"""Draw each boat's own sea: one water mesh per environment, displaced from usv_core's sea.

What it shows is exactly what the physics uses: the same sea object, the same
time, and the same voyage position (plot position + treadmill offset).

    /World/envs/env_i/Water   a square grid mesh `size` metres across, `resolution`
                              vertices per side, centred on boat i and following it.
                              Under the env root, so RTX scene partitioning tags it
                              env_i: each boat's camera sees only its own sea.

The patch follows its boat because the treadmill keeps the boat within
`recenter_radius` of its plot centre but not at it; a fixed patch would leave a
boat near an edge. When the treadmill moves a boat back, the patch moves with it
and the water it draws is unchanged (heights come from voyage coordinates), so
the jump is invisible.

Writing to the renderer
-----------------------
The RTX renderer draws from Fabric (Isaac Sim's runtime copy of the scene), not
from USD. Fabric is filled from USD when the simulation resets, so the meshes
must be created BEFORE that (in the env's _setup_scene); prims added later never
reach the renderer. And editing the USD `points` attribute afterwards changes
the stored scene but not the picture. So each update writes `points` and `normals` straight
into Fabric through usdrt, as Isaac Lab's own Fabric backend does for meshes.
Points are written relative to the mesh's parent, /World/envs/env_i, which is
already placed at the env origin: adding the origin again would draw every patch
at twice its origin. The mesh has no transform of its own, so only the point and
normal arrays ever change. The USD extent is authored large enough to cover
everywhere the patch can move, so the renderer never culls it.

Two ways to write (gpu=True is the default):
  GPU  one Warp kernel computes every vertex of every mesh and writes it into Fabric on the GPU
       (water_kernel.py); no CPU copy, one launch for all boats. Each mesh carries an int attribute
       `mmws:waterEnv` naming its environment, because Fabric hands meshes over in its own order.
  CPU  heights computed on the GPU, copied to the CPU, written mesh by mesh: fine for the few boats
       someone watches, far too slow for thousands. Kept as a fallback.
"""

from __future__ import annotations

import numpy as np
import torch
import usdrt
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils

from usv_core import SineWaveField, WaveComponents


class WaterPatches:
    def __init__(self, env_ids: list[int], size: float, resolution: int,
                 reach: float, device, color=(0.02, 0.12, 0.22), gpu: bool = True):
        """reach: how far [m] the patch centre can move from the env origin (the treadmill radius)."""
        self.env_ids = list(env_ids)
        self.device = device
        self.gpu = gpu and torch.device(device).type == "cuda"
        self._flat = None
        n = resolution
        lin = torch.linspace(-size / 2, size / 2, n, device=device)
        gx, gy = torch.meshgrid(lin, lin, indexing="xy")
        self.grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)          # [M, 2] patch-local xy

        idx = np.arange(n * n).reshape(n, n)
        quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1)
        stage = sim_utils.get_current_stage()
        half = size / 2 + reach
        # built once and shared by every mesh: per-mesh conversion is what made creating thousands slow
        counts = Vt.IntArray([4] * ((n - 1) * (n - 1)))
        indices = Vt.IntArray.FromNumpy(quads.astype(np.int32))
        flat = np.zeros((n * n, 3), np.float32)
        flat[:, :2] = self.grid.cpu().numpy()
        flat_points = Vt.Vec3fArray.FromNumpy(flat)
        up = np.zeros_like(flat)
        up[:, 2] = 1.0
        up_normals = Vt.Vec3fArray.FromNumpy(up)
        colour = Vt.Vec3fArray([Gf.Vec3f(*color)])
        extent = Vt.Vec3fArray([Gf.Vec3f(-half, -half, -20.0), Gf.Vec3f(half, half, 20.0)])
        paths = []
        for i in self.env_ids:
            path = f"/World/envs/env_{i}/Water"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr(colour)
            mesh.CreatePointsAttr(flat_points)
            mesh.CreateNormalsAttr(up_normals)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateExtentAttr(extent)
            mesh.GetPrim().CreateAttribute("mmws:waterEnv", Sdf.ValueTypeNames.Int).Set(int(i))   # for the GPU path
            paths.append(path)
        self._paths = paths
        self._rt_points = self._rt_normals = None

    def bind_fabric(self) -> None:
        """Call once after the simulation has reset (Fabric populated): handles to write the meshes each frame."""
        rt_stage = sim_utils.get_current_stage(fabric=True)
        prims = [rt_stage.GetPrimAtPath(p) for p in self._paths]
        missing = [p for p, prim in zip(self._paths, prims) if not prim.IsValid()]
        if missing:
            raise RuntimeError(f"water meshes not in Fabric (created after the reset?): {missing[:3]}")
        self._rt_points = [prim.GetAttribute("points") for prim in prims]
        self._rt_normals = [prim.GetAttribute("normals") for prim in prims]

    def update(self, sea, t: float, plot_xy: torch.Tensor, sea_offset: torch.Tensor):
        """Redraw. plot_xy/sea_offset are the full [N, ...] tensors; only drawn envs are used."""
        if not hasattr(sea, "waves"):                    # flat water: one zero-height wave per env
            if self._flat is None:
                z = torch.zeros(plot_xy.shape[0], 1, device=plot_xy.device)
                self._flat = SineWaveField.from_components(WaveComponents(
                    amp=z, omega=z, k=z, direction=z, cos_dir=z + 1.0, sin_dir=z, phase=z,
                    hs=z[:, 0], tp=z[:, 0], heading=z[:, 0], spread_s=z[:, 0]))
            sea = self._flat
        if self.gpu:
            try:
                return self._update_gpu(sea, t, plot_xy, sea_offset)
            except Exception as err:                     # never stop a run over drawing; say so once, use the CPU
                print(f"[water] GPU drawing failed, falling back to the CPU path: {type(err).__name__}: {err}", flush=True)
                self.gpu = False
        if self._rt_points is None:
            self.bind_fabric()
        ids = torch.tensor(self.env_ids, device=self.device)
        waves = type(sea.waves)(**{k: getattr(sea.waves, k)[ids] for k in sea.waves.__dataclass_fields__})
        view = SineWaveField.from_components(waves)                             # those envs' seas only
        centre = plot_xy[ids]                                                   # [K, 2] patch follows its boat
        voyage = self.grid[None] + centre[:, None, :] + sea_offset[ids, None, :2]
        eta = view.height(voyage, t)                                            # [K, M]
        slope = view.slope(voyage, t)                                           # [K, M, 2]
        local_xy = self.grid[None] + centre[:, None, :]                         # relative to the env origin
        points = torch.cat([local_xy, eta[..., None]], -1).float().cpu().numpy()
        normals = torch.cat([-slope, torch.ones_like(eta)[..., None]], -1)
        normals = (normals / normals.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        for k in range(len(self.env_ids)):
            self._rt_points[k].Set(usdrt.Vt.Vec3fArray(points[k]))
            self._rt_normals[k].Set(usdrt.Vt.Vec3fArray(normals[k]))

    def _update_gpu(self, sea, t: float, plot_xy: torch.Tensor, sea_offset: torch.Tensor) -> None:
        import warp as wp

        from .water_kernel import write_water

        stage = sim_utils.get_current_stage(fabric=True)
        # every water mesh at once, on the GPU; selected each frame since Fabric may move its storage
        self._selection = stage.SelectPrims(               # kept: the kernel follows pointers it owns
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.Normal3fArray, "normals", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.Int, "mmws:waterEnv", usdrt.Usd.Access.Read),
            ],
            device=str(self.device),
        )
        points = wp.fabricarrayarray(data=self._selection, attrib="points", dtype=wp.vec3)
        normals = wp.fabricarrayarray(data=self._selection, attrib="normals", dtype=wp.vec3)
        env_of = wp.fabricarray(data=self._selection, attrib="mmws:waterEnv", dtype=wp.int32)
        if env_of.size != len(self.env_ids):
            raise RuntimeError(f"Fabric returned {env_of.size} water meshes, expected {len(self.env_ids)}")
        write_water(points, normals, env_of, self.grid.shape[0], self.grid, sea.waves,
                    plot_xy, sea_offset[:, :2], t)
