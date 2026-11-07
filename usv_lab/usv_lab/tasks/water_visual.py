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

Vertices are written to USD from the CPU each update, which is fine for the
handful of environments someone watches and far too slow for thousands: draw
only the environments being viewed.
"""

from __future__ import annotations

import numpy as np
import torch
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils

from usv_core import SineWaveField


class WaterPatches:
    def __init__(self, env_ids: list[int], size: float, resolution: int, device, color=(0.05, 0.28, 0.45)):
        self.env_ids = list(env_ids)
        self.device = device
        n = resolution
        lin = torch.linspace(-size / 2, size / 2, n, device=device)
        gx, gy = torch.meshgrid(lin, lin, indexing="xy")
        self.grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)          # [M, 2] patch-local xy

        # two triangles per grid cell, as quads
        idx = np.arange(n * n).reshape(n, n)
        quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1)
        stage = sim_utils.get_current_stage()
        self.meshes, self.translates = [], []
        for i in self.env_ids:
            mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{i}/Water")
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * ((n - 1) * (n - 1))))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(quads.tolist()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.zeros((n * n, 3), np.float32)))
            self.translates.append(mesh.AddTranslateOp())
            self.meshes.append(mesh)

    def update(self, sea: SineWaveField, t: float, env_origins: torch.Tensor, plot_xy: torch.Tensor, sea_offset: torch.Tensor):
        """Redraw. env_origins/plot_xy/sea_offset are the full [N, ...] tensors; only drawn envs are used."""
        ids = torch.tensor(self.env_ids, device=self.device)
        waves = type(sea.waves)(**{k: getattr(sea.waves, k)[ids] for k in sea.waves.__dataclass_fields__})
        view = SineWaveField.from_components(waves)                             # those envs' seas only
        centre = plot_xy[ids]                                                   # [K, 2] patch follows its boat
        voyage = self.grid[None] + centre[:, None, :] + sea_offset[ids, None, :2]
        eta = view.height(voyage, t)                                            # [K, M]
        points = torch.cat([self.grid[None].expand(len(self.env_ids), -1, -1), eta[..., None]], -1)
        points = points.float().cpu().numpy()
        origin = (env_origins[ids, :2] + centre).cpu().numpy()
        for k, mesh in enumerate(self.meshes):
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points[k]))
            self.translates[k].Set(Gf.Vec3d(float(origin[k, 0]), float(origin[k, 1]), 0.0))
