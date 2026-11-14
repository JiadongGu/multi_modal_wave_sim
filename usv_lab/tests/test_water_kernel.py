"""GPU water kernel, run on the CPU against fake Fabric meshes, checked vertex by vertex against usv_core's sea.

No Isaac Sim needed: Warp accepts any object shaped like Fabric's array interface, so the meshes here are
plain Warp arrays laid out the way Fabric lays them out (one bucket, meshes in a shuffled env order).

    python3 -m pytest usv_lab/tests/test_water_kernel.py
"""

import importlib.util
from pathlib import Path

import pytest
import torch

wp = pytest.importorskip("warp")

from usv_core import SeaCfg, SineWaveField  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "water_kernel", Path(__file__).resolve().parents[1] / "usv_lab" / "tasks" / "water_kernel.py")
water_kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(water_kernel)          # by path: the tasks package itself imports Isaac Lab


def fake_fabric_array_of_arrays(rows: list, attrib: str) -> dict:
    """Fabric's interface for an array-valued attribute (e.g. mesh points), one bucket."""
    pointers = wp.array([r.ptr for r in rows], dtype=wp.uint64, device="cpu")
    lengths = wp.array([r.size for r in rows], dtype=wp.uint64, device="cpu")
    info = {"type": (True, "f4", 3, 1, "vector"), "access": 2, "pointers": [int(pointers.ptr)],
            "counts": [len(rows)], "array_lengths": [int(lengths.ptr)]}
    return {"version": 1, "device": "cpu", "attribs": {attrib: info}, "_refs": (rows, pointers, lengths)}


def fake_fabric_ints(values: wp.array, attrib: str) -> dict:
    info = {"type": (True, "i4", 1, 0, ""), "access": 1, "pointers": [int(values.ptr)], "counts": [values.size]}
    return {"version": 1, "device": "cpu", "attribs": {attrib: info}, "_refs": (values,)}


def test_kernel_matches_the_physics_sea_at_every_vertex():
    wp.init()
    n_envs, n = 6, 9                                            # 6 seas, 9 x 9 vertex patches
    size = 40.0
    lin = torch.linspace(-size / 2, size / 2, n)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)    # [M, 2]
    m = grid.shape[0]
    sea = SineWaveField(SeaCfg(hs=(1.0, 4.0), tp=(5.0, 10.0)), n_envs, seed=3)
    g = torch.Generator().manual_seed(0)
    centre = (torch.rand(n_envs, 2, generator=g) - 0.5) * 20    # boat's place on its plot
    offset = (torch.rand(n_envs, 2, generator=g) - 0.5) * 500   # treadmill offset
    t = 123.4

    drawn = [4, 1, 5]                                           # meshes for 3 of the 6 envs, Fabric's order
    pts = [wp.zeros(m, dtype=wp.vec3, device="cpu") for _ in drawn]
    nrm = [wp.zeros(m, dtype=wp.vec3, device="cpu") for _ in drawn]
    tags = wp.array(drawn, dtype=wp.int32, device="cpu")
    # the interfaces own the pointer tables the kernel follows: keep them alive until it has run
    ifaces = (fake_fabric_array_of_arrays(pts, "points"), fake_fabric_array_of_arrays(nrm, "normals"),
              fake_fabric_ints(tags, "mmws:waterEnv"))
    points = wp.fabricarrayarray(data=ifaces[0], attrib="points")
    normals = wp.fabricarrayarray(data=ifaces[1], attrib="normals")
    env_of = wp.fabricarray(data=ifaces[2], attrib="mmws:waterEnv")

    water_kernel.write_water(points, normals, env_of, m, grid, sea.waves, centre, offset, t)
    wp.synchronize()

    voyage = grid[None] + centre[:, None, :] + offset[:, None, :]          # [N, M, 2]
    eta = sea.height(voyage, t)
    slope = sea.slope(voyage, t)
    for row, e in enumerate(drawn):
        p = torch.from_numpy(pts[row].numpy())
        q = torch.from_numpy(nrm[row].numpy())
        assert torch.allclose(p[:, :2], grid + centre[e], atol=1e-5), "vertex drawn in the wrong place"
        assert torch.allclose(p[:, 2], eta[e], atol=2e-3), f"height differs: {(p[:, 2] - eta[e]).abs().max()}"
        want = torch.cat([-slope[e], torch.ones(m, 1)], -1)
        want = want / want.norm(dim=-1, keepdim=True)
        assert torch.allclose(q, want, atol=2e-3)
        assert eta[e].abs().max() > 0.3                                     # a real sea, not a flat one
