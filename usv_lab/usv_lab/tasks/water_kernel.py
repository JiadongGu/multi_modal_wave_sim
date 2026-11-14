"""GPU water: wave heights and normals written straight into the renderer's meshes, no CPU copy.

One Warp kernel thread per (mesh, vertex). Each thread evaluates the same sum of sine waves as
usv_core's SineWaveField (height and slope) at that vertex's voyage position, and writes the vertex
position and normal into Fabric, the scene data the RTX renderer draws from. All boats' meshes in one
launch; nothing goes through the CPU.

Kept free of Isaac Sim imports so it can be tested anywhere Warp runs (see usv_lab/tests).

Inputs, per launch:
  points, normals   Fabric arrays of arrays: [meshes][vertices] vec3, written
  env_of            Fabric array: which environment each mesh belongs to (the order Fabric hands
                    meshes over in is not the env order, so every mesh carries its env index)
  grid              [M] vec2: each vertex's (x, y) within its patch, centred on the boat
  centre, offset    [N] vec2: patch centre relative to the env origin (the boat's plot position),
                    and the treadmill offset. Heights use voyage position = grid + centre + offset;
                    the drawn vertex sits at grid + centre (relative to the env origin, its parent)
  waves             [N, K] amplitude, wavenumber, angular frequency, phase, cos and sin of direction
"""

from __future__ import annotations

import torch
import warp as wp


@wp.kernel
def _water_kernel(
    points: wp.fabricarrayarray(dtype=wp.vec3),
    normals: wp.fabricarrayarray(dtype=wp.vec3),
    env_of: wp.fabricarray(dtype=wp.int32),
    grid: wp.array(dtype=wp.vec2),
    centre: wp.array(dtype=wp.vec2),
    offset: wp.array(dtype=wp.vec2),
    amp: wp.array2d(dtype=float),
    wavenumber: wp.array2d(dtype=float),
    omega: wp.array2d(dtype=float),
    phase: wp.array2d(dtype=float),
    cos_dir: wp.array2d(dtype=float),
    sin_dir: wp.array2d(dtype=float),
    t: float,
):
    i, j = wp.tid()
    e = env_of[i]
    local = grid[j] + centre[e]
    voyage = local + offset[e]
    eta = float(0.0)
    sx = float(0.0)
    sy = float(0.0)
    for c in range(amp.shape[1]):
        cd = cos_dir[e, c]
        sd = sin_dir[e, c]
        psi = wavenumber[e, c] * (voyage[0] * cd + voyage[1] * sd) - omega[e, c] * t + phase[e, c]
        eta += amp[e, c] * wp.sin(psi)
        g = amp[e, c] * wavenumber[e, c] * wp.cos(psi)
        sx += g * cd
        sy += g * sd
    points[i][j] = wp.vec3(local[0], local[1], eta)
    normals[i][j] = wp.normalize(wp.vec3(-sx, -sy, 1.0))


def _vec2(x: torch.Tensor) -> wp.array:
    return wp.from_torch(x.float().contiguous(), dtype=wp.vec2)


def write_water(points, normals, env_of, num_vertices: int, grid: torch.Tensor, waves, centre: torch.Tensor,
                offset: torch.Tensor, t: float) -> None:
    """Fill every selected mesh. points/normals/env_of: Warp Fabric arrays (or anything with the same
    interface). grid [M, 2]; centre, offset [N, 2] torch tensors on the Warp device; waves: WaveComponents."""
    f = lambda x: wp.from_torch(x.float().contiguous())                   # [N, K], shared memory, no copy
    inputs = [points, normals, env_of, _vec2(grid), _vec2(centre), _vec2(offset),
              f(waves.amp), f(waves.k), f(waves.omega), f(waves.phase), f(waves.cos_dir), f(waves.sin_dir), float(t)]
    dim = (env_of.size, num_vertices)
    if grid.device.type == "cuda":
        # launch on PyTorch's stream: the boat positions and seas come from PyTorch work that may still be queued
        with wp.ScopedStream(wp.stream_from_torch(grid.device)):
            wp.launch(_water_kernel, dim=dim, inputs=inputs)
    else:
        wp.launch(_water_kernel, dim=dim, inputs=inputs, device="cpu")
