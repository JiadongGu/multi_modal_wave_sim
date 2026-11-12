"""Lidar against the sea: one independent lidar per boat, each seeing only its own sea.

Isaac Lab's ray caster only hits meshes frozen at startup, and the sea is a moving
formula, so the rays are intersected with the sea directly through the sea socket.
Row i of the scan uses environment i's sea and boat i's pose; nothing is shared
between boats and nothing touches the rendered scene.

Pattern: `channels` elevation angles times a full circle of azimuths, fixed to the
hull (x forward, y port, z up), so the scan rolls and pitches with the boat.

Finding the hit, for a ray origin o and direction d: the ray is above the water at
range s while f(s) = o_z + s d_z - eta(o_xy + s d_xy, t) > 0.
  1. Limit the search. No wave is higher than the sum of its amplitudes, so the ray
     can only meet the water where its height is within that bound.
  2. March through that window in `march_steps` steps; the first step with f <= 0
     brackets the hit.
  3. Bisect the bracket.
A crest narrower than one march step can be jumped over (steps are longest on near-level
rays at long range). A sensor already under the water reads range 0. No return reads inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .conventions import BodyState, body_to_world


@dataclass
class LidarCfg:
    """channels × (360 / horizontal_res) rays. mount: lidar position in the body frame, from the CoM [m]."""

    channels: int = 16
    vertical_fov: tuple[float, float] = (-15.0, 15.0)   # degrees
    horizontal_res: float = 2.0                         # degrees
    max_range: float = 100.0                            # [m]
    mount: tuple[float, float, float] = (0.0, 0.0, 1.5)
    update_period: float = 0.1                          # [s] of simulated time, used by the environment
    march_steps: int = 32
    refine_steps: int = 10
    wave_bound: float = 15.0                            # [m] height limit for a sea that does not report one


@dataclass
class LidarScan:
    """ranges [N, R] in metres (inf: no return). hits [N, R, 3] in the same frame as state.pos (inf: no return)."""

    ranges: torch.Tensor
    hits: torch.Tensor


class SeaLidar:
    def __init__(self, cfg: LidarCfg | None = None, device="cpu", dtype=torch.float32):
        self.cfg = cfg or LidarCfg()
        c = self.cfg
        elevation = torch.deg2rad(torch.linspace(c.vertical_fov[0], c.vertical_fov[1], c.channels, device=device, dtype=dtype))
        azimuth_count = int(round(360.0 / c.horizontal_res))
        azimuth = torch.arange(azimuth_count, device=device, dtype=dtype) * 2.0 * math.pi / azimuth_count
        el, az = torch.meshgrid(elevation, azimuth, indexing="ij")
        self.directions = torch.stack([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], -1).reshape(-1, 3)  # [R, 3] body
        self.mount = torch.tensor(c.mount, device=device, dtype=dtype)

    @property
    def num_rays(self) -> int:
        return self.directions.shape[0]

    def scan(self, state: BodyState, sea, t) -> LidarScan:
        c, n, r = self.cfg, state.num_envs, self.num_rays
        origin = state.pos + body_to_world(state.quat, self.mount.expand(n, 3))             # [N, 3]
        d = body_to_world(state.quat, self.directions.expand(n, -1, -1))                    # [N, R, 3]
        o = origin[:, None, :].expand(n, r, 3)

        def above_water(s):                                                                 # f(s), [N, R]
            p = o + s[..., None] * d
            return p[..., 2] - sea.height(p[..., :2], t)

        bound = sea.height_bound() if hasattr(sea, "height_bound") else None
        bound = torch.full((n,), c.wave_bound, device=o.device, dtype=o.dtype) if bound is None else bound.to(o)
        bound = bound[:, None] + 0.01                                 # margin: flat water's bound is exactly 0
        oz, dz = o[..., 2], d[..., 2]
        safe_dz = torch.where(dz.abs() < 1e-6, torch.full_like(dz, 1e-6), dz)
        to_bound_top, to_bound_bottom = (bound - oz) / safe_dz, (-bound - oz) / safe_dz
        start = torch.minimum(to_bound_top, to_bound_bottom).clamp_min(0.0)                 # ray enters the wave band
        end = torch.maximum(to_bound_top, to_bound_bottom).clamp_max(c.max_range)           # ray leaves it, or range limit
        level = dz.abs() < 1e-6
        inside = oz.abs() <= bound
        start = torch.where(level, torch.zeros_like(start), start)
        end = torch.where(level, torch.where(inside, torch.full_like(end, c.max_range), torch.full_like(end, -1.0)), end)

        submerged = above_water(torch.zeros(n, r, device=o.device, dtype=o.dtype)) <= 0.0
        searching = (end > start) & ~submerged
        low = start.clone()
        high = end.clone()
        found = torch.zeros_like(searching)
        previous = start.clone()
        for step in range(1, c.march_steps + 1):
            s = start + (end - start) * step / c.march_steps
            crossed = searching & ~found & (above_water(s) <= 0.0)
            low = torch.where(crossed, previous, low)
            high = torch.where(crossed, s, high)
            found |= crossed
            previous = s
            if not bool((searching & ~found).any()):
                break

        for _ in range(c.refine_steps):
            middle = 0.5 * (low + high)
            below = above_water(middle) <= 0.0
            high = torch.where(below, middle, high)
            low = torch.where(below, low, middle)

        ranges = torch.where(found, high, torch.full_like(high, math.inf))
        ranges = torch.where(submerged, torch.zeros_like(ranges), ranges)
        hits = torch.where(torch.isfinite(ranges)[..., None], o + torch.nan_to_num(ranges, posinf=0.0)[..., None] * d,
                           torch.full_like(o, math.inf))
        return LidarScan(ranges, hits)
