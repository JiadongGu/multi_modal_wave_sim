"""Hull geometry as vertical columns, and buoyancy from them.

The column method
-----------------
Seen from above, the hull's waterplane is cut into a grid of cells. Each cell
becomes a vertical column (a prism) with

    xy      its centre, hull frame (x forward from midship, y to port)
    area    the waterplane area inside the cell at the design draft
    bottom  z of its floor, chosen so the column's volume at the design draft
            equals the hull's true volume over that cell
    top     z of the deck (design waterline + freeboard)

Hull frame: x forward, y port, z up, origin at midship on the design
waterline. The column floors are averages, so the hull's displaced volume,
waterplane area (heave stiffness) and their distribution along and across the
hull are right; the exact underwater shape inside each cell is not.

Each step, per column per boat: rotate the column into the world with the
boat's pose, ask the sea for the water height there, and take the submerged
depth d = clamp(eta - z_floor, 0, z_deck - z_floor). Then

    vertical force    rho g A d                       (Archimedes, per column)
    horizontal force  -rho g A d (d eta/dx, d eta/dy)  (the wave slope pushes the
                      submerged volume downslope: the hydrostatic pressure under
                      a tilted surface has a horizontal gradient)

applied at the column's horizontal position; torque about the centre of mass
is the sum of r x F. Roll and pitch righting moments are not programmed in:
they come out of the low side's columns sinking deeper than the high side's.

Simplifications
---------------
Columns stay vertical in the world as the hull tilts (their floor and deck
points move with the hull, but the prism is not slanted), which is accurate
for moderate angles. Pressure is hydrostatic under the local surface (no
diffraction, no radiation, no dynamic pressure), the same assumption as the
extension's triangle-cutting BuoyancyPressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .constants import GRAVITY, WATER_DENSITY_SALT
from .conventions import BodyState, Wrench, body_to_world
from .model import HullModel, ModelContext


@dataclass
class HullCfg:
    """A hull. Lengths in metres; defaults are the Journee (1992) Wigley model the extension uses.

    shape      "wigley": y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2) below the waterline,
               wall-sided above it; "box": rectangular
    draft      depth of the keel below the design waterline (T)
    freeboard  deck height above the design waterline
    columns    (along, across) grid of buoyancy columns
    com        centre of mass in the hull frame (z measured from the design waterline)
    gyration   radii of gyration about x, y, z as fractions of (beam, length, length);
               rule-of-thumb ship values, used only for the inertia handed to the physics engine
    """

    shape: Literal["wigley", "box"] = "wigley"
    length: float = 6.10
    beam: float = 2.72
    draft: float = 0.60
    freeboard: float = 0.40
    columns: tuple[int, int] = (8, 4)
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float = WATER_DENSITY_SALT
    gravity: float = GRAVITY
    gyration: tuple[float, float, float] = (0.35, 0.25, 0.25)
    integration_samples: int = 64  # per cell side, for the exact per-cell volume at construction


class HullGeometry:
    """Columns for one hull, shared by every boat. Tensors are [C] or [C, 2] (C = number of columns)."""

    def __init__(self, cfg: HullCfg, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64):
        self.cfg = cfg
        nx, ny = cfg.columns
        L, B, T, s = cfg.length, cfg.beam, cfg.draft, cfg.integration_samples

        # Fine sample grid over each cell (float64 on the CPU; done once).
        cx = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx * L - L / 2
        cy = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny * B - B / 2
        fx = (torch.arange(s, dtype=torch.float64) + 0.5) / s - 0.5          # offsets within a cell, in cells
        x = cx[:, None, None, None] + fx[None, None, :, None] * (L / nx)     # [nx, 1, s, 1]
        y = cy[None, :, None, None] + fx[None, None, None, :] * (B / ny)     # [1, ny, 1, s]
        dA = (L / nx) * (B / ny) / (s * s)

        if cfg.shape == "box":
            inside = torch.ones(nx, ny, s, s, dtype=torch.bool)
            depth = torch.full((nx, ny, s, s), T, dtype=torch.float64)
        elif cfg.shape == "wigley":
            half = (B / 2) * (1 - (2 * x / L) ** 2)                           # waterline half-breadth b(x)
            ratio = (y.abs() / half.clamp_min(1e-12)).expand(nx, ny, s, s)
            inside = (ratio < 1.0) & (half > 0).expand(nx, ny, s, s)
            depth = T * torch.sqrt((1 - ratio).clamp_min(0.0))               # hull depth below the waterline at (x, y)
        else:
            raise ValueError(f"unknown hull shape {cfg.shape!r}")

        area = inside.sum((-1, -2)).to(torch.float64) * dA                   # [nx, ny] waterplane area per cell
        volume = (depth * inside).sum((-1, -2)) * dA                         # [nx, ny] design volume per cell
        keep = area > 1e-9 * L * B
        xs = (x * inside).sum((-1, -2)) * dA / area.clamp_min(1e-30)         # area centroid of each cell
        ys = (y * inside).sum((-1, -2)) * dA / area.clamp_min(1e-30)

        self.area = area[keep].to(device, dtype)
        self.xy = torch.stack([xs[keep], ys[keep]], -1).to(device, dtype)
        self.bottom = (-volume[keep] / area[keep]).to(device, dtype)          # floor z: volume / area below z = 0
        self.top = torch.full_like(self.bottom, cfg.freeboard)

        self.volume = volume[keep].sum().item()                              # design displacement [m^3]
        self.waterplane_area = area[keep].sum().item()
        self.mass = cfg.density * self.volume                                # floats at the design draft
        self.center_of_buoyancy = (
            (xs[keep] * volume[keep]).sum().item() / self.volume,
            (ys[keep] * volume[keep]).sum().item() / self.volume,
            -(volume[keep] ** 2 / area[keep]).sum().item() / (2 * self.volume),
        )

    @property
    def num_columns(self) -> int:
        return self.area.shape[0]

    def inertia(self) -> tuple[float, float, float]:
        """Diagonal inertia about the CoM [kg m^2] from the radii of gyration. Hand this to the physics engine."""
        kx, ky, kz = self.cfg.gyration
        c = self.cfg
        return (self.mass * (kx * c.beam) ** 2, self.mass * (ky * c.length) ** 2, self.mass * (kz * c.length) ** 2)


class ColumnBuoyancy(HullModel):
    """Buoyancy and wave-slope force from the hull's columns. Needs ctx.sea (None means still water)."""

    def __init__(self, geometry: HullGeometry):
        self.geo = geometry
        com = torch.tensor(geometry.cfg.com, dtype=geometry.area.dtype, device=geometry.area.device)
        xy0 = torch.cat([geometry.xy, torch.zeros_like(geometry.area)[:, None]], -1)   # [C, 3] at z = 0
        self._floor = xy0 + torch.stack([torch.zeros_like(geometry.bottom)] * 2 + [geometry.bottom], -1) - com
        self._deck = xy0 + torch.stack([torch.zeros_like(geometry.top)] * 2 + [geometry.top], -1) - com
        self._rho_g_area = geometry.cfg.density * geometry.cfg.gravity * geometry.area   # [C]

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        dtype, dev = state.pos.dtype, state.device
        n = state.num_envs
        floor_b = self._floor.to(dev, dtype).expand(n, -1, -1)                          # [N, C, 3] body
        deck_b = self._deck.to(dev, dtype).expand(n, -1, -1)
        floor = state.pos[:, None, :] + body_to_world(state.quat, floor_b)              # [N, C, 3] env-local
        deck_z = state.pos[:, None, 2] + body_to_world(state.quat, deck_b)[..., 2]       # [N, C]

        if ctx.sea is None:
            eta = torch.zeros_like(deck_z)
            slope = torch.zeros_like(floor[..., :2])
        else:
            eta = ctx.sea.height(floor[..., :2], ctx.t)
            slope = ctx.sea.slope(floor[..., :2], ctx.t)

        depth = torch.minimum((eta - floor[..., 2]).clamp_min(0.0), (deck_z - floor[..., 2]).clamp_min(0.0))
        lift = self._rho_g_area.to(dev, dtype) * depth                                  # [N, C]
        force = torch.cat([-lift[..., None] * slope, lift[..., None]], -1)              # [N, C, 3]

        # Apply each column's force at the middle of its wet part, measured up the hull's own vertical
        # axis. On a tilted hull that point sits sideways of the floor; using the floor instead would put
        # the centre of buoyancy at keel depth and lose the KB term of the righting moment.
        up_body = torch.zeros(n, 3, device=dev, dtype=dtype)
        up_body[:, 2] = 1.0
        hull_up = body_to_world(state.quat, up_body)                                    # [N, 3]
        along = depth / hull_up[:, 2:3].clamp_min(1e-6)                                 # wet length along the axis
        r = floor - state.pos[:, None, :] + (0.5 * along)[..., None] * hull_up[:, None, :]
        torque = torch.cross(r, force, dim=-1).sum(1)
        return Wrench(force.sum(1), torque)
