"""Hydrodynamic damping: what stops a floating hull bouncing forever.

Buoyancy alone is a spring. Damping is the water resisting motion, Fossen's

    tau = -(D_lin + D_quad |nu|) nu          (per axis, body frame)

with nu = (u, v, w, p, q, r) the body-frame velocity. The linear term rules at
low speed (skin friction, small waves made by the hull), the quadratic at
speed (water shoved aside, swirling wake).

Coefficients from the hull's size, as in the extension's damping.py
--------------------------------------------------------------------
    translational  D_quad = 1/2 rho Cd A, A the area facing that way (from the
                   hull's bounding box), Cd = (0.05, 1, 1): streamlined ahead,
                   flat-plate sideways and vertically
    rotational     drag integrated along the span the rotation sweeps, since a
                   spinning hull's ends move fastest: rho Cd w S^4 / 64
    linear         a tenth of the quadratic, so damping still acts below ~0.1 m/s

Two changes from the extension
------------------------------
Relative to the water   the translational velocity damped is the hull's minus
                        the water's orbital velocity at the middle of the hull's
                        underwater body. A boat drifting with the wave's own
                        motion is not fighting the water and is not slowed.
Wet fraction            scaled by submerged volume / design volume from the
                        hull's columns (hull.py), so a hull in the air falls
                        freely instead of dragging against nothing.

Stable at any speed (kept from the extension)
---------------------------------------------
Applied naively, -k nu with a fixed step dt overshoots once k dt / m > 1:
the drag removes more speed than the boat has, reverses it, and the next step
is worse. Solving nu' = -(k/m) nu exactly over one step gives

    tau = -k nu / (1 + k dt / m)

which equals -k nu as dt -> 0 and can at most bring the boat to rest in one
step, never past it. It needs ctx.dt, and m is mass (translation) or inertia
(rotation) from the hull geometry. ctx.dt = 0 means the plain form.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .conventions import BodyState, Wrench, body_to_world, world_to_body
from .hull import HullGeometry
from .model import HullModel, ModelContext


@dataclass
class DampingCfg:
    """drag_coefficient   Cd for (surge, sway, heave); rotation uses the bluffest of sway/heave
    linear_fraction       linear coefficient as a fraction of the quadratic
    quadratic, linear     6 values (surge, sway, heave, roll, pitch, yaw) to override the derived ones
    relative_to_water     damp velocity relative to the water's orbital motion
    """

    drag_coefficient: tuple[float, float, float] = (0.05, 1.0, 1.0)
    linear_fraction: float = 0.1
    quadratic: tuple[float, ...] | None = None
    linear: tuple[float, ...] | None = None
    relative_to_water: bool = True


def derived_coefficients(geometry: HullGeometry, cfg: DampingCfg) -> tuple[list[float], list[float]]:
    """(linear, quadratic), 6 each, from the hull's bounding box. Same formulas as the extension."""
    c = geometry.cfg
    lx, ly, lz = c.length, c.beam, c.draft + c.freeboard
    rho, cd = c.density, cfg.drag_coefficient
    areas = (ly * lz, lx * lz, lx * ly)
    quad = [0.5 * rho * cd[i] * areas[i] for i in range(3)]
    cd_rot = max(cd[1], cd[2])
    quad += [rho * cd_rot / 64.0 * v for v in (lx * ly**4, ly * lx**4, lz * lx**4)]
    if cfg.quadratic is not None:
        quad = list(cfg.quadratic)
    lin = list(cfg.linear) if cfg.linear is not None else [cfg.linear_fraction * q for q in quad]
    return lin, quad


class HydroDamping(HullModel):
    """Linear + quadratic damping in the body frame, relative to the water, scaled by wetness, stable at any speed."""

    def __init__(self, geometry: HullGeometry, cfg: DampingCfg | None = None):
        self.geo = geometry
        self.cfg = cfg or DampingCfg()
        lin, quad = derived_coefficients(geometry, self.cfg)
        self.linear = torch.tensor(lin, dtype=torch.float64)
        self.quadratic = torch.tensor(quad, dtype=torch.float64)
        self.generalised_mass = torch.tensor([geometry.mass] * 3 + list(geometry.inertia()), dtype=torch.float64)
        # where the water's velocity is sampled: middle of the underwater body, below the CoM's hull-frame spot
        com = geometry.cfg.com
        self._probe_body = torch.tensor([0.0 - com[0], 0.0 - com[1], -0.5 * geometry.cfg.draft - com[2]], dtype=torch.float64)

    def _constants(self, dev, dtype):
        """(linear, quadratic, generalised mass, probe point) on the device. Cached: copying them every call
        would make the CPU wait for the GPU each time."""
        key = (dev, dtype)
        cache = self.__dict__.setdefault("_const_cache", {})
        if key not in cache:
            cache[key] = tuple(x.to(dev, dtype) for x in (self.linear, self.quadratic, self.generalised_mass, self._probe_body))
        return cache[key]

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        dev, dtype, n = state.device, state.pos.dtype, state.num_envs
        linear, quadratic, mass, probe_body = self._constants(dev, dtype)
        _, depth = self.geo.submersion_for(state, ctx)
        wet = self.geo.wet_fraction(depth)                                             # [N]

        lin_vel = state.lin_vel
        if self.cfg.relative_to_water and ctx.sea is not None:
            probe = state.pos + body_to_world(state.quat, probe_body.expand(n, 3))
            lin_vel = lin_vel - ctx.sea.velocity(probe[:, None, :], ctx.t)[:, 0, :]
        nu = torch.cat([world_to_body(state.quat, lin_vel), world_to_body(state.quat, state.ang_vel)], -1)  # [N, 6]

        k = wet[:, None] * (linear + quadratic * nu.abs())                                             # [N, 6]
        tau = -k * nu / (1.0 + k * ctx.dt / mass)
        return Wrench(body_to_world(state.quat, tau[:, :3]), body_to_world(state.quat, tau[:, 3:]))
