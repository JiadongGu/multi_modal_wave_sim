"""Hydrodynamic damping - what stops a hull ringing forever.

Buoyancy alone is a spring. A hull dropped into water oscillates about its
waterline, and with nothing to dissipate energy it never stops. Worse, the
water surface is moving, so an undamped body bouncing on it gains energy every
cycle - the same mechanism as a ball on a vibrating plate - and departs.

Fossen writes the damping as a 6x6 matrix acting on the body-frame velocity:

    tau = -D(nu) nu,    D(nu) = D_linear + D_quadratic * |nu|

The linear term dominates at low speed (skin friction, wave radiation), the
quadratic at speed (form drag, flow separation). This is the same split
ASTRA's fossen.py uses.

Defaults are derived, not invented
----------------------------------
The quadratic coefficients come from the hull itself:

    D_quad = 0.5 * rho * Cd * A

with A the projected area on each axis, taken from the hull's bounding box,
and Cd a shape factor near 1 for a bluff body. So a given hull gets sensible
damping without anyone tuning numbers, which is the whole complaint about
per-vessel constants elsewhere in this project.

Rotational terms use the same form with a representative lever arm, which is
cruder - a proper value needs the hull's second moment of area. Good enough to
stop the ringing; replace it when the damping is being matched to data.

Why this is solved implicitly
-----------------------------
Applied explicitly, quadratic drag is only stable while

    (D/m) |nu| dt < 1

which for a 1 m cube at 60 Hz means |nu| < 24 m/s. Past that the drag
overshoots, reverses the velocity, and amplifies it every step: a hull at
60 m/s diverges within ten steps. Since an undamped hull can easily reach
those speeds before the damping ever engages, the damping itself becomes the
thing that destroys the simulation.

So the step is solved exactly instead. For nu' = -(k/m) nu the solution over
dt is nu_new = nu / (1 + k dt/m), giving

    tau = -k nu / (1 + k dt / m)

which equals the explicit form as dt -> 0 and is bounded for any speed. It
needs the step size, hence `timestep` - set it to the physics step.
"""

from __future__ import annotations

import numpy as np

from ..globals import WATER_DENSITY
from ..registry import FORCES
from .base import BodyState, ForceModel, Wrench


@FORCES.register("damping")
class LinearQuadraticDamping(ForceModel):
    """tau = -(D_lin + D_quad * |nu|) nu, in the body frame."""

    def __init__(self, linear=None, quadratic=None, extents=None,
                 drag_coefficient: float = 1.0, timestep: float = 1.0 / 60.0,
                 density: float = WATER_DENSITY) -> None:
        if timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        self.timestep = float(timestep)
        if quadratic is None:
            if extents is None:
                raise ValueError("damping needs either `quadratic` or hull `extents`")
            lx, ly, lz = (float(v) for v in np.asarray(extents, dtype=float).ravel()[:3])
            areas = np.array([ly * lz, lx * lz, lx * ly])
            q = 0.5 * density * drag_coefficient * areas
            # Rotational: same drag over a representative lever arm, squared
            # once for the moment arm and once for the velocity it sees.
            arm = np.array([max(ly, lz), max(lx, lz), max(lx, ly)]) / 2.0
            quadratic = np.concatenate([q, q * arm**2])
        self.quadratic = np.asarray(quadratic, dtype=float).ravel()
        if self.quadratic.size != 6:
            raise ValueError(f"quadratic must have 6 entries, got {self.quadratic.size}")
        if linear is None:
            # A tenth of the quadratic coefficient, which gives the linear term
            # authority below about 0.1 m/s where the quadratic one vanishes.
            linear = 0.1 * self.quadratic
        self.linear = np.asarray(linear, dtype=float).ravel()
        if self.linear.size != 6:
            raise ValueError(f"linear must have 6 entries, got {self.linear.size}")

    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        nu = np.concatenate([
            np.asarray(body.linear_velocity, dtype=float),
            np.asarray(body.angular_velocity, dtype=float),
        ])
        k = self.linear + self.quadratic * np.abs(nu)
        inertia = np.asarray(body.inertia, dtype=float).ravel()
        if inertia.size != 3 or not np.all(inertia > 0.0):
            inertia = np.ones(3)
        generalised_mass = np.concatenate([np.full(3, max(body.mass, 1e-9)), inertia])
        tau = -k * nu / (1.0 + k * self.timestep / generalised_mass)
        return Wrench(tau[:3], tau[3:])
