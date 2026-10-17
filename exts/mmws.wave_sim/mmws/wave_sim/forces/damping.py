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

    # Drag is not the same in every direction, and treating it as if it were
    # is why a 5.3 m speedboat had a top speed of 0.6 m/s. A hull is
    # streamlined going ahead and bluff going sideways:
    #
    #   surge  a Whaler 170 does ~16.5 m/s on 90 hp, so its resistance there
    #          is about P*eta/v = 2.2 kN, an effective 8 N/(m/s)^2 against the
    #          851 that Cd = 1 on the frontal area gives. Hence ~0.1, and even
    #          that is generous once a planing hull is up and running.
    #   sway   beam-on, a hull really is close to a flat plate.
    #   heave  likewise, and this is the term that damps the vertical mode,
    #          so it is left alone.
    #
    # A scalar still works and is applied to all three, for anyone who wants
    # the old isotropic behaviour or is modelling something that is not a
    # hull.
    # 0.05 ahead rather than 0.10: a planing hull rises and its resistance
    # collapses, and with no lift model the only place that can go is the
    # drag coefficient. Labelled as the stand-in it is, not a measured Cd.
    DRAG_SURGE_SWAY_HEAVE = (0.05, 1.0, 1.0)

    def __init__(self, linear=None, quadratic=None, extents=None,
                 drag_coefficient=None, timestep: float = 1.0 / 60.0,
                 density: float = WATER_DENSITY, height: float = None) -> None:
        if timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        self.timestep = float(timestep)
        if quadratic is None:
            if extents is None:
                raise ValueError("damping needs either `quadratic` or hull `extents`")
            lx, ly, lz = (float(v) for v in np.asarray(extents, dtype=float).ravel()[:3])
            areas = np.array([ly * lz, lx * lz, lx * ly])
            cd = (np.array(self.DRAG_SURGE_SWAY_HEAVE, dtype=float)
                  if drag_coefficient is None
                  else np.broadcast_to(
                      np.asarray(drag_coefficient, dtype=float).ravel(), (3,)))
            q = 0.5 * density * cd * areas
            # Rotational, integrated along the hull rather than lumped at a
            # single radius. A body rotating at omega has local speed omega*r,
            # so drag per unit span is 0.5 rho Cd (omega r)^2 and its moment
            # is that times r again:
            #
            #     N = 2 * int(0, S/2) 0.5 rho Cd w (omega r)^2 r dr
            #       = rho Cd w omega^2 S^4 / 64
            #
            # with S the span the rotation sweeps and w the width of the
            # strip. Yaw sweeps the SIDE of the hull along its length; roll
            # sweeps the bottom across the beam; pitch sweeps the bottom along
            # the length. The previous version used the plan area and one
            # lever for all three, which left yaw damping 4.4x too large: a
            # 5.2 m boat needed 29 s to turn around. Rotation is broadside
            # whatever the hull is doing in surge, so it takes the bluff
            # coefficient.
            cd_rot = float(np.max(cd[1:]))
            quadratic = np.concatenate([q, density * cd_rot / 64.0 * np.array([
                lx * ly**4,     # roll  about x
                ly * lx**4,     # pitch about y
                lz * lx**4,     # yaw   about z
            ])])
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
        # Hull height, used to work out how much of it is actually in the water.
        if height is None and extents is not None:
            height = float(np.asarray(extents, dtype=float).ravel()[2])
        self.height = float(height) if height else 1.0

    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        nu = np.concatenate([
            np.asarray(body.linear_velocity, dtype=float),
            np.asarray(body.angular_velocity, dtype=float),
        ])
        # Water damps; air does not, to any degree that matters here. Scale by
        # how much of the hull is submerged, so a hull in flight is in free
        # fall rather than dragging against nothing.
        wet = 1.0
        if wave_field is not None:
            z = float(np.asarray(body.position, dtype=float)[2])
            x, y = (float(v) for v in np.asarray(body.position, dtype=float)[:2])
            surface = float(np.asarray(wave_field.height(np.array([x]), np.array([y])))[0])
            wet = float(np.clip((surface - (z - 0.5 * self.height)) / self.height, 0.0, 1.0))
        if wet <= 0.0:
            return Wrench()

        k = wet * (self.linear + self.quadratic * np.abs(nu))
        inertia = np.asarray(body.inertia, dtype=float).ravel()
        if inertia.size != 3 or not np.all(inertia > 0.0):
            inertia = np.ones(3)
        generalised_mass = np.concatenate([np.full(3, max(body.mass, 1e-9)), inertia])
        tau = -k * nu / (1.0 + k * self.timestep / generalised_mass)
        return Wrench(tau[:3], tau[3:])
