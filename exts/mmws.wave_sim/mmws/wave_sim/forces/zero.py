"""No hydrodynamic forces at all - the control case.

With this model selected the vessel is a plain rigid body under gravity: it
should fall. That is the point. It confirms the force path is wired up and that
the physics engine is receiving what you send, before any of the physics is
right. Every force you later see is attributable, because you watched the
baseline first.

Replace this with the real models:

    buoyancy_column    ASTRA's method - sum draft over hull points, scale to a
                       volume. Cheap, stable, vertical-only. Worth implementing
                       first precisely because it is the thing to beat.
    buoyancy_pressure  integrate hydrostatic pressure over the wetted hull
                       triangles. Gives Archimedes exactly with no calibration
                       constant, a true moving centre of pressure, and the
                       horizontal components. Needs waterline clipping.
    fossen_damping     -D(nu) nu, linear plus quadratic.
    added_mass         the hard one. Added mass sits on the left-hand side of
                       the equation of motion, and PhysX owns the integration,
                       so it cannot pass through a force API. Rotational added
                       inertia can go into the rigid body's diagonalInertia
                       (three independent values); translational added mass has
                       only a single scalar mass to live in, and surge, sway and
                       heave want very different numbers. Read before coding.
"""

from __future__ import annotations

from ..registry import FORCES
from .base import BodyState, ForceModel, Wrench


@FORCES.register("zero")
class ZeroForceModel(ForceModel):
    """Returns no force. The vessel falls under gravity."""

    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        return Wrench()
