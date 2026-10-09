"""USD values to plain numpy - the error-prone half of the bridge.

Nothing here imports USD. Every function takes plain numbers or arrays and
returns plain numbers or arrays, so the conversions can be tested on a laptop.
The thin layer that actually reads prims lives in usd_bridge.py and does
nothing but fetch attributes and call these.

Three conventions differ between USD and the rest of this package, and each one
fails silently if you get it wrong:

* **Quaternion order.** Gf.Quatd keeps the scalar separate: GetReal() is w,
  GetImaginary() is (x, y, z). Our BodyState wants a flat (w, x, y, z). A
  (x, y, z, w) ordering still normalises, still rotates, and is wrong.
* **Angular velocity units.** USD reports it in DEGREES per second. Everything
  downstream is radians. A factor of 57.3 in a damping term looks like an
  unstable solver, not a unit bug.
* **Frames.** USD velocities are world-frame; BodyState is body-frame, because
  that is what PhysxForceAPI wants back. Skipping the rotation only shows up
  once the vessel turns, by which time you are debugging the wrong thing.
"""

from __future__ import annotations

import numpy as np

from ..forces import BodyState, quat_to_matrix


def quat_to_wxyz(q):
    """Gf.Quatd, or any 4-sequence, to a (w, x, y, z) array."""
    # TODO(jiadong): see the module docstring.
    raise NotImplementedError("quat_to_wxyz is not implemented yet")


def triangulate(face_vertex_counts, face_vertex_indices):
    """USD face lists to a (T, 3) triangle array, by fan triangulation.

    A UsdGeom.Mesh may hold quads or larger n-gons; the force model needs
    triangles. A convex face [v0 .. vn-1] becomes n-2 triangles all sharing v0,
    which preserves the winding, so normals keep pointing the same way.
    """
    # TODO(jiadong): see the module docstring.
    raise NotImplementedError("triangulate is not implemented yet")


def body_state_from_usd(position, orientation, linear_velocity=None,
                        angular_velocity_deg=None, mass: float = 1.0,
                        inertia=None, center_of_mass=None) -> BodyState:
    """Build a BodyState from what USD hands you.

    position, linear_velocity and angular_velocity_deg are WORLD frame, as USD
    reports them. The returned BodyState carries body-frame velocities, in
    radians per second.
    """
    # TODO(jiadong): see the module docstring.
    raise NotImplementedError("body_state_from_usd is not implemented yet")
