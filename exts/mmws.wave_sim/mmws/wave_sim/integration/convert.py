from __future__ import annotations

import numpy as np

from ..forces import BodyState, quat_to_matrix


def quat_to_wxyz(q):
    """Gf.Quatd, or any 4-sequence, to a (w, x, y, z) array."""
    if hasattr(q, "GetReal") and hasattr(q, "GetImaginary"):
        im = q.GetImaginary()
        return np.array([q.GetReal(),im[0],im[1],im[2]], dtype = float)
    a = np.asarray(q,dtype=float).ravel()
    if a.size != 4:
        raise ValueError(f"expected 4 quaternion components, got {a.size}")
    return a


def triangulate(face_vertex_counts, face_vertex_indices):
    """USD face lists to a (T, 3) triangle array, by fan triangulation.

    A UsdGeom.Mesh may hold quads or larger n-gons; the force model needs
    triangles. A convex face [v0 .. vn-1] becomes n-2 triangles all sharing v0,
    which preserves the winding, so normals keep pointing the same way.
    """
    counts = np.asarray(face_vertex_counts, dtype=int).ravel()
    indices = np.asarray(face_vertex_indices,dtype=int).ravel()
    if counts.sum() != indices.size:
        raise ValueError(
            f"counts sum to {counts.sum()} but there are {indices.size} indices"
        )
    tris,at = [],0
    for n in counts:
        if n < 3:
            raise ValueError(f"a face needs at least 3 vertices, got {n}")
        face = indices[at:at + n]
        for k in range(1,n-1):
            tris.append((face[0],face[k],face[k+1]))
        at += n
    return np.array(tris,dtype=int) if tris else np.zeros((0,3),dtype=int)



def body_state_from_usd(position, orientation, linear_velocity=None,
                        angular_velocity_deg=None, mass: float = 1.0,
                        inertia=None, center_of_mass=None) -> BodyState:
    """Build a BodyState from what USD hands you.

    position, linear_velocity and angular_velocity_deg are WORLD frame, as USD
    reports them. The returned BodyState carries body-frame velocities, in
    radians per second.
    """
    # TODO(jiadong): see the module docstring.
    quat = quat_to_wxyz(orientation)
    R = quat_to_matrix(quat)
    lin_world = np.zeros(3) if linear_velocity is None else np.asarray(linear_velocity, dtype=float)
    ang_world = np.zeros(3) if angular_velocity_deg is None else np.deg2rad(
        np.asarray(angular_velocity_deg, dtype=float))
    return BodyState(
        position=np.asarray(position, dtype=float),
        orientation=quat,
        linear_velocity=R.T @ lin_world,
        angular_velocity=R.T @ ang_world,
        mass=float(mass),
        inertia=np.ones(3) if inertia is None else np.asarray(inertia, dtype=float),
        center_of_mass=np.zeros(3) if center_of_mass is None else np.asarray(
            center_of_mass, dtype=float),
    )


def transform_points(points, matrix):
    """Apply a USD 4x4 transform to (M, 3) points.

    USD uses the ROW-vector convention: p' = p * M, and the translation lives
    in the last ROW, not the last column. Most graphics maths you will find
    online uses the column convention, where it is the last column and you
    write M @ p. Mixing them gives a transform that rotates correctly and
    translates to nonsense, which looks like a hull offset bug.
    """
    # TODO(jiadong): see the docstring. Row-vector convention.
    raise NotImplementedError("transform_points is not implemented yet")


def relative_transform(child_to_world, parent_to_world):
    """Child-to-parent transform, both given as USD local-to-world matrices.

    A hull mesh is usually a child of the rigid body prim, often with an offset
    of its own. The force model wants the hull in the BODY frame, so compose:
    child -> world -> parent.

    Row-vector convention again, so it is C @ inv(P), not inv(P) @ C.
    """
    # TODO(jiadong): see the docstring. Row-vector convention.
    raise NotImplementedError("relative_transform is not implemented yet")
