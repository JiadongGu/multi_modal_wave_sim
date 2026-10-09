"""The one file that talks to USD.

Everything awkward about the boundary - quaternion order, degrees, frames, the
row-vector convention - already lives in convert.py, where fifteen offline
tests cover it. What is left here is fetching attributes and handing them over,
which is why this file is thin.

It is also the only module in the project with no test coverage, because
verifying it means launching Isaac Sim. Keep it boring.

    bind()        once   find the prims, read the hull, apply PhysxForceAPI
    read_state()  60 Hz  pose and velocities -> BodyState
    apply()       60 Hz  a Wrench -> PhysxForceAPI

Assumptions, each of which raises rather than guesses:
* the body prim has UsdPhysics.RigidBodyAPI
* the hull prim is a UsdGeom.Mesh (the body prim itself, by default)
* mass properties come from UsdPhysics.MassAPI when it is present

Known limitation: when the hull prim IS the body prim, the child-to-parent
transform is the identity, so a scale authored on that prim is not applied to
the hull points. Author the mesh at its real size, or put the hull on a child
prim. Fixing it properly means pulling the scale out of the local-to-world
transform separately from the rigid part.
"""

from __future__ import annotations

import numpy as np

from ..forces import BodyState, Wrench
from .repair import open_edges, orient_faces, signed_volume, weld_vertices
from .convert import (
    body_state_from_usd,
    quat_to_wxyz,
    relative_transform,
    transform_points,
    triangulate,
)


class VesselBridge:
    """Binds one USD rigid body to the force model. Resolve once, use every step."""

    def __init__(self, stage, body_path: str, hull_path: str = None) -> None:
        self.stage = stage
        self.body_path = str(body_path)
        self.hull_path = str(hull_path or body_path)
        self.vertices = None
        self.triangles = None
        self.mass = 1.0
        self.inertia = np.ones(3)
        self.center_of_mass = np.zeros(3)
        self._force_api = None

    def bind(self) -> None:
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        body = self.stage.GetPrimAtPath(self.body_path)
        if not body or not body.IsValid():
            raise ValueError(f"no prim at {self.body_path}")
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"{self.body_path} has no RigidBodyAPI")

        mesh = UsdGeom.Mesh.Get(self.stage, self.hull_path)
        if not mesh:
            raise ValueError(f"no UsdGeom.Mesh at {self.hull_path}")

        now = Usd.TimeCode.Default()
        mesh_to_world = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(now))
        body_to_world = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(now))

        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        self.vertices = transform_points(points, relative_transform(mesh_to_world, body_to_world))
        self.triangles = triangulate(
            mesh.GetFaceVertexCountsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),
        )

        # Authored meshes cannot be trusted to wind consistently, and an
        # inward-facing face has its pressure pulling instead of pushing. The
        # side faces then add instead of cancelling, so a cube in still water
        # picks up kilonewtons of sideways force and the system stops being
        # conservative - the hull gains energy every cycle until it leaves.
        # Weld first so neighbouring faces actually share edges, otherwise the
        # adjacency walk has nothing to walk.
        self.vertices, self.triangles = weld_vertices(self.vertices, self.triangles)
        self.triangles = orient_faces(self.vertices, self.triangles)
        self.open_edges = open_edges(self.triangles)
        self.volume = signed_volume(self.vertices, self.triangles)
        if self.open_edges:
            raise ValueError(
                f"hull is not watertight: {self.open_edges} open edges. "
                "Pressure integration needs a closed surface."
            )
        if self.volume <= 0.0:
            raise ValueError(f"hull encloses no volume (got {self.volume:.6f} m^3)")

        mass_api = UsdPhysics.MassAPI.Get(self.stage, self.body_path)
        if mass_api:
            value = mass_api.GetMassAttr().Get()
            if value is not None and value > 0.0:
                self.mass = float(value)
            # USD uses sentinels for "unspecified, let PhysX work it out":
            # centerOfMass defaults to (-inf, -inf, -inf) and diagonalInertia
            # to (0, 0, 0). Reading either literally is poison - an infinite
            # centre of mass makes every moment arm infinite, PhysX receives a
            # NaN torque, and the body leaves the scene at implausible speed.
            value = mass_api.GetDiagonalInertiaAttr().Get()
            if value is not None:
                inertia = np.array(value, dtype=float)
                if np.all(np.isfinite(inertia)) and np.any(inertia > 0.0):
                    self.inertia = inertia
            value = mass_api.GetCenterOfMassAttr().Get()
            if value is not None:
                com = np.array(value, dtype=float)
                if np.all(np.isfinite(com)):
                    self.center_of_mass = com

        self._force_api = PhysxSchema.PhysxForceAPI.Apply(body)
        self._force_api.CreateWorldFrameEnabledAttr(False)
        self._force_api.CreateForceEnabledAttr(True)
        self._force_api.CreateForceAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        self._force_api.CreateTorqueAttr(Gf.Vec3f(0.0, 0.0, 0.0))

    def read_state(self) -> BodyState:
        from pxr import Usd, UsdGeom

        body = self.stage.GetPrimAtPath(self.body_path)
        xf = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return body_state_from_usd(
            position=np.array(xf.ExtractTranslation(), dtype=float),
            orientation=quat_to_wxyz(xf.ExtractRotationQuat()),
            linear_velocity=body.GetAttribute("physics:velocity").Get(),
            angular_velocity_deg=body.GetAttribute("physics:angularVelocity").Get(),
            mass=self.mass,
            inertia=self.inertia,
            center_of_mass=self.center_of_mass,
        )

    def apply(self, wrench: Wrench) -> None:
        from pxr import Gf

        if self._force_api is None:
            raise RuntimeError("bind() before apply()")
        f, m = wrench.force, wrench.moment
        # Never hand PhysX a non-finite wrench. One NaN torque is enough to
        # fling a body out of the scene, and the symptom looks nothing like
        # its cause, so fail here with something readable instead.
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(m))):
            raise ValueError(f"non-finite wrench: force={f}, moment={m}")
        self._force_api.GetForceAttr().Set(Gf.Vec3f(float(f[0]), float(f[1]), float(f[2])))
        self._force_api.GetTorqueAttr().Set(Gf.Vec3f(float(m[0]), float(m[1]), float(m[2])))