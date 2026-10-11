"""Build the demo scene from nothing. Run inside Isaac Sim's Script Editor:

    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/setup_demo_scene.py").read())

Creates a physics scene and a 1 m cube at /World/Vessel with 200 kg of mass,
placed at (150, 150, 3) - inside the water patch, a few metres up.

The cube's geometry comes from forces.box_hull rather than the Create menu, so
the winding is known to be outward and the size is known to be exactly 1 m.
That matters: the whole point of this scene is that the answer is known in
advance. 1 m cube, 200 kg, fresh water -> 0.200 m draft, every time.

A collider IS applied, even though buoyancy needs no contact. PhysX derives a
body's mass properties from its collision shapes, and a rigid body with none
may not honour MassAPI at all - which shows up as a hull accelerating as if it
weighed a fraction of what you set. Nothing else in this scene has a collider,
so there is nothing for it to hit.

Everything else under /World that carries a rigid body or a collider is
deleted first. A second dynamic body in the scene is not a second experiment,
it is a contaminant: it falls, it collides, and its impulses arrive in our
trace looking like forces we applied.

Stop the timeline before running this - while it is playing, PhysX owns the
transform and anything set here is overwritten immediately.
"""

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

VESSEL = "/World/Vessel"
WATER = "/World/Water"
MASS = 200.0
# x, y must land inside the water patch, which spans [0, tile_size) - see
# field/base.py mesh_points(). 64 is the centre of the 128 m default tile.
POSITION = (64.0, 64.0, 3.0)
SIZE = (1.0, 1.0, 1.0)

stage = omni.usd.get_context().get_stage()

if not stage.GetPrimAtPath("/World"):
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

# --- clear out every other physics-active prim -------------------------------
# Not just /World/Vessel. Anything with a rigid body falls; anything with a
# collider can be depenetrated against. Either one injects impulses that look
# exactly like a bug in the force model.
doomed = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path in (VESSEL, WATER) or not path.startswith("/World/"):
        continue
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
        doomed.append(path)
for path in doomed:
    stage.RemovePrim(path)
    print(f"removed {path} (stray rigid body or collider)")

if not stage.GetPrimAtPath("/World/PhysicsScene"):
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    print("created /World/PhysicsScene")

if stage.GetPrimAtPath(VESSEL):
    stage.RemovePrim(VESSEL)
    print(f"removed the old {VESSEL}")

import sys
from pathlib import Path

here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
pkg = here.parent / "exts" / "mmws.wave_sim"
if str(pkg) not in sys.path:
    sys.path.insert(0, str(pkg))
from mmws.wave_sim.forces import box_hull

vertices, triangles = box_hull(*SIZE)

mesh = UsdGeom.Mesh.Define(stage, VESSEL)
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.ravel().astype(np.int32)))
mesh.CreateSubdivisionSchemeAttr().Set("none")
# Gf.Vec3f takes Python floats, not numpy scalars - unpacking a numpy array
# straight into it raises an ArgumentError that names every overload but the
# one you wanted.
mesh.CreateExtentAttr(Vt.Vec3fArray([
    Gf.Vec3f(*(float(v) for v in vertices.min(axis=0))),
    Gf.Vec3f(*(float(v) for v in vertices.max(axis=0))),
]))

prim = mesh.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.CollisionAPI.Apply(prim)

mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateMassAttr(MASS)
# centerOfMass defaults to (-inf, -inf, -inf), the "work it out from the
# geometry" sentinel. Reading it back and using it as a moment arm gives an
# infinite lever and a NaN torque, so author it explicitly.
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
# Solid box about its centre: I = m/12 * (a^2 + b^2) per axis.
sx, sy, sz = SIZE
mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(
    float(MASS / 12.0 * (sy * sy + sz * sz)),
    float(MASS / 12.0 * (sx * sx + sz * sz)),
    float(MASS / 12.0 * (sx * sx + sy * sy)),
))
mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

xf = UsdGeom.Xformable(prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(*POSITION))

# --- cap what PhysX is allowed to do on its own ------------------------------
# maxDepenetrationVelocity is the speed limit on penetration recovery. Left
# unbounded, an overlap of a few centimetres is resolved in one 60 Hz step at
# tens of metres per second - which arrives in the trace as a huge dv on a step
# where our applied force was zero. A floating hull never needs to be pushed
# out of anything faster than 1 m/s. maxLinearVelocity then makes any remaining
# runaway visible as a clamp rather than as a disappearing cube.
try:
    from pxr import PhysxSchema

    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    rb.CreateMaxLinearVelocityAttr(20.0)
    rb.CreateMaxAngularVelocityAttr(180.0)  # deg/s
    print("  applied PhysxRigidBodyAPI velocity caps")

    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    sc = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    sc.CreateMaxBiasCoefficientAttr(0.8)
    print("  set scene maxBiasCoefficient = 0.8 (was unbounded)")
except ImportError:
    print("  PhysxSchema unavailable - velocity caps NOT applied")

print()
print(f"  {VESSEL}")
print(f"    mesh          {prim.IsA(UsdGeom.Mesh)}  ({len(vertices)} verts, {len(triangles)} tris)")
print(f"    rigid body    {prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
print(f"    collider      {prim.HasAPI(UsdPhysics.CollisionAPI)}  (for mass properties)")
print(f"    mass          {UsdPhysics.MassAPI(prim).GetMassAttr().Get()} kg")
print(f"    centre of mass {UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()}")
print(f"    inertia       {UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get()}")
print(f"    position      {POSITION}")
print()
others = [str(p.GetPath()) for p in stage.Traverse()
          if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()) != VESSEL]
print(f"  other rigid bodies in the scene: {others or 'none'}")
print(f"  expected draft  {MASS / 1000.0:.3f} m   (mass / rho / waterplane area)")
print("  press Play")
