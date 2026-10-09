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

Stop the timeline before running this - while it is playing, PhysX owns the
transform and anything set here is overwritten immediately.
"""

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

VESSEL = "/World/Vessel"
MASS = 200.0
POSITION = (150.0, 150.0, 3.0)

stage = omni.usd.get_context().get_stage()

if not stage.GetPrimAtPath("/World"):
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

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

vertices, triangles = box_hull(1.0, 1.0, 1.0)

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
UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(MASS)

xf = UsdGeom.Xformable(prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(*POSITION))

print()
print(f"  {VESSEL}")
print(f"    mesh          {prim.IsA(UsdGeom.Mesh)}  ({len(vertices)} verts, {len(triangles)} tris)")
print(f"    rigid body    {prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
print(f"    collider      {prim.HasAPI(UsdPhysics.CollisionAPI)}  (for mass properties)")
print(f"    mass          {UsdPhysics.MassAPI(prim).GetMassAttr().Get()} kg")
print(f"    position      {POSITION}")
print()
print(f"  expected draft  {MASS / 1000.0:.3f} m   (mass / rho / waterplane area)")
print("  press Play")
