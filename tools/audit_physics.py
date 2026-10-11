"""What else in this scene can touch the vessel? Run in the Script Editor:

    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/audit_physics.py").read())

A one-step impulse that reverses a body's velocity is almost always PhysX
depenetration, not an applied force. Something has to be overlapping. This
lists every collider on the stage with its world position, every rigid body,
and the PhysX scene settings that govern how hard it pushes things apart.
"""

import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
now = Usd.TimeCode.Default()

print("\n--- colliders ---")
found = 0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Plane):
        found += 1
        try:
            pos = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(now).ExtractTranslation()
            where = f"at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        except Exception:
            where = "position unavailable"
        enabled = prim.GetAttribute("physics:collisionEnabled")
        state = enabled.Get() if enabled and enabled.IsValid() else "(default on)"
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]  {where}  enabled={state}")
if not found:
    print("  none")

print("\n--- rigid bodies ---")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        m = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        print(f"  {prim.GetPath()}  mass={m}")

print("\n--- everything directly under /World ---")
world = stage.GetPrimAtPath("/World")
if world:
    for child in world.GetChildren():
        kinds = []
        if child.HasAPI(UsdPhysics.CollisionAPI):
            kinds.append("collider")
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            kinds.append("rigidbody")
        n = len(list(Usd.PrimRange(child)))
        print(f"  {child.GetPath()}  [{child.GetTypeName()}]  {n} prims in subtree  {kinds}")

print("\n--- physics scene ---")
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        print(f"  {prim.GetPath()}")
        for name in ("physics:gravityDirection", "physics:gravityMagnitude"):
            a = prim.GetAttribute(name)
            if a and a.IsValid():
                print(f"    {name} = {a.Get()}")
        px = PhysxSchema.PhysxSceneAPI(prim)
        if px:
            for label, attr in (("timeStepsPerSecond", px.GetTimeStepsPerSecondAttr()),
                                ("maxBiasCoefficient", px.GetMaxBiasCoefficientAttr()),
                                ("solverType", px.GetSolverTypeAttr())):
                try:
                    print(f"    {label} = {attr.Get()}")
                except Exception:
                    pass

print("\n--- vessel depenetration limits ---")
v = stage.GetPrimAtPath("/World/Vessel")
if v:
    body = PhysxSchema.PhysxRigidBodyAPI(v)
    if body:
        for label, attr in (("maxDepenetrationVelocity", body.GetMaxDepenetrationVelocityAttr()),
                            ("maxLinearVelocity", body.GetMaxLinearVelocityAttr()),
                            ("linearDamping", body.GetLinearDampingAttr()),
                            ("angularDamping", body.GetAngularDampingAttr())):
            try:
                print(f"  {label} = {attr.Get()}")
            except Exception:
                print(f"  {label} = (unset)")
    else:
        print("  no PhysxRigidBodyAPI - defaults apply")
print()
