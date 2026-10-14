"""Float the hull on flat water and check the draft against hand arithmetic.

Run in Isaac Sim's Script Editor with the timeline STOPPED:

    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/still_water_test.py").read())

then press Play. The hull should not move.

Why flat water, and why released AT the waterline rather than dropped
--------------------------------------------------------------------
A 1 m cube at 200 kg has a metacentric height of about 17 mm:

    BM = I / V = (1/12) / 0.2   = 0.417 m      waterplane second moment
    KB = draft / 2              = 0.100 m      centre of buoyancy above keel
    KG = 0.5 m                                 centre of a uniform cube
    GM = KB + BM - KG           = 0.017 m

Seventeen millimetres is neutral stability for practical purposes. The cube
will lie over given any excuse at all, and once it does, every force in the
log - which is in the BODY frame - rotates with it. That is why dropping it
from 3 m is the wrong experiment: the drop puts

    E = m g h = 200 * 9.81 * 2.7 = 5300 J

into a body that has almost no righting moment, and what you get back is a
capsize, not a draft. Released at rest on its own waterline there is nothing
to excite and the answer is known in advance:

    draft = mass / (rho * waterplane area) = 200 / (1000 * 1) = 0.200 m
    z     = half-height - draft            = 0.5 - 0.2 = +0.300 m
    depth = water - z                      = -0.300     (the logged column)

PASS is z holding at +0.300 and tilt holding near 0. Both must hold: a hull
that settles at the right draft while slowly rolling over has only told you
that the vertical force is right.
"""

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

import mmws.wave_sim.extension as E

VESSEL = "/World/Vessel"
RHO = 1000.0

ext = E.get()
if ext is None:
    raise RuntimeError("extension is not enabled - turn on mmws.wave_sim first")

# This stage is ours now. The extension writes to no other one,
# so pressing play in an unrelated file does nothing.
ext.activate()

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath(VESSEL)
if not prim:
    raise RuntimeError(f"no {VESSEL} - run tools/setup_demo_scene.py first")

ext.select(field="flat")

mass = float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
lo, hi = pts.min(axis=0), pts.max(axis=0)
area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
half_height = float(hi[2])

draft = mass / (RHO * area)
z_eq = half_height - draft

# Place it exactly on its waterline, upright, at rest. Clearing the op order
# first also clears any rotation left over from a previous run - a hull that
# starts 2 degrees over will keep going over, and the run tells you nothing.
# The water mesh spans [0, tile_size) in x and y, so a hull parked outside
# that square floats next to the water rather than on it. The physics does not
# care - height() is defined everywhere - but you cannot see what you are
# debugging. Centre it on whatever tile the current field uses.
centre = ext.field.tile_size / 2.0

xf = UsdGeom.Xformable(prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(float(centre), float(centre), float(z_eq)))
xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

rb = UsdPhysics.RigidBodyAPI(prim)
rb.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
rb.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))

# Stability of THIS hull, whatever shape it turned out to be.
bm = (hi[0] - lo[0]) * (hi[1] - lo[1]) ** 3 / 12.0 / (draft * area)
gm = draft / 2.0 + bm - half_height

print()
print("  still-water test armed")
print(f"    field            {ext.field!r}")
print(f"    mass             {mass:.1f} kg")
print(f"    waterplane       {area:.3f} m^2")
print(f"    expected draft   {draft:.3f} m")
print(f"    placed at        ({centre:.1f}, {centre:.1f}, {z_eq:+.3f}) m, upright, at rest")
print(f"    water patch      0 .. {ext.field.tile_size:.0f} m square")
print(f"    expected depth   {-z_eq:+.3f}   (the logged column)")
print()
print(f"    GM               {gm:+.3f} m", end="")
print("   <-- marginal; expect it to loll" if gm < 0.1 else "   (stable)")
print()
print("  PASS = z stays at "
      f"{z_eq:+.3f} and tilt stays near 0. Press Play.")
