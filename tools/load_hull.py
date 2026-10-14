"""Build /World/Vessel from a real hull file, and report its hydrostatics.

Run in Isaac Sim's Script Editor with the timeline STOPPED:

    HULL = "/home/asv/Downloads/BlueBoat_hulls_only.usdc"
    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/load_hull.py").read())

With no mass it prints a mass-to-draft table and stops, because a hull file
carries geometry and nothing else. Supply one of:

    MASS  = 23.5     kilograms, if you know it
    DRAFT = 0.08     metres, and the mass that floats it there is solved for

A UsdPhysics.MassAPI authored on the source file is used ahead of either, and
the report names where the number came from.

Why this exists
---------------
A cube is a good first hull because the answer is closed form: draft is
mass / (rho * waterplane area), and the waterplane area does not change as it
sinks. Neither is true of a boat. A BlueBoat is two curved pontoons, so the
waterplane area grows with draft and the equilibrium has to be solved for.
The same goes for stability - there is no formula for BM on an arbitrary
hull.

So this script solves both numerically, using the force model itself rather
than a separate derivation. That matters: if the answer came from independent
hydrostatics code, agreement would only prove the two implementations match.
Here the prediction and the simulation share the force model, and what is
being checked is that PhysX integrates it into the equilibrium it implies.
The independent check stays the cube, where the closed form exists.

What is assumed, and what that costs
------------------------------------
  * Uniform density. The centre of mass is placed at the hull's volume
    centroid plus COM_OFFSET. A real BlueBoat carries batteries low, so its
    true KG is lower and its true GM is LARGER than reported here. This is
    the conservative direction, but it is an assumption, not a measurement.
  * Inertia from the bounding box, as a solid box of the given mass. For a
    catamaran this overestimates roll inertia, because the real mass is not
    spread out to the bounding box corners. Overestimating I lengthens the
    roll period, so treat the reported period as an upper bound.
  Both are single constants below. Replace them with measured values when
  anyone has them.
"""

import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

VESSEL = "/World/Vessel"
HULL = globals().get("HULL", "/home/asv/Downloads/BlueBoat_hulls_only.usdc")
# No default, deliberately. A hull file carries geometry, not mass, and a mass
# invented to make the script run is an input nobody chose that every later
# number depends on: the draft, the waterplane area, GM, the roll period, and
# therefore whether the hull looks stable. Set MASS, or set DRAFT and let the
# script solve for the mass that floats the hull there.
MASS = globals().get("MASS", None)
DRAFT = globals().get("DRAFT", None)
COM_OFFSET = np.array(globals().get("COM_OFFSET", (0.0, 0.0, 0.0)), dtype=float)
RHO = 1000.0
GRAVITY = 9.80665

import sys
from pathlib import Path

# The package is already importable - the extension is enabled, so Kit put
# it on sys.path. Deriving a path from __file__ here would point at Kit's
# throwaway /tmp/carb.XXXX/script_NNN.py instead.

from mmws.wave_sim.forces import BodyState
from mmws.wave_sim.integration.repair import repair
from mmws.wave_sim.registry import FIELDS, FORCES

# --- read every mesh out of the hull file -----------------------------------
# Opened as its own stage rather than referenced onto ours. A reference brings
# the file's own transforms, default prim and layer structure with it, and any
# of those being unexpected shows up much later as a hull in the wrong place.
# Reading the points out and authoring one clean mesh has no such failure.
src = Usd.Stage.Open(HULL)
if src is None:
    raise RuntimeError(f"could not open {HULL}")

parts = []
visual_parts = []
for prim in src.Traverse():
    m = UsdGeom.Mesh(prim)
    if not m:
        continue
    # Anything called Super or Visual is scenery, not structure. Merging it
    # into the hull would wreck the enclosed volume and the centroid every
    # later number is measured from - and it sits above the waterline, so it
    # displaces nothing and belongs in no force model.
    visual = prim.GetName().lower().startswith(("super", "visual"))
    pts = m.GetPointsAttr().Get()
    counts = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if pts is None or counts is None or idx is None:
        continue
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()))
    v = np.array(pts, dtype=float)
    # USD is row-vector: p' = p * M, translation in the last ROW.
    v = v @ xf[:3, :3] + xf[3, :3]
    tris = []
    off = 0
    for n in counts:
        for k in range(1, n - 1):          # fan triangulation
            tris.append((idx[off], idx[off + k], idx[off + k + 1]))
        off += n
    if visual:
        visual_parts.append((v, np.array(tris, dtype=int)))
        print(f"  read {prim.GetPath()}  {len(v)} pts, {len(tris)} tris  (visual)")
    else:
        parts.append((v, np.array(tris, dtype=int)))
        print(f"  read {prim.GetPath()}  {len(v)} pts, {len(tris)} tris")

if not parts:
    raise RuntimeError(f"no UsdGeom.Mesh found anywhere in {HULL}")

# If whoever authored the file put a mass on it, that beats anything typed at
# the prompt, and the report says where it came from.
mass_source = "you"
if MASS is None:
    for prim in src.Traverse():
        api = UsdPhysics.MassAPI.Get(src, prim.GetPath())
        if api:
            value = api.GetMassAttr().Get()
            if value is not None and value > 0.0:
                MASS, mass_source = float(value), f"{prim.GetPath()} in the hull file"
                break

vertices, triangles, report = repair(parts)
volume = float(report["volume"])
open_count = int(report["open_edges_after"])
print()
print(f"  merged -> {report['vertices']} pts, {report['triangles']} tris")
print(f"  open edges   {report['open_edges_before']} before welding -> {open_count} after"
      f"   {'(watertight)' if open_count == 0 else '<-- NOT CLOSED'}")
print(f"  volume       {volume:.5f} m^3   (displaces {volume * RHO:.1f} kg fully submerged)")
if open_count:
    raise RuntimeError("hull is not watertight; pressure integration needs a closed surface")

# --- re-centre on the volume centroid ---------------------------------------
# Every moment in the force model is taken about the centre of mass, so the
# body origin and the centre of mass want to be the same point unless there is
# a reason otherwise. Tetrahedron decomposition about the origin: each face
# with the origin forms a tet of signed volume dot(a, b x c)/6 whose centroid
# is (a + b + c)/4, and the volume-weighted mean of those is the centroid of
# the whole closed solid.
a, b, c = (vertices[triangles[:, i]] for i in range(3))
tet_vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
centroid = (tet_vol[:, None] * (a + b + c) / 4.0).sum(axis=0) / tet_vol.sum()
vertices = vertices - centroid
print(f"  centroid     {centroid.round(4)} m  (subtracted; body origin is now the centroid)")

lo, hi = vertices.min(axis=0), vertices.max(axis=0)
extents = hi - lo
print(f"  extents      {extents.round(3)} m")

# --- resolve the mass -------------------------------------------------------
field = FIELDS.create("flat")
model = FORCES.create("buoyancy_pressure", vertices=vertices, triangles=triangles)


def lift_at(z):
    """Upward buoyancy, in newtons, with the body centre at height z.

    Independent of mass - buoyancy is a property of the displaced water, so
    the BodyState mass here is a placeholder the model never reads.
    """
    state = BodyState(position=np.array([0.0, 0.0, float(z)]), mass=1.0)
    return float(model.wrench(state, field, 0.0).force[2])


def settle(mass):
    """Height of the body centre at which buoyancy balances this weight.

    Buoyancy falls monotonically as the hull rises, so bisection cannot miss.
    Bracket: fully under (lift = rho*g*V) to fully clear (lift = 0).
    """
    weight = mass * GRAVITY
    a_z, b_z = float(lo[2]) - 1.0, float(hi[2]) + 1.0
    for _ in range(80):
        mid = 0.5 * (a_z + b_z)
        a_z, b_z = (mid, b_z) if lift_at(mid) > weight else (a_z, mid)
    return 0.5 * (a_z + b_z)


if MASS is None and DRAFT is not None:
    # The mass that floats the hull at the draft you asked for is just the
    # buoyancy with the keel that far under, divided by g. No iteration.
    MASS = lift_at(-(float(DRAFT) + float(lo[2]))) / GRAVITY
    mass_source = f"solved from DRAFT = {float(DRAFT):.3f} m"

if MASS is None:
    print()
    print("  No mass. This file carries geometry only, and a mass invented here")
    print("  would silently set the draft, the GM and the roll period.")
    print()
    print("  Pick one:")
    print("    MASS  = <kg>     if you know it")
    print("    DRAFT = <m>      and the mass that floats it there is solved for")
    print()
    print("  mass -> draft for this hull, to check against a photo or a spec:")
    print("      mass (kg)   draft (m)   submerged")
    for frac in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        m = frac * volume * RHO
        print(f"      {m:8.1f}   {-(settle(m) + float(lo[2])):9.4f}   {100 * frac:5.0f}%")
    print(f"      {volume * RHO:8.1f}   {float(hi[2] - lo[2]):9.4f}     100%  (awash)")
    raise SystemExit("  set MASS or DRAFT and run again")

MASS = float(MASS)
if MASS >= volume * RHO:
    raise RuntimeError(
        f"mass {MASS:.1f} kg exceeds full displacement {volume * RHO:.1f} kg - it will sink")
print(f"  mass         {MASS:.2f} kg   (from: {mass_source})")

z_eq = settle(MASS)

# Draft is measured DOWN from the waterline to the keel. The keel sits at
# world height z_eq + lo[2], and the flat water is at z = 0, so the depth of
# the keel below the surface is -(z_eq + lo[2]). Getting this sign wrong
# reports the hull's full height minus its draft and looks plausible.
draft = -(z_eq + float(lo[2]))
displaced = MASS / RHO
# Waterplane area is dV/dz, which is d(lift)/dz / (rho*g). Central difference.
h = 1e-4
waterplane = (lift_at(z_eq - h) - lift_at(z_eq + h)) / (2 * h) / (RHO * GRAVITY)

# --- GM, straight out of the force model ------------------------------------
# Heel the hull a small angle and read the restoring moment. Moments in the
# model are taken about the centre of mass and gravity has no moment about its
# own centre of mass, so what comes back IS the righting moment:
#     M = rho * g * V_sub * GM * sin(theta)
# Done at two angles to confirm the response is still linear - if GM differs
# between them the hull is already outside the small-angle region and a single
# number does not describe it.
def gm_at(deg):
    th = np.radians(deg)
    q = np.array([np.cos(th / 2), np.sin(th / 2), 0.0, 0.0])   # roll, (w,x,y,z)
    state = BodyState(position=np.array([0.0, 0.0, z_eq]), orientation=q,
                      mass=MASS, center_of_mass=COM_OFFSET)
    w = model.wrench(state, field, 0.0)
    return -float(w.moment[0]) / (RHO * GRAVITY * displaced * np.sin(th))


gm1, gm2 = gm_at(1.0), gm_at(3.0)
gm = gm1

print()
print("  --- hydrostatics at equilibrium ---")
print(f"    mass             {MASS:.1f} kg")
print(f"    displaced        {displaced:.4f} m^3  ({100 * displaced / volume:.1f}% of the hull)")
print(f"    body centre z    {z_eq:+.4f} m   (water at z = 0)")
print(f"    draft            {draft:.4f} m   (waterline above the keel)")
print(f"    waterplane area  {waterplane:.4f} m^2")
print(f"    GM               {gm:+.4f} m   (at 3 deg: {gm2:+.4f} m)")
if abs(gm2 - gm1) > 0.2 * abs(gm1):
    print("      ^ GM changes a lot between 1 and 3 deg: not a small-angle hull,")
    print("        treat the single number with suspicion")

heave_period = 2 * np.pi * np.sqrt(MASS / (RHO * GRAVITY * waterplane))
inertia = np.array([
    MASS / 12.0 * (extents[1] ** 2 + extents[2] ** 2),
    MASS / 12.0 * (extents[0] ** 2 + extents[2] ** 2),
    MASS / 12.0 * (extents[0] ** 2 + extents[1] ** 2),
])
if gm > 0:
    roll_period = 2 * np.pi * np.sqrt(inertia[0] / (RHO * GRAVITY * displaced * gm))
else:
    roll_period = float("inf")
print(f"    heave period     {heave_period:.2f} s")
print(f"    roll period      {roll_period:.2f} s   (upper bound; bounding-box inertia)")
if gm <= 0:
    print("      ^ GM <= 0: this hull has no righting moment and will capsize")

# --- author the prim --------------------------------------------------------
stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath("/World"):
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
if not stage.GetPrimAtPath("/World/PhysicsScene"):
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

for prim in list(stage.Traverse()):
    path = str(prim.GetPath())
    if path.startswith("/World/") and path not in (VESSEL, "/World/Water"):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            stage.RemovePrim(path)
            print(f"  removed {path} (stray rigid body or collider)")
if stage.GetPrimAtPath(VESSEL):
    stage.RemovePrim(VESSEL)

mesh = UsdGeom.Mesh.Define(stage, VESSEL)
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.ravel().astype(np.int32)))
mesh.CreateSubdivisionSchemeAttr().Set("none")
mesh.CreateExtentAttr(Vt.Vec3fArray([
    Gf.Vec3f(*(float(v) for v in lo)), Gf.Vec3f(*(float(v) for v in hi)),
]))

prim = mesh.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.CollisionAPI.Apply(prim)
mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateMassAttr(MASS)
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in COM_OFFSET)))
mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in inertia)))
mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

try:
    from pxr import PhysxSchema

    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateMaxDepenetrationVelocityAttr(1.0)
    rb.CreateMaxLinearVelocityAttr(20.0)
    rb.CreateMaxAngularVelocityAttr(180.0)
    sc = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    sc.CreateMaxBiasCoefficientAttr(0.8)
except ImportError:
    print("  PhysxSchema unavailable - velocity caps NOT applied")

# Park it on its own waterline, upright, at rest, in the middle of whatever
# water patch the extension currently has.
import mmws.wave_sim.extension as E

ext = E.get()
centre = ext.field.tile_size / 2.0 if (ext and ext.field) else 64.0
# The superstructure rides as a CHILD of the vessel, so it inherits every
# transform PhysX writes without being part of the body in any other way: no
# collider, no mass, never seen by the force model.
if visual_parts:
    sv = np.concatenate([p[0] for p in visual_parts], axis=0) - centroid
    sf, off = [], 0
    for pv, pf in visual_parts:
        sf.append(pf + off)
        off += len(pv)
    sf = np.concatenate(sf, axis=0)
    sup = UsdGeom.Mesh.Define(stage, f"{VESSEL}/Super")
    sup.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(sv.astype(np.float32)))
    sup.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(sf), 3, np.int32)))
    sup.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(sf.ravel().astype(np.int32)))
    sup.CreateSubdivisionSchemeAttr().Set("none")
    sup.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(*(float(v) for v in sv.min(axis=0))),
        Gf.Vec3f(*(float(v) for v in sv.max(axis=0))),
    ]))
    print(f"  {VESSEL}/Super  {len(sf)} tris, visual only (no collider, no mass)")

xf = UsdGeom.Xformable(prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(float(centre), float(centre), float(z_eq)))
xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
rb_api = UsdPhysics.RigidBodyAPI(prim)
rb_api.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
rb_api.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))

print()
print(f"  {VESSEL} placed at ({centre:.1f}, {centre:.1f}, {z_eq:+.4f}), upright, at rest")
print(f"  PASS on flat water = z holds at {z_eq:+.4f} and tilt stays near 0")
print("  select(field='flat') first, then press Play")

# Forget the inputs: the Script Editor keeps globals between runs, so a
# value set once would silently apply to every run after it.
for _name in ('HULL', 'MASS', 'DRAFT', 'COM_OFFSET'):
    globals().pop(_name, None)
