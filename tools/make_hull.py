"""Write a Wigley-form hull to a USD file, for load_hull.py to pick up.

Run in Isaac Sim's Script Editor:

    LENGTH, BEAM, DEPTH = 6.10, 2.72, 0.60      # J/24 waterline dimensions
    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/make_hull.py").read())

then feed the file it writes to load_hull.py with the published displacement:

    HULL, MASS = "/tmp/wigley_hull.usda", 1406.0
    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/load_hull.py").read())

Why a generated hull and not a downloaded one
---------------------------------------------
Every hull file so far has arrived without the one number that matters. The
BlueBoat mesh has no mass, so its draft was whatever anyone guessed. A Wigley
hull has the opposite problem solved: the geometry is a formula, so there is
nothing to download, no licence, no unit ambiguity and nothing to repair, and
real boats come with published displacements that can be typed in beside it.

It is also the hull Journee (1992) towed in regular waves, which is the
standard benchmark for measured heave and pitch RAOs. Flotation is checked
today; when this project needs to show it predicts MOTION, that is the data
to answer to.

Picking dimensions
------------------
Use the WATERLINE length, not the overall length - the overhangs of a yacht
carry no displacement at rest. DEPTH is the canoe body depth, NOT the quoted
draft: a yacht's published draft is mostly fin keel, which displaces almost
nothing. A J/24 draws 1.2 m with its keel and about 0.3 m without it.

FULLNESS and TAPER raise the exponents in

    y = (B/2) (1 - (2x/L)^2)^FULLNESS (1 - (z/T)^2)^TAPER

Both 1.0 is the classic Wigley, Cb = 4/9. A yacht canoe body is finer than
that, so raise them until the reported Cb looks like the boat you mean.
"""

import numpy as np
from pxr import Usd, UsdGeom, Vt

import sys
from pathlib import Path

here = Path(__file__).resolve().parent if "__file__" in dir() else Path(
    "/home/asv/dev/multi_modal_wave_sim/tools")
pkg = here.parent / "exts" / "mmws.wave_sim"
if str(pkg) not in sys.path:
    sys.path.insert(0, str(pkg))

from mmws.wave_sim.forces.hulls import planing_hull, wigley_hull
from mmws.wave_sim.integration.repair import repair

# "wigley"  - the analytic benchmark hull. Sailing-yacht-ish, and the one with
#             published towing tank data (Journee 1992).
# "planing" - a warped-V outboard runabout, from principal dimensions. The
#             defaults are a Boston Whaler 170 Montauk: 5.28 x 2.21 m, 16 deg
#             of deadrise at the transom, 771 kg dry, 0.30 m published draft.
HULL_TYPE = str(globals().get("HULL_TYPE", "wigley")).lower()
_WHALER = HULL_TYPE == "planing"

OUT = globals().get("OUT", f"/tmp/{HULL_TYPE}_hull.usda")
LENGTH = float(globals().get("LENGTH", 5.28 if _WHALER else 6.10))
BEAM = float(globals().get("BEAM", 2.21 if _WHALER else 2.72))
DEPTH = float(globals().get("DEPTH", 0.60))
STATIONS = int(globals().get("STATIONS", 41))
SECTIONS = int(globals().get("SECTIONS", 7 if _WHALER else 17))
# wigley only
FULLNESS = float(globals().get("FULLNESS", 1.0))
TAPER = float(globals().get("TAPER", 1.0))
# planing only
DEADRISE_TRANSOM = float(globals().get("DEADRISE_TRANSOM", 16.0))
DEADRISE_BOW = float(globals().get("DEADRISE_BOW", 50.0))

if _WHALER:
    vertices, triangles = planing_hull(
        length=LENGTH, beam=BEAM, depth=DEPTH, stations=STATIONS,
        sections=SECTIONS, deadrise_transom=DEADRISE_TRANSOM,
        deadrise_bow=DEADRISE_BOW,
    )
elif HULL_TYPE == "wigley":
    vertices, triangles = wigley_hull(
        length=LENGTH, beam=BEAM, depth=DEPTH, stations=STATIONS,
        sections=SECTIONS, fullness=FULLNESS, taper=TAPER,
    )
else:
    raise ValueError(f"HULL_TYPE must be 'wigley' or 'planing', got {HULL_TYPE!r}")
# Round-trip through the same repair the bridge uses, so what gets written is
# what the force model would have seen anyway.
vertices, triangles, report = repair([(vertices, triangles)])
block = LENGTH * BEAM * DEPTH

stage = Usd.Stage.CreateNew(OUT) if not Path(OUT).exists() else Usd.Stage.Open(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
mesh = UsdGeom.Mesh.Define(stage, "/Hull")
stage.SetDefaultPrim(mesh.GetPrim())
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.ravel().astype(np.int32)))
mesh.CreateSubdivisionSchemeAttr().Set("none")
stage.GetRootLayer().Save()

print()
print(f"  wrote {OUT}")
print(f"    type           {HULL_TYPE}")
print(f"    dimensions     {LENGTH:.2f} x {BEAM:.2f} m, depth {DEPTH:.2f} m"
      + (f", deadrise {DEADRISE_TRANSOM:.0f}-{DEADRISE_BOW:.0f} deg" if _WHALER else ""))
print(f"    mesh           {report['vertices']} pts, {report['triangles']} tris, "
      f"watertight={report['watertight']}")
print(f"    hull volume    {report['volume']:.3f} m^3")
print(f"    Cb             {report['volume'] / block:.3f}   "
      + ("(a planing hull's block coefficient is low: it is mostly air above "
         "the chine)" if _WHALER else
         "(classic Wigley is 0.444; a yacht canoe body is finer)"))
print()
print("    full displacement at this depth:")
for rho, label in ((1000.0, "fresh"), (1025.0, "sea")):
    print(f"      {label:6s} water   {report['volume'] * rho:8.1f} kg if submerged to the deck")
print()
print("  now run load_hull.py with this file and the boat's published displacement:")
print(f"    HULL, MASS = {OUT!r}, <kg>")
