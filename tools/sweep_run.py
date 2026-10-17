"""Flat-water force_lead sweep, start to finish, with nobody at the keyboard.

    MMWS_AUTORUN=$PWD/tools/sweep_run.py ./tools/run_isaac.sh --stream

Builds the scene, binds the hull, switches to flat water, starts the
timeline and runs the sweep. Exists because every hand-driven attempt at
this measurement was confounded by something - two settings changed at once,
four runs overwriting one filename, an amplitude read while it was still
growing - and because the Script Editor needs a human at a GUI, which makes
a run over ssh impossible.

Flat water is the point: on still water a floating hull must settle and stay
settled, so any motion left is the simulator talking to itself.
"""

import os
from pathlib import Path

import omni.timeline

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent

BOAT = os.environ.get("MMWS_SWEEP_BOAT", "planing")
VALUES = [float(v) for v in
          os.environ.get("MMWS_SWEEP_VALUES", "0,0.5,1,1.5,2,3").split(",")]
SETTLE = float(os.environ.get("MMWS_SWEEP_SETTLE", 8.0))
MEASURE = float(os.environ.get("MMWS_SWEEP_MEASURE", 20.0))

print(f"[sweep_run] building the {BOAT} scene")
g = {"BOAT": BOAT, "SUN_ELEV": 58.0, "SUN_AZ": 135.0, "BRIGHTNESS": 0.45}
exec(open(TOOLS / "make_scene.py").read(), g)

import mmws.wave_sim.extension as E                      # noqa: E402

ext = E.get()
ext.activate(True)
if ext.vessel is None:
    ext.bind_vessel()
# Flat water, but keep the tile the scene was built on - a smaller tile would
# leave the hull sitting off the edge of the mesh.
ext.select(field="flat")

print(f"[sweep_run] vessel {getattr(ext.vessel, 'body_path', None)} "
      f"mass {getattr(ext.vessel, 'mass', None)}")
print(f"[sweep_run] sweeping force_lead over {VALUES}, "
      f"{SETTLE:.0f}s settle + {MEASURE:.0f}s measure each "
      f"(~{len(VALUES) * (SETTLE + MEASURE) / 60.0:.1f} min)")

omni.timeline.get_timeline_interface().play()
ext.sweep_lead(VALUES, settle=SETTLE, measure=MEASURE)
