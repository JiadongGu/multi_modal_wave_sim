"""Build the scene and leave it running, ready to drive. Via MMWS_AUTORUN:

    MMWS_AUTORUN=$PWD/tools/start_scene.py ./tools/run_isaac.sh --stream

Everything is an environment variable so a session can be started from a
shell without touching the Script Editor:

    MMWS_BOAT=planing|wigley|survey   MMWS_FIELD=fft|flat|wake
    MMWS_HS=0.5  MMWS_TP=4.0          MMWS_PLAY=0 to build without starting

Defaults are a Boston Whaler in a 0.5 m, 4 s sea. Hs 2 m on a 5.3 m boat -
the slider default - is a survival condition, not a working one.
"""

import os
from pathlib import Path

import omni.timeline

TOOLS = Path(__file__).resolve().parent

BOAT = os.environ.get("MMWS_BOAT", "planing")
FIELD = os.environ.get("MMWS_FIELD", "fft")
HS = float(os.environ.get("MMWS_HS", 0.5))
TP = float(os.environ.get("MMWS_TP", 4.0))
PLAY = os.environ.get("MMWS_PLAY", "1") != "0"

print(f"[start_scene] building {BOAT}")
g = {"BOAT": BOAT, "SUN_ELEV": 58.0, "SUN_AZ": 135.0, "BRIGHTNESS": 0.45}
exec(open(TOOLS / "make_scene.py").read(), g)

import mmws.wave_sim.extension as E                      # noqa: E402

ext = E.get()
ext.activate(True)
if ext.vessel is None:
    ext.bind_vessel()

kwargs = {"tile_size": 256.0, "resolution": 64}
if FIELD == "wake":
    # Wake on top of the spectral sea, not instead of it.
    kwargs = {"tile_size": 256.0, "resolution": 64, "base": "fft",
              "amplitude": 0.15, "wavelength": 6.0}
ext.select(spectrum="jonswap",
           spectrum_kwargs={"hs": HS, "tp": TP, "spread_s": 5.0},
           field=FIELD, field_kwargs=kwargs)

# Frame the hull, or the camera stares at empty sky and the scene looks dead.
ext.follow_vessel(True)
ext.keyboard_drive(True)

print(f"[start_scene] {BOAT} on {FIELD}, Hs={HS} Tp={TP}")
print(f"[start_scene] force_lead={ext.force_lead} rot_lead={ext.rot_lead}")
print("[start_scene] drive: I/K ahead and astern, J/L turn")
if PLAY:
    omni.timeline.get_timeline_interface().play()
    print("[start_scene] playing")
