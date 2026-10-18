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
# Thrust per screw, as a fraction of the vessel weight.
THRUST = float(os.environ.get("MMWS_THRUST", 0.25))

print(f"[start_scene] building {BOAT}")
g = {"BOAT": BOAT, "SUN_ELEV": 58.0, "SUN_AZ": 135.0, "BRIGHTNESS": 0.45,
     "CLOUDS": float(os.environ.get("MMWS_CLOUDS", 0.45))}
exec(open(TOOLS / "make_scene.py").read(), g)

import mmws.wave_sim.extension as E                      # noqa: E402

ext = E.get()
ext.thrust_fraction = THRUST      # before binding - bind builds the thruster
ext.activate(True)
if ext.vessel is None:
    ext.bind_vessel()

# 1 m cells, not 4. A 256 m tile at resolution 64 cannot show a wave
# shorter than 8 m, which is most of what a sea looks like.
kwargs = {"tile_size": 256.0, "resolution": 256}
if FIELD == "wake":
    # Wake on top of the spectral sea, not instead of it.
    #
    # And a finer grid, because the default cannot represent a wake at all:
    # a 256 m tile at 64 is 4 m cells against a 6 m wake wavelength, which is
    # below Nyquist. 128 m at 128 gives 1 m cells, six per wave. That used to
    # be unaffordable - points x rings - until wake_height started culling
    # everything outside the track's reach, which cut it 10x.
    kwargs = {"tile_size": 128.0, "resolution": 128, "base": "fft",
              "amplitude": 0.15, "wavelength": 6.0,
              "exaggerate": float(os.environ.get("MMWS_WAKE_EXAGGERATE", 1.0))}
ext.select(spectrum="jonswap",
           spectrum_kwargs={"hs": HS, "tp": TP, "spread_s": 5.0},
           field=FIELD, field_kwargs=kwargs)

# Frame the hull, or the camera stares at empty sky and the scene looks dead.
ext.follow_vessel(True)
ext.keyboard_drive(True)

print(f"[start_scene] {BOAT} on {FIELD}, Hs={HS} Tp={TP}")
print(f"[start_scene] force_lead={ext.force_lead} rot_lead={ext.rot_lead}")
if ext.thruster is not None:
    print(f"[start_scene] thrust {ext.thruster.thrust:.0f} N per screw")
print("[start_scene] drive: I/K ahead and astern, J/L turn")
if PLAY:
    omni.timeline.get_timeline_interface().play()
    print("[start_scene] playing")
