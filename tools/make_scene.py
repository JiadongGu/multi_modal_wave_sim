"""Sky, sun and water material. Run in Isaac Sim's Script Editor:

    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/make_scene.py").read())

Optional, before the exec:

    SUN_ELEV, SUN_AZ = 58.0, 135.0     # degrees above the horizon, and heading
    BRIGHTNESS = 0.45                  # the whole scene's exposure, see below
    BOAT = "planing"                   # also build and load a hull
    MASS = 771.0                       # kg; defaults per hull type

Why the sky is generated and not downloaded
-------------------------------------------
Isaac's own sky HDRIs live on Nucleus, and this machine's log is full of
"OmniHub failed to launch" - an asset that cannot be fetched is worse than no
asset, because the dome light silently renders black and the scene looks
broken rather than unlit. This writes an equirectangular PNG with numpy and
the standard library, so there is nothing to resolve, nothing to authenticate
against, and it works with the network off.

A dome light samples an equirectangular texture with u spanning azimuth and v
spanning zenith to nadir, so the horizon lands on the middle row. Get that
wrong and the sea meets the sky in the wrong place - which is why the sun is
computed as an angle between direction vectors rather than a distance in
pixels, and why the sun prim below is aimed from the same two angles that
painted the disc. A painted sun the shadows disagree with reads as wrong
immediately, even to someone who could not say why.
"""

import struct
import zlib

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

SKY_PNG = globals().get("SKY_PNG", "/tmp/mmws_sky.png")
SUN_ELEV = float(globals().get("SUN_ELEV", 58.0))
SUN_AZ = float(globals().get("SUN_AZ", 135.0))
WIDTH = int(globals().get("SKY_WIDTH", 2048))
# One dial for the whole scene's exposure. The sea is a mirror, so almost all
# of its brightness is reflected sky - turning the sky down turns the water
# down with it, which is why there is no separate water brightness.
BRIGHTNESS = float(globals().get("BRIGHTNESS", 0.45))
BOAT = globals().get("BOAT", None)      # None, "planing" or "wigley"
MASS = globals().get("MASS", None)      # kg; defaults per hull type below


def _write_png(path, rgb):
    """rgb: (H, W, 3) uint8. PNG is simple enough not to need a dependency."""
    h, w, _ = rgb.shape
    # Every scanline carries a leading filter-type byte; 0 means "none".
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def _sky(width, sun_elev_deg, sun_az_deg, brightness=1.0, turbidity=0.35):
    height = width // 2
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    # Full 2-D grids: the sun term mixes products of both axes with a term
    # that depends on only one, and (H,1) against (H,W) will not stack.
    az, zenith = np.meshgrid(u * 2.0 * np.pi, v * np.pi, indexing="xy")
    elev = np.pi / 2.0 - zenith

    ce = np.cos(elev)
    d = np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(elev)], axis=-1)
    se, sa = np.radians(sun_elev_deg), np.radians(sun_az_deg)
    sun = np.array([np.cos(se) * np.cos(sa), np.cos(se) * np.sin(sa), np.sin(se)])
    cos_sun = np.clip((d * sun).sum(axis=-1), -1.0, 1.0)

    up = np.clip(np.sin(elev), 0.0, 1.0)
    t = up ** (0.45 + turbidity)
    col = (np.array([0.42, 0.52, 0.64]) * (1 - t)[..., None]
           + np.array([0.06, 0.13, 0.30]) * t[..., None])
    # The sun and its glow are NOT scaled by brightness. A dim sky with a
    # still-bright sun is a sky with contrast; scaling everything together
    # just gives a flat grey one.
    col += np.array([1.0, 0.86, 0.66]) * (np.clip(cos_sun, 0, 1) ** 220)[..., None] * 0.9
    col *= brightness
    col += np.array([1.0, 0.97, 0.90]) * (cos_sun > np.cos(np.radians(1.2)))[..., None]
    # Below the horizon a dome light still contributes; black down there reads
    # as a hole under the water rather than as distance.
    below = np.clip(-np.sin(elev), 0.0, 1.0)[..., None]
    col = col * (1 - below) + np.array([0.03, 0.05, 0.07]) * brightness * below
    col += (np.array([0.10, 0.10, 0.08]) * brightness
            * np.exp(-(elev[..., None] / 0.045) ** 2) * 0.5)
    return (np.clip(col, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), sun


def _surface(stage, path, diffuse, roughness, metallic=0.0, opacity=1.0,
             read_display_color=False):
    """A UsdPreviewSurface material. Renders in RTX without MDL or Nucleus."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    if read_display_color:
        # Wire diffuseColor to the mesh's own displayColor primvar, which is
        # where the field writes its foam. Without this the material would
        # win and every whitecap would be painted over in deep-water blue.
        reader = UsdShade.Shader.Define(stage, f"{path}/DisplayColor")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*diffuse))
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "result")
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath("/World"):
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

image, sun_vec = _sky(WIDTH, SUN_ELEV, SUN_AZ, BRIGHTNESS)
_write_png(SKY_PNG, image)
print(f"  wrote {SKY_PNG}  ({image.shape[1]}x{image.shape[0]})")

# --- dome -------------------------------------------------------------------
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateIntensityAttr(1200.0 * BRIGHTNESS)
dome.CreateTextureFileAttr().Set(Sdf.AssetPath(SKY_PNG))
dome.CreateTextureFormatAttr().Set("latlong")
# A latlong dome is authored around the Y axis by convention; this stage is
# Z-up. Newer USD exposes poleAxis for exactly this, and where it does not,
# a -90 degree turn about X does the same job.
pole = False
try:
    dome.CreatePoleAxisAttr().Set("Z")
    pole = True
except Exception:                                     # noqa: BLE001
    xf = UsdGeom.Xformable(dome.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXOp().Set(90.0)
print(f"  dome light   /World/Sky   ({'poleAxis=Z' if pole else 'rotated +90 about X'})")

# --- sun --------------------------------------------------------------------
# Aimed from the same two angles that painted the disc, so the shadows and the
# picture agree. A DistantLight emits along its own -Z, so rotate -Z onto the
# direction the light TRAVELS, which is minus the direction of the sun.
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3000.0)
sun.CreateAngleAttr(0.53)                 # the real sun's angular diameter
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
travel = Gf.Vec3d(*(-sun_vec))
rot = Gf.Rotation(Gf.Vec3d(0, 0, -1), travel)
xf = UsdGeom.Xformable(sun.GetPrim())
xf.ClearXformOpOrder()
xf.AddOrientOp().Set(Gf.Quatf(Gf.Quatd(rot.GetQuat())))
print(f"  sun          /World/Sun   elevation {SUN_ELEV:.0f} deg, heading {SUN_AZ:.0f} deg")

# --- materials --------------------------------------------------------------
water_mat = _surface(stage, "/World/Looks/Water",
                     diffuse=(0.015, 0.055, 0.075), roughness=0.06, metallic=0.0,
                     read_display_color=True)
water = stage.GetPrimAtPath("/World/Water")
if water:
    UsdShade.MaterialBindingAPI.Apply(water).Bind(water_mat)
    print("  water material bound to /World/Water  (dark, roughness 0.06)")
else:
    print("  no /World/Water yet - press Play once, then re-run to bind the water")

hull_mat = _surface(stage, "/World/Looks/Hull",
                    diffuse=(0.82, 0.84, 0.86), roughness=0.35, metallic=0.0)
vessel = stage.GetPrimAtPath("/World/Vessel")
if vessel:
    UsdShade.MaterialBindingAPI.Apply(vessel).Bind(hull_mat)
    print("  hull material bound to /World/Vessel")

# --- the boat ---------------------------------------------------------------
# Optional, and it runs the same two tools you would run by hand - the scene
# builder is a convenience over them, never a second way of doing it.
if BOAT:
    import runpy
    import sys
    from pathlib import Path as _Path

    tools = _Path("/home/asv/dev/multi_modal_wave_sim/tools")
    if "__file__" in dir():
        tools = _Path(__file__).resolve().parent

    defaults = {"planing": 771.0, "wigley": 1406.0}
    if BOAT not in defaults:
        raise ValueError(f"BOAT must be 'planing' or 'wigley', got {BOAT!r}")

    print()
    print(f"  building the {BOAT} hull")
    g = globals()
    g["HULL_TYPE"] = BOAT
    exec(open(tools / "make_hull.py").read(), g)
    g["HULL"] = g["OUT"]
    g["MASS"] = float(MASS) if MASS else defaults[BOAT]
    exec(open(tools / "load_hull.py").read(), g)

print()
print("  the water is dark on purpose: a sea is mostly a mirror, so almost")
print("  everything you see in it is the sky. Make the water bright and it")
print("  stops reflecting and starts looking like plastic. BRIGHTNESS turns")
print("  the sky down, and the sea follows it - that is the whole exposure.")
