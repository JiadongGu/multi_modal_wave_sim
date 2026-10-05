#!/usr/bin/env python3
"""Watch a sea surface get built, one step at a time.

    python3 tools/demo_wave_generation.py

A teaching script, not part of the extension. It sums every wave component
directly - slow, obvious, easy to read. The real field/fft.py will do the same
arithmetic with an inverse FFT, which is the same answer computed in
O(N^2 log N) instead of O(N^4). Write that one yourself; read this one to see
what it has to produce.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exts" / "mmws.wave_sim"))
from mmws.wave_sim import SPECTRA                              # noqa: E402
from mmws.wave_sim.field import deep_water_omega               # noqa: E402
from mmws.wave_sim.globals import GRAVITY                      # noqa: E402

HS, TP, WIND_DIR, SPREAD_S = 2.0, 7.0, 0.0, 5.0
L, N = 256.0, 48         # tile size [m], grid points per side
SHADE = " .:-=+*#%@"


def rule(title):
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def heightmap(z, label):
    lo, hi = z.min(), z.max()
    span = max(hi - lo, 1e-9)
    print(f"\n  {label}   (low {lo:+.2f} m  ..  high {hi:+.2f} m)\n")
    for row in z:
        cells = "".join(SHADE[min(len(SHADE) - 1, int(len(SHADE) * (v - lo) / span))] * 2
                        for v in row)
        print(f"   {cells}")


# ---------------------------------------------------------------- step 0
rule("STEP 0   the sea we want")
spec = SPECTRA.create("pierson_moskowitz", hs=HS, tp=TP,
                      wind_dir=WIND_DIR, spread_s=SPREAD_S)
print(f"""
  spectrum   {spec.name}
  Hs         {HS} m          significant wave height
  Tp         {TP} s          peak period
  wind_dir   {WIND_DIR} rad        waves travel along +x
  spread_s   {SPREAD_S}            how tightly energy clusters around that direction

  tile       {L:.0f} m x {L:.0f} m, {N} x {N} grid  ->  cell = {L / N:.2f} m
""")

# ---------------------------------------------------------------- step 1
rule("STEP 1   lay out the wave components (a grid in wavenumber space)")
dk = 2.0 * math.pi / L
n = np.fft.fftfreq(N, d=1.0 / N)
KX, KY = np.meshgrid(n * dk, n * dk, indexing="ij")
KMAG = np.hypot(KX, KY)
THETA = np.arctan2(KY, KX)
print(f"""
  Every grid point IS one wave component. Its position in the grid sets its
  wavelength and its direction:

      k = (kx, ky)        the wave vector
      |k|                 -> wavelength  = 2*pi / |k|
      atan2(ky, kx)       -> direction

  Grid spacing is forced by the tile, it is not a free choice:

      dk = 2*pi / L = 2*pi / {L:.0f} = {dk:.5f} rad/m

  A few of the {N * N} components:
""")
print(f"  {'kx':>9} {'ky':>9} {'|k|':>9} {'wavelength':>11} {'direction':>10}")
print("  " + "-" * 52)
for i, j in [(1, 0), (2, 0), (4, 0), (0, 3), (3, 3), (8, 2)]:
    print(f"  {KX[i, j]:9.4f} {KY[i, j]:9.4f} {KMAG[i, j]:9.4f} "
          f"{2 * math.pi / KMAG[i, j]:10.2f} m {math.degrees(THETA[i, j]):9.1f} deg")

# ---------------------------------------------------------------- step 2
rule("STEP 2   why everything is in radians")
k_ex = KMAG[4, 0]
print(f"""
  Because a wave is a cosine, and cosine eats an ANGLE.

      eta = a * cos(k*x - w*t + phase)
                 ^^^^^^^^^^^^^^^^^^^^  this whole thing must be in radians

  So the two constants are defined to make that true:

      k = 2*pi / wavelength   [rad/m]   "phase gained per metre travelled"
      w = 2*pi / period       [rad/s]   "phase gained per second waited"

  Take the component at k = {k_ex:.4f} rad/m:

      wavelength = 2*pi/k = {2 * math.pi / k_ex:.2f} m
      walk {2 * math.pi / k_ex:.2f} m  ->  phase advances by {k_ex * (2 * math.pi / k_ex):.3f} rad = 2*pi = one full wave
      walk {2 * math.pi / k_ex / 2:.2f} m  ->  phase advances by {k_ex * (2 * math.pi / k_ex) / 2:.3f} rad = pi    = half a wave, crest to trough

  Radians are not a convention here, they are the only unit that makes
  cos() mean anything. Hz and metres-per-wavelength both need a 2*pi
  glued on first, so the code carries k and w and never needs one.
""")

# ---------------------------------------------------------------- step 3
rule("STEP 3   give each component an amplitude, from the spectrum")
OMEGA = np.asarray(deep_water_omega(KMAG))
with np.errstate(divide="ignore", invalid="ignore"):
    S = np.where(KMAG > 0, np.asarray(spec.energy(OMEGA)), 0.0)
    dwdk = np.where(KMAG > 0, 0.5 * np.sqrt(GRAVITY / np.maximum(KMAG, 1e-12)), 0.0)
    D = np.asarray(spec.spreading(THETA))
    # dkx dky = k dk dtheta, so going from the (k, theta) the spectrum is
    # written in to the (kx, ky) grid costs a factor of 1/k. Dropping it makes
    # every sea come out too small, and no change of tile size will reveal why.
    var = S * dwdk * D / np.maximum(KMAG, 1e-12) * dk * dk
    AMP = np.sqrt(np.maximum(2.0 * var, 0.0))
print(f"""
      a = sqrt( 2 * S(w) * (dw/dk) * D(theta) / k * dkx * dky )
                    ^        ^          ^        ^     ^^^^^^^
               spectrum   convert    spreading  polar  cell area
                          w -> k                 ->
                                               cartesian

  * S(w)      how much energy at that period          (the spectrum)
  * dw/dk     the grid is in k, the spectrum is in w  (chain rule)
  * D(theta)  how much of it travels THIS direction   (spreading)
  * 1/k       dkx dky = k dk dtheta, so the change of variables costs a 1/k
  * dkx*dky   how big a slice of the sea this one component stands for

  The cell area is the "bin width" - and you can see it is not a dial you
  turn, it is just dk*dk, fixed by the tile size back in step 1.

  Tallest components:
""")
flat = np.argsort(AMP.ravel())[::-1][:6]
print(f"  {'wavelength':>11} {'direction':>10} {'period':>8} {'amplitude':>10}")
print("  " + "-" * 44)
for idx in flat:
    i, j = np.unravel_index(idx, AMP.shape)
    print(f"  {2 * math.pi / KMAG[i, j]:10.2f} m {math.degrees(THETA[i, j]):9.1f} deg "
          f"{2 * math.pi / OMEGA[i, j]:7.2f}s {AMP[i, j] * 100:8.2f} cm")

# ---------------------------------------------------------------- step 4
rule("STEP 4   give each component a random phase")
rng = np.random.default_rng(7)
PHASE = rng.uniform(0.0, 2.0 * math.pi, KMAG.shape)
print("""
  This is the ONLY randomness in the whole process. Same spectrum + different
  phases = a different-looking sea that measures identically. It is what stops
  every wave crest lining up at the origin.
""")

# ---------------------------------------------------------------- step 5
rule("STEP 5   dispersion: each component moves at its own speed")
print(f"""
      w = sqrt(g * k)        phase speed c = w/k = sqrt(g/k)

  {'wavelength':>11} {'period':>9} {'phase speed':>12}""")
for lam in (100.0, 50.0, 20.0, 5.0):
    k = 2 * math.pi / lam
    w = math.sqrt(GRAVITY * k)
    print(f"  {lam:10.1f} m {2 * math.pi / w:8.2f}s {w / k:10.2f} m/s")
print("""
  Long waves outrun short ones. ASTRA had none of this - time entered its
  wave model as a bare "+ t", so a 5 m ripple travelled as fast as a 100 m
  swell. It is most of why a real sea reads as water and that one does not.
""")

# ---------------------------------------------------------------- step 6
rule("STEP 6   add them all up")
x = np.linspace(0.0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="ij")


def surface(t):
    phase = (KX[..., None, None] * X + KY[..., None, None] * Y
             - OMEGA[..., None, None] * t + PHASE[..., None, None])
    return (AMP[..., None, None] * np.cos(phase)).sum(axis=(0, 1))


z0 = surface(0.0)
print(f"""
      eta(x, y, t) = SUM over all {N * N} components of
                       a * cos(kx*x + ky*y - w*t + phase)

  Every component contributes to every point. {N * N} components x {N * N} points
  = {N**4:,} cosines. Direct summation like this is why the real thing uses an FFT.
""")
heightmap(z0, "t = 0 s")

# ---------------------------------------------------------------- step 7
rule("STEP 7   does it match the sea we asked for?")
print(f"""
  asked for Hs    = {HS:.3f} m
  4 * std(surface)= {4 * np.std(z0):.3f} m
  highest crest   = {z0.max():+.2f} m
  deepest trough  = {z0.min():+.2f} m

  That round trip - ask for a sea, build it, measure it, get the same number
  back - is the only check that matters. It is what field/base.py's
  significant_height() does, and what you should assert the moment you have
  an FFT field.
""")
heightmap(surface(3.0), "t = 3 s   (same sea, 3 seconds later)")
print("""
  Compare the two maps: the pattern shifts, and it does not shift rigidly,
  because every component is moving at a different speed. That is dispersion
  doing its job.
""")
