"""FFT wave field - YOUR IMPLEMENTATION.

A real sea surface, synthesised by giving every wave component a random
amplitude drawn from the spectrum and summing them with an inverse FFT. This
is the Tessendorf method, and it is what `asv_wave_sim` uses.

tools/demo_wave_generation.py already does this by direct summation. Same
physics; this computes it in one transform instead of N^4 cosines.

The construction
----------------
Every cell of an N x N grid in wavenumber space is one wave component:

    k_axis = 2*pi * fftfreq(N, d=L/N)        spacing dk = 2*pi/L
    k      = (kx, ky),  |k| -> wavelength,  atan2(ky,kx) -> direction

Each gets a variance from the spectrum, exactly as in the demo:

    var(k) = S(w) * (dw/dk) * D(theta) / |k| * dk * dk

and a complex Gaussian amplitude with that variance:

    h0(k) = (xi_r + i xi_i)/sqrt(2) * sqrt(var(k) / 2)

Time evolution uses the two-term form:

    h(k,t) = h0(k) e^{i w t} + conj(h0(-k)) e^{-i w t}

Two reasons for the two terms rather than one. It makes waves *travel*
rather than stand. And it is automatically Hermitian - h(-k) = conj(h(k))
holds by construction - so the inverse FFT returns a real surface with no
symmetry fixing up afterwards. Assert the imaginary residual is ~1e-20 and
you know you got it right.

Finally:

    eta = real(ifft2(h)) * N^2

The N^2 undoes numpy's 1/N^2 normalisation in ifft2, so that
eta(x) = sum_k h(k) exp(i k.x) as written above.

What to write
-------------
1. `_build()`   - the k grid, omega, and h0. Runs once.
2. `advance(t)` - the two-term evolution and the inverse FFT. Runs every step.
3. `height(x,y)`- bilinear interpolation into the grid, with wrapping, since
                  the field tiles every `tile_size`.

Leave horizontal displacement (choppy waves) alone for now. It makes
`height()` much harder - the point of the mesh above (x,y) is no longer the
point generated from (x,y), so you need a triangle search and a vertical ray
rather than a lookup. Get the flat-grid version correct and tested first.

Expect Hs to scatter between realisations: about +/-4% at L=512 N=256,
shrinking as the tile grows. That is physical, not a bug - average over seeds
before concluding anything is wrong.
"""

from __future__ import annotations

import math

import numpy as np

from ..registry import FIELDS
from .base import WaveField, deep_water_omega


@FIELDS.register("fft")
class FftWaveField(WaveField):
    """A tiling sea surface synthesised from a spectrum with an inverse FFT."""

    def __init__(self, spectrum=None, tile_size: float = 256.0,
                 resolution: int = 128, seed: int = 0, **kwargs) -> None:
        super().__init__(spectrum=spectrum, tile_size=tile_size, resolution=resolution)
        if spectrum is None:
            raise ValueError("FftWaveField needs a spectrum to draw amplitudes from")
        self.seed = int(seed)
        self._build()
        self.advance(0.0)

    def _build(self) -> None:
        """Build the k grid, the dispersion table and the base amplitudes h0."""
        # TODO(jiadong): see the module docstring.
        raise NotImplementedError("FftWaveField._build is not implemented yet")

    def advance(self, t: float) -> None:
        """Evolve to absolute time t and inverse-FFT into a height grid."""
        # TODO(jiadong): see the module docstring.
        raise NotImplementedError("FftWaveField.advance is not implemented yet")

    def height(self, x, y):
        """Bilinear sample of the height grid, wrapping at the tile edges."""
        # TODO(jiadong): see the module docstring.
        raise NotImplementedError("FftWaveField.height is not implemented yet")
