from __future__ import annotations

import math

import numpy as np

from ..registry import FIELDS #NOT SPECTRA!
from ..globals import GRAVITY #chain-rule factor
from .base import WaveField, deep_water_omega


@FIELDS.register("fft")

#son of WaveField
class FftWaveField(WaveField):
    """tiling sea surface synthesized from a spectrum with inverse FFT"""

    def __init__(self,spectrum=None, tile_size:float = 256.0,
                 resolution: int = 128, seed: int = 0, **kwargs) -> None:
        #resolution is grid points per side* NxN height samples
        super().__init__(spectrum=spectrum,tile_size=tile_size,resolution=resolution)
        #WaveField init contains validation for tile_size and resolution.
        if spectrum is None:
            raise ValueError("FftWaveField needs a spectrum to draw amplitudes from")

        self.seed = int(seed)

        self._build() #expensive setup
        self.advance(0.0) #this is to make sure that the object is IMMEDIATELY usable.

    def _build(self) -> None:
        """build k grid, dispersion table, and base amplitudes"""
        N,L = self.resolution, self.tile_size #keeps formulas readable

        k_axis = 2.0 * math.pi * np.fft.fftfreq(N, d = L/N)
        """fftfreq returns frequency for each FFT bin. d is grid spacing in meters
        so basically we get cycles/meter"""

        self._kx, self._ky = np.meshgrid(k_axis,k_axis,indexing="ij") #[i,j] can access the wave vector of a cell
        k = np.hypot(self._kx, self._ky)

        self._omega = np.asarray(deep_water_omega(k))
        # = sqrt(g*k) for every cell. THIS IS DISPERSION!

        k_safe = np.maximum(k, 1e-12) #clamp

        dk = 2.0 * math.pi/L #grid spacing in wavenumber space

        S = np.asarray(self.spectrum.energy(self._omega))
        #spectrum evaluated at every cell's omega

        dwdk = 0.5*np.sqrt(GRAVITY/k_safe) #chain rule

        D = np.asarray(self.spectrum.spreading(np.arctan2(self._ky,self._kx)))
        #arctan is direction theta of each cell's wave vector in radians
        #D is the cos-2s spreading weight (how much of that frequency's energy travels this direction)

        var = np.where(k > 0.0, S * dwdk *D / k_safe *dk * dk, 0.0)

        rng = np.random.default_rng(self.seed)

        xi = rng.normal(size=(N,N)) + 1j * rng.normal(size=(N,N))
        #this is a complex Gaussian per cell, thats why we need the randomness.

        self._h0 = (xi / math.sqrt(2.0)) * np.sqrt(var / 2.0)

    def advance(self, t:float) -> None:
        """Evolve to absolute time t and inverse-FFT into a height grid"""
        self.time = float(t)
        phase = np.exp(1j * self._omega * t)
        h = self._h0 * phase + np.conj(self._negated(self._h0)) * np.conj(phase)
        #h is the amplitude of every component at time t
        eta = np.fft.ifft2(h) * self.resolution**2
        self._imag_residual = float(np.max(np.abs(eta.imag)))
        self._z = eta.real

    @staticmethod
    def _negated(a):
        """a evaluated at -k: index (N-i) mod N, (N-j) mod N."""
        return np.roll(np.flip(a, axis=(0, 1)), shift=(1, 1), axis=(0, 1))

    def height(self,x,y):
        """Bilinear sample of the height grid, wrapping at the tile edges."""
        N,L = self.resolution, self.tile_size
        fx = (np.asarray(x, dtype=float) % L) / L * N
        fy = (np.asarray(y, dtype=float) % L) / L * N
        i0 = fx.astype(np.int64) % N   #the cell the point lands in
        j0 = fy.astype(np.int64) % N
        i1, j1 = (i0 + 1) % N, (j0 + 1) % N
        tx, ty = fx - np.floor(fx), fy - np.floor(fy)
        z = self._z
        return (z[i0, j0] * (1.0 - tx) * (1.0 - ty)
            + z[i1, j0] * tx * (1.0 - ty)
            + z[i0, j1] * (1.0 - tx) * ty
            + z[i1, j1] * tx * ty)