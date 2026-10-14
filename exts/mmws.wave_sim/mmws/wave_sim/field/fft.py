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
                 resolution: int = 128, seed: int = 0,
                 choppiness: float = 1.0, foam_threshold: float = 0.55,
                 foam_decay: float = 2.5,
                 **kwargs) -> None:
        #resolution is grid points per side* NxN height samples
        super().__init__(spectrum=spectrum,tile_size=tile_size,resolution=resolution)
        #WaveField init contains validation for tile_size and resolution.
        if spectrum is None:
            raise ValueError("FftWaveField needs a spectrum to draw amplitudes from")

        self.seed = int(seed)
        # Choppiness: how far each point is dragged HORIZONTALLY, on top of
        # being lifted vertically. A pure height field gives sinusoids, which
        # a real sea is not - real waves have sharp crests and broad flat
        # troughs. Tessendorf's horizontal displacement
        #
        #     D(x,t) = sum over k of  -i (k/|k|) h(k,t) e^(ikx)
        #
        # moves water toward the crests and away from the troughs, which is
        # exactly the Gerstner trochoid, and it is what makes a surface look
        # like water rather than like a vibrating sheet.
        #
        # 0 is a plain height field. 1 is the standard value. Past about 1.5
        # the surface starts folding through itself.
        self.choppiness = float(choppiness)
        # Where the surface folds it has broken, and a broken wave is foam.
        # The Jacobian of the displacement map is 1 on flat water and falls
        # toward 0 as a crest sharpens; below this it is marked as whitecap.
        self.foam_threshold = float(foam_threshold)
        # How long a whitecap lingers after the crest has moved on, in
        # seconds. Instantaneous foam - straight from the Jacobian with no
        # memory - flickers on and off with the fold and reads as noise. Real
        # foam is entrained air, and air takes time to surface, so a wave
        # leaves a patch behind it that fades. Without this, cresting looks
        # like static rather than like breaking.
        self.foam_decay = float(foam_decay)
        self._dx = self._dy = self._foam = None
        self._foam_accum = None
        self._last_t = None

        # IMPORTANT: height() does NOT include the horizontal displacement.
        # Once points move sideways the surface is no longer a function of
        # (x, y) - a sharp crest overhangs - so there is no exact height to
        # return. The force model therefore integrates against the undisplaced
        # surface, and choppiness is a change to what you SEE, not to what the
        # hull feels. Keep it that way until someone needs breaking-wave loads,
        # which needs a different force model anyway.

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

        # Unit wave vector, for the horizontal displacement. k=0 is the mean
        # level and has no direction, so it contributes nothing and is left at
        # zero rather than dividing by the clamp and inventing a direction.
        self._kx_hat = np.where(k > 0.0, self._kx / k_safe, 0.0)
        self._ky_hat = np.where(k > 0.0, self._ky / k_safe, 0.0)

    def advance(self, t:float) -> None:
        """Evolve to absolute time t and inverse-FFT into a height grid"""
        self.time = float(t)
        phase = np.exp(1j * self._omega * t)
        h = self._h0 * phase + np.conj(self._negated(self._h0)) * np.conj(phase)
        #h is the amplitude of every component at time t
        eta = np.fft.ifft2(h) * self.resolution**2
        self._imag_residual = float(np.max(np.abs(eta.imag)))
        self._z = eta.real

        dt = 0.0 if self._last_t is None else max(self.time - self._last_t, 0.0)
        self._last_t = self.time

        if self.choppiness <= 0.0:
            self._dx = self._dy = self._foam = self._foam_accum = None
            return

        N2 = self.resolution ** 2
        # D = ifft(-i k_hat h). Multiplying by -i k_hat keeps the Hermitian
        # symmetry that makes eta real, so these come back real too.
        self._dx = np.fft.ifft2(-1j * self._kx_hat * h).real * N2
        self._dy = np.fft.ifft2(-1j * self._ky_hat * h).real * N2

        # Jacobian of the map (x, y) -> (x + L*Dx, y + L*Dy), spectrally:
        # d/dx of ifft(-i kx_hat h) is ifft(i kx * -i kx_hat h) = ifft(kx kx_hat h).
        lam = self.choppiness
        jxx = np.fft.ifft2(self._kx * self._kx_hat * h).real * N2
        jyy = np.fft.ifft2(self._ky * self._ky_hat * h).real * N2
        jxy = np.fft.ifft2(self._kx * self._ky_hat * h).real * N2
        jacobian = (1.0 + lam * jxx) * (1.0 + lam * jyy) - (lam * jxy) ** 2
        # 1 is undisturbed, 0 is a fold. Map the span below the threshold onto
        # 0..1 so there is a gradient to shade with rather than a hard mask.
        born = np.clip(
            (self.foam_threshold - jacobian) / max(self.foam_threshold, 1e-6),
            0.0, 1.0)

        # Take the brighter of "breaking now" and "was breaking recently".
        # Maximum rather than a sum, so foam saturates at 1 instead of
        # accumulating into a solid white sea wherever waves keep passing.
        if self._foam_accum is None or self._foam_accum.shape != born.shape:
            self._foam_accum = born
        elif self.foam_decay > 0.0 and dt > 0.0:
            self._foam_accum = np.maximum(
                self._foam_accum * math.exp(-dt / self.foam_decay), born)
        else:
            self._foam_accum = born
        self._foam = self._foam_accum

    def mesh_points(self):
        """Grid points, displaced horizontally if choppiness is on."""
        pts = super().mesh_points()
        if self._dx is None:
            return pts
        lam = self.choppiness
        pts[:, 0] += lam * self._dx.ravel()
        pts[:, 1] += lam * self._dy.ravel()
        return pts

    def vertex_colors(self):
        """White where the surface has folded, the sea's own colour elsewhere.

        Returned as displayColor. make_scene.py wires the water material to
        read it, so foam shows through a shaded, reflective surface instead of
        replacing it.
        """
        if self._foam is None:
            return None
        f = self._foam.ravel()[:, None]
        deep = np.array([[0.015, 0.055, 0.075]])
        foam = np.array([[0.90, 0.93, 0.95]])
        return deep * (1.0 - f) + foam * f

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