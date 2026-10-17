"""The wake a moving hull leaves behind it, on top of whatever sea is there.

No simulator we know of puts any effort into ASV-generated wake, which is the
reason this exists. The first version is deliberately the simplest thing that
is still wave mechanics rather than decoration: the hull is treated as a point
disturbance, every disturbance radiates a damped ring, and the wake is the sum
of the rings it has left along its track.

Why rings and not a drawn wedge
-------------------------------
A Kelvin wake *is* the superposition of circular waves emitted continuously
along the vessel's path - that is Huygens' construction, and the famous
19.47-degree half-angle is a result of it, not an input to it. Starting from
rings means the wedge appears on its own once the physics is right, and it
means the obvious next step is a real one: emit a spectrum of wavenumbers
instead of the single k here, and stationary phase gives the full Kelvin
pattern with its transverse and divergent systems.

A hand-drawn wedge would look correct immediately and teach us nothing, and
it could never answer the question we actually care about - what the wake does
to another hull floating in it.

Superposition with the ambient sea
----------------------------------
`height` is `base.height + wake`, a plain sum. That is exactly right to first
order: the linearised free-surface equations are linear, so two wave systems
pass through each other and add. It stops being right at second order in
steepness, where wave-wave interaction transfers energy between components -
the same approximation the rest of this package already makes by building a
sea from independent Fourier modes. Worth remembering before trusting a wake
crossing a steep JONSWAP crest.

So `base="flat"` gives Tim's starting case - a wake on still water, where
there is nothing else to confuse it - and `base="fft"` is the same wake added
to a spectral sea, with no code change in between.
"""

from __future__ import annotations

import numpy as np

from ..globals import GRAVITY
from ..registry import FIELDS
from .base import WaveField, deep_water_omega


@FIELDS.register("wake")
class WakeField(WaveField):
    """A base field, plus damped rings shed along a hull's track."""

    # Deliberately not sea colours. Pale cyan on blue water was too close to
    # the water to pick out a few centimetres of wake, which was the whole
    # problem. Warm crest against cool trough reads at a glance and cannot be
    # mistaken for the sea itself - this is an overlay, and it should look
    # like one.
    CREST_COLOUR = (1.00, 0.55, 0.10)      # orange
    TROUGH_COLOUR = (0.10, 0.80, 0.45)     # green

    def __init__(
        self,
        spectrum=None,
        tile_size: float = 128.0,
        resolution: int = 128,
        base: str = "flat",
        base_kwargs=None,
        amplitude: float = 0.12,
        wavelength: float = 6.0,
        decay_length: float = 40.0,
        lifetime: float = 12.0,
        speed_ref: float = 3.0,
        core: float = 2.0,
        highlight: bool = True,
        highlight_scale: float = None,
        exaggerate: float = 1.0,
        min_spacing: float = 0.75,
        min_interval: float = 0.08,
        max_samples: int = 48,
        **kwargs,
    ) -> None:
        super().__init__(spectrum=spectrum, tile_size=tile_size,
                         resolution=resolution)

        # The sea this wake rides on. Created through the registry, so the
        # wake does not care whether it is flat water or a JONSWAP spectrum,
        # and a third field added later works here with no edit.
        self.base = FIELDS.create(
            base, spectrum=spectrum, tile_size=tile_size,
            resolution=resolution, **(base_kwargs or {}),
        )
        self.base_name = base
        # Mirror the base's discretisation so write_mesh and write_grid draw
        # the same grid the base would have drawn.
        self.tile_size = self.base.tile_size
        self.resolution = self.base.resolution

        if amplitude < 0.0:
            raise ValueError(f"amplitude must be non-negative, got {amplitude}")
        if wavelength <= 0.0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        if decay_length <= 0.0 or lifetime <= 0.0:
            raise ValueError("decay_length and lifetime must be positive")
        if speed_ref <= 0.0:
            raise ValueError(f"speed_ref must be positive, got {speed_ref}")

        self.amplitude = float(amplitude)
        self.wavelength = float(wavelength)
        # One wavenumber, and the frequency the dispersion relation forces on
        # it. Not a free parameter: a 6 m wave travels at the speed a 6 m wave
        # travels at, and using the same omega = sqrt(gk) as the spectral
        # field is what lets the two be added without one of them being wrong.
        self.k = 2.0 * np.pi / self.wavelength
        self.omega = float(deep_water_omega(self.k))
        # Energy travels at the group velocity, which in deep water is half
        # the phase speed. This is what makes the ring an expanding ring
        # rather than a pattern that exists everywhere the instant it is shed.
        self.group_speed = 0.5 * np.sqrt(GRAVITY / self.k)

        # A point source has a singular near field: at d = 0 every ring that
        # has been shed is in phase and they add coherently, which produced
        # 0.55 m of wake from an amplitude of 0.12. The hull would then float
        # on a half-metre hump of its own making - height() feeds the force
        # model, so the wake is not only drawn, it is felt. Giving the source
        # a finite size removes the singularity: the effective distance is
        # hypot(d, core), so nothing ever sees d = 0. Roughly the beam of the
        # vessel is the right scale, since that is the size of the thing
        # actually pushing the water.
        self.core = float(core)
        self.decay_length = float(decay_length)   # [m]  spatial damping
        self.lifetime = float(lifetime)           # [s]  temporal damping
        self.speed_ref = float(speed_ref)         # [m/s] full-amplitude speed

        # A wake a few centimetres high on a 5 m boat is invisible against a
        # half-metre sea, so the water is tinted where the wake is. This is
        # the honest way round: the SHAPE stays physical and only the COLOUR
        # is amplified, so nothing the hull feels is changed by making it
        # easier to look at.
        self.highlight = bool(highlight)
        self.highlight_scale = (float(highlight_scale)
                                if highlight_scale else max(self.amplitude, 1e-6))
        # If the shape itself needs exaggerating, that is a separate dial and
        # it is NOT free - it scales the rendered surface only, so what you
        # see stops matching what the hull floats on. 1.0 is honest.
        self.exaggerate = float(exaggerate)
        self._mesh_wake = None
        self.min_spacing = float(min_spacing)
        self.min_interval = float(min_interval)
        self.max_samples = int(max_samples)

        # Each row is one shed ring: x, y, emission time, amplitude.
        self._trail: list[tuple[float, float, float, float]] = []
        self._cache = None

        self.advance(0.0)

    # -------------------------------------------------------------- the source

    def set_source(self, position, velocity=None) -> None:
        """Tell the field where the hull is, and how fast it is going.

        Called once per physics step by the extension. Samples are shed only
        when the hull has actually moved - a vessel sitting still leaves no
        wake, and emitting anyway would pile hundreds of coincident rings on
        one spot and produce a standing pimple in the water.
        """
        position = np.asarray(position, dtype=float).reshape(-1)
        speed = 0.0
        if velocity is not None:
            v = np.asarray(velocity, dtype=float).reshape(-1)
            speed = float(np.hypot(v[0], v[1]))     # horizontal speed only

        ds = None
        if self._trail:
            lx, ly, lt, _ = self._trail[-1]
            moved = np.hypot(position[0] - lx, position[1] - ly)
            if moved < self.min_spacing and (self.time - lt) < self.min_interval:
                return
            ds = float(moved)

        # Amplitude grows with speed and saturates: a real wake grows with
        # Froude number and this is the crude stand-in for that. No motion, no
        # wake, which is the behaviour that matters most for believability.
        a = self.amplitude * float(np.clip(speed / self.speed_ref, 0.0, 1.0))
        if a <= 0.0:
            return

        # Weight by the arc length this ring stands for, so the sum is a
        # discrete line integral along the track rather than a count of
        # samples. Without this the wake's height depends on how often
        # set_source happens to be called - halve the physics timestep and
        # the wake doubles, which is a property of the code and not of the
        # water. Measured before the fix: amplitude 0.15 produced a 1.49 m
        # wave, because 48 rings sat nearly on top of each other and added in
        # phase. Lambda is the natural length to divide by: it makes the
        # weight dimensionless and ties the scale to the wave being shed.
        if ds is None:
            ds = max(speed * self.min_interval, 1e-6)   # first sample
        a *= ds / self.wavelength

        self._trail.append((float(position[0]), float(position[1]),
                            float(self.time), a))
        if len(self._trail) > self.max_samples:
            del self._trail[: len(self._trail) - self.max_samples]
        self._cache = None

    def clear(self) -> None:
        """Forget the track. Use between runs so an old wake cannot haunt a new one."""
        self._trail.clear()
        self._cache = None

    # ------------------------------------------------------------------ update

    def advance(self, t: float) -> None:
        self.time = float(t)
        self.base.advance(self.time)
        # Drop rings that have decayed to nothing. Five lifetimes is a factor
        # of e^-5, under a percent, and keeping them only costs arithmetic.
        if self._trail:
            cutoff = self.time - 5.0 * self.lifetime
            keep = [s for s in self._trail if s[2] >= cutoff]
            if len(keep) != len(self._trail):
                self._trail = keep
                self._cache = None

    # ----------------------------------------------------------------- queries

    def _arrays(self):
        """Trail as (x, y, t, a) arrays, rebuilt only when it changes."""
        if self._cache is None:
            if self._trail:
                self._cache = tuple(
                    np.asarray(c, dtype=float) for c in zip(*self._trail))
            else:
                z = np.zeros(0)
                self._cache = (z, z, z, z)
        return self._cache

    def wake_height(self, x, y):
        """Just the wake, with the ambient sea left out.

        Separated from `height` so it can be tested and plotted on its own -
        a wake buried inside a 2 m sea is impossible to eyeball.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        px, py, pt, pa = self._arrays()
        if px.size == 0:
            return np.zeros(x.shape)

        # Cull before the expensive part. Cost here is points x rings, and on
        # a 128x128 grid with 40 rings that is 650k distance evaluations for a
        # wake that covers a few percent of the tile - the rest is zeroed by
        # the causality gate after being computed in full. A bounding box
        # around the track, grown by the furthest any ring can have travelled,
        # throws most of it away and is exact: nothing outside that box can
        # have been reached yet.
        reach = float(np.max(self.group_speed * (self.time - pt)))
        near = ((x >= px.min() - reach) & (x <= px.max() + reach)
                & (y >= py.min() - reach) & (y <= py.max() + reach))
        if not near.all():
            out = np.zeros(x.shape)
            if near.any():
                out[near] = self.wake_height(x[near], y[near])
            return out

        # (..., n) - every query point against every shed ring.
        d = np.hypot(x[..., None] - px, y[..., None] - py)
        age = self.time - pt
        # Softened distance for the wave itself. Causality below still uses
        # the true d - the front travels at the group velocity from where the
        # ring was actually shed, not from the edge of a fictitious core.
        d_soft = np.hypot(d, self.core)

        # Causality: the ring's energy has only reached radius c_g * age.
        # Without this the whole pattern springs into existence at once, which
        # reads as the water flickering rather than a wake spreading.
        arrived = d <= self.group_speed * age

        # The ring itself: a sine in distance, damped in space and in time.
        # This is the piece to replace when moving past a single wavenumber -
        # everything else here is bookkeeping.
        eta = (pa
               * np.exp(-d_soft / self.decay_length)
               * np.exp(-age / self.lifetime)
               * np.sin(self.k * d_soft - self.omega * age))
        return np.sum(np.where(arrived, eta, 0.0), axis=-1)

    def height(self, x, y):
        """Ambient sea plus wake. Linear superposition - see the module docstring."""
        return np.asarray(self.base.height(x, y), dtype=float) + self.wake_height(x, y)

    # ------------------------------------------------------------------ visuals

    def mesh_points(self):
        """The base's grid, lifted by the wake.

        The base may already have displaced its points horizontally (FFT
        choppiness does), so the wake is sampled at where the points actually
        ended up rather than where they started.
        """
        pts = self.base.mesh_points()
        # Kept for vertex_colors, which runs straight after this and would
        # otherwise have to recompute the whole thing.
        self._mesh_wake = self.wake_height(pts[:, 0], pts[:, 1])
        pts[:, 2] += self._mesh_wake * self.exaggerate
        return pts

    def mesh_topology(self):
        return self.base.mesh_topology()

    def vertex_colors(self):
        """The base's colours, brightened where the wake is.

        Without this the wake is physically present and visually invisible:
        a few centimetres of it does not read against a half-metre sea, and
        on a 5 m boat that is the normal case. Tinting is preferable to
        exaggerating the geometry, because the surface the hull floats on
        stays the surface you are looking at.
        """
        base = self.base.vertex_colors()
        if not self.highlight or self._mesh_wake is None:
            return base
        if base is None:
            deep = np.array([[0.015, 0.055, 0.075]])
            base = np.repeat(deep, self._mesh_wake.size, axis=0)
        else:
            base = np.asarray(base, dtype=float)
        # Signed, so crests and troughs of the wake read differently - a wake
        # is a wave train, and colouring |eta| would show it as a series of
        # bars at twice the real frequency.
        w = np.clip(self._mesh_wake[:, None] / self.highlight_scale, -1.0, 1.0)
        crest = np.array([self.CREST_COLOUR])
        trough = np.array([self.TROUGH_COLOUR])
        tint = np.where(w > 0.0, crest, trough)
        return base * (1.0 - np.abs(w)) + tint * np.abs(w)

    def __repr__(self) -> str:
        return (f"WakeField(base={self.base_name!r}, "
                f"A={self.amplitude}, lambda={self.wavelength}, "
                f"rings={len(self._trail)}, t={self.time:.3f})")
