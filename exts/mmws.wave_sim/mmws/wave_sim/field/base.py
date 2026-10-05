"""The wave field contract - the most important file in the project.

One field, two consumers
------------------------
A wave field has exactly two customers: the renderer, which needs the whole
surface as geometry, and the physics, which needs the height at a handful of
scattered points. ASTRA served them with two separate implementations of the
same formula - a Warp GPU kernel wrote the visual mesh, and a NumPy function
recomputed the identical expression for buoyancy. Nothing ever read the GPU
buffer back.

That duplication had two consequences worth remembering:

* the two copies could disagree, and nothing would detect it;
* their JONSWAP module could not be switched on at all, because it existed
  only as a GPU kernel with no CPU twin to answer height queries. A finished,
  working, cited wave model sat disabled for want of one function.

So in this project the field is the *single owner* of the surface state:

    advance(t)          move the field to time t. Called once per update.
    height(x, y)        elevation query. Physics calls this.
    displacement(x, y)  horizontal motion of the surface (choppy waves).
    normal(x, y)        surface normal, for slope-dependent forces.
    write_mesh(...)     push the current state to the renderer.

Both consumers read the same state, by construction. A subclass physically
cannot let them drift.

Dispersion
----------
`deep_water_wavenumber` and `deep_water_omega` are here because dispersion is
a property of the field, not of the spectrum. ASTRA had none at all: time
entered its wave model as a bare `+ t`, so every component advanced at 1 rad/s
regardless of wavelength, and a 2 m ripple travelled as fast as a 100 m swell.
Getting this right is what makes a wave field look like water rather than like
a vibrating sheet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..globals import GRAVITY


def deep_water_wavenumber(omega):
    """k from omega in deep water: omega^2 = g k, so k = omega^2 / g."""
    omega = np.asarray(omega, dtype=float)
    return omega**2 / GRAVITY


def deep_water_omega(k):
    """omega from k in deep water: omega = sqrt(g k).

    Phase speed is omega/k = sqrt(g/k), so long waves are fast and short waves
    are slow. That single fact is most of what makes a sea surface read as
    water to the eye.

    Deep water means depth > about half the wavelength. The pond in ASTRA's
    scene is shallow enough that the full relation omega^2 = g k tanh(k h)
    matters for the longer components; add a finite-depth variant when a scene
    needs it rather than fudging this one.
    """
    k = np.asarray(k, dtype=float)
    return np.sqrt(GRAVITY * np.abs(k))


class WaveField(ABC):
    """Base class for all wave fields.

    A field is constructed from a spectrum plus whatever discretisation it
    needs (grid size, tile length, number of components), then advanced in time
    and queried.
    """

    name: str = "<unregistered>"

    def __init__(self, spectrum=None, tile_size: float = 128.0, resolution: int = 128) -> None:
        if tile_size <= 0.0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        self.spectrum = spectrum
        self.tile_size = float(tile_size)
        self.resolution = int(resolution)
        self.time = 0.0

    @property
    def cell_size(self) -> float:
        """Spacing between surface samples [m].

        Worth knowing: 1 m cells capture essentially all of the wave-*height*
        energy of a typical sea but only about a third of the *slope* energy,
        and slope is what drives the horizontal forces. Resolution is not a
        cosmetic setting.
        """
        return self.tile_size / self.resolution

    # ------------------------------------------------------------------ update

    @abstractmethod
    def advance(self, t: float) -> None:
        """Move the field to absolute time `t` [s].

        Absolute rather than a dt, so a field is reproducible: advancing to the
        same t twice must give the same surface. That is what makes a
        regression test on a wave field possible at all.
        """

    # ------------------------------------------------------------------ queries

    @abstractmethod
    def height(self, x, y):
        """Surface elevation at world (x, y) [m]. Scalar or array in, same out.

        This is the one function the whole force path depends on. It must be
        cheap enough to call for every hull sample point, every physics step.
        """

    def displacement(self, x, y):
        """Horizontal surface displacement (dx, dy) [m]. Zero by default.

        Real wave crests are sharper than sine waves because water particles
        move horizontally as well as vertically, bunching up at the crest.
        Adding this is what turns round swells into recognisable choppy sea.

        It also breaks the simple height query: once the surface is displaced
        horizontally, the point of the mesh above (x, y) is no longer the point
        generated *from* (x, y), so `height(x, y)` has to search for the right
        patch rather than evaluating a formula. Gazebo solves this by locating
        the containing triangle and casting a vertical ray. Expect to do the
        same, and do not try to invert the displacement iteratively - it
        diverges near crests.
        """
        zeros = np.zeros_like(np.asarray(x, dtype=float))
        return zeros, np.zeros_like(zeros)

    def normal(self, x, y, eps: float | None = None):
        """Unit surface normal, by central differences on `height` by default.

        Override it when the field can produce slopes analytically - an FFT
        field gets them nearly free from the same transform.
        """
        if eps is None:
            eps = 0.5 * self.cell_size
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        dzdx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2.0 * eps)
        dzdy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2.0 * eps)
        n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    # -------------------------------------------------------------------- stats

    def significant_height(self, count: int = 64) -> float:
        """Hs measured from the field itself: 4 * std(elevation).

        The round-trip check that matters. Ask a spectrum for Hs = 2 m, build a
        field from it, measure the field, and you should get 2 m back. If you
        do not, the discretisation is losing or inventing energy - and that is
        a bug you will otherwise only notice as "the boat feels wrong".
        """
        g = np.linspace(0.0, self.tile_size, count, endpoint=False)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        return float(4.0 * np.std(np.asarray(self.height(gx, gy))))

    # ------------------------------------------------------------------ visuals

    def write_mesh(self, stage=None, prim_path: str = "/World/Water") -> None:
        """Push the current surface to the renderer. No-op by default.

        The seam for all the visual work. The fast path in Isaac Sim is USDRT
        / Fabric: get the mesh's points attribute once, then write into it from
        a Warp kernel so the data never round-trips through Python. Going
        through the plain USD API instead works and is much easier to debug,
        and is the right first version.

        Deliberately separate from `advance` so the visual rate and the physics
        rate can differ - ASTRA updated its mesh at 20 Hz while physics stepped
        far faster. Keep that split, but note that it means the renderer shows
        a frozen surface between updates, so interpolate if it shows.
        """
        return None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, tile_size={self.tile_size}, "
            f"resolution={self.resolution}, t={self.time:.3f})"
        )
