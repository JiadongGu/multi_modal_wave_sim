
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

    #visuals start here

    def mesh_points(self):
        N = self.resolution
        axis = np.arange(N,dtype=float) * self.cell_size #1-D coordinate axis
        X,Y = np.meshgrid(axis,axis,indexing="ij") #NxN coordinate grid
        Z = np.asarray(self.height(X,Y),dtype=float) #Ask field for elevation at every grid point
        return np.stack([X.ravel(),Y.ravel(),Z.ravel()],axis=-1) #turns into n^2, 3 array of xyz triples

    def mesh_topology(self):
        N = self.resolution
        i,j = np.meshgrid(np.arange(N-1),np.arange(N-1),indexing="ij")
        v00 = (i * N + j).ravel() #vertex index of each quad's bottom left corner
        quads = np.stack([v00,v00+N,v00+N+1,v00+1],axis=-1) #four corners in order
        counts = np.full((N-1) ** 2, 4, dtype = np.int32) #vertice counts
        return counts, quads.ravel().astype(np.int32)

    def write_mesh(self, stage=None, prim_path: str = "/World/Water"):
        if stage is None:
            return None
        from pxr import UsdGeom, Vt #IGNORE THIS FLAG, PXR IS INSTALLED IN ISAACSIM

        mesh = UsdGeom.Mesh.Get(stage,prim_path) #fetch mesh if it exists
        if not mesh:
            mesh = UsdGeom.Mesh.Define(stage,prim_path)
            counts, indices = self.mesh_topology() #creates prim, topology attributes get set
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
            mesh.CreatePointsAttr()
        pts=self.mesh_points().astype(np.float32) #USD points are Vec3f, so convert from float64 down to float32
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
        return mesh
        

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, tile_size={self.tile_size}, "
            f"resolution={self.resolution}, t={self.time:.3f})"
        )
