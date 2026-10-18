
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

    def write_grid(self, stage=None, prim_path: str = "/World/WaterGrid",
                   stride: int = 8, visible: bool = True):
        """Draw every `stride`-th grid line as curves following the surface.

        The water renders as a shaded sheet, which hides what the field is
        actually made of. These are the FFT grid lines themselves, lifted (and
        displaced, when choppiness is on) with the water - so you can see the
        resolution you are working at, watch the tile wrap, and see the
        horizontal drag toward the crests that a shaded surface disguises.

        Curves rather than a wireframe draw mode: USD has no per-prim
        wireframe, and this way the lines are real geometry with a width that
        does not change as the camera moves.
        """
        if stage is None:
            return None
        from pxr import Sdf, UsdGeom, Vt

        curves = UsdGeom.BasisCurves.Get(stage, prim_path)
        if not curves:
            curves = UsdGeom.BasisCurves.Define(stage, prim_path)
            curves.CreateTypeAttr().Set("linear")
            curves.CreateWidthsAttr().SetMetadata("interpolation", "constant")
            UsdGeom.PrimvarsAPI(curves).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([(0.95, 0.95, 1.0)]))

        UsdGeom.Imageable(curves).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
        if not visible:
            return curves

        N = self.resolution
        stride = max(1, int(stride))
        grid = self.mesh_points().reshape(N, N, 3)
        lines = [grid[i, :, :] for i in range(0, N, stride)]
        lines += [grid[:, j, :] for j in range(0, N, stride)]
        points = np.concatenate(lines, axis=0)
        curves.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        curves.CreateCurveVertexCountsAttr().Set(
            Vt.IntArray([N] * len(lines)))
        curves.GetWidthsAttr().Set(Vt.FloatArray([self.cell_size * 0.06]))
        return curves

    def overlay_colors(self):
        """Per-vertex EMISSIVE colour, or None. A debug overlay, not the sea.

        Separate from vertex_colors because the two are asked different
        questions by the material: displayColor is what the water looks like
        and is lit and reflected like water, while this is painted on top and
        does not care about the lighting. A field with nothing to overlay
        returns None and the primvar is never written.
        """
        return None

    def vertex_colors(self):
        """Per-point RGB for the surface, or None to leave it unshaded.

        A hook, not a feature: most fields have nothing to say about colour,
        but a field that knows where it has broken (FftWaveField, via the
        Jacobian of its displacement) can return foam here and have write_mesh
        carry it without the mesh code knowing what foam is.
        """
        return None

    def write_mesh(self, stage=None, prim_path: str = "/World/Water"):
        if stage is None:
            return None
        from pxr import Gf, Sdf, UsdGeom, Vt #IGNORE THIS FLAG, PXR IS INSTALLED IN ISAACSIM

        mesh = UsdGeom.Mesh.Get(stage,prim_path) #fetch mesh if it exists
        if not mesh:
            mesh = UsdGeom.Mesh.Define(stage,prim_path)
            mesh.CreatePointsAttr()
            # Quads, not a subdivision cage. Left unauthored, subdivisionScheme
            # resolves to catmullClark and the renderer smooths a surface you
            # computed exactly.
            mesh.CreateSubdivisionSchemeAttr().Set("none")
            mesh.CreateDoubleSidedAttr().Set(True)

        pts=self.mesh_points().astype(np.float32) #USD points are Vec3f, so convert from float64 down to float32

        # Rewrite the topology whenever the point count no longer matches it.
        #
        # This used to run only when the prim was created, which silently broke
        # every field swap: /World/Water outlives the field that made it, so a
        # 64-resolution fft tile left behind indices with a row stride of 64,
        # and the next field wrote 128*128 points into them. The indices still
        # address valid points, so nothing errors - each row of quads just
        # shears one step sideways as it advances and the surface renders as
        # diagonal stripes.
        existing = mesh.GetPointsAttr().Get()
        if existing is None or len(existing) != len(pts):
            counts, indices = self.mesh_topology()
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))

        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))

        # Per-vertex normals. Without them the renderer uses FACE normals and
        # shades every triangle flat, so a 1 m grid under a 1.5 m sea renders
        # as a field of visible facets - the single worst thing about how this
        # looks, and nothing to do with the wave model. subdivisionScheme is
        # "none" (deliberately: the surface is computed, not a cage to smooth),
        # which means nothing else is going to supply them.
        #
        # Taken from the height grid rather than from the triangles: the
        # surface is known analytically, so its slope is too, and a gradient
        # of the grid is both cheaper and smoother than averaging face
        # normals. With choppiness on, the points have also moved sideways
        # and this becomes an approximation - a good one, because the
        # displacement is smooth at the scale of a cell.
        N = self.resolution
        if len(pts) == N * N:
            z = pts[:, 2].reshape(N, N).astype(np.float64)
            dzdx, dzdy = np.gradient(z, self.cell_size)
            nrm = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1).reshape(-1, 3)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
            mesh.CreateNormalsAttr().Set(
                Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        overlay = self.overlay_colors()
        if overlay is not None:
            primvars = UsdGeom.PrimvarsAPI(mesh)
            ov = primvars.GetPrimvar("wakeTint")
            if not ov:
                ov = primvars.CreatePrimvar(
                    "wakeTint", Sdf.ValueTypeNames.Color3fArray,
                    UsdGeom.Tokens.vertex)
            ov.SetInterpolation(UsdGeom.Tokens.vertex)
            ov.Set(Vt.Vec3fArray.FromNumpy(np.asarray(overlay, dtype=np.float32)))

        colors = self.vertex_colors()
        if colors is not None:
            # "vertex" interpolation: one colour per point, interpolated
            # across each face. The alternative, "constant", would paint the
            # whole sea one colour and the foam would vanish.
            primvars = UsdGeom.PrimvarsAPI(mesh)
            pv = primvars.GetPrimvar("displayColor")
            if not pv:
                pv = primvars.CreatePrimvar(
                    "displayColor", Sdf.ValueTypeNames.Color3fArray,
                    UsdGeom.Tokens.vertex)
            # Set the interpolation every time, not only on creation.
            # UsdGeomGprim DECLARES primvars:displayColor as part of its
            # schema, with interpolation "constant". So GetPrimvar returns a
            # valid primvar before anything has been authored, the branch
            # above never runs, and the interpolation stays constant - under
            # which USD uses the FIRST value for the whole mesh and throws
            # the other 16383 away. Every per-vertex colour this field has
            # ever produced was discarded here.
            pv.SetInterpolation(UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec3fArray.FromNumpy(
                np.asarray(colors, dtype=np.float32)))
        # Hydra culls against the extent. A stale one hides parts of the
        # surface, which looks like another meshing bug.
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        mesh.CreateExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(*(float(v) for v in lo)),
            Gf.Vec3f(*(float(v) for v in hi)),
        ]))
        return mesh

    

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, tile_size={self.tile_size}, "
            f"resolution={self.resolution}, t={self.time:.3f})"
        )
