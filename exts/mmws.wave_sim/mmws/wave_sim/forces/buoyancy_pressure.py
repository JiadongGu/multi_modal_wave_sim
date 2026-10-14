from __future__ import annotations

import numpy as np

from ..globals import GRAVITY, WATER_DENSITY
from ..registry import FORCES
from .base import BodyState, ForceModel, Wrench, quat_to_matrix


@FORCES.register("buoyancy_pressure")
class BuoyancyPressure(ForceModel):
    """Hydrostatic pressure integrated over the wetted hull."""

    def __init__(self, vertices=None, triangles=None,
                 density: float = WATER_DENSITY) -> None:
        if vertices is None or triangles is None:
            raise ValueError("BuoyancyPressure needs a hull: vertices and triangles")
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (M, 3), got {self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (T, 3), got {self.triangles.shape}")
        if self.triangles.max(initial=0) >= len(self.vertices):
            raise ValueError("triangle index out of range")
        self.density = float(density)
        # Set True by the extension when the force arrows or the run
        # recorder are on. Nothing else reads these.
        self.record_faces = False
        self.last_wet = self.last_centroids = self.last_forces = None

    @staticmethod
    def _clip(tri_pts, tri_depth):
        """Wet (sub-)triangles. tri_pts (T,3,3) world, tri_depth (T,3)."""
        wet_count = (tri_depth > 0.0).sum(axis=1)
        pieces = []
        full = tri_pts[wet_count==3]
        if len(full):
            pieces.append(full) #fully wetted triangles dont need any work

        for idx in np.flatnonzero((wet_count== 1) | (wet_count == 2)):
            p,d = tri_pts[idx],tri_depth[idx]
            below = [i for i in range(3) if d[i] > 0.0]
            above = [i for i in range(3) if d[i] <= 0.0] #which local vertex indices are wet and dry

            def cut(i,j):
                return p[i] + d[i] / (d[i]-d[j]) * (p[j] - p[i])

            if len(below) == 1: #one vertex wet, vertex plus 2 points where edges cross the surface
                b = below[0]
                pieces.append(np.array([[p[b],cut(b,(b+1)%3),cut(b,(b+2)%3)]]))

            else: #two vertices wet, the wet region is a quadrilateral
                b1,b2 = below
                a = above[0]
                if(b1 + 1) %3 != b2: 
                    b1,b2 = b2,b1
                pieces.append(np.array([
                    [p[b1], p[b2], cut(b2, a)],
                    [p[b1], cut(b2, a), cut(b1, a)],
                ])) #split the quad into two triangles sharing the diagonal from p[b1] to cut(b2,a)

        if not pieces: #nothing wet, empty (0,3,3) array rather than None
            return np.zeros((0,3,3))
        return np.concatenate(pieces,axis=0)


    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        R = quat_to_matrix(body.orientation)
        position = np.asarray(body.position,dtype=float)
        world = position + self.vertices @ R.T #transpose 
        depth = np.asarray(wave_field.height(world[:,0],world[:,1])) - world[:,2] #vectorized call for all M vertices
        wet = self._clip(world[self.triangles],depth[self.triangles])
        if len(wet) == 0:
            # Clear the debug capture too, or a hull that lifts clear of the
            # water keeps drawing the arrows it had on the way out.
            if self.record_faces:
                self.last_wet = np.zeros((0, 3, 3))
                self.last_centroids = np.zeros((0, 3))
                self.last_forces = np.zeros((0, 3))
            return Wrench()

        #standard triangle geometry
        a,b,c = wet[:,0],wet[:,1],wet[:,2]
        cross = np.cross(b-a,c-a)
        norm = np.linalg.norm(cross,axis=1)
        area = 0.5 * norm
        normal = cross / np.maximum(norm,1e-12)[:,None]
        centroid = (a+b+c)/3.0 

        d_c = np.maximum(
            np.asarray(wave_field.height(centroid[:,0], centroid[:,1])) - centroid[:,2],
            0.0,
        ) #depth at each individual centroid

        face = -(self.density * GRAVITY * d_c *area)[:, None] * normal
        com_world = position + R @ np.asarray(body.center_of_mass, dtype=float)
        moment = np.cross(centroid-com_world,face)

        # Optional capture of the per-face terms, for drawing the arrows and
        # for the run recorder. Off by default: these are references to arrays
        # that already exist, so the cost is a few pointers, but holding them
        # keeps a frame's worth of geometry alive and there is no reason to
        # when nobody is looking. World frame, like everything above.
        if self.record_faces:
            self.last_wet = wet
            self.last_centroids = centroid
            self.last_forces = face

        return Wrench(R.T @ face.sum(axis=0), R.T @ moment.sum(axis=0))