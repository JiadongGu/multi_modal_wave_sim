"""Pressure-based buoyancy - YOUR IMPLEMENTATION, and the point of the project.

Integrate hydrostatic pressure over the wetted hull instead of estimating a
submerged volume. Four things fall out that the volume method cannot give:

* Archimedes exactly, with no calibration constant
* horizontal force, because a wave face is a tilted pressure surface
* a centre of pressure that moves on its own as the hull heels
* nothing per-vessel - give it a mesh, get forces

The method
----------
1. Transform the hull's vertices into world space.
2. Depth of each vertex below the surface: d = field.height(x, y) - z.
3. Classify each triangle by how many vertices have d > 0:
       3 -> fully wet, keep it
       0 -> dry, discard
       1 or 2 -> straddles the waterline, CLIP it
4. For every wet triangle: pressure at its centroid, p = rho * g * d.
5. Force on that face, normal to it and pushing inward:

       F_i = -p_i * A_i * n_i

6. Sum, and take moments about the centre of mass:

       F = sum F_i          M = sum (c_i - com) x F_i

Why the minus sign: n is the OUTWARD normal, and pressure pushes in. On the
bottom of a hull n points down, so -p*A*n points up. On the top it points
down. The bottom is deeper, so the pressures do not cancel - that difference
is the buoyant force, and summing it over a closed surface gives rho*g*V by
the divergence theorem. You never write rho*g*V anywhere; it emerges.

Clipping
--------
The fiddly part, and where the bugs live. For a triangle straddling the
surface, interpolate along each edge that crosses it:

    t = d_i / (d_i - d_j)        cut = p_i + t * (p_j - p_i)

* one vertex wet  -> one sub-triangle  (that vertex plus two cut points)
* two vertices wet -> a quad           (split into two triangles)

Keep the winding consistent with the parent triangle or the normal flips and
that face pushes the wrong way. Cost scales with the WATERLINE, not the hull
area - only triangles crossing the surface need this, and they lie along a
1-D curve through a 2-D mesh.

A known approximation
---------------------
Pressure is evaluated at each triangle's centroid. Because pressure varies
linearly over a face, that gives the FORCE exactly - the centroid is the mean.
It does not give the moment exactly, because the resultant really acts at the
centre of pressure, which sits lower. A level box therefore shows a small
spurious moment that converges away with refinement (measured: 27.8 mm of
equivalent arm at 12 triangles, 13.9 at 48, 3.7 at 192, 1.0 at 768, while the
force stays exact to 0.0000% throughout).

Gazebo applies the centre-of-pressure correction analytically, which removes
it and lets a coarse hull stay accurate. Worth doing once this works - it is
the difference between needing a 5,000-triangle hull and a 500-triangle one,
which matters at 60 Hz.

What to write
-------------
1. `__init__`: store vertices, triangles, density. Validate shapes.
2. `_clip`: wet sub-triangles from (T,3,3) world points and (T,3) depths.
   Handle the fully-wet case in bulk and loop only over the straddlers.
3. `wrench`: steps 1-6 above, returning a body-frame Wrench.
4. Delete the raises.

Verify with tools/selftest.py. A box hull has an analytic submerged volume, so
the Archimedes checks are exact rather than approximate - expect 0.000% error
at every draft, and Fx/Fz equal to the water slope on a tilted plane.
"""

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

    @staticmethod
    def _clip(tri_pts, tri_depth):
        """Wet (sub-)triangles. tri_pts (T,3,3) world, tri_depth (T,3)."""
        # TODO(jiadong): see the module docstring.
        raise NotImplementedError("BuoyancyPressure._clip is not implemented yet")

    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        # TODO(jiadong): see the module docstring.
        raise NotImplementedError("BuoyancyPressure.wrench is not implemented yet")
