"""Simple closed hull meshes, for tests and demos.

A force model needs a watertight triangle mesh with outward-facing normals.
Real hulls come from USD; these exist so the physics can be checked against
shapes whose displaced volume is known analytically.
"""

from __future__ import annotations

import numpy as np


def box_hull(length: float = 2.0, beam: float = 1.0, depth: float = 0.5):
    """A closed box centred on the origin. Returns (vertices, triangles).

    Submerged volume at draft d is exactly length * beam * d, which makes it
    the right shape for asserting Archimedes to the last decimal.
    """
    hx, hy, hz = length / 2.0, beam / 2.0, depth / 2.0
    vertices = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=float)
    # Wound so every normal points out of the box.
    triangles = np.array([
        [0, 2, 1], [0, 3, 2],     # bottom, -z
        [4, 5, 6], [4, 6, 7],     # top, +z
        [0, 1, 5], [0, 5, 4],     # -y
        [2, 3, 7], [2, 7, 6],     # +y
        [1, 2, 6], [1, 6, 5],     # +x
        [3, 0, 4], [3, 4, 7],     # -x
    ], dtype=int)
    return vertices, triangles
