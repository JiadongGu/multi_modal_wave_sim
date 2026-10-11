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


def wigley_hull(length: float = 6.10, beam: float = 2.72, depth: float = 0.60,
                stations: int = 41, sections: int = 17,
                fullness: float = 1.0, taper: float = 1.0,
                end_fraction: float = 0.98):
    """A closed Wigley-form hull. Returns (vertices, triangles).

    The Wigley hull is the standard analytic hull of ship hydrodynamics:

        y(x, z) = (B/2) * (1 - (2x/L)^2)^p * (1 - (z/T)^2)^q

    with p = q = 1 giving the classic form. It matters here for two reasons.
    It is defined by a formula, so there is no file to download, no licence,
    no units to guess and no mesh to repair - the geometry is exact and
    watertight by construction. And it is the hull that Journee (1992) towed
    in regular waves, so measured heave and pitch RAOs exist for it. When this
    project eventually needs to show that it predicts motion and not just
    flotation, that is the dataset to answer to.

    p = `fullness` and q = `taper` move the block coefficient without leaving
    the family: p = q = 1 gives Cb = 4/9 = 0.444, and raising either makes the
    hull finer. A sailing yacht canoe body is finer than a Wigley, so expect
    to raise them; the tool reports the Cb it actually got.

    Construction is a loft. Each station is a closed ring running starboard
    deck edge -> down to the keel -> up to the port deck edge, with the ring's
    closing edge being the deck line itself, so the deck is ruled between
    consecutive stations and needs no separate cap. Consecutive rings join as
    quads split into triangles; the two end rings are closed with a fan about
    their centroid.

    `end_fraction` < 1 stops the stations short of the true ends, where the
    section collapses to a zero-width line. A degenerate ring has no usable
    normal, and orient_faces walks normals - so the ends are left as small
    transoms instead. At 0.98 that costs well under 1% of the volume, and a
    real yacht has a transom anyway.
    """
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError(f"end_fraction must be in (0, 1], got {end_fraction}")
    if stations < 3 or sections < 2:
        raise ValueError("need at least 3 stations and 2 sections")

    half_l, half_b = length / 2.0, beam / 2.0
    xs = np.linspace(-half_l, half_l, stations) * end_fraction
    zs = np.linspace(0.0, -depth, sections)

    def half_width(x, z):
        a = max(1.0 - (x / half_l) ** 2, 0.0) ** fullness
        b = max(1.0 - (z / depth) ** 2, 0.0) ** taper
        return half_b * a * b

    # One ring per station: starboard from deck to keel, then port back up,
    # sharing the keel point and leaving the deck line as the closing edge.
    rings = []
    for x in xs:
        starboard = [(x, +half_width(x, z), z) for z in zs]
        port = [(x, -half_width(x, z), z) for z in reversed(zs[:-1])]
        rings.append(np.array(starboard + port, dtype=float))

    ring_size = len(rings[0])
    vertices = np.concatenate(rings, axis=0)
    triangles = []
    for i in range(stations - 1):
        base, nxt = i * ring_size, (i + 1) * ring_size
        for k in range(ring_size):
            k2 = (k + 1) % ring_size          # wraps across the deck line
            a, b = base + k, base + k2
            c, d = nxt + k2, nxt + k
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    # Close the two ends with a fan about each ring's centroid.
    for i, sign in ((0, 0), (stations - 1, 1)):
        base = i * ring_size
        centre = len(vertices)
        vertices = np.vstack([vertices, rings[i].mean(axis=0)])
        for k in range(ring_size):
            k2 = (k + 1) % ring_size
            # Wind the two caps opposite ways; orient_faces would fix it, but
            # a mesh that is already consistent is one less thing to trust.
            if sign:
                triangles.append((centre, base + k, base + k2))
            else:
                triangles.append((centre, base + k2, base + k))

    return vertices, np.array(triangles, dtype=int)
