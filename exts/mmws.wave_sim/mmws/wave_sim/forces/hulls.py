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


def planing_hull(length: float = 5.28, beam: float = 2.21, depth: float = 0.60,
                 deadrise_transom: float = 16.0, deadrise_bow: float = 50.0,
                 keel_rise: float = 0.55, sheer_rise: float = 0.30,
                 flare: float = 1.05, fullness: float = 0.45,
                 stations: int = 41, sections: int = 7,
                 bow_fraction: float = 0.985):
    """A warped-V planing hull from principal dimensions. (vertices, triangles).

    Built for a small outboard runabout - a Boston Whaler 170 Montauk is
    5.28 x 2.21 m with 16 degrees of deadrise at the transom - but it is a
    hull form, not that boat. Nobody publishes offsets for a production hull,
    so this is the standard shape generated to the published dimensions, in
    the same spirit as wigley_hull: watertight by construction, no file to
    download, and a draft that can be checked against the maker's figure.

    The geometry each station carries
    ---------------------------------
    Three points define half a section, and everything else is interpolation:

        keel    (y=0, low)      the centreline, rising toward the bow
        chine   the hard corner where the V bottom meets the topsides
        sheer   the deck edge, rising toward the bow

    `deadrise` is the angle of the V from horizontal, and it is WARPED: 16
    degrees at the transom opening out to 50 at the bow. That is what makes a
    planing boat work - flat aft to lift onto the plane, deep forward to cut
    chop instead of slamming - and it is the single most consequential number
    in the shape.

    The chine is clamped below the sheer, which is also what real hulls do:
    forward, the chine runs out and merges into the stem, so there is no
    corner left at the bow.

    Caps: the transom is a genuine flat face, so its ring is capped as-is. The
    bow would taper to a zero-width line, which has no usable normal, so the
    stations stop at `bow_fraction` and leave a small stem face - under 2% of
    the length, and real boats have a stem band there anyway.
    """
    if not 0.0 < bow_fraction <= 1.0:
        raise ValueError(f"bow_fraction must be in (0, 1], got {bow_fraction}")
    if stations < 3 or sections < 2:
        raise ValueError("need at least 3 stations and 2 sections")

    # `beam` is the widest part of the boat, which is the deck edge, not the
    # chine - the topsides flare outward above it. Divide the flare back out
    # so the hull measures what was asked for rather than `flare` times it.
    half_b = beam / 2.0 / flare
    t = np.linspace(0.0, bow_fraction, stations)        # 0 transom, 1 bow

    # Plan view: full aft, fining toward the bow.
    half_width = half_b * (1.0 - t ** 3) ** fullness
    # Warped V: flat aft, deep forward.
    deadrise = np.radians(deadrise_transom
                          + (deadrise_bow - deadrise_transom) * t ** 1.6)
    keel_z = -depth + keel_rise * depth * t ** 2.2
    sheer_z = sheer_rise * depth * t ** 2.0
    chine_z = keel_z + half_width * np.tan(deadrise)
    # Forward the chine runs out into the stem; without this it climbs above
    # the deck and the section turns inside out.
    chine_z = np.minimum(chine_z, sheer_z - 0.04 * depth)

    # Half a section: sheer -> chine down the topside, chine -> keel down the V.
    topside = np.linspace(0.0, 1.0, sections)[:-1]      # excludes the chine
    bottom = np.linspace(0.0, 1.0, sections)            # includes keel at 1

    rings = []
    for i in range(stations):
        x = float(t[i]) * length
        w, sz, cz, kz = (float(half_width[i]), float(sheer_z[i]),
                         float(chine_z[i]), float(keel_z[i]))
        starboard = []
        for f in topside:                               # deck edge down to chine
            starboard.append((x, w * (flare + (1.0 - flare) * f), sz + (cz - sz) * f))
        for f in bottom:                                # chine down to the keel
            starboard.append((x, w * (1.0 - f), cz + (kz - cz) * f))
        port = [(px, -py, pz) for px, py, pz in reversed(starboard[:-1])]
        rings.append(np.array(starboard + port, dtype=float))

    ring_size = len(rings[0])
    vertices = np.concatenate(rings, axis=0)
    triangles = []
    for i in range(stations - 1):
        base, nxt = i * ring_size, (i + 1) * ring_size
        for k in range(ring_size):
            k2 = (k + 1) % ring_size        # the wrap is the deck panel
            triangles.append((base + k, base + k2, nxt + k2))
            triangles.append((base + k, nxt + k2, nxt + k))

    for i, flip in ((0, 0), (stations - 1, 1)):
        base = i * ring_size
        centre = len(vertices)
        vertices = np.vstack([vertices, rings[i].mean(axis=0)])
        for k in range(ring_size):
            k2 = (k + 1) % ring_size
            triangles.append((centre, base + k, base + k2) if flip
                             else (centre, base + k2, base + k))

    return vertices, np.array(triangles, dtype=int)
