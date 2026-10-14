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


def displacement_hull(length: float = 31.5, beam: float = 7.7, depth: float = 4.6,
                      bilge: float = 0.82, plan_fullness: float = 0.90,
                      transom: float = 0.72,
                      flare: float = 1.04, forefoot: float = 0.55,
                      sheer_rise: float = 0.14, stations: int = 45,
                      sections: int = 13, bow_fraction: float = 0.99):
    """A round-bilge displacement hull. Returns (vertices, triangles).

    Defaults are the Portuguese Andromeda class coastal hydrographic vessel -
    31.5 x 7.7 m, 3.1 m draft, 245 t - which is about the smallest thing
    anyone calls a survey ship.

    Those two published figures are not the same loading condition, and the
    difference is visible. 245 t at 3.1 m draught would need Cb = 0.35 on this
    waterline, which is fast-patrol-boat fine; a survey ship runs 0.45 to
    0.55. So 245 t is light ship and 3.1 m is the loaded draught. At these
    defaults the hull floats at about 2.3 m light and 3.1 m at 380 t, Cb 0.53.
    Loading it at 245 t leaves it riding 1.3 m high, showing twice the
    topsides it should, and it looks like a bathtub.

    What makes this shape different from planing_hull
    -------------------------------------------------
    A planing hull is a hard-chined V that climbs onto its own bow wave. A
    displacement hull never does: it pushes water aside and closes it back in
    behind, so it has no chine at all. The section is a rounded turn of bilge
    running into near-vertical topsides, which is what `bilge` controls - the
    fraction of the depth over which the bottom curves out to full beam. Small
    is a sharp garboard turn, large is a slack bilge.

    The stern is a transom at `transom` of the maximum beam rather than a
    point, because a survey ship needs deck space aft for the over-the-side
    gear, and the keel lifts over the last tenth of the length into the run.
    The forefoot rises forward of 0.7 L, which is the rocker that lets the bow
    lift to a sea instead of driving through it.

    Freeboard is deliberately generous: `depth` is keel to main deck, not
    draft. For the Andromeda numbers that is 4.6 m of hull under about 3.1 m
    of draft, and the 1.5 m left over is what keeps the working deck dry.
    """
    if not 0.0 < bow_fraction <= 1.0:
        raise ValueError(f"bow_fraction must be in (0, 1], got {bow_fraction}")
    if not 0.0 < bilge < 1.0:
        raise ValueError(f"bilge must be in (0, 1), got {bilge}")

    half_b = beam / 2.0 / flare
    t = np.linspace(0.0, bow_fraction, stations)        # 0 transom, 1 stem

    # Plan: a transom aft, full amidships, fine forward.
    plan = (1.0 - t ** 3.0) ** plan_fullness
    plan *= 1.0 - (1.0 - transom) * np.clip((0.30 - t) / 0.30, 0.0, 1.0) ** 2
    half_width = half_b * plan

    # Keel: flat for most of the length, lifting into the run aft and over the
    # forefoot forward.
    keel_z = (-depth
              + forefoot * depth * np.clip((t - 0.70) / 0.30, 0.0, 1.0) ** 2.0
              + 0.10 * depth * np.clip((0.12 - t) / 0.12, 0.0, 1.0) ** 2.0)
    deck_z = sheer_rise * depth * t ** 2.0

    rings = []
    for i in range(stations):
        x = float(t[i]) * length
        w, kz, dz = float(half_width[i]), float(keel_z[i]), float(deck_z[i])
        span = max(dz - kz, 1e-6)
        starboard = []
        for u in np.linspace(0.0, 1.0, sections):       # 0 keel, 1 deck edge
            if u <= bilge:
                # Quarter ellipse: zero width on the centreline at the keel,
                # full beam by the top of the bilge, with the tangent vertical
                # where it meets the topsides so there is no crease.
                frac = float(np.sqrt(max(1.0 - ((bilge - u) / bilge) ** 2, 0.0)))
            else:
                # Topsides, very slightly flared outward toward the deck.
                frac = 1.0 + (flare - 1.0) * (u - bilge) / (1.0 - bilge)
            # The floor lift is strongest on the centreline, where frac is 0,
            # and dies away by the turn of the bilge - which is what makes it
            # a rise of floor rather than simply a shallower hull.
            starboard.append((x, w * frac, kz + span * u))
        port = [(px, -py, pz) for px, py, pz in reversed(starboard[:-1])]
        rings.append(np.array(starboard + port, dtype=float))

    ring_size = len(rings[0])
    vertices = np.concatenate(rings, axis=0)
    triangles = []
    for i in range(stations - 1):
        base, nxt = i * ring_size, (i + 1) * ring_size
        for k in range(ring_size):
            k2 = (k + 1) % ring_size          # the wrap is the deck
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


def superstructure(length: float = 31.5, beam: float = 7.7, depth: float = 4.6,
                   sheer_rise: float = 0.14):
    """Deckhouse, bridge, mast and a stern A-frame. Returns (vertices, triangles).

    VISUAL ONLY. This is never handed to a force model: it sits above the
    waterline, so it displaces nothing, and merging it into the hull would
    corrupt the enclosed volume and the centroid that everything else is
    measured from. Keep it on a separate prim with no physics APIs.

    Not watertight either, and it does not need to be - the pieces are closed
    boxes but they interpenetrate, which is exactly what repair() would
    complain about and what a renderer does not care about at all.
    """
    deck = sheer_rise * depth * 0.25            # roughly the deck amidships
    parts = []

    def box(cx, cy, cz, lx, ly, lz):
        v, f = box_hull(lx, ly, lz)
        parts.append((v + np.array([cx, cy, cz]), f))

    # Main deckhouse, set forward of midships the way a survey ship is - the
    # whole point of the aft two-thirds is clear working deck.
    box(0.56 * length, 0.0, deck + 1.25, 0.34 * length, 0.62 * beam, 2.5)
    # Bridge, one deck up and narrower, with the wings a survey ship needs.
    box(0.62 * length, 0.0, deck + 3.45, 0.18 * length, 0.50 * beam, 2.0)
    box(0.62 * length, 0.0, deck + 3.45, 0.08 * length, 0.80 * beam, 0.9)
    # Mast above the bridge, and a radar bar across it.
    box(0.59 * length, 0.0, deck + 6.60, 0.35, 0.35, 4.2)
    box(0.59 * length, 0.0, deck + 8.40, 0.3, 2.2, 0.25)
    # Funnel, aft of the bridge.
    box(0.46 * length, 0.0, deck + 3.60, 0.08 * length, 0.17 * beam, 2.4)
    # Stern A-frame: two legs and a cross beam, which is the single most
    # recognisable thing on a survey ship - it is how the gear goes over.
    for side in (-1.0, 1.0):
        box(0.045 * length, side * 0.33 * beam, deck + 1.9, 0.35, 0.35, 3.9)
    box(0.045 * length, 0.0, deck + 3.75, 0.35, 0.70 * beam, 0.35)
    # Bulwark around the working deck, so the aft end is not a bare slab.
    for side in (-1.0, 1.0):
        box(0.27 * length, side * 0.48 * beam, deck + 0.45, 0.50 * length, 0.18, 0.9)

    vertices = np.concatenate([p[0] for p in parts], axis=0)
    triangles, offset = [], 0
    for v, f in parts:
        triangles.append(f + offset)
        offset += len(v)
    return vertices, np.concatenate(triangles, axis=0)
