"""Turning a pile of panels into a closed hull.

Pressure integration needs a watertight mesh with outward normals. A lot of
exported assets are neither: the same solid arrives as one prim per face, each
with its own copy of the shared corner points, and with whatever winding the
exporter happened to emit.

That is a topology problem, not a geometry one. The panels may meet perfectly
in space and still share no edges, because "the same point" means "the same
index", not "the same coordinate". Three steps fix it:

    merge    stack the parts, offsetting indices so they live in one array
    weld     collapse coincident points to one index, so edges become shared
    orient   make every face agree, then flip the whole thing if it came out
             inside-out

`open_edges` is the check. Zero means every edge has exactly two faces, which
is what the divergence theorem needs. Everything here is plain numpy, so it is
tested on a laptop.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def merge_meshes(parts):
    """[(vertices, triangles), ...] -> one (vertices, triangles) pair."""
    verts, tris, offset = [], [], 0
    for v, t in parts:
        v = np.asarray(v, dtype=float)
        t = np.asarray(t, dtype=int)
        if len(v) == 0 or len(t) == 0:
            continue
        verts.append(v)
        tris.append(t + offset)
        offset += len(v)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.vstack(verts), np.vstack(tris)


def weld_vertices(vertices, triangles, tol: float = 1e-5):
    """Collapse points within `tol` onto one index, dropping degenerate faces.

    Snapping to a grid of size `tol` means two points either side of a grid
    line will not merge however close they are. Good enough for assets authored
    on round numbers, which is most of them; raise tol if a hull stays open.
    """
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(triangles, dtype=int)
    if len(v) == 0:
        return v, t
    key = np.round(v / tol).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_new = int(inverse.max()) + 1
    welded = np.zeros((n_new, 3))
    np.add.at(welded, inverse, v)
    welded /= np.bincount(inverse, minlength=n_new)[:, None]
    new_t = inverse[t]
    keep = ((new_t[:, 0] != new_t[:, 1])
            & (new_t[:, 1] != new_t[:, 2])
            & (new_t[:, 0] != new_t[:, 2]))
    return welded, new_t[keep]


def open_edges(triangles) -> int:
    """Edges with other than exactly two faces. Zero means watertight."""
    counts = Counter()
    for a, b, c in np.asarray(triangles, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(min(u, v), max(u, v))] += 1
    return sum(1 for n in counts.values() if n != 2)


def signed_volume(vertices, triangles) -> float:
    """Enclosed volume via the divergence theorem. Negative means inside-out."""
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(triangles, dtype=int)
    if len(t) == 0:
        return 0.0
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0)


def orient_faces(vertices, triangles):
    """Make every face wind the same way, with normals pointing outward.

    Walk the face adjacency graph. Two correctly oriented neighbours traverse
    their shared edge in OPPOSITE directions; if they agree, one is flipped.
    Then check the enclosed volume and flip everything if it came out negative,
    which is the case where the whole mesh is consistently inside-out.
    """
    t = np.asarray(triangles, dtype=int).copy()
    if len(t) == 0:
        return t
    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(t):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(u, v), max(u, v))].append(fi)

    seen = np.zeros(len(t), dtype=bool)
    for seed in range(len(t)):
        if seen[seed]:
            continue
        seen[seed] = True
        stack = [seed]
        while stack:
            fi = stack.pop()
            a, b, c = t[fi]
            directed = ((a, b), (b, c), (c, a))
            for u, v in directed:
                for fj in edge_faces[(min(u, v), max(u, v))]:
                    if fj == fi or seen[fj]:
                        continue
                    x, y, z = t[fj]
                    if (u, v) in ((x, y), (y, z), (z, x)):
                        t[fj] = (x, z, y)        # same direction -> flipped
                    seen[fj] = True
                    stack.append(fj)

    if signed_volume(vertices, t) < 0.0:
        t = t[:, [0, 2, 1]]
    return t


def repair(parts, tol: float = 1e-5):
    """merge -> weld -> orient. Returns (vertices, triangles, report)."""
    v, t = merge_meshes(parts)
    before = open_edges(t)
    v, t = weld_vertices(v, t, tol)
    after_weld = open_edges(t)
    t = orient_faces(v, t)
    return v, t, {
        "open_edges_before": before,
        "open_edges_after": after_weld,
        "vertices": len(v),
        "triangles": len(t),
        "volume": signed_volume(v, t),
        "watertight": after_weld == 0,
    }
