"""Fracture rims: contact region -> boundary loop -> resampled contour -> angles.

The fracture region is given, not detected: for Fantastic Breaks it is the set
of faces within a contact threshold of the mating fragment in the assembled
pose; for PRESIOUS it is contact with any other fragment of the object. The
pairing itself never enters the matcher.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as G
from .descriptors import descriptors

MIN_PTS = 40           # loops with fewer mesh vertices are dropped


def contact_mask(mesh, other_points: np.ndarray, eps: float) -> np.ndarray:
    """Faces of `mesh` whose three vertices lie within `eps` of `other_points`."""
    d = cKDTree(other_points).query(np.asarray(mesh.points), workers=-1)[0]
    return (d < eps)[G.faces_of(mesh)].all(1)


def rim_of(mesh, fmask: np.ndarray, ds: float, sm: float, min_pts: int = MIN_PTS,
           rule: str = "intact"):
    """The longest boundary loop of `fmask` as a rim (Fantastic Breaks path).

    Returns None when the region has no loop of `min_pts` vertices. Keys:
      P, N        full resampled loop (n + 2 samples) and its normals
      pts, nrm    the same truncated to n samples, as the published caches stored
      alpha, beta, kappa, tau   n values each, at smoothing `sm`
      loop        mesh vertex ids of the loop
    The traversal sense is fixed by `geometry.orientation`.
    """
    tri = G.faces_of(mesh)
    loops = G.region_boundary_loops(tri, fmask, min_pts)
    if not loops:
        return None
    lp = max(loops, key=len)
    fn = G.canonical_normals(mesh)
    fids, start = G.vertex_face_index(tri)
    N0 = G.loop_normals(fn, fids, start, lp, fmask, rule)
    P, N = G.resample(mesh.points[lp], N0, ds)
    if G.orientation(P, N) < 0:
        P, N = P[::-1].copy(), N[::-1].copy()
    a, b, k, t = descriptors(P, N, ds, sm, closed=True)
    n = min(len(a), len(k), len(P))
    return dict(P=P, N=N, pts=P[:n], nrm=N[:n], alpha=a[:n], beta=b[:n],
                kappa=k[:n], tau=t[:n], loop=np.asarray(lp),
                loop_closed=bool(lp[0] == lp[-1]), n_loops=len(loops))


def component_rims(mesh, fmask: np.ndarray, ds: float, sm: float,
                   min_faces: int = 200, min_pts: int = MIN_PTS):
    """One rim per connected component of `fmask` (PRESIOUS path).

    Within a component the longest valid loop is kept. Normals use the intact
    side of the whole mask. Points are not reoriented, as in the published code.
    """
    tri = G.faces_of(mesh)
    comps = [c for c in G.face_components(tri, fmask) if len(c) >= min_faces]
    if not comps:
        return []
    fn = G.canonical_normals(mesh)
    fids, start = G.vertex_face_index(tri)
    out = []
    for ci, c in enumerate(comps):
        cm = np.zeros(len(tri), bool)
        cm[c] = True
        best = None
        for lp in G.region_boundary_loops(tri, cm, min_pts):
            P0 = mesh.points[lp]
            N0 = G.loop_normals(fn, fids, start, lp, fmask)
            P, N = G.resample(P0, N0, ds)
            if len(P) < 20:
                continue
            closed = bool(lp[0] == lp[-1])
            a, b, kk, tt = descriptors(P, N, ds, sm, closed=closed)
            if len(a) < 12 or len(kk) < 12:
                continue
            cand = dict(P=P, N=N, pts=P, nrm=N, alpha=a, beta=b, kappa=kk, tau=tt,
                        closed=closed, loop=np.asarray(lp))
            if best is None or len(cand["alpha"]) > len(best["alpha"]):
                best = cand
        if best is None:
            continue
        best.update(rim=ci, faces=int(len(c)),
                    centroid=mesh.points[tri[c]].reshape(-1, 3).mean(0))
        out.append(best)
    return out


def closed_geometry(x: dict, ds: float):
    """Closed polyline used by the tuned comparison (Table II, lower block).

    Fantastic Breaks caches stored the loop truncated to n samples, so the
    tuned runs put the two dropped samples back on the chord from the last
    point to the first (spacing ds, or a third of the chord when it is shorter
    than 2.5 ds), normals taken from the nearest end. This reproduces that
    rule. PRESIOUS rims kept the full loop (n + 2 samples) and are used as they
    are.
    """
    n = len(x["alpha"])
    if len(x["pts"]) == n + 2:
        return np.asarray(x["pts"], float), np.asarray(x["nrm"], float)
    assert len(x["pts"]) == n, (len(x["pts"]), n)
    P, N = np.asarray(x["pts"], float), np.asarray(x["nrm"], float)
    g = P[0] - P[-1]
    L = np.linalg.norm(g)
    u = g / (L + 1e-12)
    step = ds if L > 2.5 * ds else L / 3.0
    add = np.stack([P[-1] + k * step * u for k in (1, 2)])
    addN = np.stack([N[-1], N[0]])
    return np.vstack([P, add]), np.vstack([N, addN])
