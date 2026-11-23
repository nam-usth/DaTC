"""Mesh -> fracture rim: normals, region boundary loops, resampling.

Every function here is a port of the code that produced the paper's numbers
(`fracontour.py`, `fbarc.py`, `reassemble.py`, `measure_regions.py` of the
research tree). The ports keep the same floating-point operations and the same
iteration order, because the start vertex of a loop decides where resampling
begins, and that changes every descriptor value downstream.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ normals
def canonical_normals(mesh) -> np.ndarray:
    """Face normals whose sign survives rigid motion.

    Winding-consistent normals, global sign fixed so they point away from the
    face-centre centroid on average. pyvista's auto-orientation is not used: it
    can flip after a rotation, which would flip alpha and beta.
    """
    m = mesh.compute_normals(cell_normals=True, point_normals=False,
                             consistent_normals=True, auto_orient_normals=False)
    fn = np.asarray(m.cell_data["Normals"], float)
    c = m.cell_centers().points
    if np.mean(np.einsum("ij,ij->i", fn, c - c.mean(0))) < 0:
        fn = -fn
    return fn


def faces_of(mesh) -> np.ndarray:
    """(F, 3) vertex indices of a triangulated pyvista mesh."""
    return mesh.faces.reshape(-1, 4)[:, 1:]


def vertex_face_index(tri: np.ndarray):
    """Vertex -> incident faces, as (face ids sorted by vertex, start offsets)."""
    flat = tri.ravel()
    order = np.argsort(flat, kind="stable")
    fids = order // 3
    start = np.searchsorted(flat[order], np.arange(int(flat.max()) + 2))
    return fids, start


def loop_normals(fn, fids, start, loop, fmask, rule="intact") -> np.ndarray:
    """Surface normal at each loop vertex, averaged over incident faces.

    The rim is the seam between the fracture surface and the intact surface, so
    the normal there is two-valued and a side must be chosen:

      intact    faces outside the fracture region (default; every reported
                number uses it): the surface both mates share
      all       every incident face
      fracture  faces inside the region only

    A vertex with no face on the chosen side falls back to all its faces.
    """
    out = np.zeros((len(loop), 3))
    for i, v in enumerate(loop):
        f = fids[start[v]:start[v + 1]]
        if rule == "intact":
            keep = f[~fmask[f]]
        elif rule == "fracture":
            keep = f[fmask[f]]
        elif rule == "all":
            keep = f
        else:
            raise ValueError(f"unknown normal rule {rule!r}")
        if not len(keep):
            keep = f
        nv = fn[keep].mean(0)
        out[i] = nv / (np.linalg.norm(nv) + 1e-12)
    return out


# ------------------------------------------------------------------ region -> loops
def region_boundary_loops(tri: np.ndarray, mask: np.ndarray, min_pts: int = 40):
    """Closed loops formed by the edges with exactly one incident masked face.

    Every vertex of that edge set has even degree, so it decomposes into closed
    loops. Loops with fewer than `min_pts` vertices are dropped. The walk order
    (and hence each loop's start vertex) follows the mesh indexing exactly as in
    the published code.
    """
    cnt = {}
    for f in np.where(mask)[0]:
        a, b, c = tri[f]
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            cnt[k] = cnt.get(k, 0) + 1
    nbr = {}
    for (a, b), n in cnt.items():
        if n == 1:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)

    loops, used = [], set()
    for s in list(nbr):
        for t in nbr[s]:
            k = (min(s, t), max(s, t))
            if k in used:
                continue
            used.add(k)
            lp = [s, t]
            while True:
                cur, prv = lp[-1], lp[-2]
                nx = [w for w in nbr.get(cur, [])
                      if w != prv and (min(cur, w), max(cur, w)) not in used]
                if not nx:
                    break
                used.add((min(cur, nx[0]), max(cur, nx[0])))
                lp.append(nx[0])
                if lp[-1] == lp[0]:
                    break
            if len(lp) >= min_pts:
                loops.append(lp)
    return loops


def face_components(tri: np.ndarray, fmask: np.ndarray):
    """Edge-connected components of the masked faces, largest first.

    Components are numbered in the order of their smallest face index and then
    sorted by size with a stable sort, the order of the published code.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    idx = np.where(fmask)[0]
    if not len(idx):
        return []
    T = tri[idx]
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e.sort(1)
    f = np.tile(np.arange(len(idx)), 3)
    key = e[:, 0].astype(np.int64) * (int(tri.max()) + 1) + e[:, 1]
    o = np.argsort(key, kind="stable")
    key, f = key[o], f[o]
    same = key[1:] == key[:-1]
    a, b = f[:-1][same], f[1:][same]
    G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(idx), len(idx)))
    _, lab = connected_components(G, directed=False)
    # relabel by first appearance so the numbering does not depend on scipy
    first = {}
    for i, l in enumerate(lab):
        first.setdefault(int(l), i)
    order = sorted(first, key=first.get)
    comps = [idx[lab == l] for l in order]
    comps.sort(key=len, reverse=True)
    return comps


# ------------------------------------------------------------------ polyline
def resample(p: np.ndarray, n: np.ndarray, ds: float):
    """Constant arc-length resampling of points and normals (open polyline).

    Returns empty arrays when the polyline is shorter than 3 ds.
    """
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 3 * ds:
        return p[:0], n[:0]
    t = np.arange(0, s[-1], ds)
    P = np.stack([np.interp(t, s, p[:, k]) for k in range(3)], 1)
    N = np.stack([np.interp(t, s, n[:, k]) for k in range(3)], 1)
    return P, N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def orientation(P: np.ndarray, N: np.ndarray) -> float:
    """Signed area vector of the loop against the mean normal; its sign fixes
    the traversal sense."""
    Qc = P - P.mean(0)
    A = 0.5 * np.cross(Qc, np.roll(Qc, -1, axis=0)).sum(0)
    return float(np.dot(A, N.mean(0)))


def smooth_curve(P: np.ndarray, sm: float, closed: bool = True) -> np.ndarray:
    """Gaussian smoothing along the polyline; `sm` is the standard deviation in
    samples, the kernel is truncated at 3 sm, closed loops wrap."""
    if sm <= 0 or len(P) < 5:
        return P
    r = max(1, int(round(3 * sm)))
    t = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (t / sm) ** 2)
    k /= k.sum()
    mode = "wrap" if closed else "edge"
    pad = np.pad(P, ((r, r), (0, 0)), mode=mode)
    return np.stack([np.convolve(pad[:, c], k, mode="valid")
                     for c in range(P.shape[1])], axis=1)


def planarity(P: np.ndarray) -> float:
    """Smallest over largest singular value of the centred points: 0 for a
    planar loop. Invariant under rotation, translation and uniform scale."""
    Q = P - P.mean(0)
    s = np.linalg.svd(Q, compute_uv=False)
    return float(s[-1] / (s[0] + 1e-12))
