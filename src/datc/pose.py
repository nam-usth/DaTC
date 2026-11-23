"""Placement: matched rim samples -> rigid transform, by RANSAC.

Ported from the code behind the paper's placement paragraph (Section V-F):
RANSAC (Fischler and Bolles 1981) over minimal samples of three matched pairs,
Kabsch on each sample, inliers within tau = 3 ds, refit on the largest
consensus; the traversal direction is chosen by fracture-plane opposition
(two mating fracture faces meet face to face, so their outward plane normals
are roughly antiparallel). The paper reports this placement as validation, not
as a result: it puts 6 of 12 Fantastic Breaks objects within 30 degrees.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

TAU_DS = 3.0          # inlier threshold, in sampling steps
N_ITER = 2000
DEGEN = 0.05          # smallest allowed s1/s0 of a minimal sample (rejects near-collinear triples)


def kabsch(P, Q):
    """Rigid transform (R, t) taking P onto Q, both (n, 3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def _hypothesis(P, Q, idx, degen):
    S = P[idx] - P[idx].mean(0)
    if degen > 0:
        s = np.linalg.svd(S, compute_uv=False)
        if s[0] < 1e-12 or s[1] / s[0] < degen:
            return None
    return kabsch(P[idx], Q[idx])


def ransac_kabsch(P, Q, tau, n_iter=N_ITER, m=3, degen=DEGEN, seed=0):
    """Best (R, t, inlier mask) taking P onto Q (P[i] claimed to match Q[i]),
    or None when no hypothesis gathers three inliers."""
    n = len(P)
    if n < max(m, 3):
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_iter):
        idx = rng.choice(n, m, replace=False)
        h = _hypothesis(P, Q, idx, degen)
        if h is None:
            continue
        R, t = h
        mask = np.linalg.norm((R @ P.T).T + t - Q, axis=1) < tau
        k = int(mask.sum())
        if k >= 3 and (best is None or k > best[0]):
            best = (k, mask)
    if best is None:
        return None
    mask = best[1]
    R, t = kabsch(P[mask], Q[mask])
    return R, t, mask


def plane_normal(P, bulk_centroid):
    """Best-fit plane normal of a rim, oriented away from the fragment's bulk."""
    Q = P - P.mean(0)
    n = np.linalg.svd(Q, full_matrices=False)[2][-1]
    if np.dot(n, P.mean(0) - bulk_centroid) < 0:
        n = -n
    return n


def rotation_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _plane_step(X, Q, N):
    """One linearised point-to-plane step (Chen and Medioni 1992): the small rotation w and
    translation v minimising sum(((X + w x X + v) - Q) . N)^2. Returns (dR, v)."""
    A = np.c_[np.cross(X, N), N]
    b = np.einsum("ij,ij->i", Q - X, N)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    w, v = x[:3], x[3:]
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3), v
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx, v


def icp(src, dst, R, t, tau, dst_normals=None, n_iter=60, tol=1e-4, min_pairs=10):
    """Trimmed ICP started from (R, t).

    Each iteration pairs every moved source point with its nearest destination
    point and drops the pairs farther apart than tau. With dst_normals the rest
    are fitted point-to-plane (Chen and Medioni 1992), which lets the surfaces
    slide along each other; without, point-to-point by Kabsch (Besl and McKay
    1992), which stalls near the sampling spacing on smooth surfaces. Stops when
    the RMS point distance of the kept pairs changes by less than tol x tau.
    Returns dict(R, t, rms0, rms, kept0, kept, iters, converged, metric), or None
    when fewer than min_pairs pairs lie within tau at the start."""
    tree = cKDTree(dst)

    def pairs(R, t):
        X = src @ R.T + t
        d, j = tree.query(X, workers=-1)
        return X, d, j, d < tau

    X, d, j, k = pairs(R, t)
    if k.sum() < min_pairs:
        return None
    rms0, kept0 = float(np.sqrt(np.mean(d[k] ** 2))), int(k.sum())
    prev, it, converged = rms0, 0, False
    for it in range(1, n_iter + 1):
        if dst_normals is None:
            R1, t1 = kabsch(src[k], dst[j[k]])
        else:
            dR, v = _plane_step(X[k], dst[j[k]], dst_normals[j[k]])
            R1, t1 = dR @ R, dR @ t + v
        X1, d1, j1, k1 = pairs(R1, t1)
        if k1.sum() < min_pairs:        # the refit lost the overlap: keep the last good pose
            it -= 1
            break
        R, t, X, d, j, k = R1, t1, X1, d1, j1, k1
        rms = float(np.sqrt(np.mean(d[k] ** 2)))
        if abs(prev - rms) < tol * tau:
            converged = True
            break
        prev = rms
    return dict(R=R, t=t, rms0=rms0, rms=float(np.sqrt(np.mean(d[k] ** 2))), kept0=kept0, kept=int(k.sum()),
                iters=it, converged=converged, metric="point-to-point" if dst_normals is None else "point-to-plane")
