"""Placement: matched rim samples -> rigid transform, by RANSAC.

Ported from the code behind the paper's placement paragraph (Section VI):
RANSAC (Fischler and Bolles 1981) over minimal samples of three matched pairs,
Kabsch on each sample, inliers within tau = 3 ds, refit on the largest
consensus; the traversal direction is chosen by fracture-plane opposition
(two mating fracture faces meet face to face, so their outward plane normals
are roughly antiparallel). The paper reports this placement as validation, not
as a result: it puts 6 of 12 Fantastic Breaks objects within 30 degrees.
"""
from __future__ import annotations

import numpy as np

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
