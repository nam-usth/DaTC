"""The compiled kernels against slow reference implementations.

The references are the NumPy/Python code that produced the paper's matrices
(fracontour.dtw with band=None, fracontour.sw_soft, fbarc.dtw_cyclic), copied
here. Equality is exact, not approximate.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datc import kernels as K  # noqa: E402

NEG = -1e18


def ref_dtw(x, y):
    x = np.atleast_2d(np.asarray(x).T).T
    y = np.atleast_2d(np.asarray(y).T).T
    n, m = len(x), len(y)
    if n < 3 or m < 3:
        return 1e9
    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        prev, cur = D[i - 1], D[i]
        for j in range(1, m + 1):
            cur[j] = cost[i - 1, j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
    return float(D[n, m] / (n + m))


def ref_dtw_cyclic(x, y, n_offsets=32):
    y = np.atleast_2d(np.asarray(y).T).T
    n = len(y)
    offs = np.unique(np.linspace(0, n, n_offsets, endpoint=False).astype(int))
    return min(ref_dtw(x, np.roll(y, -int(o), axis=0)) for o in offs)


def ref_sw_soft(s1, s2, a_ctr, b_ctr, match=2.0, wa=6.0, wb=4.0, gap_open=-4.0, gap_ext=-0.6):
    """Anti-diagonal Smith-Waterman on symbols, as published."""
    n, m = len(s1), len(s2)
    if n < 3 or m < 3:
        return 0.0
    sa = float(np.ptp(a_ctr)) + 1e-9
    sb = float(np.ptp(b_ctr)) + 1e-9
    da = np.abs(a_ctr[s1[:, 0]][:, None] - a_ctr[s2[:, 0]][None, :]) / sa
    db = np.abs(b_ctr[s1[:, 1]][:, None] - b_ctr[s2[:, 1]][None, :]) / sb
    S = match - wa * da - wb * db
    H = np.zeros((n + 1, m + 1))
    E = np.full((n + 1, m + 1), NEG)
    F = np.full((n + 1, m + 1), NEG)
    best = 0.0
    for d in range(2, n + m + 1):
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        if not len(i):
            continue
        j = d - i
        E[i, j] = np.maximum(E[i, j - 1] + gap_ext, H[i, j - 1] + gap_open)
        F[i, j] = np.maximum(F[i - 1, j] + gap_ext, H[i - 1, j] + gap_open)
        h = np.maximum(H[i - 1, j - 1] + S[i - 1, j - 1], np.maximum(E[i, j], F[i, j]))
        h = np.maximum(h, 0.0)
        H[i, j] = h
        best = max(best, float(h.max()))
    return best


def test_dtw_exact():
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.normal(size=(int(rng.integers(3, 40)), 2))
        y = rng.normal(size=(int(rng.integers(3, 40)), 2))
        assert K.dtw_nd(K.c(x), K.c(y)) == ref_dtw(x, y)


def test_dtw_cyclic_exact():
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.normal(size=(int(rng.integers(5, 40)), 2))
        y = rng.normal(size=(int(rng.integers(5, 40)), 2))
        assert K.dtw_cyc_nd(K.c(x), K.c(y), K.cyc_offsets(len(y))) == ref_dtw_cyclic(x, y)


def test_sw_symbols_exact():
    rng = np.random.default_rng(3)
    a_ctr = np.sort(rng.normal(size=9))
    b_ctr = np.sort(rng.normal(size=5))
    scale = np.array([float(np.ptp(a_ctr)) + 1e-9, float(np.ptp(b_ctr)) + 1e-9])
    for _ in range(20):
        n1, n2 = int(rng.integers(3, 50)), int(rng.integers(3, 50))
        s1 = np.c_[rng.integers(0, 9, n1), rng.integers(0, 5, n1)]
        s2 = np.c_[rng.integers(0, 9, n2), rng.integers(0, 5, n2)]
        x = K.c(np.c_[a_ctr[s1[:, 0]], b_ctr[s1[:, 1]]])
        y = K.c(np.c_[a_ctr[s2[:, 0]], b_ctr[s2[:, 1]]])
        assert K.sw_nd(x, y, scale, np.array([6.0, 4.0])) == ref_sw_soft(s1, s2, a_ctr, b_ctr)


def test_sw_extent_score_and_path():
    rng = np.random.default_rng(4)
    scale, w = np.array([1.0, 1.0]), np.array([6.0, 4.0])
    for _ in range(10):
        x = K.c(rng.normal(size=(30, 2)))
        y = K.c(np.vstack([rng.normal(size=(10, 2)), x[5:20] + 0.01, rng.normal(size=(8, 2))]))
        s, path = K.sw_nd_extent(x, y, scale, w)
        assert s == K.sw_nd(x, y, scale, w)
        assert len(path) > 0
        assert np.all(np.diff(path[:, 0]) >= 0) and np.all(np.diff(path[:, 1]) >= 0)
