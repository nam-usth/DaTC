"""Alignment kernels, compiled with numba.

They perform the same floating-point operations, in the same order, as the
NumPy code that produced the paper's matrices (fracontour.dtw / sw_soft), so the
scores agree bit for bit; `tests/test_kernels.py` checks this against slow
reference implementations.

  sw_nd      Smith-Waterman local alignment, affine gaps, soft substitution
             match - sum_c w_c |x_c - y_c| / s_c
  dtw_nd     dynamic time warping, Euclidean local cost, normalised by n + m
  dtw_cyc_nd DTW minimised over evenly spaced cyclic rotations of y
"""
from __future__ import annotations

import math
import sys

import numba as nb
import numpy as np

NEG = -1e18
SW_MATCH, SW_GAP_OPEN, SW_GAP_EXT = 2.0, -4.0, -0.6   # hand-set, never tuned
N_OFFSETS = 32
# numba keys its disk cache on the .py source, which a frozen release does not ship;
# there the kernels are compiled once per process instead (same code, same results).
CACHE = not getattr(sys, "frozen", False)


@nb.njit(cache=CACHE)
def sw_nd(x, y, scale, w, match=SW_MATCH, gap_open=SW_GAP_OPEN, gap_ext=SW_GAP_EXT):
    """Best local alignment score of x (n, d) against y (m, d)."""
    n, m, d = x.shape[0], y.shape[0], x.shape[1]
    if n < 3 or m < 3:
        return 0.0
    H = np.zeros((n + 1, m + 1))
    E = np.full((n + 1, m + 1), NEG)
    F = np.full((n + 1, m + 1), NEG)
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e = max(E[i, j - 1] + gap_ext, H[i, j - 1] + gap_open)
            f = max(F[i - 1, j] + gap_ext, H[i - 1, j] + gap_open)
            E[i, j] = e
            F[i, j] = f
            s = match
            for c in range(d):
                s = s - w[c] * (abs(x[i - 1, c] - y[j - 1, c]) / scale[c])
            h = max(H[i - 1, j - 1] + s, max(e, f))
            if h < 0.0:
                h = 0.0
            H[i, j] = h
            if h > best:
                best = h
    return best


@nb.njit(cache=CACHE)
def sw_nd_extent(x, y, scale, w, match=SW_MATCH, gap_open=SW_GAP_OPEN, gap_ext=SW_GAP_EXT):
    """sw_nd plus the traceback of the best alignment: (score, path (k, 2)).

    The path lists matched index pairs (i, j), 0-based, from start to end."""
    n, m, d = x.shape[0], y.shape[0], x.shape[1]
    path = np.zeros((0, 2), np.int64)
    if n < 3 or m < 3:
        return 0.0, path
    H = np.zeros((n + 1, m + 1))
    E = np.full((n + 1, m + 1), NEG)
    F = np.full((n + 1, m + 1), NEG)
    S = np.zeros((n + 1, m + 1))
    best, bi, bj = 0.0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e = max(E[i, j - 1] + gap_ext, H[i, j - 1] + gap_open)
            f = max(F[i - 1, j] + gap_ext, H[i - 1, j] + gap_open)
            E[i, j] = e
            F[i, j] = f
            s = match
            for c in range(d):
                s = s - w[c] * (abs(x[i - 1, c] - y[j - 1, c]) / scale[c])
            S[i, j] = s
            h = max(H[i - 1, j - 1] + s, max(e, f))
            if h < 0.0:
                h = 0.0
            H[i, j] = h
            if h > best:
                best, bi, bj = h, i, j
    # traceback over the three matrices (state 0 = H, 1 = E gap in x, 2 = F gap in y)
    out = np.zeros((bi + bj, 2), np.int64)
    k = 0
    i, j, state = bi, bj, 0
    while i > 0 and j > 0:
        if state == 0:
            if H[i, j] <= 0.0:
                break
            if H[i, j] == H[i - 1, j - 1] + S[i, j]:
                out[k, 0] = i - 1
                out[k, 1] = j - 1
                k += 1
                i -= 1
                j -= 1
            elif H[i, j] == E[i, j]:
                state = 1
            else:
                state = 2
        elif state == 1:
            if E[i, j] == E[i, j - 1] + gap_ext:
                j -= 1
            else:
                j -= 1
                state = 0
        else:
            if F[i, j] == F[i - 1, j] + gap_ext:
                i -= 1
            else:
                i -= 1
                state = 0
    path = out[:k][::-1].copy()
    return best, path


@nb.njit(cache=CACHE)
def dtw_nd(x, y):
    """DTW distance of x (n, d) and y (m, d), divided by n + m; 1e9 if too short."""
    n, m, d = x.shape[0], y.shape[0], x.shape[1]
    if n < 3 or m < 3:
        return 1e9
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc = 0.0
            for c in range(d):
                t = x[i - 1, c] - y[j - 1, c]
                acc += t * t
            D[i, j] = math.sqrt(acc) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


@nb.njit(cache=CACHE)
def dtw_cyc_nd(x, y, offs):
    """min over the rotations `offs` of dtw_nd(x, roll(y, -o))."""
    m = y.shape[0]
    best = np.inf
    yr = np.empty_like(y)
    for o in offs:
        for t in range(m):
            yr[t] = y[(t + o) % m]
        v = dtw_nd(x, yr)
        if v < best:
            best = v
    return best


def cyc_offsets(m: int, n_offsets: int = N_OFFSETS) -> np.ndarray:
    """Evenly spaced start offsets for a cyclic sequence of length m."""
    return np.unique(np.linspace(0, m, n_offsets, endpoint=False).astype(int)).astype(np.int64)


def derivative(x: np.ndarray) -> np.ndarray:
    """Keogh-Pazzani derivative estimate used by derivative DTW."""
    x = np.atleast_2d(np.asarray(x).T).T
    d = np.zeros_like(x, dtype=float)
    if len(x) < 3:
        return d
    d[1:-1] = ((x[1:-1] - x[:-2]) + (x[2:] - x[:-2]) / 2) / 2
    d[0], d[-1] = d[1], d[-2]
    return d


def c(a) -> np.ndarray:
    """Contiguous float64 2-D copy."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return np.ascontiguousarray(a)
