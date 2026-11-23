"""Pair scores and score matrices.

Two families, both used in Table II of the paper.

Reviewed configuration (upper block), one function per arm:
  "sw"           DaTC symbol code + cyclic Smith-Waterman on bin centres
  "dtw_ab_cyc"   (alpha, beta) z-scored + cyclic DTW
  "ddtw_kt_cyc"  (kappa, tau) z-scored + cyclic derivative DTW
  plus rim length and planarity, which need no pair loop.

Tuned comparison (lower block): sequences built by `normalise.build_seqs`,
scored with "swc" (cyclic SW on continuous values) or "dtw" (cyclic DTW).

Every score is "higher = more alike". Both traversal directions of the
candidate are tried and the maximum kept, because a loop's direction comes
from mesh indexing. The candidate is the sequence that is rotated (DTW) or
doubled (SW); the query is fixed.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import kernels as K

SW_W_AB = np.array([6.0, 4.0])       # substitution weights for (alpha, beta), hand-set


# ------------------------------------------------------------------ reviewed arms
def prepare_reviewed(items, alph):
    """Per-rim arrays for the reviewed arms.

    alpha/beta/kappa/tau are z-scored with pool statistics (label free); the
    symbol string is encoded from the raw alpha/beta with `alph`.
    """
    stats = {}
    for k in ("alpha", "beta", "kappa", "tau"):
        v = np.concatenate([x[k] for x in items])
        stats[k] = (float(v.mean()), float(v.std()) or 1.0)
    out = []
    for x in items:
        z = {k: (x[k] - mu) / sd for k, (mu, sd) in stats.items()}
        sym = alph.encode(x["alpha"], x["beta"])
        out.append(dict(
            sym_fwd=alph.centres(sym), sym_mate=alph.centres(alph.reverse(sym)),
            ab_fwd=K.c(np.c_[z["alpha"], z["beta"]]),
            ab_mate=K.c(np.c_[-z["alpha"][::-1], z["beta"][::-1]]),
            dkt_fwd=K.c(K.derivative(np.c_[z["kappa"], z["tau"]])),
            dkt_mate=K.c(K.derivative(np.c_[z["kappa"][::-1], z["tau"][::-1]])),
            closed=bool(x.get("closed", True)), n=len(x["alpha"])))
    scale = np.array([float(np.ptp(alph.a_ctr)) + 1e-9, float(np.ptp(alph.b_ctr)) + 1e-9])
    return out, stats, scale


def reviewed_pair(a, b, method, scale):
    """Score of query `a` against candidate `b` (dicts from prepare_reviewed)."""
    if method == "sw":
        cyc = a["closed"] and b["closed"]
        best = -np.inf
        for C in (b["sym_mate"], b["sym_fwd"]):
            Y = np.ascontiguousarray(np.vstack([C, C])) if cyc else C
            best = max(best, K.sw_nd(a["sym_fwd"], Y, scale, SW_W_AB))
        return best
    if method == "dtw_ab_cyc":
        return max(-K.dtw_cyc_nd(a["ab_fwd"], C, K.cyc_offsets(len(C)))
                   for C in (b["ab_mate"], b["ab_fwd"]))
    if method == "ddtw_kt_cyc":
        return max(-K.dtw_cyc_nd(a["dkt_fwd"], C, K.cyc_offsets(len(C)))
                   for C in (b["dkt_mate"], b["dkt_fwd"]))
    raise ValueError(f"unknown method {method!r}")


# ------------------------------------------------------------------ tuned arms
def tuned_pair(A, B, Bm, kind, scale, w):
    """Query A against candidate B (Bm = B as its mate traverses it)."""
    if kind == "swc":
        best = -np.inf
        for C in (Bm, B):
            best = max(best, K.sw_nd(A, np.ascontiguousarray(np.vstack([C, C])), scale, w))
        return best / (len(A) + len(B))
    if kind == "dtw":
        return max(-K.dtw_cyc_nd(A, C, K.cyc_offsets(C.shape[0])) for C in (Bm, B))
    raise ValueError(f"unknown matcher {kind!r}")


# ------------------------------------------------------------------ matrices
def _work(job):
    fam, data, pairs, arg = job
    if fam == "reviewed":
        method, scale = arg
        return [float(reviewed_pair(data[i], data[j], method, scale)) for i, j in pairs]
    seqs, mates = data
    kind, scale, w = arg
    return [float(tuned_pair(seqs[i], seqs[j], mates[j], kind, scale, w)) for i, j in pairs]


def symmetric_matrix(family, data, arg, use=None, procs=None, progress=None):
    """S[i, j] = S[j, i] = score(query i, candidate j) for i < j in `use`.

    Entries outside `use` and the diagonal are -inf. `procs` worker processes
    (default: CPU count - 1, at least 1); `progress(done, total)` is called as
    chunks finish.
    """
    m = len(data) if family == "reviewed" else len(data[0])
    use = sorted(range(m) if use is None else use)
    pairs = [(a, b) for k, a in enumerate(use) for b in use[k + 1:]]
    S = np.full((m, m), -np.inf)
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    nchunk = max(1, min(len(pairs), procs * 8))
    chunks = [pairs[k::nchunk] for k in range(nchunk)]
    done = 0
    if procs == 1:
        results = (_work((family, data, ch, arg)) for ch in chunks)
        for ch, vals in zip(chunks, results):
            for (a, b), v in zip(ch, vals):
                S[a, b] = S[b, a] = v
            done += len(ch)
            if progress:
                progress(done, len(pairs))
        return S
    with ProcessPoolExecutor(procs) as ex:
        futs = [ex.submit(_work, (family, data, ch, arg)) for ch in chunks]
        for ch, f in zip(chunks, futs):
            for (a, b), v in zip(ch, f.result()):
                S[a, b] = S[b, a] = v
            done += len(ch)
            if progress:
                progress(done, len(pairs))
    return S


def length_matrix(v):
    v = np.asarray(v, float)
    return -np.abs(v[:, None] - v[None, :])
