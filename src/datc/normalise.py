"""Channel transforms for the tuned comparison (Table I, lower block).

Every transform is fitted without labels on a set of rims and then applied to
the rims being scored; the two sets may differ (frozen on development objects,
or on the 60-object pool for the held-out pool).

  z      pool mean and standard deviation
  rank   pooled empirical CDF (mid-rank), centred, times sqrt(12)

alpha is always fitted on the symmetrised pool {alpha} u {-alpha} (and its z
mean is 0), so the transformed alpha of the mate is the negation.
"""
from __future__ import annotations

import math

import numpy as np

from .descriptors import descriptors
from .rims import closed_geometry

CHSETS = {"D": ("alpha", "beta"), "K": ("kappa", "tau"),
          "H4": ("alpha", "beta", "kappa", "tau"), "H3": ("alpha", "beta", "kappa")}
NLEV = {"alpha": 9, "beta": 5, "kappa": 9, "tau": 5}
SW_W = {"D": (6.0, 4.0), "K": (6.0, 4.0), "H4": (3.0, 2.0, 3.0, 2.0),
        "H3": (3.75, 2.5, 3.75)}


def descs(items, sm, ds):
    """channel -> list of arrays, recomputed from each rim's closed geometry at
    smoothing `sm` and truncated to the rim's stored length n."""
    out = {c: [] for c in ("alpha", "beta", "kappa", "tau")}
    for x in items:
        P, N = closed_geometry(x, ds)
        a, b, k, t = descriptors(P, N, ds, sm, closed=True)
        n = len(x["alpha"])
        for c, v in zip(("alpha", "beta", "kappa", "tau"), (a, b, k, t)):
            out[c].append(np.asarray(v[:n], float))
    return out


def subset(D, idx):
    return {c: [v[i] for i in idx] for c, v in D.items()}


def _centres(v, N, robust=0.995):
    edges = np.quantile(v, np.linspace(0, 1, N + 1)[1:-1])
    lo, hi = (1 - robust) / 2, 1 - (1 - robust) / 2
    e = np.concatenate([[np.quantile(v, lo)], edges, [np.quantile(v, hi)]])
    return 0.5 * (e[:-1] + e[1:])


def _midcdf(srt, v):
    return (np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / (2.0 * len(srt))


def fit_channel(vals, ch, kind):
    """Fit on a list of arrays; returns a function v -> v'."""
    v = np.concatenate(vals)
    sym = ch == "alpha"
    if sym:
        v = np.concatenate([v, -v])
    if kind == "z":
        mu = 0.0 if sym else float(v.mean())
        sd = float(v.std()) or 1.0
        return lambda x: (x - mu) / sd
    if kind == "rank":
        srt = np.sort(v)
        return lambda x: (_midcdf(srt, x) - 0.5) * math.sqrt(12.0)
    raise ValueError(f"unknown transform {kind!r}")


def build_seqs(D_fit, D_apply, chset, norm):
    """Transform fitted on D_fit, applied to D_apply.

    Returns (seqs, mates, scale, weights): mates are the sequences reversed with
    alpha negated; scale is the SW scale per channel (spread of the bin centres
    of an N_c-level quantile code fitted on the transformed pool)."""
    chans = CHSETS[chset]
    fs = {c: fit_channel(D_fit[c], c, norm) for c in chans}
    seqs, mates = [], []
    for r in range(len(D_apply[chans[0]])):
        X = np.ascontiguousarray(np.stack([fs[c](D_apply[c][r]) for c in chans], 1), float)
        M = X[::-1].copy()
        if "alpha" in chans:
            M[:, chans.index("alpha")] *= -1.0
        seqs.append(X)
        mates.append(np.ascontiguousarray(M))
    scale = []
    for c in chans:
        v = np.concatenate([fs[c](x) for x in D_fit[c]])
        if c == "alpha":
            v = np.concatenate([v, -v])
        scale.append(float(np.ptp(_centres(v, NLEV[c]))) + 1e-9)
    return seqs, mates, np.array(scale), np.array(SW_W[chset])
