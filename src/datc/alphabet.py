"""Symbol code: quantile bins on (alpha, beta).

The bin edges are quantiles of the pooled values, so symbols are roughly
equiprobable and the code does not depend on the sampling step or the object
scale. Fitting uses no labels. alpha is fitted on the symmetrised pool
{alpha} u {-alpha}, so that mirroring the alpha bin equals negating alpha,
which is what the mate's reversed traversal needs.
"""
from __future__ import annotations

import json

import numpy as np


class Alphabet:
    """na x nb symbols; `a_ctr`, `b_ctr` are the bin centres used by the
    soft substitution score."""

    def __init__(self, na: int = 9, nb: int = 5, symmetric: bool = True):
        self.na, self.nb, self.symmetric = int(na), int(nb), bool(symmetric)
        self.a_edges = self.b_edges = None
        self.a_ctr = self.b_ctr = None

    def fit(self, alphas, betas, robust: float = 0.995) -> "Alphabet":
        if self.symmetric:
            alphas = list(alphas) + [-np.asarray(x) for x in alphas]
        a = np.concatenate([np.asarray(x).ravel() for x in alphas])
        b = np.concatenate([np.asarray(x).ravel() for x in betas])
        self.a_edges = np.quantile(a, np.linspace(0, 1, self.na + 1)[1:-1])
        self.b_edges = np.quantile(b, np.linspace(0, 1, self.nb + 1)[1:-1])
        lo, hi = (1 - robust) / 2, 1 - (1 - robust) / 2
        ea = np.concatenate([[np.quantile(a, lo)], self.a_edges, [np.quantile(a, hi)]])
        eb = np.concatenate([[np.quantile(b, lo)], self.b_edges, [np.quantile(b, hi)]])
        self.a_ctr = 0.5 * (ea[:-1] + ea[1:])
        self.b_ctr = 0.5 * (eb[:-1] + eb[1:])
        return self

    def encode(self, a, b) -> np.ndarray:
        """(n, 2) int16 symbols: (alpha bin, beta bin)."""
        return np.stack([np.digitize(a, self.a_edges),
                         np.digitize(b, self.b_edges)], 1).astype(np.int16)

    def reverse(self, sym: np.ndarray) -> np.ndarray:
        """The same rim walked the other way: order reversed, alpha bin mirrored."""
        out = sym[::-1].copy()
        out[:, 0] = (self.na - 1) - out[:, 0]
        return out

    def centres(self, sym: np.ndarray) -> np.ndarray:
        """(n, 2) bin centres of a symbol string."""
        return np.ascontiguousarray(np.c_[self.a_ctr[sym[:, 0]], self.b_ctr[sym[:, 1]]], float)

    def to_str(self, sym: np.ndarray) -> str:
        """Letters for the alpha bin, digits for the beta bin, e.g. 'E2F2D3'."""
        A = "ABCDEFGHIJKLMNOP"[:self.na]
        B = "0123456789"[:self.nb]
        return "".join(A[i] + B[j] for i, j in sym)

    # ---------------------------------------------------------------- io
    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(na=self.na, nb=self.nb, symmetric=self.symmetric,
                           a_edges=self.a_edges.tolist(), b_edges=self.b_edges.tolist(),
                           a_ctr=self.a_ctr.tolist(), b_ctr=self.b_ctr.tolist()), f, indent=1)

    @classmethod
    def from_json(cls, path: str) -> "Alphabet":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        al = cls(d["na"], d["nb"], d["symmetric"])
        for k in ("a_edges", "b_edges", "a_ctr", "b_ctr"):
            setattr(al, k, np.asarray(d[k], float))
        return al
