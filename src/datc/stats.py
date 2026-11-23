"""Retrieval statistics.

Ties count at their expected rank: a query whose mate is beaten by g
candidates and tied with t others scores top-k = min(1, max(0, (k - g)/(t + 1))).
The object-level sign test counts an object for the method that ranks its
partners better (by mid-rank) on more of its queries; objects with equal counts
are dropped and n = wins + losses.
"""
from __future__ import annotations

import math

import numpy as np


def mates_by_object(objects):
    """Query index -> mate index, for pools where each object has two rims."""
    return {i: j for i, o in enumerate(objects) for j, q in enumerate(objects)
            if i != j and o == q}


def per_query(S, mate, use=None, queries=None):
    """Per-query expected top-k, reciprocal rank and mid-rank of the mate.

    `use`: candidate set (default all); `queries`: subset of queries."""
    m = S.shape[0]
    use = np.arange(m) if use is None else np.asarray(sorted(use))
    U = set(use.tolist())
    out = {}
    for i in (use if queries is None else queries):
        j = mate.get(int(i))
        if j is None or j not in U:
            continue
        s = S[i, use].astype(float)
        s[use == i] = -np.inf
        sj = S[i, j]
        if not np.isfinite(sj):
            raise ValueError(f"no score for query {i} against its mate {j}")
        better = int((s > sj).sum())
        tied = int((s == sj).sum()) - 1
        tk = {k: min(1.0, max(0.0, (k - better) / (tied + 1))) for k in (1, 5, 10)}
        rr = float(np.mean([1.0 / p for p in range(better + 1, better + tied + 2)]))
        out[int(i)] = dict(top1=tk[1], top5=tk[5], top10=tk[10], rr=rr,
                           mid=better + tied / 2 + 1, better=better, tied=tied)
    return out


def summary(q):
    if not q:
        return {}
    return dict(n=len(q), **{k: float(np.mean([v[k] for v in q.values()]))
                             for k in ("top1", "top5", "top10", "rr")},
                median_rank=float(np.median([v["mid"] for v in q.values()])))


def sign_test(k, n):
    """Exact two-sided sign test, 2 P(X <= k), X ~ Bin(n, 1/2), capped at 1."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def clustered(qa, qb, obj):
    """Method a against method b, by query and by object (mid-rank of the mate)."""
    ids = sorted(set(qa) & set(qb))
    qw = sum(qa[i]["mid"] < qb[i]["mid"] for i in ids)
    ql = sum(qa[i]["mid"] > qb[i]["mid"] for i in ids)
    cw = cl = ct = 0
    for o in sorted(set(obj[i] for i in ids)):
        idx = [i for i in ids if obj[i] == o]
        w = sum(qa[i]["mid"] < qb[i]["mid"] for i in idx)
        l = sum(qa[i]["mid"] > qb[i]["mid"] for i in idx)
        cw += w > l
        cl += l > w
        ct += w == l
    return dict(q_win=int(qw), q_lose=int(ql), q_tie=int(len(ids) - qw - ql),
                q_p=sign_test(min(qw, ql), qw + ql),
                o_win=int(cw), o_lose=int(cl), o_split=int(ct),
                o_p=sign_test(min(cw, cl), cw + cl))


def detection(S, mate):
    """ROC AUC and average precision of 'this pair mates', over unordered pairs."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = S.shape[0]
    iu, ju = np.triu_indices(m, 1)
    lab = np.array([mate.get(int(i)) == int(j) for i, j in zip(iu, ju)])
    sc = S[iu, ju]
    ok = np.isfinite(sc)
    return dict(auc=float(roc_auc_score(lab[ok], sc[ok])),
                ap=float(average_precision_score(lab[ok], sc[ok])),
                n_pos=int(lab.sum()), n_pairs=int(len(lab)))
