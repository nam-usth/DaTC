"""Thresholds of the app's match / no-match verdict, and their measured error.

For every unordered pair of rims inside a half of the Fantastic Breaks retrieval
pool, compute the two verdict inputs of datc.pipeline.pair_features:
  score    Smith-Waterman score / (n_A + n_B), 9 x 5 code fitted on the two rims
  inliers  share of the aligned samples within 3 ds after RANSAC (best direction)
Three rules are fitted on the development half (30 objects, 1770 pairs, 30 mated),
each threshold maximising Youden's J = TPR - FPR:
  score only, inliers only, score AND inliers.
The rule with the highest development J is selected; all three are then measured
on the test half (30 other objects) and on PRESIOUS (47 rims, other scans, other
step), with no refitting.

Usage
  python -u experiments/pair_verdict.py [--procs 6]
Writes reference/verdict.json and results/pair_verdict/features.npz.
Last line on success: 'VERDICT DONE'.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from datc import datasets as DS  # noqa: E402
from datc.pipeline import pair_features  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "results", "pair_verdict")
KEYS = ("score", "inliers")


def _job(args):
    rims, pairs, ds = args
    out = []
    for i, j in pairs:
        f = pair_features(rims[i], rims[j], ds)
        out.append((f["score"], f["inliers"]))
    return out


def features(items, pairs, ds, procs):
    rims = [dict(alpha=x["alpha"], beta=x["beta"], pts=np.asarray(x["pts"])[:len(x["alpha"])]) for x in items]
    chunks = [pairs[k::procs * 4] for k in range(procs * 4)]
    F = {}
    with ProcessPoolExecutor(procs) as ex:
        for ch, res in zip(chunks, ex.map(_job, [(rims, c, ds) for c in chunks])):
            for p, v in zip(ch, res):
                F[p] = v
    return np.array([F[p] for p in pairs])


def rates(X, y, thr):
    acc = np.ones(len(y), bool)
    for k, t in thr.items():
        acc &= X[:, KEYS.index(k)] >= t
    tpr, fpr = float(acc[y].mean()), float(acc[~y].mean())
    return dict(tpr=tpr, fpr=fpr, j=tpr - fpr, n_pos=int(y.sum()), n_neg=int((~y).sum()))


def best_single(X, y, k):
    v = np.unique(X[:, KEYS.index(k)])
    best = max(v, key=lambda t: (rates(X, y, {k: t})["j"], -t))
    return {k: float(best)}


def best_and(X, y):
    qs = {k: np.unique(np.quantile(X[:, i], np.linspace(0, 1, 101))) for i, k in enumerate(KEYS)}
    best, bj = None, -2
    for a in qs["score"]:
        for b in qs["inliers"]:
            j = rates(X, y, {"score": a, "inliers": b})["j"]
            if j > bj:
                best, bj = {"score": float(a), "inliers": float(b)}, j
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    pool = pickle.load(open(os.path.join(CACHE, "fb_pool60.pkl"), "rb"))["items"]
    pres = pickle.load(open(os.path.join(CACHE, "presious.pkl"), "rb"))["items"]
    sp = DS.splits()
    obj = [x["object"] for x in pool]
    sets = {}
    for half in ("dev", "test"):
        idx = [i for i, o in enumerate(obj) if o in set(sp[half])]
        pairs = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
        y = np.array([obj[i] == obj[j] for i, j in pairs])
        print(f"{half}: {len(pairs)} pairs, {int(y.sum())} mated ...", flush=True)
        sets[half] = (features(pool, pairs, DS.FB_GLOBAL_DS, a.procs), y)
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)
    pairs = [(i, j) for i in range(len(pres)) for j in range(i + 1, len(pres))]
    y = np.array([pres[i]["mate"] == j or pres[j]["mate"] == i for i, j in pairs])
    print(f"PRESIOUS: {len(pairs)} pairs, {int(y.sum())} mated (either direction) ...", flush=True)
    sets["presious"] = (features(pres, pairs, DS.PR_GLOBAL_DS, a.procs), y)
    np.savez_compressed(os.path.join(OUT, "features.npz"),
                        **{f"{k}_X": v[0] for k, v in sets.items()}, **{f"{k}_y": v[1] for k, v in sets.items()})

    Xd, yd = sets["dev"]
    rules = {"score only": best_single(Xd, yd, "score"),
             "inliers only": best_single(Xd, yd, "inliers"),
             "score and inliers": best_and(Xd, yd)}
    res = dict(note=("Thresholds of the DaTC Studio verdict. Fitted on development-half pairs of the "
                     "Fantastic Breaks retrieval pool (Youden J); measured on test-half pairs and on PRESIOUS "
                     "without refitting. Inputs: datc.pipeline.pair_features."),
               rules={})
    for name, thr in rules.items():
        res["rules"][name] = dict(thresholds=thr, **{k: rates(X, y, thr) for k, (X, y) in sets.items()})
        r = res["rules"][name]
        print(f"RULE {name:18s} {thr}  dev J {r['dev']['j']:.3f} | test TPR {r['test']['tpr']:.3f} "
              f"FPR {r['test']['fpr']:.3f} | PRESIOUS TPR {r['presious']['tpr']:.3f} FPR {r['presious']['fpr']:.3f}",
              flush=True)
    sel = max(res["rules"], key=lambda k: res["rules"][k]["dev"]["j"])
    res["selected"] = dict(name=sel, **res["rules"][sel])
    res["seconds"] = time.time() - t0
    with open(os.path.join(HERE, "..", "reference", "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"selected on development J: {sel}")
    print("VERDICT DONE", flush=True)


if __name__ == "__main__":
    main()
