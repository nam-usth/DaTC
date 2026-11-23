"""Table I, upper block: the reviewed configuration, untuned for the table.

Arms (one sampling step per pool, smoothing 2 samples, 9 x 5 symbol code with
bin edges fitted on the pool without labels):
  DaTC + local alignment       symbol code, cyclic Smith-Waterman, score / (n_i + n_j)
  DaTC + cyclic DTW            (alpha, beta) z-scored, cyclic DTW
  (kappa,tau) + cyclic dDTW    (kappa, tau) z-scored, cyclic derivative DTW
  rim length                   -|n_i - n_j|
  rim planarity                -|p_i - p_j|

Usage
  python -u experiments/table1_reviewed.py [--procs 4] [--pools fb presious]
Needs cache/fb_pool60.pkl and cache/presious.pkl (experiments/build_pools.py).
Writes results/table1_reviewed/{fb,presious}_scores.npz, table1_reviewed.json,
fb_alphabet.json. Last line on success: 'REVIEWED DONE'.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from datc import scoring as SC, stats as ST  # noqa: E402
from datc.alphabet import Alphabet  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "results", "table1_reviewed")
ARMS = ("DaTC + local alignment", "DaTC + cyclic DTW", "(kappa,tau) + cyclic dDTW",
        "rim length", "rim planarity")
METHOD = {"DaTC + local alignment": "sw", "DaTC + cyclic DTW": "dtw_ab_cyc",
          "(kappa,tau) + cyclic dDTW": "ddtw_kt_cyc"}


def load(name):
    p = os.path.join(CACHE, name + ".pkl")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing: run experiments/build_pools.py first")
    with open(p, "rb") as f:
        return pickle.load(f)["items"]


def run_pool(tag, items, mate, procs, log):
    n = np.array([len(x["alpha"]) for x in items], float)
    pl = np.array([x["planarity"] for x in items], float)
    alph = Alphabet(9, 5).fit([x["alpha"] for x in items], [x["beta"] for x in items])
    data, zstats, scale = SC.prepare_reviewed(items, alph)
    S = {}
    for arm in ARMS:
        t0 = time.time()
        if arm == "rim length":
            S[arm] = SC.length_matrix(n)
        elif arm == "rim planarity":
            S[arm] = SC.length_matrix(pl)
        else:
            raw = SC.symmetric_matrix("reviewed", data, (METHOD[arm], scale), procs=procs)
            S[arm] = raw / (n[:, None] + n[None, :]) if METHOD[arm] == "sw" else raw
        log(f"  {tag}: scored {arm} ({time.time()-t0:.0f}s)")
    pq = {arm: ST.per_query(S[arm], mate) for arm in ARMS}
    table = {arm: {k: 100 * v for k, v in ST.summary(pq[arm]).items() if k in ("top1", "top5", "top10")}
             for arm in ARMS}
    for arm in ARMS:
        table[arm]["n_queries"] = len(pq[arm])
    obj = [x["object"] for x in items]
    vs = {arm: ST.clustered(pq[ARMS[0]], pq[arm], obj) for arm in ARMS[1:]}
    det = {arm: ST.detection(S[arm], mate) for arm in (ARMS[0], "rim length")}
    names = np.array([x["stem"] for x in items])
    np.savez_compressed(os.path.join(OUT, f"{tag}_scores.npz"), names=names, n=n, planarity=pl,
                        mate=np.array([mate.get(i, -1) for i in range(len(items))]),
                        **{"S_" + METHOD.get(a, a.replace(" ", "_")): S[a] for a in ARMS})
    return dict(n_rims=len(items), n_queries=len(pq[ARMS[0]]), chance_top1=100.0 / (len(items) - 1),
                table=table, datc_vs=vs, detection=det,
                zscore_stats=zstats), alph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--pools", nargs="*", default=["fb", "presious"], choices=["fb", "presious"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"== table1_reviewed {time.ctime()} {' '.join(sys.argv)}")
    res = {}
    if "fb" in a.pools:
        items = load("fb_pool60")
        mate = ST.mates_by_object([x["object"] for x in items])
        res["fantastic_breaks"], alph = run_pool("fb", items, mate, a.procs, log)
        alph.to_json(os.path.join(OUT, "fb_alphabet.json"))
    if "presious" in a.pools:
        items = load("presious")
        mate = {i: int(x["mate"]) for i, x in enumerate(items) if int(x["mate"]) >= 0}
        res["presious"], _ = run_pool("presious", items, mate, a.procs, log)
    for pool, r in res.items():
        log(f"\n{pool}: {r['n_rims']} rims, {r['n_queries']} queries, chance top-1 {r['chance_top1']:.1f}%")
        for arm, t in r["table"].items():
            log(f"  ROW {pool:16s} {arm:28s} top1 {t['top1']:5.1f}  top5 {t['top5']:5.1f}")
        for arm, c in r["datc_vs"].items():
            log(f"  DaTC vs {arm:28s} objects {c['o_win']}:{c['o_lose']} (p {c['o_p']:.4f})"
                f"  queries {c['q_win']}:{c['q_lose']}")
        for arm, d in r["detection"].items():
            log(f"  detection {arm:28s} AUC {d['auc']:.3f}  AP {d['ap']:.3f}")
    path = os.path.join(OUT, "table1_reviewed.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1)
    log("REVIEWED DONE")


if __name__ == "__main__":
    main()
