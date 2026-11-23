"""Table II, lower block: DaTC and (kappa, tau) tuned alike.

The three configurations were selected on the 30 development objects of the
retrieval pool over the same 63-point grid (smoothing x matcher x transform);
this script evaluates the selected ones, it does not repeat the selection:

  DaTC rank code + local al.        (alpha, beta), sm 5, rank transform, cyclic SW
  (kappa,tau) tuned + cyclic DTW    (kappa, tau),  sm 8, rank transform, cyclic DTW
  (kappa,tau) tuned + local al.     (kappa, tau),  sm 8, rank transform, cyclic SW

Evaluation sets
  test half   60 rims of the 30 test objects; the transform is fitted on the
              development rims and frozen; a query ranks the other 59 test rims
  held-out    92 rims of 46 objects outside the pool; transform fitted on the
              120 pool rims; a query ranks the other 91
  PRESIOUS    47 rims; transform fitted on its own rims (label free)

Usage
  python -u experiments/table1_tuned.py [--procs 4]
Needs cache/fb_pool60.pkl, cache/fb_heldout.pkl, cache/presious.pkl.
Writes results/table1_tuned/table1_tuned.json and the score matrices.
Last line on success: 'TUNED DONE'.
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
from datc import datasets as DS, normalise as NZ, scoring as SC, stats as ST  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")
OUT = os.path.join(HERE, "..", "results", "table1_tuned")
CONFIGS = {  # label: (channel set, smoothing, matcher)
    "DaTC rank code + local al.": ("D", 5.0, "swc"),
    "(kappa,tau) tuned + cyclic DTW": ("K", 8.0, "dtw"),
    "(kappa,tau) tuned + local al.": ("K", 8.0, "swc"),
}
DATC = "DaTC rank code + local al."


def load(name):
    p = os.path.join(CACHE, name + ".pkl")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing: run experiments/build_pools.py first")
    with open(p, "rb") as f:
        return pickle.load(f)["items"]


def score(tag, seqs, mates, scale, w, kind, use, procs, log):
    t0 = time.time()
    S = SC.symmetric_matrix("tuned", (seqs, mates), (kind, scale, w), use=use, procs=procs)
    log(f"    {tag} ({time.time()-t0:.0f}s)")
    np.savez_compressed(os.path.join(OUT, f"S_{tag}.npz"), S=S)
    return S


def block(name, S_by_cfg, mate, obj, use, log):
    pq = {k: ST.per_query(S, mate, use) for k, S in S_by_cfg.items()}
    table = {k: {m: 100 * v for m, v in ST.summary(q).items() if m in ("top1", "top5", "top10")}
             for k, q in pq.items()}
    vs = {k: ST.clustered(pq[DATC], pq[k], obj) for k in pq if k != DATC}
    log(f"  {name}: {len(pq[DATC])} queries")
    for k, t in table.items():
        log(f"  ROW {name:10s} {k:32s} top1 {t['top1']:5.1f}  top5 {t['top5']:5.1f}")
    for k, c in vs.items():
        log(f"  DaTC vs {k:32s} objects {c['o_win']}:{c['o_lose']} (p {c['o_p']:.3f})"
            f"  queries {c['q_win']}:{c['q_lose']}")
    return dict(n_queries=len(pq[DATC]), table=table, datc_vs=vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"== table1_tuned {time.ctime()} {' '.join(sys.argv)}")
    sp = DS.splits()
    pool = load("fb_pool60")
    held = load("fb_heldout")
    pres = load("presious")
    obj = [x["object"] for x in pool]
    mate = ST.mates_by_object(obj)
    dev = [i for i, o in enumerate(obj) if o in set(sp["dev"])]
    test = [i for i, o in enumerate(obj) if o in set(sp["test"])]
    assert len(dev) == 60 and len(test) == 60 and not set(dev) & set(test)
    hobj = [x["object"] for x in held]
    hmate = ST.mates_by_object(hobj)
    pobj = [x["object"] for x in pres]
    pmate = {i: int(x["mate"]) for i, x in enumerate(pres) if int(x["mate"]) >= 0}
    pds = float(pres[0]["ds"])

    St, Sh, Sp = {}, {}, {}
    for lab, (chset, sm, kind) in CONFIGS.items():
        log(f"== {lab}: channels {NZ.CHSETS[chset]}, sm {sm:g}, rank, {kind}")
        Dall = NZ.descs(pool, sm, DS.FB_GLOBAL_DS)
        seqs, mates, scale, w = NZ.build_seqs(NZ.subset(Dall, dev), Dall, chset, "rank")
        St[lab] = score(f"test_{chset}_sm{sm:g}_{kind}", seqs, mates, scale, w, kind, test, a.procs, log)
        Dh = NZ.descs(held, sm, DS.FB_GLOBAL_DS)
        seqs, mates, scale, w = NZ.build_seqs(Dall, Dh, chset, "rank")
        Sh[lab] = score(f"heldout_{chset}_sm{sm:g}_{kind}", seqs, mates, scale, w, kind, None, a.procs, log)
        Dp = NZ.descs(pres, sm, pds)
        seqs, mates, scale, w = NZ.build_seqs(Dp, Dp, chset, "rank")
        Sp[lab] = score(f"presious_{chset}_sm{sm:g}_{kind}", seqs, mates, scale, w, kind, None, a.procs, log)

    res = dict(configs={k: dict(channels=list(NZ.CHSETS[c]), sm=s, transform="rank", matcher=m)
                        for k, (c, s, m) in CONFIGS.items()},
               test_half=block("test half", St, mate, obj, test, log),
               heldout=block("held-out", Sh, hmate, hobj, None, log),
               presious=block("PRESIOUS", Sp, pmate, pobj, None, log))
    res["test_half"]["chance_top1"] = 100 / 59
    res["heldout"]["chance_top1"] = 100 / (len(held) - 1)
    res["presious"]["chance_top1"] = 100 / (len(pres) - 1)
    with open(os.path.join(OUT, "table1_tuned.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    log("TUNED DONE")


if __name__ == "__main__":
    main()
