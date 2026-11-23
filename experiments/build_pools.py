"""Build the three rim pools of the paper and cache them.

  fb_pool60     Fantastic Breaks retrieval pool: 60 objects, 120 rims
  fb_heldout    Fantastic Breaks held-out pool: 46 objects, 92 rims
  presious      PRESIOUS: 5 objects, 47 rims, oracle mates from the assembled pose

Usage
  python -u experiments/build_pools.py --fb-root E:/Fantastic_break --presious-root D:/presious
  python -u experiments/build_pools.py --only fb_pool60          # one pool
Writes cache/<pool>.pkl and cache/<pool>.json (counts, parameters, timings).
About 6 s per Fantastic Breaks object and a few minutes per PRESIOUS object.
Last line on success: 'POOLS DONE'.
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from datc import __version__, datasets as DS  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")
POOLS = ("fb_pool60", "fb_heldout", "presious")


def save(name, items, meta):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name + ".pkl"), "wb") as f:
        pickle.dump(dict(items=items, meta=meta), f)
    with open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"-> cache/{name}.pkl ({len(items)} rims)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb-root", default=None, help="Fantastic Breaks root (or DATC_FB_ROOT)")
    ap.add_argument("--presious-root", default=None, help="PRESIOUS root (or DATC_PRESIOUS_ROOT)")
    ap.add_argument("--only", nargs="*", choices=POOLS, default=list(POOLS))
    a = ap.parse_args()
    sp = DS.splits()
    for name in a.only:
        t0 = time.time()
        print(f"== {name}", flush=True)
        if name.startswith("fb_"):
            root = DS.root_of(a.fb_root, "DATC_FB_ROOT")
            objs = sp["pool60"] if name == "fb_pool60" else sp["heldout"]
            items = DS.fb_pool(root, objs, progress=lambda s: print(s, flush=True))
            meta = dict(pool=name, version=__version__, n_objects=len(objs), n_rims=len(items),
                        ds=DS.FB_GLOBAL_DS, eps=f"{DS.FB_EPS} x {DS.FB_DS_REL} x joint diagonal",
                        sm=DS.SM_PUBLISHED, normals="intact", seconds=time.time() - t0)
        else:
            root = DS.root_of(a.presious_root, "DATC_PRESIOUS_ROOT")
            items, mate = DS.presious_pool(root, progress=lambda s: print(s, flush=True))
            for x, j in zip(items, mate):
                x["mate"] = int(j)
            meta = dict(pool=name, version=__version__, objects=list(DS.PR_OBJECTS), n_rims=len(items),
                        n_mated=int((mate >= 0).sum()), ds=DS.PR_GLOBAL_DS,
                        eps=f"{DS.PR_EPS} x {DS.PR_DS_REL} x median fragment diagonal",
                        sm=DS.SM_PUBLISHED, normals="intact", resolution="med",
                        seconds=time.time() - t0)
        save(name, items, meta)
    print("POOLS DONE", flush=True)


if __name__ == "__main__":
    main()
