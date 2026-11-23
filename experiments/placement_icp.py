"""Placement error of RANSAC and of RANSAC + ICP on the 60 mated pairs of the
Fantastic Breaks retrieval pool (not in the paper; documents the app's ICP option).

Fantastic Breaks stores each pair assembled, so the true pose of B is the identity.
For each object: the pool rims (cache/fb_pool60.pkl), datc.pipeline.match_rims
(9 x 5 code fitted on the pair, cyclic Smith-Waterman, RANSAC), then
datc.pipeline.icp_refine on the contact regions (datasets.fb_masks) with the app's
default reject distance, 3 ds. Errors: rotation angle of the fitted pose, and the
RMS displacement of B's vertices as a percentage of the joint diagonal D.

Usage
  python -u experiments/placement_icp.py [--fb-root DIR]
Writes results/placement_icp/placement_icp.json; last line 'PLACEMENT DONE'.
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
from datc import datasets as DS  # noqa: E402
from datc.pipeline import icp_refine, match_rims  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "placement_icp")
TAU_DS = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb-root", default=None)
    a = ap.parse_args()
    root = DS.root_of(a.fb_root, "DATC_FB_ROOT")
    os.makedirs(OUT, exist_ok=True)
    items = pickle.load(open(os.path.join(HERE, "..", "cache", "fb_pool60.pkl"), "rb"))["items"]
    by = {}
    for x in items:
        by.setdefault(x["object"], {})[x["side"]] = x
    rows = []
    for k, (obj, d) in enumerate(sorted(by.items())):
        t0 = time.time()
        mb, mr = DS.fb_pair(root, obj)
        fb_, fr_, _, _ = DS.fb_masks(mb, mr)
        rims = [dict(alpha=np.asarray(d[s]["alpha"]), beta=np.asarray(d[s]["beta"]),
                     pts=np.asarray(d[s]["pts"], float)[:len(d[s]["alpha"])]) for s in ("b", "r")]
        r = match_rims(rims[0], rims[1], DS.FB_GLOBAL_DS, np.asarray(mb.points).mean(0),
                       np.asarray(mr.points).mean(0))
        V = np.asarray(mr.points)
        D = float(np.linalg.norm(np.ptp(np.vstack([np.asarray(mb.points), V]), 0)))

        def err(p):
            if p is None or p["R"] is None:
                return None, None
            return p["rotation_deg"], 100 * float(np.sqrt(((V @ p["R"].T + p["t"] - V) ** 2).sum(1).mean())) / D

        log = []
        pi = icp_refine(mb, fb_, mr, fr_, r["pose"], DS.FB_GLOBAL_DS, TAU_DS, log)
        (r0, e0), (r1, e1) = err(r["pose"]), err(pi)
        rows.append(dict(object=obj, verdict=bool(r["verdict"]["match"]), ransac_rot_deg=r0, ransac_rms_pct_D=e0,
                         icp_rot_deg=r1, icp_rms_pct_D=e1, icp_log=log[-1] if log else None))
        f = lambda v, fmt: "  n/a " if v is None else format(v, fmt)  # noqa: E731
        print(f"[{k + 1:2d}/{len(by)}] {obj} {'MATCH' if r['verdict']['match'] else 'NO   '} "
              f"RANSAC rot {f(r0, '6.1f')} rms {f(e0, '6.2f')}%D | ICP rot {f(r1, '6.1f')} rms {f(e1, '6.2f')}%D "
              f"({time.time() - t0:.0f}s)", flush=True)

    def final(r, key):      # the placement the app shows: ICP when it ran, else RANSAC
        v = r[f"icp_{key}"]
        return r[f"ransac_{key}"] if v is None else v

    nan = lambda v: np.nan if v is None else v  # noqa: E731
    e0 = np.array([nan(r["ransac_rms_pct_D"]) for r in rows])
    e1 = np.array([nan(final(r, "rms_pct_D")) for r in rows])
    a0 = np.array([nan(r["ransac_rot_deg"]) for r in rows])
    a1 = np.array([nan(final(r, "rot_deg")) for r in rows])
    ran = [r["icp_rms_pct_D"] is not None for r in rows]
    summ = dict(n=len(rows), icp_ran=int(sum(ran)),
                within_pct_D={str(t): [int(np.sum(e0 < t)), int(np.sum(e1 < t))] for t in (1, 2, 5)},
                within_deg={str(t): [int(np.sum(a0 < t)), int(np.sum(a1 < t))] for t in (5, 30)},
                icp_lowered=int(np.sum(e1 < e0)), icp_raised=int(np.sum(e1 > e0)),
                median_rms_pct_D=[float(np.nanmedian(e0)), float(np.nanmedian(e1))])
    print(json.dumps(summ))
    with open(os.path.join(OUT, "placement_icp.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(note=__doc__.strip().splitlines()[0] + " Pairs: [RANSAC, RANSAC + ICP].", tau_ds=TAU_DS,
                       summary=summ, rows=rows), fh, indent=1)
    print("PLACEMENT DONE", flush=True)


if __name__ == "__main__":
    main()
