"""One pair of fragments through the whole method, for the app and for scripts.

    meshes -> fracture regions -> rims -> (alpha, beta) -> symbols
           -> cyclic Smith-Waterman -> RANSAC placement -> match / no match

The regions come either from contact in the assembled pose (as in the paper)
or from a per-vertex mask file supplied by the user. Table I is not produced
here; the experiment scripts call the same modules directly.

The match / no-match verdict compares two numbers of the pair with thresholds
chosen on the development objects of the Fantastic Breaks pool and measured on
its test objects (experiments/pair_verdict.py -> reference/verdict.json):
  score    Smith-Waterman score / (n_A + n_B), code fitted on the two rims
  inliers  share of the aligned samples within 3 ds after the RANSAC fit
The verdict reports those measured rates; it is a decision rule with known
error, not a proof that two fragments belong together.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as G
from . import kernels as K
from . import pose as PO
from .alphabet import Alphabet
from .rims import contact_mask, rim_of

VERDICT = Path(__file__).resolve().parents[2] / "reference" / "verdict.json"
SW_W = np.array([6.0, 4.0])


@dataclass
class Settings:
    ds: float = 0.0142          # sampling step (dataset units)
    ds_rel: float | None = None  # if set: ds = ds_rel x joint diagonal
    eps: float = 1.5            # contact threshold in units of 0.01 x joint diagonal
    sm: float = 2.0             # Gaussian smoothing of the points, in samples
    normals: str = "intact"     # intact | all | fracture
    na: int = 9
    nb: int = 5
    min_pts: int = 40


@dataclass
class PairResult:
    settings: Settings
    ds: float
    masks: tuple
    rims: tuple
    alphabet: Alphabet
    alphabet_source: str
    symbols: tuple
    score: float                # SW score / (n_A + n_B), the Table I scale
    raw_score: float
    direction: str              # direction of the best score: "mate" (B reversed, alpha negated) or "forward"
    path: np.ndarray            # (k, 2) matched samples of that direction: index on A, index on B
    pose: dict                  # RANSAC placement of B onto A (R, t, inliers, direction, rotation_deg)
    features: dict              # verdict inputs (score with the pair code, inlier share)
    verdict: dict               # match / no match, the rule and its measured rates
    timings: dict = field(default_factory=dict)
    log: list = field(default_factory=list)


def load_mask(path: str, n_points: int) -> np.ndarray:
    """Per-vertex boolean mask from .npy, .txt (one 0/1 per line) or .npz ('mask')."""
    if path.endswith(".npz"):
        m = np.load(path)["mask"]
    elif path.endswith(".npy"):
        m = np.load(path)
    else:
        m = np.loadtxt(path)
    m = np.asarray(m).astype(bool).ravel()
    if len(m) != n_points:
        raise ValueError(f"mask has {len(m)} entries, mesh has {n_points} vertices")
    return m


def frame_at(rim: dict, i: int, sm: float):
    """Point, tangent t, in-surface direction g = n x t and normal n at sample i,
    on the smoothed polyline the angles were computed from."""
    P = G.smooth_curve(rim["P"], sm, closed=True)
    N = rim["N"]
    i = int(np.clip(i, 0, len(P) - 3))
    t = P[i + 1] - P[i]
    t /= np.linalg.norm(t) + 1e-12
    n = N[i + 1] - np.dot(N[i + 1], t) * t
    n /= np.linalg.norm(n) + 1e-12
    g = np.cross(n, t)
    return P[i + 1], t, g, n


# ------------------------------------------------------------------ rims -> alignment -> pose
def align(rimA, rimB, alphabet):
    """Cyclic Smith-Waterman of A against B in both directions of travel.

    Returns {direction: dict(raw, path)}; path rows are (sample of A, sample of B)."""
    sA = alphabet.encode(rimA["alpha"], rimA["beta"])
    sB = alphabet.encode(rimB["alpha"], rimB["beta"])
    scale = np.array([float(np.ptp(alphabet.a_ctr)) + 1e-9, float(np.ptp(alphabet.b_ctr)) + 1e-9])
    X = alphabet.centres(sA)
    m = len(sB)
    out = {}
    for direction, s2 in (("mate", alphabet.reverse(sB)), ("forward", sB)):
        Y = alphabet.centres(s2)
        raw, path = K.sw_nd_extent(X, np.ascontiguousarray(np.vstack([Y, Y])), scale, SW_W)
        path = path.copy()
        path[:, 1] %= m
        if direction == "mate":        # index on the reversed string -> index on B
            path[:, 1] = m - 1 - path[:, 1]
        out[direction] = dict(raw=float(raw), path=path)
    return out, (sA, sB)


def fit_poses(rimA, rimB, aligned, ds, tau_ds=PO.TAU_DS):
    """RANSAC rigid fit of B's matched samples onto A's, for each direction."""
    for d, a in aligned.items():
        pr = a["path"]
        fit = None
        if len(pr) >= 3:
            fit = PO.ransac_kabsch(rimB["pts"][pr[:, 1]], rimA["pts"][pr[:, 0]], tau_ds * ds)
        if fit is None:
            a.update(R=None, t=None, inliers=np.zeros(len(pr), bool), inlier_share=0.0)
        else:
            R, t, mask = fit
            a.update(R=R, t=t, inliers=mask, inlier_share=float(mask.mean()))
    return aligned


def pair_features(rimA, rimB, ds, na=9, nb=5):
    """The two verdict inputs, with the code fitted on the two rims (direction free)."""
    al = Alphabet(na, nb).fit([rimA["alpha"], rimB["alpha"]], [rimA["beta"], rimB["beta"]])
    aligned, _ = align(rimA, rimB, al)
    fit_poses(rimA, rimB, aligned, ds)
    n = len(rimA["alpha"]) + len(rimB["alpha"])
    return dict(score=max(a["raw"] for a in aligned.values()) / n,
                inliers=max(a["inlier_share"] for a in aligned.values()))


def verdict_of(features):
    """Apply reference/verdict.json; returns dict(match, rule, text)."""
    if not VERDICT.exists():
        return dict(match=None, rule=None, text="no verdict: reference/verdict.json missing "
                                               "(run experiments/pair_verdict.py)")
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    rule = v["selected"]
    ok = all(features[k] >= thr for k, thr in rule["thresholds"].items())
    te, pr = rule["test"], rule["presious"]
    cond = " and ".join(f"{k} >= {thr:.4g}" for k, thr in rule["thresholds"].items())
    text = (f"{'MATCH' if ok else 'NO MATCH'}: rule '{cond}' (thresholds set on development objects). "
            f"On the {te['n_pos']} mated and {te['n_neg']} unmated test-half pairs of Fantastic Breaks "
            f"it accepted {100 * te['tpr']:.1f}% of mates and {100 * te['fpr']:.1f}% of non-mates; "
            f"on the {pr['n_pos']} mated and {pr['n_neg']} unmated PRESIOUS pairs, {100 * pr['tpr']:.1f}% and "
            f"{100 * pr['fpr']:.1f}%.")
    return dict(match=bool(ok), rule=rule, text=text)


def run_pair(mesh_a, mesh_b, st: Settings, mask_a=None, mask_b=None, alphabet: Alphabet | None = None,
             progress=None) -> PairResult:
    """Run DaTC on two fragments.

    mask_a / mask_b: per-vertex masks of the fracture region; when None the
    region is contact with the other fragment (both must be in their assembled
    pose). alphabet: a fitted code (e.g. the 120-rim Fantastic Breaks pool code
    saved by table1_reviewed.py) for the displayed alignment; when None it is
    fitted on the two rims. The verdict always uses the code fitted on the rims,
    the setting its thresholds were measured with.
    """
    say = progress or (lambda p, s: None)
    log, tm = [], {}
    t0 = time.time()
    PA, PB = np.asarray(mesh_a.points), np.asarray(mesh_b.points)
    diag = float(np.linalg.norm(np.ptp(np.vstack([PA, PB]), 0)))
    ds = st.ds if st.ds_rel is None else st.ds_rel * diag
    say(5, "fracture regions")
    if mask_a is None or mask_b is None:
        e = st.eps * 0.01 * diag
        fa, fb = contact_mask(mesh_a, PB, e), contact_mask(mesh_b, PA, e)
        log.append(f"regions: contact within {e:.4g} ({st.eps:g} x 0.01 x joint diagonal {diag:.4g})"
                   " -- datc.rims.contact_mask")
    else:
        fa = mask_a[G.faces_of(mesh_a)].all(1)
        fb = mask_b[G.faces_of(mesh_b)].all(1)
        log.append("regions: faces whose three vertices are in the given vertex masks")
    log.append(f"region faces: A {int(fa.sum())} of {mesh_a.n_cells}, B {int(fb.sum())} of {mesh_b.n_cells}")
    tm["regions"] = time.time() - t0
    say(25, "rims")
    t1 = time.time()
    rims = []
    for tag, m, f in (("A", mesh_a, fa), ("B", mesh_b, fb)):
        if not f.any():
            raise ValueError(f"fragment {tag}: empty fracture region (no contact: the fragments may not be "
                             "in their assembled pose, or eps is too small)")
        r = rim_of(m, f, ds, st.sm, st.min_pts, st.normals)
        if r is None:
            raise ValueError(f"fragment {tag}: no boundary loop of at least {st.min_pts} vertices")
        r["bulk"] = np.asarray(m.points).mean(0)
        rims.append(r)
        log.append(f"rim {tag}: {r['n_loops']} loop(s), longest kept, n = {len(r['alpha'])} samples at ds = {ds:.4g}"
                   " -- datc.rims.rim_of")
    tm["rims"] = time.time() - t1
    say(50, "symbols, alignment, placement")
    t2 = time.time()
    m = match_rims(rims[0], rims[1], ds, rims[0]["bulk"], rims[1]["bulk"], st.na, st.nb, alphabet, log)
    tm["match"] = time.time() - t2
    say(100, "done")
    return PairResult(st, float(ds), (fa, fb), tuple(rims), m["alphabet"], m["alphabet_source"], m["symbols"],
                      m["score"], m["raw"], m["direction"], m["path"], m["pose"], m["features"], m["verdict"],
                      tm, log)


def match_rims(rimA, rimB, ds, bulkA, bulkB, na=9, nb=5, alphabet=None, log=None):
    """Rims -> symbols -> cyclic SW (both directions) -> RANSAC placement -> verdict.

    bulkA / bulkB: vertex centroids of the two fragments (orient the fracture planes).
    Returns a dict with alphabet, symbols, score, raw, direction, path, aligned, pose,
    features, verdict. Rims must be in their fragments' frames."""
    log = [] if log is None else log
    if alphabet is None:
        alphabet = Alphabet(na, nb).fit([rimA["alpha"], rimB["alpha"]], [rimA["beta"], rimB["beta"]])
        src = "fitted on these two rims"
    else:
        src = "loaded"
    log.append(f"code: {alphabet.na} x {alphabet.nb} quantile bins, {src} -- datc.alphabet.Alphabet")
    aligned, sy = align(rimA, rimB, alphabet)
    direction = max(aligned, key=lambda d: aligned[d]["raw"])
    raw, path = aligned[direction]["raw"], aligned[direction]["path"]
    score = raw / (len(sy[0]) + len(sy[1]))
    log.append(f"alignment: cyclic Smith-Waterman, best direction '{direction}', {len(path)} matched samples, "
               f"score {raw:.2f} / (n_A + n_B) = {score:.4f} -- datc.pipeline.align")
    fit_poses(rimA, rimB, aligned, ds)
    nA = PO.plane_normal(rimA["pts"], bulkA)
    nB = PO.plane_normal(rimB["pts"], bulkB)
    cand = [(d, a) for d, a in aligned.items() if a["R"] is not None]
    if cand:   # fracture-plane opposition picks the direction of travel (the paper's placement rule)
        d, a = max(cand, key=lambda da: -float(np.dot(nA, da[1]["R"] @ nB)))
        pose = dict(direction=d, R=a["R"], t=a["t"], path=a["path"], inliers=a["inliers"],
                    inlier_share=a["inlier_share"], rotation_deg=PO.rotation_angle(a["R"]),
                    opposition=float(-np.dot(nA, a["R"] @ nB)))
        log.append(f"placement: RANSAC (tau {PO.TAU_DS:g} ds, {PO.N_ITER} iterations) on direction '{d}' "
                   f"(fracture planes opposed {pose['opposition']:+.2f}), {int(a['inliers'].sum())} of "
                   f"{len(a['path'])} matched samples inliers; turns B by {pose['rotation_deg']:.1f} deg "
                   "-- datc.pose.ransac_kabsch")
    else:
        pose = dict(direction=None, R=None, t=None, path=np.zeros((0, 2), int), inliers=np.zeros(0, bool),
                    inlier_share=0.0, rotation_deg=None, opposition=None)
        log.append("placement: RANSAC found no consensus of 3 samples")
    if src == "fitted on these two rims" and (alphabet.na, alphabet.nb) == (9, 5):
        feats = dict(score=score, inliers=max(a["inlier_share"] for a in aligned.values()))
    else:   # the verdict was calibrated with the 9 x 5 code fitted on the pair
        feats = pair_features(rimA, rimB, ds)
    v = verdict_of(feats)
    log.append(f"verdict inputs: score {feats['score']:.4f}, inlier share {feats['inliers']:.3f}")
    log.append(v["text"])
    return dict(alphabet=alphabet, alphabet_source=src, symbols=sy, score=float(score), raw=float(raw),
                direction=direction, path=path, aligned=aligned, pose=pose, features=feats, verdict=v)
