"""Dataset loaders and rim pools.

Fantastic Breaks (Lamb et al., CVPR 2023)
    <root>/<category>/<object>/model_b_0.ply, model_r_0.ply
    The two fragments of one broken object, already in their assembled pose.
PRESIOUS (Papaioannou et al. 2017; Theoharis and Papaioannou 2013)
    <root>/<Object>/**/<Fragment>_low.obj, <Fragment>_med.obj, *_Reassembled.obj
    The reassembled file is the low-resolution fragments concatenated after
    rigid transforms; the poses are recovered from it by Kabsch per block.

Roots come from the arguments, else from the environment variables
DATC_FB_ROOT and DATC_PRESIOUS_ROOT. Nothing is downloaded.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from . import geometry as G
from .rims import component_rims, contact_mask, rim_of

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "splits" / "fantastic_breaks.json"

# Protocol constants of the paper (Section IV)
FB_DS_REL = 0.01          # contact threshold unit: 0.01 x joint bounding-box diagonal
FB_EPS = 1.5              # contact threshold, in that unit
FB_GLOBAL_DS = 0.0142     # one sampling step for every Fantastic Breaks rim
PR_DS_REL = 0.01          # PRESIOUS: 0.01 x median fragment diagonal
PR_EPS = 1.0              # PRESIOUS contact threshold, in that unit
PR_GLOBAL_DS = 4.85       # one sampling step for every PRESIOUS rim
PR_OBJECTS = ("DoraBlock", "DoraColumn", "DoraColumnBase", "DoraEmbrasure", "Tombstone")
PR_MIN_FACES = 200
SM_PUBLISHED = 2.0        # smoothing of the symbol code, in samples


def root_of(arg, env):
    r = arg or os.environ.get(env)
    if not r:
        raise SystemExit(f"dataset root not given: pass it or set {env}")
    if not os.path.isdir(r):
        raise SystemExit(f"dataset root {r!r} is not a directory")
    return r


def splits():
    with open(SPLITS, encoding="utf-8") as f:
        return json.load(f)


def read_mesh(path):
    return pv.read(path).extract_surface().triangulate()


# ------------------------------------------------------------------ Fantastic Breaks
def fb_pair(root, name):
    """(mesh_b, mesh_r) of object `name` = '<cat>_<obj>'."""
    cat, obj = name.split("_", 1)
    d = os.path.join(root, cat, obj)
    return read_mesh(os.path.join(d, "model_b_0.ply")), read_mesh(os.path.join(d, "model_r_0.ply"))


def fb_masks(mb, mr, ds_rel=FB_DS_REL, eps=FB_EPS):
    """Contact masks of both fragments; threshold eps x ds_rel x joint diagonal."""
    PB, PR = np.asarray(mb.points), np.asarray(mr.points)
    diag = float(np.linalg.norm(np.ptp(np.vstack([PB, PR]), 0)))
    e = eps * ds_rel * diag
    return contact_mask(mb, PR, e), contact_mask(mr, PB, e), diag, e


def fb_object_rims(root, name, ds=FB_GLOBAL_DS, sm=SM_PUBLISHED, eps=FB_EPS, rule="intact"):
    """Both rims of one object, or None if either side has no loop."""
    mb, mr = fb_pair(root, name)
    fb_, fr_, diag, e = fb_masks(mb, mr, eps=eps)
    A = rim_of(mb, fb_, ds, sm, rule=rule)
    B = rim_of(mr, fr_, ds, sm, rule=rule)
    if A is None or B is None:
        return None
    out = []
    for side, C in (("b", A), ("r", B)):
        n = len(C["alpha"])
        out.append(dict(object=name, side=side, stem=f"{name}/{side}", ds=float(ds),
                        n=n, arc=float(n * ds), planarity=G.planarity(C["pts"]),
                        closed=True, diag=diag, eps=e, **C))
    return out


def fb_pool(root, names, progress=print, **kw):
    """Rims of the listed objects, two per object, in list order."""
    items = []
    for k, name in enumerate(names):
        got = fb_object_rims(root, name, **kw)
        if got is None:
            raise RuntimeError(f"{name}: no rim on one side; the listed pools all have both")
        items += got
        if progress:
            progress(f"  [{k+1}/{len(names)}] {name}  b:n={got[0]['n']:4d}  r:n={got[1]['n']:4d}")
    return items


# ------------------------------------------------------------------ PRESIOUS
def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def presious_object(root, name, res="med"):
    """Fragments of one PRESIOUS object in the assembled frame, at `res`.

    Returns {'<Object>/<Fragment>': mesh} in the order of the blocks of the
    reassembled file, and the worst pose residual. The OBJ files are triangle
    soups; coincident vertices are merged after the rigid transform.
    """
    try:  # these OBJs carry lines VTK does not parse; silence its warning channel only
        import vtkmodules.vtkCommonCore as vcc
        vcc.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass
    d = os.path.join(root, name)
    objs = glob.glob(os.path.join(d, "**", "*.obj"), recursive=True)
    reas = [p for p in objs if "eassembl" in p.lower()]
    lows = sorted(p for p in objs if "eassembl" not in p.lower() and p.endswith("_low.obj"))
    if not reas or len(lows) < 2:
        raise RuntimeError(f"{name}: no reassembled file or fewer than two fragments under {d}")
    RP = np.asarray(read_mesh(reas[0]).points)
    F = [(os.path.splitext(os.path.basename(p))[0].replace("_low", ""), read_mesh(p), p) for p in lows]
    if sum(m.n_points for _, m, _ in F) != len(RP):
        raise RuntimeError(f"{name}: vertex total of the fragments != reassembled file")
    placed = {}

    def walk(off, unused):          # contiguous blocks, backtracking
        if off == len(RP):
            return True
        for i in list(unused):
            nm, m, _ = F[i]
            P = np.asarray(m.points)
            n = len(P)
            if off + n > len(RP):
                continue
            R, t = _kabsch(P, RP[off:off + n])
            e = float(np.linalg.norm((R @ P.T).T + t - RP[off:off + n], axis=1).mean())
            if e > 1e-2:
                continue
            placed[i] = (nm, F[i][2], R, t, e)
            unused.discard(i)
            if walk(off + n, unused):
                return True
            unused.add(i)
            placed.pop(i, None)
        return False

    if not walk(0, set(range(len(F)))):
        raise RuntimeError(f"{name}: no rigid decomposition of the reassembled file")
    out, worst = {}, 0.0
    for i, (nm, low, R, t, e) in placed.items():
        p = low if res == "low" else low[:-len("_low.obj")] + (".obj" if res == "full" else f"_{res}.obj")
        if not os.path.exists(p):
            raise RuntimeError(f"{name}: {os.path.basename(p)} missing")
        mm = read_mesh(p)
        mm.points = (R @ np.asarray(mm.points).T).T + t
        mm = mm.clean(tolerance=1e-8).extract_surface().triangulate()
        out[f"{name}/{nm}"] = mm
        worst = max(worst, e)
    return out, worst


def presious_rims(meshes, eps_rel=PR_EPS, ds_rel=PR_DS_REL, sm=SM_PUBLISHED,
                  min_faces=PR_MIN_FACES, ds_override=PR_GLOBAL_DS):
    """Rims of one object: contact with any other fragment (the neighbour is
    never named), one rim per connected component of the contact region."""
    scale = float(np.median([np.linalg.norm(np.ptp(np.asarray(m.points), 0)) for m in meshes.values()]))
    ds_own = ds_rel * scale
    trees = {k: cKDTree(np.asarray(m.points)) for k, m in meshes.items()}
    ds_use = ds_own if ds_override is None else float(ds_override)
    items = []
    for k, m in meshes.items():
        P = np.asarray(m.points)
        best = np.full(len(P), np.inf)
        for j, t in trees.items():
            if j != k:
                best = np.minimum(best, t.query(P, workers=-1)[0])
        fmask = (best < eps_rel * ds_own)[G.faces_of(m)].all(1)
        if not fmask.any():
            continue
        for r in component_rims(m, fmask, ds_use, sm, min_faces):
            r.update(frag=k, ds=ds_use, eps=eps_rel * ds_own, stem=f"{k}#r{r['rim']}")
            items.append(r)
    return items


def presious_pool(root, objects=PR_OBJECTS, res="med", progress=print, **kw):
    """All rims of the listed objects and their oracle mates.

    The mate of a rim is the rim on another fragment of the same object whose
    region centroid is nearest in the assembled pose. It is used only for
    scoring; the relation need not be symmetric."""
    items = []
    for name in objects:
        meshes, worst = presious_object(root, name, res)
        got = presious_rims(meshes, **kw)
        for x in got:
            x["object"] = name
            x["stem"] = f"{name}/{x['stem']}"
            x["n"] = len(x["alpha"])
            x["arc"] = float(x["n"] * x["ds"])
            x["planarity"] = G.planarity(x["pts"])
        items += got
        if progress:
            progress(f"  {name}: {len(meshes)} fragments, pose residual {worst:.1e}, {len(got)} rims")
    mate = np.full(len(items), -1, int)
    for i, a in enumerate(items):
        best, bd = -1, np.inf
        for j, b in enumerate(items):
            if b["object"] != a["object"] or b["frag"] == a["frag"]:
                continue
            dd = float(np.linalg.norm(a["centroid"] - b["centroid"]))
            if dd < bd:
                best, bd = j, dd
        mate[i] = best
    return items, mate
