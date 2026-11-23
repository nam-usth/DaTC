# DaTC: Darboux Turn Code for thin-shell fragment matching

Code for

> Vinh Nam Huynh, Hoang Ha Nguyen, Romain Raffin.
> **Matching Thin-Shell Fragments with a Darboux Turn Code of Their Fracture Boundaries.**
> RIVF 2026.

The fracture boundary of a thin-shell fragment (its *rim*) is a closed 3D curve
that lies on the original surface. DaTC records, at each sample along the rim,
how far the rim turns inside that surface (alpha) and out of it (beta), measured
in the Darboux frame built from the intact-side surface normal. The two angles
are quantised into a 9 x 5 symbol code (or rank-transformed), and two rims are
compared by cyclic local alignment (Smith-Waterman), trying both directions of
travel. The repository reproduces Table I of the paper from the meshes and
contains **DaTC Studio**, a PyQt application that runs the method on a pair of
fragments and shows each step.

![DaTC Studio](figures/app/selftest_00_00002_window.png)

## Contents

```
src/datc/
  geometry.py     normals, region boundary loops, resampling, smoothing, planarity
  descriptors.py  Darboux (alpha, beta) and Frenet (kappa, tau) turn angles
  rims.py         contact region -> longest boundary loop -> rim (one per component for PRESIOUS)
  alphabet.py     quantile symbol code, mirrored for the reversed walk of the mate
  kernels.py      Smith-Waterman and cyclic DTW kernels (numba)
  scoring.py      pair scores and score matrices for every arm of Table I
  normalise.py    z and rank transforms of the tuned comparison
  stats.py        expected-tie top-k, object-level sign test, ROC AUC / AP
  datasets.py     Fantastic Breaks and PRESIOUS loaders, the three rim pools
  pose.py         RANSAC + Kabsch placement from the matched samples
  pipeline.py     one pair of fragments end to end, with placement and match verdict (used by the app)
experiments/
  build_pools.py      the three rim pools, cached under cache/
  table1_reviewed.py  Table I, upper block
  table1_tuned.py     Table I, lower block
  make_table1.py      prints Table I and checks every cell against the paper
  pair_verdict.py     fits and measures the app's match / no-match rule
app/datc_studio.py    DaTC Studio (PyQt5 + PyVista)
splits/               the object lists of every pool and split
reference/            the values printed in the paper (make_table1.py) and the verdict rule (verdict.json)
tests/                kernels against slow references; invariance and tie-rule checks
```

## Install

Python 3.9 or later.

```
pip install -r requirements.txt
```

or `conda env create -f environment.yml`. The code was run with numpy 2.0.2,
scipy 1.13.1, pyvista 0.46.5, vtk 9.4.2, numba 0.60.0, scikit-learn 1.5.2 and
PyQt5 5.15.10 on Windows 11.

## Data

Neither dataset is redistributed here.

| dataset | where to get it | layout expected |
|---|---|---|
| Fantastic Breaks (Lamb et al., CVPR 2023) | the authors' release | `<root>/<category>/<object>/model_b_0.ply`, `model_r_0.ply` |
| PRESIOUS (Theoharis and Papaioannou 2013) | https://www.ntnu.edu/presious/3d-data-sets | `<root>/<Object>/**/<Fragment>_low.obj`, `<Fragment>_med.obj`, `*_Reassembled.obj` |

Our copy of Fantastic Breaks holds 132 of the 150 pairs of the public release;
every object used in the paper is listed in `splits/fantastic_breaks.json`.
PRESIOUS objects used: DoraBlock, DoraColumn, DoraColumnBase, DoraEmbrasure,
Tombstone, at the `_med` resolution. Fantastic Breaks fragments are already in
their assembled pose; PRESIOUS poses are recovered from the reassembled file.

Set the roots once:

```
set DATC_FB_ROOT=E:\Fantastic_break
set DATC_PRESIOUS_ROOT=D:\data\presious
```

(or pass `--fb-root` and `--presious-root` to `build_pools.py`).

## Reproduce Table I

```
python -u experiments/build_pools.py            # three pools -> cache/        (about 45 min)
python -u experiments/table1_reviewed.py        # upper block -> results/      (a few minutes)
python -u experiments/table1_tuned.py           # lower block -> results/      (a few minutes)
python experiments/make_table1.py               # prints Table I, checks it, exit code 0 if it matches
```

Times are for an 8-core laptop. `make_table1.py` compares every percentage to
the value printed in the paper (to its 0.05 precision), the object-level
sign-test counts exactly and the detection AUC / AP to 0.0005, and ends with
`TABLE I REPRODUCED` or `TABLE I MISMATCH`. Its output is written to
`results/table1.md`. The same four steps can be run from the *Retrieval* tab
of DaTC Studio.

What each block measures:

| block | pool | protocol |
|---|---|---|
| upper | Fantastic Breaks retrieval pool, 60 objects, 120 rims; PRESIOUS, 5 objects, 47 rims | settings fixed before this table (smoothing 2 samples, 9 x 5 code with bins fitted on the pool without labels); a query ranks every other rim of its pool |
| lower, test half | 30 test objects of the retrieval pool, 60 rims | configurations selected on the 30 development objects over the same 63-point grid; transforms fitted on development rims and frozen; a query ranks the other 59 test rims |
| lower, held-out | 46 objects outside the pool, 92 rims | every setting frozen; transforms fitted on the 120 pool rims |
| lower, PRESIOUS | 47 rims | transforms fitted on its own rims (label free) |

Ties count at their expected rank. The selection over the 63-point grid is not
repeated here: `table1_tuned.py` evaluates the three selected configurations
(listed at the top of the script).

## DaTC Studio

```
python app/datc_studio.py                                   # empty window
python app/datc_studio.py E:\Fantastic_break\00\00002       # one Fantastic Breaks object
python app/datc_studio.py frag_a.ply frag_b.ply             # any two fragments in assembled pose
```

*Pair* tab: choose two fragments, press **Run DaTC**. The 3D view shows the
fracture regions, the two rims, the samples the alignment matched, and the
Darboux frame (t, g, n) at a sample chosen with the slider; below it, alpha and
beta of rim A with the aligned samples of rim B, and a log naming the function
behind every step. B can be placed by the RANSAC fit of the matched samples,
and the pairing exported as a GIF. A MATCH / NO MATCH verdict is always shown.
The computation runs in a worker thread.
*Retrieval* tab: run the four reproduction steps, then browse each query's
ranking. A selected query and its first candidate (or its mate) are drawn in a
2 x 3 grid: the two fragments, the RANSAC placement; the two rims coloured by
their code, and the Smith-Waterman matches between them.

![Retrieval tab](figures/app/selftest_00_00002_retrieval_mate.png)
*Help* tab: `app/README.md`.

### Match verdict

The verdict answers whether the two rims are mates. Its
rule, fitted by `experiments/pair_verdict.py` on the development half of the
Fantastic Breaks retrieval pool (Youden J) and stored in
`reference/verdict.json`, is: score >= 0.0847 **and** inlier share >= 0.597,
where the inlier share is the fraction of matched samples within 3 ds after
RANSAC. Measured without refitting:

| pairs | mates accepted | non-mates accepted |
|---|---|---|
| Fantastic Breaks test half (30 mated, 1740 unmated) | 86.7% | 16.2% |
| PRESIOUS (26 mated, 1055 unmated) | 65.4% | 52.9% |

So on Fantastic Breaks about one non-mate pair in six is still called MATCH,
and on PRESIOUS the verdict barely separates mates from non-mates. Read it as
a screening flag, not as a decision. The verdict always uses the 9 x 5 code
fitted on the two rims, the setting it was measured with.

The rule is refitted with

```
python -u experiments/pair_verdict.py     # about 5 min with --procs 8
```

It is not part of the paper; the paper reports retrieval ranks only.

A headless check, used before every release:

```
python app/datc_studio.py --selftest E:\Fantastic_break\00\00002 figures/app/selftest_00_00002.png
```

It runs the pair through the window's worker, asserts that the rims equal those
of `datasets.fb_object_rims` and that a verdict is given, writes the GIF, and,
when the pools are built, draws three retrieval pairs (a query whose first
candidate is its mate, a query whose first candidate is not, and that query
with its mate) and saves screenshots. The object used, 00_00002, is itself one
of the missed mates: its two rims get NO MATCH (inlier share 29.8%).

## Tests

```
python -m pytest tests
```

The compiled kernels are compared with the slow NumPy implementations that
produced the paper's matrices, and must agree exactly. Other tests check rigid
invariance, that the reversed walk negates alpha and keeps beta, that the
mirrored symbol code equals the code of the negated angles, the Frenet values of
a helix, and the tie rule.

## Scope and known limits

This release covers the method and Table I. The other experiments of the paper
(partial overlap, edge wear and warp, resolution, the normal-source ablation)
are not in this repository. The limits stated in the paper apply: the fracture
region is given (contact in the assembled pose, or a mask), not detected; the
counterpart rims of Fantastic Breaks come from the same scan, so the retrieval
pool is the easy case; the Smith-Waterman parameters (match 2, gap open -4,
gap extend -0.6, weights 6 and 4) were set by hand and never tuned.

## Citation

```
@inproceedings{huynh2026datc,
  author    = {Huynh, Vinh Nam and Nguyen, Hoang Ha and Raffin, Romain},
  title     = {Matching Thin-Shell Fragments with a Darboux Turn Code of Their Fracture Boundaries},
  booktitle = {Proc. RIVF International Conference on Computing and Communication Technologies},
  year      = {2026}
}
```

## License

MIT, see `LICENSE`. The datasets keep their own licenses.
