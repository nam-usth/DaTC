# DaTC Studio

Runs the Darboux Turn Code on a pair of thin-shell fragments and shows every
step, and reproduces Table I of the paper.

```
python app/datc_studio.py [fragment_a fragment_b | fantastic_breaks_object_folder]
```

## Pair tab

1. **Fragments.** *Choose a Folder*, select a mesh and press *Set as A*, then
   another and *Set as B*. A Fantastic Breaks object folder (one holding
   `model_b_0.ply` and `model_r_0.ply`) sets both at once. The two fragments
   must be in their assembled pose when the region comes from contact.
2. **Fracture region.** *Contact in the assembled pose* (the paper's rule):
   the faces whose three vertices lie within eps of the other fragment.
   *Vertex mask files*: one 0/1 value per vertex for each fragment (`.txt`,
   `.npy`, or `.npz` with an array `mask`).
3. **Darboux turn code**, *Rims* group:
   - *step ds*: sampling step along the rim in mesh units (Fantastic Breaks
     0.0142, PRESIOUS 4.85); or tick *0.01 x joint diagonal*.
   - *contact eps*: contact threshold in units of 0.01 x the joint bounding-box
     diagonal (paper: 1.5 for Fantastic Breaks).
   - *smoothing sm*: Gaussian smoothing of the rim points, in samples (paper: 2).
   - *normals*: the faces whose normals build the frame at the rim; *intact*
     (paper) uses the faces outside the fracture region, the surface both mates
     share.

   *Code* group: the number of alpha and beta bins (paper: 9 x 5), and where the
   bin edges come from: fitted on the two rims, or the code fitted on the 120
   rims of the Fantastic Breaks retrieval pool (available after step 2 of the
   *Retrieval* tab). **Load default** restores the paper's values.
4. **Run DaTC.** The computation runs in a worker thread.

What is drawn:

| item | meaning |
|---|---|
| orange / teal faces | the fracture region of A / B |
| red / blue tube | rim A / rim B (the longest boundary loop of the region) |
| yellow on a tube | samples matched by the best local alignment |
| arrows | the Darboux frame at the chosen sample of rim A: tangent t (red), g = n x t in the surface (green), surface normal n (blue) |
| plot, top | alpha along rim A (red) and alpha of the aligned samples of rim B (blue; negated when B is walked in reverse, as the mate is) |
| plot, bottom | beta likewise; the shaded band is the aligned stretch of A |

The score is the Smith-Waterman score divided by n_A + n_B, the scale of
Table I. *Place B by the RANSAC fit* moves B by the rigid transform that RANSAC
(Fischler and Bolles 1981; 2000 minimal samples of three matched pairs, Kabsch
on each, inliers within 3 ds, refit on the largest consensus) finds from the
matched samples; of the two directions of travel, the one whose fracture planes
face each other is used. The fit uses only the matched samples, not the stored
pose, so for a true Fantastic Breaks pair (stored assembled) it should turn B
by close to 0 degrees. *Pull the fragments apart* moves B away from A along
the line between their centres. *Export* saves the 3D view (PNG), the result
(JSON: settings, score, symbol strings, matched sample pairs, pose, verdict)
and a GIF of B travelling from apart to the RANSAC pose.

The verdict (*Result* region) answers whether the two rims are mates. Its
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

## Retrieval tab

*Reproduce Table I* runs the scripts of `experiments/` as separate processes and
streams their output below the 3D view:

1. *Build the rim pools* (needs the two dataset folders)
2. *Upper block (reviewed settings)*
3. *Lower block (tuned alike)*
4. *Check against the paper*: prints Table I and ends with
   `TABLE I REPRODUCED` or `TABLE I MISMATCH`

*Rankings*: choose the pool and the arm, press *Load rankings*. Each row is a
query: its mate's rank (ties at their mid-rank) and the first candidate. Select
a row, and choose whether fragment 2 is the first candidate or the mate. The
pair is run through the same steps as the *Pair* tab (pool rims unchanged, 9 x 5
code fitted on the two rims, RANSAC, verdict) and drawn in six panels:

| | column 1 | column 2 | column 3 |
|---|---|---|---|
| row 1 | fragment 1 (query) and its rim | fragment 2 and its rim | fragment 2 placed by RANSAC, with the verdict |
| row 2 | rim 1 coloured by alpha bin, and its code from the black dot | rim 2 likewise | the Smith-Waterman matches: green within 3 ds after RANSAC, grey outside |

The rims of row 2 are drawn in the plane of rim 1 (its principal axes); rim 2
is first placed by RANSAC, so matched samples line up when the fit is good.
The meshes are read from the dataset folders under *Datasets*. The verdict and
its measured rates appear under the table.

## Terms

- **rim**: the boundary loop of a fragment's fracture region, resampled at a
  constant step ds.
- **alpha, beta**: the turn of the rim between two samples, split into the
  part inside the surface (along g) and the part out of it (along n).
- **mate**: the rim on the other side of the same fracture. It walks the
  shared curve the other way, so its alpha is negated and reversed.
- **expected rank**: a mate tied with t other candidates and beaten by g
  counts as top-k with probability min(1, max(0, (k - g)/(t + 1))).
