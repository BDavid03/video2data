# mesh_extents

Video-measured object proportions, and the closed primitive (ellipsoid or box) built from them, for
the Track 3 CD-O mesh file (D1).

## Latest: D1 v2

V2 uses the adopted **R2 vessel-handle** rule with `aspect_threshold=1.3` and
`handle_height_fraction=0.6`, chosen on all ten objects after cross-fitted adoption. It reuses
v1's exact 23 measure runs; vessel membership comes from `_rule_scores.CATEGORIES`, shared with
the rule loop. The spoon and the other non-vessels retain their boxes.

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_extents build --request scripts/track3/requests/d1/build-v2-r2.json
```

Build: `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/run.json`.
Fresh local CD-O: **0.9251 cm** per object (`outputs/mesh_eval/d1-ctx03/v2-cdo.json`), the
all-object fitted development score; the adoption headline remains **0.92572 cm cross-fitted**.
All ten scanned meshes match the loop's R2/1.3 candidates byte-for-byte. Legacy box and ellipsoid
builds retain identical vertex arrays, face arrays and PLY files over all 47 meshes each
(`outputs/mesh_eval/d1-ctx03/v2-equivalence.json`).

Packed: `submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet`, **294,912 rows**, identical
row order to the kit sample; preflight `--track track_3 --skip-kaggle`: **Ready to submit**
(warnings: no pose results, Kaggle checks skipped). The commit URL is still the requested
placeholder; SHA-256 and reproduction commands are in the adjacent `manifest.json` and
`docs/chain/SUBMISSION.md`.

## Why this slice exists

CD-O (kit `metric_code/track_3/CD-O.py`) scales each candidate by the ratio of surface RMS radii and
registers it rigidly to the scan before a symmetric Chamfer. Absolute scale and pose cost nothing;
proportions do. Context 01 measured, on the 10 public scans, that a primitive with the scan's own
principal extents scores 1.17 cm against 2.52 cm for a cube (analysis only). This slice measures those
proportions from the egocentric video instead, so no scan geometry enters a submitted mesh (the
challenge FAQ requires participants' own reconstructions).

## Operations

| Operation | Produces | GPU |
|---|---|---|
| `measure` | Per frame: support plane, table-relative clusters (label image, extents); per assigned object: extents per frame and their median; `overlay_<frame>.png` of the first frame | No |
| `build` | One primitive per object pooled over measure runs, and `meshes/episode_XXXXXX/<object>.ply` for every scored pair, ready for the kit's `pack_reconstruction.py track3-meshes` | No |

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python scripts/track3/d1_measure.py --out .runtime/tmp/d1-runs.json
PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_extents build --request <build.json>
```

Inputs: one completed `depth.generate` run per episode (MoGe2, ego_cam_a, opening frames 0-19).
No stereo anchor is used: a per-frame scale multiplies all three extents alike, so proportions do
not depend on it.

## Method

Per frame, in camera coordinates:

1. **Support plane.** RANSAC the largest plane in the lower 70 % of the image (the floor), then take
   the sharpest histogram peak of heights 0.2-1.2 m above it (the round table top) and refit there.
   Context 02 found that the camera_motion world plane can be the floor, which made the whole table
   one "object" on public episode 39.
2. **Clusters.** Points 8 mm-0.5 m above the table and within the table radius + 15 cm horizontally
   (objects overhang the rim), joined by 4-connected metric edges of at most 2 cm, at least 300 px.
3. **Local plane.** MoGe2 table tops are not flat: depth bleeds a few mm upwards around objects. Each
   cluster's plane is refitted on table pixels in a 40 px ring around it and the cluster re-cut.
4. **Extents.** In-plane PCA extents (2-98 % range) and height (98th percentile above the local
   plane): `[in_plane_long, in_plane_short, height]`.

An object's clusters are named once, on the first measured frame, in `assignments.json` (a list, so an
object split by depth gaps is reunited). Later frames take every cluster lying mostly inside the
object's first-frame mask dilated by 15 px. Only clean, isolated, fully visible objects are assigned;
merged objects (a pot on the rack) are left out of that episode. `build` divides each episode's
median by its geometric mean, takes the per-axis median over episodes and restores the median size.

## Known limits

- Undersides are never seen; height is the visible top above the table, so hollow objects become solid.
- The first-frame assignment is a recorded visual judgement by the chain context (see
  `assignments.json`), not a detector.
- Scale is MoGe2's, not metric. Simulation (D2) needs a metric rescale before these meshes are used.

## Evidence

`outputs/mesh_eval/d1-ctx02/` (local CD-O against the public scans, measurement only).

## Shape rules

Run all three rules and their parameter grids on CPU, or resume the same command:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/kit-env/bin/python -B -m track3.features.mesh_extents.rules --queue track3/features/mesh_extents/rules.json
```

The editable queue is `rules.json`. The handler reuses the v1 pooled video extents and exact
box files from `outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff/`, reproduces the
baseline, then imports `scripts/track3/local_cdo.py` for the kit's mesh budget and CD-O scorer
(seed episodes 1, 3, 5). Rule evaluation preserves v1's box files; `_geometry.py` and `measure.py`
remain unchanged.
No scans enter the shape builders in `_shapes.py`.

- **R1 vessel-cylinder:** round vessels become capped elliptical cylinders; other categories
  retain their v1 boxes. No free parameter.
- **R2 vessel-handle:** round vessels with measured long/short aspect above the threshold
  become circular cylinders plus a closed handle bar using the long-axis excess; other round
  vessels become elliptical cylinders. Threshold grid: 1.05, 1.15, 1.30. Handle height is fixed
  at 60% of measured height and width at 15% of the short extent. Tuning the threshold tests
  when a handle is appropriate while leaving just one free parameter.
- **R3 thin-tool-head:** thin tools become a box head plus narrow box handle, aligned along the
  measured in-plane long axis. Head length grid: 0.25, 0.35, 0.50 of that extent; handle width is
  fixed at 20% of the short extent. Both components retain measured height. The unscanned spoon
  shares this rule's category but is neither scored nor rebuilt by this evaluation.

Categories, folds and adoption gates follow `docs/chain/mesh-selection-protocol.md`. Each
parameter is chosen on fold A and scored on B, then chosen on B and scored on A; ties take the
first grid entry. The headline combines the ten held-out object scores. The in-sample column
chooses one parameter on all ten objects and is optimistic. Pair weighting applies the v1 roster
counts to the cross-fitted scores (26 scanned pairs). Adoption requires a gain greater than
0.03 cm, no object regression greater than 0.15 cm, and no pair-weighted regression.

Outputs under `outputs/mesh_eval/rules/`: `summary.md`, machine-readable `summary.json`,
`rules-cdo.png`, append-only `ledger.jsonl`, content-addressed meshes and scores, and per-object
trial `run.json` records. Complete trials resume by a hash of the rule, parameter, object,
video extents, candidate/reference bytes, protocol, code and scoring environment. Identical
mesh/reference pairs reuse their score across rules. Run records mark the evaluation as
`public_asset_diagnostic`; candidate shapes still derive solely from video. The kit environment
does the scoring; the existing `.runtime/linux-cuda` interpreter renders the matplotlib figure
on CPU without requiring another install.

Handle meshes concatenate overlapping, individually watertight components. They contain internal
faces; this run checks the kit budget and area-weighted CD-O sampling, not Boolean union or
physical simulation validity. No submission files are built or repacked by this command.

Results: development estimates against the ten public scans, with parameters chosen on one fold
and scored on the other. Box v1 is **1.00764 cm per object**, **1.25662 cm pair-weighted**.

| Rule | Chosen on A → scored B | Chosen on B → scored A | Cross-fitted mean (cm) | In-sample mean (cm) | Pair-weighted mean (cm) | Worst regression (cm) | Gain / object / pair gates | Adopt |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| R2 vessel-handle | 1.3 | 1.05 | 0.92572 | 0.92506 | 1.08742 | +0.10599 | pass / pass / pass | yes |
| R1 vessel-cylinder | none | none | 0.95644 | 0.95644 | 1.18422 | +0.00668 | pass / pass / pass | yes |
| R3 thin-tool-head | 0.5 | 0.25 | 0.99027 | 0.98365 | 1.24326 | +0.00000 | fail / pass / pass | no |

R2 ranks first: gain **0.08192 cm per object**, with its worst regression on water_pitcher
(1.57686 → 1.68285 cm, +0.10599 cm). R1 also qualifies, with a smaller gain of 0.05120 cm.
R3 improves both scanned tools but its held-out gain is only 0.01737 cm, so it fails the
0.03 cm requirement; this evidence does not support changing the unscanned spoon. No combined
rule was tried. The adoption column means eligible under the fixed protocol; packing remains
the lead context's decision.

The full CPU run produced 80 object trials (10 baseline + 70 rule/parameter/object combinations),
25 unique mesh scores and 55 exact mesh-score reuses. All ten baseline scores reproduce the
stored v1 JSON within its rounding. An independent artifact-based calculation agrees with all
fold selections, means and gates (`outputs/mesh_eval/rules/independent-recheck.json`).
Repeating the command resumed all 80 completed trials with zero new scores, left the ledger at
80 rows, and reproduced the ranking exactly (`outputs/mesh_eval/rules/resume-recheck.json`).
