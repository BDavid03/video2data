# mesh_extents

Video-measured object proportions, and the closed primitive (ellipsoid or box) built from them, for
the Track 3 CD-O mesh file (D1).

## Latest result (2026-11-02): R6/0.10 passes the exact veto

**Adopt R6 peg-rack with wall/radius fraction 0.10.** The frozen round-2 loop selects 0.10 on
both folds and on all ten scanned objects. It passes every development gate and the exact packed
v2 veto. The amendment, attributes, grids and constructors remain identical to commit `c7e2b56`.
No retuning followed the veto. These are development estimates on ten reused public scans;
choosing among three nested rules is optimistic. The spoon has no public scan and is unchanged.

| Rule | Chosen on A → scored B | Chosen on B → scored A | Cross-fitted mean (cm) | In-sample mean (cm) | Pair-weighted mean (cm) | Worst regression (cm) | Gain / object / pair gates | Development eligible (veto pending) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| R6 peg-rack | 0.1 | 0.1 | 0.77315 | 0.77315 | 0.81675 | +0.00000 | pass / pass / pass | yes |
| R5 hollow-vessels | 0.1 | 0.1 | 0.84555 | 0.84555 | 0.95600 | +0.00000 | pass / pass / pass | yes |
| R4 rim-handle | — | None | 0.92061 | 0.92061 | 1.07740 | +0.00000 | fail / pass / pass | no |

R4 has no free parameter (`—` / `None` in the unedited loop table). Seed SD is the population
standard deviation over kit episode seeds 1/3/5 for each held-out object; this is not a confidence
interval for independent performance.

| Rule | Mean object seed SD (cm) | Maximum object seed SD (cm) |
| --- | ---: | ---: |
| R6 peg-rack | 0.006887335425561346 | 0.02136714096735155 |
| R5 hollow-vessels | 0.008641243641626373 | 0.03585776336634344 |
| R4 rim-handle | 0.009542621334146224 | 0.03585776336634344 |

| Packed file | Pair-weighted CD-O (cm) | Equal-object CD-O (cm) |
| --- | ---: | ---: |
| v2 | 1.085023637656605 | 0.9257250385709256 |
| v3 | 0.8114951342417491 | 0.7710770332786878 |

Exact evaluation covers 26/36 pairs: **19 improve, 7 are unchanged, 0 worsen**. Each pair uses its
own kit episode/slot seed through the unmodified `mesh_eval.pairs` handler and scorer. The ten
spoon pairs remain unscored. Both veto means are better than packed v2; entry-point/direct
episode-1 equivalence error is 0.0 cm.

The scoring loop produced 70 parameter/object records, 23 unique scores and 47 identical-mesh
score reuses. A repeat resumed all 70 records with zero new scores or ledger rows. Ranked table,
per-object figure, seed spreads and full-precision independent gate recomputation live under
`outputs/mesh_eval/rules-round2/`; exact pair results are in
`outputs/mesh_eval/pairs/mesh-round2-v2-v3/`.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .runtime/kit-env/bin/python -B .runtime/tmp/mesh-round2/score_round2.py --queue track3/features/mesh_extents/rules-round2.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B -m track3.features.mesh_extents build --request scripts/track3/requests/d1/build-v3.json
bash scripts/reproduce/d1_meshes.sh --version v3
```

The scratch launcher redirects caches into this job's permitted scratch path and calls the
unmodified loop. The ordinary module entry point remains available with the same queue.
The reproduction command prints a CPU plan; add an explicit commit URL and `--run` to execute.
Default v2 reproduction has a byte-identical plan and all 47 fresh v2 mesh hashes remain identical.
For v3, the script uses a submitted v3 manifest if present, otherwise this job's local pack manifest;
`--manifest <path>` selects an explicit recorded manifest.

V3 build: `outputs/mesh_extents/build/20261102T030138.153550Z-118eeb04/run.json`.
All 11 objects and 36 pair copies were built. All ten scored meshes match selected loop candidates
byte-for-byte; the spoon matches v2. Local placeholder pack:
`.runtime/tmp/mesh-round2/pack/track3_meshes_v3.parquet`, **294,912 rows**, SHA-256
`2ea5bb2da753dc4c324c1a327ed49b0f29348e9948fea485461720be8e8bf7b6`. Kit preflight `--track track_3 --skip-kaggle`: **Ready to submit**; warnings
are no policy results and skipped Kaggle checks. No `submissions/` files were written; final
packing and release remain the lead's. See `outputs/mesh_eval/rules-round2/job-b.md` for commands
and full-precision gate numbers.

## Round 2 geometry preparation (Job A)

Job A appends Amendment 1 to `docs/chain/mesh-selection-protocol.md`, freezes `attributes.json`
and `rules-round2.json`, and adds R4 rim handles, R5 vessel shells and R6 rack pegs. No new CD-O
was computed and no scan file was opened, loaded or rendered. `build.py` and the round-1 queue
were unchanged in Job A. The lead committed the amendment and attributes as `c7e2b56` before Job B scoring.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.mesh_extents.rules --queue track3/features/mesh_extents/rules-round2.json --dry-build
```

The dry loop reads all 11 video-measured extents from the recorded v2 build, constructs every
candidate, checks every connected component for watertightness and positive volume, checks both
budgets (<4096) and extents (relative tolerance 1e-6), and writes content-addressed PLY files and
candidate-only three-view PNGs. No scorer is imported. Identities include code, protocol, attributes,
incumbent ledger/scores, queue and extents. Completed trials resume; records append to
`outputs/mesh_eval/rules-round2/dry/ledger.jsonl`, with the active 77 results in
`dry/summary.json` and `dry/summary.md`. Earlier code revisions remain in the append-only ledger.
Queue budgets are 70 scored object trials (job B), 77 dry builds, and 7200 seconds.

| Object | Frozen attributes | Frame evidence (ego_cam_a) | Confidence |
| --- | --- | --- | --- |
| beige_cup | handle="none", open_top=true | public 11, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000011/20261101T103701.928174Z-f63cc238/overlay_000000.png)); public 11, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000011/20261101T103701.928174Z-f63cc238/overlay_000000.png)) | high |
| blue_cup | handle="none", open_top=true | public 11, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000011/20261101T103701.928174Z-f63cc238/overlay_000000.png)); public 11, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000011/20261101T103701.928174Z-f63cc238/overlay_000000.png)) | high |
| white_pot | handle="rim_flat", open_top=true | evaluation 5, frame 0 ([overlay](../../../outputs/mesh_extents/evaluation/episode_000005/20261101T103721.877826Z-55c95204/overlay_000000.png)); public 42, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000042/20261101T103715.090424Z-d410dfac/overlay_000000.png)); evaluation 1, frame 110 ([overlay](../../../outputs/mesh_extents/evaluation/episode_000001/20261101T103716.335071Z-0993ba31/overlay_000110.png)) | high |
| water_pitcher | handle="loop", open_top=true | public 21, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000021/20261101T103707.151959Z-05bf6bac/overlay_000000.png)); public 21, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000021/20261101T103707.151959Z-05bf6bac/overlay_000000.png)) | high |
| white_pot_lid | handle="none", open_top=false | evaluation 1, frame 110 ([overlay](../../../outputs/mesh_extents/evaluation/episode_000001/20261101T103716.335071Z-0993ba31/overlay_000110.png)) | high |
| plastic_dish_rack | pegs={"rows": 2, "count_per_row": 6} | public 12, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000012/20261101T103703.949004Z-546c56c2/overlay_000000.png)); evaluation 15, frame 0 ([overlay](../../../outputs/mesh_extents/evaluation/episode_000015/20261101T103736.789367Z-86a79be7/overlay_000000.png)); evaluation 16, frame 0 ([overlay](../../../outputs/mesh_extents/evaluation/episode_000016/20261101T103740.185103Z-0f0a559d/overlay_000000.png)) | high |
| wooden_piece_1 | unchanged R2 | public 23, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000023/20261101T103708.924989Z-52ed3be1/overlay_000000.png)) | high |
| wooden_piece_2 | unchanged R2 | public 23, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000023/20261101T103708.924989Z-52ed3be1/overlay_000000.png)) | high |
| mini_dust_pan | unchanged R2 | public 18, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000018/20261101T103706.195611Z-8e4c4ca5/overlay_000000.png)) | high |
| mini_sweeper | unchanged R2 | public 18, frame 0 ([overlay](../../../outputs/mesh_extents/public/episode_000018/20261101T103706.195611Z-8e4c4ca5/overlay_000000.png)) | high |
| wooden_spoon | unchanged R2 | evaluation 5, frame 0 ([overlay](../../../outputs/mesh_extents/evaluation/episode_000005/20261101T103721.877826Z-55c95204/overlay_000000.png)) | high |

The observations and exact overlay SHA-256 values live beside each attribute in `attributes.json`.
The pot is visibly open in evaluation 5 and public 42 opening frames; usual lid-on frequency is
not established. The lid is solid. Both cups remain without handles because aspect is below 1.3.
The pitcher handle is vertical and retains its R2 bar. Public 12 partially occludes one rack tine;
unobstructed evaluation 15 and 16 each show two rows of six. Board identity remains arbitrary.
Wall/floor thickness, undersides and internal construction were not measured from video. The
rack rule deliberately omits its visible utensil compartment and rim. Unknown or uncertain
required attributes preserve the entire R2 shape, rather than applying part of a nested rule.

Geometry conventions are frozen in the amendment: 64-section shells, wall/floor thickness equal
fraction x short-body radius with grid `[0.02, 0.05, 0.10]`; rim handle height 0.15 x vessel height;
rack base height 0.25 x height, pegs 0.5 x long-axis spacing by 0.3 x total short extent, two rows
with half-peg-width end margins, overlapping the base by 0.01 x height. Vessel shell bodies are
single connected surfaces. Handles and rack parts are overlapping closed components, with internal
faces; the check does not claim a Boolean union or simulation validity. `_shapes.py` and
`_rule_scores.py` contain pure construction/selection, while `rules.py` handles files and records.

Incumbent R2/1.3 recomputed from the existing three-seed ledger: **0.9250628448 cm per object**,
**1.0876784204 cm pair-weighted**. Development gates require cross-fitted mean **<0.8950628448**,
no object regression >0.15 cm and no pair-weighted regression. All three nested rules are candidates;
the lowest qualifying cross-fitted mean wins. Choosing among them on ten reused scans is optimistic.
One final all-object parameter is fixed before the production build/pack; the exact packed-v2 veto
requires both means no worse than **1.08502364 cm pair-weighted / 0.92572504 cm per object**.
No retuning or rule switching follows that veto. Production changes and scoring remain job B.

Checks completed on CPU (no test suites):

- `py_compile` and CPU imports passed for all three edited Python modules.
- All three round-1 `select` outputs match the saved implementation exactly on the 80-row ledger,
  including thresholds, gates, per-object scores and aggregates; all 33 legacy constructor PLY
  comparisons are byte-identical. Evidence: `.runtime/tmp/mesh-round2/round1-equivalence.json`.
- Fresh unchanged production build `outputs/mesh_extents/build/20261102T024758.526008Z-486c5f5d/`
  matches **all 47 PLY SHA-256 values** (11 objects plus 36 pair copies). Evidence:
  `.runtime/tmp/mesh-round2/v2-build-equivalence.json`.
- Dry build: **77/77 candidates**, **24 distinct meshes**, **14 distinct three-view renders**;
  every component watertight and positive volume, largest mesh **266 vertices / 524 faces**.
  Maximum construction extent relative error **2.22e-16**; after PLY serialization/reload **5.30e-8**.
  Repeating the final loop resumes **77/77**, adds **0** trials and produces identical results.
  A Python audit hook rejects scan opens and scoring imports: **0 scan-open attempts**, scorer absent.
  Evidence: `.runtime/tmp/mesh-round2/dry-audit.json`; active per-object table and render links:
  `outputs/mesh_eval/rules-round2/dry/summary.md`.
- At the end of Job A, seed spread was unmeasured. Job B results above now name the incumbent
  and report the cross-fitted per-object seed SD, its mean and maximum; ties use frozen grid order.

Full handoff, amendment text, commands and file inventory: `outputs/mesh_eval/rules-round2/job-a.md`
and `job-a-files.txt`. Job B scoring results and commands are recorded above.

## D1 v2 (reproduction baseline)

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

- Undersides and wall/floor thickness are not measured. V3 shells use the frozen thickness fraction; visible geometry remains approximate.
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
