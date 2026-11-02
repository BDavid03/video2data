# mesh_extents

Measure object proportions from cached egocentric depth, build closed procedural meshes, and
run frozen shape-rule or silhouette diagnostics. CD-O aligns each mesh with its own Sim(3),
so shape matters while absolute scale and placement do not. Meshes use video evidence only.
Monocular scale is unanchored; simulation needs separate metric calibration. Episode 41 and
holdout episodes 43/44 are excluded. The owner handles uploads.

## Operations

Each operation owns its request, runtime admission and run records. Commands below are CPU
commands; `measure` consumes completed depth and does not run MoGe2. Use the existing environments.

| Operation | Input and output |
| --- | --- |
| `measure` | Assigned clusters from a completed `depth.generate` run; labels, planes, overlays and `extents.json`. |
| `build` | Completed measure runs; pooled shapes, `objects.json`, and one PLY per object/pair. |
| `rules` | Frozen queue; resumable candidate builds, recorded scores and ranked summaries. `--dry-build` omits scoring. |
| `adopt` | Frozen rule winner; production admission, pack and matched packed-pair veto. |
| `proxy` | Frozen meshes, cached video measurements and existing CD-O scalars; IoU ledger and concordance report. |
| `proxy-insights` | Amendment-4 selected-yaw replay and descriptive within-object pair comparisons. |

`measure` request keys are `source_split`, `episode_index`, `depth_run`, `frames` and object-to-cluster
`objects`; `assignments.json` records reviewed identities. `<measure.json>` names such a request.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .runtime/linux-cuda/bin/python -B -m track3.features.mesh_extents measure --request <measure.json>
```

`build` requests name `measure_runs`, shape/rule and optional frozen attributes; `<build.json>` is
one request. Existing examples include `scripts/track3/requests/d1/build-v2-r2.json` and `build-v3.json`.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .runtime/linux-cuda/bin/python -B -m track3.features.mesh_extents build --request <build.json>
```

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/kit-env/bin/python -B -m track3.features.mesh_extents.rules --queue track3/features/mesh_extents/rules-round3.json
```

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/kit-env/bin/python -B -m track3.features.mesh_extents adopt --request track3/features/mesh_extents/adopt-round3.json
```

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.mesh_extents proxy --request track3/features/mesh_extents/proxy-spoon.json
```

Measurement finds the floor in the lower 70% of the image, then a table-height peak 0.200–1.200
above it. Above-table clusters use height 0.008–0.500, overhang 0.150, neighbour distance 0.020,
minimum 300 pixels and a 40-pixel local plane ring. Extents use in-plane PCA percentiles 2–98
and height percentile 98. Tracking dilates the assigned first-frame mask by 15 pixels. Pooling
normalizes each episode by geometric-mean size, takes median proportions and restores median size.
Undersides, cavity depths and thicknesses remain approximate. Closed overlapping components
can retain internal faces; closure is not a Boolean-union or simulation-validity claim.

Proxy placement uses the local plane, in-plane PCA, footprint-bounds centre and base on plane;
one uniform scale matches long footprint extent. x/y are long/short footprint axes, z is up;
the R8 sweeper's longitudinal axis is z. Require x >= y*(1-1e-6). Evaluate yaw 0/pi, retaining
the larger IoU, and record both yaws and projected/target areas. Each assigned run supplies its
first frame: camera a, frame 0 except lid frame 110. Spoon views are evaluation 5/15/16.
The panel has 52 scanned candidates, 26 scanned views/143 trials, plus two spoon meshes over
three views/six trials: 149 trials total. Primitive-era spoon box/ellipsoid are references only.

`_shapes.py` and `_rule_scores.py` own construction/selection; `_geometry.py` owns shared pure
table selection. `_proxy.py` owns array preparation, projection, averaging, correlations and report formatting;
`proxy.py` owns files and records. `measure.py` is unchanged. No scan or scorer is opened by proxy.
The frozen request pins input/source hashes and package versions. Its baseline pointers are
request data; baseline run/input hashes authenticate the saved original request without Git.
Commit the protocol/request/code before first scoring as required by the protocol; the lead
owns that ordering. `proxy_authorized` defaults true. `--dry-run` writes target-only outputs
under `outputs/mesh_extents/spoon-proxy-dry/summary.{json,md}` and `overlays/` without scores.
The 149-trial/7200-second budget is per invocation. Resume filters by frozen identity and
preserves older records. Only a truncated final ledger append is quarantined; other malformed
rows raise. Missing inputs produce NOT VALIDATED; unavailable values are null with reasons.
Re-freezing refreshes source hashes only; the authenticated baseline fixes the admitted panel.
Adding a slice operation changes `__main__.py`/`runtime.json`, so affected frozen requests must be re-frozen and re-proved.
Its E scores were admitted by path/status/seeds/build SHA and A/B/C by seeds 1/3/5; E's legacy
score JSON has no SHA itself. Matching object/SHA aliases were merged only for equal scores.

## Latest results

### D1 v4: adopted R9

R9 combines the frozen R7 rack and R8 tools; R8 independently passed on the scanned objects.
This prevents rack gains from masking scanned-tool failure; it does not validate the spoon.

| Recorded comparison | Equal-object CD-O (cm) | Pair-weighted CD-O (cm) |
| --- | ---: | ---: |
| Packed v3 | 0.771 | 0.811 |
| Packed v4 | 0.653 | 0.663 |
| R7 development | 0.720 | 0.715 |
| R8 development | 0.706 | 0.765 |
| R9 development | 0.653 | 0.663 |

Coverage is 10 scanned objects / 26 of 36 pairs; ten spoon pairs are unscored. Packed results:
9 improved, 17 tied, 0 worse; worst regression 0.000. Gates use the frozen full-precision
ledger/queue values, not rounded table entries: object gain >0.030, no regression >0.150,
packed object mean <v3−0.030 and pair mean no worse. Third adaptive round on reused scans:
these are development results, not an independent estimate. R9 seed SD mean/max: 0.005/0.021 cm.
First new score finished 2026-11-02 05:06:45 UTC after the pre-score commit `bc601cf`.

Packed file `submissions/20261102T0516Z-d1-v4/track3_meshes_v4.parquet`: 294,912 rows, official
order, unique IDs, preflight Ready to submit with no poses and Kaggle checks skipped; manifest
hashes 6/6. SHA-256: `6663a1c7e0491c0cfa53b27fe36d7904455351dad548181e4e59a2263215af50`.
Fresh v3/v4 reproduction each matches 47/47 PLYs, including 36 pair files; v4 packed bytes
match its reproduction. Full evidence: `outputs/mesh_extents/round3-job-b/job-b.md`,
`outputs/mesh_extents/rules-round3/summary.md`, `outputs/mesh_eval/pairs/mesh-round3-veto/summary.md`.

### Spoon proxy: NOT VALIDATED

| Pre-registered gate | Value | Requirement | Result |
| --- | ---: | ---: | --- |
| Mean tau-b (10 objects) | 0.544 | >=0.400 | pass |
| Qualifying median rho | 0.655 | >=0.600 | pass |
| Qualifying objects passing rho | 5/8 | >=6 | fail |
| Both scanned tools | 1/2 | all | fail |
| Dust pan rho | 0.600 | >=0.600 | pass |
| Sweeper rho | -0.086 | >=0.600 | fail |
| Correlations defined / inputs complete | all / all | all / all | pass |

| Object | Candidates | Rho | Tau-b | IoU ties |
| --- | ---: | ---: | ---: | ---: |
| beige_cup | 7 | 0.709 | 0.620 | 3 |
| blue_cup | 6 | 0.845 | 0.775 | 3 |
| mini_dust_pan | 6 | 0.600 | 0.467 | 0 |
| mini_sweeper | 6 | -0.086 | -0.200 | 0 |
| plastic_dish_rack | 4 | 1.000 | 1.000 | 0 |
| water_pitcher | 7 | -0.059 | 0.056 | 3 |
| white_pot | 8 | 0.862 | 0.725 | 3 |
| white_pot_lid | 4 | -0.200 | 0.000 | 0 |
| wooden_piece_1 | 2 | 1.000 | 1.000 | 0 |
| wooden_piece_2 | 2 | 1.000 | 1.000 | 0 |

IoU ties count variants beyond distinct values; CD-O ties are zero. Report-only R8-minus-box
IoU is +0.143 for dust pan and +0.186 for sweeper; R8 ranks above box for both.

| Spoon evaluation view, frame 0 | v3 box IoU | v4 bowl IoU | Difference |
| --- | ---: | ---: | ---: |
| 5 | 0.365 | 0.724 | 0.360 |
| 15 | 0.556 | 0.702 | 0.147 |
| 16 | 0.542 | 0.725 | 0.183 |
| Equal-view mean difference | | | 0.230 |

**Proxy not validated for spoon CD-O inference; proxy says nothing about spoon CD-O; v4 unchanged.**
Full-precision values, both yaws, mask/projected areas and overlays: `outputs/mesh_extents/spoon-proxy/summary.json`.
The ledger is `outputs/mesh_extents/spoon-proxy/ledger.jsonl`.

**Post-result reporting, not another gate:** mean tau-b over the eight objects with >=4 variants
is 0.430, beside the pre-registered ten-object mean 0.544. Both two-variant wooden blocks
contribute +1.000; two untied variants force tau-b to ±1. Dust pan passes at exactly rho 0.600.
**Hypothesis:** silhouettes may miss walls or brush depth; this explanation has not been
separated from mask and placement error. Construction views were reused and samples are small.

### Descriptive proxy failure analysis (Amendment 4)

Replays the ledger placement and selected yaw directly; all 135 IoUs and projected/target areas match exactly.
The failed-group signature is **not separated**. At both 0.03/0.06 cm cutoffs only the lid retains
outline-score disagreement; sweeper and both cups are gap-sensitive. Failed-minus-contrast B is -0.167
(descriptive flag true); the T difference/flag is undefined because the lid has one view.
All signatures remain conditional on recorded views, normalization and selected yaw; hidden geometry
versus mask/placement error remains unproven. Amendment 3 remains NOT VALIDATED; packed meshes unchanged.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.mesh_extents proxy-insights --request track3/features/mesh_extents/proxy-insights.json
```

Writes `outputs/mesh_extents/proxy-insights/<run-id>/{run.json,summary.json,summary.md,pairs.json,pairs.md,replays/}`,
plus stable `summary.{json,md}` and an identity-resumable `replay-ledger.jsonl` in that output root.

## History

### Rounds 1–2 and earlier engineering

| Round / rule | Object mean (cm) | In-sample (cm) | Pair mean (cm) | Worst regression (cm) | Selection |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 box | 1.008 | 1.008 | 1.257 | — | baseline |
| R1 cylinder | 0.956 | 0.956 | 1.184 | 0.007 | fixed, eligible |
| R2 handle | 0.926 | 0.925 | 1.087 | 0.106 | folds 1.300 / 1.050; adopted |
| R3 head | 0.990 | 0.984 | 1.243 | 0.000 | folds 0.500 / 0.250; failed gain |
| R4 rim handle | 0.921 | 0.921 | 1.077 | 0.000 | fixed, failed gain |
| R5 hollow | 0.846 | 0.846 | 0.956 | 0.000 | thickness 0.100, eligible |
| R6 peg rack | 0.773 | 0.773 | 0.817 | 0.000 | thickness 0.100, adopted as v3 |

R2 improved object mean by 0.082; pitcher regressed 1.577→1.683 (+0.106). R1 gain was 0.051;
R3 gain 0.017 failed >0.030. R2 aspect grid: 1.050/1.150/1.300, handle height/width 0.600/0.150.
R3 head grid: 0.250/0.350/0.500, handle width 0.200. Round 1: 80 trials, 25 unique scores,
55 reuses; resume added zero. V2 reused 23 measure runs and the non-vessel/spoon boxes.
Round-2 incumbent fixed R2/1.300: 0.925 object / 1.088 pair; object gate <0.895, worst <=0.150.
R4 rim handle height is 0.150; R5 wall/floor grid 0.020/0.050/0.100, 64 sections; R6 rack has
two rows of six pegs, base height 0.250, peg length 0.500 of spacing, width 0.300, overlap 0.010.
R6/R5/R4 seed-SD mean/max: 0.007/0.021, 0.009/0.036, 0.010/0.036 cm.
Round 2: 70 scores, 23 unique, 47 reuses; packed v2→v3 object 0.926→0.771, pair 1.085→0.811;
19 improved, seven tied, zero worse. Dry: 77/77 candidates, 24 distinct meshes, 14 renders,
266 vertices/524 faces maximum; 33 legacy constructors and 47 production PLYs unchanged.
Construction error <=2.22e-16, reloaded <=5.30e-8; all dry/scored resumes added zero work.
Portability checks reproduced 47/47 v2/v3 PLYs, all 11 pooled extents, 80/70 cached rule rows
and all 52 packed-pair values. Ten scanned meshes matched selected candidates; spoon matched v2.
Temporary v3 pack SHA: `2ea5bb2da753dc4c324c1a327ed49b0f29348e9948fea485461720be8e8bf7b6`;
shipped v3 SHA: `a61f02b12078a05c940037034e37ec4201e3e919a55e4bea9e0058dfa2f45f40`.
An earlier oracle analysis scored 1.170 versus cube 2.520 cm; oracle proportions never enter builds.

### Round 3 and proxy engineering

R7 changes rack, R8 changes pan/sweeper/spoon, R9 composes both, with no fitted parameter.
Video profiles retain rack floor 1/4, compartment length 155/900, rim top 7/16, peg top 5/8,
peg footprint 40/900 by 40/230 and wall 10/230; rack height uncertainty is ±0.100.
Pan head/neck/grip/start: 0.700/0.250/0.550/0.650; floor/grip/wall: 6/42, 12/42, 9/280.
Sweeper brush/neck/grip/start/depth: 0.500/0.750/0.700/0.450/0.650; upright longitudinal axis.
Spoon head/neck/grip/start: 1/3, 0.240, 0.220, 0.300; inner radius/floor/thickness: 0.900/0.250/0.500;
bowl uses 48 angular sections and six radial intervals. Markers and holes are omitted.
Round-3 dry checks: 110 legacy comparisons, 11/11 old meshes, ten evidence overlays unchanged;
33/33 candidates, 15 distinct meshes, four renders/12 views and four video comparisons; max
634 vertices/1260 faces, 141 PLY hashes checked, 41/30/24 R7/R8/R9 files unchanged from v3.
Both construction/reload errors remain <=2.22e-16 / 5.30e-8. Development: 30 trials, 13 new
scores, 17 reuses; later resume reused all 30 and 52 packed-pair trials with zero new work.
Incumbent object/pair 0.773/0.817; frozen object gate <0.743 (a prior transcription differed by 5e-7 cm).
Buried rack surface v3/R7: 10.326%/10.537%, SE 0.015/0.013 percentage points; coincident boundary
0.307%/9.948%, areas 0.815/0.836 unanchored squared units; three replicates of 20,000 samples/component.
These geometry-only diagnostics use axis-aligned containment and never select a rule.

Post-result proxy fixes provide null/reason failure reports and strict score admission, share
table selection, and keep numerical averaging pure. Old/new geometry matches exactly over
92 recorded frames / 18 runs / 30 assigned objects. Recomputations match 149/149 first-run
IoUs, both yaws, areas, placements, PNGs, every rho/tau/gate and spoon difference; second
invocations add zero trials. Original run/trial/cache artifacts and ledger prefixes are retained.
Prior engineering timings were 6.844 seconds recomputation / 1.568 seconds resume; 602 original
artifacts stayed unchanged. Evidence scripts: `.runtime/tmp/spoon-proxy-engineering/`.

Critic cleanup removes proxy's Git dependency and tightens ledger recovery. Audited reruns took
9.862/3.385 seconds with zero subprocess or `.git` file-access attempts; resume added zero trials.
All 149 values still match the first run and 1,218 historical artifacts are unchanged. Current proof:
`outputs/mesh_extents/spoon-proxy/critic/result-comparison.json` and `critic/portability.json`.

### Evidence index

All prior evidence paths remain below; detailed values and command inventories stay in their
recorded JSON/Markdown artifacts. Earlier shorthand paths are retained with their original bases.
- `../../../outputs/mesh_extents/evaluation/episode_000001/20261101T103716.335071Z-0993ba31/overlay_000110.png`; `../../../outputs/mesh_extents/evaluation/episode_000005/20261101T103721.877826Z-55c95204/overlay_000000.png`; `../../../outputs/mesh_extents/evaluation/episode_000015/20261101T103736.789367Z-86a79be7/overlay_000000.png`
- `../../../outputs/mesh_extents/evaluation/episode_000016/20261101T103740.185103Z-0f0a559d/overlay_000000.png`; `../../../outputs/mesh_extents/public/episode_000011/20261101T103701.928174Z-f63cc238/overlay_000000.png`; `../../../outputs/mesh_extents/public/episode_000012/20261101T103703.949004Z-546c56c2/overlay_000000.png`
- `../../../outputs/mesh_extents/public/episode_000018/20261101T103706.195611Z-8e4c4ca5/overlay_000000.png`; `../../../outputs/mesh_extents/public/episode_000021/20261101T103707.151959Z-05bf6bac/overlay_000000.png`; `../../../outputs/mesh_extents/public/episode_000023/20261101T103708.924989Z-52ed3be1/overlay_000000.png`
- `../../../outputs/mesh_extents/public/episode_000042/20261101T103715.090424Z-d410dfac/overlay_000000.png`; `.runtime/tmp/d1-runs.json`; `.runtime/tmp/mesh-round2/dry-audit.json`
- `.runtime/tmp/mesh-round2/pack/track3_meshes_v3.parquet`; `.runtime/tmp/mesh-round2/round1-equivalence.json`; `.runtime/tmp/mesh-round2/v2-build-equivalence.json`
- `docs/chain/SUBMISSION.md`; `docs/chain/mesh-selection-protocol.md`; `dry/summary.json`
- `meshes/episode_XXXXXX/<object>.ply`; `metric_code/track_3/CD-O.py`; `outputs/mesh_eval/d1-ctx02/`
- `outputs/mesh_eval/d1-ctx03/v2-cdo.json`; `outputs/mesh_eval/d1-ctx03/v2-equivalence.json`; `outputs/mesh_eval/d1-v3-reproduction/`
- `outputs/mesh_eval/equivalence/20261102T055518.792315Z-eb3c237a/summary.md`; `outputs/mesh_eval/mesh-fix/`; `outputs/mesh_eval/pairs/shipped-v3/`
- `outputs/mesh_eval/rules-round2/dry/ledger.jsonl`; `outputs/mesh_eval/rules-round2/dry/summary.md`; `outputs/mesh_eval/rules-round2/job-a.md`
- `outputs/mesh_eval/rules-round2/job-b.md`; `outputs/mesh_eval/rules/independent-recheck.json`; `outputs/mesh_eval/rules/resume-recheck.json`
- `outputs/mesh_extents/adopt-round3/development/summary.md`; `outputs/mesh_extents/adopt-round3/runs/<run>/result.json`; `outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff/`
- `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/run.json`; `outputs/mesh_extents/build/20261102T024758.526008Z-486c5f5d/`; `outputs/mesh_extents/build/20261102T030138.153550Z-118eeb04/run.json`
- `outputs/mesh_extents/maintenance/20261102T055350Z/verification.json`; `outputs/mesh_extents/reproduction/20261102T052748Z/v4/manifest.json`; `outputs/mesh_extents/round3-diagnostic/20261102T044817.813770Z-132cdab2/summary.md`
- `outputs/mesh_extents/round3-job-a/`; `outputs/mesh_extents/round3-job-b/reproduction-checks.json`; `outputs/mesh_extents/round3-job-b/reproduction-v3/`
- `outputs/mesh_extents/rules-round3/dry/summary.md`; `outputs/mesh_extents/rules-round3/job-b/pair-request.json`; `outputs/mesh_extents/spoon-proxy/engineering/result-comparison.json`
- `reproduction-v4/`; `round2/summary.md`; `scripts/reproduce/d1_meshes.sh`
- `scripts/reproduce/verify.sh`; `scripts/track3/d1_measure.py`; `scripts/track3/local_cdo.py`
- `scripts/track3/requests/d1/build-v3.json`; `submissions/*-d1-v3/manifest.json`; `submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet`
- `submissions/20261102T0516Z-d1-v4/manifest.json`; `track3/features/mesh_extents/rules-round2.json`; `track3/features/mesh_extents/rules.json`

Additional indexed evidence: `outputs/mesh_eval/rules-round2/job-a-files.txt`; `outputs/mesh_extents/maintenance/20261102T055350Z/round2/summary.md`; `outputs/mesh_extents/round3-job-b/reproduction-v4/`.
