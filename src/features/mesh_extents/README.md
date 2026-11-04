# Mesh extents

Measures object proportions from cached video depth and builds closed procedural meshes. CD-O aligns each
mesh with its own Sim(3), so shape matters while absolute scale and placement do not. Meshes use video
evidence only, but the rules were selected on public scans. Monocular scale is unanchored, so simulation
needs separate metric calibration. Episodes 41, 43 and 44 are excluded.

## Operations

All are CPU. `measure` reads a completed depth run and does not run MoGe2.

| Operation | Input and output |
| --- | --- |
| `measure` | Assigned clusters from a `depth.generate` run; labels, planes, overlays, `extents.json` |
| `build` | Measure runs; pooled shapes, `objects.json`, one PLY per object and pair |
| `rules` | Frozen queue; candidate builds, scores, ranked summaries. `--dry-build` skips scoring |
| `adopt` | Frozen winner; admission, pack, matched packed-pair check |
| `proxy` | Frozen meshes and video measurements; IoU ledger and concordance report |
| `proxy-insights` | Replay of selected yaw and within-object pair comparisons |

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B src/features/mesh_extents/measure_loop.py --out data/.runtime/tmp/d1-runs.json
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents measure --request <measure.json>
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents build --request <build.json>
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents.rules --queue src/features/mesh_extents/rules-round3.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents adopt --request src/features/mesh_extents/adopt-round3.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents proxy --request src/features/mesh_extents/proxy-spoon.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents proxy-insights --request src/features/mesh_extents/proxy-insights.json
```

`measure` requests name `source_split`, `episode_index`, `depth_run`, `frames` and object-to-cluster
`objects`; `assignments.json` records reviewed identities. `build` requests name `measure_runs`, the rule and any
frozen attributes; examples are `requests/build-v2-r2.json` and `requests/build-v3.json`.

The proxy and proxy-insights commands refuse this checkout, because their frozen requests pin older
`adopt.py`, `build.py`, `rules.py` and protocol hashes. Reproduce them at their recorded sources; new work needs a
new protocol. Rule queues hash the scorer and `rules.py`, so a new campaign needs a new output directory.

Measurement finds the floor in the lower 70% of the image, then a table-height peak 0.200 to 1.200 above
it. Extents use in-plane PCA percentiles 2 to 98 and height percentile 98. Pooling normalises each episode by
geometric-mean size, takes median proportions and restores median size. Undersides, cavity depths and thicknesses
stay approximate. Closed overlapping parts can keep internal faces, so closure does not imply a Boolean union or
simulation validity.

`_shapes.py` and `_rule_scores.py` own construction and selection. `_union.py` is the R11 rack wrapper. `_geometry.py`
is shared table selection. `_proxy.py` is pure proxy maths and `proxy.py` owns its files and records.

## Results

D1 v4 uses rule R9, which is the R7 compartment rack plus R8 thin-tool profiles for dust pan, sweeper and spoon.

| Comparison | Equal-object CD-O (cm) | Pair-weighted CD-O (cm) |
| --- | ---: | ---: |
| Packed v3 | 0.771 | 0.811 |
| Packed v4 | 0.653 | 0.663 |
| R7 development | 0.720 | 0.715 |
| R8 development | 0.706 | 0.765 |
| R9 development | 0.653 | 0.663 |

Coverage is 10 scanned objects and 26 of 36 pairs; ten spoon pairs are unscored. Packed: 9 improved, 17 tied, none
worse. This is a third adaptive round on reused scans, so these are development results and not an independent
estimate. R9 does not validate the spoon. File: `submission/20261102T0516Z-d1-v4/track3_meshes_v4.parquet`, 294,912
rows, SHA-256 `6663a1c7e0491c0cfa53b27fe36d7904455351dad548181e4e59a2263215af50`. Record:
`data/outputs/mesh_extents/round3-job-b/job-b.md`.

Earlier rounds (object mean, pair mean, in cm):

| Rule | Object | Pair | Selection |
| --- | ---: | ---: | --- |
| v1 box | 1.008 | 1.257 | Baseline |
| R1 cylinder | 0.956 | 1.184 | Eligible |
| R2 handle | 0.926 | 1.087 | Adopted, pitcher regressed 1.577 to 1.683 |
| R3 head | 0.990 | 1.243 | Failed gain |
| R4 rim handle | 0.921 | 1.077 | Failed gain |
| R5 hollow | 0.846 | 0.956 | Eligible |
| R6 peg rack | 0.773 | 0.817 | Adopted as v3 |

Rule parameters are in `_shapes.py` and the queues `rules-round2.json` and `rules-round3.json`.

## Round 4: no change

`rules-round4.json` admits only C11, which is v4 plus R11 (the rack's 21 boxes replaced by their exact exterior
boundary, 157 vertices, 310 faces). R10 pooled vessel profiles failed a video-only test on tuning 42 (subset knots
45.7% of radius apart, limit under 5%). R11 scored worse in development (rack 0.860 to 0.904 cm, equal-object 0.6571 against 0.6527 cm), so v4 stayed. See
[the submission ledger](../../../docs/chain/SUBMISSION.md).

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents.rules --queue src/features/mesh_extents/rules-round4.json --dry-build
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents.rules --queue src/features/mesh_extents/rules-round4.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents adopt --request src/features/mesh_extents/adopt-round4.json
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents.profiles_diagnostic --output data/outputs/mesh_extents/round4-profiles-rerun
```

The last is the R10 test. This checkout refuses to resume or `--verify` its recorded outputs; verify at the recorded
commit and use a new output directory for a new run.

v4 geometry check (plan, then the same build, then compare all 47 PLY hashes):

```bash
bash scripts/reproduce/d1_meshes.sh https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/0000000000000000000000000000000000000000 --version v4
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents build --request src/features/mesh_extents/requests/build-v4.json
```

## Spoon proxy: not validated

The spoon has no public scan, so we tested whether silhouette IoU could stand in for CD-O. It failed, and it says
nothing about spoon CD-O. v4 is unchanged.

| Pre-registered gate | Value | Requirement | Result |
| --- | ---: | ---: | --- |
| Mean tau-b (10 objects) | 0.544 | at least 0.400 | Pass |
| Qualifying median rho | 0.655 | at least 0.600 | Pass |
| Qualifying objects passing rho | 5/8 | at least 6 | Fail |
| Both scanned tools | 1/2 | all | Fail |
| Dust pan rho | 0.600 | at least 0.600 | Pass |
| Sweeper rho | -0.086 | at least 0.600 | Fail |

Mean tau-b over the eight objects with at least 4 variants is 0.430; two-variant objects force tau-b to plus or minus
1. Construction views were reused and samples are small. The later failure analysis found no separating signature, and
hidden geometry against mask or placement error remains unproven. Records:
`data/outputs/mesh_extents/spoon-proxy/` and `data/outputs/mesh_extents/proxy-insights/`.
