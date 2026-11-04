# Mesh extents

Pipeline role: step 2 (shapes). It measures each object's proportions from MoGe-2 video depth and builds a closed,
hand-designed (procedural) mesh from a rule for that object type. The submitted file uses these procedural shapes
(the v4 set, rule R9).

CD-O is the shape score: the Chamfer distance in cm between our mesh and the object's scan, after the best
similarity alignment (rotation, shift and scale). Lower is better. Because of that alignment, shape matters and absolute
size and placement do not. Meshes use video only, but we chose the rules by scoring them on public scans. MoGe-2 scale is
not anchored, so simulation needs metric scale from elsewhere (stereo). Episodes 41, 43 and 44 are excluded.

## Operations

All CPU. `measure` reads a finished depth run; it does not run MoGe-2.

| Operation | Input and output |
| --- | --- |
| `measure` | Assigned table clusters from a `depth.generate` run; labels, planes, overlays, `extents.json` |
| `build` | Measure runs; pooled shapes, `objects.json`, one PLY per object and pair |
| `rules` | Frozen queue of candidate rules; builds, scores, ranked summaries. `--dry-build` skips scoring |
| `adopt` | Frozen winner; admission check, pack, packed-pair check |
| `proxy` | Frozen meshes and video measurements; silhouette IoU ledger and report |
| `proxy-insights` | Replay of selected yaw and pair comparisons |

A pack is the Parquet file the kit's `pack_submission.py` writes for upload. An admission check is a pass/fail rule,
fixed before the result is known, that a result must pass before it is used.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B src/features/mesh_extents/measure_loop.py --out data/.runtime/tmp/d1-runs.json
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents measure --request <measure.json>
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents build --request <build.json>
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents.rules --queue src/features/mesh_extents/rules-round3.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents adopt --request src/features/mesh_extents/adopt-round3.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents proxy --request src/features/mesh_extents/proxy-spoon.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents proxy-insights --request src/features/mesh_extents/proxy-insights.json
```

`measure` requests name `source_split`, `episode_index`, `depth_run`, `frames` and the object-to-cluster map `objects`;
`assignments.json` records the checked identities. `build` requests name `measure_runs`, the rule and any frozen
attributes; see `requests/build-v1-box.json` to `requests/build-v5.json`.

`proxy` and `proxy-insights` refuse to run at this commit: their frozen requests pin older source hashes. Run them at
their recorded commit. Rule queues hash the scorer and `rules.py`, so a new campaign needs a new output folder.

## Method

Find the floor in the lower 70% of the image, then a table-height peak 0.200 to 1.200 above it. Object extents are
in-plane PCA percentiles 2 to 98 and height percentile 98. Across episodes, normalise each by its geometric-mean size,
take median proportions, restore median size. Undersides, cavities and wall thickness are rough. Overlapping closed parts
can keep internal faces, so a closed mesh is not a clean union.

Code: `_shapes.py` and `_rule_scores.py` build and select shapes, `_union.py` is the R11 rack wrapper, `_geometry.py`
picks the table, `_proxy.py` is the proxy maths.

## Results (CD-O on public scans, cm)

v4 uses rule R9: the R7 compartment rack plus R8 thin-tool profiles for dust pan, sweeper and spoon.

| Rule | Object mean | Pair mean | Outcome |
| --- | ---: | ---: | --- |
| v1 box | 1.008 | 1.257 | Baseline |
| R1 cylinder | 0.956 | 1.184 | Eligible |
| R2 handle | 0.926 | 1.087 | Adopted (pitcher worse, 1.577 to 1.683) |
| R3 head | 0.990 | 1.243 | No gain |
| R4 rim handle | 0.921 | 1.077 | No gain |
| R5 hollow | 0.846 | 0.956 | Eligible |
| R6 peg rack | 0.773 | 0.817 | Adopted as v3 |
| R7 | 0.720 | 0.715 | Development |
| R8 | 0.706 | 0.765 | Development |
| R9 = packed v4 | 0.653 | 0.663 | Adopted as v4 |

Coverage: 10 scanned objects, 26 of 36 pairs. Ten spoon pairs have no scan and are not scored. v4 against v3: 9 pairs
better, 17 tied, none worse. These are development numbers: three rounds reused the same scans. The v4 file was
`submission/20261102T0516Z-d1-v4/track3_meshes_v4.parquet` (294,912 rows, SHA-256
`6663a1c7e0491c0cfa53b27fe36d7904455351dad548181e4e59a2263215af50`; not in this checkout). Record:
`data/outputs/mesh_extents/round3-job-b/job-b.md`.

Round 4 changed nothing. Candidate C11 (v4 plus R11, the rack as one exterior boundary of 157 vertices and 310 faces)
scored worse (rack 0.860 to 0.904 cm; object mean 0.6571 against 0.6527 cm). R10 pooled vessel profiles failed a
video-only check on tuning episode 42 (knots 45.7% of radius apart; limit 5%).

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents.rules --queue src/features/mesh_extents/rules-round4.json --dry-build
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents.rules --queue src/features/mesh_extents/rules-round4.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_extents adopt --request src/features/mesh_extents/adopt-round4.json
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents.profiles_diagnostic --output data/outputs/mesh_extents/round4-profiles-rerun
```

The last command is the R10 check. Recorded round-4 outputs verify only at their recorded commit; use a new output
folder for a new run.

Rebuild v4 and compare all 47 PLY hashes (plan first, then the build):

```bash
bash scripts/reproduce/d1_meshes.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d --version v4
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_extents build --request src/features/mesh_extents/requests/build-v4.json
```

## Spoon proxy: failed

The spoon has no public scan. We tested whether silhouette IoU against the video could stand in for CD-O. It failed its
pre-registered gates, so it says nothing about spoon CD-O, and v4 is unchanged.

| Gate | Value | Needed | Result |
| --- | ---: | ---: | --- |
| Mean tau-b (10 objects) | 0.544 | at least 0.400 | Pass |
| Qualifying median rho | 0.655 | at least 0.600 | Pass |
| Qualifying objects passing rho | 5/8 | at least 6 | Fail |
| Both scanned tools | 1/2 | all | Fail |
| Dust pan rho | 0.600 | at least 0.600 | Pass |
| Sweeper rho | -0.086 | at least 0.600 | Fail |

Samples are small and views were reused. Records: `data/outputs/mesh_extents/spoon-proxy/`.
