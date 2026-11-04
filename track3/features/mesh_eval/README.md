# Packed mesh pair evaluation

This CPU slice evaluates packed D1 Parquets with the unmodified kit
`CD-O.py:score` entry point against public scans. It never changes a submission.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .runtime/kit-env/bin/python -B -m track3.features.mesh_eval pairs \
  --request track3/features/mesh_eval/pairs.json
```

`pairs.json` is the editable queue: labelled Parquet files, the official roster,
a run directory under `outputs/mesh_eval/`, a trial budget and a time budget.
`comparison: ["v1", "v2"]` requests the difference **v2 minus v1**; other labels
and additional candidates are supported. Each file expands to one trial per
covered `(episode, object_slot)`. Choose a fresh output directory for changed
inputs or code; repeat the same request to resume matching completed trials.
The checked-in request writes `outputs/mesh_eval/pairs/critic-fixes-20261102/`.
The historical first run's top-level files in `outputs/mesh_eval/pairs/` stay intact.

`local_cdo.py` moved into this slice after that run, which changes the source
identity, so this request refuses its existing output. For current-code scoring, copy the
request with a fresh `output`; use the recorded commit to reproduce a historical run.

The runtime gate refuses 41/43/44 before loading scans. It reads object scans,
never episode data. The existing kit environment supplies all dependencies;
cache and scratch files go under `.cache/mesh_eval/`. No GPU operation is used.

`pairs.py` owns gating, I/O, hashes, the resumable queue and summary writing.
`_metrics.py` contains pure reference-frame construction and aggregation. It
holds no previous-report constants or fixed candidate labels. The request,
Parquets, roster, public scans, runtime, kit scorer sources, slice sources and
package versions enter the run identity. Narrative documents do not.

The handler reuses `track3.features.mesh_eval.local_cdo.scan_path` and `budgeted`, retaining
the kit's 4,096-face/vertex budget. Each trial passes a reference frame and the
matching packed submission rows to `score`. The kit decodes the actual packed
meshes and uses `_object_seed(episode, slot)`; no candidate is rebuilt from GLBs.
Both episode-1 slots for every file also call `_object_chamfer_cm` on the decoded
arrays and record direct-versus-entry-point equivalence.

The scorer also runs directly as `.runtime/kit-env/bin/python -B track3/features/mesh_eval/local_cdo.py --help`. It loads kit dependencies on import;
callers import it inside scoring functions, so importing the other slice modules
does not require loading the kit scorer.

Each run directory contains:

- `run.json`: request, hashed inputs/sources, versions, coverage, status, timing
  and an artifact hash inventory. A completed record is preserved on rerun.
- `ledger.jsonl`: one durable result per file/pair, including seed, identity,
  CD-O in cm, duration and the direct equivalence result when applicable.
- `summary.json` and `summary.md`: per-pair and per-object scores, both means,
  the requested label comparison (`difference_cm`), largest difference and
  regression, improved/unchanged/worsened counts, and equivalence results.

The pair mean weights all 26 covered pairs equally. The object mean averages
actual pair scores within each object, then weights the ten objects equally.
The ten wooden-spoon pairs have no public scan and remain unscored. Public
scans are measurement references, not host references or submission geometry.

## Recorded result (2026-11-02)

The corrected CPU run took **137.337 s** to recompute 52 scores (two files ×
26 pairs). Every score is
bitwise identical to the last completed record for its identity in the historical
`outputs/mesh_eval/pairs/ledger.jsonl`.

| Packed file | Pair mean (cm) | Equal-object mean (cm) |
| --- | ---: | ---: |
| v1 | 1.25473106 | 1.00910253 |
| v2 | 1.08502364 | 0.92572504 |

V2 improves 13 pairs, leaves 11 unchanged and worsens two; its pair mean is
0.16970742 cm lower. All four direct-versus-entry-point errors are exactly zero.
Previous-report estimates are omitted from generated summaries because their
seed and fitted-threshold protocols differ from this packed-pair measurement.

Canonical summaries: `outputs/mesh_eval/pairs/critic-fixes-20261102/summary.md`
and `summary.json`. Equality evidence: `verification.json` beside those summaries.
The earlier summary and verification remain historical evidence at the parent path.

## Round 3 exact packed veto (2026-11-02)

The pre-score inputs were committed as `bc601cf`. R9 won development selection, with independently passing R8, and passed every stricter Amendment 2 packed gate. The request is `outputs/mesh_extents/rules-round3/job-b/pair-request.json`; it compares shipped v3 with the selected R9 placeholder pack.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/kit-env/bin/python -B -m track3.features.mesh_eval pairs --request outputs/mesh_extents/rules-round3/job-b/pair-request.json
```

| File | Pair mean (cm) | Equal-object mean (cm) |
| --- | ---: | ---: |
| v3 | 0.8114951342417491 | 0.7710770332786878 |
| candidate | 0.6629214779189021 | 0.6533839589873099 |

All **52** file/pair trials completed in **128.515 s**; coverage is **26/36 pairs**, ten public-scan objects. Improved/unchanged/worsened: **9/17/0**. Every object is no worse than its own packed-v3 mean; worst regression **0.0 cm**. All episode-1 entry-point/direct differences are exactly zero.

Object mean **0.6533839589873099 <0.74107703**, pair mean **0.6629214779189021 <=0.81149513**, and maximum regression **0.0 <=0.15 cm**: all packed-veto gates pass. Independent aggregation of the pair ledger reproduces every number.

The adopted file is `submissions/20261102T0516Z-d1-v4/track3_meshes_v4.parquet`. Its geometry and row order are identical to the veto candidate across all 36 pairs, including the unscored spoon. Scorer results therefore transfer by geometry identity, without another scoring pass. The spoon's ten pairs have no public reference; no spoon improvement is measured.

Evidence: `outputs/mesh_eval/pairs/mesh-round3-veto/{run.json,ledger.jsonl,summary.json,summary.md}`, `outputs/mesh_extents/round3-job-b/independent-packed-recheck.json`, `outputs/mesh_extents/round3-job-b/reproduction-checks.json`.

This third adaptive round reuses the same ten public scans and follows earlier scan-render inspection. These are development estimates, not an independent assessment. No scan geometry entered the frozen constructors or the submitted meshes.

## Changed-geometry host equivalence

The additional recorded CPU check covers **episode 19 slot 1, mini_dust_pan**, and
**episode 14 slot 1, plastic_dish_rack**, on both shipped v3 and v4 files. Each check calls the
unmodified host `score`, then `_object_chamfer_cm` on the same decoded arrays with the pair's own
seed. All four absolute differences are **0.0 cm**, and every score exactly matches its earlier
veto-pair value. This covers changed geometry in addition to the historical episode-1 pot/lid checks.

Evidence: `outputs/mesh_eval/equivalence/20261102T055518.792315Z-eb3c237a/{run.json,results.json,summary.md}`.
The request-driven mesh-extents adoption operation reuses the already completed 52-trial pair run;
this verification did not rewrite it. These are development checks on public scans, not independent
performance estimates.
