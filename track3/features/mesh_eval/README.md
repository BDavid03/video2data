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

The runtime gate refuses 41/43/44 before loading scans. It reads object scans,
never episode data. The existing kit environment supplies all dependencies;
cache and scratch files go under `.cache/mesh_eval/`. No GPU operation is used.

`pairs.py` owns gating, I/O, hashes, the resumable queue and summary writing.
`_metrics.py` contains pure reference-frame construction and aggregation. It
holds no previous-report constants or fixed candidate labels. The request,
Parquets, roster, public scans, runtime, kit scorer sources, slice sources and
package versions enter the run identity. Narrative documents do not.

The handler reuses `scripts.track3.local_cdo.scan_path` and `budgeted`, retaining
the kit's 4,096-face/vertex budget. Each trial passes a reference frame and the
matching packed submission rows to `score`. The kit decodes the actual packed
meshes and uses `_object_seed(episode, slot)`; no candidate is rebuilt from GLBs.
Both episode-1 slots for every file also call `_object_chamfer_cm` on the decoded
arrays and record direct-versus-entry-point equivalence.

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
