# Packed mesh evaluation

Scores packed mesh Parquets with the kit's unmodified `CD-O.py:score` against public scans. CPU only.
It never changes a submission.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  data/.runtime/kit-env/bin/python -B -m src.features.mesh_eval pairs \
  --request src/features/mesh_eval/pairs.json
```

`pairs.json` lists labelled Parquet files, the roster, an output directory under
`data/outputs/mesh_eval/`, and trial and time budgets. `comparison: ["v1", "v2"]` reports v2 minus v1.
Repeat a request to resume matching trials; use a new output directory for changed inputs or code. The
checked-in request refuses its existing output, because `local_cdo.py` moved into this slice and changed
the source identity. Copy the request with a new `output`, or use the recorded commit for a historical run.

The scorer also runs alone: `data/.runtime/kit-env/bin/python -B src/features/mesh_eval/local_cdo.py --help`.

Episodes 41, 43 and 44 are refused before scans load. `pairs.py` owns I/O, hashes, the resumable queue
and summaries. `_metrics.py` holds the pure maths. Caches go to `data/.cache/mesh_eval/`. Each run writes
`run.json`, `ledger.jsonl`, `summary.json` and `summary.md`.

The pair mean weights all 26 covered pairs equally. The object mean averages pairs per object, then
weights ten objects equally. Ten wooden-spoon pairs have no public scan and are unscored. Public scans are
measurement references, not host references, and none enters submission geometry.

## Recorded results

| Date | File | Pair mean (cm) | Object mean (cm) |
| --- | --- | ---: | ---: |
| 2026-11-02 | v1 | 1.25473106 | 1.00910253 |
| 2026-11-02 | v2 | 1.08502364 | 0.92572504 |
| 2026-11-02 | v3 | 0.81149513 | 0.77107703 |
| 2026-11-02 | v4 candidate (R9) | 0.66292148 | 0.65338396 |

v2 against v1: 13 pairs better, 11 unchanged, 2 worse. v4 against v3: 9 better, 17 unchanged, none worse,
worst regression 0.0 cm. The adopted v4 file has the same geometry and row order as the candidate, so
scores carry over without another pass. No spoon improvement is measured.

These rounds reuse the same ten public scans after scan-render inspection, so they are development
estimates and not an independent assessment. Summaries:
`data/outputs/mesh_eval/pairs/critic-fixes-20261102/` and `data/outputs/mesh_eval/pairs/mesh-round3-veto/`.

Round 3 packed check:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_eval pairs --request data/outputs/mesh_extents/rules-round3/job-b/pair-request.json
```

Calling `score` and `_object_chamfer_cm` on the same decoded arrays gave differences of exactly 0.0 cm on
the episode-1 pairs and on the changed episode 19 slot 1 and episode 14 slot 1 pairs. Record:
`data/outputs/mesh_eval/equivalence/20261102T055518.792315Z-eb3c237a/`.
