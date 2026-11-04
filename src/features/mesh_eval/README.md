# Packed mesh evaluation

Pipeline role: step 2 (shapes), scoring. It scores packed mesh files with the kit's own CD-O code against public scans,
so we can compare shape rules. It never changes a submission.

CD-O is the Chamfer distance in cm between a mesh and the object's scan after the best similarity alignment. Lower is
better. A packed file (pack) is the Parquet the kit's packer writes for upload.

## Run

CPU only, with the kit interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  data/.runtime/kit-env/bin/python -B -m src.features.mesh_eval pairs \
  --request src/features/mesh_eval/pairs.json
```

`pairs.json` lists the packed files, the object list, an output folder under `data/outputs/mesh_eval/`, and trial and
time budgets. `comparison: ["v1", "v2"]` reports v2 minus v1. Rerunning a request resumes matching trials. The checked-in
request refuses its old output folder because the code moved and its hash changed: copy it with a new `output`, or run at
the recorded commit.

The scorer alone: `data/.runtime/kit-env/bin/python -B src/features/mesh_eval/local_cdo.py --help`.

Episodes 41, 43 and 44 are refused before any scan loads. Caches go to `data/.cache/mesh_eval/`. Each run writes
`run.json`, `ledger.jsonl`, `summary.json` and `summary.md`. `pairs.py` does I/O and the queue; `_metrics.py` is the maths.

Pair mean: all 26 scored pairs weigh the same. Object mean: average per object, then the ten objects weigh the same.
Ten wooden-spoon pairs have no scan and are not scored. Public scans are only a measuring stick; none goes into a
submitted mesh.

## Results

| File | Pair mean (cm) | Object mean (cm) |
| --- | ---: | ---: |
| v1 | 1.25473106 | 1.00910253 |
| v2 | 1.08502364 | 0.92572504 |
| v3 | 0.81149513 | 0.77107703 |
| v4 (R9) | 0.66292148 | 0.65338396 |

v2 against v1: 13 pairs better, 11 same, 2 worse. v4 against v3: 9 better, 17 same, none worse. The adopted v4 file has
the candidate's geometry and row order, so its score carries over. These rounds reuse the same ten scans, so they are
development estimates, not an independent test. Summaries: `data/outputs/mesh_eval/pairs/critic-fixes-20261102/` and
`data/outputs/mesh_eval/pairs/mesh-round3-veto/`.

Round 3 packed check:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/kit-env/bin/python -B -m src.features.mesh_eval pairs --request data/outputs/mesh_extents/rules-round3/job-b/pair-request.json
```

The kit's `score` and our `_object_chamfer_cm` give identical results (difference 0.0 cm) on the tested pairs. Record:
`data/outputs/mesh_eval/equivalence/20261102T055518.792315Z-eb3c237a/`.
