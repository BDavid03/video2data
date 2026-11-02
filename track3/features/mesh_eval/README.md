# Packed mesh pair evaluation

This slice evaluates the actual D1 v1/v2 Parquet files with the unmodified kit
`CD-O.py:score` entry point. It is separate from `mesh_extents` so evaluation does
not change the sources pinned by D1 reproduction.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .runtime/kit-env/bin/python -B -m track3.features.mesh_eval pairs \
  --request track3/features/mesh_eval/pairs.json
```

`pairs.json` is the editable input queue: candidate labels/files, the official
episode/object roster, output directory, trial budget, and time budget. Each file
expands to one trial per covered `(episode, object_slot)` in roster order. The
runtime gate permits only evaluation episodes and refuses 41/43/44 before loading
scans. This entry reads public object scans, never episode data. It uses the
existing kit environment; no dependency installation or GPU operation is needed.

The handler imports `scripts.track3.local_cdo.scan_path` and `budgeted`, retaining
the host's 4,096-face/vertex budget. The pure `_metrics.py` builds reference solution
frames using the kit's own row-id/array definitions. Every trial passes its
reference frame and the matching rows from the packed submission into `score`;
the kit performs candidate decoding, provenance checks, and sampling with
`_object_seed(episode, slot)`. Candidate meshes are never reconstructed from source
GLBs. Both slots of episode 1 for both files are additionally evaluated directly
with `_object_chamfer_cm` on kit-decoded arrays to record numerical equivalence.

`pairs.py` owns I/O, the runtime gate, input/source hashes, and the resumable loop.
`_metrics.py` owns only pure frame construction and aggregation. No existing
`mesh_extents` or reproduction source is modified.

Outputs are under `outputs/mesh_eval/pairs/`:

- `run.json`: hashed Parquet files, roster, scans, sources, dependency versions,
  request, coverage, completion status, wall time, and hashed artifacts.
- `ledger.jsonl`: a durable result per file/pair, including its seed, score,
  identity, duration, and the equivalence result where applicable.
- `summary.json` and `summary.md`: per-pair scores, equal-pair and equal-object
  means, historical comparisons, largest pair difference, and equivalence results.

Completed identities are skipped after interruption; an incomplete final ledger
line is discarded before the next durable append. A completed `run.json` is
preserved on repeat invocation. Different input/source identities are refused in
an occupied output directory, preserving the existing record. Scratch/cache
files are confined to `.runtime/tmp/ctx06/cdo/`.

The pair mean averages the 26 covered pairs equally. The object mean first
averages each object's actual pair scores, then averages the ten object means.
Earlier results reused each object's score over seed episodes 1/3/5 at slot 0;
the cross-fitted v2 estimate also used fold-selected thresholds instead of the
packed build's fixed threshold 1.3. These estimates answer different questions.

Public scans are measurement references, not host references. The ten spoon
pairs have no public scan and remain unscored; no geometry is fed into submissions.

The completed CPU run took **142.165 seconds** for 52 trials, covering **26/36
pairs over ten objects** per file:

| Packed file | Exact pair mean (cm) | Equal-object mean (cm) |
| --- | ---: | ---: |
| v1 | 1.25473106 | 1.00910253 |
| v2 | 1.08502364 | 0.92572504 |

The v2 pair mean is 0.16970742 cm lower than v1. Thirteen pairs improve, eleven
are unchanged, and two regress. The largest absolute change is episode 8, slot 1
(`white_pot`, seed 8000125): 1.83234075 → 1.31825076 cm, a 0.51408999 cm gain.
The largest regression is `water_pitcher`, episode 20, slot 1: +0.11116771 cm.
The pair means differ from the earlier reused-object estimates by −0.00188894 cm
(v1) and −0.00239636 cm (v2). V2's equal-object mean is +0.00062504 cm above the
earlier fitted 0.9251 cm estimate. All four direct-versus-`score` checks have
exactly zero error. Independent pandas aggregation agrees within 2.23e-16 cm;
every recorded input, source, and artifact hash was rechecked successfully.

Full results: `outputs/mesh_eval/pairs/summary.md` and `summary.json`. Verification
commands and packed-file hashes: `outputs/mesh_eval/pairs/verification.md`.
