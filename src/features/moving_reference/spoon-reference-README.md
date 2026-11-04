# Spoon reference continuation

The ordered queue reuses the completed masks for evaluation episodes 35, 36 and 37.
The lead waived the texture-usability fraction and longest texture-unusable gap on
4 November 2026. Stereo support and every other M1 final limit remain unchanged.
Historical mask-screen records are preserved. This continuation writes a separate
`reference-mask-waiver/` directory for each episode.

Run from the checkout, using the existing project interpreter:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.spoon_reference_object
```

The camera phase calls the existing `candidate_c.prepare_frame0`,
`camera_ba.observations`, `bundle_adjust` and `heldout_rows`. Its final measurements
use the unchanged limits in `_final_gates.FINAL_LIMITS`, including the perpendicular
world-residual instrument admitted by the existing sealed condition. The table
phase calls `full_c.cache_frames` and `full_c.table_fit` without changing settings.
Each phase stops an episode at a failed gate and moves to the next queued episode.
Camera observations and bundle-adjusted poses are reused when already saved.
A camera or table pass alone does not admit a reference or training.

Outputs include hashed input/source requests, camera observations and diagnostics,
bundle-adjustment reports, table evidence, UTC status lines and a per-gate summary:
`data/outputs/moving_policy/spoon-reference-summary.md`. The summary marks downstream
checks as NOT RUN after a stop. The object phase uses the same forward/backward model fitting and smoothing as
candidate C. The pan-specific handle-hole initializer is replaced by M1's existing
generic stereo/PCA four-heading silhouette initializer, with candidate C's fixed
base. Its tracking outline comes from the shared own spoon mesh. This object
adaptation is disclosed and fixed across the three episodes. The current entry
returns to the lead if an episode passes camera, table fit and object checks,
before the remaining table drift audits; it never starts training.

`spoon-reference-equivalence.json` independently compares the six camera measurements
with episode 38's archived final gate record: all values agree exactly, and all
limits and verdicts agree. This establishes instrument equivalence, not spoon pose
accuracy. No spoon labels or scans are read.

Object-loop equivalence: on three real episode-38 frames, forward and backward
poses agreed exactly with the original loop when supplied the same initial pose
and model; every diagnostic also agreed. Evidence:
`data/outputs/moving_policy/spoon-reference-object-equivalence.json`.

## Outcome (4 November 2026)

| Episode | First failed remaining gate | Measurement | Limit |
| --- | --- | ---: | ---: |
| 35 | Maximum angular rate | 1,895.293 degrees/s | 540 degrees/s |
| 36 | Table patch normal disagreement | 3.260364 degrees | 3 degrees |
| 37 | Table patch normal disagreement | 3.827163 degrees | 3 degrees |

Episode 35 also failed frame-zero penetration (14.425 mm, limit 5 mm).
Episodes 36 and 37 stopped before object tracking. All downstream unmeasured
checks remain unmeasured. No Stage 0, training, GPU operation or ready file was
produced. The waiver did not yield an admitted reference.

Final per-episode records: `epNN/reference-mask-waiver/gates-final.json` under
`data/outputs/moving_reference/`. `spoon-reference-audit.json` independently
recomputed episode 35's peak rate and speed and verified 997 recorded input hashes.
The rate peak is a 94.765-degree step between frames 189 and 190 (50 ms).

Exact source versions used by the runs are archived in
`data/outputs/moving_policy/spoon-reference-source-snapshots/`. After the runs,
gate composition moved from the pure array module to its handler; the array
math did not change. `spoon-reference-cleanup-equivalence.json` verifies exact
agreement on all four available camera records (38 and 35–37).
