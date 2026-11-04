# S1 silhouette diagnostic

Measures the precommitted S1 mask agreement of saved HR2/H3 basins on CPU, with
no pose refitting. `_silhouette.py` owns pure geometry, clustering, matching and
scores; `silhouette_loop.py` admits inputs, records predictions and separately
assesses the spent cases. The exact 13-object roster and numerical thresholds
are pinned in `silhouette_queue.json` and bound to the committed protocol.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.silhouette_loop
```

`--operation predict` stops after all 13 predictions and the figure; `--operation
assess --prediction-report <run.json>` admits that complete report before opening
any mocap. Unchanged trial identities resume from the append-only ledger.

Outputs live under `outputs/pose_insurance/silhouette/`: immutable per-object
`trials/*/{run.json,prediction.json,renders.npz}`, `predictions/*` with the complete
candidate table and one `candidate-outlines.png`, and `assessments/*` with label
errors, controls and verdict. Root `predictions.json` and `assessment.json` point
to those reports; `predictions.md` and `summary.md` provide readable summaries.
Score order is original/eroded/dilated, IoU is unitless, contour chamfer is in
pixels and capped at 8 px. Masks are depth-derived partial regions. Control
comparisons use the best and farthest basin regardless of loss; other objects
use only basins within 5% of their search's minimum depth loss.

Latest result (3 November 2026): **FAIL; episode-19 application withheld**.
All 13 predictions finished at 14:08:18 UTC, before the separate assessment
started at 14:08:49 UTC. Episode 18's correct sweeper basin wins every comparison,
but its minimum IoU lead is 0.02481043, below 0.05. Its search-zero best/rival IoUs
are 0.792341/0.764124 and C errors are 5.6309/176.8835 degrees. Both piece-1 cases
lack an upright candidate; both piece-2 cases abstain. No wooden flip is resolved.
Only 2/7 controls (39 pan and 27 pot) have complete repeated matches; both retain
the depth-best. The other five controls are incomplete, with no far-basin veto.
Episode 19's saved prediction also abstains; its search-zero best/rival IoUs are
0.782869/0.787093. The independent raw-input check agrees exactly on all four
search-zero sweeper candidates, including all three mask variants (maximum
absolute IoU difference 0).

The initial complete panel was saved and assessed before a reporting correction:
failed control matches had unnecessarily rendered unrelated representatives.
Those scores never entered a decision. The corrected loop records an empty
comparison for an unmatched control search. Both panels and their assessments
are retained; thresholds, sweeper scores and verdicts are unchanged. An earlier
admission error, before any projection, was fixed by using the depth producer's
explicit preprocessing artifact hash. See `outputs/pose_insurance/silhouette/report.md`
for all candidate/search/variant values, timestamps, commands and integrity evidence.
This diagnostic informs the owner's D2 v5 risk assessment; it changes neither H3
nor a submission.
