# Scored stereo-anchored S1 follow-up

Run the fixed queue serially with the approved environments:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -u -m track3.features.pose_insurance.scored_scale_loop
```

`queue_v3.json` fixes the sixteen two-object evaluation episodes, the four untouched
single-object episodes and the v2 512/1/32 budget. `scored_scale_loop` prepares exact
stereo/depth identities, records immutable plans, then invokes the slice handler
once per changed scene. The handler reuses the admitted v2 task and established
training/evaluator functions under `track3.gpu.GpuLease`. Only slot 1's table-plane
position changes. `_scored_scale.py` contains pure scene editing and physical checks.
The other modules orchestrate stereo, evaluator execution, reports and packaging.

Completed physical failures are final per-episode fallbacks and are never retried.
A resumed started attempt is admitted from its run record; a live owner is waited
on and an interrupted attempt falls back without a retry. Re-running `--phase all`
resumes the existing fixed plans; `--phase pack` only composes/packages the frozen
decisions. Existing run records and v1/v2 files are immutable.

Outputs are `outputs/pose_insurance/v3/{ledger,plans,decisions,finished}.json*`,
per-episode stereo and evaluator logs, `build-v3/`, `run-v3/`, `results-v3/`, and a
new timestamped v3 submission directory. `v3/report.md` is the authoritative result:
scales, physical checks, no-ops/fallbacks, composition, pack hash, preflight,
source verification and CPU package reproduction. `scored_scale_loop --phase prepare`
prepares the same plan without entering the GPU stage.

The evaluation rig is assumed to share the public stereo calibration. Shortening
an incorrectly directed offset may hedge its error, so no host-score gain is
claimed and AUC/SP-SR are expected to remain effectively unchanged.
