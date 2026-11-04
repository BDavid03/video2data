# pose_refresh: frame-zero FoundationPose line without wall-clock deadlines

The frame-zero FoundationPose modules in `src/features/pose_insurance/` (`frame0_fp.py`, `frame0_fp_worker.py`,
`frame0_fp_policy.py`, `frame0_fp_retrain.py`) are pinned by the packed candidate manifest of 4 November 2026 and carry
hard-coded cutoffs that have passed (registration queue 21:30 UTC, retrain 21:15 UTC, static-policy admission 21:25 UTC).
This slice holds copies of those four modules with every wall-clock gate removed and nothing else changed, so the
registration, conversion, static RL recipe and physical guards are the same instruments. The pinned originals are not edited.

What differs from the pinned copies:

- no `cutoff_utc` in the registration queue, no `before_cutoff` term in the physical admission, no retrain cutoff;
- outputs under `data/outputs/refresh/frame0-fp/` (`registration/`, `policies/`) instead of `pose_insurance/frame0-fp/`;
- GPU lease stages `pose_refresh.frame0_fp` and `pose_refresh.frame0_fp_static`;
- the worker launcher is `run_worker.sh` in this directory, with caches under the registration output;
- admission records have kind `pose_refresh.frame0_fp_admission`;
- the pure helper `_frame0_fp_frames.py` is imported from `pose_insurance` unchanged.

The mask stage's `stop_utc` (`src/features/moving_reference/runtime.json`, read by `masks.py` and `_worker.py`) is now
optional: `null` means no deadline. Those files are not pinned by any manifest.

Commands (project venv, from the checkout root):

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_refresh.frame0_fp          # registration/queue.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_refresh.frame0_fp_retrain  # policies/queue.json
```

`registration/queue.json` is `{"episodes": [...]}`; `policies/queue.json` the same. Adoption limits, the train/tuning
calibration and the scored decision still need their own protocol file; the integrity gates (episode allowlist, one GPU
lease, limits decided before scored results) are unchanged.
