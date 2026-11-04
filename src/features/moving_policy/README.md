# Moving policy

Trains and scores a moving FlashSAC policy under FlashCHORD on a reconstructed dust pan, on
tuning episode 39 and evaluation episode 38. There are two routes: mocap goal on 39 (M2,
`__main__`), and video-derived goal on 39 and 38 (M2v, `video.py`). No scan geometry enters a
simulated asset. Protocols and results are in
[moving-m2-protocol.md](../../../docs/chain/moving-m2-protocol.md),
[moving-m2-outcome.md](../../../docs/chain/moving-m2-outcome.md),
[moving-m2v-protocol.md](../../../docs/chain/moving-m2v-protocol.md) and
[moving-m2v-outcome.md](../../../docs/chain/moving-m2v-outcome.md).

Train and score commands only print the upstream command unless given `--run`. Execution also
needs the runtime flag, a passing Stage 0, unchanged hashed inputs and the GPU lease.
`stage0.json` refuses to be overwritten.

| File | Role |
| --- | --- |
| `geometry.py`, `video_geometry.py` | Pure calculations: grasp, penetration, support, verdicts, self-reference metrics |
| `handler.py`, `video.py` | Files, candidate selection, admission, train and score commands |
| `cpu.py`, `amended_cpu.py` | CPU checks using the installed FK, evaluator import and physics |

## Mocap route on 39 (M2)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy stage0
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy train
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy score
```

Output: `data/outputs/moving_policy/ep39-a1/`. `ep39/` and `diagnosis/` are historical. The object
reference is mocap, so this is an upper bound and not a video-only entry. The asset is the
`thin_tool_profile` reconstruction sized from episode 39's video depth. Fingers come from the shipped
template frame, and both hands follow the object rigidly after acquisition. The wrist controller in
the support diagnostic is never exported or scored.

Frozen support queue, stopping at the first pass: S1 a 1 cm cylinder 1 mm below the imported hull; S2
adds fixed mass 0.3 kg and diagonal inertia 0.001 kg m²; S3 thickens thin walls to 5 mm. A grasp pair
needs two opposing groups (normal dot at most -0.8), distance at most 1 cm on at least 90% of frames,
and penetration at most 1 cm. The smallest peak wrist speed wins.

Stage 0 result: PASS.

| Check | Result |
| --- | --- |
| Support S1 | FAIL: cleared 60 steps, 4.426 mm, 4.137 degrees |
| Support S2 | PASS: acquisition 2.308 mm, 0.000948 degrees; cleared 2.308 mm, 1.841 degrees |
| Longitudinal grasp (selected) | Closure 308/308, penetration 0, peak wrist 1.59152 m/s |
| Transverse grasp | Ineligible: closure 0%, penetration 9.180 mm |
| Wrists, mean / max position | Left 1.902 / 10.685 cm; right 1.516 / 11.067 cm |

Zero residual stays poor (mean position error 6.477 and 5.731 cm), and the lever-arm allowance is
exceeded (0.414 and 0.525 m against 0.19 m). The diagnostic shows wrist reachability with the object
absent. It does not show learned tracking or any score. S2's contact forces of 129 to 134 N are solver
estimates.

Recipe: seed 0, 1,024 worlds, ratio 2.0, replay 1,048,576 with minimum 6,400, checkpoint interval
504,832, 66,763,776 transitions, 4,500 s deadline. Only checkpoints 16,154,624, 32,814,080 and
66,763,776 are admitted, and only the last is judged. GO needs T at least 0.50, Q at least 0.50,
E at most min(0.5 E_S, 3 cm), Theta at most min(0.5 Theta_S, 15 degrees), and AUC at least 0.45. SIGNAL
needs T at least 0.20, Q at least 0.10, E at most min(E_S, 6 cm), Theta at most min(0.9 Theta_S, 30
degrees), and AUC at least 0.20. Otherwise NULL. Scan points appear only in the later metric.

## Video route on 39 and 38 (M2v)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video stage0 --episode 39 --reference data/outputs/moving_reference/ep39/candidate-c/reference.npz --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video stage0 --episode 38 --reference data/outputs/moving_reference/ep38/candidate-c/reference.npz --output data/outputs/moving_policy/ep38-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video train --episode 39 --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video score-self --episode 39 --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video score-mocap --episode 39 --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video kill-check --episode 39 --output data/outputs/moving_policy/ep39-v1
```

The last four print commands by default. Each needs `--run` and its own flag
(`video_train_authorized`, `video_score_self_authorized`, `video_score_mocap_authorized`,
`video_kill_check_authorized`), plus M1 admission in the snapshotted reference. Training and the final
evaluator hold the GPU lease from `src/gpu.py`.

Stage 0 never reads labels. One OBJ and URDF under `data/outputs/moving_policy/video-asset/` serve both
tasks (0.3 kg, diagonal inertia 0.001 kg m²). The installed evaluator decomposes the asset with
unedited CoACD. Each build snapshots the reference to `reference-source.npz` and writes
`reference-task.npz` with one recorded translation to the existing S2 start position, a constant
copied without reading M2's mocap, rotation or grasp. A failed check stops that episode, and there is
no repair queue.

`score-self` measures the final checkpoint against the hashed task reference with no scan or label.
`score-mocap` runs on 39 only, once: its attempt record blocks a second pass even after a failure, and
scan points enter only the kit metric. Episode 38 has no label scoring. A mocap GO alone is not adoption: it still
needs the self bounds and the separately performed frozen-camera reprojection gates.

`kill-check` replays `policy_32814080.safetensors` on CPU through `scripts/report/policy_hands.py`,
writes under `data/outputs/visuals/policy-hands/moving-video-ep39-boundary/`, and sets `kill=true` when
self-T is below 0.5. It does not stop any process or score labels. The protocol's explicit kill rule
is for 39.

The M1 body file describes its tracking extrusion. For the profile,
`video_geometry.profile_body_transform` composes the embedded scan bounding constant with the inverse
profile constant, with no further scan read or pose registration.

Video Stage 0 (`ep39-v1/`, `ep38-v1/`; earlier snapshots in
`data/outputs/moving_policy/reference-snapshots/`):

| Episode | Stage 0 | Cleared 60 steps | Worst wrist mean / max position | Worst wrist orientation |
| --- | --- | ---: | ---: | ---: |
| 39 | PASS | 2.68623 mm, 0.72649 degrees | 18.2655 / 100.069 mm | 0.135256 degrees |
| 38 | PASS | 5.22025 mm, 1.77248 degrees | 21.7585 / 116.601 mm | 0.138424 degrees |

Both use the longitudinal grasp with no approach.

Final outcomes, from `moving-m2v-outcome.md`:

| Episode | Against | Result |
| --- | --- | --- |
| 39 | Its own reference | T 1.001, Q 0.974, 1.21 cm, 1.49 degrees, AUC 0.936 |
| 39 | Mocap, raw frames | NULL: the recorded verdict, comparing two unrelated world frames |
| 39 | Mocap, first-pose alignment | GO after that correction: 2.67 cm (limit 3), 8.31 degrees, AUC 0.775 old scorer |
| 38 | Its own reference | T 1.026, Q 0.973, 1.42 cm, 1.59 degrees, 263 of 263 frames, no resets |

The alignment was chosen after seeing the NULL. Episode 38 failed badly at the midpoint (T 0.25 at 32.8M), and we do not know why. The kill rule fired at 34.3M and the run was resumed, which that rule alone would not have allowed. The owner stopped it again at about 40.7M. It lost its replay each time, so it is a different experiment from 39's fresh run. It has no labels, so these numbers do not show its goal
is correct or that it beats its static export. On the new scorer, 39's AUC falls to 0.669 with the
simulated mesh and 0.591 with the tracking mesh, and MP-SR falls from 1 to 0.
