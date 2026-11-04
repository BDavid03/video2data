# M2 Amendment 1: CPU Stage 0 and gated policy preparation

This slice implements [Amendment 1](../../../docs/chain/moving-m2-protocol.md)
for tuning episode 39. `data/outputs/moving_policy/ep39/` and `diagnosis/` are
immutable historical records. All new artifacts and caches live in
`data/outputs/moving_policy/ep39-a1/`. No scan geometry enters a simulated asset.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy stage0
```

The command refuses to overwrite `stage0.json`. Before that record exists, its
fixed queue resumes completed candidates only when their input/source identity
matches. Each candidate owns a task, asset, imported collision audit and measured
checks. `ledger.jsonl` records support outcomes. Failed implementation attempts
are retained under `revisions/`; they do not change a scientific gate.

`geometry.py` computes grasp transforms, collision penetration, acquisition
stationarity, support dimensions, diagnostic residuals and the amended verdict.
`handler.py` owns files, candidate selection, subprocess records, admission and
training/scoring commands. `cpu.py` uses installed FK and the evaluator's exact
object import. `amended_cpu.py` runs the two support horizons and the isolated
wrist diagnostic with the installed Newton/MuJoCo physics and wrist controller.

The asset is the existing `thin_tool_profile` reconstruction, sized from public
39's frozen video-depth measurement, leveled in the mocap body's first frame.
The object reference is mocap: this is an upper bound, not a video-only entry.
Fingers are fixed from the deterministic shipped template frame. Both hands
follow the object rigidly after acquisition; no object is actuated in support
checks. The diagnostic feed-forward wrist controller is reference-only,
physically clipped, never a learned policy, and never exported or scored.

The support queue is frozen: S1 uses a 1 cm cylinder covering the imported hull
footprint, with top 1 mm below its bottom and original mass/inertia. S2 carries
S1 and uses the shipped URDF's fixed mass 0.3 kg and diagonal inertia 0.001 kg m².
S3 carries S2 and thickens thin closed-prism walls to at least 5 mm before the
unedited evaluator's CoACD. Stop at the first support pass; no S4 exists.
Each trial measures task-trajectory hands through frame 49 and hands cleared
for 60 steps. The acquisition interval and the entire cleared-hand horizon must
both meet 1 cm / 2°. Through-frame-49 motion is also reported.

Grasp candidates are longitudinal ends and transverse scoop sides at its centre.
Eligibility requires at least two distinct groups on opposing exterior faces,
normal dot ≤−0.8, distance ≤1 cm on ≥90% of post-acquisition frames, and maximum
penetration ≤1 cm. The eligible pair with the smallest peak wrist speed wins.
The approach lasts 0.5 s only when the reference stays within 1 mm / 1°; otherwise
hands start at the grasp. No rest is prepended. Speed is descriptive. Finger
limits remain gated. Each diagnostic wrist must meet mean position ≤3 cm,
mean orientation ≤15°, and maximum position ≤15 cm after acquisition.

## Latest result

**PASS**, recorded at `data/outputs/moving_policy/ep39-a1/stage0.json`.
The one-page `summary.md` contains all support and grasp candidates and wrist
measurements. No training or GPU operation ran.

| Check | Measured result |
| --- | --- |
| Support S1 | FAIL: cleared 60 steps, 4.426 mm / 4.137° |
| Support S2 | PASS: acquisition 2.308 mm / 0.000948°; cleared 60 steps 2.308 mm / 1.841° |
| Support S3 | Not run: queue stops at S2's first pass |
| Selected longitudinal grasp | Closure 308/308 frames; penetration 0 m; peak wrist speed 1.59152 m/s |
| Transverse grasp | Ineligible: closure 0%; penetration 9.180 mm; peak wrist speed 0.792462 m/s |
| 4b left wrist | Mean / maximum position 1.902 / 10.685 cm; mean orientation 0.1314° |
| 4b right wrist | Mean / maximum position 1.516 / 11.067 cm; mean orientation 0.1179° |
| Finger limits | Maximum excess 5.02e−9 rad |
| Reference truth | Maximum translation error 2.19011e−7 m; kit identity AUC 1.0 |

Zero residual remains poor: left/right mean position errors 6.477 / 5.731 cm,
maximum errors 22.155 / 31.162 cm, and maximum lever-arm allowance
0.414 / 0.525 m against 0.19 m. These are descriptive, as the amendment requires.
The feed-forward diagnostic's normalized input peaks at 0.987862; its physical
position/orientation residuals remain within 0.2 m / 1 rad per component. This
proves only reference-only wrist reachability with the object absent; it does not
establish learned object tracking or any challenge score.

The first 0.5 s of reference motion is 0.545248 mm / 0.205889°, so the approach
is admitted. The longitudinal grasp transforms and motion summary match HEAD
exactly in `equivalence.json`. Independent float64 recalculation in `recheck.json`
confirms the passing raw support/wrist gates and all frozen hashes. The largest
wrist-position maximum discrepancy is below 1.2e−8 m. Raw solver contact force
peaks for S2 are 129.063 N with hands and 134.155 N cleared; these are solver
estimates, not verified delivered impulses.

## Prepared training and scoring

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy train
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy score
```

These commands only print the upstream argv. Execution additionally requires
`--run`, the lead's runtime flag, passing Stage 0, unchanged scientific hashes,
and `GpuLease`. This CPU job leaves every runtime flag false and starts no
training. The prepared fresh seed-0 recipe has 1,024 worlds, ratio 2.0, replay
capacity 1,048,576/minimum 6,400, checkpoint interval 504,832, unchanged D1
curriculum thresholds, 66,763,776 transitions, and a 4,500 s cooperative deadline.

Only checkpoints **16,154,624**, **32,814,080**, and **66,763,776** are admitted.
The first two are descriptive; the final is gated. The unedited evaluator exports
achieved poses. `score_exports` reuses `partial_eval.motion_metrics`, computes
T/Q/E/Theta and held-still E_S/Theta_S, and records kit AUC, SP-SR, RPE and MPPE.
It retains per-frame achieved-minus-reference errors for M1's allowance screen.
Scan points are allowed only in that later metric, never in an asset or Stage 0.

GO requires T≥0.50, Q≥0.50, E≤min(0.5 E_S, 3 cm),
Theta≤min(0.5 Theta_S, 15°), AUC≥0.45. SIGNAL requires T≥0.20, Q≥0.10,
E≤min(E_S, 6 cm), Theta≤min(0.9 Theta_S, 30°), AUC≥0.20. Otherwise NULL.
No training or scoring claim exists until the lead runs the gated commands.

## Video-reference M2v and scored episode 38

The separate `video.py` entry point keeps the mocap route above intact. It
accepts only 39 and 38, never reads their labels during construction, and uses
M1's geometric body convention. The M2 profile already has that convention:
its profile-to-canonical transform is identity under the same bounding-box,
end-width and dense-floor rule. No scan read is needed. One OBJ and URDF under
`data/outputs/moving_policy/video-asset/` are referenced by both tasks. Mass is
0.3 kg and diagonal inertia is 0.001 kg m². The installed evaluator imports
and decomposes the asset with unedited CoACD.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video stage0 --episode 39 --reference data/outputs/moving_reference/ep39/candidate-c/reference.npz --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video stage0 --episode 38 --reference data/outputs/moving_reference/ep38/candidate-c/reference.npz --output data/outputs/moving_policy/ep38-v1
```

Each build snapshots the source bytes to `reference-source.npz`, records their
SHA-256, then writes `reference-task.npz` with a single recorded translation to
M2's original initial simulator position. It preserves every orientation and
records the residual between source timestamps and the uniform task clock.
The initial position is the pre-existing S2 simulator translation, copied as
a constant; construction reads no M2 mocap, rotation or grasp. Grasp selection,
closure, penetration, acquisition and the CPU physics checks reuse the existing
slice. Both grasp pairs are measured per episode; eligible pairs are ranked
by peak wrist speed. The support is placed 1 mm below imported hulls.

`stage0.json` refuses overwrite. `summary.md` starts with PASS or the failed
checks and gives both asset hashes, the reference hashes, support motion,
feed-forward wrist error, speed and angular-rate facts. A failed check stops
that episode. There is no repair queue for video tasks. `video_geometry.py`
contains the pure self-reference metric calculation; `video.py` owns I/O,
workers, admission, and run records.

`commands.json` and `*.argv.txt` prepare the unchanged seed-0, 66,763,776-step
training recipe with a 4,500 s deadline. `score-self` runs the unedited evaluator
only at that final checkpoint and measures its export against the hashed task
reference. It uses no scan or label. On 39 alone, `score-mocap` subsequently
measures that same export once against mocap, with scan points used only for
the kit metric. Its exclusive attempt record prevents a second label pass,
including after a failed attempt. Episode 38 has no label-scoring operation.
Mocap GO still needs self bounds and the separately performed frozen-camera
reprojection gates before adoption; the score alone never records adoption.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video train --episode 39 --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video score-self --episode 39 --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video score-mocap --episode 39 --output data/outputs/moving_policy/ep39-v1
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy.video kill-check --episode 39 --output data/outputs/moving_policy/ep39-v1
```

These print dry commands. Execution needs `--run` and the independent
`video_train_authorized`, `video_score_self_authorized`,
`video_score_mocap_authorized`, or `video_kill_check_authorized` runtime flag.
All four are false. Existing mocap runtime flags are retained. Training also
requires passing M1 admission in the snapshotted reference, unchanged live
reference bytes, passing Stage 0 and unchanged scientific inputs. Training
and final evaluator execution acquire `src/gpu.py`'s GPU lease.

The kill check invokes `scripts/report/policy_hands.py` with the installed
FlashCHORD interpreter, `policy_32814080.safetensors`, the task and
`--no-official`, with CUDA disabled and JAX on CPU. It writes a diagnostic
replay under `data/outputs/visuals/policy-hands/moving-video-ep39-boundary/`,
then computes self-`T` against `reference-task.npz`. `kill-check.json` marks
`kill=true` when `T < 0.5`. The lead stops training and records NULL on that
result; this CPU command never stops another process. It performs no label
score. The same command is available for 38, though the protocol's explicit
kill rule is for M2v on 39.

The final M1 body file describes its tracking extrusion. For the profile,
`video_geometry.profile_body_transform` uses the embedded original scan bounding
constant and verifies `scan_T_asset @ asset_T_bounding_body` agrees with it,
then composes the inverse profile bounding constant. This removes the tracking
extrusion convention without another scan read or any pose registration.

### Video Stage 0 result (final-reference build)

Both source files still report M1 admission false; training remains blocked.
The immutable per-episode Stage 0 and one-page summaries are under `ep39-v1/`
and `ep38-v1/`. Earlier source/build snapshots are preserved in
`data/outputs/moving_policy/reference-snapshots/` with an explanatory README.

| Episode | Stage 0 | Cleared 60 support, mm / degrees | Worst mean / max wrist position, mm | Worst mean wrist orientation, degrees |
| --- | --- | ---: | ---: | ---: |
| 39 | PASS | 2.68623 / 0.72649 | 18.2655 / 100.069 | 0.135256 |
| 38 | PASS | 5.22025 / 1.77248 | 21.7585 / 116.601 | 0.138424 |

Both choose longitudinal grasp and no approach, so composed acquisition has zero gated steps; hands-cleared support still has 60. All raw-array checks and frozen hashes pass in `recheck.json`. The legacy construct, grasp details and training argv remain exactly equal to HEAD under the default mocap path in `data/.runtime/tmp/moving-video-equivalence/equivalence.json`.
