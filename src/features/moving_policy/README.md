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
