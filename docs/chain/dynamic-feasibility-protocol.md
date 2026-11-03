# Dynamic feasibility D0: can our policy stage move an object along a reference on this machine? (context 19; written before any run; commit time is the record)

Route: `outputs/chain/codex/20261103T140047Z-route-struct-19.answer.md`. Owner focus: `PROGRAM.md` section 0, to-do 1.

## Why

D2 v4 is static: objects hold their frame-0 placement. At evaluation FlashCHORD sets its virtual object controller to
zero, so an object moves only if simulated hands manipulate it; the policy adds residual wrist/finger targets around a
hand reference. Before any work on video-derived hand and object trajectories, we measure whether a policy trained
within our budget (RTX 5080, 16 GB; the released recipe wants 48 GB and 250M transitions) reproduces the motion of
the shipped tissue-box example task (390 frames, 30 Hz, hand and object references, about 17 cm of object travel).
This is a capability diagnostic on a shipped example, never a submission, and it reads no challenge episode.

## Method (fixed now)

- Task: the shipped example task, unchanged, its declared fps passed as `sim.fps` (no resampling).
- Evaluation: the unchanged official `evaluate_policy.py` path (VOC scale zero), exporting simulated object poses.
- Arms, in this order, in one resumable queue with a ledger and a summary (PROGRAM.md section 9):
  - **Z**: zero-update checkpoint (hands follow the reference through their controllers, no learning).
  - **L1**: the existing feasibility budget (64 worlds, about 128k transitions), seeds 0 and 1.
  - **L2**: only if L1's better seed improves on Z's mean object position error by >= 20 %: the largest budget that
    fits 16 GB within 45 GPU-minutes, seeds 0 and 1. The VOC curriculum is rescaled so assistance reaches zero
    before the end of training in every learned arm.
- Comparator **S**: the object held at its initial pose over the reference timeline (computed, not simulated).
- Measures per arm and seed, over the full sequence: mean and 90th-percentile object position error (cm), mean
  orientation error (deg), the kit ADD-AUC (1-9 cm) on the task's object mesh points, rollout completion (finite,
  all frames, no early termination), wall time and peak GPU memory.

## Go / no-go (fixed now)

**GO** if some learned arm, on BOTH seeds: completes finite full rollouts, mean object position error <= 3 cm, and
<= 0.5 x comparator S's mean error. Otherwise **NO-GO**: dynamic trajectories cannot land by the 12:00 UTC 4 Nov pack
cut-off with this policy stage, recorded as a measured finding (not a claim that FlashCHORD cannot manipulate), and the
effort moves to to-do 3 (whole-clip meshes). Z's result is reported either way (it says how much the hand reference
alone moves the object).

Stop rule: 75 GPU-minutes in total, including compilation. A crash or non-finite state in every seed of an arm is that
arm's failure; it is not retried with new settings after the fact.

Episodes 41, 43, 44 are never read; no challenge episode is read; no leaderboard use.
