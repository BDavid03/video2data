# Dynamic feasibility

Step 4 groundwork: we tested whether a reinforcement-learning (RL) policy can carry a moving object
through the official evaluator before we tried it on episode 38. The recipe that passed here (M0b/M0c) is
the one we used for episode 38.

## What it is

All runs use FlashCHORD's shipped sample task `tissue_box_simple` (390 frames at 30 Hz), with its own
reference path and mesh. No challenge episode is read and nothing here goes into a submission. The task,
FlashCHORD and the kit evaluator are read-only. Each stage had a written protocol in `docs/chain/`,
committed before any of its results.

Terms:

- **Policy**: the trained controller for the simulated robot hands.
- **FlashSAC**: FlashCHORD's soft actor-critic RL trainer.
- **Assistance (VOC)**: a virtual force that helps the object along during training. The curriculum lowers it
  to zero. The official evaluator always turns it off.
- **S**: the still baseline. The object's first pose is held for every frame.
- **T (transport)**: how much of the reference motion the object makes. 1 = all of it, 0 = none.
- **Q (motion score)**: how well half-second motion steps match the reference. 1 = perfect, 0 = no better than still.
- **E**: mean position error in cm. **Perpendicular error**: sideways error along the path, in cm.
- **ADD-AUC**: the kit's pose-accuracy area under the curve, on 1,000 seeded surface points. 1 is best.

These are measurements on the sample task. They are not challenge scores.

## Results

| Stage | Question | Verdict | Key numbers |
| --- | --- | --- | --- |
| D0 | Does a short run (128,000 transitions, 64 worlds, seeds 0/1) beat zero updates? | NO-GO | Best gain over zero updates 0.83%. Position error S/Z 6.889/7.012 cm; seeds 7.026/6.954 cm; ADD-AUC 0.428/0.448. 271 GPU s. |
| D1 | Does a scaled run (1,024 worlds, 66,763,776 transitions per seed) finish in 30 min? | NO-GO, not finished | Sizing timed only simulator steps (28.7k-49.5k steps/s). Real training ran at 12.9k-13.6k/s, so the budget was 3.7x too large. Seeds stopped at 21.6M / 22.6M transitions. 3,601 GPU s. |
| M0 | Do D1's unfinished policies carry the box? | NULL | T 0.035 / 0.009, Q 0.075 / 0.009, E 6.80 / 6.83 cm. All 390 jump-and-hold controls fail too. 42 GPU s. |
| M0b | Does D1 seed 0, resumed to the full schedule, carry it? | GO | Final: T 0.987, Q 0.875, E 0.562 cm, perpendicular 0.339 cm, ADD-AUC 0.981. 35.7 GPU min. |
| M0c | Does the same recipe work from zero, seed 1? | GO | Final: T 0.992, Q 0.872, E 0.516 cm, perpendicular 0.251 cm, ADD-AUC 0.979. 2,959 s wall. |

M0b/M0c recipe: update ratio 2.0, replay capacity 1,048,576, learning-rate horizon 130,398, checkpoint
every 504,832 transitions, 66,763,776 transitions. GO needed T >= 0.50, Q >= 0.50 and E <= half of S's error.
Ratio, capacity, horizon and length changed together, so we cannot say which one mattered. One seed each.

Hand clearance (CPU, M0b preparation): opposing hand groups sit within 1 cm of the object's surfaces in
259 of 390 reference frames, from frame 28. Both hands rise during the carry.

Lesson from D1: size a run from measured end-to-end training seconds per transition, times 1.1. Never from
simulator-only speed. `_sizing.py` does this.

## How to run

FlashCHORD interpreter for training and evaluation; project interpreter for CPU work. Each operation has its
own flag in `runtime.json`; all default to off. Add `--dry-run` to M0b/M0c commands to print the exact commands
without running anything.

```bash
# D0 and D1
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/flashchord/venv/bin/python -B -m src.features.dynamic_feasibility
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/flashchord/venv/bin/python -B -m src.features.dynamic_feasibility --queue src/features/dynamic_feasibility/scaling_queue.json
# Re-render a past campaign on CPU (no training)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.dynamic_feasibility --queue src/features/dynamic_feasibility/scaling_queue.json --render-only data/.runtime/tmp/dynamic-rerender
# M0
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/flashchord/venv/bin/python -B -m src.features.dynamic_feasibility.partial_eval
# Hand clearance (CPU)
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 data/.runtime/linux-cuda/bin/python -B -m src.features.dynamic_feasibility.hand_clearance
# M0b, in order
PYTHONPATH=. data/.runtime/flashchord/venv/bin/python -B -m src.features.dynamic_feasibility.m0b smoke   # then probe, select, train, score, contact
# M0c
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/flashchord/venv/bin/python -B -m src.features.dynamic_feasibility.m0c train
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/flashchord/venv/bin/python -B -m src.features.dynamic_feasibility.m0c score
```

`train` detaches and writes `launch.json`. Wait for `train-finished.json` before `score`. Failed training is
never retried, and an existing export is never re-run. Past campaigns are render-only on current code: their
recorded source hashes no longer match, and the code refuses to resume them. `m0c --check-m0b-identity`
reports this for M0b.

## Outputs

Everything goes under `data/outputs/dynamic_feasibility/`: D0 at the top level, then `d1/sizing-fixed/`, `m0/`,
`m0b/` (with `m0b/clearance/`) and `m0c/`. Each holds `run.json` records, a `ledger.jsonl`, `summary.md`,
`final.json`, `audit.json` (an independent recomputation) and `finished.json`. Retained policy files are in
each stage's `retained/`. Old records are never rewritten.

Protocols and outcomes: `docs/chain/dynamic-feasibility-*.md`, `dynamic-scaling-*.md`, `moving-m0-*.md`,
`moving-m0b-*.md`, `moving-m0c-protocol.md`.

## Code

`_metrics.py`, `_sizing.py` and `_render.py` are pure maths. `handler.py` runs admission, the GPU lease and
subprocesses. `measure.py` exports and scores on CPU. `loop.py` runs queues, resume and the ledger. `_worker.py`
and `_m0b_worker.py` run unchanged upstream training with a cooperative deadline. The contact diagnostic was
skipped: reliable force and slip measurement did not fit its 20-minute build allowance.
