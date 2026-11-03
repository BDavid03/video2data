# Dynamic feasibility

D0 and D1 measure moving-reference tracking on the unchanged, shipped
`tissue_box_simple` task (390 frames at 30 Hz). They read no challenge episode
and produce no submission. The task, FlashCHORD and kit evaluator are read-only.

## D0: bounded feasibility

Run the fixed [D0 protocol](../../../docs/chain/dynamic-feasibility-protocol.md):

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/flashchord/venv/bin/python -B -m track3.features.dynamic_feasibility
```

`queue.json` fixes Z (zero updates), L1 (64 worlds, 128,000 transitions,
seeds 0/1), and the conditional L2 extension. L2 needs at least 20% improvement
over Z and uses the slower measured L1 training seconds/transition, multiplied
by **1.1**, with evaluation time reserved. The total GPU ceiling is 75 minutes.

Writes `outputs/dynamic_feasibility/{campaign,comparator_S,l2_decision,summary,finished}.json`,
`summary.md`, an append-only `ledger.jsonl`, and immutable `runs/<arm>-seed<seed>/`
records, checkpoints, logs and official exports. `finished.json` signals completion.

Latest result: **NO-GO**, 271.04 GPU seconds. S/Z mean position errors are
6.889343/7.011673 cm; L1 seeds give 7.025931/6.953809 cm and ADD-AUC
0.427564/0.447115. Both complete all 390 frames with 237 learner / 119 actor
updates. The best improvement over Z is 0.8252%, so L2 was not run.
The independent `audit.json` and `equivalence.json` reproduce the measurements;
`resume_audit.json` records zero new trials on the unchanged historical source.

## D1: scaling

The [D1 protocol](../../../docs/chain/dynamic-scaling-protocol.md) is now a
second queue for the same loop, handler, measurement boundary and trial ledger:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/flashchord/venv/bin/python -B -m track3.features.dynamic_feasibility --queue track3/features/dynamic_feasibility/scaling_queue.json
```

Writes `outputs/dynamic_feasibility/d1/sizing-fixed/`: campaign, frozen
`selection.json`, ledger, per-seed records, checkpoints/exports when complete,
`summary.{json,md}` and `finished.json`. The slice's single `runtime.json` keeps
D1's separate operation authorizations in `D1.execution`; its run flag is checked
before any campaign preparation or write. D0's top-level flags are unchanged;
D1 training still also requires D0's run flag, while D1 measurement uses only
its own measure flag. Both seeds have 1800-second allowances within
a 3900-second total cap. The common handler retains only recent checkpoints in
owned scratch and copies the final policy/state unchanged into the trial record.

**Correct sizing:** use measured end-to-end **training seconds per transition
multiplied by 1.1**, then floor the time-limited transition count to the measured
world-count grid. This includes learning, compilation and checkpoint I/O;
it must not extrapolate an environment-only rate to a new world count.
`_sizing.py` uses the slower replicate at each measured count. `training_runs`
provides the recorded measurements. D0 already used this conservative rule for
L2. A new selection requires a new fixed protocol/campaign; no new training was
performed during this refactor.

The historical `chosen` decision is retained exactly. Its probes timed only
environment stepping with learner state resident and zero optimizer updates:
28,703.724 / 39,570.707 / 49,455.158 steps/s at 256 / 512 / 1024 worlds. This
incorrectly sized 66,763,776 transitions per seed. Both seeds timed out before
a final checkpoint/export; **NO-GO by completion, tracking unmeasured**.
Total GPU time was 3600.928 s, including 174.795 s for all probes and setup
failures. Last saved steps were 21,575,680 / 22,599,680, with 2632 / 2757
learner updates. Actual training was about 12.9k–13.6k transitions/s.

Retired probe world counts, fallback, timing and method settings live only in
`scaling_queue.json`'s `D1.history.probe`. Identity reconstructs their original
plan shape so moving them does not change the queue's identity contribution;
they never control execution. The live `probe` block keeps only the memory
ceiling and training duration consumed by sizing. A saved `selection.json`
missing its `budget_frozen` ledger event is durably appended before training
resumes, without duplicating an existing selection key.

Original setup failures remain under `outputs/dynamic_feasibility/d1/` and
are charged to the corrected campaign. `audit.json`, `partial_checkpoints.json`
and `resume_audit.json` retain the original evidence. See the
[D1 outcome](../../../docs/chain/dynamic-scaling-outcome.md) for limitations.

## Historical rendering and resume

Re-render either historical campaign on CPU, without admitting changed source
hashes or invoking training (substitute the D0 queue to render D0):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.dynamic_feasibility --queue track3/features/dynamic_feasibility/scaling_queue.json --render-only .runtime/tmp/dynamic-rerender
```

The renderer reads the immutable ledger, comparators and frozen decisions.
D0/D1 `summary.json` and `summary.md` retain their original bytes. Historical
run records and ledgers are never rewritten. Execution resume still requires
matching task/protocol/queue/runtime/source hashes; this refactor changes those
hashes, so historical campaigns are **render-only** on the new source. A fresh
campaign needs its own authorized output identity; overwriting old records is
not supported. Long authorized campaigns should use `setsid nohup` with a log.

`_metrics.py`, `_sizing.py` and `_render.py` compute from plain values without
I/O. `handler.py` owns admission, GPU leases and subprocesses; `measure.py` owns
CPU export/kit I/O; `loop.py` owns queues, resume and durable ledger appends.
`_worker.py` runs unchanged upstream scripts with a cooperative deadline,
sending no signals. Native calls cannot be preempted until Python resumes.

GO needs both learned seeds to complete finite full rollouts with mean position
at most 3 cm and half S. Position/orientation are raw world-frame errors.
ADD-AUC uses the kit's initial slot-0 alignment and 1,000 seeded task-surface
samples. These are diagnostic sample-task measurements, not challenge scores.
Peak GPU memory is sampled total device use and may miss brief peaks.
