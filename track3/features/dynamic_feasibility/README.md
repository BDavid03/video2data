# Dynamic feasibility D0

Run the committed `docs/chain/dynamic-feasibility-protocol.md` on the unchanged
shipped `tissue_box_simple` task. This diagnostic reads no challenge episode and
produces no submission. Task data, native FlashCHORD and kit evaluator remain
read-only. The source task has 390 frames at 30 Hz.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/flashchord/venv/bin/python -B -m track3.features.dynamic_feasibility
```

For the complete campaign, launch the command with `setsid nohup` and redirect
its log to `outputs/dynamic_feasibility/loop.log`. `finished.json` is its finish
signal. `queue.json` contains the Z/L1 budgets and the predeclared L2 allocation
rule. Completed or failed trials are retained, hashed and never retried. Exact
resume requires unchanged task, protocol, runtime, queue and Python sources.

Z runs one 64-world collection and must save zero optimizer updates. L1 uses
the existing 64-world, 128,000-transition contact-feasibility recipe, seeds 0/1.
The released network, mixed precision, collision/support settings, 512 `njmax`
and 256 `nconmax` stay at their defaults. Learned curriculum thresholds are
rescaled; no task resampling or evaluator assistance is enabled.

L2 requires the better completed L1 seed to improve mean position error over Z
by at least 20%. It retains L1's memory settings, allocates at most 45 minutes
across both seeds, and chooses the largest multiple-of-64 transition budget
predicted to fit using the slower L1 training time per transition, a 10% time
margin and an evaluation reserve. These rules are frozen before the first run.
Each initial arm has a 15-minute ceiling; all GPU lease time, including
compilation, counts against the 75-minute campaign ceiling.

`_worker.py` runs the unchanged upstream scripts through `runpy` with a
cooperative Python-event deadline. It changes no training or evaluator code and
sends no process signals. A long native/CUDA call cannot be preempted until it
returns to Python. Evaluator provenance continues to hash its original script.

`_metrics.py` contains only pure array statistics and budget arithmetic. The
handler owns input hashing, the GPU lease, subprocesses and output records;
`loop.py` owns queue/resume, the append-only ledger and the protocol verdict.
Kit scoring uses the existing perception environment on CPU because the
FlashCHORD environment has no pandas. `measure.py` owns this I/O boundary;
training and official evaluation use the FlashCHORD environment throughout.

Mean/p90 position and mean/p90 orientation are raw world-frame errors. ADD-AUC
uses the kit's 1–9 cm function after its initial slot-0 rigid alignment; raw
ADD-AUC is also saved. The point set is 1,000 seeded surface samples of the
shipped task mesh. Comparator S holds its true initial pose over all 390 frames.
These are sample-task diagnostic measurements, not challenge scores.

Outputs are under `outputs/dynamic_feasibility/`: `campaign.json`,
`comparator_S.json`, `ledger.jsonl`, `l2_decision.json`, `summary.md/json`,
`finished.json`, and immutable `runs/<arm>-seed<seed>/` records, logs,
checkpoints, official trajectories and metrics. Peak GPU memory is sampled
device-wide use; it includes display allocation and may miss brief peaks.

GO requires one learned arm on both seeds to complete finite full rollouts and
achieve mean position error at most 3 cm and at most half S. A failed arm or
budget exhaustion cannot promote a candidate.

## Measured result: NO-GO

The campaign completed on 3 November 2026 in **271.04 GPU seconds (4.52 min)**,
including compilation. Every rollout is finite and covers all 390 frames at
30 Hz. No settings changed after launch. The unchanged evaluator code digest is
`b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`.

| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | P90 ori deg | ADD-AUC | Wall s | Peak device MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S, computed | — | 6.889343 | 16.548991 | 2.217122 | 3.750897 | 0.459135 | — | — |
| Z | 0 | 7.011673 | 16.266067 | 3.322483 | 5.235784 | 0.428045 | 119.34 | 3183 |
| L1 | 0 | 7.025931 | 16.395528 | 3.414495 | 5.258379 | 0.427564 | 77.27 | 2915 |
| L1 | 1 | 6.953809 | 16.351698 | 2.150268 | 3.794408 | 0.447115 | 75.52 | 2915 |

Z saved zero learner and actor updates. The object moved at most **1.499 cm**
from its first exported pose, so the hand reference alone did not reproduce the
roughly 17 cm demonstration motion. L1 performed **237 learner / 119 actor Adam
updates per seed** and finished at zero VOC assistance. Its better seed improves
mean position over Z by only **0.8252%**, below the protocol's 20% L2 gate;
therefore L2 was not run. Neither learned seed meets the 3 cm GO limit.

`audit.json` independently recomputes all position/orientation and ADD-AUC
statistics from the real exports, checks all 85 campaign input hashes, and
reproduces NO-GO. Position and ADD-AUC match exactly; the maximum mean-angle
difference against SciPy is 2.1e-9 degrees. A second queue invocation resumed
all three trials, appended no ledger rows and launched no new GPU work.
`equivalence.json` separately compares S with independent NumPy/SciPy statistics.

This bounded policy stage did not establish moving-object tracking in time for
the pack deadline. The protocol directs subsequent effort to whole-clip meshes;
the result does not establish that longer or different manipulation training
would fail. No contact-buffer overflow warning was found in these logs; the
runtime's automatic robot-inertia correction warnings are retained.
