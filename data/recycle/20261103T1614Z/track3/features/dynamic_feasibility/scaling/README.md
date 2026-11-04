# Dynamic scaling D1

Run `docs/chain/dynamic-scaling-protocol.md` on the unchanged shipped example,
using D0's task loader, training command, checkpoint admission, official evaluator
and pure metrics. All original D0 source, queue and output files remain unchanged.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 setsid nohup .runtime/flashchord/venv/bin/python -B -m track3.features.dynamic_feasibility.scaling > outputs/dynamic_feasibility/d1.log 2>&1 &
```

The finish signal is `outputs/dynamic_feasibility/d1/sizing-fixed/finished.json`.
`scaling/queue.json` owns the distinct D1 entry. Before training, the probe
measures 256/512/1024 worlds, with 384/768 as declared fallback midpoints after
a failed upper probe. All sizing operations together have 300 GPU seconds.

Each probe builds the same captured environment and initializes the full L1
learner/replay state, retained during measurement. It performs zero optimizer
updates and never reads tracking-error, reward or pose-quality values. Throughput
is 64 synchronized zero-residual environment collections after eight warmups;
initialization/compilation counts against the probe time cap. This measures
environment throughput, not optimizer or checkpoint throughput. The <=13.5 GiB
ceiling applies to sampled total device memory, including display allocation.

The fastest fitting probe sets worlds and `floor(steps/s * 1350 / worlds) * worlds`
transitions. The selected values and scaled VOC curriculum are written into the
queue, `selection.json` and the append-only ledger before seed zero starts.
All other L1 settings remain fixed, including the 5120-transition checkpoint
interval. Intermediate checkpoints are written into newly owned scratch,
retaining the two most recent generations there to bound disk use; the final
policy/state files are copied unchanged into the immutable output record.

Seeds 0 and 1 each have at most 1800 GPU seconds including setup/evaluation,
within the total 3900-second cap. Timeouts exit cooperatively through D0's
unchanged deadline wrapper. Failed seeds are never retried with different settings.
Only final checkpoints are evaluated. CPU measurements use D0's measurement
function in the existing perception environment. No challenge episode is read.

Outputs live under `outputs/dynamic_feasibility/d1/sizing-fixed/`: campaign and prior-D0
hashes, probe records, frozen selection, ledger, per-seed records, final
checkpoints, official exports, and `summary.md/json`. Resume re-admits finished
operations and never reruns a failed seed. The source plan excludes only the
declared mutable `chosen` queue field from its identity.

GO requires both finite full 390-frame rollouts to achieve mean position <=3 cm
and <=0.5*S. The sizing decision uses only throughput and memory; S/Z are
reporting comparators.

The original three sizing attempts failed on a flattened-observation setup
error before any throughput measurements or training. They remain under
`outputs/dynamic_feasibility/d1/probes/`; their 83.21 GPU seconds count against
both original caps. The repaired worker uses the existing WarpRLEnv reset view,
and the separate `sizing-fixed/` records preserve the earlier source snapshot.
No model, physics, optimizer or task setting changed for this correction.

## Measured result: NO-GO from clock failures

The corrected probe measured the following rates with the L1 learner/replay
state resident and zero optimizer updates:

| Worlds | Environment steps/s | Peak device MiB |
|---:|---:|---:|
| 256 | 28,703.724 | 2620 |
| 512 | 39,570.707 | 3292 |
| 1024 | 49,455.158 | 4124 |

At **2026-11-03 14:48:28.272706 UTC**, before seed zero started at
14:48:28.280818 UTC, the queue and ledger froze **1024 worlds and 66,763,776
transitions per seed**. This reproduces the protocol's rate-times-1350 rule.
All six sizing attempts, including the setup failures, consumed **174.795 s**.

| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | P90 ori deg | ADD-AUC | Completion | Wall s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| S, computed | — | 6.889343 | 16.548991 | 2.217122 | 3.750897 | 0.459135 | Reference | — | — |
| Z, D0 | 0 | 7.011673 | 16.266067 | 3.322483 | 5.235784 | 0.428045 | 390/390 finite | 119.34 | 3183 |
| D1 | 0 | — | — | — | — | — | No final export | 1713.13 | 4710 |
| D1 | 1 | — | — | — | — | — | No final export | 1713.01 | 4714 |

Both seeds exited through the cooperative wall-clock guard before their final
checkpoints. Total GPU lease time was **3600.928 s / 60.015 min**, including
failed probes, setup and compilation, below the 65-minute cap. The two-seed
completion rule therefore gives NO-GO. Earlier checkpoints were not substituted
or evaluated. Independent pose re-measurement is unavailable for both seeds.

| Seed | Last complete saved transitions | Learner updates | Actor Adam steps | State finite | Training VOC |
|---:|---:|---:|---:|---|---:|
| 0 | 21,575,680 | 2632 | 1316 | Yes | 0.05 |
| 1 | 22,599,680 | 2757 | 1379 | Yes | 0.05 |

The budget-sizing implementation measured environment stepping without optimizer
or checkpoint overhead. Full training achieved roughly 12.9k–13.6k transitions/s,
well below the probe's 49.5k, so the chosen budget did not fit its intended time.
The unchanged 5120-transition checkpoint interval incurred substantial I/O even
though scratch retention bounded live disk use. This is a budget-sizing and
completion failure; D1 supplies no measured tracking-quality result and does
not establish that the policy cannot learn transport. A new capability protocol
would need end-to-end throughput, including learning and checkpoint writes.

`audit.json` independently reproduces sizing and selection ordering, verifies
the absence of both final checkpoints/exports, and confirms all **175 prior D0
files** and **91 campaign inputs** are unchanged. `partial_checkpoints.json`
hashes the retained finite states without scoring them. `resume_audit.json`
confirms unchanged run records/ledger and zero new GPU operations. No source or
training setting changed after the budget was selected. D0's original identity,
records and verdict remain unchanged.
