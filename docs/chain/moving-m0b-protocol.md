# M0b protocol: does a finished, unassisted curriculum move the box? (`COMPLETION-PLAN.md` W3; fixed before any training number below exists)

## Question

M0's two partial policies moved the box at most 2.03 of 17.38 cm (`moving-m0-outcome.md`), but they had 2,632 learner updates and training assistance still at VOC 0.05. M0b finishes one policy and asks whether it transports the object. **Scope: FlashCHORD's shipped `tissue_box_simple` task, its own reference and mesh, one seed. It opens no challenge episode, it is not a challenge result, and it produces nothing for a submission.** The biggest unproven link stays transfer from this task to a packable challenge policy.

## What was already known when this was written (disclosure)

- M0's table, and its reading: `T` 0.035 / 0.009, `Q` 0.075 / 0.009.
- `outputs/dynamic_feasibility/m0b/clearance/` (CPU, hashed): the **reference carries the box with a geometrically plausible grasp**, opposing contact groups in 152 of 161 frames above 5 cm, hand and box heights correlated 0.997. The earlier critic's "bump" reading is unsupported for the reference. Whether any policy grasped is unknown, because policy hands and contacts were not exported.
- `resume-facts.md` in the same directory, source-read and unrun: resume restores learner state; replay starts empty, so the first update comes after 9,216 new transitions; the 32,814,080 checkpoint is saved at VOC 0.01 with no completed collection at VOC 0, and the first checkpoint with VOC 0 metadata is 32,819,200; 66,763,776 is an explicit final save at VOC 0.
- **Two variables hidden inside D1's budget:**

| Variable | D1 | Upstream `sharpa_flash_sac` |
| --- | ---: | ---: |
| `updates_per_collection` | 0.125 | 8.0 |
| Updates over a whole run | 8,149 | 488,288 |
| Replay `capacity` | 16,384 (16 collections) | 4,194,304 (1,024 collections) |
| Replay `minimum_size` | 6,400 | 409,600 |
| Worlds | 1,024 | 4,096 |

Neither ratio nor capacity has ever been varied here. A resumed run must keep 1,024 worlds. Replay is empty on resume, so capacity is free to change.

## Fixed recipe

D1 seed 0's state `.runtime/tmp/dynamic-scaling-seed0-k62pzjmq/training/state_21575680.safetensors`, sha256 `33056ac8d5fa7196...`; its copy is hashed into the record, because `.runtime/tmp` is scratch. D1's frozen curriculum thresholds and total of 66,763,776 are kept (`outputs/dynamic_feasibility/d1/sizing-fixed/selection.json`). Unchanged: 1,024 worlds, batch size, network, the shipped task, seed 0, `save_replay=false`, `minimum_size` 6,400, `n_step` 3, `warmup_action=policy`. **Learning-rate horizon, pinned explicitly including in the smoke and probes:** {0.5: 32,599; 1.0: 65,199; 2.0: 130,398}, upstream's `floor(total / worlds * ratio)` at the final total. Restored optimizer counters continue, so the critic's learning rate moves from 0.00026459 to about 0.00030 at ratio 2. The schedule runs 3e-4 to 1.5e-4, and D1's 8,149-update horizon would expire early in any higher-ratio run. **Checkpoint interval 504,832 for probes and training:** it divides 32,814,080 exactly, and the final save is unconditional. D1's 5,120 would write a 158 MiB pair every five collections, about 1.36 TiB over the rest of this run, so that is a defect in carrying D1's value forward and not a choice. Disclosed: ratio, capacity, horizon and interval all change together. **M0b finishes a policy on the best recipe the hardware admits; it does not isolate any one of them.**

## Stage 1: resume smoke (under 2 GPU-minutes)

Ratio 0.5, capacity 16,384, total 21,584,896 (start + 9,216), checkpoint interval 5,120, no evaluator, new directories. Passes only if the final state shows `environment_steps` 21,584,896, `global_update_step` 2,633, `update_credit` 0.25, stage 5, VOC 0.05, the first progress line starts at 21,575,680, and actor and critic Adam counters and parameters advanced. **Failure stops M0b:** recorded, resume is out, and M2 must plan a fresh schedule.

## Stage 2: throughput probe (at most 20 GPU-minutes)

Four arms, each independently resuming the same state, sequentially, in new directories, under `GpuLease`. An arm runs until its replay ring has been full for 60 seconds, or to a 300-second cooperative deadline, whichever comes first. No policy is evaluated and no quality is read.

| Arm | Ratio | Capacity |
| --- | ---: | ---: |
| a | 0.5 | 16,384 |
| b | 2.0 | 16,384 |
| c | 0.5 | 1,048,576 |
| d | 2.0 | 1,048,576 |

The largest capacity is upstream's replay horizon in collections, at our world count; replay is allocated at once on the GPU, about 2,366 MiB, so memory is visible at the start. Each arm records: steady transitions/s (over the final 60 seconds, or from 60 seconds to the end if the ring did not fill, stating which), compile seconds, updates done, and peak sampled device MiB. A failed arm (out of memory, crash) is an inadmissible arm, not a stop.

## Rule for what runs (mechanical; written before the probe)

1. **Capacity:** 1,048,576 if at least one of c and d completed its probe with peak at most 13,824 MiB (the limit D1 sized against), else 16,384.
2. **Ratio:** from the chosen capacity's two arms, fit per-collection time `t(r) = a + b r`. Predicted hours are `1.25 * 45,188,096 / 1024 * t(r) / 3600`; 1.25 is a planning margin, not a guarantee, because a short probe cannot see later curriculum stages or checkpoint cost. Choose the largest `r` in {2.0, 1.0, 0.5} with predicted time at most **100 minutes**. A ratio is excluded if its arm failed. 1.0 is interpolated, not probed, and only between two passing arms.
3. **If even 0.5 misses 100 minutes:** run 0.5 with a 100-minute deadline, score the last complete checkpoint, and label the result incomplete. Assistance is still removed by then if the rate reaches 1,873 transitions/s.
4. **No arm passed at all:** M0b stops, recorded.

## Stage 3: the training run

One uninterrupted resume to 66,763,776, detached, under `GpuLease`, with a hard 100-minute cooperative deadline and new output directories. Retained, with hashes: the complete policy and state pairs at 32,814,080, and at 66,763,776 or the last complete pair. Launch no later than 05:00 UTC. **No other GPU operation runs meanwhile.** The evaluator runs only after training ends.

## Stage 4: scoring

Each retained policy runs once through the kit's `evaluate_policy.py`, using `partial_eval.py`'s path and settings (`evaluation.world_count=1 evaluation.reset_mode=explicit evaluation.start_frame=0`), with M0's metrics and M0's five validity checks; check 5's controls are the same reference and need no recomputation beyond a hash match. A run that crashes without an export may be retried once, disclosed. A run with an export is never repeated. Intermediate checkpoints are not scored, so there is no best-of selection. The 32,814,080 policy is labelled "boundary, saved at VOC 0.01"; the evaluator removes assistance itself.

## Gate (on the final policy; the 32,814,080 row is descriptive)

`E_S` is S's mean position 6.889 cm at full precision, `E` the policy's mean position error, and `T_base = max(T_Z, T_L1-seed0, T_L1-seed1, T_D1p-seed0, T_D1p-seed1)` = 0.0352.

- **GO-1**: `T >= 0.50`, `Q >= 0.50`, `E <= 0.5 E_S`. M0's GO required both seeds; this has one, so it is named differently and carries that caveat.
- **SIGNAL**: `T >= max(0.20, T_base + 0.10)`, `Q >= 0.10`, `E <= E_S`.
- **NULL**: anything else. NULL means this screen was not met on this one run.

Raw values, no rounding, no extra tolerance. Beside the gate: each row's perpendicular error, largest displacement, orientation error, ADD-AUC, the ratio, capacity and total updates, and the training curve's own success fields if logged.

**Contact diagnostic, outside the gate, so that a NULL can be read.** After the official evaluations, one separate rollout of the final policy with the same settings, from an owned instrumented copy of the evaluation loop, never the kit's file, logging opposed force-bearing hand-object contact dwell during the reference's lift and relative slip. It is labelled a diagnostic, it feeds no gate or metric, and it is not an official export. If it would take more than about 20 minutes of work or about 2 GPU-minutes, it is skipped and that is said.

## What each result does

| Result | Consequence |
| --- | --- |
| GO-1 or SIGNAL | Learning moves the object on the hand reference with these settings. M2 runs on this recipe, or on a compressed version if time forces it, against the same hand-reference result. M1 and M3 become eligible under their own gates |
| NULL, run complete | A finished, unassisted curriculum at this recipe earned no credit. M2 still tests the object-driven reference once, on this recipe; if that also fails, W3 closes for this deadline with the finding in the report |
| NULL, incomplete | As above, and the report says the curriculum did not finish |

**Budget note.** A full `M2` arm costs what Stage 3 cost, so M0b's measured time is M2's budget; the last GPU launch is 10:30 UTC.

## Independent check

If the result is GO-1 or SIGNAL, the lead recomputes `T`, `Q` and perpendicular error from the parquet with a separate script before believing it (`PROGRAM.md` section 4). The outcome goes in `docs/chain/moving-m0b-outcome.md`, whichever way it falls.

## Integrity

No challenge episode, no labels of scored episodes, no scan. No pack manifest, `AGENTS.md` or recorded `run.json` is touched. New files under `outputs/dynamic_feasibility/m0b/`, and a new `M0b` block in the slice's `runtime.json`. Nothing here enters a submission; a later pack would need its own gate.
