# M0c protocol: the same recipe from nothing, on a new seed (`COMPLETION-PLAN.md` W3; fixed before any number below exists)

## Question

M0b's GO-1 used one seed and a warm start from D1's 21.6M transitions. Every later policy, on a new task, starts from nothing. M0c asks two things the plan's budget depends on: does this recipe reach transport without the warm start and on a different seed, and how soon? **Scope: shipped `tissue_box_simple`, one seed; it is not a challenge result and nothing here enters a submission.**

## What was already known when this was written (disclosure)

M0b's table (`moving-m0b-outcome.md`), in full, including its boundary row; D1's training records; the probe rates. So thresholds below are not blind to one seed's outcome.

## Fixed recipe

M0b's selected recipe, unchanged: 1,024 worlds, ratio 2.0, capacity 1,048,576, `minimum_size` 6,400, `n_step` 3, `warmup_action=policy`, learning-rate horizon 130,398, checkpoint interval 504,832, D1's thresholds, total 66,763,776, the shipped task. **Seed 1.** No resume: a fresh start. A hard 75-minute cooperative deadline; on a miss, the last complete checkpoint is scored and labelled incomplete. Launched only when no other GPU job runs. No retry with different settings.

## Scoring

Three checkpoints, fixed now, each once through the kit's evaluator with M0's settings, metrics and validity checks: **16,154,624** (32 x 504,832, stage 3, training VOC 0.25), **32,814,080** (65 x 504,832, the boundary, saved at VOC 0.01), **66,763,776** (the final, VOC 0). Each row is labelled with its saved VOC, and the evaluator removes assistance. No other checkpoint is scored, so there is no best-of selection.

## Gate (on the final policy; the earlier rows are descriptive)

Thresholds as in M0b: `T_base` and `E_S` from M0's table.

- **GO-1**: `T >= 0.50`, `Q >= 0.50`, `E <= 0.5 E_S` (3.44 cm).
- **SIGNAL**: `T >= 0.20`, `Q >= 0.10`, `E <= E_S`.
- **NULL**: anything else.

Beside the gate: both seeds' rows together, as this deadline's seed spread (two seeds, different starts, so a spread and not a variance).

## What each result does

| Result | Consequence |
| --- | --- |
| GO-1 | The recipe does not rely on D1's prefix. Challenge policies budget from the earliest row that itself meets GO-1's thresholds, with its minutes, plus a margin |
| SIGNAL or NULL | M0b's result depends on the warm start or on the seed. The report says so. A challenge route must plan for it: full schedule at least, or warm start where contracts admit one |

The lead independently recomputes `T`, `Q` and `E` from the parquets for any row used in a budget. The outcome is appended to `moving-m0b-outcome.md`.

## Integrity

No challenge episode, no scan, no pack-pinned file, no recorded `run.json` is touched. New files under `outputs/dynamic_feasibility/m0c/`.
