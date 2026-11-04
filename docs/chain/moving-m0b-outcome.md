# M0b outcome (4 Nov 2026, about 04:50 UTC): GO-1

Protocol `docs/chain/moving-m0b-protocol.md`, committed `9bbd221` before any training number. Evidence: `outputs/dynamic_feasibility/m0b/` (`summary.md`, `final.json`, `audit.json`, `selection.json`, `trials/train/run.json`, `scoring/runs/`).

**A finished, unassisted curriculum tracks the moving box through the official evaluator.** One seed, FlashCHORD's shipped `tissue_box_simple` task with its own reference and mesh. **This is not a challenge result and nothing here enters a submission.**

| Arm | T | Q | Perpendicular cm | Position cm | Max displacement cm | ADD-AUC | Learner updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S (held still) | 0 | 0 | 0 | 6.889 | 0 | 0.4591 | none |
| D1p seed 0, M0 | 0.0352 | 0.0751 | 0.478 | 6.803 | 2.03 | 0.4438 | 2,632 |
| M0b boundary, 32,814,080, saved at VOC 0.01, descriptive | 0.9856 | 0.8657 | 0.549 | 0.953 | 17.40 | 0.9635 | 24,566 |
| **M0b final, 66,763,776, VOC 0, gated** | **0.9874** | **0.8751** | **0.339** | **0.562** | **17.22** | **0.9808** | **90,874** |

GO-1 needed `T >= 0.50`, `Q >= 0.50`, `E <= 3.44 cm`. The reference travels up to 17.38 cm.

**Recipe the rule chose:** ratio 2.0, replay capacity 1,048,576, learning-rate horizon 130,398, checkpoint interval 504,832. It finished in 35.7 GPU-minutes against a prediction of 35.9, resuming D1 seed 0 from 21,575,680.

## Checks

- **All five of M0's validity checks passed,** and no jump-and-hold control met a gate.
- **The lead's separate script,** run on the raw parquets, reproduces `T`, `Q` and position error, and confirms the ways this could have been false did not occur:
  - Not a replay: residual from the reference ranges from 0.01 to 2.23 cm, with no frame exactly equal to it.
  - Assistance really off: the evaluator forces `voc_scale`, `max_force` and `max_torque` to 0 (`harness.py:230-233`), and its hash equals the kit's.
  - A working control: the same evaluator path gave Z, L1 and D1p at most 2.03 cm on this task.
  - Checkpoint identity: the hashes in the record match the retained files.

## Reading

- **The earlier NO-GO reflected the budget, not the method.** D1 had 2,632 learner updates, and the boundary policy has 24,566. **Ratio, replay capacity, horizon and transitions all changed together, so this does not say which one mattered,** only that the old recipe was far short.
- **Most of the gain arrives early.** The boundary policy, 11.2M transitions and about 14 minutes after the resume, is already at `T` 0.986 and AUC 0.963. It had a warm start from D1's 21.6M transitions, so a fresh run's time is not yet measured.
- **Throughput was also misjudged.** Probes measured 26,189 to 35,495 transitions/s against D1's 13,000, at far higher update ratios. D1 saved a 158 MiB checkpoint pair every five collections; this run saved every 493 collections. That fits the difference, but no run has isolated it. Either way, the plan's budgets were pessimistic: a 66.76M schedule takes about 43 minutes at the measured rate, not 95.
- **This may bear on the leaderboard's 0.99, but nothing here shows it.** The method behind that score is still unknown. This task's reference and mesh are high quality, and 0.98 here says nothing yet about a challenge episode.

## Limits

One seed, with a warm start. The contact diagnostic was skipped within the protocol's allowance, so whether the hands hold the box as a person would is not measured; the box's motion is. These metrics see position; orientation error here is 0.50 degrees, which is reported beside them.

**The biggest unproven link is unchanged, and it is now the whole question:** transfer to a challenge episode. That needs task assembly, an embodiment, a video-derived reference of good enough quality, and physical tracking of a noisier reference than this one.

## Consequence, by the protocol

Learning moves the object on the hand reference. M2 and M3 run on this recipe under their own gates; M1 becomes eligible. `COMPLETION-PLAN.md` section 5 is revised accordingly.
