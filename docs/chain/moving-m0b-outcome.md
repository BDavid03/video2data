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

- **The revised recipe succeeds where the earlier, incomplete one failed.** D1 never ran its full schedule, so it was never a test of that recipe. At the boundary, transitions are 1.5 times D1's and learner updates 9.3 times. **Ratio, replay capacity, horizon and transitions all changed together, so this does not say which one mattered.** M0c tests the warm start.
- **Most of the gain arrives early.** The boundary policy, 11.2M transitions after the resume, is already at `T` 0.986 and AUC 0.963. It had a warm start from D1's 21.6M transitions, so a fresh run's time is not yet measured.
- **Throughput was also misjudged.** Probes measured 26,189 to 35,495 transitions/s against D1's 13,000, at far higher update ratios. D1 saved a 158 MiB checkpoint pair every five collections; this run saved every 493 collections. That fits the difference, but no run has isolated it. **This run itself averaged 21,100 transitions/s** (45.19M in 35.7 minutes), so the 1.25 planning margin was all used: a full 66.76M from nothing would take about 53 minutes, not the probe's 43. That is still far below the plan's earlier 95.
- **This may bear on the leaderboard's 0.99, but nothing here shows it.** The method behind that score is still unknown. This task's reference and mesh are high quality, and 0.98 here says nothing yet about a challenge episode.

## M0c: the same recipe from nothing, on seed 1 (`moving-m0c-protocol.md`): GO-1

Evidence: `outputs/dynamic_feasibility/m0c/` (`summary.md`, `final.json`). Fresh start, no resume, seed 1, 130,382 learner updates. All five validity checks passed; the lead's separate script reproduces `T`, `Q` and `E` from the parquets.

| Checkpoint | Saved VOC | Wall minutes from start | T | Q | Position cm | ADD-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,154,624 | 0.25 | 15.0 | 0.9683 | 0.8315 | 0.940 | 0.9606 |
| 32,814,080 | 0.01 | 26.6 | 0.9930 | 0.8607 | 0.605 | 0.9798 |
| **66,763,776, gated** | 0 | 49.3 | **0.9917** | **0.8715** | **0.516** | **0.9790** |

**So the recipe does not rely on D1's prefix,** on this task. The earliest row meeting GO-1's thresholds, by the protocol's budget rule, is 16,154,624, 15 minutes including startup, though it was still training with assistance at VOC 0.25; the evaluator removed that assistance for its score. **Two seeds now agree, on an easy task.** Whether a harder task finishes this early is unknown: 39 moves about eight times faster and rotates 109 degrees.

## Limits

Two seeds, one with a warm start. The contact diagnostic was skipped within the protocol's allowance, so whether the hands hold the box as a person would is not measured; the box's motion is. A palm-under lift would satisfy every metric here. These metrics see position; orientation error here is 0.50 degrees, reported beside them.

**This task is easy in ways challenge episodes are not (Claude critic 25).** 93.7 % of the reference's displacement energy is vertical, its peak speed is 9.08 cm/s, and it rotates only 5.4 degrees. The held-still arm already has AUC 0.459. Early frames are not tracked: at frames 0 to 60 the box moves 0.31 cm against 1.55 cm, and the largest error is at frame 40. Tuning episode 39, by comparison, travels 39.1 cm, peaks at 71 cm/s and rotates 109 degrees, so GO-1 here predicts little about it.

**The biggest unproven link is unchanged, and it is now the whole question:** transfer to a challenge episode. That needs task assembly, an embodiment, a video-derived reference of good enough quality, and physical tracking of a noisier reference than this one.

## Consequence, by the protocol

Learning moves the object on the hand reference. M2 and M3 may use this recipe under their own gates; M1 becomes eligible. **M2's Stage 0 has since stopped** (see `moving-m2-protocol.md`, "State"), so a challenge-type task does not yet exist. `COMPLETION-PLAN.md` section 5 is revised accordingly.
