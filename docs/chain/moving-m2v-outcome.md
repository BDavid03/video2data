# M2v outcome (4 Nov 2026, 10:35 UTC): GO with the scorer's alignment, after the first verdict read NULL on raw frames

Protocol `moving-m2v-protocol.md`, fixed before any M2v number. Evidence: `data/outputs/moving_policy/ep39-v1/`. Seed 0, fresh start, 66,763,776 transitions, 3,694 s of training (09:25 to 10:27 UTC). **A learned policy trained in simulation by RL on M1's scan-free video path, moving our own reconstructed pan through the kit's unedited evaluator, on tuning episode 39. Not a challenge result; 39 is a practice episode.**

| Against | `T` | `Q` | `E` cm | `Theta` deg | Kit AUC | SP-SR | MPPE cm | Limits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Its own reference | 1.001 | 0.974 | 1.21 | 1.49 | 0.936 | 1 | 1.27 | `T`, `Q` 0.50; 3 cm; 15 deg |
| Mocap, raw frames: **the code's recorded verdict, NULL** | 0.055 | -0.880 | 21.31 | 176.87 | 0.775 | 1 | 2.73 | |
| **Mocap, scorer's single first-pose alignment: GO** | **1.003** | **0.945** | **2.67** | **8.31** | **0.775** | 1 | 2.73 | `T`, `Q` 0.50; `E` at most 3 cm; `Theta` at most 15 deg; AUC 0.45 |

End reached, no reset. Reprojection on its own run: median IoU 0.80, p10 0.66, 319 of 319 frames, which is blind to camera drift. Kill check at 32.8M: self-`T` 1.01. Beside it: held still 0.18, the path alone 0.833, the same recipe on mocap 0.982, and the allowance screen's forecast 0.80. **A policy following a video path loses about 0.06 of AUC from the path alone.** 2.67 cm against a 3 cm limit: passing, not by much.

## The NULL, read this

The handler computed `T`, `Q`, `E` and `Theta` against mocap on raw arrays, as it had on mocap-39, where the policy's world **was** mocap's world. On M2v the world is M1's, so the same arrays compare poses across two unrelated world frames: the 176.87 degrees and -0.88 are coordinate-dependent discrepancies, not tracking fidelity. The kit's own AUC aligns by itself, so its unchanged 0.7749 (`mocap_metrics_aligned.json`) says little on its own. The aligned arithmetic was reviewed line by line by a fresh Codex critic (`critic2-27`): **GO stands, as a disclosed retrospective correction.** Alignment hides initial absolute placement and orientation error, which has its own adoption gate, and cannot hide drift, scale or timing error. The critic's sentence for the abstract: "M1 passed only amended gates and used a public-scan body convention; M2v's initial NULL became GO after a retrospective first-pose alignment correction, with label evaluation confined to tuning episode 39 and episode 38 assessed only by video and physical gates."

**A risk the protocol had not named (critic2-27): there is no label-based noninferiority guarantee.** 38 could score below its static v5 export while passing every gate. The evidence from 39 only suggests otherwise: held still 0.18, moving 0.775.

**This is a defect in comparing frames, not a gate moved: every numeric limit is what the protocol wrote.** But the protocol's text said "against mocap" and did not say "aligned"; the alignment is taken from M1's protocol and the kit's scorer, after seeing the NULL. It means reading the same export and mocap a second time: "scored once" is true of the policy, not of the arithmetic. `mocap_metrics.json` and `result.md`'s NULL are kept unchanged.

## Consequence

The route to 38 stays open. 38 launched at about 10:33 UTC, three minutes after the plan's 10:30, because the job stopped on the raw NULL. Adoption by `moving-m2v-protocol.md`, "Scored 38".
