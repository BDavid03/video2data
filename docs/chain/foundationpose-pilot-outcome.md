# FoundationPose pilot outcome (4 Nov 2026, about 05:20 UTC): PASS, on two used train pairs

Protocol `foundationpose-pilot-protocol.md`, with Amendment 1 committed `933863e` before any pose existed. Evidence: `outputs/foundationpose/pilot/` (`summary.md`, `summary.json`, `predictions.json`, `run.json`, overlays under `amendment1/`). Install: `docs/INSTALL.md`, noncommercial licence recorded. Nothing here changes a submitted file. These are scan-based measurements; scans enter no mesh or simulated asset.

| Pair | Objects | `b` error cm | `C` error deg | Gate (3 cm, 30 deg) |
| --- | --- | ---: | ---: | --- |
| 12 | `blue_cup`, `plastic_dish_rack` | 1.810 | 2.36 | PASS |
| 31 | `mini_sweeper`, `plastic_dish_rack` | 1.231 | 8.83 | PASS |

**Kinematic, descriptive, with a T1 slot 0 anchor:**

| Pair / method | AUC | SP-SR | RPE cm | MPPE cm |
| --- | ---: | ---: | ---: | ---: |
| 12 / T1 (the panel's arms leave 12 at T1) | 0.1041 | 0 | 35.02 | 22.14 |
| 12 / FP rotations, TW4's positions | 0.3854 | 0 | 18.64 | 9.64 |
| 12 / FP full relative pose | **0.5729** | 0 | 18.45 | 8.21 |
| 31 / T1 | 0.098 | 0 | 30.38 | 19.60 |
| 31 / panel TW4 | 0.3171 | 1 | 10.08 | 6.96 |
| 31 / FP rotations, TW4's positions | 0.3967 | 1 | 9.56 | 6.09 |
| 31 / FP full relative pose | **0.5046** | 1 | 10.20 | 5.07 |

## Checks

- **Order:** predictions sealed 05:14:11 UTC (sha256 `1dec4f02...`), first mocap access 05:16:04.
- **Independent:** the lead's separate script, from the sealed file and raw parquets with its own formula, gives 1.8102 cm / 2.355 deg and 1.2313 cm / 8.825 deg, the same as recorded. The predicted offsets are 19.39 and 11.97 cm against 19.04 and 11.45 cm, and identity would be 86.5 and 98.3 degrees wrong, so this is not a trivial pass.
- **Overlays:** Codex and the lead looked. Cup and rack sit closely on the image; the sweeper is the least convincing, with its bristle end narrower and offset.

## Disclosures

- **Used pairs:** this is a screen, not validation. Train 31 was in the panel and 12 in earlier work; no untouched train pair remains.
- **Two earlier crashes:** the cup's first run and permitted retry crashed on a dtype mismatch (float32 intrinsics, float64 points) before any pose existed. Amendment 1 allowed one type-only fix and one more attempt, which is what ran. Each other object ran once.
- **Flips persist (Claude critic 25, from `predictions.npz`).** The recorded top-hypothesis margins, 0.117, 0.004, 0.000 and 0.000, mislead, because the runners-up are the same basin at 0.5 to 1.3 degrees. The true rival gaps are small: cup 0.19 % at 120 degrees, rack on 12 0.95 % at 174, sweeper 0.25 % at 179, rack on 31 0.37 % at 174. **So 31 avoided its flip here, and FoundationPose has not been shown to resolve flips.** On 31, `C` of 8.8 degrees barely beats the panel's TW at 9.1; the sweeper had 1,493 points and silhouette IoU 0.68.
- **Resources:** 11 to 73 seconds per `register`, 8.6 GiB Torch peak, about 16 GiB sampled device peak including the display. It is tight, and its guard stopped four times, after each output; fresh processes ran only uncompleted objects.

## What it may mean

Registration failed its panel with a rival within 0.117 % (`frame0-panel-outcome.md`), and HR1 found cup heading unrecoverable from stereo shape. On these two pairs, image evidence plus a model reached under 2 cm of offset error and under 9 degrees, including on one cup-anchored pair, of HR1's four cups. If a full panel confirmed this, it might reach cup-anchored pairs at a level the tolerance curve says pays. A static result still stops short of a moving one: the ceiling stays O3's 0.532 on train.

## Consequence, by the protocol, and the next step

The protocol's next step is a full panel. **The plan defers it:** it is static, its physical admission is unsolved (the panel's episode 30 failed by falling over at its measured offset), and W3 is the main line. **Tracking is a candidate for M1 and has never been run.** Using scan-tracked trajectories on a scored episode also needs the owner's answer first (`NEEDS-YOU.md`, top): the scan line names rotation training and H3, the FAQ speaks of reconstructing objects to generate trajectories, and this model's licence is noncommercial against the Apache-2.0 the challenge asks of winning code.
