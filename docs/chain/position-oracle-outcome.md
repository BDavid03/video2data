# W1.0 outcome (4 Nov 2026, about 02:08 UTC): position alone wins no AUC; the gain is in the joint correction

Protocol `docs/chain/position-oracle-protocol.md`, committed `1ec1e90` before any new arm existed. Evidence: `outputs/pose_insurance/position-oracle/` (`summary.md`, `marginals.md`, `scenarios.md`). Train labels, oracle arms only, never a pack candidate. All five validity checks passed (T1, O2 and O3 reproduced exactly).

| Arm (eight-episode mean) | AUC | RPE cm | MPPE cm | SP-SR |
| --- | ---: | ---: | ---: | ---: |
| T1 | 0.2767 | 29.01 | 17.68 | 0 |
| PL-W1: 0/26/31, true length, yaw swept | 0.2767 | 27.02 | 16.56 | 0 |
| PO-W1: 0/26/31, slot 1 exact | 0.2767 | 24.89 | 15.53 | 0 |
| PO: all eight, slot 1 exact | 0.2886 | 15.11 | 9.03 | 0 |
| RC: relative rotation only | 0.2926 | 29.01 | 17.29 | 0.125 |
| O2: relative rotation plus anchor direction | 0.3527 | 20.27 | 11.35 | 0.25 |
| O3: rotation and offset | 0.5319 | 15.11 | 7.40 | 0.625 |

| Marginal | AUC | RPE cm | MPPE cm | SP-SR |
| --- | ---: | ---: | ---: | ---: |
| Position first: PO - T1 | +0.0119 | -13.90 | -8.65 | 0 |
| then O3 - PO | +0.2433 | 0 | -1.63 | +0.625 |
| Relative rotation first: RC - T1 | +0.0159 | 0 | -0.39 | +0.125 |
| then O3 - RC | +0.2393 | -13.90 | -9.88 | +0.500 |

**Findings.**

1. **Slot 1 exact at frame 0 gains no AUC at all on 0, 26 and 31**, in any direction. Their RPE and MPPE do improve. The plan's 0.387 for correcting those three came from O3, so nearly all of it belongs to the joint correction and none to position alone.
2. **The levers interact strongly.** Position alone is +0.012 and relative rotation alone +0.016, while together they are +0.255. Either one first leaves about +0.24 for the other. Neither "position then rotation" nor "rotation then position" is a fair additive story, and a route that fixes only one looks nearly worthless on AUC whichever it is.
3. **Position alone is a real RPE and MPPE lever.** PO halves RPE (29.0 to 15.1 cm), and it equals O3 on RPE by construction, since RPE reads positions only.
4. **Unsolved direction costs most of W1's gain on 26 and 31.** With yaw swept, 26's RPE averages 20.9 cm (11.3 to 29.4) against 18.7 for T1, so the mean is worse than the stack, and 31's averages 17.3 (9.6 to 23.1). Episode 0, a nearly vertical stack, barely depends on yaw.

**Scenario table (a calculation, not a schedule; AUC / RPE / MPPE per hour, scored reach out of 20).** Perfect-model figures for the rotation rows, so it favours them by construction:

| Route | `g x r / (20 h)` |
| --- | --- |
| W1, length with unsolved yaw | 0 / -0.355 / -0.198 |
| W1, direction solved | 0 / -0.733 / -0.382 |
| Rotation, W2a or H3 | 0.0042 / -0.628 / -0.455 |
| Rotation and offset, W2b (4 h unverified) | 0.0248 / -1.285 / -0.958 |

**Consequences.** W1 does not justify a position-only pack on AUC. Its masks are still worth building, because they unlock the joint routes: `frame0-panel-protocol.md` (committed `f6ba8bc`, before these results were read) gates rotation and offset together on 0, 26, 31 and 23. Pose estimation that recovers both is the largest figure in the table, and it reaches the cup-anchored pairs H3 cannot, so the FoundationPose pilot goes ahead of W2a training.

**Independent check.** The lead recomputed T1 and PO from raw mocap with a different formula (`R0 R0_ref^T d_ref`), through the kit's functions: largest residual per episode 7e-15; means T1 0.276705, PO 0.288631, RPE 29.014 to 15.114. These arms repair full XYZ, so they do not forecast the XY-only placement code, and they are not upper bounds on whole-clip static AUC.
