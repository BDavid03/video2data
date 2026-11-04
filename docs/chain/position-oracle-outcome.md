# W1.0 outcome (4 Nov 2026, about 02:08 UTC): position alone wins no AUC on W1's roster and +0.012 across train; the gain is in the joint correction

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

1. **Slot 1 exact at frame 0 gains no AUC on 0, 26 and 31**, nor does any of the 24 yaw directions tested at the reference's height and radius. Their RPE and MPPE do improve. Across all eight episodes position alone gains +0.012, from 11 (+0.042) and 18 (+0.052). The plan's 0.387 for correcting those three came from O3, so under these constructions none of it belongs to position alone.
2. **The levers interact strongly, on the mean.** Position alone is +0.012 and relative rotation alone +0.016, while together they are +0.255 (interaction +0.227). Either one first leaves about +0.24 for the other. The mean hides episodes: RC's whole gain is episode 23 (+0.128), PO's is 11 and 18, and 12 and 31 each jump exactly +0.5 under O3, which is 49 % of O3's gain. A likely mechanism, not established here: the kit counts frames with ADD under 1 to 9 cm per body (`AUC.py:19,87-94`), so a large rotation or position error alone earns nothing for that body.
3. **Position alone is a real RPE and MPPE lever.** PO halves RPE (29.0 to 15.1 cm), and it equals O3 on RPE by construction, since RPE reads positions only.
4. **Unsolved direction matters a lot on 26, less on 31.** With yaw swept, 26's RPE averages 20.9 cm (11.3 to 29.4) against 18.7 for T1, so all of the gain is lost on average and the mean is worse than the stack. 31's averages 17.3 cm (9.6 to 23.1) against T1's 30.4, keeping 63 % of PO's RPE gain. Episode 0, a nearly vertical stack, barely depends on yaw.

**Scenario table (a calculation, not a schedule; AUC / RPE / MPPE per hour, scored reach out of 20).** Perfect-model figures for the rotation rows, so it favours them by construction:

| Route | `g x r / (20 h)` |
| --- | --- |
| W1, length with unsolved yaw | 0 / -0.355 / -0.198 |
| W1, direction solved | 0 / -0.733 / -0.382 |
| Rotation, W2a or H3 | 0.0042 / -0.628 / -0.455 |
| Rotation and offset, W2b (4 h unverified) | 0.0248 / -1.285 / -0.958 |

**Consequences.** On AUC, W1 gives no reason to expect a position-only pack to pay; Gate P of `frame0-panel-protocol.md` still lets RPE and MPPE earn one, and W1.0 kills nothing. Its masks are still worth building, because they unlock the joint routes: that protocol (committed `f6ba8bc`, after W1.0 ran; the lead had not read the results, which is disclosed and not independently verifiable) gates rotation and offset together on 0, 26, 31 and 23. Pose estimation that recovers both is the largest figure in the table, by perfect-model assumption and on unverified hours, and it reaches the cup-anchored pairs H3 cannot. FoundationPose-before-W2a was decided by the route review before these results existed; they do not contradict it.

**Independent check.** The lead recomputed T1 and PO from raw mocap with a different formula (`R0 R0_ref^T d_ref`), through the kit's functions: largest residual per episode 7e-15; means T1 0.276705, PO 0.288631, RPE 29.014 to 15.114. These arms repair full XYZ, so they do not forecast the XY-only placement code, and they are not upper bounds on whole-clip static AUC.
