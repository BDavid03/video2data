# FoundationPose pilot protocol: can a pretrained model-based estimator give a correct frame-0 pair? (`COMPLETION-PLAN.md` W2b; written before any install or prediction)

## Question

Scan-to-stereo registration earned real AUC on the panel but failed its gate, with flip ambiguity the largest failure (`frame0-panel-outcome.md`). HR1 found cup heading unrecoverable from stereo shape. FoundationPose also uses image texture, so it might break flips and reach cup-anchored pairs. This pilot tests that on two train pairs before any full panel. **It changes no submitted file.**

## Hard limits

- **90 minutes in total, from 04:15 UTC, ending 05:45 UTC,** for toolchain, weights, build and one real frame. Milestones: toolchain and downloads by minute 25, builds by minute 70, a real frame by minute 90. A missed milestone stops the pilot, and it is recorded as infeasible on this machine under this cap, not as a finding about the method.
- Own environment under `.runtime/foundationpose/`. The main venv is untouched. Licence: NVIDIA's noncommercial source licence, recorded in `docs/INSTALL.md` and disclosed in the abstract if any result is used (owner's approval: `PROGRAM.md` section 0).
- GPU only when no other GPU operation runs; M0b's training and scoring come first.

## What was already known when this was written (disclosure)

No untouched train pair remains. Train 31 was evaluated in the panel: TW offset residual 5.12 cm, `b` angle 26.0 deg, `C` 9.1 deg, with a rival within 0.117 % at 179 deg. Train 12's reference offset and the joint oracle's +0.5 AUC jump on it are on record, and it was HR1's and T1's evidence. T1 and oracle rows for both are known. All scans have UV-mapped base-colour textures (`route-fp-25`); `mini_sweeper` uses its visual mesh. Stage A masks exist for both pairs. So this is a screen on used pairs, not independent validation.

## Inputs, fixed now

Train **12** (`blue_cup`, `plastic_dish_rack`) and **31** (`mini_sweeper`, `plastic_dish_rack`), frame 0 only. Camera-A remapped RGB, the stereo depth T1 used, on its grid and intrinsics, holes left invalid. Stage A's overlap-resolved masks, unchanged (`outputs/pose_insurance/frame0-panel/stage-a/`), with no re-prompt. Scan GLBs, read only. FoundationPose's upstream checkpoints, one seed, five refinement iterations, frame 0 registration with no tracking. One run per object. A run that crashes without output may be retried once, disclosed. A run with output is never repeated, and no parameter changes after one exists.

## Order

1. **Kill test (under 10 minutes after install):** register `blue_cup` once, save hypotheses and an overlay, no label read. Fewer than four valid points after upstream's filtering, or no finite plausible hypothesis, stops the pilot, recorded.
2. **Predictions:** all four objects. Rotations and translations are restored to the scan's original frame, and saved with hashes and overlays, **before any label is loaded.**
3. **Scoring.** On the two pairs, with `b = R0^T (t1 - t0)` and `C = R0^T R1` against mocap, computed by `_heading_h3.errors`' definitions.

## Gate (both pairs must pass; numbers fixed now)

`|b_pred - b_ref| < 3 cm` and `C` error under 30 degrees, on **both** 12 and 31. A flip, a decline or a missing value fails. No symmetry allowance is applied, because the kit's metrics apply none; the cup's handle is what would resolve it. The limits come from the tolerance curve in hand-off 24 (unprotocolled scratch): O3 plus slot-1 error of 3 cm / 15 deg still gains +0.165 mean train AUC, and 5 cm / 30 deg only +0.066.

**Beside the gate, descriptive only:** 5 cm passes; each object's top-hypothesis margin; for 31, whether it avoids the recorded rival; and each pair's AUC, SP-SR, RPE and MPPE with its prediction substituted into T1.

## What each result does

| Result | Consequence |
| --- | --- |
| PASS | A full panel protocol is written first, over 0, 11, 18, 21, 23, 26 and the cup-anchored scored roster, with TW's gates including a per-object stability measure. Nothing is packed from this pilot |
| FAIL or infeasible | FoundationPose stops for this deadline; recorded in the outcome and the report. W2a's kill test goes next |

## Integrity

Episodes 41, 43, 44 are never read. Validation 2, 13, 22, 24, 27 are not used. Scans enter only measurement; no scan geometry enters a submitted mesh or a simulated asset. The outcome goes in `docs/chain/foundationpose-pilot-outcome.md`, whichever way it falls, and the lead independently recomputes the pair errors before believing a PASS.
