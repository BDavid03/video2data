# Completion plan: the target for every chain context until the freeze

Owner, 2026-11-04 about 01:40 UTC: "you still havent even got a model trained or submission written to improve on, why have
you ended the chain, do a widespread plan for completion that the chain can target". `PROGRAM.md` section 0 points here.
Numbers below were checked against code and records by a fresh Codex critic (`outputs/chain/codex/*plan-critic-23.answer.md`).

## 1. Where the entry really stands

| Claim | Fact | Evidence |
| --- | --- | --- |
| A submission exists | D1 v5 and D2 v5 are packed and pass preflight locally. **Nothing is uploaded, so there is no leaderboard score.** No Kaggle token is on this machine | `SUBMISSION.md`, `NEEDS-YOU.md` |
| Policies are trained | **Effectively not.** Every shipped policy ran a 512-step budget in one world. Objects stay where they start | D2 v5 manifest `budget` |
| Pose quality | Train AUC 0.277, SP-SR 0, MP-SR 0, RPE 29 cm, MPPE 18 cm | `outputs/pose_insurance/stereo-placement/summary.md` |
| Frame-0 oracles, static (ceilings, not forecasts) | Rotation only (O2): AUC 0.353, SP-SR 0.25. Rotation and full offset (O3): AUC 0.532, SP-SR 0.625, RPE 15 cm. Train 18 alone supplies 45 % of O2's gain, and its analogue, episode 19, already ships it | `outputs/pose_insurance/heading/summary.md` |
| Moving objects | D0 NO-GO at the cheapest budget. **D1 was never scored**: both seeds timed out at about 22M transitions; the partial policies exist and load-check as complete, with assistance still at VOC 0.05. Stopping the route was a planning decision, not a finding | `dynamic-scaling-outcome.md` |
| Meshes | Measured primitives plus two pretrained shapes, CD-O 0.56 cm on public scans; 10 of 36 scored pairs (the spoon) have no scan | `mesh-generative-outcome.md` |

The third-party 0.99 and how it was reached are unknown. This plan does not promise it. One public team reports 0.29 from a full
video pipeline at 64 GPU-hours.

## 2. Clock and budget

- **Hard deadline 17:00 UTC 4 Nov; last pack 12:00 UTC**, both from the owner's line in `AGENTS.md`, so the owner can push and upload. The kit's freeze is 22:00 UTC (section 7).
- **GPU:** one RTX 5080, one operation at a time, about 10 serial hours to the pack cut-off. Last GPU launch 10:30 UTC.
- **Throughput:** about 13k transitions/s at 1,024 worlds and 4.7 GiB, measured at `updates_per_collection` 0.125; two-object tasks may cost more. Budget at 1.1x measured training time.
- The released 250M recipe takes 5.3 h per episode, 107 h for 20. Assistance reaches zero only at 122.88M, 2.6 h per episode at best. **So twenty undertrained policies are the wrong target: complete a compressed, unassisted curriculum on a few episodes, keep v5 elsewhere.**

## 3. Rules for this plan

- Cheapest decisive test first, with a written protocol before any number (`PROGRAM.md` sections 0 and 4).
- A candidate must beat its same-episode incumbent on train/tuning labels under a gate fixed first, with stated tolerances. Scored episodes have no labels, so per-episode adoption uses frozen video and physical admission rules only, never scored AUC. A complete weaker entry beats an unfinished stronger one.
- **The chain does not stop for owner-only items while a workstream is open.** It records them in `NEEDS-YOU.md` and carries on, and no gate depends on an owner action. `CHAIN: STOP` is allowed only when every workstream has reached its gate result or its cut-off, or after the freeze.
- Rollouts come from the official `evaluate_policy.py` with a learned checkpoint and physical contact. No kit edits, direct object actuation or hand-built trajectories. No holdout 43/44, no episode 41, no confidential reference. Leaderboard scores steer nothing: not recipes, not this plan's allocation.
- **Scans:** no scan geometry enters a submitted mesh or a simulated asset (`SUBMISSION.md`, rules). New use of scans for anything else waits for the owner's line (section 7).
- A new pack's manifest must not pin `AGENTS.md`.

## 4. Workstreams

### W0. A real score (owner, 15 minutes, any time; nothing below waits on it)

`git push origin main`, accept the rules on all six Track 3 competitions, save a token (`~/.kaggle/access_token` or `KAGGLE_API_TOKEN`), upload both v5 files (`NEEDS-YOU.md`, top), write the scores there. The FAQ allows unlimited uploads in the final three days. Pushing publishes the whole history, including the coursework notebooks and four episode-41 frames.

### W1. Frame-0 positions on merged episodes (installed tools; about 3 h, under 1 GPU-hour)

Paired non-T1 exports are scored **1, 5, 28, 30** (1 and 28 have a zero offset; only 30 is a recorded mask fallback). Train 0, 26, 31 have the same problem. SAM2.1 and Grounding DINO are now installed.
- **W1.0, first (CPU, minutes):** position-only oracle on train. The recorded 0.387 for correcting 0/26/31 is a joint correction, not W1 alone, and T1 showed a correct length alone barely moves AUC, so expect RPE and MPPE more than AUC. Report marginal gains, and rank W1 against W2 by gain per hour.
- **W1.1:** per-object frame-0 masks from fixed object-name prompts, direct stereo offset, on train 0/26/31. Gate fixed first, on **vector** body-origin error in the frame the T1 protocol uses, not length alone: at least two of three within 3 cm of mocap, with no train episode worse beyond a stated tolerance. A video rule decides genuinely stacked objects (episode 1's lid).
- **W1.2:** kinematic train gate, about 30 s GPU per scored episode, the physical check, composition.
- **W1.3 (needs the scan line, section 7):** those masks make the H3 method shipped on episode 19 applicable to more pairs, such as 28 and 30. Prepare the protocol; do not pack without the line.

### W2. Frame-0 rotation: the first model we actually train (kill test 90 minutes; then up to 3 GPU-hours; kill at 07:00 UTC)

Reach: at most 9 of 20 episodes, the two-scan pairs 1, 8, 10, 14, 20, 25, 28, 30 and 19 (done). Paired spoon episodes 3, 5, 6, 15, 16, 32, 33 cannot be labelled, because the spoon's frame is unknown. **35, 36, 37 (spoon) and 38 (dust pan) are single-object, so static rotation is aligned for free.**
- **Data:** 2,003 labelled train frames (4,006 object observations) from 8 correlated clips. There is **no camera-to-mocap extrinsic**, so targets must be camera-invariant: the pair's relative rotation and anchor-frame offset, with scan-frame conventions preserved and no z-up assumption. Synthetic data renders **our own reconstructions** (`PROGRAM.md` section 0), not scans. A scan-render arm waits for the line in section 7.
- **Kill test, first:** (1) prove headless rendering: `moderngl` is installed but defaults to X11, so test an EGL context and read back one known pose. (2) A small fixed retrieval baseline on one held-out train episode, with flipped rivals. HR2 found a 179-degree rival within 2 %: flips count as failures. If this cannot separate rivals, record it and stop W2.
- **Gate, leave-one-episode-out:** relative rotation under 30 degrees on at least half the pairs; then train AUC at least +0.02 with SP-SR above 0 against T1, no episode worse beyond tolerance. Check on tuning 39/42 where applicable.
- **W2b (needs the owner's line):** FoundationPose on the same panel, in its own venv, after a licence check. It uses scans, so it also needs the scan line.

### W3. Moving objects: decisive tests before any large build

Each step can kill the route; together they bound it.
- **M0 (about 3 GPU-minutes, now):** score D1's partial policies `policy_21580800` and `policy_22604800` through `evaluate_policy.py`, using a new evaluation-only slice operation under `GpuLease`; the training handler refuses nonzero final VOC. Inventory and hash the actual latest checkpoints, because `partial_checkpoints.json` lists earlier ones. New metric, numeric gate fixed first: **transport** = sum(dp_achieved . dp_ref) / sum(|dp_ref|^2), with perpendicular error beside it (S scores 0). The project has never measured this.
- **M1 (about 2 h, no policy):** tracked masks, stereo and a per-frame rigid fit on train **11 and 12**, which each have one nearly stationary object for removing camera motion (object travel 0.49/25.20 and 22.23/0.41 cm). Score the kinematic reference with `static_pose_baseline.score`. Gate: beats the same episode's shipped T1 (0.500, 0.104) by a margin fixed first, plus a measured allowance for physical tracking error.
- **M2 (about 2 h GPU):** **object-driven grasp reference** against the current hand reference on the tissue box, same compressed budget, same unassisted evaluator, transport metric. This replaces human-hand reconstruction as the first route (D0: hands alone move the box 1.5 of 17.4 cm). It is Codex's proposal, not a result. Floating hands are allowed by the FAQ and may be the easier embodiment. Budget it on D1's measured throughput; D1's rescaled 66.76M schedule takes about 95 minutes.
- **M3 (only if M0-M2 pass, GPU free by 07:30 UTC):** complete compressed curricula on at most 3 episodes, single-object first (35-38: motion is all they lack). Restore `retargeting`, `scene_assembly` and `hand_motion` from `recycle/20261104T010336Z/` with `git mv` as needed; their integration is real extra work. Shared initialisation is separate work (resume validates schemas and articulation contracts), not assumed. Gate first on train, then adopt by frozen admission rules.
- MediaPipe hands (approved) only if the object-driven route fails and hours remain. WiLoR + MANO need the owner's registration and gate nothing.
- **Honest odds:** subjective, under 15 % of beating static on any scored episode by 12:00 UTC. It runs because the upper bound (AUC 1.0 against a static 0.53) is what the owner asked to go after, and M0 and M1 are cheap. If killed, the finding goes in the report and W6 continues it.

### W4. Meshes (only when the GPU is idle; capped at 2 h; never allocated from leaderboard rank)

New prospective protocols only, never a repair of G3: multi-seed consensus and whole-handle masks (IDEAS 60), visibility-aware deformation (IDEAS 56). The spoon stays a known risk.

### W5. Pack, report, release (continuous)

- A new pose file only if W1, W2 or W3 passes; at most two more packs, last at 12:00 UTC. Each updates `scripts/reproduce/release.json` so `check_release.sh` stays the one pass/fail.
- A pack is prepared and marked as waiting for its commit to be pushed; that wait blocks no other work.
- The abstract follows every adopted change, at two main pages, with a cold reading each time.
- Before 12:00 UTC: confirm the Apache-2.0 licence the challenge requires of winning code is in the repository (`SUBMISSION.md`, rules).

### W6. After the freeze

Record final scores, complete the report, carry W3 on full budgets if the owner wants it, run independent-object mesh evaluation, settle `recycle/`, and reset this plan.

## 5. Order for the next contexts

| When (UTC) | Context does | GPU |
| --- | --- | --- |
| Now | Protocols; W1.0; M0 | minutes |
| To 04:00 | W1.1-W1.2; M1; W2 kill test | about 1 h |
| To 07:00 | M2; W2 train and gate if alive; W3 go or kill | about 3 h |
| To 10:30 | W1/W2 scored runs; M3 if alive; W4 if idle | rest |
| To 12:00 | Final pack, abstract, `check_release.sh`, `NEEDS-YOU.md` commands | none |
| To 17:00 | Owner uploads. Chain records scores, works W6 report items | none |

## 6. What done means

| Lens | Now | Done means |
| --- | --- | --- |
| Readiness | 3 | v5 or better uploaded, scores recorded, commit public |
| Models | 3 | A model we trained changes a scored file through a gate, or a recorded gate says why not |
| Metrics | 4 | Train and tuning numbers for every adopted change, leaderboard scores recorded beside them |
| Insights | 4 | Transport measured; reference quality bounded; marginal frame-0 gains by episode |
| Presentation, Visualisations | 4 | Abstract describes what shipped and those findings |
| Engineering | 4 | `check_release.sh` READY on the final files; D3's external inputs checked |

## 7. Owner decisions (the chain runs the default until a line appears)

Add the answer to `PROGRAM.md` section 0, not `AGENTS.md` (pinned by D2 v5), or say it in the session.

1. **Keep the 12:00 pack cut-off and 17:00 deadline, or plan to the kit's 22:00 UTC?** Default: keep. 17:00 UTC is about 03:00 AEST; 22:00 UTC is 08:00 AEST. Moving the last pack to 20:00 UTC would almost double the GPU budget, and is what could make M3 real.
2. **Scan use beyond measuring:** may scans train a rotation model, and may H3 extend past episode 19? Default: no. The FAQ says participants "should reconstruct the objects"; D2 v5's episode 19 is the only exception, by your approval.
3. **May the chain push and upload?** Default: no; you do both, and every pack waits for you.
4. **FoundationPose install?** Default: not installed. It needs a licence check as well as decision 2.
