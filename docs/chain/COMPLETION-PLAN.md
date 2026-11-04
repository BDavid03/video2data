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
- **Reinforcement learning in simulation is required (owner, 2026-11-04 about 05:06 UTC: "put in docs, reinforcement learning required using simulations").** Every policy behind a submitted pose is trained by RL in simulation; frame-0 estimates, video-derived references and pretrained models are inputs to that, never a replacement for it.
- Rollouts come from the official `evaluate_policy.py` with a learned checkpoint and physical contact. No kit edits, direct object actuation or hand-built trajectories. No holdout 43/44, no episode 41, no confidential reference. Leaderboard scores steer nothing: not recipes, not this plan's allocation.
- **Scans:** no scan geometry enters a submitted mesh or a simulated asset (`SUBMISSION.md`, rules). New use of scans for anything else waits for the owner's line (section 7).
- A new pack's manifest must not pin `AGENTS.md`.

## 4. Workstreams

### W0. A real score (owner, 15 minutes, any time; nothing below waits on it)

`git push origin main`, accept the rules on all six Track 3 competitions, save a token (`~/.kaggle/access_token` or `KAGGLE_API_TOKEN`), upload both v5 files (`NEEDS-YOU.md`, top), write the scores there. The FAQ allows unlimited uploads in the final three days. Pushing publishes the whole history, including the coursework notebooks and four episode-41 frames.

### W1. Frame-0 positions on merged episodes (installed tools; about 3 h, under 1 GPU-hour)

Paired non-T1 exports are scored **1, 5, 28, 30** (1 and 28 have a zero offset; only 30 is a recorded mask fallback). Train 0, 26, 31 have the same problem. SAM2.1 and Grounding DINO are now installed.
- **W1.0 done, position alone gains no AUC** (`position-oracle-outcome.md`): exact slot 1 on 0/26/31 is +0 AUC, position alone +0.012 and relative rotation alone +0.016, jointly +0.255. W1's masks stay worthwhile because they unlock the joint routes; a position-only pack would have to earn its place on RPE and MPPE through Gate P. W1.1 to W1.3 now run as `frame0-panel-protocol.md`, which moves the vector gate to where a registered anchor supplies the rotation.
- *Original text.* **W1.0, first (CPU, minutes):** position-only oracle on train. The recorded 0.387 for correcting 0/26/31 is a joint correction, not W1 alone, and T1 showed a correct length alone barely moves AUC, so expect RPE and MPPE more than AUC. Report marginal gains, and rank W1 against W2 by gain per hour.
- **W1.1:** per-object frame-0 masks from fixed object-name prompts, direct stereo offset, on train 0/26/31. Gate fixed first, on **vector** body-origin error in the frame the T1 protocol uses, not length alone: at least two of three within 3 cm of mocap, with no train episode worse beyond a stated tolerance. A video rule decides genuinely stacked objects (episode 1's lid).
- **W1.2:** kinematic train gate, about 30 s GPU per scored episode, the physical check, composition.
- **W1.3 (needs the scan line, section 7):** those masks make the H3 method shipped on episode 19 applicable to more pairs, such as 28 and 30. Prepare the protocol; do not pack without the line.

### W2. Frame-0 rotation: the first model we actually train (kill test 90 minutes; then up to 3 GPU-hours; kill at 07:00 UTC)

Reach: at most 9 of 20 episodes, the two-scan pairs 1, 8, 10, 14, 20, 25, 28, 30 and 19 (done). Paired spoon episodes 3, 5, 6, 15, 16, 32, 33 cannot be labelled, because the spoon's frame is unknown. **35, 36, 37 (spoon) and 38 (dust pan) are single-object, so static rotation is aligned for free.**
- **Data:** 2,003 labelled train frames (4,006 object observations) from 8 correlated clips. There is **no camera-to-mocap extrinsic**, so targets must be camera-invariant: the pair's relative rotation and anchor-frame offset, with scan-frame conventions preserved and no z-up assumption. Synthetic data renders **our own reconstructions** (`PROGRAM.md` section 0), not scans. A scan-render arm waits for the line in section 7.
- **Kill test, first:** (1) prove headless rendering: `moderngl` is installed but defaults to X11, so test an EGL context and read back one known pose. (2) A small fixed retrieval baseline on one held-out train episode, with flipped rivals. HR2 found a 179-degree rival within 2 %: flips count as failures. If this cannot separate rivals, record it and stop W2.
- **Gate, leave-one-episode-out:** relative rotation under 30 degrees on at least half the pairs; then train AUC at least +0.02 with SP-SR above 0 against T1, no episode worse beyond tolerance. Check on tuning 39/42 where applicable.
- **W2b, ahead of W2a training (route-m0-w1-24):** FoundationPose in its own venv, owner-approved. Bounded feasibility first, capped at 90 minutes for weights, build and one real frame; this machine has no `gcc`, and upstream needs CUDA source builds. Then predictions frozen on train 12 and 31 before reading labels, needing relative rotation under 30 degrees and offset vector error under 3 cm on both to justify a full panel. It is the only route here that reaches cup-anchored pairs. Noncommercial licence, recorded and disclosed. Its protocol is written before it runs.

### W3. Moving objects: decisive tests before any large build

Each step can kill the route; together they bound it.
- **M0 done, NULL** (`moving-m0-outcome.md`): `T` 0.035 / 0.009, `Q` 0.075 / 0.009, box moved at most 2.0 of 17.4 cm. It closes nothing by itself, and M2's protocol must now choose between a test matched to D1 and a completed curriculum.
- *Original text.* **M0 (about 3 GPU-minutes, now):** score D1's partial policies `policy_21580800` and `policy_22604800` through `evaluate_policy.py`, using a new evaluation-only slice operation under `GpuLease`; the training handler refuses nonzero final VOC. Inventory and hash the actual latest checkpoints, because `partial_checkpoints.json` lists earlier ones. New metric, numeric gate fixed first: **transport** = sum(dp_achieved . dp_ref) / sum(|dp_ref|^2), with perpendicular error beside it (S scores 0). The project has never measured this.
- **M0b (new, about 1.5 GPU-hours; Claude critic 24):** finish a policy. A probe of about 10 minutes at `updates_per_collection` 0.5 and 2.0 measures training throughput, then D1 seed 0 resumes from its saved state (`state_21575680`; replay was not saved, resume compatibility untested; `workflow/train_flash_sac.yaml:48-49`) to the schedule's zero-assistance stage at 32,814,080 and its end at 66,763,776, about 14 and 58 minutes at 13k/s and 0.125. `T` and `Q` are scored at both points with the 42-second evaluator, against the M0 table. Its own protocol comes first, including which ratio runs. It yields a finished unassisted policy on the shipped task, and tests the variable M0 could not. Before it, a CPU check of hand-object clearance against the reference, because the export shows a bump near frames 146-211 rather than a carry; hands are not in the trajectory export.
- **M1 (about 2 h, no policy; deferred until some policy passes a transport screen, since it bounds a reference nothing can yet follow):** tracked masks, stereo and a per-frame rigid fit on train **11 and 12**, which each have one nearly stationary object for removing camera motion (object travel 0.49/25.20 and 22.23/0.41 cm). Score the kinematic reference with `static_pose_baseline.score`. Gate: beats the same episode's shipped T1 (0.500, 0.104) by a margin fixed first, plus a measured allowance for physical tracking error.
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

**Revised 4 Nov about 05:00 UTC, after M0b's GO-1 (`moving-m0b-outcome.md`).** The panel closed with no pack, and a finished curriculum tracks the shipped task at T 0.987. W3 is now the main line, and object reference quality is its bottleneck. Route: Codex `route-w3-25`. Protocols: `moving-m2-protocol.md`, `moving-m0c-protocol.md`.

| When (UTC) | Context does | GPU |
| --- | --- | --- |
| Now to about 06:05 | M0c (the same recipe from nothing, seed 1), launched 05:19 | about 45 min |
| Done 05:20 | FoundationPose pilot **PASS** on two used train pairs, 1.81 cm / 2.4 deg and 1.23 cm / 8.8 deg (`foundationpose-pilot-outcome.md`). Its full panel and its tracking mode as M1's leading candidate are now in play | |
| **First, now** | **Settle M2's Stage 0 (`moving-m2-protocol.md`, "State"): it stopped, so M2 needs a disclosed amendment or a recorded stop before any GPU minute is planned around it.** Decide on upstream's route (`route-v2d-25`, owner's pointer to `video_to_data`), and fix M2's gates as critic 25 asks. CPU | none |
| To about 07:00 | Amended task through Stage 0; M1's protocol on 39 and 38, with camera motion from the background. Scan-free stereo plus masks unless the owner allows scan tracking on scored episodes (`NEEDS-YOU.md`). Also build a moving pack path: `pose_insurance/handler.py:79` refuses drift above 1 cm, and no moving pack path exists | short |
| 07:00 to 09:00 | M2: mocap-39, the upper bound (about 55 min). Then M1's gate, then M2v: video-39 (about 55 min) | about 2 h |
| 09:00 to 10:30 | Scored policy for whatever passed: **realistically 38 alone**, about 55 min | rest |
| To 12:00 | One gated pack, abstract, `check_release.sh`, push and upload | none |
| To 17:00 | Chain records scores, works W6 report items | none |

**Cut:** W4, W2a's rotation model, the FoundationPose static panel (static, physical admission unsolved), spoon expansion, and broad restoration of the recycled slices. MediaPipe is cut unless `route-v2d-25` shows it can feed upstream's retargeting. First to go if time runs short after that: M2v's second variation. **The biggest unproven link is still transfer to a packable challenge policy: task assembly, embodiment, reference quality and physical tracking.** Name it in any W3 pass; do not call a shipped-task result a challenge result. Earliest kill check: Stage 0's grasp closure and first-frame support. If M2 is NULL, W3 closes for this deadline and D2 v5 stands.

## 6. What done means

| Lens | Now | Done means |
| --- | --- | --- |
| Readiness | 3 | v5 or better uploaded, scores recorded, commit public |
| Models | 3 | A model we trained changes a scored file through a gate, or a recorded gate says why not |
| Metrics | 4 | Train and tuning numbers for every adopted change, leaderboard scores recorded beside them |
| Insights | 4 | Transport measured; reference quality bounded; marginal frame-0 gains by episode |
| Presentation, Visualisations | 4 | Abstract describes what shipped and those findings |
| Engineering | 4 | `check_release.sh` READY on the final files; D3's external inputs checked |

## 7. Owner decisions (answered 2026-11-04 01:53 UTC; these override the "needs the line" notes above)

1. **Deadline: unchanged.** Last pack 12:00 UTC, deadline 17:00 UTC, last GPU launch 10:30 UTC.
2. **Scan use: yes.** Scans may train the rotation model (add a scan-render arm to W2a) and H3 may extend past episode 19 (W1.3). Still no scan geometry in a submitted mesh or simulated asset; disclose each use in the abstract.
3. **Push and upload: yes, by the chain.** Terms in `PROGRAM.md` section 0. W0's upload runs as soon as the owner has accepted the rules and saved a token, which stay the owner's.
4. **FoundationPose: yes.** W2b runs, with the licence recorded.
