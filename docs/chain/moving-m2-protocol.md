# M2 protocol: does the recipe transfer to a challenge-type episode? (`COMPLETION-PLAN.md` W3; fixed before any number below exists)

## State (4 Nov 2026, 05:35 UTC): Stage 0 STOPPED; no training has run; awaiting a disclosed amendment

The narrow builder ran on real 39 (`outputs/moving_policy/ep39/stage0.json`, `final-checks.json`). **Failed, after the one allowed repair:** check 4, wrist speed 1.59 m/s, 7.43 times the example against a limit of 2; check 5, the object reaches 10.5 m in 11 CPU physics steps with 0.73 degrees of rotation, which looks like a solver blow-up (hands start 4 to 6 cm inside the support), not a fall. Passed: schema, closure, penetration, truth; closure and truth are close to vacuous for a rigid grasp and an identical reference.

**The next context settles this before any GPU minute is planned around it: a disclosed amendment, or a recorded stop.** No training number exists, so a redesign remains clean. Inputs to it:
- **Upstream's own route (owner, about 05:30 UTC: "worth using: https://github.com/nvidia-isaac/video_to_data.git"):** the pinned checkout has `reconstruction/`, `ego_recon_to_sharpa.py`, `contact_labels.py`, `generate_rigid_urdfs.py`, `reconstruct_support_surfaces.py`, `data_quality_checks/` and `docs/e2e_workflow.md`. See Codex `route-v2d-25`.
- **Check 4's limit was a guess from one example,** and human motion on 39 is genuinely fast: 71 against 9 cm/s. Decide it on what the tracker can follow, not by loosening it silently.
- **Gate fixes from critic 25, to take with the amendment.** AUC 0.275 is too weak: with position perfect and rotation held, 39 reaches 0.413 from a static 0.178, so GO must include rotation or a higher AUC. Scored adoption clause 3 scores against its own reference, so a wrong path passes; it needs `E` and an orientation bound, and a **label-free reprojection of achieved poses on the video's masks**, since residual camera motion could become a trackable, wrong reference. Clause 2 is undefined until M1's protocol. Check 5 cannot literally precede all simulation; it is a CPU physics check. The spoon inherits no scan-free labelled test.

## Question

M0b's recipe tracks the shipped tissue box (`moving-m0b-outcome.md`). That task has a captured hand reference and its own mesh. A challenge policy has neither. M2 asks whether a policy tracks an object on a challenge episode when the hands come from the object's own trajectory. **Tuning episode 39 (dust pan), single object, mocap reference: an upper bound.** A failure here rejects this recipe for this deadline, not the idea. A pass says nothing yet about a video-derived reference; that is M2v below. Route: Codex `route-w3-25`.

## What was already known when this was written (disclosure)

M0b's table; D0, D1 and M0; route-w3's reading of the task contract; T1 AUC on 39 of 0.1750 and on 42 of 0.1941, and that 39's static row has been read before as tuning evidence. 39 and 42 are tuning episodes with labels, used here as `PROGRAM.md` section 4 allows; 39 was chosen because it matches scored 38's object. No moving rollout of any challenge episode exists.

## The task, built before any training

A narrow task builder in a new slice, not the recycled chain. **Object reference:** 39's mocap poses, at the dataset's frame timing. **Object asset:** our own reconstructed dust pan, from the same route as D1 (no scan geometry in a simulated asset). The scan is not read to build it. **Embodiment:** the example's two floating Sharpa hands, one or both active. **Hand reference:** after a fixed acquisition phase, `T_WH(t) = T_WO(t) T_OH`, one rigid grasp per hand, transforming wrist, finger frames and contact points together. Grasp `T_OH` and finger joints come from a deterministic rule on our asset's geometry, chosen before training and never revised after any score. No per-frame hand-tuning. The object is moved only by contact: **no direct object actuation, no hand-built object motion in a rollout.** Rollouts come only from the kit's unedited `evaluate_policy.py`, with a learned checkpoint.

## Stage 0: label-free and reference-only checks, before any GPU minute

The checks below use the mocap reference and our asset, and run on CPU. **Any failure stops M2 with the reason recorded.** One repair of the rule is allowed, disclosed, while no training has run.

1. **Schema:** the parquet loads through FlashCHORD's loader and passes its contracts, with the example's fields and shapes.
2. **Grasp closure, by the clearance criterion in `hand_clearance.py`:** at least two distinct contact groups within 1 cm of opposite exterior faces with outward normals dot at most -0.8, on at least 90 % of the frames after acquisition.
3. **Penetration:** deepest hand-object penetration at most 1 cm at any frame after acquisition.
4. **Kinematics:** finger joints inside limits at every frame; wrist speed within the example's observed maximum, or the reference is reported as exceeding it, and this is a stop if it is above twice that.
5. **First frame:** with zero action in the evaluator's physics, the object stays within 1 cm and 2 degrees for the acquisition phase; a body falling here fails.
6. **Reference truth:** the task's object reference equals mocap within 1e-6 m after the loader, and the kit's score of the reference against mocap gives AUC above 0.99.

## Stage 1: training and scoring

M0b's recipe, unchanged: 1,024 worlds, ratio 2.0, capacity 1,048,576, `minimum_size` 6,400, checkpoint interval 504,832, D1's thresholds, seed 0, fresh start, hard 75-minute cooperative deadline, `GpuLease`. Horizon by the upstream formula at the final total. Episode length differs from the example; the thresholds stay fixed in transitions.

Each of 32,814,080 and 66,763,776 is scored once through the kit's evaluator with M0's settings. Metrics, on the exported achieved poses: `T`, `Q`, `E` against the task reference, and the kit's AUC, SP-SR, RPE and MPPE with `static_pose_baseline.score` against public mocap (39 is tuning; this is the label use the plan allows). The boundary row is descriptive. `E_S` is the held-still arm's mean position error on 39, recomputed from mocap at scoring and recorded.

## Gate (final policy, raw values; if the run misses its deadline, the last complete checkpoint, labelled)

- **GO:** `T >= 0.50`, `Q >= 0.50`, `E <= 0.5 E_S`, and kit AUC at least **0.275** (T1 0.175 plus 0.10).
- **SIGNAL:** `T >= 0.20`, `Q >= 0.10`, `E <= E_S`, and AUC above 0.175.
- **NULL:** anything else.

The lead independently recomputes `T`, `Q`, `E` and AUC from the parquet for any GO or SIGNAL.

## What each result does

| Result | Consequence |
| --- | --- |
| GO | M2v runs, and the scored route is budgeted from M2's measured minutes |
| SIGNAL | The hand reference or the recipe limits it. One disclosed variation, chosen before a second run: an object-driven reference without the acquisition phase's fixed timing, or the remaining schedule at the same recipe. Then M2v only if GO |
| NULL | W3 closes for this deadline; `D2 v5` stands; the finding goes in the report with whichever Stage 0 or training evidence explains it |

## M2v: the same with a video-derived reference (own short protocol before it runs)

Object reference from M1's per-frame estimate on 39, replacing mocap; everything else the same. M1's own gate comes first: its kinematic reference, scored against mocap with `static_pose_baseline.score`, beats T1 on the same episode by a margin fixed in its protocol, with a stated allowance for physical tracking error. The policy's gate is scored against public mocap as above.

## Scored stage (only after M2v's GO; written now so the adoption rules are not fitted later)

**Scored episodes have no labels. Per-episode adoption uses frozen video and physical rules, never a scored AUC.** Order: 38 (dust pan, 39's analogue), then 35, 36 and 37 (spoon, no scan; a spoon policy needs M1 without a scan), then pairs only after their own gates. An episode changes only if:
1. M2v passed;
2. M1's reference on that episode meets its label-free quality rule, fixed in M1's protocol;
3. the policy meets `T >= 0.50` and `Q >= 0.50` against its own task reference, and reaches its end without a reset, through the kit's evaluator;
4. the object's first pose matches its reference within 1 cm and 2 degrees;
5. the pack's `check_release.sh` is READY, its manifest does not pin `AGENTS.md`, and the abstract discloses the method.

Any failure keeps that episode's v5 export.

## Limits and integrity

A pass on 39 is one seed on one tuning episode. Mocap is available on train and tuning only, so M2's success is not reachable on scored episodes without M2v. Episodes 41, 43 and 44 are never read. Validation 2, 13, 22, 24 and 27 are not used. No kit edit. The outcome goes in `docs/chain/moving-m2-outcome.md`, whichever way it falls.
