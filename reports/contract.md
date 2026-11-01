# Track 3 contract

Revision 2, 2026-09-30, with design-review documentation corrections. This is a project record of what the organisers have
published. It is not the challenge rules.

The first revision, dated 2026-09-24, found no evaluator in any published source
and listed twelve unresolved requirements. It is kept unchanged as
[contract-2026-09-24.md](contract-2026-09-24.md). On 2026-09-29 the organisers
announced that submissions are open and published a Submission Kit. This revision
records what the kit and the source it names settle, and what they leave open.

**Eight of the twelve requirements are resolved, three are partly resolved and
one is open. The open one, the allowed inputs at evaluation, affects the design.**
The machine-readable form is `configs/evaluator/contract.json`.

## Method

Read-only source reading. No kit script, evaluator, packer or uploader was run.
No Parquet sample was opened. Three sub-agents and the coordinator read the
material; their reports hold the line-level citations.

| Source | Version | Report |
|---|---|---|
| Submission Kit archive, 691,560 bytes | SHA-256 `a2de2aeed5821c28efedf36ddbe08a7e595f9b7d7298e4be2c07fe4665cf91b3`, served with `Last-Modified: Tue, 29 Sep 2026 06:52:06 GMT` | [submission-kit.md](contract-evidence/2026-09-30/submission-kit.md) |
| FlashCHORD policy runtime | `nvidia-isaac/video_to_data` commit `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, named by the kit's manifest | [flashchord-runtime.md](contract-evidence/2026-09-30/flashchord-runtime.md) |
| Upstream retargeting and motion schema | Same commit | [upstream-retargeting.md](contract-evidence/2026-09-30/upstream-retargeting.md) |
| Challenge site, repository state, dataset state | Retrieved 2026-09-30 | [web-sources.md](contract-evidence/2026-09-30/web-sources.md) |

The web report came through a page fetcher that returns extracted text, so its
quotations should be checked against the pages before anyone relies on their
exact wording. The dataset is unchanged: its current revision equals the pinned
`5f68335f3acc802033d1e80728c1633197521de8`.

`KIT/`, `FC/` and `RG/` below mean the kit root, the FlashCHORD folder and the
upstream `robotic_grounding` folder under `upstream/`.

## What is submitted

| File | Content | Leaderboards |
|---|---|---|
| Pose CSV | 9,120 rows: for each of 20 episodes, world 0, every frame and every object slot, the object's pose during a policy rollout | AUC, SP-SR, MP-SR, RPE, MPPE |
| Mesh Parquet | One triangle mesh for each of 36 episode-object pairs | CD-O |

Each file also carries a GitHub commit URL. The pose file carries five SHA-256
values: the policy checkpoint per episode, the evaluator script, the evaluation
code, the packer and the uploader.

The website still says submissions are packaged with `eval_e2e.py`. No file of
that name exists in the kit or in any inspected source tree. The kit's
`evaluate_policy.py`, `tools/pack_submission.py` and
`tools/pack_reconstruction.py` do that work.

## Scored episodes

Twenty evaluation episodes: 1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32,
33, 35, 36, 37 and 38 (`KIT/config/tracks.json:206-227`). Five further held-out
episodes have no released video and are not scored.

| Episode | Slot 0 | Slot 1 | Frames |
|---|---|---|---|
| 1 | white_pot | white_pot_lid | 335 |
| 3 | white_pot | wooden_spoon | 313 |
| 5 | white_pot | wooden_spoon | 288 |
| 6 | white_pot | wooden_spoon | 243 |
| 8 | beige_cup | white_pot | 219 |
| 10 | blue_cup | beige_cup | 161 |
| 14 | blue_cup | plastic_dish_rack | 158 |
| 15 | plastic_dish_rack | wooden_spoon | 166 |
| 16 | plastic_dish_rack | wooden_spoon | 173 |
| 19 | mini_sweeper | mini_dust_pan | 235 |
| 20 | blue_cup | water_pitcher | 202 |
| 25 | wooden_piece_1 | wooden_piece_2 | 356 |
| 28 | plastic_dish_rack | white_pot | 220 |
| 30 | mini_sweeper | plastic_dish_rack | 263 |
| 32 | blue_cup | wooden_spoon | 259 |
| 33 | blue_cup | wooden_spoon | 348 |
| 35 | wooden_spoon | | 421 |
| 36 | wooden_spoon | | 407 |
| 37 | wooden_spoon | | 151 |
| 38 | mini_dust_pan | | 263 |

Total 5,181 frames. Slot order is the reference's order and is not always
alphabetical. The frame counts equal the decoded frame counts of the evaluation
clips recorded by this project's data audit on 2026-09-24, for all 20 episodes.

`wooden_spoon` is in ten scored episodes and has no mesh in the public release.

## Requirements from the first revision

| ID | Requirement | Status | Finding |
|---|---|---|---|
| U01 | Evaluator source and command | Resolved | The kit, with `evaluate_policy.py` matching upstream commit `aee95f25` |
| U02 | Embodiment and simulator | Partly | Simulator: FlashCHORD on Newton with the MuJoCo Warp solver. Embodiment: three are available and none is named as required. |
| U03 | Policy observations | Resolved | Defined by the runtime's recipes. They include each object's simulated pose and the current frame's reference targets. No separate rule restricts them. |
| U04 | Policy actions | Resolved | For floating hands, 56 residuals on the reference wrist pose and finger joints, at 20 control steps per second |
| U05 | Reference access | Resolved | The policy tracks a reference built by the participant from the video. The organisers' reference is confidential and never available. |
| U06 | Per-episode policies | Resolved | One policy per sequence |
| U07 | Permitted inputs at evaluation | **Open** | The kit says to reconstruct "from the released egocentric videos". Nothing states whether the stereo streams, the calibration file, the public scans or manual annotation may be used. |
| U08 | Reconstruction contents | Resolved | No hand, body or scene reconstruction is submitted. Object meshes are, for CD-O. |
| U09 | Rollout protocol | Resolved | One world, reset to frame 0 of the reference, deterministic policy, assistance off, failure terminations disabled, horizon equal to the reference length |
| U10 | Scoring | Partly | Formulas and thresholds are published in code. How the six metrics combine into a ranking, and tie-breaking, are not. |
| U11 | Submission schema | Resolved | See [submission-kit.md](contract-evidence/2026-09-30/submission-kit.md), sections 1, 2 and 5 |
| U12 | Registration and final rules | Partly | Team size 10, two final submissions per competition, Apache-2.0 release for winners, from the kit's configuration. The Kaggle rules text was not retrieved. The FAQ was not found. The submission week's reset time is unpublished. |

## Scoring

From `KIT/v2dlb/object_pose.py`, which the kit says is inlined into every Kaggle
scorer.

**Alignment.** For each episode, one rigid rotation and translation is computed
from object slot 0 at the first frame, so that its submitted pose equals its
reference pose. The same transform is applied to every object and frame. No scale
is fitted (lines 122-163).

**Metrics.**

| Metric | Definition | Lines |
|---|---|---|
| AUC | For each object, the mean distance between fixed object points under the submitted and reference poses. The share of frames under each of nine thresholds, 1 to 9 cm, is integrated. Mean over objects. No symmetry handling. | 80-104 |
| SP-SR | Success when position error, averaged over frames and objects, is at most 0.1 m and angle at most 0.5 rad | 173-178 |
| MP-SR | Success when every object's frame-averaged error is under 0.03 m and 30 degrees | 181-191 |
| RPE | Position of object 1 in object 0's frame, compared with the reference, in cm. Zero for single-object episodes. | 194-224 |
| MPPE | Mean error of six points 5 cm from each object's origin along its axes, in cm | 227-253 |

Each metric is averaged over the 20 episodes with equal weight. Four episodes
have one object, so their RPE is zero by definition.

**CD-O.** The scorer rescales the submitted mesh to the reference's size,
registers it by rotation and translation from up to 25 starting orientations, and
reports the symmetric mean surface distance in cm over 10,000 sampled points per
surface (`KIT/v2dlb/object_mesh.py:59-216`). Units, origin and orientation of the
submitted mesh do not affect it.

**No partial credit.** A submission missing any roster row is rejected
(`KIT/metric_code/track_3/AUC.py:399-405`).

**No local score.** The reference is confidential, so the packer cannot compute
Track 3 metrics (`KIT/tools/pack_submission.py:447-451`).

## Rules encoded in the kit's configuration

From `KIT/config/tracks.json`. The file's own comment says the website prevails
where they disagree.

| Item | Value | Line |
|---|---|---|
| Submission limit | Five per week, unlimited in the final three days | 304, 333 |
| Kaggle daily limit as configured | 5 | 325 |
| Final submissions per competition | 2 | 326 |
| Maximum team size | 10 | 324 |
| Public and private split | All rows public | 23 |
| Winner's licence | Apache-2.0 for the submission and its source. Pretrained models and data under incompatible licences are exempt. | 334-337 |
| Freeze | 4 November 2026, 17:00 EST | 301 |
| Evaluator hash allowlists | Empty, which disables the check. Entries may be appended later and do not rescore earlier submissions. | 17-22, 27-28 |

## What the code enforces and what the documents only say

| Statement in the kit's documents | Enforced by code |
|---|---|
| Use the official evaluator | No. The packer accepts any Parquet of the right shape, including a documented legacy layout, and the allowlists are empty. |
| The commit must be in a repository shared with the organisers | No. Only the URL's shape is checked. |
| Positions are in metres | No. The unit is never read. |
| Frames are the dataset's frames | No. Rows are matched by integer, and extra frames are dropped silently. |

This project follows the documents, not the gaps. See decision D03 in
[decisions.md](../docs/design/decisions.md).

## Findings that shaped the design

| Finding | Consequence |
|---|---|
| The released floating-hand recipe plays the reference at half speed, so the evaluator writes two steps per dataset frame | Train at unit motion speed. See [policy_rollout](../track3/features/policy_rollout/README.md). |
| Only the floating-hand route accepts two tracked objects from egocentric input | Floating Sharpa hands |
| The retargeter needs both hands on every frame, with joint orientations in MANO's convention | Hand completion in scene assembly; MANO kinematics |
| The alignment removes a world-side offset only | Public scans for tracking and simulation, so body frames match the reference |
| The runtime documents 48 GiB for full training | A measured budget before any plan depends on training time |
| The runtime's logging defaults to online upload | `logging.mode=disabled` on every command |

## Open

| Question | Blocks |
|---|---|
| Permitted inputs at evaluation (U07) | Whether stereo, calibration and public scans may be used as designed |
| Body frame of `wooden_spoon` | Accuracy on ten episodes |
| How rollout frames are meant to align with dataset frames | Confidence in the unit-speed choice |
| Required embodiment, if any | Floating hands |
| Combined ranking and tie-breaking (U10) | Which metric to favour when they trade off |
| Submission week reset, Kaggle rules text, FAQ location (U12) | Quota planning |
| Whether the official evaluator is mandatory and evaluator-hash allowlists will be filled | Evaluator choice and reproducible LF source bytes (D03, R14) |
| Reference rows where motion capture lost the object | Expected score floor and treatment of invisible frames |

The full list, in the order to ask, is in
[risks and open questions](../docs/design/risks-and-open-questions.md).

The organiser announcement supplied by the user is recorded in
[organiser-announcement.md](contract-evidence/2026-09-30/organiser-announcement.md).
It supplies the webinar time, Discord channel, early-submission request and
approximate L40S training claim that the earlier web extraction did not retrieve.
It does not resolve the open questions above.

## Before packaging

Compare the kit archive's hash with the value above. The kit is served from an
unversioned URL, and upstream commits on 2026-09-29 already changed its scoring
code and Track 2 conversion after first publication.
