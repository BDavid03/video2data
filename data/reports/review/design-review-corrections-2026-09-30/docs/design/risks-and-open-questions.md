# Risks and open questions

Written 2026-09-30, 35 days before the leaderboard freeze. Risks are ordered by
how much of the entry they can take down. Nothing in the project has run since
the no-testing instruction, so every likelihood below is a judgement, not a
measurement.

## Risks

| # | Risk | What it costs | First evidence that would settle it |
|---|---|---|---|
| R01 | Policy training does not fit in 16 GB, or is too slow to train 20 policies | The whole pose submission | A short training run at reduced world count, with device memory and steps per second recorded |
| R02 | The policy stack does not run under WSL 2 on the RTX 5080 | The whole pose submission | Loading the upstream example sequence in the scene viewer |
| R03 | No run is authorized in time | Everything | The user's decision |
| R04 | The body frame of `wooden_spoon` stays unknown | Accuracy on 10 of 20 episodes | An organiser answer |
| R05 | Rollout frames do not line up with video frames | Every pose metric, silently | An organiser answer; the `inspect` check in policy_rollout |
| R06 | Metric scale is wrong | Every pose metric | Measuring a scanned object's known size in stereo depth |
| R07 | The first-frame pose of object slot 0 is wrong | Every pose metric for that episode | The spread of the opening-frame poses |
| R08 | A perception dependency cannot be built for this GPU or used under its licence | The affected slice falls back to a weaker method | A build attempt; an organiser answer on licences |
| R09 | Hands are unusable in too many frames | Policies that do not grasp | Hand overlays on development episodes |
| R10 | One episode has no valid rollout | The submission is rejected whole | The fallback ladder in policy_rollout |
| R11 | A packing mistake spends quota | One of five weekly submissions per mistake | The `check` operation in submission |
| R12 | The rules change or are clarified against a design choice | Rework | Organiser answers, the 7 October webinar |
| R13 | The code cannot be released as required | Award eligibility | A licence review before the first commit |
| R14 | CRLF conversion changes the evaluator-code hash | Pose uploads rejected if organiser allowlists are filled | Future comparison of rollout metadata with the LF source hash and published allowlists |

### R01. Training on 16 GB

The runtime's README states: "Full training: 48 GiB GPU with ≥46,000 MiB free"
(**Official**, `FC/README.md:14`). The released floating-hand recipe uses 4,096
parallel worlds, a replay buffer of 4,194,304 transitions and 250 million
environment steps (**Official**,
`FC/src/flash_chord/configs/experiment/sharpa_flash_sac.yaml`). The organisers'
message gives about two hours on a single L40S; this is an organiser claim, not
a measurement on this machine (see the
[announcement record](../../reports/contract-evidence/2026-09-30/organiser-announcement.md)).

This machine has one RTX 5080 with 16 GB, of which Windows and other processes held about
2.4 GiB at the last measurement (**Measured**, `reports/runtime/README.md`).

World count and replay capacity are configuration values, and the README shows
`scene.world_count=1024` as an override. What a smaller world count does to
training quality and time is not stated anywhere in the source. Twenty policies
have to be trained at least once, and preferably twice, between 12 October and
1 November.

**If it does not fit or is too slow,** the choice is the user's: rent a larger
GPU, which breaks the local-hardware constraint in `TRACK3-PLAN.md`, or accept
weaker policies trained with fewer worlds and steps.

### R03. No run authorized

`AGENTS.md` forbids tests, smoke checks, validation commands and model execution.
The design set can be finished under that rule. The entry cannot. The
[roadmap](roadmap.md) gives the latest date for each authorization the plan
needs. The first is 2 October.

### R04. The spoon

`wooden_spoon` is in ten scored episodes and is slot 0 in three. It has no public
scan, so its reference body frame is unpublished. The scorer's alignment removes
a world-side offset only (**Official**, `KIT/v2dlb/object_pose.py:122-163`). A
body-frame offset is cancelled at frame 0 for slot 0, then can produce error as
the object moves, including translation if the offset has a rotation. For any
other slot the offset is never removed and is scored in full from the first
frame. `wooden_spoon` is slot 1 in seven of its ten episodes: 3, 5, 6, 15, 16,
32 and 33 (`KIT/data/track_3_evaluation_objects.json`).

The spoon is also the hardest object to reconstruct and track: thin, with little
texture, and partly hidden in the hand whenever it is used.

### R05. Frame alignment

Covered in [policy_rollout](../../track3/features/policy_rollout/README.md). The
danger is that the failure is silent: the packer accepts the file.

### R07. The anchor pose

An orientation error of 10 degrees in slot 0's first pose moves a point 30 cm
away by about 5 cm after alignment. MP-SR's position gate is 3 cm. Objects that
look the same from every side, such as cups, have an orientation about their axis
that only texture or a handle can fix, and AUC has no symmetry handling
(**Official**, `KIT/v2dlb/object_pose.py:80-104`).

### R13. Code release

Winners must publish their submission's source under Apache-2.0 (**Official**,
`KIT/config/tracks.json:334-337`). Pretrained models under incompatible licences
are exempt from release, by the same text. Three things need attention before
the first commit:

| Item | Issue |
|---|---|
| `coursework/` | Course material and two 347 MB embedding files. It may not be the user's to publish, and the large files do not belong in Git. |
| The MANO kinematics library | GPL-3.0. Kept in a separate environment and run as a separate process, as upstream does. |
| MANO model files, hand and pose model weights | Licence-gated or non-commercial. Never committed. |

### R14. Evaluator source bytes

Review E-04 records CRLF endings in all four evaluation modules in the Windows
reference checkout. The evaluator hashes their installed raw bytes
(`FC/src/flash_chord/evaluation/harness.py:123-135`), so installing from that
checkout would produce a different hash from an LF build of the same commit.
Filled allowlists reject other hashes (`KIT/config/tracks.json:17-22`).
The proposed installation source is an LF checkout of `aee95f25` inside the
project-owned WSL disk, with CRLF conversion disabled. Keep the Windows reference
unchanged. Future rollout inspection and submission checks compare the recorded
hash with those LF sources and any organiser allowlist; nothing is checked now.

## Questions for the organisers

In priority order. The user posts them; Discord `#v2d-challenge` and the webinar
on 7 October at 11:00 Pacific (8 October, 05:00 in Sydney) are the channels the
organisers named. The same questions are written out as ready-to-send posts in
[organiser questions](../../reports/organiser-questions-2026-09-30.md).

| # | Question | Decides |
|---|---|---|
| Q01 | The released floating-hand recipe plays the reference at half speed, so the evaluator writes two steps per dataset frame, and the packer matches rows by integer frame. How do your baselines keep rollout frames equal to dataset frames? Is training at `motion_speed=1.0` the intended way? | D06 |
| Q02 | `wooden_spoon` appears in ten scored Track 3 episodes and has no mesh in the public release. Will its scan be released? If not, what is the origin and orientation of its reference body frame? | D09, R04 |
| Q03 | At evaluation, may a method use the two mono stereo streams, the calibration file published under `public/meta/`, and the public object scans and URDFs? | D07, D09 |
| Q04 | Must rollouts come from the official `evaluate_policy.py`? Will the evaluator-hash allowlists be filled, and would that invalidate earlier submissions? | D03 |
| Q05 | Is any embodiment required for Track 3? | D05 |
| Q06 | For CD-O, may the public scans be submitted? May one reconstruction be submitted for several episodes that contain the same object? | D09 |
| Q07 | When does the submission week reset, and when exactly do the final three days begin? How many final submissions are selected per leaderboard? | Submission ledger |
| Q08 | Does the confidential reference hold a pose for frames where motion capture lost the object? How are those rows scored? | Expected score floor |
| Q09 | Is manual annotation of an evaluation video allowed, for example one click to start a mask? | object_masks fallback |
| Q10 | May a method use pretrained models released under non-commercial or research-only licences? | hand_motion, object_pose |
| Q11 | What mass and friction do your reference scenes use for the objects? The public URDFs mark their mass as a placeholder. | retargeting |
| Q12 | Is training on a GPU with less than 48 GB supported? Which settings do you recommend for 16 GB? | R01 |
| Q13 | Where is the FAQ? It is not linked from the challenge page. | All of the above |

## Questions only a run can answer

| Question | Slice |
|---|---|
| How much device memory and time does training take at 256, 512 and 1,024 worlds? | policy_training |
| Is the stereo scale stable within an episode? | stereo_depth |
| Does feature tracking hold when hands cover the image centre? | camera_motion |
| How far do opening-frame object poses scatter? | object_pose |
| In what share of frames is each hand usable? | hand_motion |
| Does a resting object stay still on the simulator's ground plane? | scene_assembly |
| Does the policy track a full-speed reference? | policy_training |

## Questions only the user can answer

| Question | Needed by |
|---|---|
| Will execution be authorized, and for which operations first? | 2 October |
| May an isolated Linux environment be created under `.runtime/wsl/` and the policy stack installed in it? | 2 October |
| Will you create the GitHub repository, the Kaggle token, and accept the six competitions' rules? | 4 October |
| Will you register for MANO and download the model files? | 4 October |
| What happens to `coursework/` when the project becomes a repository? | Before the first commit |
| If training does not fit on 16 GB, is a rented GPU acceptable? | When R01's measurement arrives |
