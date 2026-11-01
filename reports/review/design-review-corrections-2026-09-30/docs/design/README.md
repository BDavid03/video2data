# Track 3 design documents

Design set for the NVIDIA Video-to-Data Challenge, Track 3 (Egocentric). Started
2026-09-30, after the organisers published the Submission Kit and opened the
leaderboards. These documents describe what the project intends to build and why.
They are not evidence that anything runs.

**Execution policy.** `AGENTS.md` prohibits tests, smoke checks, validation commands
and model execution until the user changes that instruction. Nothing in this design
set was run, imported, compiled or measured while writing it. Every operation
described here is disabled by default and needs its own explicit authorization.

## Reading order

| # | Document | Answers |
|---|---|---|
| 1 | [System overview](system-overview.md) | What is scored, the end-to-end flow, the slices and the runtime environments |
| 2 | [Artifact contracts](artifact-contracts.md) | What each slice reads and writes, in which frame, unit and clock |
| 3 | [Decisions](decisions.md) | Choices made, alternatives rejected, and what would reverse each one |
| 4 | [Risks and open questions](risks-and-open-questions.md) | What can sink the entry, and which questions go to the organisers |
| 5 | [Roadmap](roadmap.md) | Dated plan to the 4 November freeze, with gates that need the user |
| 6 | [Feature design template](feature-template.md) | The structure every feature README follows |

The official contract is recorded separately, because it is evidence and not design:
[contract report](../../reports/contract.md), with its machine-readable form in
`configs/evaluator/contract.json` and the source readings under
`reports/contract-evidence/2026-09-30/`.

The questions for the organisers are also written out as posts the user can send:
[organiser questions](../../reports/organiser-questions-2026-09-30.md).

The [design review](../../reports/review/design-review-2026-09-30.md) is retained
unchanged. Its 39 findings are addressed in the
[response record](../../reports/review/design-review-response-2026-09-30.md).
The [organiser announcement supplied by the user](../../reports/contract-evidence/2026-09-30/organiser-announcement.md)
records the submission opening, RL claim, quota, Discord channel and webinar.

## Feature designs

Vertical slice architecture keeps each feature's design next to its code. A slice
marked *design only* has a README and no implementation yet.

| Slice | Folder | State |
|---|---|---|
| Monocular depth (MoGe2) | [`track3/features/depth/`](../../track3/features/depth/README.md) | Implemented, never run |
| Coursework depth CNN | [`track3/features/scratch_depth/`](../../track3/features/scratch_depth/README.md) | Implemented, never run |
| Stereo depth and metric scale | [`track3/features/stereo_depth/`](../../track3/features/stereo_depth/README.md) | Design only |
| Object and hand masks | [`track3/features/object_masks/`](../../track3/features/object_masks/README.md) | Design only |
| Camera motion | [`track3/features/camera_motion/`](../../track3/features/camera_motion/README.md) | Design only |
| Object mesh | [`track3/features/object_mesh/`](../../track3/features/object_mesh/README.md) | Design only |
| Object pose | [`track3/features/object_pose/`](../../track3/features/object_pose/README.md) | Design only |
| Hand motion | [`track3/features/hand_motion/`](../../track3/features/hand_motion/README.md) | Design only |
| Scene assembly | [`track3/features/scene_assembly/`](../../track3/features/scene_assembly/README.md) | Design only |
| Retargeting | [`track3/features/retargeting/`](../../track3/features/retargeting/README.md) | Design only |
| Policy training | [`track3/features/policy_training/`](../../track3/features/policy_training/README.md) | Design only |
| Policy rollout | [`track3/features/policy_rollout/`](../../track3/features/policy_rollout/README.md) | Design only |
| Submission | [`track3/features/submission/`](../../track3/features/submission/README.md) | Design only |

## Conventions used in every document

**Claim labels.** Each non-obvious statement carries one of four labels, so a reader
can tell a published rule from a plan.

| Label | Meaning |
|---|---|
| **Official** | Stated by an organiser-published source: the Submission Kit, the pinned upstream source, the challenge site or the dataset. A citation follows. |
| **Measured** | Produced by a recorded local run. Only runs from before the no-testing instruction qualify. |
| **Design** | This project's choice. Not validated. |
| **Open** | Unresolved. The entry names who can resolve it and how. |

**Citations.** `KIT/` means `upstream/v2d_submission_kit/v2d_submission_kit/`.
`FC/` means `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/`.
`RG/` means `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/`.
All three are read-only inputs pinned in `configs/sources.lock.json`.

**Names.** An *episode* is one recorded demonstration, identified by
`(source_split, episode_index)`. *Public* episodes have pose labels; *evaluation*
episodes have video only and are the ones scored. The same integer can appear in
both splits and never means the same recording.

**Frames and units.** Metres, seconds and radians. Poses are written
`parent_T_child` and map child coordinates into the parent. Quaternion order is
stated at every interface, because the dataset stores scalar-first values and the
scorer uses scalar-last.
