# Design notes

Most pages here date from October 2026, before the entry was built, and describe a
larger planned system. Treat them as history. The current state is in the
[README](../../README.md) and the [submission ledger](../chain/SUBMISSION.md).

Older pages may name `track3/`, `outputs/` or `reports/`. These are now `src/`,
`data/outputs/` and `data/reports/`. Older design notes and retired slices were deleted
on 4 Nov 2026; they remain in Git history before the cleanup commit (parent `ef884fd`).

| Topic | Page |
| --- | --- |
| Artifacts | [Artifact contracts](artifact-contracts.md) |
| Design tradeoffs | [Decisions](decisions.md) |
| Adding a feature | [Feature template](feature-template.md), [code style](simplification-standard.md) |
| Competition contract | [Contract report](../../data/reports/contract.md), [source reading](../../data/reports/contract-evidence/2026-09-30/submission-kit.md) |

## Feature directory

Each folder owns its requests, handlers, runtime flags and README. Being listed here
does not mean an operation is enabled or has been run.

| Feature | What it does | Operations |
| --- | --- | --- |
| [Depth](../../src/features/depth/README.md) | Calibrated monocular depth | generate, offline challenger ingestion |
| [Stereo depth](../../src/features/stereo_depth/README.md) | Stereo geometry and metric scale | rectify, generate, anchor, correspondences, report |
| [Object masks](../../src/features/object_masks/README.md) | Opening clusters and full-video masks | opening-clusters, generate, correspondences |
| [Camera motion](../../src/features/camera_motion/README.md) | Metric camera path and table world | estimate |
| [Mesh extents](../../src/features/mesh_extents/README.md) | Video measurements and procedural meshes | measure, build, rules, adopt, proxy |
| [Mesh evaluation](../../src/features/mesh_eval/README.md) | Packed mesh scoring on public scans | pairs, local CD-O script |
| [Mesh generative](../../src/features/mesh_generative/README.md) | Pretrained shape campaigns | smoke, pilot, g2, g3, insights |
| [Pose insurance](../../src/features/pose_insurance/README.md) | Static pose policies and packing | scene build, policy run, diagnostics, packing |
| [Moving reference](../../src/features/moving_reference/README.md) | Object path from video for episodes 39 and 38 | masks, track, candidate C, gate records |
| [Moving policy](../../src/features/moving_policy/README.md) | Moving RL policy on episodes 39 and 38 | stage0, train, score, video variants |
| [Policy training](../../src/features/policy_training/README.md) | General training tools, never run; only `budget.json` is in use | compose, train, select, comparison |
| [Mesh fusion](../../src/features/mesh_fusion/README.md) | Failed fusion route, kept as evidence | stage-a |
| [Dynamic feasibility](../../src/features/dynamic_feasibility/README.md) | Dynamic-route evidence cited in the abstract | feasibility and scaling queues |

Policy training's compose, train, select and comparison need the retargeting, scene_assembly
and policy_rollout producers, which are in Git history before the cleanup commit.

## Labels on older pages

Feature pages may carry these labels: Official (from pinned local sources), Measured (from
a named historical report), Derived (arithmetic, no measurement), Design (thresholds not
proven as tolerances) and Open. None of them is a fresh external check. A completed run
can still contain invalid measurements. The core design pages were rebuilt from source
after their files were found unreadable, so they do not recover the original decision
history.
