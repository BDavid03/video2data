# Design notes

How the code is organised. For what we did and the scores, read the [README](../../README.md) first.

| Topic | Page |
| --- | --- |
| Run records and file formats | [Artifact contracts](artifact-contracts.md) |
| Design choices | [Decisions](decisions.md) |
| Adding a feature | [Feature template](feature-template.md), [code style](simplification-standard.md) |
| Competition rules | [Contract report](../../data/reports/contract.md), [kit reading](../../data/reports/contract-evidence/2026-09-30/submission-kit.md) |

Some older records name `track3/`, `outputs/` or `reports/`. These are now `src/`, `data/outputs/`
and `data/reports/`. `src/paths.py` maps the old names through `scripts/layout/mapping.json`.
Deleted notes and retired features are in Git history before the cleanup commit (parent `ef884fd`).

## Feature directory

A *slice* is one folder under `src/features/`. It holds one feature's requests, code, run flags
(`runtime.json`) and README. Run one with:

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.<slice> <operation> --request <file.json>
```

Being listed here does not mean an operation is enabled. Each slice checks its own flags first.

The step numbers match "What we did" in the README: 2 shapes, 3 static poses, 4 episode 38 moving
policy, 5 submission.

| Slice | What it does | Step |
| --- | --- | --- |
| [depth](../../src/features/depth/README.md) | Per-frame MoGe-2 depth on a calibrated image | 2 |
| [stereo_depth](../../src/features/stereo_depth/README.md) | Stereo depth and metric scale; places static objects | 2, 3 |
| [object_masks](../../src/features/object_masks/README.md) | Object masks for every video frame | 2, 3 |
| [camera_motion](../../src/features/camera_motion/README.md) | Camera path and a table-based world frame | 3, 4 |
| [mesh_extents](../../src/features/mesh_extents/README.md) | Measures object size from depth; builds procedural meshes | 2 |
| [mesh_eval](../../src/features/mesh_eval/README.md) | Scores packed meshes against public scans (CD-O) | 2 |
| [mesh_generative](../../src/features/mesh_generative/README.md) | Pretrained shape generator trials; not in the final file | 2 |
| [mesh_fusion](../../src/features/mesh_fusion/README.md) | Depth fusion trial that failed; kept as evidence | 2 |
| [pose_insurance](../../src/features/pose_insurance/README.md) | Static scenes, short RL policy runs, pose packing | 3, 5 |
| [moving_reference](../../src/features/moving_reference/README.md) | Tracks the object path in video (episodes 39 and 38) | 4 |
| [moving_policy](../../src/features/moving_policy/README.md) | Trains and scores the moving RL policy (episodes 39 and 38) | 4 |
| [dynamic_feasibility](../../src/features/dynamic_feasibility/README.md) | Early tests that a moving policy can learn at all | 4 |
| [policy_training](../../src/features/policy_training/README.md) | General training tools, never run; only `budget.json` is read | none |
| `foundationpose_pilot.py` | One-off FoundationPose pose pilot; not in the entry ([install](../INSTALL.md)) | none |

`policy_training`'s compose, train, select and comparison need the retargeting, scene_assembly and
policy_rollout slices. Those are in Git history before the cleanup commit.

## Terms

- *Episode*: one recorded video of a person doing a tabletop task. The challenge numbers them.
- *Run record*: the `run.json` a slice writes beside its outputs (request, input hashes, status).
- *Pack*: the kit's tool that turns our results into the one Parquet file Kaggle accepts.
- *Admission gate*: a check, written before the run, that a result must pass before it is packed.
- *CD-O*: surface Chamfer distance between our mesh and the object scan after aligning the two, in cm.
  Lower is better.
- *AUC*: pose accuracy (ADD) averaged over 1 to 9 cm error thresholds. Higher is better.
