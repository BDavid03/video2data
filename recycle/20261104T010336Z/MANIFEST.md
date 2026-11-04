# Recycled 20261104T010336Z

Restructure phase 5 (`docs/chain/restructure-plan.md`; owner go-ahead in `docs/chain/PROGRAM.md` section 0). Nothing here is deleted.
Restore an entry with `git mv recycle/20261104T010336Z/<original path> <original path>`; recycled slices import each other by their original `track3.features.<slice>` names, so restore a dependent group together.

Shared reason for every slice below: No deliverable's reproduction reaches it, no pack manifest pins it, and no code that stays imports it.

## `track3/features/auto_research`

Closed loop of the finished depth study; only `scripts/research_loop.py` imported it. Its ledger and frozen selection stay under `outputs/auto_research/`.

17 tracked files:

- `track3/features/auto_research/README.md`
- `track3/features/auto_research/__init__.py`
- `track3/features/auto_research/__main__.py`
- `track3/features/auto_research/_common.py`
- `track3/features/auto_research/advisor.py`
- `track3/features/auto_research/campaign.py`
- `track3/features/auto_research/finalize.json`
- `track3/features/auto_research/finalize.py`
- `track3/features/auto_research/ledger.py`
- `track3/features/auto_research/loop.json`
- `track3/features/auto_research/loop.py`
- `track3/features/auto_research/metrics.py`
- `track3/features/auto_research/recipes.py`
- `track3/features/auto_research/report.json`
- `track3/features/auto_research/report.py`
- `track3/features/auto_research/runtime.json`
- `track3/features/auto_research/visualize.py`

## `track3/features/depth_student`

Depth study model family, frozen on 2026-10-29; the abstract does not use it. Its ONNX export and runs stay under `outputs/`.

14 tracked files:

- `track3/features/depth_student/README.md`
- `track3/features/depth_student/__init__.py`
- `track3/features/depth_student/__main__.py`
- `track3/features/depth_student/_common.py`
- `track3/features/depth_student/_config.py`
- `track3/features/depth_student/_losses.py`
- `track3/features/depth_student/_models.py`
- `track3/features/depth_student/export.json`
- `track3/features/depth_student/export.py`
- `track3/features/depth_student/predict.json`
- `track3/features/depth_student/predict.py`
- `track3/features/depth_student/runtime.json`
- `track3/features/depth_student/train.json`
- `track3/features/depth_student/train.py`

## `track3/features/learning_experiments`

Depth study scorer and model families, imported only by other recycled depth-study slices.

37 tracked files:

- `track3/features/learning_experiments/README.md`
- `track3/features/learning_experiments/__init__.py`
- `track3/features/learning_experiments/__main__.py`
- `track3/features/learning_experiments/_common.py`
- `track3/features/learning_experiments/_config.py`
- `track3/features/learning_experiments/boost-predict.json`
- `track3/features/learning_experiments/boost-train.json`
- `track3/features/learning_experiments/boosting.py`
- `track3/features/learning_experiments/boosting_operations.py`
- `track3/features/learning_experiments/checkpoints.py`
- `track3/features/learning_experiments/data.py`
- `track3/features/learning_experiments/ensemble.json`
- `track3/features/learning_experiments/ensemble.py`
- `track3/features/learning_experiments/ensemble_operations.py`
- `track3/features/learning_experiments/execute.json`
- `track3/features/learning_experiments/execute.py`
- `track3/features/learning_experiments/geometry-freeze.json`
- `track3/features/learning_experiments/geometry-selection.json`
- `track3/features/learning_experiments/investigate.json`
- `track3/features/learning_experiments/investigate.py`
- `track3/features/learning_experiments/models.py`
- `track3/features/learning_experiments/plan.json`
- `track3/features/learning_experiments/planning.py`
- `track3/features/learning_experiments/predict.json`
- `track3/features/learning_experiments/predict.py`
- `track3/features/learning_experiments/report.json`
- `track3/features/learning_experiments/report.py`
- `track3/features/learning_experiments/runtime.json`
- `track3/features/learning_experiments/score.json`
- `track3/features/learning_experiments/score.py`
- `track3/features/learning_experiments/scratch_adapter.py`
- `track3/features/learning_experiments/search.json`
- `track3/features/learning_experiments/selection.py`
- `track3/features/learning_experiments/train.json`
- `track3/features/learning_experiments/train.py`
- `track3/features/learning_experiments/tune.json`
- `track3/features/learning_experiments/tune.py`

## `track3/features/scratch_depth`

Coursework depth model of the depth study, imported only by other recycled depth-study slices.

26 tracked files:

- `track3/features/scratch_depth/README.md`
- `track3/features/scratch_depth/__init__.py`
- `track3/features/scratch_depth/__main__.py`
- `track3/features/scratch_depth/_checkpoint.py`
- `track3/features/scratch_depth/_common.py`
- `track3/features/scratch_depth/_cuda_loss.py`
- `track3/features/scratch_depth/_network.py`
- `track3/features/scratch_depth/_optimization.py`
- `track3/features/scratch_depth/_research_network.py`
- `track3/features/scratch_depth/compiler_experiments.py`
- `track3/features/scratch_depth/coursework.json`
- `track3/features/scratch_depth/cuda/masked_log_depth.cpp`
- `track3/features/scratch_depth/cuda/masked_log_depth.cu`
- `track3/features/scratch_depth/external-compiler.json`
- `track3/features/scratch_depth/external_cuda_experiment.py`
- `track3/features/scratch_depth/predict.json`
- `track3/features/scratch_depth/predict.py`
- `track3/features/scratch_depth/prepare.json`
- `track3/features/scratch_depth/prepare.py`
- `track3/features/scratch_depth/research-predict.json`
- `track3/features/scratch_depth/research-prepare.json`
- `track3/features/scratch_depth/research-train.json`
- `track3/features/scratch_depth/research.py`
- `track3/features/scratch_depth/runtime.json`
- `track3/features/scratch_depth/train.json`
- `track3/features/scratch_depth/train.py`

## `track3/features/opening_experiments`

Depth study experiments; nothing imports it, and its recorded result was consumed only by recycled `learning_experiments`.

6 tracked files:

- `track3/features/opening_experiments/README.md`
- `track3/features/opening_experiments/__init__.py`
- `track3/features/opening_experiments/__main__.py`
- `track3/features/opening_experiments/run.json`
- `track3/features/opening_experiments/run.py`
- `track3/features/opening_experiments/runtime.json`

## `track3/features/synthetic_depth`

Synthetic depth data of the depth study; it ran, its datasets stay under `outputs/`, and only recycled `learning_experiments` consumed them.

6 tracked files:

- `track3/features/synthetic_depth/README.md`
- `track3/features/synthetic_depth/__init__.py`
- `track3/features/synthetic_depth/__main__.py`
- `track3/features/synthetic_depth/generate.json`
- `track3/features/synthetic_depth/generate.py`
- `track3/features/synthetic_depth/runtime.json`

## `track3/features/hand_motion`

Unrun scaffolding for the moving-object route, paused for this deadline. Restore it with `git mv` if that route resumes.

22 tracked files:

- `track3/features/hand_motion/README.md`
- `track3/features/hand_motion/__init__.py`
- `track3/features/hand_motion/__main__.py`
- `track3/features/hand_motion/_common.py`
- `track3/features/hand_motion/_mano.py`
- `track3/features/hand_motion/_wilor.py`
- `track3/features/hand_motion/backends.json`
- `track3/features/hand_motion/contacts.json`
- `track3/features/hand_motion/contacts.py`
- `track3/features/hand_motion/estimate.json`
- `track3/features/hand_motion/estimate.py`
- `track3/features/hand_motion/kinematics.json`
- `track3/features/hand_motion/kinematics.py`
- `track3/features/hand_motion/lift.json`
- `track3/features/hand_motion/lift.py`
- `track3/features/hand_motion/parked-template.json`
- `track3/features/hand_motion/parked_template.py`
- `track3/features/hand_motion/preview.json`
- `track3/features/hand_motion/preview.py`
- `track3/features/hand_motion/refine.json`
- `track3/features/hand_motion/refine.py`
- `track3/features/hand_motion/runtime.json`

## `track3/features/object_mesh`

Unrun scaffolding; `mesh_extents` and `mesh_generative` build the meshes. D1 scoring's `v2dlb.object_mesh` is kit code, not this slice.

15 tracked files:

- `track3/features/object_mesh/README.md`
- `track3/features/object_mesh/__init__.py`
- `track3/features/object_mesh/__main__.py`
- `track3/features/object_mesh/_common.py`
- `track3/features/object_mesh/_fusion.py`
- `track3/features/object_mesh/_points.py`
- `track3/features/object_mesh/fuse.json`
- `track3/features/object_mesh/fuse.py`
- `track3/features/object_mesh/reconstruct-tsdf.json`
- `track3/features/object_mesh/reconstruct.json`
- `track3/features/object_mesh/reconstruct.py`
- `track3/features/object_mesh/resolve.json`
- `track3/features/object_mesh/resolve.py`
- `track3/features/object_mesh/runtime.json`
- `track3/features/object_mesh/sources.json`

## `track3/features/object_pose`

Unrun scaffolding; `pose_insurance` makes the poses.

13 tracked files:

- `track3/features/object_pose/README.md`
- `track3/features/object_pose/__init__.py`
- `track3/features/object_pose/__main__.py`
- `track3/features/object_pose/_common.py`
- `track3/features/object_pose/compose.json`
- `track3/features/object_pose/compose.py`
- `track3/features/object_pose/initialise.json`
- `track3/features/object_pose/initialise.py`
- `track3/features/object_pose/refine.json`
- `track3/features/object_pose/refine.py`
- `track3/features/object_pose/runtime.json`
- `track3/features/object_pose/track.json`
- `track3/features/object_pose/track.py`

## `track3/features/policy_rollout`

Unrun scaffolding; `pose_insurance` runs the kit's evaluator directly.

11 tracked files:

- `track3/features/policy_rollout/README.md`
- `track3/features/policy_rollout/__init__.py`
- `track3/features/policy_rollout/__main__.py`
- `track3/features/policy_rollout/_common.py`
- `track3/features/policy_rollout/achieved_comparison.py`
- `track3/features/policy_rollout/compare-achieved.json`
- `track3/features/policy_rollout/inspect.json`
- `track3/features/policy_rollout/operations.py`
- `track3/features/policy_rollout/rollout.json`
- `track3/features/policy_rollout/runtime.json`
- `track3/features/policy_rollout/self-score.json`

## `track3/features/retargeting`

Unrun scaffolding for the moving-object route, paused for this deadline.

12 tracked files:

- `track3/features/retargeting/README.md`
- `track3/features/retargeting/__init__.py`
- `track3/features/retargeting/__main__.py`
- `track3/features/retargeting/_common.py`
- `track3/features/retargeting/_evidence.py`
- `track3/features/retargeting/assets.json`
- `track3/features/retargeting/export.json`
- `track3/features/retargeting/inspect.json`
- `track3/features/retargeting/operations.py`
- `track3/features/retargeting/retarget.json`
- `track3/features/retargeting/runtime.json`
- `track3/features/retargeting/support.json`

## `track3/features/scene_assembly`

Unrun scaffolding; `pose_insurance` assembles its own scenes.

16 tracked files:

- `track3/features/scene_assembly/README.md`
- `track3/features/scene_assembly/__init__.py`
- `track3/features/scene_assembly/__main__.py`
- `track3/features/scene_assembly/_common.py`
- `track3/features/scene_assembly/_contacts.py`
- `track3/features/scene_assembly/_geometry.py`
- `track3/features/scene_assembly/_hands.py`
- `track3/features/scene_assembly/assemble.json`
- `track3/features/scene_assembly/assemble.py`
- `track3/features/scene_assembly/parked_hand.json`
- `track3/features/scene_assembly/prepare_refinement.json`
- `track3/features/scene_assembly/prepare_refinement.py`
- `track3/features/scene_assembly/refine.json`
- `track3/features/scene_assembly/refine.py`
- `track3/features/scene_assembly/refinement-design.json`
- `track3/features/scene_assembly/runtime.json`

## `track3/features/campaign`

Unrun scaffolding of the general pipeline; packs use `pose_insurance` and `mesh_generative`.

18 tracked files:

- `track3/features/campaign/README.md`
- `track3/features/campaign/__init__.py`
- `track3/features/campaign/__main__.py`
- `track3/features/campaign/_planning.py`
- `track3/features/campaign/board.json`
- `track3/features/campaign/evidence.json`
- `track3/features/campaign/evidence.py`
- `track3/features/campaign/freeze.json`
- `track3/features/campaign/materialize.json`
- `track3/features/campaign/materialize.py`
- `track3/features/campaign/operations.py`
- `track3/features/campaign/pin.json`
- `track3/features/campaign/priorities.json`
- `track3/features/campaign/queue.json`
- `track3/features/campaign/request_drafts.py`
- `track3/features/campaign/runtime.json`
- `track3/features/campaign/selections.py`
- `track3/features/campaign/stale.json`

## `track3/features/submission`

Unrun scaffolding; packs use the kit's packers directly. This is not the top-level `submissions/` directory, which is untouched.

13 tracked files:

- `track3/features/submission/README.md`
- `track3/features/submission/__init__.py`
- `track3/features/submission/__main__.py`
- `track3/features/submission/_campaign.py`
- `track3/features/submission/_common.py`
- `track3/features/submission/_packed.py`
- `track3/features/submission/check.json`
- `track3/features/submission/operations.py`
- `track3/features/submission/pack-meshes.json`
- `track3/features/submission/pack-poses.json`
- `track3/features/submission/plan.json`
- `track3/features/submission/runtime.json`
- `track3/features/submission/upload.json`

## `track3/features/inspection`

Unrun scaffolding; nothing imports it.

9 tracked files:

- `track3/features/inspection/README.md`
- `track3/features/inspection/__init__.py`
- `track3/features/inspection/__main__.py`
- `track3/features/inspection/annotations.py`
- `track3/features/inspection/encode.json`
- `track3/features/inspection/export.json`
- `track3/features/inspection/operations.py`
- `track3/features/inspection/reels.py`
- `track3/features/inspection/runtime.json`

## `track3/features/public_diagnostics`

Unrun scaffolding; `mesh_eval` and `pose_insurance` hold the live diagnostics.

13 tracked files:

- `track3/features/public_diagnostics/README.md`
- `track3/features/public_diagnostics/__init__.py`
- `track3/features/public_diagnostics/__main__.py`
- `track3/features/public_diagnostics/_mesh.py`
- `track3/features/public_diagnostics/advanced.py`
- `track3/features/public_diagnostics/agreement.json`
- `track3/features/public_diagnostics/diagnostics.py`
- `track3/features/public_diagnostics/floor.json`
- `track3/features/public_diagnostics/frame-convention.json`
- `track3/features/public_diagnostics/mesh.json`
- `track3/features/public_diagnostics/oracle.json`
- `track3/features/public_diagnostics/runtime.json`
- `track3/features/public_diagnostics/score.json`

## `track3/features/runtime_measurement`

Unrun scaffolding; nothing imports it.

8 tracked files:

- `track3/features/runtime_measurement/README.md`
- `track3/features/runtime_measurement/__init__.py`
- `track3/features/runtime_measurement/__main__.py`
- `track3/features/runtime_measurement/aggregate.json`
- `track3/features/runtime_measurement/grouped-accounting.json`
- `track3/features/runtime_measurement/grouped.py`
- `track3/features/runtime_measurement/operations.py`
- `track3/features/runtime_measurement/runtime.json`

## `scripts/research_loop.py`

Entry point of recycled `auto_research`.

1 tracked files:

- `scripts/research_loop.py`

## `scripts/use-scratch-depth.ps1`

PowerShell launcher for recycled `scratch_depth`.

1 tracked files:

- `scripts/use-scratch-depth.ps1`

## `scripts/use-policy-env.ps1`

PowerShell launcher that uses the retired Z: mount.

1 tracked files:

- `scripts/use-policy-env.ps1`

## `scripts/wsl`

Policy-environment installer and activation that hard-code the retired Z: mount; `.runtime/flashchord/venv` replaced them.

5 tracked files:

- `scripts/wsl/README.md`
- `scripts/wsl/create-policy-distro.ps1`
- `scripts/wsl/install-policy-env.sh`
- `scripts/wsl/retarget-requirements.txt`
- `scripts/wsl/use-policy-env.sh`

## Untracked

- `docs/report/track3/figures/d1-v4-objects.pdf`, at `untracked/docs/report/track3/figures/d1-v4-objects.pdf`: orphaned by a generator rename; `make_figures.py` now writes `d1-objects.pdf`. No tracked file uses it.
- `docs/report/track3/figures/d1-v4-qualitative.pdf`, at `untracked/docs/report/track3/figures/d1-v4-qualitative.pdf`: stale generated copy (5ce15a36..., rebuilt 61ab41e8...). `d3_report.sh` regenerates this figure inside its own archive build; an in-place build needs `make_figures.py --qualitative` first.
- `docs/report/track3/main.pdf`, at `untracked/docs/report/track3/main.pdf`: stale in-place build (7d7bece5...), moved so it cannot be mistaken for the abstract to upload (62967a99..., `scripts/reproduce/check_release.sh`).
