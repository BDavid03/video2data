# Project design and operating reference

Documentation baseline: **5 October 2026**. The target is the full Track 3
version 1: reconstruction, robot policy, achieved simulator trajectories,
reconstructed meshes, packaging and accepted submission evidence.

Implementation spans all major feature slices, but the complete pipeline has
not run. Source presence is not evidence of accuracy, compatibility, resource
fit or delivery. The [delivery ledger](version1.md) records what is implemented
and what still needs evidence.

## Current working agreement

[AGENTS.md](../../AGENTS.md) controls current work. Implementation and documentation
may continue. Tests, smoke checks, validation commands, dependency-resolution
experiments, model execution, simulation and coursework notebook execution are
not authorized. Scratch-depth preparation, training and inference remain disabled;
episode 41 remains excluded from the current depth work. Operation flags remain
false. Reading a command or procedure here does not authorize running it.

New production dependencies require confirmation. The specifically approved
MoGe2 installation does not approve any other backend or execution. Generated
artifacts, environments, caches and temporary outputs belong under
`Z:/hugging-face/video2data`; external environments and upstream references are
read-only.

## Reading paths

For a single-file collection, see the
[documentation compiler](../../scripts/documentation/README.md). Its standard-library
script collects project Markdown, including feature pages and historical reports,
into `docs/compiled/all-documentation.md`. It has not been executed.

| Need | Read |
|---|---|
| Understand the complete system | [System overview](system-overview.md), then [artifact contracts](artifact-contracts.md) |
| Find exact operations and future operating order | [Operating reference](operations.md) and the owning feature README |
| Understand remaining delivery work | [Version 1 ledger](version1.md), [roadmap](roadmap.md), [risks and open questions](risks-and-open-questions.md) |
| Understand fallback and tracked selection | [Entry tiers](entry-tiers.md) |
| Understand a design tradeoff | [Decisions](decisions.md), [error budget](error-budget.md) |
| Understand experimental models, tuning and optimization | [Learning buildout](learning-buildout.md) |
| Study research papers, architectures, mathematics, Candle and Burn | [Research volumes](../research/README.md), [complete compendium](../research/egocentric-video-to-policy.md) |
| Add or change a feature | [Feature authoring guide](feature-template.md) |
| Read dated implementation history | [Implementation status](../../reports/implementation-status.md) |
| Read the pinned competition contract | [Contract report](../../reports/contract.md) and [source reading](../../reports/contract-evidence/2026-09-30/submission-kit.md) |

## Feature directory

Each linked slice owns its request, handler, integrations, persistence,
configuration and detailed documentation. The operation list below describes
source entry points, not execution readiness.

| Slice | Responsibility | Operations |
|---|---|---|
| [Depth](../../track3/features/depth/README.md) | Calibrated monocular depth | generate, through the module's direct entry point |
| [Stereo depth](../../track3/features/stereo_depth/README.md) | Calibrated stereo depth and observed scale | rectify, generate, anchor |
| [Scratch depth](../../track3/features/scratch_depth/README.md) | Original CNN and owned training artifacts | prepare, train, predict |
| [Learning experiments](../../track3/features/learning_experiments/README.md) | Owned research training, prediction, ensembles, scoring and tuning | plan, train, predict, boost-train, boost-predict, ensemble, score, tune |
| [Auto research](../../track3/features/auto_research/README.md) | Closed loop: run, record, visualise, consult Codex, improve; frozen finalize, ONNX export and LaTeX report | loop, finalize, report |
| [Depth student](../../track3/features/depth_student/README.md) | Fully convolutional depth students (UNet/dilated/coursework), SILog/gradient losses, EMA, augmentation, ONNX export | train, predict, export |
| [Object masks](../../track3/features/object_masks/README.md) | Opening clusters and persistent full-video masks | opening-clusters, generate |
| [Camera motion](../../track3/features/camera_motion/README.md) | Metric camera trajectory and table world | estimate |
| [Object mesh](../../track3/features/object_mesh/README.md) | Tracking/collision assets and episode reconstructions | resolve, reconstruct |
| [Object pose](../../track3/features/object_pose/README.md) | Opening registration, tracking and world composition | initialise, track, compose |
| [Hand motion](../../track3/features/hand_motion/README.md) | Anatomical estimates, metric lifting and review | estimate, kinematics, lift, parked-template, preview |
| [Scene assembly](../../track3/features/scene_assembly/README.md) | Complete references, supports and contact proxies | assemble |
| [Retargeting](../../track3/features/retargeting/README.md) | Robot references and simulator assets | export, assets, retarget, support, inspect |
| [Policy training](../../track3/features/policy_training/README.md) | Resolved recipes, checkpoints and selection | compose, train, select |
| [Policy rollout](../../track3/features/policy_rollout/README.md) | Achieved simulator trajectories and comparison | rollout, inspect, self-score |
| [Public diagnostics](../../track3/features/public_diagnostics/README.md) | Isolated public-reference measurements | floor, score, frame-convention, oracle, agreement, mesh |
| [Campaign](../../track3/features/campaign/README.md) | Explicit selections, ancestry, work plans and reviewable requests | board, pin, stale, queue, materialize |
| [Submission](../../track3/features/submission/README.md) | Immutable plans, packages and upload records | plan, check, pack-poses, pack-meshes, upload |

`object_mesh.register` and generative mesh reconstruction remain proposals.
A reserved configuration field is not an implemented operation.

## Evidence vocabulary

**Implemented** means a source path exists. **Disabled** describes an execution
flag. **Completed run** means an operation produced its completion record; it
can still contain explicitly invalid measurements. **Inspected** refers to a
particular inspection artifact and its scope. **Selected** means an explicit
campaign choice, not that the latest file won. **Delivered** requires the
external receipt and acceptance evidence required by the delivery ledger.

Feature pages contain historical **Official**, **Measured**, **Derived**,
**Design** and **Open** labels. Official facts refer to pinned local sources;
measured facts refer to named historical reports. Neither is a fresh external
verification. Derived arithmetic is not a resource measurement. Design thresholds
are not proven tolerances. This documentation update ran no project operations.

The core design pages were rebuilt from readable source and feature documentation
after their files were found to contain no readable text. This set describes the
current design; it does not claim to recover the original decision history.
