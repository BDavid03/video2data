# Project design and operating reference

This index describes the eleven retained slices after the November 2026
restructure. The [submission ledger](../chain/SUBMISSION.md) tracks the shipped
mesh/pose packages and their evidence; the [version 1 ledger](version1.md) keeps
the broader historical design and its uncompleted goals.

## Current working agreement

[AGENTS.md](../../AGENTS.md) and the current task brief control authorization.
Keep runtime gates, producer hashes and episode exclusions intact. New production
dependencies need approval; upstream references and external environments remain
read-only. Generated artifacts and caches stay inside the permitted Linux checkout.
A command or operation listed here grants no permission to execute it.

Older documents under `docs/design/`, `docs/experiments/`, `docs/research/` and `reports/` may name recycled slices; those paths are now under `recycle/20261104T010336Z/`.

## Reading paths

For a single-file collection, see the
[documentation compiler](../../scripts/documentation/README.md). Its standard-library
script collects project Markdown, including feature pages and historical reports,
into `docs/compiled/all-documentation.md`, which is untracked.

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

| Slice | Responsibility | Entry points |
|---|---|---|
| [Depth](../../track3/features/depth/README.md) | Calibrated monocular depth and cached teacher frames | generate; offline challenger ingestion |
| [Stereo depth](../../track3/features/stereo_depth/README.md) | Calibrated stereo geometry and observed scale | rectify, generate, anchor, correspondences, report |
| [Object masks](../../track3/features/object_masks/README.md) | Opening clusters and full-video masks | opening-clusters, generate, correspondences |
| [Camera motion](../../track3/features/camera_motion/README.md) | Metric camera trajectory and table world | estimate |
| [Mesh extents](../../track3/features/mesh_extents/README.md) | Video measurements, procedural meshes and frozen shape diagnostics | measure, build, rules, adopt, proxy |
| [Mesh evaluation](../../track3/features/mesh_eval/README.md) | Exact packed mesh scoring on public scans | pairs; local CD-O script |
| [Mesh generative](../../track3/features/mesh_generative/README.md) | Frozen pretrained-shape campaigns and diagnostics | pilot, g2, g3, insights |
| [Pose insurance](../../track3/features/pose_insurance/README.md) | Shipped pose-policy line and frozen diagnostics | scene build, policy run, diagnostic and packing modules |
| [Policy training](../../track3/features/policy_training/README.md) | Retained budget consumed by D0; general training tools | compose, train, select, comparison |
| [Mesh fusion](../../track3/features/mesh_fusion/README.md) | Unsuccessful fusion route retained as cited evidence | fixed F1/F1b queues |
| [Dynamic feasibility](../../track3/features/dynamic_feasibility/README.md) | D0/D1 dynamic-route evidence cited in the abstract | feasibility and scaling queues |

The other 17 slices and companion scripts are preserved in
[the recycle folder](../../recycle/20261104T010336Z/MANIFEST.md), whose manifest
explains restoration of dependent groups to their original paths.

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

## Historical planning

The [legacy plan](legacy-plan.md) records the original execution milestones. The
[research backlog](../experiments/README.md) records the ordered experiment plans
and their implementation evidence. Current authorizations remain in the root
[AGENTS.md](../../AGENTS.md).
