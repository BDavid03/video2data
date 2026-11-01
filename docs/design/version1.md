# Version 1 delivery ledger

Objective: **full complete version1**. Status: incomplete; source implementation and documentation ongoing.

This ledger records the full project scope, including the tracked reconstruction,
policy and submission path. A resting fallback alone does not complete version 1.
Source files are implementation evidence only. No runtime behavior, quality,
compatibility or end-to-end completion follows from their presence.

On 18 October 2026 the user expanded authorization to bounded research execution,
including MoGe2 target generation, preparation, learning experiments, original
coursework CNN training, compiler experiments, ensembles, and assessment on the
final holdout after the recipe is frozen. The available episode 39 opening
geometry experiment is also authorized. Episode 41 remains excluded. Tests,
smoke checks, validation commands and dependency-resolution experiments remain
prohibited. New dependency installations still require a concrete proposal and
confirmation; broad execution authorization does not supply missing assets or
licensed models. Submission uploads require explicit authorization.

The first bounded learning campaign completed 12 trials on 40 prepared pairs.
Its transformer winner recorded a tuning log-depth MAE of approximately 0.15815
against MoGe2 pseudo-depth, not measured ground truth. The expanded campaign
completed in 1470.21 seconds: 35 successful experiment outcomes and two failed
primary-environment compiler attempts. Its tuning-frozen eager/BF16 original-CNN
ensemble scored 0.1239556042 tuning MAE and 0.1521616317 holdout MAE. All 13
candidates completed tuning and holdout assessment; these outcomes are not 35
separately trained models. The
[research results](../../track3/features/learning_experiments/README.md#expanded-investigation--completed)
retain the frozen selection and artifact links.

A separate procedural generator completed 128 synthetic training-only samples
in 3.330 seconds, with analytic optical-axis depth and procedural instance IDs.
They were not mixed into the measured research campaign and establish no
training benefit or real-scene fidelity. The first episode-39 opening attempt
failed during rectification with an OpenCV `gemm` error. Explicit contiguous
matrix shapes resolved it; the rerun completed rectify/stereo/anchor/camera,
then clustering failed its quiet-opening/border guard, blocking hull and TSDF.
Separately, a read-only external PyTorch `2.12.1+cu132` environment compiled and
executed the CUDA loss with one-real-batch agreement inside recorded tolerance.
The two primary compiler failures remain preserved; external single-batch timing
establishes no general speedup or primary-runtime compatibility. None of these bounded experiments completes
the full reconstruction and policy deliverables below. Execution flags remain
authorized; tests and new dependency installations remain excluded.

## Documentation coverage

The [design index](README.md) links the architecture, artifact contracts, operating
reference, selection tiers, decisions, error budget, risks, roadmap and feature
authoring guide. This ledger includes the bounded execution update of 18 October
2026; individual design pages retain their own evidence dates. Exact
feature contracts remain beside their implementations.

## Deliverables and current evidence

| Requirement | Current source evidence | Remaining completion evidence or work |
|---|---|---|
| Immutable dataset, calibrated camera grids, distinct clocks and explicit provenance | `track3/data/`, `track3/calibration/`, `track3/provenance.py`; historical foundation reports | Integrate with complete reconstruction; historical checks do not establish current pipeline correctness |
| Metric monocular/stereo depth and observed scale anchors | `features/depth/`, `features/stereo_depth/` | Authorized execution, complete artifacts and accuracy/resource evidence |
| Coursework model from random initialization | `features/scratch_depth/`; five original-CNN variants completed training and calibrated tuning/holdout prediction on isolated train-only data; checkpoints bind source, owning run, data contract and counters; notebooks remain unexecuted | Measured real-scene accuracy and broader generalization; explicit production integration before promotion |
| PyTorch/CUDA optimization and experimental learning | `features/learning_experiments/`; baseline plus completed expanded comparison; frozen ensemble tuning 0.12396 / holdout 0.15216 against pseudo-targets; primary compiler failures preserved; separate external CUDA real-batch agreement within recorded tolerance | Broader numerical/performance coverage and primary-runtime compatibility; production integration and evidence before promotion |
| Procedural synthetic training data | `features/synthetic_depth/`; 128 generated training-only samples with analytic depth and procedural instance IDs, separately owned manifest and preview | Real-scene fidelity and measured training benefit; no mixing into the completed campaign or quality gain is claimed |
| Full-video object and hand masks with persistent identity | `features/object_masks/generate.py`, `_identity.py`, `_models.py`, plus `opening_clusters.py` | Approve/provision pinned model dependencies/assets; establish identity/occlusion and window-boundary evidence |
| Metric camera trajectory and support world | `features/camera_motion/estimate.py` | Authorized full-clip evidence, camera gaps and world/scale quality |
| Scanned object resolution and per-episode reconstructed meshes | `features/object_mesh/`; bounded opening-point accumulation, closed hulls and optional observed TSDF; opening rerun completed rectify/stereo/anchor/camera, failed clustering and blocked hull/TSDF | Persistent valid object support, approved spoon body-frame convention, complete episode meshes and measured shape/coverage; TSDF does not complete unseen surfaces |
| Opening poses, full-clip tracking and world composition | `features/object_pose/initialise.py`, `track.py`, `compose.py` | Provisioned mask backend; authorized trajectory evidence and handling of symmetry/recovery failures |
| Anatomical hand estimation, metric kinematics and world lifting | `features/hand_motion/estimate.py`, `_wilor.py`, `kinematics.py`, `_mano.py`, `lift.py`; disabled selected-frame projection preview | Approve/provision mask and isolated WiLoR/MANO backends and licensed assets; rendered review, full-clip identity, depth and orientation evidence |
| Complete resting and tracked scenes, contacts and supports | `features/scene_assembly/`; sourced parked template, parent-frame hand completion preserving interpolated bone lengths, explicit scene template lineage | Approved MANO assets/runtime, authorized template generation and measured full-scene inputs; orientation, gap, seating and contact evidence; sampled link contacts remain unmeasured |
| Robot references and simulator assets | `features/retargeting/` | Complete local runtime/assets, authorized retargeting/support operations and inspection evidence |
| Per-episode policy composition, training and checkpoint selection | `features/policy_training/`, including resolved recipe snapshots and shared checkpoint admission | Approved runtime provision, measured 16 GB resource budget, authorized training and checkpoint evidence |
| Simulator rollouts at the exact scoring clock | `features/policy_rollout/` | Authorized rollouts, inspection, checkpoint/reference lineage and measured task performance |
| Diagnostics separated from production inputs | `features/public_diagnostics/`; optional same-object mesh baseline comparison and explicit unavailable-reference coverage | Authorized diagnostic evidence using the pinned metric implementation; no label-derived production inputs; spoon shape cannot be scored against an unpublished scan |
| Campaign inventory, selections, stale lineage and work ordering | `features/campaign/`; role-aware selections, public first-run roster bootstrap, full task graph, template/schema-bound requests and disabled review-bundle materialization | Authorized campaign records and selection/lineage evidence for every scored episode; explicit physics/asset choices and actual producer artifacts |
| Complete pose/mesh packages and release provenance | `features/submission/`, including campaign integration, comparison evidence and packed-file admission required before upload | All roster outputs, passing authorized checks, clean pushed source revision and reproducible package manifests |
| Submission delivery | `features/submission/operations.py` | Account/access and explicit upload authorization, followed by an accepted receipt; implementation alone is not delivery |

All `features/` paths above are relative to `track3/`. Feature READMEs describe
their detailed contracts, authorization gates and current limitations. Runtime
flags must reflect the specifically authorized operation; enabling a flag alone
is not execution evidence.

## Integration invariants

- Each new feature owns its request, handler, integrations, persistence,
  configuration and documentation. Shared code contains stable technical
  primitives rather than horizontal service/repository layers.
- Each producer records explicit input/source hashes, split, episode, frame
  identity, status and provenance. Incomplete, changed or diagnostic-only
  producers cannot silently become production inputs.
- Reconstruction uses the common pinhole grid and metric world. Public labels
  remain diagnostic. Object slot and anatomical hand order are explicit.
- Missing measurements remain invalid until a feature with an explicit filling
  policy handles them. A pose prediction or held seed is not an observation.
- Policy outputs must come from actual simulator rollouts on the required clock.
  A copied perception trajectory is not a policy result.
- All generated artifacts, environments, caches and temporary outputs remain in
  the project storage rooted at `Z:/hugging-face/video2data`; external environments
  remain read-only.
- Authorized GPU work is serialized through the shared lease. No operation or
  configuration change grants its own execution permission.

## Next implementation work

Continue resolving source-level integration gaps across perception, scenes,
retargeting, policy and packaging. Current handoffs include the sourced neutral
parked-hand template, explicit 16-link contact layout, role-aware campaign and
submission selections, common-scene fallback comparisons, and exact resolved
recipe/checkpoint admission, scratch-depth checkpoint ownership and source
identity, observed opening TSDF reconstruction with explicit campaign selection,
and admission of the exact packed deliverable before upload. Evidence from the
bounded learning campaign covers only that campaign; the broader reconstruction,
retargeting, simulation and delivery path remains incomplete.

The template and model backends still require approved assets/runtime. Campaign request drafts remain incomplete
until explicit inputs and per-episode choices are supplied. Dependency/asset
proposals must name concrete packages, pins and user-provided files before
seeking installation approval. Implementation approval does not authorize
installation or acquisition of licensed assets.

Completion remains unproven until every deliverable has evidence at its own scope.
As authorized runs complete, record evidence against this ledger without
replacing the full target with the stages that happen to work first.
