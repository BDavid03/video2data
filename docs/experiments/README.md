# Beyond-coursework research and experiment queue

Created 2026-10-18. Updated after ordered evidence reconciliation, primary-source
research, source improvements and bounded comparison designs for all 16 items.
No new experiment or verification command was executed from this backlog.
**The backlog remains open:** empirical prerequisites and final production
evidence are blocked. See [the evidence handoff](16-evidence-report.md).

Following “with subagents build each”, feature owners built the eligible source
handlers and their integration boundaries across the queue. See the
[source buildout map](BUILDOUT.md) for entry points, delivered behaviour and
remaining limitations. All new operations are disabled and unexecuted.

## Objective and working order

Improve the full video-to-robot chain: metrically anchored camera/depth geometry,
persistent object identity, reconstructed object and hand trajectories, credible
contacts, and achieved robot behaviour. Teacher imitation loss is a research
metric, not independent evidence of physical accuracy or task success.

Work through every numbered item below. Investigate every named approach, but
adopt only those justified by evidence and integration cost. Alternative models
are challengers to compare, not a requirement to stack every model into one
pipeline. Record an explicit disposition for each alternative.

The root `AGENTS.md` controls authorization. In particular, the no-testing
direction remains active. Proposed verification and simulator comparisons below
must remain unexecuted where current permissions exclude them. This document
does not enable flags, authorize new dependencies or enlarge a run's budget.

| Order | Work item | Status | Prerequisites |
|---|---|---|---|
| 01 | Reconcile existing evidence and evaluation contracts | [Reconciled; metadata inventory handler built](01-evidence.md) | None |
| 02 | Design independent geometric evidence and error attribution | [Report/correspondence source built; unrun](02-metrics.md) | 01 |
| 03 | Build the synchronized inspection and visualization slice | [Inspection/tracks/reel encoder built; unrun](03-inspection.md) | 01–02 |
| 04 | Establish calibrated stereo and metric anchoring | [Heldback support/aggregation/report source built](04-metric-anchor.md) | 01–03 |
| 05 | Compare joint camera/depth and temporal-depth approaches | [Offline challenger ingestion built; no model execution](05-camera-depth.md) | 04 |
| 06 | Improve masks, correspondences and persistent identity | [Identity/visibility/correction handler built; unrun](06-identity.md) | 03–04 |
| 07 | Improve object pose and observed surface reconstruction | [Pose refinement and moving fusion built; unrun](07-pose-surface.md) | 05–06 |
| 08 | Refine hands and contact intervals | [Wrist/contact hypothesis handlers built; unrun](08-hands-contact.md) | 05–07 |
| 09 | Add bounded joint scene refinement | [Bounded solver/ablation handler built; unrun](09-joint-refinement.md) | 05–08 |
| 10 | Extend the original learning models beyond teacher imitation | [Preparation/trainer/tuning inference/objectives built](10-learning.md) | 02, 04–06 |
| 11 | Assess uncertainty, ensembles and bounded model selection | [Geometry comparison/freeze handlers built; unrun](11-selection.md) | 07–10 |
| 12 | Integrate contact-aware retargeting | [Contact admission and matched recipes built; unrun](12-retargeting.md) | 07–09, 11 |
| 13 | Design and, if authorized, assess achieved robot behaviour | [Frozen plan/achieved comparison handlers built; unrun](13-achieved-behaviour.md) | 12 |
| 14 | Assess end-to-end policy alternatives | [Feasibility/action contracts/tiny module built; unfitted](14-policy-alternatives.md) | 01, 13 |
| 15 | Optimize demonstrated runtime bottlenecks | [Phase instrumentation/receipt accounting built; unrun](15-runtime.md) | Stable candidate and measurable workload |
| 16 | Freeze the final candidate and report final evidence | [Conditional freeze/report handler built; no final candidate](16-evidence-report.md) | Dispositions for 01–15 |

Source implementation and a research disposition do not establish experimental
success. Dependencies below still apply to runtime comparisons; independent
research/design continued when those comparisons were blocked. Existing campaigns
were reused, never restarted. All 43/44 holdout outcomes are consumed; no new
selection used them. Episode 41 remains excluded.

Current execution blocker: canonical Z: storage is outside this session's writable
roots, and approval is unavailable. No runtime artifact can be redirected to the
implementation checkout or `/tmp`. New dependencies remain unapproved, simulation
and checks remain prohibited, and missing upstream evidence is recorded per item.
Last completed: eligible source buildouts described in [BUILDOUT.md](BUILDOUT.md).
First unresolved: 03 inspection assessment and missing producer evidence. Next
conditional execution: 03 then 04,
after canonical access and admitted assets, within existing authorization. Full
artifact identities, limitations and continuation are in [item16](16-evidence-report.md).

## 01 — Reconcile existing evidence and evaluation contracts

Owners: `track3/features/learning_experiments/`, `public_diagnostics/`,
`submission/`, and the feature documents that own each producer.

- Read completed records from the existing bounded and expanded campaigns.
  Separate completed, running, failed and unassessed stages; retain their source,
  dataset, configuration and artifact identities. Do not relaunch existing work.
- Establish whether final holdout 43/44 has been accessed, which decisions were
  frozen beforehand, and what remains legitimately untouched.
- Preserve training episodes 0/11/12/18/21/23/26/31 and tuning episodes 39/42
  unless a separately justified protocol changes them. Episode 41 stays excluded.
- Reconcile the pinned Submission Kit and FlashCHORD contracts with project
  producers. Online documentation is research context, not a replacement for
  the pinned official metric implementation.
- Record coordinate frames, object origin conventions, units, pose ordering,
  calibration transformations, timestamps and missing-data semantics.

Deliverable: an evidence inventory and contract map. Done when each headline
claim has a supporting completed record or is explicitly marked unestablished.

## 02 — Design independent geometric evidence and error attribution

Owner: `public_diagnostics/`, with feature-owned non-label diagnostics.

- Separate teacher agreement, independent image geometry, reference-label
  diagnostics and achieved simulator behaviour in every report.
- Specify stereo correspondence error, unused-view reprojection where available,
  scale drift, depth-edge alignment, object rigidity and coverage.
- Specify motion-compensated temporal comparisons with occlusion and dynamic
  masks. Raw frame-to-frame depth differences are not a consistency metric.
- Use the existing diagnostic scale, anchor, body-frame, timing and resting-drift
  substitutions to attribute error when their execution is authorized. These
  fits remain diagnostic reports; they never correct production outputs.
- Report per-episode, per-object and difficult-event results, including missing
  support and worst failures. Define matched support for comparisons.
- Do not independently rescale each frame when assessing metric depth. Report
  any scale-aligned supplementary metric separately.
- Define baseline and candidate comparisons before inspecting their outcomes.
  Bootstrap whole episodes if useful, and expose the weak statistical support
  of only two tuning episodes rather than treating pixels as independent trials.

Deliverable: a metric and error-attribution specification, plus feature-owned
report implementations where appropriate. No verification commands are implied.

## 03 — Build the synchronized inspection and visualization slice

Owner: the existing producer slices; a dedicated visualization vertical slice
only if cross-feature playback requires its own request, handler and artifacts.

- Inspect existing viewer integrations first. Compare extending Viser with using
  Rerun; choose one primary viewer and avoid duplicate visualization stacks.
  Any new production dependency needs confirmation before installation.
- Link source RGB, masks and IDs, hand landmarks, point tracks, fixed-scale
  depth, invalid regions and reprojection residual arrows to one source clock.
- Link a world view with camera frusta, axes, point clouds, object meshes, hand
  geometry and collision surfaces. Distinguish observed and inferred surfaces.
- Add timelines for contact, visibility, support, uncertainty and tracking
  discontinuities. Clicking an event should identify its frame and producer.
- Design candidate A/B views with identical camera, scale and colour limits.
- Produce event-oriented failure reels for grasp, lift, release, full occlusion
  and reacquisition; retain the failures, not just attractive examples.
- Specify exportable scorecards for per-episode quality, coverage, latency and
  memory, with source and artifact identities.

Deliverable: a synchronized inspection interface and documented export format.
Playback or verification remains subject to current execution authorization.

## 04 — Establish calibrated stereo and metric anchoring

Owners: `stereo_depth/`, `depth/`, `camera_motion/`.

- Reconcile calibration, distortion, resizing, stream synchronization, baseline
  and optical-axis depth conventions before proposing a new estimator.
- Treat calibrated stereo as independent metric evidence where correspondence
  is reliable. Specify confidence and left/right consistency admission.
- Investigate robust anchoring of monocular depth to stereo-supported regions,
  separating static background, hands and moving objects.
- Explicitly handle textureless, reflective and occluded regions; stereo is not
  automatically ground truth there.

Deliverable: a metric-anchor design and eligible bounded comparison request.
Adoption requires improved independent evidence with visible coverage limits.

## 05 — Compare joint camera/depth and temporal-depth approaches

Owners: `camera_motion/`, `depth/`; `learning_experiments/` owns comparisons.

- Retain the current calibrated MoGe2/stereo approach as the incumbent.
- Investigate ViPE for joint camera and near-metric depth estimation; preserve
  available calibration and assess metric anchoring rather than assuming scale.
- Investigate Depth Anything 3 for geometry across multiple views, and Video
  Depth Anything for temporal depth consistency. Record the role, input contract,
  license, weights, dependency and resource requirements of each.
- Compare challengers separately before considering combinations. Evaluate
  static-background drift, dynamic-object contamination and occlusion recovery.
- Define matched samples, resolution and bounded resources. Establish memory
  fit from evidence; model family names do not establish GPU requirements.

Deliverable: a disposition for all three challengers and one evidence-backed
camera/depth candidate, or an explicit decision to retain the incumbent.

## 06 — Improve masks, correspondences and persistent identity

Owner: `object_masks/`, with feature-owned tracking integration.

- Compare the available segmentation baseline with SAM 2 and the SAM 3 family;
  investigate CoTracker3 as complementary point correspondence.
- Combine extent, temporal tracks, visibility and geometric rejection rather
  than treating a semantic mask as persistent object identity.
- Address hand/object overlap, similar objects, complete occlusion, re-entry,
  identity switches and abrupt mask contamination.
- Define manual correction inputs and provenance without using reserved labels.

Deliverable: an identity/visibility artifact contract, challenger dispositions
and event-specific tracking evidence when execution is authorized.

## 07 — Improve object pose and observed surface reconstruction

Owners: `object_pose/`, `object_mesh/`.

- Investigate FoundationPose with admissible reconstructed geometry and observed
  RGB/depth; inspect BundleSDF as a joint pose/shape alternative.
- Refine temporal SE(3) trajectories using depth, silhouettes and accepted
  correspondences. Handle rotational symmetry and competing pose hypotheses.
- Fuse surfaces in object coordinates so moving objects are not smeared into
  the static scene. Compare a simple observed-surface fusion baseline before
  committing to neural implicit reconstruction.
- Track observation coverage and uncertainty separately from completed shape.
  Gaussian or generative representations may provide appearance/shape priors;
  their appearance does not establish metric collision-surface accuracy.
- Compare mesh dimensions, observed surface agreement and downstream contact
  sensitivity. Diagnostic reference scans remain outside production estimation.

Deliverable: pose and mesh candidates with explicit failure and coverage reports;
dispositions for FoundationPose, BundleSDF and shape-completion approaches.

## 08 — Refine hands and contact intervals

Owner: `hand_motion/`, with contact fields consumed by `retargeting/`.

- Investigate HaMeR and WiLoR as initializers, accounting for MANO access/license
  requirements and existing dependency limits.
- Refine wrist location, articulated pose and stable hand shape against images,
  scene depth and temporal evidence. Distinguish occluded joints from observed
  joints rather than inventing confident measurements.
- Infer approach, contact, grasp, lift, release and support-transition intervals.
- Specify hand/object distance, excessive penetration, contact sliding, bone
  consistency and timestamp-aware trajectory derivatives as diagnostics.
- Avoid using smoothing to erase genuine rapid motion or conceal tracking loss.

Deliverable: temporally refined hand/contact artifacts and dispositions for both
initializers. Contact estimates must expose ambiguity and failure intervals.

## 09 — Add bounded joint scene refinement

Owner: `scene_assembly/`, retaining feature-owned producer integrations.

- Optimize short overlapping windows over admitted camera/object/hand estimates.
- Combine robust reprojection, depth, silhouette, correspondence, anatomical,
  contact and temporal residuals with explicit visibility and support masks.
- Fix gauge freedoms and metric anchors; prevent global scale or coordinate
  changes from masquerading as improvements.
- Compare independent estimates, simple sequential refinement and joint
  refinement at a declared compute budget. Keep ambiguous hypotheses explicit.
- Retain the unrefined artifacts so each correction and failure is attributable.

Deliverable: a bounded refinement request/handler and an ablation specification.
Adopt only if it improves independent evidence without hiding missing support.

## 10 — Extend original learning beyond teacher imitation

Owners: `learning_experiments/`, `scratch_depth/`.

- Reuse existing ANN, CNN, transformer, boosting and original coursework-CNN
  results as baselines; do not repeat them without a stated evidence gap.
- Investigate a compact residual CNN/U-Net correcting admitted depth, using RGB,
  depth validity and confidence, against direct prediction at matched resources.
- Investigate temporal context, stereo-supported supervision, robust log-depth,
  depth-boundary, normal and motion-compensated temporal objectives separately.
- Sample whole clips and difficult events, not only early frames. Preserve
  episode grouping and fit all learned preprocessing on training episodes only.
- Consider diverse teachers only after identifying complementary error patterns.
  Keep pseudo-label quality and student imitation performance separate.
- Record budgets and stopping rules before fitting; separate original randomly
  initialized models from any pretrained/adapted challenger in reports.

Deliverable: a bounded ablation sequence and evidence for each added source of
supervision/context. A lower teacher loss alone does not promote a model.

## 11 — Assess uncertainty, ensembles and model selection

Owner: `learning_experiments/`, with consumer-visible confidence semantics.

- Reconcile the expanded campaign's frozen ensembles before proposing new ones.
- Measure complementary errors and quality/resource tradeoffs before adding
  members. Define missing-member and invalid-pixel behaviour explicitly.
- Investigate confidence thresholds through error-versus-coverage curves.
  Member disagreement is a diagnostic, not calibrated uncertainty by itself.
- Fit weights, thresholds and preprocessing on training/tuning evidence only.
- Predeclare how conflicting geometric, temporal and teacher metrics determine
  a candidate. Avoid choosing favourable metrics after seeing outcomes.
- Freeze all choices and record holdout eligibility before final assessment.

Deliverable: a frozen candidate or justified incumbent retention, with uncertainty
limits, lineage and a complete record of rejected/deferred candidates.

## 12 — Integrate contact-aware retargeting

Owners: `retargeting/`, `scene_assembly/`, `policy_training/`.

- Keep pinned FlashCHORD as the integration baseline. Investigate CHORD's
  object-centric contact-wrench representation and SPIDER as research comparators.
- Preserve intended object motion and contact effects while respecting robot
  reachability, joint/velocity limits, collision geometry and contact feasibility.
- Propagate measured scene uncertainty into proposed training randomization;
  avoid arbitrary ranges that obscure reconstruction defects.
- Compare motion-only and contact-aware objectives with matched resources.

Deliverable: retargeting artifacts and a matched downstream comparison design.
Implement permitted work; simulation stays deferred unless explicitly authorized.

## 13 — Assess achieved robot behaviour when authorized

Owners: `policy_training/`, `policy_rollout/`, `public_diagnostics/`.

- Design fixed-budget, matched-seed comparisons of achieved object trajectories
  using the pinned official metric code and complete coverage reporting.
- Include resting and existing-policy baselines. Distinguish reconstructed
  demonstration poses from object motion actually achieved in simulation.
- Isolate perception contributions through controlled depth/scale, tracking,
  geometry and contact ablations rather than changing every component at once.
- Record simulator failures, recovery, contact loss and worst episodes alongside
  aggregate scores. Inspect whether label-free candidate rankings agree with
  development diagnostics before relying on them elsewhere.

Deliverable: a concrete comparison protocol and, only with applicable simulation
authorization, recorded achieved-rollout evidence. No uploads are implied.

## 14 — Assess end-to-end policy alternatives

Owners: `policy_training/`, `learning_experiments/`.

- Investigate pretrained VLA, world/action-model and hybrid alternatives against
  the explicit reconstruction route, including data/action representation fit.
- Document embodiment adaptation, available demonstration volume, licenses,
  dependencies and total resource cost before proposing execution.
- Treat these as a separate bounded research track. Do not assume that a larger
  pretrained model is better for the available tasks or evaluation contract.
- Compare achieved outcomes under declared budgets if execution is authorized;
  otherwise record an evidence-based feasibility disposition.

Deliverable: explicit dispositions for end-to-end and hybrid policy approaches,
including the reason for any deferral and the evidence needed to revisit it.

## 15 — Optimize demonstrated runtime bottlenecks

Owners: `scratch_depth/` and each affected producer slice.

- Reconcile existing eager, BF16, channels-last, fused-optimizer, TorchInductor
  and custom-CUDA experiments before adding work.
- Design measurements that separate decode, preprocessing, model, transfer,
  compilation and artifact I/O. Separate first-call and steady-state costs.
- Compare matched numerical workloads and actual end-to-end resource costs;
  record allocated/reserved peaks and sampled device memory distinctly.
- Investigate compilation and custom kernels only for a material bottleneck.
  Preserve explicit backend failures rather than silently reporting a fallback.
- Proposed future correctness evidence includes reference outputs/gradients,
  gradcheck where applicable, and CUDA memory/race/synchronization diagnostics.
  These checks remain prohibited by the current no-testing instruction.

Deliverable: an optimization disposition and, where already authorized, bounded
performance evidence with limitations. A single successful compiler call is
neither general correctness evidence nor a steady-state speedup claim.

## 16 — Freeze and report final evidence

Owners: `learning_experiments/`, `campaign/`, `public_diagnostics/`, `submission/`.

- Give every preceding approach an evidence-backed disposition; unresolved
  blockers remain visible rather than being marked complete.
- Freeze models, weights, thresholds, geometry settings and policy choices using
  tuning evidence before any eligible final holdout assessment.
- If holdout outcomes have already influenced work, say so. Do not claim an
  untouched final assessment or select a new holdout after inspecting outcomes.
- Report independent geometry, coverage, achieved behaviour where available,
  runtime/resource costs, failures, sample limitations and exact artifact lineage.
- Export fixed-scale comparisons, event reels and per-episode scorecards.
- Keep research completion, production promotion and submission/upload separate.

Deliverable: a reproducible evidence report identifying what is demonstrated,
what remains uncertain and whether any production integration is justified.

## Per-item agent record

Keep concise notes beside this queue as work progresses. Use one Markdown record
per item when needed; store runtime artifacts only in the canonical Z: checkout.

```text
Item / status / owner:
Hypothesis and expected task benefit:
Existing implementation and evidence reused:
Baseline / changed variable / alternatives:
Episode partition and holdout eligibility:
Bounded compute, samples, seeds and stopping rule:
Primary metric / independent evidence / coverage:
Authorization and dependency constraints:
Artifact and source identities:
Observed results, failures and limitations:
Disposition and reason:
Next eligible item / exact blockers:
```

## Primary research references

- [Official reconstruction baseline](https://nvidia-isaac.github.io/video_to_data/reconstruction/)
  and [CHORD](https://nvidia-isaac.github.io/video_to_data/chord/).
- [ViPE](https://research.nvidia.com/labs/toronto-ai/vipe/),
  [Depth Anything 3](https://depth-anything-3.github.io/) and
  [Video Depth Anything](https://videodepthanything.github.io/).
- [SAM 2](https://ai.meta.com/research/sam2/),
  [SAM 3](https://github.com/facebookresearch/sam3) and
  [CoTracker3](https://cotracker3.github.io/).
- [FoundationPose](https://nvlabs.github.io/FoundationPose/) and
  [BundleSDF](https://github.com/NVlabs/BundleSDF).
- [HaMeR](https://github.com/geopavlakos/hamer),
  [WiLoR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf)
  and [SPIDER](https://github.com/facebookresearch/spider).
- [Rerun](https://docs.rerun.io/) and
  [Viser](https://viser.studio/main/api/core/server/).
- [PyTorch gradcheck](https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html)
  and [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

These references support research directions, not project-specific superiority,
installation approval, compatibility or measured results. Pin versions, weight
identities and licenses in each concrete proposal before integration.
