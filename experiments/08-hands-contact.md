# 08 — Hands and contact intervals

Status: initializer research reconciled; bounded sequential wrist refinement and
observed-surface contact hypothesis source implemented, unrun; articulated fitting
and empirical deliverables blocked. Owner: `hand_motion/`; consumers `retargeting/`.
Sources accessed2026-10-18; mutable branches/checkpoints are not execution pins.

## Reuse and initializer dispositions

`track3/features/hand_motion/{_wilor.py,estimate.py,_mano.py,kinematics.py,lift.py}`
already own optional pinned WiLoR-mini estimation, licensed MANO integration,
anatomical FK, stable episode shape and robust metric depth lifting. Preserve
these contracts; detector confidence is not calibrated joint visibility.

| Approach | Role/input and gap | License/assets/dependencies/resources | Disposition |
|---|---|---|---|
| HaMeR | RGB hand crop initializes MANO pose/shape; not independent metric wrist depth or contact. Needs same crop, side, camera and parameter conventions as existing estimator. | MIT repository; separate MANO registration/licensed right-hand asset and downloaded learned weights. Upstream environment includes Torch/CUDA and ViTPose; local memory/compatibility unmeasured. | Investigated; defer additional adapter while incumbent unassessed. |
| WiLoR | RGB detects/reconstructs hands; existing adapter accepts assigned anatomy and refuses implicit downloads. Missing temporal articulated refinement and measured joint visibility. | Official model CC-BY-NC-ND; MANO and Ultralytics separate terms. Requires approved installation record, pinned installed files/checkpoints and licensed MANO asset. Upstream Torch2/CUDA11.7 recipe does not establish local compatibility or memory fit. | Implemented optional integration, unassessed; retain initializer design. |

Sources: [HaMeR](https://raw.githubusercontent.com/geopavlakos/hamer/main/README.md),
[WiLoR](https://raw.githubusercontent.com/rolpotamias/WiLoR/main/README.md).
Existing feature README additionally documents the pinned WiLoR-mini adapter;
the official repository recipe is not approval to replace that pin.

Hypothesis: image/depth-supported wrist/articulation refinement and explicit
contact ambiguity improve downstream constraints without smoothing away rapid
motion or disguising tracking loss. Stable shape must fit training/episode image
evidence only; predicted occluded joints stay inferred rather than observed.

## Proposed artifact and metrics

Retain source frame/PTS, grid, anatomical side, MANO convention and initializer
lineage. Separate joint observation/support, inferred articulation, valid camera
motion and metric wrist support. Retain unrefined estimates and each correction.

Contact intervals carry hand/object IDs, start/end source PTS, phase hypothesis
(`approach`, `contact`, `grasp`, `lift`, `release`, `support_transition`), observed
support and ambiguity reason. Distance/proximity alone does not establish grasp;
occlusion or uncertain object surface makes contact unknown. Record supporting
surface provenance so inferred backsides do not become confident contacts.

Diagnostics: supported joint reprojection, hand/object surface distance,
penetration, sliding after object-motion compensation, bone consistency and
timestamp-aware velocity/acceleration. Distinguish unsupported joints from large
residuals. Do not take derivatives across gaps or use nominal FPS when PTS exist.
Report coverage, rapid movement, overlap, full occlusion and release failures.

## Proposed bounded comparison

After05–07, freeze six32-frame event windows per tuning episode39/42 covering
approach/grasp/lift/release and occlusion. Compare existing initializer/lift,
sequential supported refinement and optional alternative initializer separately;
at most384 frames/arm, one fixed checkpoint,30 GPU minutes/arm,12 GiB process
ceiling and2 GiB device headroom. Refinement: at most16-frame windows and20
iterations/window; fixed robust weights from training/tuning protocol, not holdout.
Stop on missing side/identity/scale, unsupported geometry, budget or OOM; preserve
invalid/ambiguous intervals. Do not change prompts/thresholds after outcomes.

Compare matched support, all-support coverage and worst events. Freeze any
candidate on real tuning geometry/contact evidence. Episode41 excluded;
43/44 are consumed and cannot provide a fresh unbiased final assessment.

No new execution outcomes, contact accuracy, local memory fit or physical
performance demonstrated. Exact blockers:05–07 geometry/identity/surface evidence
unavailable; canonical Z: storage read-only; optional WiLoR/MANO installation and
licensed assets not provisioned, new dependencies require confirmation. No
checks/installations or simulator work performed. Next independent design:09;
empirical08 remains open.

## Build disposition — source only

Added `hand_motion/refine.py`, `contacts.py`, request examples and CLI dispatch.
New `refine_authorized` and `contacts_authorized` flags remain false. Existing
operations and defaults retain their gates. New operations additionally restrict
inputs to the ten public training/tuning episodes; 41/43/44 remain excluded.

The sequential baseline refines only metric-depth-supported wrist translation
with a bounded five-frame local linear fit, at most 5 mm correction, explicit
384-frame source interval, no tracking-loss fill and rapid-acceleration bypass.
Original joints and every correction/disposition remain in the output. FK pose,
rotations, stable episode shape and bone lengths are unchanged. Per-joint
`joint_observed` remains false: a depth-supported whole hand and detector score
do not establish observed articulated landmarks. Inferred articulation,
depth-supported wrists and missing joints are separate fields. PTS-aware
velocity/acceleration and bone-length correction diagnostics are emitted.

Observed-surface contact hypotheses consume exact same-camera composed object
poses and moving-fusion voxel support from `object_mesh.fuse`. They require the
same tracked object parent and inherited body origin. No completed backsides
become observed contacts. Approach/contact/grasp/lift/release/support-transition
phase hypotheses combine geometry and object-relative motion; missing support
stays unknown. Inferred joints, partial coverage, absent opposing-force evidence
and unobserved supporting surfaces stay explicit ambiguity reasons. All intervals
remain unaccepted and physically unconfirmed. Distances and sliding are unsigned
observed-voxel diagnostics, not penetration/contact labels. Penetration remains
unavailable rather than fabricating a signed collision surface.

Source bounded resources: at most 384 hand frames, 15-frame temporal windows
(default 5), one linear solve per supported hand/frame, one million observed
support voxels per object. The existing proposed 384-frame tuning comparison and
30-minute/arm stop remain the future assessment protocol, not a run performed.
Both producer manifests and referenced arrays use existing exact hashes,
completed video-only parent admission and canonical Z: persistence. No new
production dependencies, licensed model imports, installations, execution,
tests, checks or outcome-based selection occurred.

Disposition: temporal wrist refinement and contact hypotheses implemented,
unassessed; image/depth-supported articulated MANO/shape fitting deferred because
the current producer supplies inferred FK rather than measured landmarks, and
licensed initializer/MANO assets remain unavailable. HaMeR remains deferred and
WiLoR remains the existing optional initializer; neither is duplicated. Runtime
assessment additionally needs admitted05–07 outputs and writable canonical Z:.
