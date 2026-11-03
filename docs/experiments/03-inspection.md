# 03 — Synchronized inspection

Status: eligible source implementation delivered; full visualization deliverable
partially deferred. Owner: `track3/features/inspection/`.

Hypothesis: synchronized producer artifacts with shared scale and explicit missing
support expose geometric and tracking failures that attractive isolated previews
hide. Baseline reused: `hand_motion/preview.py` already exports selected calibrated
RGB and MANO21 projections. Pinned upstream Viser viewers and producer depth/mask/
hand contracts were read. No campaign was rerun or holdout outcome reopened.

Changed variable: a cross-feature index and optional viewer, rather than another
estimator. Primary Viser chosen to reuse upstream integration. Rerun investigated
through [official documentation](https://rerun.io/docs/overview/what-is-rerun) and
deferred because a second SDK/recording stack has no demonstrated project benefit.
Viser 1.0.30 tagged GUI/scene sources were consulted; no dependency installed.
The old robotic-grounding `viser<0.3` pin conflicts with the selected adapter.

Implemented source: bounded request/handler, all-false runtime gates, canonical Z:
output guard, completed video-only producer lineage and SHA-256 admission, fixed
public train/tuning episode allowlist, source-PTS selected-frame synchronization,
A/B shared calibrated grid and fixed depth limits, saved RGB/hand overlays,
optional depth/invalid pixels and mask-instance layers, optional world-valid hand
landmarks, common-world camera axes/representable frusta, assembled visual/collision
meshes and geometric contact proxy points, per-PTS scene support/contact/completion
state fields, identical A/B camera parent and assembly shift, clickable producer-frame events,
coverage scorecard and manual event-window/reel manifest export. Missing producer
frames remain events, not silently discarded. Export timing/failure classification
and canonical Windows/WSL cache/temporary/bytecode confinement are implemented.
An optional `stereo_depth.generate` adapter adds a deterministic bounded 20k-point
observed metric cloud on the exact RGB preprocessing grid, requiring exact
camera/stereo/depth lineage and internal indices/PTS. No raw MoGe cloud is
metricized. Scene strips show object state/filled, hand source and contact presence
across selected PTS, with fixed colours and a current-frame marker. Selected
sequence order is explicitly not constant-rate playback.
See the feature README for
exact schemas and unsupported channels. No output was rendered or generated.

Budget/protocol: source work only in this pass; future export bounded to 256
selected frames and two candidates, using existing artifacts without models,
GPU work or metric fitting. Episode grouping unchanged: 0/11/12/18/21/23/26/31
train; 39/42 tuning. 41 excluded; 43/44 consumed and unavailable for fresh
assessment; evaluation refused. Limits are selected in the request before review.
Coverage means artifact availability only. Quality/latency/memory remain null.
No image metric, temporal consistency, physical accuracy or policy result claimed.

Results/failures: no playback, imports, checks, tests, export, installation or
dependency-resolution experiments were run. Runtime artifacts: none. Source
identities are the new slice, `hand_motion/preview.py`, `hand_motion/lift.py`,
`depth/generate.py`, `object_masks/generate.py`, and existing clock/provenance
primitives. Future exports hash their exact sources and every consumed artifact.

Additional source build (subagent, 2026-10-18): reused the identity owner's actual
`object_masks.correspondences` contract for optional visible/admitted point tracks,
explicit identity visibility and discontinuity records. Its exact selected masks
parent, source clock, grid and artifact hashes are required. Added a bounded
producer-owned annotation contract for reprojection arrows, visibility/support,
uncalibrated disagreement and observed/inferred/mixed/unknown surface metadata.
An annotation must already be a hashed artifact of one admitted video-only
producer; no arbitrary file, reference label, calibrated confidence or per-face
observation is invented. No current producer is claimed to emit the optional
annotation contract. Its adapter is source implementation, not measured evidence.

The source adapter also accepts the depth owner's `depth.preview` RGB-only
producer, requiring its exact selected depth parent and shared calibrated grid.
This removes the licensed-hand prerequisite from early geometric image review.
Existing `hand_motion.preview` admission remains strict about both saved RGB and
hand overlay and its selected hands/depth parents. RGB-only previews never
fabricate hand overlays; reels use RGB explicitly when an overlay is absent and
record the image channel used in each output-frame mapping.

The independent stereo geometry producer was reconciled with this viewer: its
residual endpoints live in rectified C/B pixels, whereas inspection RGB is in
the calibrated A grid. Those arrows are deliberately not projected onto A.
The optional A-grid annotation adapter does not claim those incompatible
rectified residuals have been rendered or compared.

Added a separately disabled offline OpenCV encoder for manual event windows,
with A/B saved overlays and retained missing-frame panels. Predeclared 1–4096
total frame budget, 128–1024 maximum side and 1–60 FPS bound the entire plan.
Source-PTS sample-and-hold and an exact output-to-source mapping expose CFR
resampling; missing following timestamps and encoder failures are explicit.
Partial/failed artifacts remain failure records, never successful evidence.
Successful writer completion would be labelled encoded-unreviewed, not checked
decodability. Existing OpenCV only; no codec installation. No encoding ran.
The default export still emits only a manifest. Rational source-clock admission
and explicit per-frame rendering failure handling were also strengthened.

Exact blockers: canonical `/mnt/z/hugging-face/video2data` is read-only in this
sandbox; no approved compatible Viser runtime has been established; no executed
inspection artifact bundle exists. Reprojection/annotation artifacts still need
an admitted producer; observed/inferred per-face segmentation lacks labels;
calibrated uncertainty lacks a calibrated producer. Continuous wall-clock timeline plotting,
automatic event classification and runtime assessment remain pending.
Manual image-review events retain exact source-PTS
windows with reviewer/notes. Preview availability itself is not assumed: explicit
completed compatible producer records must be supplied. Do not mark this item
fully assessed or claim the complete world/event interface is implemented.

Disposition: retain Viser as one primary viewer; useful adapters, producer-owned
annotation admission and bounded offline encoding are implemented but unrun.
Runtime assessment and missing producer evidence remain blocked explicitly.
Later independent design work can proceed; any evidence-led
adoption requiring inspected outputs remains blocked. Next numbered item: 04.
