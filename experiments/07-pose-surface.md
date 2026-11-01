# 07 — Object pose and observed surfaces

Status: investigated; existing baselines retained; empirical deliverables blocked.
Owners: `object_pose/`, `object_mesh/`. Sources accessed2026-10-18; mutable upstream
branches/model cards are research context, not approved executable pins.

## Existing implementation and expected benefit

Reuse `track3/features/object_pose/{initialise.py,track.py,compose.py}`: geometric
registration, bounded competing hypotheses/reinitialization and camera→world
composition. Reuse `object_mesh/{_points.py,_fusion.py,reconstruct.py}`: bounded
opening point accumulation, hull and observed TSDF extraction. TSDF retains open
unseen surfaces; appearance is not collision-surface evidence. These implementations
have no newly established runtime outcomes. Full-trajectory object-coordinate
fusion remains an integration gap; opening fusion is not its implementation.

Hypothesis: better rigid alignment and admitted object-coordinate fusion improve
observed surface agreement and contact stability without inventing metric evidence
for unobserved surfaces. Compare the simple baseline before neural reconstruction.

| Approach | Input/role and concrete integration gap | License, assets and resource constraints | Disposition |
|---|---|---|---|
| FoundationPose | RGB/depth, intrinsics, initial mask and admissible mesh; alternatively reference-view neural field. Import into existing `camera_T_object` convention, retaining origin, symmetry hypotheses and validity. Diagnostic scans cannot provide its production mesh. | NVIDIA research/evaluation-only license; separate refiner/scorer weight downloads; PyTorch3D/NVDiffRast and CUDA/native compilation. Local memory/compatibility unmeasured. | Investigated; defer approved pinned adapter until baseline geometry/coverage supports comparison. |
| BundleSDF | RGBD video, K and first-frame segmentation jointly track rigid pose and neural shape. Needs PTS-preserving RGB/depth conversion: upstream custom data expects uint16 millimetres, our depth is float metres/NaN. | NVIDIA research/evaluation-only license; LoFTR/XMem weights; Docker/native CUDA builds. XMem code excluded upstream due to licensing. Exact dependency/license inventory and memory fit absent. | Investigated; defer; simple observed fusion first. |
| Observed hull/TSDF | Admitted masked depth and poses to measured surfaces; extend to moving-object coordinates with per-surface support. | Existing bounded code, no new model/dependency. Grid/face/point budgets already explicit. Unknown local runtime. | Implemented opening baseline, unassessed; retained. Moving fusion remains planned. |
| Gaussian/neural appearance priors | Appearance reconstruction or completion hypothesis, not proof of a metric contact surface. Requires separate observed/inferred provenance and depth/silhouette evidence. | Additional models/renderers require pinned licenses, weights and approval; no local resources established. | Investigated category; defer adoption. |
| InstantMesh completion | Single image→generated views→mesh; would require metric alignment against measured surfaces and uncertainty for hidden geometry. | Apache repository; Python≥3.10/Torch≥2.1/CUDA≥12.1, xformers; reconstruction/Zero123++ assets and their terms must be pinned separately. Automatic downloads prohibited. | Investigated exemplar; defer because generated views cannot independently establish collision accuracy. |
| TRELLIS completion | Image/text-conditioned mesh, Gaussian or radiance-field prior; same observed/inferred gap. | Models/majority code MIT, renderer exceptions. Upstream requires≥16 GB NVIDIA GPU/Linux and native CUDA modules; no local measured fit. | Investigated exemplar; defer due resources/dependencies and missing independent geometry evidence. |

Sources: [FoundationPose README](https://raw.githubusercontent.com/NVlabs/FoundationPose/main/readme.md),
[license](https://raw.githubusercontent.com/NVlabs/FoundationPose/main/LICENSE),
[BundleSDF](https://raw.githubusercontent.com/NVlabs/BundleSDF/master/readme.md),
[license](https://github.com/NVlabs/BundleSDF/blob/master/LICENSE.txt),
[InstantMesh](https://github.com/TencentARC/InstantMesh),
[TRELLIS](https://github.com/microsoft/TRELLIS).

## Proposed bounded comparison and reporting

After05–06, select two objects per tuning episode39/42, one static and one moving
where present, with fixed four64-frame manipulation/occlusion windows per episode.
Reuse identical admitted masks/depth/camera transforms; compare geometric tracker,
then FoundationPose separately, then BundleSDF separately. At most512 source
frames/arm, 30 GPU minutes/object,12 GiB memory ceiling and2 GiB headroom. Keep
baseline voxel/face budgets; any completion comparison is a distinct amended
proposal, not permission to exceed resources. Stop on bad scale, unsupported
identity, budget/OOM or unavailable geometry; retain all losses/partial surfaces.

Report rigidity, silhouette/depth residuals and unused-view reprojection where
available, symmetry-equivalent pose error, competing-hypothesis ambiguity,
observed surface coverage, dimensions and gaps. Object-coordinate fusion uses
`object_T_cam = inverse(world_T_object) @ world_T_cam`; never fuse a moving object
into a static world mesh. Do not fill holes by diagnostic scans. Compare contact
distances/penetration sensitivity separately from appearance. Use matched support
and per-object worst failures; diagnostic labels stay isolated from production.

No new pose/mesh experiment, local resource measurement or promotion evidence.
Exact blockers:05–06 admitted geometry/identity unavailable; canonical storage
read-only; new challengers need dependency approval, component/license pins and
weight provisioning. Episode41 excluded; consumed43/44 cannot select candidates.
No tests/checks run. Next independent research:08; empirical07 remains open.

## Pose implementation continuation — source only

Added `object_pose.refine` with a feature-owned bounded request, handler, disabled
runtime flag and documented evidence receipt. It reuses accepted composition,
identity/correspondence and direct stereo artifacts rather than launching another
tracker. The handler checks measured body-point backprojection, same-image accepted
track support, exact pixel calibration, frame/PTS clocks and parent hashes before
robust reprojection/depth refinement. Timestamp-aware correction priors break on
loss, unsupported evidence, reinitialization, large motion and gaps. Original poses
and validity, rejected windows, solver exhaustion and explicit symmetry alternatives
remain available; unresolved symmetry receives no correction. No silhouette cost
is claimed without a renderer. FoundationPose/BundleSDF remain deferred challengers,
with no new imports, model assets or dependencies.

Implementation is unexecuted and unassessed. Actual geometry and manually admitted
receipts remain required; bounds/weights are source defaults, never selected from
consumed 43/44 outcomes. Baseline versus corrected reprojection/depth on matched
support, coverage and worst-event failures are the declared comparison; no result
or production promotion is established. Canonical storage and upstream support
remain exact execution blockers. No tests/checks/imports or experiments ran.
