# 05 — Joint camera and temporal depth

Status: research/design dispositions recorded; empirical comparison blocked.
Owners: `camera_motion/`, `depth/`; comparison belongs to `learning_experiments/`.
Research accessed 2026-10-18. Upstream `main` and model cards are mutable;
the links below are research references, not executable source/weight pins.

## Reused evidence and hypothesis

Reuse calibrated MoGe2 generation (`track3/features/depth/generate.py`), stereo
anchoring (`stereo_depth/anchor.py`) and static-background camera tracking
(`camera_motion/estimate.py`). The completed learning campaign measures teacher
agreement, not independent geometric superiority of these challengers. No campaign
is relaunched. Item 01 records consumed holdout 43/44; no new choice may use it.

Hypothesis: joint or temporal estimates could reduce background drift and
occlusion-recovery errors without corrupting moving-object depth or metric scale.
Retain the incumbent until a matched independent comparison supports promotion.

## Approach dispositions

| Approach | Role, inputs and integration gap | License, weights, dependencies and resources | Disposition |
|---|---|---|---|
| ViPE | RGB video to intrinsics, camera trajectory and near-metric depth; retain supplied calibration and independent stereo scale. Needs PTS/grid-preserving camera/depth adapters and explicit pose-direction conversion. | Apache code except Unik3D BY-NC-SA; third-party models/software have separate terms. Native CUDA extensions, CUDA-enabled PyTorch and nvcc; current dependencies include GTSAM development build and Ray. `pose_only_long` bounds keyframe memory and uses MoGe2, but local memory/runtime are unmeasured. | Investigated; defer integration pending approved pinned components, storage and geometry evidence. |
| Depth Anything 3 | Multiple RGB views, optionally known camera poses, to depth/confidence/intrinsics and OpenCV world-to-camera extrinsics. Preserve calibrated intrinsics, invert extrinsics explicitly for `world_T_camA`; distinguish relative, metric and nested outputs. | Base/small weights Apache; large/giant/nested CC BY-NC; standalone metric/mono large Apache. PyTorch≥2 and xformers; Gaussian head additionally gsplat. Streaming advertises <12 GB, not a measurement for our workload. Exact checkpoint revision/hash remains unselected. | Investigated; defer. Calibrated DA3-Base is the first proposed any-view challenger, not a frozen selection. |
| Video Depth Anything | RGB video to temporally consistent relative or metric depth; does not supply camera motion. Needs bounded clip inference preserving remap, PTS, validity and stereo scale. | Small model Apache; base/large CC BY-NC. Requirements pin Torch2.1.1/torchvision0.16.1/xformers0.0.23. Upstream A100 FP16 memory at 32×518×518: small6.8 GB, large23.6 GB. These figures do not establish local fit or compatibility. | Investigated; defer. Small metric model is the first proposed temporal challenger. |

Sources: [ViPE](https://github.com/nv-tlabs/vipe),
[installation](https://raw.githubusercontent.com/nv-tlabs/vipe/main/docs/installation.md),
[dependencies](https://raw.githubusercontent.com/nv-tlabs/vipe/main/pyproject.toml),
[DA3 model cards and API](https://github.com/ByteDance-Seed/Depth-Anything-3),
[VDA models/resource measurements](https://github.com/DepthAnything/Video-Depth-Anything),
[VDA dependencies](https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/main/requirements.txt).

## Proposed comparison, not an executable request

After items 02–04 supply admitted geometry, freeze six 32-frame event windows per
tuning episode 39/42: static head motion, manipulation, occlusion and recovery;
record exact indices/PTS before inference. Use identical calibrated source grids,
maximum side518, one challenger at a time, one fixed pretrained checkpoint,
at most384 evaluated frames/model, 30 GPU minutes/model and 12 GiB process memory
ceiling with 2 GiB device headroom. Record preprocessing/resize differences.
No install or automatic checkpoint retrieval is part of the budget.
Stop on budget, OOM, incompatible component or missing calibration; retain failure,
do not silently lower resolution/change model. Any amended proposal precedes its run.

Use item02's independent stereo correspondence/metric residuals, unused-view
reprojection where available, static-background drift and motion-compensated
temporal residuals. Split static/hands/moving objects; report valid coverage,
matched-support results and challenger-specific missing support. Reject raw
frame differences and per-frame scale fitting as primary metrics. Report worst
occlusion/recovery windows and separate teacher agreement. No statistical claim
should treat pixels as independent episodes.

No new outcomes, failures from model execution, measured local resources or
promotion evidence exist in this pass. Exact blockers: item04's independent metric
anchor/admitted artifacts are unavailable; canonical Z: storage is read-only in
this sandbox; challenger dependencies and weights lack approved immutable
provisioning. No tests or dependency-resolution experiments are authorized.
Next independent research item:06. Empirical05 remains open.

## Source build: offline challenger ingestion

Implemented the smallest dependency-free integration path in
`depth/challenger.py`, with the request shape `depth/ingest-challenger.json` and
camera-owned extrinsic normalization in `camera_motion/challenger.py`. All three
named approaches now have a concrete offline export-ingestion contract. The new
`challenger_ingest_authorized` flag is false; existing gates remain unchanged.
No challenger imports, weights, dependencies, downloads or model launches were
added. Immutable source archive/commit, weight and license byte receipts and a
license review are mandatory, with explicit video-only provenance.

The handler checks exact incumbent RGB identity, calibration grid and frame PTS;
retains invalid support; separates relative optical-z from metric optical-z;
inverts declared world-to-camera poses without fitting scale or changing gauge;
and preserves relative pose units separately. VDA poses and unsupported/nested
output conventions fail instead of falling back. Episode admission is hard-limited
to the ten training/tuning episodes, excluding 41 and consumed 43/44. The bounded
adapter permits at most 384 explicit frames at maximum side 518, and output paths
remain canonical Z: only. It writes separate challenger producer kinds rather
than silently replacing the incumbent. Production consumers must explicitly
admit challenger semantics before promotion; that remains an evidence-dependent
step rather than a default switch.

Disposition update: ViPE, DA3 and VDA offline adapters implemented at source level;
model-specific execution adapters deferred pending approved immutable components
and weights. No new measured result, resource measurement, test, import check or
validation command was run. No export is available here, so no ingestion ran.
Existing blockers and the incumbent-retention decision above still apply.
