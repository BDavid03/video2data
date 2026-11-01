# Synchronized artifact inspection

Source-only implementation, 2026-10-18. Export, serving and compatible-environment
admission and offline reel encoding are all disabled in `runtime.json`. No module, viewer, decoder, export,
check or test was executed. Runtime outputs may only be written from the canonical
`Z:/hugging-face/video2data` checkout (native Windows or `/mnt/z`); this sandbox cannot write there.

`operations.py` owns request admission, producer integration, persistence and the
optional Viser UI. Feature-owned `annotations.py` admits image evidence;
`reels.py` owns the separately gated offline encoder. No dependency was added. Viser is the primary viewer because
the pinned FlashCHORD tree already uses it; its `flash_chord/pyproject.toml` pins
`viser==1.0.30`. The separate robotic-grounding environment's `viser<0.3` pin is
incompatible with this adapter. Serving requires an existing approved compatible
environment and checks exactly 1.0.30 before lazy import. Existing environments
stay read-only. This is not a claim that a suitable environment is installed.

Rerun was investigated as a recording/timeline alternative through its [official
overview](https://rerun.io/docs/overview/what-is-rerun). It would add another SDK,
recording format and viewer integration. Deferred: no demonstrated benefit here
justifies a second dependency stack. Viser's tagged [GUI API source](https://raw.githubusercontent.com/nerfstudio-project/viser/v1.0.30/src/viser/_gui_api.py)
and [scene API source](https://raw.githubusercontent.com/nerfstudio-project/viser/v1.0.30/src/viser/_scene_api.py)
informed this adapter; API inspection is not runtime compatibility evidence.

The export request names 1–256 selected source frames and one or two candidates
labelled A/B. Each candidate explicitly names a completed `hand_motion.preview`
or `depth.preview` run. The latter exports calibrated RGB without requiring a
licensed hand initializer, and must consume the exact selected `depth_run`.
Its channel has RGB only; no hand overlay is fabricated. Existing hand previews
continue to require both saved RGB and overlay and their exact selected hand/depth
parents. An independently supplied hands producer is not claimed as drawn by an
RGB-only preview. Optional producers are `depth.generate`, `object_masks.generate` and
`hand_motion.lift`, `camera_motion.estimate`, `scene_assembly.assemble` and
`stereo_depth.generate` and `object_masks.correspondences` (`tracks_run`). Tracks
must consume the exact selected masks parent. Its producer-owned per-frame JSON
contains query identity, pixel position, visible/geometric-admission flags,
explicit identity visibility and discontinuity IDs. Only geometrically admitted
visible tracks are drawn; rejected/unknown evidence remains in the source JSON.
No producer is searched for or inferred. Parent run identity,
artifact SHA-256, dataset revision, public episode, source PTS and calibrated
preprocessing grid must agree. Preview RGB/overlay must have consumed the exact
selected hand/depth parents; masks must have consumed the exact selected depth.
Only training episodes 0/11/12/18/21/23/26/31 and tuning 39/42 are admitted.
Episode 41, consumed holdout 43/44 and evaluation are refused before opening
producer paths. Only `video_only` completed producers are admissible. Legacy
untagged records need producer-owned migration, not guessed identity.

Implemented viewer channels:

- Saved calibrated RGB and projected hand overlays from `hand_motion.preview`.
- Optional depth with predeclared identical metre limits for A/B; invalid pixels
  are magenta, out-of-range values clip, and no per-frame rescaling occurs.
- Optional individual mask layers labelled with the producer's instance IDs.
  Those IDs remain producer assertions, not a new identity estimate.
- Optional finite, world-valid MANO21 hand points in metres. A/B world rendering
  requires the identical camera-motion parent. Both occupy the same scene and
  camera without spatial offset, interpolation or scale fitting. Point clouds
  are landmarks, not reconstructed hand surfaces.
- One selected-frame slider identifies original frame and integer source PTS.
  Producer-frame buttons jump to that frame and name its source run. Full producer
  details remain in the export. This is selected-frame review, not real-time
  PTS-paced video playback; no RGB decoding or estimator is launched.
- Optional common-world camera axes and a symmetric frustum where calibrated
  principal point/focal lengths can be represented (within one pixel); otherwise
  the frustum is explicitly omitted. Camera consumes the exact selected depth.
- Optional assembled scene visual and collision meshes, displayed separately with
  different colour/opacity, bounded to one million vertices per loaded mesh.
  Existing `trimesh` is lazy and never installed. Scene consumes the exact selected
  camera. Assembly's z shift is applied to camera/raw hands for common coordinates;
  A/B scenes must have identical shift and camera parent. Geometry is shown as
  producer geometry; no observed/completed surface segmentation is fabricated.
- Optional observed stereo-supported metric point clouds from `rgb_depth_m` and
  `rgb_valid`, backprojected with `rgb_intrinsics_pixels` on the exact preprocessing
  grid and transformed by the exact selected camera parent. At most 20,000 valid
  pixels are selected deterministically in raster order; no interpolation or
  surface completion occurs. Camera and stereo internal indices/PTS must agree;
  invalid camera poses omit the cloud. RGB colour comes from the saved matching
  preview where available. Provisional calibration and stereo correspondence can
  still be wrong; observed support is not ground truth. Raw MoGe depth is never
  used to produce this metric cloud.
- Scene per-frame support/contact/completion fields track the selected PTS:
  object resting/moving state, filled poses, hand source and contact body IDs.
  Contact points are geometric proximity proxies, not measured physical contacts.
  Explicit manual image-review annotations can identify grasp/lift/release/full
  occlusion/reacquisition windows; each records reviewer and notes. At most 128
  annotations are admitted, with fully selected half-open windows of at most 256
  frames. No automatic event classification is performed.
- Scene timeline image strips show all selected source PTS in request order with
  a white current-frame marker. Rows contain each object's state, each object's
  filled flag, left/right hand source and left/right contact presence. Fixed
  state colours are blue=0, orange=1, purple=2; hand-source states distinguish
  observed/gap-completed/parked. Columns show selected PTS order, so frame gaps
  must not be interpreted as a constant-rate time axis or continuous playback.

Export format is version 1: `run.json` hashes the source files, request, runtime,
clock inventory, producer records and consumed artifacts, and hashes three JSON
outputs. `inspection.json` indexes the source clock, candidate producers, selected
frames, channel artifact references and producer observations. `scorecard.json`
reports selected-frame channel availability and exact identities. Quality,
inspection latency and peak memory stay null because they have not been measured.
Availability counts are not accuracy or successful-rendering claims.
`reel-manifest.json` retains producer observations, missing/failed-frame rows,
and explicit manually annotated windows with every original frame and PTS.
It is a manifest, not an encoded reel. Manual events are input annotations with
`provenance: manual_image_review`, never inferred from diagnostic labels.
`run.json` records export start/finish and wall time, including interrupted/failed
runs; these are not playback latency measurements. Before optional imports,
serving confines temporary directories, XDG/matplotlib cache and Python bytecode
cache under the canonical root without modifying existing environments.

Optional `annotations` names a JSON artifact already hashed in one selected
producer's `artifacts` mapping, bounded to 8 MB and 256 selected frames. It never
admits an arbitrary review file or reserved labels. Schema version 1 has exactly
`schema_version`, `grid`, `producer` (owner's relative run.json path; the consuming
export records its hash), `image_size: [width,height]`, and `frames`. Every frame
has `frame_index`, integer `source_pts`, `tracks`, `reprojection`, `states`, and
`surfaces`, including empty lists for missing channels. Each channel is bounded
to 2048 rows. This optional contract is implemented; no existing producer is
claimed to emit it yet.

- Tracks contain `id`, `xy` (pixel pair or null), and `visibility`
  (`observed`, `occluded`, `unknown`). Only observed points are drawn.
- Reprojection rows contain `id`, `observed_xy`, `predicted_xy`, and `support`
  (`admitted`, `rejected`, `unknown`). Arrows run from prediction to observation;
  rejected/unknown arrows stay visibly distinct. All endpoints must fall on the
  shared calibrated RGB grid. This adapter does not compute a geometric metric.
- State rows contain `identity`, `visibility`, `support`, `discontinuity`
  (`none`, `lost`, `reacquired`, `identity_switch`, `unknown`), and `uncertainty`.
  The latter is null or `{kind,value,units}` with kind `member_disagreement` or
  `uncalibrated_score`. Neither is a probability or calibrated confidence.
- Surface rows contain `identity`, `geometry` (`visual` or `collision`),
  `status` (`observed`, `inferred`, `mixed`, `unknown`), and an explicit `evidence`
  explanation. Supplied metadata is displayed alongside geometry; it does not
  relabel an entire mesh or fabricate per-face observation masks.

Frame labels now also show seconds derived from the single positive rational
source time base. Ambiguous clocks are refused. Rendering exceptions clear the
partially rendered frame and show the exact failure; another selection may be
reviewed without presenting stale layers as successful current-frame output.

`encode <request.json>` consumes an admitted immutable inspection export and
manual image-review windows, using existing lazy OpenCV and NumPy imports. Its
separate `encode_authorized` gate defaults false. `encode.json` declares 1–60 FPS,
128–1024 maximum side and 1–4096 total encoded frames before image loading.
Reels retain A/B saved overlays when present, otherwise saved calibrated RGB,
on a common square canvas, source frame/PTS
text and missing-image panels. The entire event plan must fit the declared
budget; no truncation or favourable-event selection occurs. Each selected window
needs a following source PTS to bound its final frame duration. CFR
sample-and-hold preserves a per-output-frame source mapping; it does not imply
constant-rate source capture or interpolate missing measurements. A failed
existing `mp4v` encoder is reported rather than installing a codec. Output and
manifest hashes, source image identities, timing and failed/interrupted states
are retained. Encoding status is `encoded_unreviewed`: writer completion has not
established successful decoding, image quality or runtime performance. No reel
has been encoded in this pass. The default export remains a manifest-only operation.

Remaining gaps: calibrated uncertainty lacks a calibrated producer; per-face
surface segmentation lacks observed support. Continuous wall-clock timeline
plotting and automatic semantic events remain pending.
Selected-frame producer details and scene state fields expose
available visibility/support/contact evidence without inventing certainty.
Native depth validity is displayed; no calibrated uncertainty is inferred from it.
No independent geometric metric or achieved robot behaviour is measured here.

When separately admitted, CLI shapes are `python -m track3.features.inspection
export <request.json>` and `python -m track3.features.inspection serve
<inspection-run.json>`. They are documentation only, not commands executed in this
pass. Serving binds loopback and ends on Enter. Runtime exports are immutable
indexes over existing artifacts; downstream processes must retain those artifacts.
