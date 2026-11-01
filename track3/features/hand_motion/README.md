# Hand motion

Status: `estimate`, `kinematics`, `lift`, `parked-template` and `preview` implemented, never run. The optional
WiLoR and MANO backends are not provisioned. All execution flags are false.
No tests, validation commands, model
execution, dependency installation or notebook execution accompanied this work.

This slice estimates the pose of both hands in every frame and places them in the
metric world. The robot's reference motion is retargeted from these hands, so they
decide how the robot approaches, grasps and moves each object. `RG/` means
`upstream/video_to_data-flashchord-aee95f2/robotic_grounding/`; the source reading
behind the upstream facts is
[upstream-retargeting.md](../../../reports/contract-evidence/2026-09-30/upstream-retargeting.md).

## Role in scoring

No leaderboard metric measures hands. They matter through the policy: a hand
reference that reaches the wrong place, or passes through the object, trains a
policy that knocks the object away or never lifts it, and the object trajectory is
what is scored.

## What the upstream retargeter requires

All **Official**, from the pinned upstream source.

| Requirement | Source |
|---|---|
| Per hand and per frame: 21 joint positions in the world, metres, in MANO order (wrist, then four joints for each of thumb, index, middle, ring and pinky) | `RG/source/robotic_grounding/robotic_grounding/retarget/params.py:7-29` |
| Per hand and per frame: 21 joint orientations, scalar-first quaternions, in MANO's joint-frame convention. The wrist and the five fingertips are active orientation targets. | `retarget/params.py:191-219`, `retarget/hand_kinematics.py:236-251` |
| Both hands on every frame. There is no validity field and no single-hand mode. | `RG/scripts/retarget/ego_recon_to_sharpa.py:307-312,521-535` |
| A Z-up world with gravity along negative z | `ego_recon_to_sharpa.py:227-243`, `retarget/support_recon.py:39,480-491` |

The MANO parameters themselves are carried through and never read by the
retargeter. The joint orientations are the hard part: upstream gets them by
running MANO forward kinematics in a separate loader, because the MANO model files
are licence-gated and the library that runs them is GPL-3.0
(`RG/workflow/data_pipeline.md:43-59`).

## Operations

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `estimate` (implemented, disabled) | Per-frame hand detections and MANO parameters in the camera frame | Planned isolated WiLoR environment at `.runtime/wilor/` | Yes |
| `kinematics` (implemented, disabled) | 21 joint positions and orientations per hand, plus metric surface samples | Planned isolated MANO environment at `.runtime/mano/` | No |
| `parked-template` (implemented, disabled) | Neutral, model-derived parked-hand joints, rotations and surface bounds | Planned isolated MANO environment at `.runtime/mano/` | No |
| `lift` (implemented, disabled) | Depth-fitted joints placed in the metric world | Existing perception environment; NumPy only | No |
| `preview` (implemented, disabled) | Selected-frame RGB/joint overlays and a local review page | Existing perception environment; NumPy and OpenCV | No |

### Selected-frame projection review

`preview.py` owns `PreviewHandsRequest`, rendering and persistence. The independent
`preview_authorized` flag remains false; split and episode allowlists still apply.
No preview has been generated. This operation reads completed `hand_motion.lift`
and `depth.generate` producers for the same episode and requires the lift to have
consumed that exact depth record. It verifies artifact hashes, the pixel grid,
RGB identity and the full inventoried MP4 frame/PTS sequence before drawing.
It does not import a model or invoke another feature's operation.

The request lists 1–64 unique ascending `frame_indices`. For each selected frame,
the handler remaps RGB to the depth pinhole grid and saves the plain image beside
an overlay of accepted camera-space joints. Left is cyan and right is orange;
wrist and fingertip labels follow MANO21. Missing or rejected hand estimates are
not drawn. Camera-valid estimates with an invalid world pose remain visible, with
their separate `world_valid` status recorded. The HTML page shows the original
fit/rejection outcomes, and can be opened locally without a server or external
assets. No frames are interpolated or assigned a new playback clock.

Outputs share the feature's unique operation directory:

- `frame_<index>_rgb.png` and `frame_<index>_overlay.png` preserve the selected images.
- `preview.json` records source identities, MP4 PTS, projected joint counts,
  validity and original per-side outcomes.
- `preview.html` presents the images and outcomes for human review.
- `run.json` records consumed artifacts and output hashes, including partial
  image outputs if a later frame fails.

The request template is `preview.json`. Future authorized usage is
`python -m track3.features.hand_motion preview --request track3/features/hand_motion/preview.json`;
this command remains prohibited under the current no-execution instruction.
The report is explicitly `review_only: true`, `quality_approved: false`. It covers
selected 2D projections, not full-clip identity, world orientation, surface contact,
metric accuracy or retargeting behavior. It does not promote campaign inputs or
change hand data. Preparation of this operation adds no dependency or permission.

### Neutral parked-hand template

`parked_template.py` owns `ParkedTemplateRequest`, generation and persistence.
The CLI operation is `parked-template`; the producer kind is
`hand_motion.parked_template`. Its independent `parked_template_authorized` flag
is false. This operation is **model execution**, even though it uses CPU and no
video. It requires the same approved, pinned MANO runtime and user-provided
licensed left/right assets as kinematics. No runtime or assets were provisioned,
and no template was generated by this implementation work.

The implemented method evaluates each anatomical side with zero shape, zero
axis-angle pose, `flat_hand_mean=True` and zero model-origin translation. It
recentres all geometry on the wrist and applies one proper rotation to joints,
surface vertices and anatomical joint frames together. The geometric convention
places wrist-to-middle-MCP along +x, right radial/left ulnar across-palm direction
along +y, and their cross product along +z. This defines the palms-down neutral
reference without reflecting either hand. Metric dimensions and proper rotations
are required. Its anatomy, orientation and retargeting behavior still need future
execution evidence; source code does not prove those properties on the licensed
models.

Outputs live in `outputs/hand_motion/templates/<run-id>/`:

| Artifact | Content |
|---|---|
| `model_rest.npz` | Raw MANO joints, frames and surfaces, zero parameters, model wrist and alignment rotations |
| `parked_hand.json` | Wrist-relative joints and surfaces, world-aligned anatomical frames, source asset/backend hashes |
| `run.json` | `scope: model_asset_template`, `observed: false`, source hashes and operation status |

The template has no episode, split or measured trajectory. It can be reused by
both scene modes through an explicit `parked_hand_run`; campaign queues accept
the same shared reference. Generated geometry is a neutral model reference and
never marks a missing hand as observed. Scene assembly retains the template and
uses its surface radius when placing hands outside the object bounds. Robot
clearance and behavior remain matters for the later retargeting/rollout stages.
The old null `scene_assembly/parked_hand.json` is no longer a scene input.

### Implemented lift boundary

`LiftHandsRequest`, the CLI, input handling, geometry, persistence, `lift.json`
and `runtime.json` live in this directory. Only project paths and provenance use
shared technical primitives. There is no hand model import or subprocess launch.

The request selects six explicit completed producers: `hand_motion.kinematics`,
`object_masks.generate`, `depth.generate`, `stereo_depth.generate`,
`stereo_depth.anchor` and `camera_motion.estimate`. These must have matching
episode, dataset revision and pinhole-grid identity, and `video_only` provenance.
The kinematics record must hash its selected mask and depth run manifests; masks
and stereo must hash the selected depth manifest; anchor must hash depth and
stereo; camera must hash depth, stereo and anchor. Each consumed NPZ/JSON is
checked against its producer's artifact hash when execution is eventually enabled.

The unprovisioned hand and full-video mask model backends remain
prerequisites. Geometric opening clusters are not accepted as hand masks.
This implementation does not make an end-to-end hand reconstruction available.

The implemented kinematics producer writes `kinematics.npz` with:

| Field | Shape | Contract |
|---|---|---|
| `sides` | 2 strings | Exactly `left`, `right` |
| `frame_index`, `pts` | int64 T | Full RGB clip clock, exactly matching camera motion |
| `valid` | bool T×2 | A finite camera-relative kinematics estimate is available |
| `joints_camA` | T×2×21×3 | Metric joints in MANO order, including the regressed camera translation |
| `joint_rotations_camA` | T×2×21×3×3 | Proper absolute joint rotations in the camera frame, MANO axes |
| `surface_points_camA` | T×2×V×3 | Metric hand surface samples in the same pose and camera translation as the joints |
| `wrist_orient_camA`, `finger_pose` | T×2×3, T×2×45 | Anatomical, un-mirrored axis-angle parameters |
| `betas` | 2×10 | Fixed episode shape; an entirely absent side may be NaN |
| `score` | T×2 | Finite detector score in [0, 1] for observed estimates |
| `mask_index` | integer T×2 | Anatomical side's index into the selected mask run's `instances` |

Each observed side must map to a distinct `hand_0` or `hand_1` layer. The lift
does not guess handedness from layer order. The kinematics manifest declares
`joint_order: MANO21`, `joint_frame_convention: MANO`, `units: metres`,
`orientation_source: mano_fk`, `hand_model`, and
`output_parameter_form: anatomical_unmirrored`. Its `input_parameter_form` is
either `anatomical_unmirrored` or `mirrored_right_hand`; `unmirroring_applied`
must be false or true respectively. The lift carries that conversion history
forward and never mirrors parameters again. The geometric orientation fallback
described below is not implemented or accepted by this first lift operation.

Depth fitting uses observed per-frame stereo/monocular scale only. Direct stereo
depth takes precedence at its valid RGB pixels; scaled monocular depth supplies
the remaining valid pixels. Missing scale never falls back to an episode median
or another frame. Without a scale, direct stereo can still supply support.

The fit projects the surface samples, keeps the nearest sample per pixel, excludes
pixels shared by the two hand masks, and uses a median/MAD fit to depth residuals.
It translates the entire hand along the wrist's camera ray, preserving the wrist's
image position and all metric bone lengths and orientations. It reprojects after
each update and measures support and residual again after the final update.
This is a bounded surface-sample approximation, not triangle rasterization;
surface density, self-occlusion and alignment quality remain unmeasured.

The request sets score, visible-joint fraction, unique depth-pixel support,
iteration count, maximum optical-axis depth shift and residual limits. Their
defaults are implementation choices, not measured accuracy claims. No temporal
filling or smoothing occurs here. A camera-relative fit can be valid while its
world pose is invalid. World rotations use `R_world_camA @ R_camA_joint`.

For future authorized use only, the entry point is
`python -m track3.features.hand_motion lift --request track3/features/hand_motion/lift.json`.
Do not run it under the current instruction. NumPy loads only after authorization;
no new production dependency is added. The template names episode 39, contains
no input runs, and grants no execution permission. Episode 41 remains excluded.

### Implemented kinematics boundary

`KinematicsRequest`, `kinematics.py`, `_mano.py` and `kinematics.json` implement
the CPU operation. For future explicitly authorized use in the isolated MANO
environment, the entry point is `python -m track3.features.hand_motion kinematics`.
This is not a command to run under the current instruction.

The request names an `hand_motion.estimate` run, the same full-video mask run and
the same depth run. The estimate must hash both parent manifests and share their
pixel grid. All producers must have completed, video-only provenance for the
same episode and dataset revision. Raw estimate arrays are loaded with pickle
disabled. The implemented estimator writes the contract below.

Its `estimates.npz` contract is:

| Field | Shape | Meaning |
|---|---|---|
| `sides` | 2 strings | Exactly `left`, `right` |
| `frame_index`, `pts` | int64 T | Ordered full-clip frame indices and original MP4 PTS |
| `valid`, `score`, `mask_index` | T×2 | Availability, detector score in [0, 1], and anatomical side's named mask layer |
| `wrist_orient_camA` | T×2×3 | Axis-angle global orientation in the declared parameter form |
| `finger_pose` | T×2×45 | Fifteen full axis-angle rotations in the same parameter form |
| `translation_camA_m` | T×2×3 | Metric translation of the MANO model origin into the camera frame |
| `betas` | 2×10 | One shape vector per side, fixed across the episode |

The estimate manifest declares `hand_model`, `output_parameter_form`
(`mirrored_right_hand` or `anatomical_unmirrored`),
`translation_convention: mano_model_origin_camA_m`, and
`pose_representation: axis_angle_full_flat_mean`. PCA coefficients, a mean-pose
offset, weak-perspective crop translations and wrist-centered translations are
not interchangeable with these fields and are refused at this boundary.

When the input declares mirrored parameters, kinematics negates the y/z
axis-angle components of the left wrist and all fifteen left finger rotations
together. It does not reflect the camera translation. Anatomical parameters pass
through once. Both side-specific MANO models use `use_pca=false`,
`flat_hand_mean=true` and `center_idx=None`; the model-origin translation is added
to their joints and all 778 surface vertices. `AxisLayerFK` supplies anatomical
joint frames, and each fingertip takes its distal joint's orientation according
to the pinned upstream `TRANSFORMS_TO_JOINTS` mapping. No geometry-derived
orientation substitute is used.

The API/convention references are the existing read-only
`upstream/video_to_data/reconstruction/modules/v2d_task_library_loader/lib/read_mano.py`
and the pinned FlashCHORD `retarget/params.py`. Neither reference is imported or
executed by this feature. The adapter imports the optional `manotorch` library
only after the operation and runtime gates pass.

`backends.json` currently points at `.runtime/mano` and `models/hand`, with null
installation-record pins. It cannot load a backend in that state. Future approved
provisioning must record:

- `backend: manotorch`, `status: completed`, `user_approved: true`, the exact
  project environment path and its `packages` version mapping, including NumPy,
  PyTorch and manotorch;
- a unique `files` list of project-relative installed-file `path`/`sha256`
  entries, including the imported `manolayer.py` and `axislayer.py`;
- `licensed_assets_acknowledged: true` and an `assets` mapping for
  `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`, each containing its project-relative
  `path` and `sha256`;
- the completed installation record's path and SHA-256 in `backends.json`.

The runtime gate compares the executing environment, recorded versions, listed
source files and licensed assets before creating MANO layers. Cache and temporary
paths are set inside project storage; bytecode writes are disabled before optional
imports. No installer, download operation, package requirement change or model
file is included in this implementation. Provisioning remains subject to the
existing dependency-confirmation rule, independently of execution authorization.

Batching is bounded by the request. Shapes, finite values, proper rotations and
positive wrist depth are required of backend results. An absent side stays
absent. Failure/interruption during kinematics preserves only successfully
computed hand batches as valid in a partial `kinematics.npz`, with a failed or
interrupted run manifest that downstream consumers reject. This behavior has not
been exercised.

### Implemented estimation boundary

`EstimateHandsRequest`, `estimate.py`, `_wilor.py` and `estimate.json` own
detection, temporal anatomical assignment, regression, shape aggregation and
persistence. The future entry point is `python -m track3.features.hand_motion
estimate` in the isolated WiLoR environment; execution remains disabled.

The handler reads the completed depth preprocessing, the same full-video mask
run and the pinned RGB video/clock inventory. Each frame is decoded in original
PTS order and remapped to the depth pinhole grid. WiLoR uses a centered camera
with a scalar focal length; the handler therefore makes a second calibrated
remap to a virtual camera with equal focal lengths, centered principal point and
the same optical axes. Its focal length is the smaller input focal length.
Masks and the valid pixel region undergo the same remap. This changes the image
sampling, not the 3D coordinate frame. Padding is invalid. `inference_grid.npz`
records the virtual-to-pinhole remap, validity and both intrinsic matrices.

The adapter follows the previously recorded WiLoR-mini commit
`ebec42f94c389070cdd7dda6fd1bf0b4a659c960`. The pinned
[pipeline source](https://github.com/warmshao/WiLoR-mini/blob/ebec42f94c389070cdd7dda6fd1bf0b4a659c960/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py)
defines the constructor focal scaling and separate detector/bounding-box APIs.
The estimator reads detector box confidence directly, matches boxes to the two
hand-mask layers, then supplies the selected anatomical side to bounding-box
regression. It never substitutes a constant confidence for a detector score.

Box-to-mask assignment maximizes confidence-weighted intersection-over-union
across both masks with unique detections. Near-tied mask assignments are dropped.
Anatomical side assignment uses detector handedness, recent spatial continuity
and mask identity. It may correct a detector's handedness flip only while a
recent side track exists within the configured displacement bound. Near-tied
side assignments remain absent. Regression runs with that selected side, so a
corrected label also changes the crop's handedness processing. Gaps retain NaN
parameters; a previous pose is never copied into an unobserved frame.

The pinned [model forward method](https://github.com/warmshao/WiLoR-mini/blob/ebec42f94c389070cdd7dda6fd1bf0b4a659c960/wilor_mini/models/wilor.py)
reverses the input channels before RGB normalization, so this adapter supplies
OpenCV BGR images to both detector and regressor. The pinned pipeline already
converts left wrist and finger axis-angle parameters into anatomical form.
Estimation records `anatomical_unmirrored`; kinematics consequently performs no
second conversion. The model origin translation remains in the shared optical
camera axes. Subsequent lifting fits its depth to metric observations.

The output includes `betas_per_frame` (float32 T×2×10) for the accepted estimates
and `boxes_virtual_xyxy` (float32 T×2×4). `betas` is the component-wise median of
accepted shape vectors for each side. Kinematics uses that fixed shape instead
of the per-frame shape. The detector-score floor defines which observations may
contribute; an entirely absent side keeps NaN shape values. Scores, match IoUs,
reported and assigned sides, and mask indices are recorded in frame outcomes.

`backends.json` points at `.runtime/wilor` and `models/hand/wilor`, with null
installation pins. Future separately approved provisioning must create a record
with `backend: wilor-mini`, the exact source commit above, `status: completed`,
`user_approved: true`, the environment path, package versions and installed-file
hashes. Required distribution entries include `numpy`, `torch`, `wilor-mini`,
`ultralytics` and `opencv-python`; the installed-file inventory must include the
imported pipeline and model modules. No dependency lock or installer is supplied
by this implementation and the backend has not been installed.

Its `assets` mapping records project-relative `path`/`sha256` entries for all four
files under `models/hand/wilor/pretrained_models/`: `mano_mean_params.npz`,
`wilor_final.ckpt`, `detector.pt`, and the user's licensed `MANO_RIGHT.pkl`.
The record must acknowledge licensed assets. The constructor cannot supply
missing weights or a MANO mirror: the adapter requires all asset hashes before
loading and replaces its download entry point with an explicit refusal while
the model is active. Offline flags and disabled Ultralytics auto-installation
are set; caches and temporary directories remain inside project storage.

GPU ownership covers model creation and every inference call. At most one
selected hand crop is regressed per call, in float32. Request memory allowances
are admission policies, not measured memory requirements. Failed/interrupted
estimation preserves completed observations in a partial NPZ and retains the
failure status; downstream kinematics accepts only completed producers.

This implementation has no runtime, identity or reconstruction-quality evidence.
Box/mask matching can reject strongly occluded hands, tracking can lose identity
at crossings or long gaps, and the virtual camera can crop or pad the field of
view. These limits remain visible through validity and run outcomes.

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `ego_cam_a` clip on the pinhole grid | Dataset, [depth](../depth/README.md) | Colour frames, intrinsics |
| `outputs/object_masks/.../frames` | [object_masks](../object_masks/README.md) | The two hand layers, to seed and check detection |
| Depth in metres | [depth](../depth/README.md), [stereo_depth](../stereo_depth/README.md) | Depth on hand pixels |
| `outputs/camera_motion/.../trajectory.npz` | [camera_motion](../camera_motion/README.md) | `world_T_camA` |
| `models/hand/models/MANO_RIGHT.pkl`, `models/hand/models/MANO_LEFT.pkl` | The user, from the MANO site | Read by `kinematics` only |

## Method

1. **Detect and assign sides.** Find up to two hands per frame and label each
   left or right. At most one hand per side is kept. A side's track is continued
   by position between frames, so a detector that flips a label for one frame
   does not swap the hands.
2. **Regress.** For each hand crop, a hand-pose network gives MANO wrist
   orientation in the camera frame, 15 finger joint rotations and a shape vector.
   Networks of this family store left-hand parameters in mirrored right-hand
   form (`RG/scripts/retarget/ego_recon_to_dexmate_sharpa.py:313-326`). The output
   records which form it holds for both wrist and finger rotations. `kinematics`
   un-mirrors both together once, using the same convention. The final
   `hands.npz` contains both in anatomical, un-mirrored form.
3. **Fix the shape.** Use one shape vector per side for the whole episode, the
   median over confident frames. A hand does not change size during a clip.
4. **Run forward kinematics.** From the parameters, compute the 21 joints and
   their orientations in MANO's convention, relative to the wrist.
5. **Place in depth.** A network estimating from one crop is reliable in image
   position and unreliable in distance. Keep the image position. Set the distance
   so the visible hand surface agrees with metric depth on the hand's mask
   pixels, using a robust fit over those pixels.
6. **Lift.** Transform to the world with `world_T_camA` only where the camera
   pose and the hand estimate are valid.
7. **Mark validity.** A hand is valid in a frame when it was detected, its score
   is above the floor, and enough of it is inside the image. Everything else is
   invalid and left as NaN. `world_valid` also requires a valid camera pose; world
   joints and rotations are NaN where it is false. [scene_assembly](../scene_assembly/README.md) decides
   what an absent hand does, because upstream needs both hands on every frame.

**Without MANO (Design fallback).** If the MANO files are not available, joint
positions can still come from a landmark model, and the wrist and fingertip
orientations can be built geometrically: the wrist frame from the wrist and the
index and pinky knuckles, each fingertip frame from the direction of its last
bone. The construction has to reproduce MANO's axis convention, which would be
written from the published model description and checked only by looking at
retargeted hands. It is a fallback because a convention error there is silent.

## Outputs

`outputs/hand_motion/<split>/episode_XXXXXX/<run-id>/hands.npz`

| Field | Type and shape | Meaning |
|---|---|---|
| `sides` | string 2 | `["left", "right"]`, the index into every array's second axis |
| `valid` | bool T×2 | Camera-relative hand estimate satisfies step 7 |
| `world_valid` | bool T×2 | Hand `valid` AND camera_motion `valid`; mask for world joints and rotations |
| `joints_world` | float64 T×2×21×3 | Metres, MANO joint order |
| `joint_rotations_world` | float64 T×2×21×3×3 | MANO joint-frame convention |
| `wrist_orient_camA` | float32 T×2×3 | Axis-angle, anatomical and un-mirrored after `kinematics`; same form as `finger_pose` |
| `finger_pose` | float32 T×2×45 | Axis-angle, anatomical and un-mirrored after `kinematics`; same form as wrist orientation |
| `betas` | float32 2×10 | One shape per side |
| `score` | float32 T×2 | Detector score |
| `in_frame_fraction` | float32 T×2 | Share of the 21 joints that project inside the image |
| `depth_support` | int32 T×2 | Hand pixels with valid depth used in step 5 |
| `frame_index`, `pts` | int64 T | Frame identity |

The implemented lift additionally writes `joints_camA` (float64 T×2×21×3),
`depth_residual_m` and `depth_shift_m` (float32 T×2). Camera joints and pose
parameters are NaN where `valid` is false; world joints and rotations are NaN
where `world_valid` is false. Depth diagnostics may remain populated for rejected
fits. `in_frame_fraction` is computed after fitting, and is NaN when fitting
could not produce candidate joints. An entirely absent side has NaN shape values.

`run.json` records the hand model, the orientation source (`mano_fk` or
`geometric`), the regressor's input parameter form and the output parameter
form (`anatomical_unmirrored`). It records whether the joint un-mirroring of
wrist and finger rotations was applied; already anatomical input is not converted
again. Intermediate regressor output preserves both rotations as regressed in
the same declared form. Intermediate kinematics output records the conversion
and camera-relative joints; these files pass between the two Windows processes.

## Configuration and authorization

`runtime.json` holds `estimate_authorized`, `kinematics_authorized`,
`parked_template_authorized`, `lift_authorized` and `preview_authorized`, all false.
Episode operations also require the episode in the explicit
split allowlist; validation and evaluation have separate authorization flags,
also false. Changing a configuration is not authorization under the current
implementation-only instruction.

## Dependencies

Proposed, not installed. Each needs the user's confirmation, and the first two
need the user's own action.

| Item | Licence | Note |
|---|---|---|
| MANO model files | MANO licence; registration and download by the user | Never committed, never redistributed |
| A MANO forward-kinematics library | `manotorch` is GPL-3.0 | Kept in its own environment and called as a separate process, as upstream does |
| WiLoR weights and code | Model under CC BY-NC-ND | 8.66 GB with the HaMeR archive, publisher figures in `reports/model-readiness/README.md` |
| The hand detector used by WiLoR | Ultralytics terms apply | Needs a licence decision |

The challenge's release terms exempt pretrained models under incompatible
licences from the release requirement (**Official**, `KIT/config/tracks.json:335`).
Whether non-commercial models may be used at all is the organisers' call.

## Resources

The implemented kinematics, lifting and preview operations are CPU-only. Estimation uses
the shared GPU lock. Memory and runtime are unmeasured.

## Failure handling

An episode in which a side is never detected is not a failure: the side is
recorded as absent throughout. An episode in which no hand is ever detected fails
with `no_hands`. If detections exist but none passes depth and visibility limits,
the lift fails with `no_valid_hands`. Low scores, absent masks, insufficient depth,
excessive shifts/residuals and invalid camera poses have explicit per-side frame
statuses. Missing or mismatched input artifacts fail the run rather than becoming
successful observations.

`run.json` records failure/interruption and frame outcomes reached before the
failure. `hands.npz` is atomically written after frame processing; an interruption
before that point does not promise a partial array artifact. The no-hands failures
retain an all-invalid NPZ for diagnosis but their run status is failed and
downstream consumers reject it. No automatic resume is implemented.

## Separation of public labels

The public release has no hand labels. Nothing to separate.

## Acceptance evidence

Once runs are authorized: overlays of the 21 projected joints sit on the hands at
approach, grasp and release; fingertip positions at grasp lie within a centimetre
or two of the object's surface; the hand's distance from the camera changes
smoothly.

`preview` implements the selected-frame overlay portion only. Its source has not
been executed or tested, and no rendered review or accuracy evidence exists.

## Open questions

| Question | Who can answer |
|---|---|
| Will the user register for and download MANO? | The user |
| May models under non-commercial licences be used? | Organisers |
| What did the organisers' own baseline do with frames in which a hand is out of view? | Organisers |

## Bounded temporal refinement and contact hypotheses

`refine.py` and `contacts.py` add opt-in request/handler operations, with examples
in `refine.json` and `contacts.json`. Both new authorization flags are false;
existing defaults and episode gates retain their behavior. These operations admit
only public training/tuning episodes 0/11/12/18/21/23/26/31/39/42. Episodes
41/43/44 and evaluation inputs are rejected even if other feature gates change.
No preparation, model import, asset download, installation or simulation is added.

`refine` consumes an exact completed `hand_motion.lift` run and its camera run,
including both as hashed direct parents. The explicit source interval is at most
384 frames. Integer source frame IDs and PTS are retained and checked against the
pinned RGB clock; seconds use its rational time base rather than nominal FPS.
A five-frame weighted local linear wrist fit changes at most 5 mm by default.
Only valid metric depth-supported wrists participate, windows never span missing
support or timestamp gaps, and high acceleration bypasses smoothing. A rejected
correction retains the initializer and its reason. There is no interpolation.

The correction translates every joint on a hand equally, preserving existing FK
articulation, rotations, bone lengths and episode shape. This is a conservative
sequential wrist refinement, not articulated MANO fitting. Camera-space fields
remain the unrefined initializer; world-space corrections are explicit.
Image-supported articulation and shape optimization remain deferred until actual
landmark/visibility evidence and licensed assets exist. Detector confidence does
not become joint visibility. `hands.npz` retains the original lift arrays and adds:

| Field | Shape and meaning |
|---|---|
| `unrefined_joints_world`, `joints_world` | T × 2 × 21 × 3, input and corrected world metres |
| `correction_world_m` | T × 2 × 3, applied rigid translation |
| `joint_observed` | T × 2 × 21 bool, all false for FK-only inputs |
| `joint_inferred` | T × 2 × 21 bool, valid inferred articulation |
| `wrist_depth_supported` | T × 2 bool, metric hand-surface depth support, not observed landmarks |
| `refinement_status` | T × 2, explicit accept/bypass/failure reason |
| `velocity_world_m_s`, `acceleration_world_m_s2` | T × 2 × 21 × 3 with corresponding `velocity_valid`, `acceleration_valid` masks |
| `bone_length_m`, `bone_length_correction_m` | T × 2 × 20, bone lengths and change due to correction |

Derivatives use adjacent valid source samples and nonuniform PTS spacing. Invalid
values remain NaN. Acceleration compares adjacent secant velocities at their true
midpoint spacing; it is not a nominal-frame second difference.

`contacts` consumes refined hands, `object_pose.compose`, and `object_mesh.fuse`
with the same exact tracking parent and inherited body origins. Both hand lift
and composed object poses must bind the same exact camera run. Selected hand
frames select their corresponding object rows without relabeling source IDs.
It uses hashed `*_support.npz` observed voxel centroids, never completed mesh
backsides, and bounds support to one million points per object. Distances are
unsigned distances to observed centroids, padded by the voxel diagonal for the
proximity hypothesis; they are not signed collision distances or calibrated
uncertainty. Partial geometry cannot establish absence of contact or penetration.

Hypotheses combine metric wrist support, inferred articulated locations,
object-relative stability, convergence/divergence and upward object motion.
They distinguish approach, contact, grasp, lift, release and support transition.
Missing support, clock gaps or unavailable derivatives produce `unknown`.
Grasp lacks observed opposition/force evidence; support transition lacks an
observed supporting surface. Both retain these ambiguity reasons. Every interval
has `accepted: false` and `physical_contact_confirmed: false`. Phase names describe
estimated event hypotheses only; downstream physical-contact consumers must not
promote them merely because several fingertips are close to a surface.

`contacts.json` schema version 1 includes source split/episode, `frame_index`,
`pts`, rational `time_base`, exact hashed hand/pose/mesh run identities, observed
surface lineage and intervals. Each interval contains `side`, `object_name`,
`start_frame`, exclusive `end_frame`, inclusive endpoint `start_pts`/`end_pts`,
`phase`, `candidate_phase`, `ambiguity`, `support`, `accepted` and
`physical_contact_confirmed`. Intervals never bridge clock gaps.
`contacts.npz` includes T × 2 × O × 21 unsigned observed-voxel distances,
object-relative joint coordinates and sliding-speed proxies, T × 2 × O
`geometry_valid` and all-false `physical_contact_confirmed`. `penetration_m`
is all NaN with an explicit unavailable reason, since this partial observed
surface is not a closed signed collision surface. No confidence values are
invented. These outputs provide diagnostics and abstention, not contact labels.

All persisted runtime files use the existing canonical Z: path gate, atomic NPZ
writer, completed-parent/hash admission and failure run records. Source code was
implemented without imports, tests, checks or runtime execution. No accuracy,
speed, memory-fit, contact or physical improvement has been measured.
