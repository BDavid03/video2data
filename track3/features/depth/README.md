# Depth feature

The feature began as implementation only. On 18 October 2026 the user authorized
bounded research execution, including MoGe2 teacher generation and wider
experiments. Episode 41 remains excluded. No tests, dependency changes or model
downloads are part of generation; this source update makes no completion claim.

## Vertical slice

The opening-frame batch command is `bash track3/features/depth/depth_opening_frames.sh <split> <episode> [<episode> ...]`.
It invokes the depth handler once per episode, with the same GPU gates and lease.

This folder owns the complete depth-generation use case:

- `generate.json`: the default request, covering frames 0 through 19 of public development episode 39.
- `runtime.json`: separate generation, evaluation and validation gates and explicit episode allowlists.
- `episode41.json`: retained historical request; episode 41 is refused by the request handler.
- `generate.py`: request type, input selection, calibrated preprocessing, local MoGe2 integration, output persistence, and operation handling.
- `track3/gpu.py`: shared technical primitive for cooperative GPU ownership and resource telemetry.
- `__main__.py`: a thin command entry point that invokes the feature handler.
- `__init__.py`: the public `GenerateDepthRequest` and `generate_depth` API.

There are no horizontal controller, service, repository, or model-adapter layers.
Shared imports are limited to repository path constraints, GPU ownership and provenance/JSON
utilities. Episode and timestamp inventories are consumed as explicit input
artifacts. Future features should own their corresponding operations in sibling
folders and consume the depth artifact contract described below.

## Request and execution boundary

`generate.json` selects RGB camera `ego_cam_a`, the first 20 frames of episode 39, a
maximum image side of 768 pixels, 1,200 model tokens, and mixed-precision inference.
Frames are processed singly. Optional normals are disabled by default.

`start_frame` is inclusive and `stop_frame` is exclusive; a null stop means the end
of the inventoried clip. Positive integer `frame_stride` defaults to one and selects
`range(start_frame, resolved_stop_frame, frame_stride)`. For example, start 3,
stop 14 and stride 5 select frames 3, 8 and 13. The decoder advances contiguously
from frame zero using `grab()` for skipped frames and `read()` only for selected
frames; remapping and inference run only on selected frames. No timestamp seeking
or synthetic frame clock is used. The default request retains contiguous frames.
`source_split` explicitly selects `public` or `evaluation`.
Public episodes must belong to the frozen development or validation split.
Validation and evaluation each require their own disabled gate, and every episode
must be explicitly present in its split's authorization allowlist. Episode 41 is
excluded. No operation opens Parquet trajectories, hand labels or object poses.

Generation requires `execution.model_execution_authorized` in
`configs/perception/moge/local.json` and `execution.generate_authorized` in this
slice's `runtime.json`, plus explicit split/episode admission. A disabled gate refuses
before importing torch, loading model weights, opening a video, or writing run
artifacts. Authorization is recorded with each operation's input configuration.

Reference invocation for an authorized request:

```powershell
. ./scripts/use-moge.ps1
python -m track3.features.depth --request track3/features/depth/generate.json
```

No test, evaluator, smoke-run, or dependency-resolution command is part of this
feature. Runtime input guards and resource limits belong to the generation
operation and do not replace separately authorized testing.

## Input and geometry contract

The feature consumes the pinned source lock, explicit episode inventory, existing
RGB presentation timestamp inventory, original camera calibration, installed
MoGe2 configuration, and local checkpoint. The timestamp inventory is historical
audit evidence, not a fresh validation performed by this feature.

Decoded RGB pixels receive no additional 180-degree rotation. All 14 factory
distortion coefficients are used in a single OpenCV remap directly to a centered,
square-pixel pinhole image at the requested size. This makes the model's centered
camera assumption explicit. The optical coordinate basis remains x-right,
y-down, z-forward. No world transform is inferred.

The output pinhole K, original K and distortion, output dimensions, and supplied
horizontal field of view are recorded. The remapping coordinates are saved so a
downstream feature can reproduce the input pixels. Border samples outside the
original image are invalid. There is no second undistortion or implicit crop.

MoGe2 receives the horizontal field of view derived from the output focal length.
Its returned normalized K is converted to pixel-center coordinates using
`[[W,0,-0.5],[0,H,-0.5],[0,0,1]] @ K_normalized`. Both the input K and model K are
retained. This does not establish metric accuracy, dense-stereo agreement, or an
official Track 3 coordinate convention.

Each frame retains its original frame index, integer MP4 PTS and PTS in seconds.
Parquet nominal timestamps and capture times are not read, rescaled, or guessed.
The three clocks must not be treated as interchangeable by downstream consumers.

## Output contract

Every generation operation creates a new timestamped directory beneath
`outputs/depth/<source_split>/episode_XXXXXX/` for the explicitly selected episode.
Existing results are never overwritten. A run contains:

- `run.json`: request (including stride), input/model provenance, preprocessing, selected-frame records, `selected_frames`, `resolved_stop_frame`, operation status, GPU telemetry, timing, and artifact hashes.
- `source-files.json`: content hashes for the feature and the three pinned source trees.
- `preprocessing.npz`: maps from each output pixel to original distorted RGB coordinates, the usable-input mask and output camera K.
- `frames/frame_000000.npz`, etc.: one compressed numeric artifact per completed frame.

| Per-frame NPZ field | Meaning |
|---|---|
| `depth_m` | H×W float32 camera-z depth; metres estimated by MoGe2 |
| `valid` | H×W boolean: model mask AND usable source sample AND positive finite depth |
| `model_mask` | Original binary MoGe2 mask |
| `input_valid` | Geometric sampling mask from preprocessing |
| `model_intrinsics_normalized` | 3×3 K returned by MoGe2 |
| `model_intrinsics_pixels` | 3×3 model K in pixel-center coordinates |
| `input_intrinsics_pixels` | 3×3 K used for the remapped input |
| `normal_camera`, `normal_valid` | Optional model normals and their separate finite-value mask |

Invalid depth is NaN and must be excluded using `valid`. The model's binary mask
is not a calibrated confidence score; confidence is explicitly unavailable. No
temporal interpolation, smoothing, contact labels, camera trajectory, or object
pose is inferred here. A consumer can backproject valid camera-z depth using the
saved model K, but must supply camera motion before constructing a world scene.

The manifest distinguishes not-processed, running, completed and failed frames.
Its `frames` list contains only selected indices; `completed_frames` counts actual
saved predictions rather than decoder advances. Each retains its original
inventory PTS. Skipped frames have no depth artifact or manifest frame record.
The overall operation can be completed, failed or interrupted. Partial artifacts
and the last recorded error remain available. A completed depth operation is not
a completed reconstruction or a task-success result.

## Shared run record and split integration

The design registry in
[artifact-contracts.md](../../../docs/design/artifact-contracts.md) describes a
shared contract implemented by this slice. Input selection and output routing
preserve public and evaluation namespaces, with separately gated validation and
explicit episode authorization.

The run record adds `kind: depth.generate`, path/hash `inputs`, inline `sources`
including the source-lock hash, `provenance: video_only` and wall-clock `timing`.
It preserves `feature`, `input_hashes` and `source-files.json`. The shared consumer
helper requires completed, explicitly tagged producers and inherits the most
restrictive tag; untagged legacy records are refused. This operation has only
explicit video, calibration, model, configuration and inventory inputs.

The existing `episode41.json` is a retained request, not an authorized run. The
user declined depth execution on episode 41, and request admission continues to
refuse it even when broader experiment flags are enabled.

## Resource and storage behavior

The implementation uses a cooperative `.runtime/gpu-stage.lock` shared with
future project GPU stages. It never removes another process's lock or
automatically deletes a stale lock. Uncoordinated external applications are still
possible, so total device memory is sampled through `nvidia-smi`.

The default admission allowance is 4,096 MiB plus 2,048 MiB of headroom. These are
provisional limits, not measured MoGe2 requirements. The feature stops between
model operations when sampled free memory crosses the headroom boundary. Sampling
can miss short peaks and cannot interrupt an in-flight CUDA operation. PyTorch
allocated/reserved peaks are recorded separately; host peak RAM is explicitly
unmeasured.

All generated caches, temporary files and outputs remain in this repository.
The loader requires the existing checkpoint and enables offline Hub access.
No weights are automatically fetched. The installed source code and environment
are not changed during generation. Output checksums are provenance generated as
part of an authorized operation, not a separate validation run.

## Offline camera/depth challengers (source only)

`challenger.py` owns a separate, disabled `depth.ingest_challenger` operation for
pre-existing ViPE, Depth Anything 3 and Video Depth Anything exports. It imports
no challenger package, launches no model, downloads nothing and fits no scale.
`runtime.json.execution.challenger_ingest_authorized` defaults to false; existing
generation flags do not authorize it. `ingest-challenger.json` is a request shape
with deliberately unresolved paths. An authorized future invocation uses
`python -m track3.features.depth.challenger --request <request.json>` on canonical
Z: only. This pass has not invoked it or created an export.

The request names an explicit manifest and completed video-only `depth.generate`
run supplying the calibrated pixel grid, RGB identity and source clock. Public
training episodes 0/11/12/18/21/23/26/31 and tuning episodes 39/42 are admitted;
41/43/44 and evaluation are always refused. At most 384 increasing unique frames,
maximum side 518, are supported. Missing paths, unsupported conventions and
receipt mismatches fail explicitly; no incumbent or model fallback is selected.
This is result ingestion, not the proposed bounded model experiment.

Export schema version 1 requires these manifest fields:

- `approach`: `vipe`, `da3` or `vda`; `status`: `completed`; `provenance`:
  `video_only`; `split`: `public`; `episode_index`; `dataset_revision`.
- `camera`: `ego_cam_a`; `camera_frame`:
  `OpenCV optical x-right y-down z-forward`; `grid`: calibrated preprocessing
  NPZ SHA256; `source_video_sha256`; `source_time_base_and_frame_rate` copied from
  the explicit calibration producer. Changed calibration or resized grids need
  a separate admitted producer; this adapter does not silently resample.
- `depth_kind`: `metric_optical_z_m` or `relative_optical_z`. Inverse depth, ray
  distance, nested outputs and near-metric claims are unsupported conventions
  requiring an explicit upstream conversion. Near-metric estimates without an
  established unit declaration must remain relative. A metric declaration does
  not establish accuracy or an independent metric anchor.
- `provisioning`: immutable 40-hex `source_commit`, `source_archive` receipt,
  nonempty `weights` and `licenses` receipt lists, and a nonempty `license_review`
  describing all components/checkpoints and restrictions. Each receipt is
  `{ "path": "project-local/path", "sha256": "64-lowercase-hex" }` and must name
  existing bytes. These receipts establish export identity and record terms;
  they do not grant installation, model execution or licensing approval.
- `frames`: increasing entries with `frame_index`, `video_pts`,
  `video_pts_seconds`, `path` and `sha256`. PTS must exactly match a completed
  frame in the calibration producer; no fabricated or positional timestamps.

Each frame NPZ contains `depth` (H×W optical z), boolean `valid` (H×W), and
`intrinsics_pixels` exactly equal to the calibration grid. Optional `confidence`
is an uncalibrated finite score on valid support, not a probability. Invalid or
nonpositive depth and invalid calibration support become NaN plus false validity;
no interpolation fills holes. Relative output uses `depth_relative`, never
`depth_m`. Metric output uses `depth_m` without asserting geometric superiority.

ViPE/DA3 exports may additionally provide a 4×4 rigid `extrinsic` and scalar
boolean `pose_valid`. The manifest then requires `pose_world` (explicit world
frame/gauge description), `extrinsic_direction` (`world_to_camera` or
`camera_to_world`), and `pose_translation_units` (`m` or `relative`). Metric pose
units require a metric depth declaration. Invalid poses stay NaN with false
support. The camera-owned helper in `camera_motion/challenger.py` explicitly
inverts world-to-camera poses, preserves the declared gauge and translation
units, and rejects scale/reflections. Outputs use `world_T_camA` for metres or
`world_T_camA_relative` otherwise. Neither is aligned to the incumbent table
world. VDA camera poses are refused because VDA supplies depth only.

The separate run owns copied calibrated preprocessing, normalized frame NPZs,
SHA256 lineage and a completed/failed `run.json`. Failures after output creation
retain a failed run and any partial frames. Preflight failures create no run.
Existing production consumers still require incumbent producer kinds: ingestion
never masquerades as `depth.generate` or `camera_motion.estimate`. Comparison
consumers must explicitly admit this challenger kind and its units/gauge before
using it. Promotion and independent geometry comparison remain unimplemented
until admitted anchor evidence and a frozen comparison protocol exist.

## Calibrated RGB preview without hand prerequisites

`preview.py` owns a disabled `depth.preview` operation for inspecting early
geometry before any hand model or licensed hand assets are available. Its only
producer parent is an explicit completed video-only `depth.generate` run.
`preview.json` shows the request shape with an unresolved parent path. A future
authorized invocation is `python -m track3.features.depth.preview --request
<request.json>`; `runtime.json.execution.preview_authorized` remains false.
Existing generation flags do not authorize preview decoding.

The producer admits only the ten frozen public training/tuning episodes and
checks the parent episode/revision, pinned clock inventory, exact source RGB hash,
source orientation, completed depth frame PTS and preprocessing hash. It reuses
saved pinhole maps once, retains the calibrated grid, blackens invalid map support
and writes only selected RGB PNGs. It invokes no model and makes no hand,
quality, metric-accuracy or approval claims. There is no `overlay` field because
no overlay was observed or generated.

The request selects 1..64 unique increasing indices and declares a source decode
ceiling (default 4096, hard maximum 32768). Sequential decoding stops at the final
selected index; missing source frames fail explicitly. Preview/cache/temp outputs
require canonical Z:, and local cache/temp configuration occurs before NumPy or
OpenCV imports. A separate run records frame indices and source PTS, PNG SHA256s,
`preview.json`, exact parent identity and failure/interruption outcomes. Each
completed row supplies `rgb: {file, sha256}` for inspection consumption. Inspection
may admit this kind explicitly as an alternative to `hand_motion.preview`; that
alternative introduces no hand dependency or synthetic overlay. No decoder,
model, smoke check or test was executed for this source implementation.
