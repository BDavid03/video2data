# Depth feature

Status: implemented only. The user explicitly selected implementation without a
run and prohibited testing. This feature has not been imported, executed, tested,
compiled, or used to produce depth results. No dependencies were added for it.

## Vertical slice

This folder owns the complete depth-generation use case:

- `episode41.json`: the default request, covering all RGB frames of development episode 41.
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

`episode41.json` selects RGB camera `ego_cam_a`, all 299 inventoried frames, a
maximum image side of 768 pixels, 1,200 model tokens, and mixed-precision inference.
Frames are processed singly. Optional normals are disabled by default.

`start_frame` is inclusive and `stop_frame` is exclusive; a null stop means the end
of the inventoried clip. Only public development episodes in the frozen split are
accepted. This bounded implementation does not process validation or evaluation
episodes, and does not open Parquet trajectories, hand labels, or object poses.

The existing `configs/perception/moge/local.json` flag
`execution.model_execution_authorized` remains false. The handler refuses to run
before importing torch, loading model weights, opening a video, or writing run
artifacts. A future explicit model-run instruction is needed to change that scope.

The following is a reference invocation, **not an instruction to execute it under
the current implementation-only scope**:

```powershell
. ./scripts/use-moge.ps1
python -m track3.features.depth --request track3/features/depth/episode41.json
```

No test, evaluator, smoke-run, or dependency-resolution command is part of this
feature. Runtime input guards and resource limits belong to the generation
operation; they have not been exercised.

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
`outputs/depth/public/episode_000041/` (or the selected development episode).
Existing results are never overwritten. A run contains:

- `run.json`: request, input/model provenance, preprocessing, a record for every requested frame, operation status, GPU telemetry, timing, and artifact hashes.
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
The overall operation can be completed, failed or interrupted. Partial artifacts
and the last recorded error remain available. A completed depth operation is not
a completed reconstruction or a task-success result.

## Planned integration changes — not implemented

The design registry in
[artifact-contracts.md](../../../docs/design/artifact-contracts.md) describes a
target contract beyond this bounded implementation. Phase 0 must extend input
selection and output routing to explicitly named evaluation episodes and separately
gated public validation episodes; preserve split namespaces and avoid pose-label
access. The current handler still refuses those episodes.

Its run record currently uses `feature`, `input_hashes` and separate
`source-files.json`, without a `provenance` tag. Add `kind`, path/hash `inputs`,
`sources` including the source lock, and explicit inherited `provenance` before
new consumers are enabled. Consumers must refuse untagged legacy records. No
code or configuration flag has changed as part of the design correction.

The existing `episode41.json` is a retained request, not an authorized run. The
user declined depth execution on episode 41; the roadmap proposes episode 39
instead, still requiring explicit authorization.

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
