# Coursework depth model from scratch

Implemented only. No notebook cells, tests, validation commands, imports, model
training, video decoding or inference were executed for this feature. No trained
checkpoint or training pairs exist yet. All three operations remain disabled in
[runtime.json](runtime.json). No packages were installed or added.

This is an original, randomly initialized CNN alongside the existing MoGe2 depth
stage. It uses the local coursework's convolution, pooling, regression, dropout,
regularization and optimization concepts. It does not load a pretrained encoder
or copy MoGe's architecture. The implemented training-data adapter consumes
explicitly selected **MoGe2 pseudo-depth targets**, so training from random weights
does not imply independence from a pretrained teacher's labels. Teacher accuracy
and metric scale would carry through to the new model and remain unmeasured.

## Coursework to implementation

The sources were read as notebook text; none of their cells were executed.
[coursework.json](coursework.json) identifies the files and zero-based cell indices.

| Local course material | Application here |
|---|---|
| Beginner guide, Part 1, cells 5–7 | Explicit forward pass, loss, backward pass and optimizer update |
| Beginner guide, Part 3, cells 33–37 | Regression targets and separation of development from held-out episodes |
| Beginner guide, Part 6, cells 65–68 | Adam weight decay and bottleneck dropout |
| Beginner guide, Part 8, cells 77–84 | RGB divided by 255, convolution/pooling and training-only brightness augmentation |
| Week 4 lab, cells 26, 27, 30, 33 | Stacked convolution blocks and capacity control |

The teaching examples use Keras. This implementation translates those concepts to
the installed PyTorch packages in `.runtime/moge-python313`; it needs neither
TensorFlow nor Keras. The upsampling decoder, skip connections, depth-specific
masking and camera geometry are original adaptations beyond the course examples.

## Vertical slice

All depth-model behavior and configuration live in this folder:

- `prepare.py` and `prepare.json`: request, local teacher-artifact reader, RGB alignment and paired-data persistence.
- `train.py` and `train.json`: request, dataset access, random initialization, optimization and checkpoint persistence.
- `predict.py` and `predict.json`: request, checkpoint loading, calibrated video preprocessing and depth persistence.
- `_network.py`: private CNN and masked regression loss.
- `_common.py`: private camera, input-selection, execution and artifact conventions for these operations.
- `runtime.json`, `coursework.json`, this README and the thin CLI/API entry points.

Shared technical primitives are repository path constraints, atomic JSON and
hashing, and `track3/gpu.py`. Both depth slices use the same cooperative GPU lock.
There are no horizontal service/repository layers or private imports from the
MoGe feature. The scratch environment selector excludes MoGe's source trees.

## Model and learning contract

The encoder uses two 3×3 convolutions with ReLU at each level, channels 24, 48,
96, and a 192-channel bottleneck, with three 2×2 max-pooling steps. The bottleneck
has 0.1 spatial dropout. Three decoder blocks upsample to their matching encoder
resolution and concatenate skip features. A 1×1 convolution and softplus plus
0.001 produce positive optical-axis z-depth at each input pixel. There is no
classification head, normal prediction, temporal model or learned confidence.

Training uses masked mean absolute error between log prediction and log target.
The loss retains absolute scale; it does not align each image's predicted scale
to the teacher. Invalid target values are sanitized before logarithms and excluded
from the loss. Defaults are 20 fixed epochs, batch size 2, Adam at 0.001 with
0.00001 weight decay, gradient norm clipping at 1, and seed 31. The reported epoch
loss is weighted by the number of supervised pixels. Only RGB brightness changes;
there are no spatial augmentations that could misalign depth or camera intrinsics.

There is no validation loader, test pass, hyperparameter search, early stopping,
best-model selection or accuracy claim. Each epoch saves a local weights checkpoint;
the final epoch is the default result. Checkpoints contain training metadata and
model weights, not optimizer state for resuming. Every training operation starts
from random weights. A seed records intent but does not establish reproducibility
across GPU/library versions.

## Input and artifact contracts

Preparation requires completed `depth` run manifests from explicitly chosen public
development episodes. The empty `teacher_runs` list is intentional: no teacher
inference has been authorized or performed. The adapter checks the recorded video,
calibration and frame-artifact identities during a future preparation operation.
It never invokes MoGe or generates its own targets implicitly.

RGB frames are remapped directly from the 14-coefficient factory Perspective camera
to a centered, square-pixel pinhole grid, default maximum side 384. No extra 180°
rotation is applied. Teacher depth is sampled onto that grid by the corresponding
pinhole rays with nearest-neighbour sampling, avoiding interpolation across depth
edges or invalid pixels. RGB and depth retain the same optical coordinate frame.
Labels outside 0.05–20 metres are excluded; samples with fewer than 64 valid labels
are recorded as rejected. The range is an initial implementation choice, not a
measured property of the scenes.

`outputs/scratch_depth/prepare/<run-id>/dataset.json` records each pair's episode,
frame index, original MP4 PTS, artifact hash and teacher artifact lineage. Samples
contain `rgb` (uint8 H×W×3), `depth_m` (float32 H×W, NaN when invalid), `valid`
(boolean H×W), and `input_intrinsics_pixels` (3×3). All included episodes must
belong to the frozen development split. Frames are never randomly split between
training and held-out sets. A single episode cannot establish generalization.

`outputs/scratch_depth/train/<run-id>/` holds the run manifest and per-epoch
checkpoints under `checkpoints/`. Loading for prediction requires an intact
checkpoint from a completed training run and uses `torch.load(weights_only=True)`
with strict state loading. Untrained random weights are not used for predictions.

`outputs/scratch_depth/predict/<run-id>/` holds `run.json`, `preprocessing.npz` and
one NPZ per requested frame. Each frame contains `depth_m`, `valid`, `input_valid`
and `input_intrinsics_pixels`. Prediction validity means an in-bounds input ray and
a finite depth within the training label range; it is not a calibrated confidence
or occlusion mask. `run.json` preserves MP4 PTS and checkpoint provenance. Parquet
poses, capture-time clocks and world transforms are not inferred.

Prediction uses the checkpoint's exact preprocessing geometry. It is currently
limited to public development RGB episodes. It processes one frame at a time and
does not load the teacher. Partial artifacts remain on failure with a failed run
record; new runs use unique directories. GPU operations reserve the common project
lock and record device-wide sampled memory separately from PyTorch peaks. These
resource mechanisms have not been exercised for this model.

## Planned shared-record migration — not implemented

The target run record in
[artifact-contracts.md](../../../docs/design/artifact-contracts.md) is not yet
implemented here. `_common.py` currently writes `feature`, `input_hashes` and
separate `source-files.json`, with no `provenance` tag. Each operation must add
`kind`, path/hash `inputs`, `sources` including the source lock, and inherited
`provenance`. Preparation inherits teacher tags, training inherits dataset tags
and prediction inherits checkpoint and episode lineage. Missing legacy tags
cannot be assumed `video_only`. The teacher reader must migrate together with
these records before downstream consumers are enabled.

These are future implementation tasks, with no change to the disabled preparation,
training or inference flags. The planned evaluation/validation split extension
is for the MoGe2 depth slice; this scratch model remains limited to development
episodes.

## Future operation sequence — disabled

The following commands document the implemented API. They have not been run and
are **not authorized under the current implementation-only instruction**.

1. After a separate teacher run is authorized and completed, enter its local
   `run.json` paths in `prepare.json`.
2. An authorized preparation operation returns `dataset.json`; put that path in
   `train.json`.
3. An authorized training operation returns a checkpoint; put that path in
   `predict.json` before any separately authorized prediction.

```powershell
. ./scripts/use-scratch-depth.ps1
python -m track3.features.scratch_depth prepare --request track3/features/scratch_depth/prepare.json
python -m track3.features.scratch_depth train --request track3/features/scratch_depth/train.json
python -m track3.features.scratch_depth predict --request track3/features/scratch_depth/predict.json
```

Each operation checks its own authorization flag before heavy imports or artifact
creation. There is no CLI switch that bypasses the flags. Caches, temporary files,
paired data, logs and checkpoints stay inside this repository. The existing
MoGe2 execution flag remains false and independent of the scratch-model flags.
