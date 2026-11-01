# Coursework depth model from scratch

New source-only ablations also have owning CLI routes `research-prepare`,
`research-train` and `research-predict`, using
[research-prepare.json](research-prepare.json),
[research-train.json](research-train.json) and
[research-predict.json](research-predict.json). Their placeholder identities must be
replaced by explicitly admitted source/dataset hashes; all three gates are
false. No preparation, fitting or inference accompanied this source buildout.

Status on 18 October 2026: the expanded
[research investigation](../learning_experiments/README.md#expanded-investigation--completed)
**completed** in the canonical `Z:/hugging-face/video2data` checkout. All five
original scratch-CNN variants trained and completed tuning/holdout inference.
Their eager/BF16 equal-weight ensemble was frozen as the tuning winner, scoring
0.1239556042 tuning MAE and 0.1521616317 holdout MAE against MoGe2 pseudo-depth.
Both primary-environment compiler attempts failed. The earlier
12-trial research baseline exercised preparation only, retaining 40 pairs from
ten training/tuning episodes; those baseline results are separate from this run.
Episode 41 remains excluded. No notebook cells or tests were executed, and no
new dependencies have been installed for the expanded campaign.

This is an original, randomly initialized CNN alongside the existing MoGe2 depth
stage. It uses the local coursework's convolution, pooling, regression, dropout,
regularization and optimization concepts. It does not load a pretrained encoder
or copy MoGe's architecture. The implemented training-data adapter consumes
explicitly selected **MoGe2 pseudo-depth targets**, so training from random weights
does not imply independence from a pretrained teacher's labels. Teacher accuracy
and metric scale would carry through to the new model and remain unmeasured.

## Coursework to implementation

The original implementation recorded reading the sources as notebook text;
none of their cells were executed. Both referenced `coursework/` notebooks are
present in this workspace as of 2026-10-11. Future authorized operations record
their actual source hashes; file presence alone establishes no training or
runtime evidence. The previous 2026-10-05 absence note is superseded.
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
- `_checkpoint.py`: private checkpoint ownership, source identity and training-counter admission.
- `_optimization.py`: explicit PyTorch training choices, retaining conservative defaults.
- `_cuda_loss.py` and `cuda/`: optional custom CUDA loss and autograd integration, with separately gated compilation.
- `compiler_experiments.py`: separately gated real-batch compiler attempts and explicit backend outcomes.
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

Sample admission requires floating depth arrays and boolean masks. It checks
depth bounds before and after float32 conversion, comparing in float64 so stored
rounding cannot widen those bounds. Loss and epoch counts use the resulting
usable-label mask. Both loss backends require identical prediction, target and
mask shapes; broadcasting is refused.

There is no validation loader, test pass, hyperparameter search, early stopping,
best-model selection or accuracy claim. Each epoch saves a local weights checkpoint;
the final epoch is the default result. Checkpoints contain training metadata and
model weights, not optimizer state for resuming. Every training operation starts
from random weights. A seed records intent but does not establish reproducibility
across GPU/library versions.

## Completed comparison campaign

The isolated dataset contained 128 accepted training pairs across eight episodes,
with 32 tuning and 32 holdout pairs assessed separately. All five variants used
ten epochs, batch size four and seed 31. Owned training wall times include data,
checkpoint and bookkeeping work; they are single-seed operation timings, not
isolated kernel measurements or established speedups.

| Variant | Training seconds | Tuning MAE | Holdout MAE |
|---|---|---|---|
| Eager FP32 | 101.79 | 0.1321539216 | 0.1702597208 |
| BF16 | 110.42 | 0.1385482141 | 0.1536269931 |
| Channels-last | 88.52 | 0.1483359807 | 0.1856198889 |
| Fused Adam | 84.63 | 0.1462275573 | 0.1577466423 |
| BF16/channels-last/fused | 92.81 | 0.2023547478 | 0.2175236167 |

The equal-weight eager/BF16 ensemble remained the selected candidate after
holdout assessment. The top-three equal ensemble's lower holdout MAE
(0.1494962110) is recorded without revising that frozen choice. These are
pseudo-label comparisons, not measured scene-depth accuracy or submission
readiness. The [full results](../learning_experiments/README.md#expanded-investigation--completed)
link the completed investigation, frozen selection and per-candidate evidence.
The [rendered report](../../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/report.md)
and [comparison figure](../../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/comparison.png)
resolve in the canonical Z: checkout and retain the broader candidate comparison.

The [investigation request](../learning_experiments/investigate.json) selects at
most 16 MoGe2 teacher frames across each training/tuning clip, using
`ceil(decoded_frame_count / 16)` as the teacher frame stride. Preparation uses
maximum side 128 and `frame_stride: 1`, retaining sparse teacher frames without
a second stride filter. Samples with too few usable labels can still be rejected.

The research orchestrator prepares a **separate train-only scratch dataset** for
episodes `0,11,12,18,21,23,26,31`. This is necessary because this slice's training
loader uses every dataset row. Tuning episodes 39/42 and holdout episodes 43/44
must not enter that training dataset. Five matched original-CNN variants use
ten epochs, batch size four and seed 31: eager FP32, BF16, channels-last, fused
Adam, and BF16 plus channels-last plus fused Adam. Each starts from random weights
with the same model and remaining training defaults.

For context, the campaign also refits the best prior ANN/CNN/transformer/boosting
recipes, with neural refits at 20 epochs and batch size four, and retains the old
winning transformer as an unchanged incumbent. Successful candidates are assessed
on tuning episodes only. Equal-weight top-two, equal-weight top-three and
inverse-tuning-MAE top-three ensembles are considered when enough candidates
succeed. Candidate definitions, weights and the selected winner are frozen before
generating holdout labels or assessing episodes 43/44. Holdout outcomes are
separate measurements and cannot revise the selection.

## Optional training optimization

Training requests can opt into bfloat16 autocast, channels-last layout, compiled
forward execution, fused Adam and the custom CUDA loss. Defaults retain eager
float32 training and the existing PyTorch loss. The loss arithmetic remains
float32 under autocast, and each batch still produces one optimizer update.
The original model owns saved weights even when a compiled forward wrapper is used.

Compilation needs separate authorization: `pytorch_compile_authorized` and
`cuda_compile_authorized` in `runtime.json`. Training authorization does not
implicitly allow either. A requested CUDA backend fails explicitly if
its compilation prerequisites are unavailable. No compiler or production package
was added. Compiler caches are routed to project-owned directories.
Authorized operations must run from the canonical `Z:/hugging-face/video2data`
checkout (or `/mnt/z/hugging-face/video2data` under WSL); this implementation
checkout cannot be used to create training artifacts elsewhere.

Training records and checkpoint payloads bind optimization settings and source
identity. The source inventory includes the optional C++/CUDA files. The CUDA
implementation did not execute in the primary-environment compiler attempt:
CUDA 13.3 compilation failed on CCCL's rejection of MSVC's traditional
preprocessor. The TorchInductor attempt separately failed with `TritonMissing`.
These primary failures remain preserved. A separate read-only external
PyTorch `2.12.1+cu132` environment subsequently compiled and executed the custom
CUDA loss on one real training sample: loss difference
`1.1920928955078125e-7`, maximum gradient difference `1.862645149230957e-9`,
within `rtol=1e-4, atol=1e-6`. Compilation took 59.7557 seconds; total operation
time was 83.1539 seconds. Warm eager/CUDA forward-plus-backward timings were
0.0048373/0.0043066 seconds. This is one-batch evidence in that recorded
environment, not general correctness, steady-state speedup or primary-runtime
compatibility. The
[external record](../../../outputs/scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json)
retains exact details. See the
[learning design](../../../docs/design/learning-buildout.md) for comparison and
episode-separation rules.

`CompilerExperimentRequest` and `run_compiler_experiments` in
[compiler_experiments.py](compiler_experiments.py) attempt TorchInductor and the
custom CUDA loss independently using existing local tools. Each uses one real
training pair and a smaller CNN (`base_channels: 8`, no dropout), recording an
eager forward/backward comparison before attempting the requested backend.
There is no optimizer update, trained checkpoint or accuracy score. Successful
attempts record loss/output/gradient differences, first-call time and GPU memory;
failed prerequisites or backend/compiler calls remain explicit failed owning
runs. The helper never substitutes eager execution for a failed backend or
installs tools. A single real-batch attempt does not establish general numerical
correctness or steady-state speed. Compiler outcomes do not select the model.

## Input and artifact contracts

Preparation requires completed `depth` run manifests from explicitly chosen public
development episodes. The template's empty `teacher_runs` list requires explicit
producer selection; the research orchestrator supplies its completed teacher run
paths through the public request. The adapter checks recorded video,
calibration and frame-artifact identities during preparation.
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
with strict state loading. Schema 2 binds each checkpoint to its exact owning
run, request, dataset hash, coursework hashes, network source bytes, model
configuration, preprocessing and target contract. Its artifact path and hash
must match both the checkpoint inventory and standard artifact map. Epoch and
optimizer-step counts must agree with the fixed-epoch, non-dropping batch loader
and completed training record. Both saving and prediction reject nonfinite
model tensors. Prediction records the admitted epoch and optimizer-step count.
Changing `_network.py` requires the checkpoint's original implementation to be
restored before loading; a matching architecture name alone is insufficient.
Untrained random weights and older checkpoint schemas are not admitted.

`outputs/scratch_depth/predict/<run-id>/` holds `run.json`, `preprocessing.npz` and
one NPZ per requested frame. Each frame contains `depth_m`, `valid`, `input_valid`
and `input_intrinsics_pixels`. Prediction validity means an in-bounds input ray and
a finite depth within the training label range; it is not a calibrated confidence
or occlusion mask. `run.json` preserves MP4 PTS and checkpoint provenance. Parquet
poses, capture-time clocks and world transforms are not inferred.

Prediction uses the checkpoint's exact preprocessing geometry. It is currently
limited to public development RGB episodes, excluding episode 41. `predict.json`
and the request type default to episode 39. `PredictDepthRequest.frame_stride`
defaults to one and selects `range(start_frame, stop_frame, frame_stride)`;
skipped frames are decoded/grabbed without model inference. The investigation
matches this stride to each sparse teacher schedule, preserving source indices
and MP4 PTS. Prediction processes one selected frame at a time and
does not load the teacher. Partial artifacts remain on failure with a failed run
record; new runs use unique directories. GPU operations reserve the common project
lock and record device-wide sampled memory separately from PyTorch peaks.

The research-owned [scratch adapter](../learning_experiments/scratch_adapter.py)
exposes `ScratchAdapterRequest` and `adapt_scratch_predictions`. It consumes
completed public scratch prediction contracts, with no private cross-feature
imports or model execution, and emits ordinary research prediction artifacts for
the shared scorer and ensembles. It verifies actual scratch checkpoint ownership,
one checkpoint across episodes, a train-only training dataset, matching camera
and target contracts, video/timestamp lineage, calibrated masks and complete
prepared-group coverage. It preserves the original checkpoint identity instead
of fabricating a research checkpoint. The train-only and scoring dataset hashes
can differ; their optical/depth contracts must agree.

## Shared run-record migration

The target run record in
[artifact-contracts.md](../../../docs/design/artifact-contracts.md) is implemented
in `_common.py`: every operation writes `kind`, path/hash `inputs`, inline `sources`
including the source lock, inherited `provenance` and wall-clock `timing`.
Coursework hashes are included in the inline sources and `source-files.json`.
Every completed or failed operation inventories its output files in the standard
`artifacts` map, excluding the owning `run.json`. Legacy fields are retained.
Preparation inherits the most restrictive teacher tag. Training requires a tagged,
completed preparation record, matching producer path and intact dataset artifact,
then embeds its tag in every
checkpoint. Prediction requires an intact checkpoint from a completed, tagged
training run and propagates that lineage alongside its RGB inputs. Missing legacy
tags are refused. Training rechecks each sample's hash when reading it, including
later epochs. Preparation and prediction also retain per-frame status.

The baseline research run enabled preparation only and closed that flag after
completion. The expanded campaign separately authorizes scratch training,
prediction and compiler attempts in the canonical checkout. The
evaluation/validation split extension applies only to MoGe2 depth; this scratch
model remains limited to frozen public development episodes other than 41.
The completed expanded campaign retains successful scratch training/prediction
and failed compiler attempts separately. Runtime authorization remains enabled
under the user's expanded instruction; tests and dependency changes stay excluded.

## Operation sequence — separately gated

The following commands document the individual API. The authorized expanded
research campaign orchestrates preparation, training and prediction through
these public handlers. Commands themselves do not grant additional execution
scope or bypass runtime gates.

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
paired data, logs and checkpoints stay inside the canonical artifact root.
MoGe2 execution authorization remains independent of the scratch-model flags;
the research campaign supplies authorization for its bounded teacher generation.

## Admitted-depth ablations — implemented source, disabled and unrun

`research.py` owns preparation and training requests/handlers;
`_research_network.py` reuses `CourseworkDepthCNN`. The historical network,
checkpoint admission and flags stay unchanged. New `research_prepare_authorized`
`research_train_authorized` and `research_predict_authorized` flags are false. Authorization precedes heavy
imports and writes; canonical Z: checkout/environment rules apply. No checks,
preparation, fitting or inference ran for these changes.

Separate entry point: `python -m track3.features.scratch_depth.research` accepts
`prepare --request <json>`, `train --request <json>` or `predict --request <json>`. Prepare requests require
`sources` plus predeclared `sources_sha256`; train requests require `dataset`
plus `dataset_sha256`. Describing this API does not authorize execution.

Preparation JSON declares pinned `dataset_revision`/`split_id`,
`source_split: public`, `real_captured_only: true`,
`pixel_geometry: undistorted-centre-pinhole-optical-z-metres`,
`frozen_protocol_before_fitting: true`, and explicit `samples`. A sample declares
`episode_index`, `frame_indices`, inventoried actual `pts_seconds`, existing
scratch pinhole `geometry`, `target_source`, `confidence_semantics`, and `arrays`.
Each array maps to `{artifact, key, sha256, producer_run}`. Input fields alone
allow explicit `add_frame_axis: true` to adapt a legacy single-frame archive;
it never creates confidence. Producer receipts bind episode, frames, PTS, grid
and array ownership before any sample is written. Source indices/PTS also bind
to the pinned RGB inventory, and legacy depth arrays bind actual intrinsics.

Required arrays: `rgb` (THW3 uint8), `base_depth_m`/`base_valid`/`confidence`
(THW), `target_depth_m`/`target_valid` (HW), `intrinsics` (3×3). T is one or three.
Optional `normal_support` is HW; `temporal_depth_m`/`temporal_support` are 2HW.
All arrays must already share the admitted pinhole grid. No resizing, motion
warping or correspondence estimation occurs in this adapter.

Known scratch-pair producers bind their hash-owned dataset sample; known
depth/stereo producers bind completed frame records. Other producers require
`learning_inputs[artifact-relative-path]` receipts with hash, public split,
revision/split ID, episode, exact frame/PTS arrays, geometry and `fields` mapping
logical names to archive keys. Receipts also bind every declared confidence,
target-source and optional support/motion semantic. Arbitrary tagged files and
self-declared source rows are insufficient. Missing evidence fails closed.

Matched direct/residual arms receive RGB, standardized admitted log-z, validity
and explicit producer confidence: six channels/frame. Residual correction is
bounded ±0.5 log metres; direct prediction uses the positive softplus head.
Both arms retain admitted centre-depth output support. Frozen input log-depth
mean/std is fit using training pixels only, RGB divides by fixed 255. Fitting
admits only 0/11/12/18/21/23/26/31; 39/42 are stored as tuning but never loaded
by the trainer. Episodes 41/43/44 are rejected, including mislabeled producer
references. Synthetic data and diverse teachers are excluded from this adapter.

Context requires consecutive indices, actual PTS gaps in (0,0.1] seconds, and
producer-bound `context_alignment: visibility-masked-motion-warp-to-centre-optical-grid`.
Robust log Huber transition is fixed 0.1. Boundary, normal and temporal additions
are mutually exclusive, fixed weight 0.1, against robust-log alone. Boundaries
use valid depth neighbours. Normals use calibrated backprojection, nondegenerate
neighbourhoods and `normal_boundaries_excluded: true`. Temporal sources require
`temporal_depth_frame: centre-camera-optical-z-metres` and
`temporal_visibility_and_motion_admitted: true`. This compares predictions to
frozen motion-transformed neighbour depths; it does not establish consistency
between student predictions. Differentiable neighbour-prediction reprojection
and a learned temporal encoder remain deferred pending admitted motion evidence.

Per-arm bounds: seed 31, ≤10 epochs, side ≤128, ≤128 real training centres,
batch-four gradient accumulation, Adam lr 0.001/weight decay 0.00001, 30 minutes,
12 GiB peak reserved device memory plus existing headroom policy, 4 GiB artifacts.
Checks are cooperative between examples, not watchdogs; one kernel may overrun
a ceiling before returning. There is no tuning loader, early stopping, best
epoch selection or fallback. An outer frozen campaign must enforce the proposed
eight-arm/four-hour aggregate; this individual-arm handler does not grant it.
Failed records remain explicit. Checkpoints bind sources/dataset/preprocessing,
use a separate schema/architecture and declare `production_admitted: false`.
Legacy inference refuses these checkpoints. `PredictLearningRequest` and
`predict_learning` provide a separately disabled research-only tuning path.
The request templates `research-prepare.json`, `research-train.json` and
`research-predict.json` contain explicit placeholder paths/digests and enable
nothing. Prediction requires exact dataset/checkpoint hashes and one selected
tuning episode (39 or 42), plus a completed same-episode `depth.generate`
calibration run. It emits every declared prepared sample of that episode, with
no frame selection override, target arrays, target support, holdout or early
stopping. Output validity uses centre base-depth/ray support and finite positive
predictions only. Source confidence remains an explicit input diagnostic.

Safe loading uses `weights_only=True`, CPU placement, ≤128-MiB checkpoint
size, ≤128 dense finite float32 state entries and ≤10-million parameters, plus
exact architecture keys/shapes. Completed owning run, source files, request,
dataset, preprocessing, model configuration, counters and checkpoint artifact
hashes must agree. Source changes refuse the checkpoint. Each prediction run
copies byte-identical calibrated `preprocessing.npz`, checks all input ray grids
and K, and writes completed per-frame depth/validity/K/PTS artifacts plus nested
episode inventory. `stereo_depth.report` can admit this research candidate kind;
no resulting evidence or production promotion is claimed. Limits remain
cooperative 30 minutes/12-GiB device memory/4-GiB output.

Optional `trace_runtime: false` records no synchronized spans by default.
Future opt-in measurement captures first 32 clip examples (first cold, next two
warmup, rest warm), GPU-synchronized host-wall phases and whole-arm elapsed time
in owned `runtime_trace.json`. Decode denotes NPZ reading; video decode happened
upstream. Nested spans overlap. No compilation or full-pipeline steady-state
speedup is claimed. `runtime_measurement.grouped` consumes this as whole-arm
accounting, preserving train/tuning groups and missing memory measurements.

Exact blockers: historical archives expose no producer confidence. The updated
dense stereo source now emits continuous `rgb_match_confidence`, explicitly
`uncalibrated_matching_quality_not_probability` from
`linear_left_right_disparity_residual`, transferred with the same visible
stereo point as base depth. The adapter admits only that named producer field
and semantics, never validity. No new matching archives were generated. Raw stereo targets
include reserved support and are refused until producer-owned exclusion evidence
exists. No admitted warped-context, boundary-support or temporal-motion archive
exists. These gaps are not filled with invented inputs. Canonical Z: output is
unavailable in this session. Source implementation establishes no empirical gain.
