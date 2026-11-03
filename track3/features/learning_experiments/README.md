# Learning experiments

The subsequent ordered [research backlog](../../../docs/experiments/README.md) reuses
these completed runs. Its source-only changes add per-pixel `member_count`,
`effective_member_count` and `log_depth_std` to **future** persisted ensemble
artifacts. They do not change weights or depth aggregation. Disagreement is
uncalibrated, is NaN with fewer than two supported positive-weight direct parents,
and counts parents rather than independent models. Existing artifacts are
unchanged. No new fit, export, score or check ran; [item11](../../../docs/experiments/11-selection.md)
specifies prospective coverage/selection rules without consumed-holdout reuse.

Status on 18 October 2026: the expanded research campaign **completed** in the
canonical `Z:/hugging-face/video2data` checkout: 35 completed experiment outcomes
and two failed compiler attempts, in 1470.21 seconds. The frozen tuning-selected
winner is the equal-weight original scratch-CNN eager/BF16 ensemble. The earlier
bounded run completed with 12 completed trials, zero failed
trials, 200 teacher frames and 40 prepared pairs; those results remain the
baseline, not results of the expanded campaign.

These are research operations, **not submission candidates**. The expanded scope
includes original scratch-CNN training/inference and bounded compiler experiments
using existing tooling. Tests, smoke checks, dependency changes, notebooks,
simulation and submission uploads remain outside this run. No new dependencies
have been installed for this campaign.

## Recorded baseline result

The selected transformer achieved tuning log-depth MAE **0.1581543745** against
MoGe2 pseudo-targets, averaged equally over tuning episodes 39 and 42. Final
holdout 43/44 was not evaluated. The run used only four early frames per episode;
these scores do not establish measured depth accuracy or full-clip performance.

| Technique | Completed trials | Best tuning log-depth MAE |
|---|---|---|
| ANN | 3 | 0.4028088411 |
| CNN | 3 | 0.2804249727 |
| Transformer | 3 | 0.1581543745 |
| Gradient boosting | 3 | 0.3841079499 |

The selected model has width 32, two layers, four heads and patch size 16. It
trained for 20 epochs with batch size one, learning rate 0.001 and no weight
decay. Total workflow time was 841.99 seconds; tuning took 659.03 seconds.

The canonical checkout retains the
[workflow result](../../../outputs/learning_experiments/execute/20261018T105355.351623Z-37d0aa68/result.json),
[selection](../../../outputs/learning_experiments/tune/20261018T105658.305032Z-7407936a/selection.json)
and [all trial records](../../../outputs/learning_experiments/tune/20261018T105658.305032Z-7407936a/run.json).
These artifact links resolve on Z:; generated files were not copied into the
implementation checkout. The selection identifies the checkpoint and source,
dataset, prediction and scoring hashes.

## Expanded investigation — completed

The frozen `top2_equal` ensemble scored **0.1239556042** tuning log-depth MAE
on episodes 39/42 and **0.1521616317** on holdout episodes 43/44. It averages
`scratch_eager` and `scratch_bf16` equally in log depth. All 13 candidates
completed tuning and holdout assessment; nine new training operations completed
and the original transformer incumbent was reused. These counts describe
experiment outcomes, not 35 separately trained models.

| Candidate | Tuning MAE | Holdout MAE |
|---|---|---|
| Refit ANN | 0.4021551989 | 0.4526321222 |
| Refit CNN | 0.2460226340 | 0.2594607760 |
| Refit transformer | 0.1890610631 | 0.1999931842 |
| Incumbent transformer | 0.1958149432 | 0.2190753796 |
| Refit gradient boosting | 0.3873144056 | 0.4339543552 |
| Scratch eager | 0.1321539216 | 0.1702597208 |
| Scratch BF16 | 0.1385482141 | 0.1536269931 |
| Scratch channels-last | 0.1483359807 | 0.1856198889 |
| Scratch fused Adam | 0.1462275573 | 0.1577466423 |
| Scratch BF16/channels-last/fused | 0.2023547478 | 0.2175236167 |
| **Frozen top-two equal** | **0.1239556042** | **0.1521616317** |
| Top-three equal | 0.1266161557 | 0.1494962110 |
| Top-three inverse MAE | 0.1261815198 | 0.1495593719 |

Top-three equal happened to score lower on holdout; it **does not replace the
frozen winner**. The incumbent's expanded tuning score differs from its earlier
0.1581543745 baseline because it is assessed on different frames. All scores
compare to MoGe2 pseudo-labels, not measured ground truth. One seed and two
holdout episodes do not establish broad generalization or complete reconstruction.

Teacher generation produced 192 sparse frames across 12 episodes. Preparation
accepted all of them: 128 training pairs, 32 tuning pairs and, after selection
was frozen, 32 holdout pairs at maximum side 128. The separate scratch dataset
contains only the 128 training pairs. The holdout preparation manifest contains
all 192 pairs; the holdout scorer selects only episodes 43/44.

Both primary-environment compiler attempts failed honestly: TorchInductor
reported `TritonMissing`, and CUDA 13.3 compilation stopped because MSVC's
traditional preprocessor was rejected by CCCL. These attempts establish no
compiled-backend parity or speedup. The separate external-toolchain CUDA attempt
subsequently completed, with the original primary failures preserved. Under
read-only PyTorch `2.12.1+cu132`, compilation took 59.7557 seconds (83.1539 seconds
total). One real batch produced a loss difference of `1.1920928955078125e-7` and
maximum gradient difference of `1.862645149230957e-9`, within recorded
`rtol=1e-4, atol=1e-6`. Warm eager/CUDA forward-plus-backward times were
0.0048373/0.0043066 seconds for that single sample; no general speedup or
primary-environment compatibility is established. Its
[owning record](../../../outputs/scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json)
retains the exact environment, tolerances, differences and timings.

The episode-39 opening rerun completed rectification, stereo, anchor and camera
stages after explicit contiguous OpenCV rotation/translation/intrinsic shapes
resolved the initial `gemm` failure. Clustering then failed with
`opening_not_quiet: an above-table cluster touches the image border`; hull and
TSDF remained blocked. The
[opening result](../../../outputs/opening_experiments/20261018T120039.470932Z-e0b15dc0/result.json)
records four completed, one failed and two blocked stages, not completed object
reconstruction. Separately, the [synthetic pilot](../synthetic_depth/README.md)
generated 128 training-only analytic-depth/instance-ID samples in 3.330 seconds;
they were not mixed into these measured fits and establish no learning benefit.
Runtime authorization remains enabled under
the user's expanded instruction; no tests or dependency installations ran.

The canonical checkout retains the
[completed result](../../../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/result.json),
[frozen selection](../../../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/frozen-selection.json)
and [owning run](../../../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/run.json).
These artifact links resolve on Z:; generated files are retained there.

The completed metadata-only renderer additionally produced the
[research report](../../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/report.md),
[tuning/holdout comparison figure](../../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/comparison.png),
[complete candidate CSV](../../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/candidates.csv)
and [owning report record](../../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/run.json).
`ReportRequest` / `write_report` in [report.py](report.py) admit the completed
investigation and its owned frozen/result hashes, plus optional opening,
external-CUDA and separate synthetic-dataset evidence. The renderer loads no
model or checkpoint weights and never reselects the frozen winner from holdout.

[investigate.py](investigate.py) owns `InvestigateRequest` and
`investigate_models`; [investigate.json](investigate.json) supplies its request.
The CLI is:

```text
python -m track3.features.learning_experiments investigate --request track3/features/learning_experiments/investigate.json
```

The request names the completed baseline tuning run, with defaults of at most
16 teacher samples per episode, prepared maximum side 128, 20 research-neural
epochs, ten scratch epochs, batch size four, seed 31 and compiler experiments
enabled. Each of the eight training and two tuning episodes is sampled across
its clip at frame indices `0, stride, 2*stride, ...`, where
`stride = ceil(decoded_frame_count / 16)`. This produces at most 16 teacher
frames per episode; preparation uses `frame_stride: 1` to retain every sparse
teacher frame that passes target-validity admission. Actual sample counts may
be lower. This replaces the baseline's four early prepared frames per episode.

The campaign takes the best completed baseline recipe for each of ANN, CNN,
transformer and gradient boosting. It refits the four recipes on the expanded
training samples; neural refits use 20 epochs, batch size four and seed 31.
Architecture, learning rate, weight decay and boosting parameters come from the
selected prior recipe. The old winning transformer checkpoint remains a separate
frozen incumbent and is assessed on the expanded tuning samples without refitting.

Five matched variants train the original coursework scratch CNN for ten epochs
with batch size four and seed 31: eager FP32, BF16, channels-last, fused Adam,
and BF16 plus channels-last plus fused Adam. Scratch preparation creates a
separate dataset containing only training episodes
`0,11,12,18,21,23,26,31`; the scratch loader fits every row in its dataset and
must never receive the research dataset containing tuning samples.

The separately gated compiler helper attempts TorchInductor and custom CUDA loss
independently on one real training pair, with a smaller scratch CNN and no
optimizer update or checkpoint. It records eager/candidate loss, output and
gradient differences, first-call time and memory when execution succeeds.
Unavailable prerequisites or compiler/backend errors remain explicit failed
experiments; no eager fallback is reported as compiled execution. These bounded
attempts do not establish general correctness or a steady-state speedup, and
their outcomes do not choose the model.

Successful individual candidates are ranked on tuning episodes 39/42. The
campaign additionally assesses equal-weight top-two, equal-weight top-three and
inverse-tuning-MAE top-three geometric ensembles when enough candidates succeed.
It writes `frozen-selection.json`, fixing candidate checkpoints, ensemble members,
weights and the tuning-selected winner **before** generating holdout teacher
labels or assessing episodes 43/44. Public holdout clock metadata may be read
earlier to plan sampling. Holdout outcomes are reported separately and never
change the frozen selection. Episode 41 remains excluded throughout.

`outputs/learning_experiments/investigate/<run-id>/run.json` retains stage,
candidate and child-run identities plus explicit failures. A completed campaign
writes `result.json`; running or failed child attempts are not completed model
results. The completed records above retain the actual successful assessments
and both compiler failures separately.

## Baseline execution workflow

[execute.py](execute.py) owns `ExecuteRequest` and `execute_tuning`; the CLI is
`python -m track3.features.learning_experiments execute --request track3/features/learning_experiments/execute.json`.
The default request first persists a plan, generates MoGe2 targets sequentially,
prepares pairs and calls the existing tuning handler. Progress and child hashes
are retained in `outputs/learning_experiments/execute/<run-id>/run.json`; only a
completed workflow writes `result.json` linking its plan, dataset and selection.

The selected budget is the default seeded 12-trial search: three trials each for
ANN, CNN, transformer and gradient boosting, using seed 31. Teacher generation
uses frames 0–19 from each of the eight training episodes and tuning episodes
39/42, at maximum side 768. Preparation uses maximum side 128 and stride five,
selecting frames 0/5/10/15 before target-validity rejection: at most 40 pairs.
This geometry keeps even patch-size-eight transformer grids within 256 tokens.
Episode 41 and final holdout episodes 43/44 are excluded throughout.

`ExecuteRequest` bounds teacher frames to 1–20, teacher maximum side to 64–768,
prepared maximum side to 64–128 and stride to 1–teacher_frames. The workflow
requires the 12-trial, four-technique plan and derives teacher episodes from its
training/tuning groups; the selected default partition remains unchanged.
`execute`, `plan`, `model`, `train`, `predict`, `score`, `tune`, `boost_train` and
`boost_predict` need explicit authorization, along with depth generation and
scratch-depth preparation in their owning slices. The baseline excluded holdout
prediction, scratch training/inference, ensembles and compiler execution. The
expanded investigation has separate authorization for those operations in the
canonical checkout.

Use the approved `.runtime/moge-python313/Scripts/python.exe` in the canonical
checkout. `scripts/use-scratch-depth.ps1` selects that environment and confines
caches; teacher generation owns its source loading. The shared GPU lease admits
physical GPU 0 only and requires 6144 MiB free with the selected configuration.
Plan/source and dataset/producer hashes are pinned; do not change the canonical
sources during execution. Historical run records retain the exact source and
authorization identities used before the execution gates were closed.

## Bounded planning and ownership

[planning.py](planning.py) owns `PlanRequest`, `plan_trials` (in-memory metadata)
and `write_plan` (the persisted operation `plan`). [plan.json](plan.json) supplies
the default request; [search.json](search.json) owns the finite search spaces.
The `plan` operation never calls a model factory, boosting fit, prediction,
a notebook or any evaluator. Both planning APIs and request-file loading
refuse while `plan_authorized` is false. There is no CLI authorization bypass.

Grid and seeded random planning cover each selected technique in round-robin
order. Random planning shuffles each technique's finite candidate pool with an
independent seeded standard-library generator, then selects without replacement.
Technique order is canonical; request ordering cannot change trial identities.
Budgets smaller than the number of selected techniques are refused. Limits are
128 trials, four techniques, 10 dimensions, 16 choices per dimension, and 4096
combinations per technique (at most 16384 total). Exhausted spaces may produce
fewer trials than the budget. There is no adaptive search. The separate `tune`
operation can execute this fixed plan only after its own and each child operation's
authorization is explicitly enabled.

Neural spaces include AdamW weight decay values `0`, `0.00001` and `0.0001`
alongside architecture, learning rate, epochs and batch size. The planner admits
neural learning rates only up to `0.1`, matching the training request. Tuning
passes weight decay to the optimizer request rather than to the model factory;
plans that omit it retain the training default of `0.00001`.

Each trial records bounded parameters, a deterministic SHA-256-derived ID and
initialization seed, and null score/checkpoint fields. IDs bind the method, seed,
parameters and episode partition; unique persisted run IDs additionally contain
time and randomness. Identical requests/configuration give identical trial
metadata, not identical run timestamps. Numerical training reproducibility across
library/device versions is not established by these seeds.

## Three-way episode separation

The policy is pinned to the existing frozen public development split and dataset
revision. Episode 41 is always excluded, as are the official validation and
evaluation namespaces. Every group must be nonempty, unique and disjoint:

| Group | Default episodes | Role |
|---|---|---|
| Training | 0, 11, 12, 18, 21, 23, 26, 31 | Fit parameters, preprocessing statistics and residuals |
| Tuning | 39, 42 | Hyperparameter and ensemble-weight decisions only |
| Final holdout | 43, 44 | Assessment after selection is frozen; never selection |

Grouping includes every frame, camera stream, object and derived artifact from
an episode. Do not randomly split pixels or frames across groups. Any future
normalization, sampling, target preparation or teacher-derived artifact adapter
must preserve this partition and lineage. The APIs check supplied episode IDs;
the low-level factories cannot establish that caller-supplied RGB/targets actually
came from those episodes. Owned operations additionally use the hash-bound public
prepared-data adapter. Teacher inference/preparation stays in its existing slices.

The selection objective is **tuning-only log-depth MAE**, minimized as an
equal-weight mean of per-episode valid-pixel means of
`abs(ln(predicted_metres) - ln(target_metres))`. Targets must be finite positive
optical-axis depths on the same geometry and absolute scale. No scale alignment
is performed by planning. The separate `score` operation implements that metric.
Final holdout must never determine hyperparameters, ensemble weights, stopping
decisions or a best model. No selection occurs during planning. `tune` selects on
tuning episodes only. Reusing final holdout outcomes to revise choices would
invalidate this separation. Episode IDs alone do not prove
independent sessions or absence of near duplicates.

## Neural models and owned training

[models.py](models.py) contains `ModelRequest` and `create_model`. Importing this
module or the package does **not** import PyTorch. Factory calls require both
`model_authorized` and their `train_authorized` or `predict_authorized` purpose
before heavy imports or construction. Each forward pass rechecks the operation
and model flags, RGB bounds and supplied batch episode IDs. Prediction requires
`eval()`; a training forward requires training mode. No pretrained weights,
downloads, model-zoo dependencies or checkpoint loading are used.

| Technique | Original randomly initialized implementation |
|---|---|
| ANN | Per-pixel RGB MLP using 1×1 convolutions and ReLU; no spatial context |
| CNN | Compact spatial 3×3 convolution/ReLU stack and 1×1 depth head |
| Transformer | Strided patch embedding, learned position vectors, bounded encoder and transposed-convolution patch decoder |

Inputs are finite floating RGB `N3HW` in `[0,1]`, batch size at most 8 and at most
65536 pixels per image. Transformer padding must fit at most 256 tokens;
depth is cropped back to the original pixel grid. Outputs are positive finite
`N1HW` optical-axis depth via softplus plus 0.001. Widths, layers, heads and patch
sizes have explicit bounds in [_config.py](_config.py). Model initialization
preserves the caller's CPU random stream. These untrained outputs have no measured
metric-scale or accuracy guarantee.

The factories themselves do not persist a checkpoint. [train.py](train.py) now
owns an explicit neural training request, training-only data loader, random
initialization, AdamW optimization, finite-gradient clipping, per-epoch weights
and run persistence. Defaults are ten epochs, batch size two, learning rate
0.001, weight decay 0.00001 and seed 31. Optional native BF16 autocast applies
only to the forward pass; log-depth loss stays float32. There is one optimizer
update per non-dropping batch. No training operation reads tuning or final-holdout
targets for fitting, early stopping or checkpoint selection. Every training run
starts from random weights; optimizer state and resume are not implemented.

[data.py](data.py) consumes the public `scratch_depth.prepare` artifact contract
through shared path/provenance primitives, without private cross-feature imports.
It checks the completed producer, dataset/sample hashes, frozen split, actual
camera calibration, exact pinhole geometry, unique episode/frame identities and
boolean target masks. Each requested group's episodes must all have samples.
Each array read rechecks its manifest, producer and sample bytes. Use separately
authorized preparation with `max_image_side: 256` or smaller; the scratch-depth
default of 384 is deliberately not silently resized by this feature.

Depth-bound comparisons promote stored values to float64 before comparing the
declared scalar bounds. Prepared targets are checked again after float32
conversion. Neural and boosting prediction, ensemble member admission and scoring
use the same comparison precision, so float32 rounding cannot widen the declared
depth interval. Unsupported values remain masked rather than clipped into range.

[checkpoints.py](checkpoints.py) owns atomic per-epoch saves and weights-only
loading. Admission binds the owning completed training run, actual dataset bytes,
architecture, episode partition, preprocessing, target bounds, seed/optimizer
request, source identity and exact epoch/optimizer counters. State tensors must
be finite and the original unwrapped model's weights. Only a strictly loaded
admitted checkpoint can be used by [predict.py](predict.py).

Neural admission also requires a unique dataset entry in the training input
inventory, unique episode/frame rows covering every declared training episode,
and an ordered checkpoint for every completed epoch. Neural and boosting
admission both require the prepared-pairs schema, positive ordered depth bounds
and canonical pinhole intrinsics; boosting also binds the pseudo-depth target
and calibration digest.

Prediction selects one declared episode group and writes per-frame NPZ depth,
validity, calibrated input-ray support and intrinsics with exact episode/frame,
MP4 PTS, input and checkpoint lineage. Validity means an in-bounds calibrated RGB
ray and finite depth inside training bounds; it is not the teacher-label mask or
confidence. A preview may use `max_samples`; scoring refuses incomplete group
coverage. `holdout_predict_authorized` is a separate flag required before
any owned final-holdout prediction or scoring. Neural GPU operations hold the
shared cooperative `GpuLease` and record memory telemetry.

These research checkpoints cannot enter the existing scratch-depth or submission pipeline.
Authorization flags are operational guards, not a security boundary around an
arbitrary caller with access to raw PyTorch modules/tensors.

## Original scratch-CNN prediction adapter

[scratch_adapter.py](scratch_adapter.py) owns `ScratchAdapterRequest` and
`adapt_scratch_predictions`. Supply a prepared `dataset`, completed scratch
`prediction_runs`, a `group` and the declared training/tuning/holdout episode
lists. The handler requires research model/prediction authorization and the
separate holdout gate for that group before imports or artifact creation.

The adapter reads the public scratch run/artifact contracts through shared path
and provenance primitives. It admits canonical completed training and prediction
producers, actual owned checkpoint hashes, one common checkpoint across episodes,
matching video hashes/timestamps and the exact calibrated optical/depth contract.
It checks the scratch training dataset covers every declared training episode
and contains no tuning or holdout samples. That train-only dataset intentionally
has a different hash from the research scoring dataset; their data contracts
must agree. Complete prediction coverage of the requested prepared group is
required. Sparse scratch prediction uses `PredictDepthRequest.frame_stride`
matching each episode's teacher schedule, retaining actual source frame indices.

The handler copies admitted arrays into a standard
`learning_experiments.predict` run with technique `coursework_scratch_cnn` and
explicit original scratch training/checkpoint/prediction identities. It executes
no model and fabricates no research checkpoint. Existing scoring and saved-array
ensembles consume this output using the same geometry, coverage and lineage
rules as the research families. Rejected artifacts remain failures; the adapter
does not resize, fill missing frames or replace prediction masks with teacher
supervision.

## Standard-library gradient boosting

[boosting.py](boosting.py) provides `BoostRequest`, `fit` and `predict`, plus the
gated `BoostModel.predict` method. `fit` checks `train_authorized` before processing
inputs; either prediction entrypoint checks `predict_authorized` first. No heavy
imports, external estimators, feature libraries or dependencies are needed.

The feature definition is exactly one `[red, green, blue]` row per selected valid
pixel, normalized to `[0,1]`; there are no position or spatial-context features.
Targets supplied to `fit` are already **natural logarithms of positive
optical-axis depth in metres**, restricted to depths in `[0.001,1000]`. Invalid
RGB/targets must be removed by a future caller; fit rejects invalid rows. A supplied
episode ID is required for every row, and fitting accepts only declared training
episodes. Tuning and final holdout targets cannot be passed to fit under their
actual episode IDs.

Fit starts with the training target mean and adds deterministic decision stumps
to the current log-depth residuals, with shrinkage by learning rate. Each stump
minimizes training squared residual error; this surrogate is different from the
future tuning log-depth MAE. Stable feature/cut order breaks ties. Constant
features/targets fall back to a constant stump. Training uses at most 2048 rows,
three RGB features, 128 estimators and 32 candidate cut thresholds per feature.
Sorted training features and residual prefix sums bound work; no exhaustive
unbounded threshold search is used. There is no tuning-based early stopping.

Prediction requires declared permitted episode IDs and the same feature/target
contract. It sums the intercept and shrunk stump outputs in log space, then
exponentiates to finite positive depths in metres, one scalar per input row.
It rejects overflow/underflow rather than silently returning invalid depths.
The low-level model is an in-memory research record. The separate
[boosting operations](boosting_operations.py) own future `boost-train` and
`boost-predict` requests, bounded training-row sampling, atomic JSON checkpoint
persistence and admission, full-grid prediction in bounded row chunks and NPZ
outputs. Checkpoint/source/data/partition ownership is checked before prediction.
No automatic fit, prediction or model persistence is wired to `plan`.

## Scoring and bounded tuning

[score.py](score.py) admits an exact completed prediction artifact and the same
prepared dataset. It requires complete episode/frame coverage for the requested
group and rejects incomplete previews. Loss is accumulated over the intersection
of finite positive target supervision and prediction support. Per-episode
pixel means are then averaged equally across episodes; supported-pixel coverage
and unsupported pixels are recorded. `minimum_prediction_fraction` defaults to
0.9 and prevents a candidate from receiving an apparently favorable score from
a small supported subset. These are comparisons against teacher pseudo-labels,
not measured depth accuracy or official evaluation.

Scoring requires the saved boolean `input_valid` mask to equal the calibrated
RGB ray support reconstructed from the dataset geometry. Declared prediction
validity must stay within that support. The saved-output ensemble enforces the
same rule for every member. File alias admission uses device/inode sets rather
than comparing each artifact with every earlier file.

[tune.py](tune.py) takes a completed owned plan and prepared dataset. It checks
the plan's exact finite search contract and source identity, then executes trials
sequentially. Neural and boosting child operations retain their separate run
records, checkpoints and GPU ownership. Prediction and scoring use **tuning
episodes only**, with every tuning sample included. A failed trial is recorded
with its stage, error and failed child-run identity when one exists;
`continue_on_failure` chooses whether to try the rest.
If no trial yields an admitted score, tuning fails without a selection.

Before starting trials, tuning requires unique in-partition sample identities
and coverage of every training and tuning episode. It pins the plan, dataset
and their producer records throughout the comparison. Successful child outputs
enter the input inventory immediately, retaining checkpoint and prediction
lineage even if a later stage fails. Score admission binds the requested
coverage threshold, full tuning frame count and unused final holdout.

After all trials, `selection.json` names the minimum equal-episode tuning MAE,
checkpoint, prediction/scoring runs and their hashes. Equal scores break by
lexicographic trial ID. It records completed/failed trial coverage and null final
holdout score. This selection is a frozen research candidate, not a submission
promotion. No refitting, adaptive search, implicit ensemble-weight fitting or
final-holdout assessment is performed by tuning.

## Masked geometric ensemble

[ensemble.py](ensemble.py) provides `ensemble_depth(predictions, valid_masks,
weights)` for 1–8 already aligned `N1HW` members. It requires ensemble, prediction
and model authorization before importing PyTorch. Each weight is finite, nonnegative and
bounded; at least one is positive. Masks must be boolean and match depth shape
and device. Member validity intersects the mask with finite positive depth.
Invalid depths are replaced before logarithms. Float64 accumulation computes
`exp(sum(w_i * ln(depth_i)) / sum(w_i))`, renormalizing weights over valid members
at each pixel. Unsupported or unrepresentable output pixels receive NaN and a
false output mask. Validity is support, not confidence.

The caller must establish matching camera geometry, optical-axis units, absolute
scale and pixel correspondence. This function does not register images, infer
geometry, invoke models or calibrate confidence. Weights must be fixed independently
of final holdout outcomes; no weight tuning is implemented here.

[ensemble_operations.py](ensemble_operations.py) adds the owned `ensemble`
operation for previously persisted predictions. It admits 1–8 completed parent
runs with matching dataset, partition, group, complete frame identities, clocks,
sample hashes, pixel geometry and calibrated ray support. NumPy combines stored
positive depths in float64 without launching any model or GPU work. The output
uses the same frame protocol as individual predictions and can be scored by the
same complete-group scorer. Parent run/frame hashes, weights and support are
recorded. Missing member pixels are excluded and remaining weights normalized
per pixel. Final-holdout parents require separate holdout authorization.

## Project paths and plan provenance

All source, request, runtime and split reads use shared `project_path` confinement.
There is no arbitrary output-path parameter. Persisted plans go only to
`outputs/learning_experiments/plan/<run-id>/plan.json` with an owning `run.json`.
Every write passes through `output_path` and an additional canonical-root check:
the checkout must resolve to **`Z:/hugging-face/video2data`**, or
`/mnt/z/hugging-face/video2data` under WSL. A checkout elsewhere may be edited and
read but cannot write this slice's generated artifacts or initialize its neural
factories. Configuring another artifact root is refused. External input/source
trees are never modified.

Authorized heavy execution confines temporary
directories, Python/PyTorch/CUDA/extension/Hugging Face caches under that root and
sets `tempfile.tempdir` to the confined `TMPDIR` even when Python has cached an
earlier location. Neural/array operations require the approved existing
`.runtime/moge-python313` interpreter. No environments are created, altered or installed.

Plan run records include resolved request configuration, runtime/search/split
input hashes, source hashes, inherited standard provenance, artifact hashes,
timestamps, timing and completion/failure status. Their `video_only` provenance
tag reflects metadata planning with no label/model consumption; it is not a
provenance admission for a future trained model. Model/training/prediction and
test/validation/evaluation execution fields remain false; scores and best-model
fields are null. Planning itself never creates model results.

Individual API/CLI references (the selected workflow orchestrates these operations):

```text
python -m track3.features.learning_experiments plan --request track3/features/learning_experiments/plan.json
python -m track3.features.learning_experiments train --request track3/features/learning_experiments/train.json
python -m track3.features.learning_experiments predict --request track3/features/learning_experiments/predict.json
python -m track3.features.learning_experiments boost-train --request track3/features/learning_experiments/boost-train.json
python -m track3.features.learning_experiments boost-predict --request track3/features/learning_experiments/boost-predict.json
python -m track3.features.learning_experiments ensemble --request track3/features/learning_experiments/ensemble.json
python -m track3.features.learning_experiments score --request track3/features/learning_experiments/score.json
python -m track3.features.learning_experiments tune --request track3/features/learning_experiments/tune.json
python -m track3.features.learning_experiments investigate --request track3/features/learning_experiments/investigate.json
```

This CLI requires operation authorization and a permitted checkout.
Planning authorization does not authorize model creation, fit or prediction.
Tuning requires `tune`, `plan`, `model`, `train`, `predict` and `score`
authorization; plans containing boosting additionally require `boost_train` and
`boost_predict`. These flags authorize separate future operations only when the
user has approved that scope. The selected run enables only its required scope.
Persisted ensembles require their separate `ensemble`, `model` and `predict`
flags; combining stored files does not authorize new model inference.
All requests, handlers, model integrations, bounds, persistence and documentation
stay together in this feature folder; only stable path/provenance primitives are shared.

### Independent geometry selection (source only, disabled)

`selection.py` owns `geometry-selection` and `geometry-freeze`. Both new runtime
flags are false. The null default request identities are intentional: there is no
admitted protocol, receipt or new assessment. These operations load stored JSON
and hash declared input/checkpoint bytes; they never instantiate or fit models.
They do not change the completed campaign winner or existing ensemble outputs.

The comparison consumes explicit hash-bound `independent-geometry-selection-v1`
receipts. A receipt is an adapter contract, not a substitute for independent
measurements. In particular, a producer without event labels, temporal/rigidity
channels or identity-loss evidence cannot satisfy the promotion gates. Source
report bytes are hash-bound without opening historical outcome records as JSON.
Every parsed query must be episode 39 or 42; 41/43/44 are refused. The existing
43/44 outcomes remain consumed, and every new freeze is ineligible for holdout.

Each request identity is exactly `{"path": "...", "sha256": "..."}`:

- `geometry-selection.json`: `protocol`, ordered `candidates` of
  `{definition, receipt}` identity pairs (maximum eight), and `incumbent` ID.
- Protocol JSON: schema, `partition: tuning`, `episode_indices: [39,42]`,
  `decision_rule: item11-geometry-90pct-5pct-v1`, `predeclared: true`,
  `reference_labels_used: false`, `scale_fitting: false`, hash identities
  `data_identity`, `source_identity`, `event_index_identity`, ordered exact
  `candidate_identities`, `event_strata` including static/dynamic, and full
  reserved `queries` (maximum 10,000). Query identity consists of
  `episode_index`, string `object_id`, `event_id`, `event_stratum`, `query_id`.
  Unsupported queries must remain present; empty declared strata block promotion.
- Candidate definition: `candidate_id`; identities for `data_identity`,
  `source_identity`, `configuration_identity`, `geometry_identity`,
  `anchor_identity`, `preprocessing_identity`; `members` with unique `member_id`,
  nonnegative `weight`, and exact `checkpoint` identity; explicit `thresholds`;
  `frozen_before_receipt: true`; `consumed_holdout_used_for_selection: false`.
  These declarations require upstream provenance review; the reader cannot infer
  chronology from a declaration or hash alone.
- Receipt JSON: schema, partition/episode indices as above; exact
  `protocol_identity`, `candidate_identity`, `source_report` identity;
  `reference_labels_used: false`, `scale_fitting: false`, `metric_scale: metres`,
  `clocks_aligned: true`, `reserved_stereo_support: true`;
  measured `complete_operation_seconds`; one row per protocol query.
- Each row: query identity; `observed` bool; signed `depth_log_residual`;
  **absolute** `depth_error_m`; `reprojection_error_px`, `temporal_residual_m`,
  `rigidity_drift_m` (nonnegative or null); `catastrophic_identity_loss` and
  `new_invalid_event` (bool or null); `member_log_residuals` with every direct
  member ID mapped to signed residual or null; `member_count` counting positive
  weighted supported direct parents; `log_depth_std` nonnegative or null (null
  below two supported parents). Missing members are explicit nulls, not zeros.
  Geometry producer signed metre residuals must become absolute in this adapter.

Comparison emits absolute support/denominators per episode, object, event and
stratum, fixed disagreement threshold error/coverage curves, and paired direct
member residual diagnostics per group. Oracle-minimum residual is only a
complementarity diagnostic and cannot be used as a prediction. Curves require two
supported direct members and never claim calibrated probabilities. Promotion
requires every item11 gate on paired support; unavailable evidence retains the
incumbent. Different paired query sets across passing challengers block ranking.
Guardrails use p90 for reprojection, temporal residual and rigidity in each episode;
this conservative fixed summary is declared before receipts. A comparison input
JSON is bounded at 64 MiB; query IDs are limited to 256 characters, diagnostic
rows to 250,000, and processing to 20 minutes before writes. These conservative
source bounds stop expansion before the 2 GiB artifact ceiling.
Outputs use the existing canonical Z: operation persistence and exact source
identities. No runtime output has been produced here.

`geometry-freeze.json` requires the exact completed comparison artifact identity.
Its separate handler binds the selected full definition, checkpoints, weights,
thresholds, geometry/anchor/data/event/config/source identities and comparison
receipt inventory, rechecks bytes, and writes an explicitly research-only freeze.
No fresh unbiased final protocol is designated, and this freeze cannot promote
production or authorize a holdout read.
