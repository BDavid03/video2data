# Learning and optimization buildout

Implementation scope was added on 11 October 2026 with execution disabled.
On 18 October both the baseline tuning run and the expanded investigation
completed in the canonical `Z:/hugging-face/video2data` checkout. The baseline
had 12 completed trials, zero failed trials, 200 teacher frames and 40 prepared
pairs. The expanded campaign recorded 35 completed experiment outcomes and two
failed compiler attempts in 1470.21 seconds. Runtime authorization remains
enabled under the user's expanded instruction. Tests, smoke checks, dependency
changes, notebooks, simulation and submission uploads remain excluded; bounded
compiler attempts using existing tooling were separately authorized.

The baseline transformer scored 0.1581543745 tuning log-depth MAE against MoGe2
pseudo-targets without holdout assessment. On the expanded sparse-clip dataset,
the frozen winner was the equal-weight original scratch-CNN eager/BF16 ensemble:
0.1239556042 tuning MAE and 0.1521616317 holdout MAE. The
[feature results](../../track3/features/learning_experiments/README.md#expanded-investigation--completed)
retain per-candidate outcomes and canonical artifact links. Neither campaign
establishes measured depth accuracy, full reconstruction or submission readiness.

## Completed expanded investigation

`InvestigateRequest` / `investigate_models` in
[investigate.py](../../track3/features/learning_experiments/investigate.py) owns
the expanded comparison. Sparse teacher sampling selected 16 frames across each
of the eight training and two tuning clips, with prepared maximum side 128 and
stride one. All 160 pairs were accepted. Scratch training used a separately
prepared train-only dataset of 128 pairs; its loader fits every supplied row.
Tuning used 32 pairs from episodes 39/42.

The best prior ANN/CNN/transformer/boosting recipes were refitted, with neural
refits using 20 epochs, batch size four and seed 31. The prior transformer
checkpoint remained an unchanged incumbent. Five original coursework-CNN
variants used ten epochs, batch size four and seed 31: eager, BF16,
channels-last, fused Adam, and BF16/channels-last/fused. All nine new training
operations completed. Ten individual candidates and three geometric ensembles
each completed tuning and holdout assessment.

Equal top-two, equal top-three and inverse-tuning-MAE top-three ensembles were
assessed on tuning. The
[frozen selection](../../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/frozen-selection.json)
fixed members, weights and winner before holdout teacher labels were generated.
The subsequent 32 holdout pairs from episodes 43/44 were accepted without
rejection. Top-three equal scored 0.1494962110 on holdout, but the selected
top-two ensemble remains fixed; holdout scores never revise the selection.
The incumbent transformer scored 0.1958149432 tuning and 0.2190753796 holdout
on these expanded frames. Its prior baseline score used different frame coverage
and is not directly interchangeable.

Both primary-environment compiler attempts failed: TorchInductor reported
`TritonMissing`; CUDA 13.3 compilation failed when CCCL rejected MSVC's
traditional preprocessor. No compiled parity or speedup is claimed from those
attempts. A separate read-only external PyTorch `2.12.1+cu132` environment
subsequently compiled and executed the custom loss on one real batch: loss
difference `1.1920928955078125e-7`, maximum gradient difference
`1.862645149230957e-9`, within recorded `rtol=1e-4, atol=1e-6`. Compilation took
59.7557 seconds (83.1539 seconds total); warm eager/CUDA forward-plus-backward
timings were 0.0048373/0.0043066 seconds. These single-sample timings establish
no general speedup or compatibility of the primary environment. The
[external record](../../outputs/scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json)
retains that scope without rewriting primary failures.

After contiguous 3×3 rotation/intrinsic and 3×1 translation arrays resolved an
initial OpenCV `gemm` error, the episode-39 opening rerun completed rectification,
stereo, anchor and camera stages. Its clustering guard rejected an above-table
cluster touching the image border (`opening_not_quiet`), leaving hull and TSDF
blocked. The [opening result](../../outputs/opening_experiments/20261018T120039.470932Z-e0b15dc0/result.json)
records four completed, one failed and two blocked stages. No complete mesh or
full-episode trajectory is established.

The separate [synthetic-depth pilot](../../track3/features/synthetic_depth/README.md)
generated 128 training-only samples with analytic optical-axis depth and
procedural instance IDs in 3.330 seconds. They remain outside the completed
MoGe2 fitting data; no synthetic training benefit or real-scene fidelity is
claimed. The completed primary
[result](../../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/result.json)
distinguishes 35 successful experiment outcomes from two compiler failures;
those counts do not mean 35 separately trained models.

The completed [research report](../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/report.md),
[comparison figure](../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/comparison.png)
and [candidate CSV](../../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/candidates.csv)
combine the recorded primary, compiler, opening and synthetic evidence without
loading model weights or selecting again on holdout. Generated-artifact links
resolve on Z:; they are not copied into the implementation checkout.

## Completed baseline run

The feature-owned `ExecuteRequest` / `execute_tuning` API in
[execute.py](../../track3/features/learning_experiments/execute.py) is available as
`python -m track3.features.learning_experiments execute`; `--request` optionally
selects a bounded request, otherwise `execute.json` supplies the selected defaults.
It persists a plan, generates MoGe2 targets sequentially, prepares aligned pairs
and invokes tuning. Its owning run records stage progress and child hashes;
`result.json` is written only after successful completion.

The budget covers the eight default training episodes plus tuning episodes 39/42.
Each supplies 20 teacher frames (0–19), at maximum side 768. Stride-five preparation
at maximum side 128 selects frames 0/5/10/15, producing at most 40 pairs before
target-validity rejection. The default seed-31 search schedules 12 trials: three
each for ANN, CNN, transformer and gradient boosting. At side 128, patch size eight
fits the 256-token transformer bound. Episode 41 and final holdout 43/44 remain
excluded in that baseline run.

Execution requires the approved existing Windows interpreter and canonical
artifact root, plus each participating slice's operation flags. No dependency
installation is part of the workflow. The baseline enabled scratch preparation
only; the subsequent expanded authorization covered scratch training/inference,
ensembles, holdout and compiler attempts. Canonical sources stayed unchanged while each run
pinned their hashes. Historical records retain the execution-time identities.

## Feature ownership

The existing [scratch-depth slice](../../track3/features/scratch_depth/README.md)
owns optimization of its original CNN training loop and the optional custom
CUDA log-depth loss. The
[learning-experiments slice](../../track3/features/learning_experiments/README.md)
owns bounded trial planning and experimental models and ensembles. The official
policy runtime and read-only FlashCHORD sources are not modified by this work.

Experimental models have feature-owned training, checkpoint, prediction and
tuning handlers. They do not inherit scratch-depth
checkpoint admission, generate production depth artifacts, or become submission
candidates merely because a model factory exists. A future integration must
define production artifact consumers before any candidate can enter the
reconstruction chain. Research checkpoint ownership and source/data identity are
implemented in the experimental slice; production promotion remains separate.

## PyTorch optimization

Keep full precision, eager execution, ordinary memory layout and the existing
Adam path as defaults. Make bfloat16 autocast, channels-last layout, compiled
forward execution and fused Adam explicit choices. Preserve the original model
for checkpoint persistence when a compiled wrapper is used. Compute log-depth
loss in float32, retain finite-gradient clipping, and keep one optimizer update
per batch so existing training-counter admission remains meaningful.

Neither reduced precision nor compilation establishes a speedup. In a
controlled comparison, hold the data, architecture, seed, batch order and
step budget constant and record optimizer options and source identity. Measure
wall time, sampled device memory, PyTorch memory peaks and numerical differences
under that separately approved scope. Compilation needs its own authorization
and project-owned compiler caches. No new dependencies were installed for this campaign.

Both scratch loss backends refuse shape broadcasting. Training sample admission
checks bounds before and after conversion to float32, with float64 comparisons;
epoch supervision counts use the same usable-label mask as the loss. Research
neural loss also requires identical prediction, target and mask shapes.

## Custom CUDA boundary

The optional kernel implements masked mean absolute log-depth error and its
prediction gradient. Its target mask is boolean validity intersected with finite,
positive targets. The denominator is the actual supervised-pixel count. Pixels
outside the mask contribute neither error nor gradient. The subgradient at equal
prediction and target is zero. Empty supervision and invalid predictions at
supervised pixels must fail rather than produce an apparently usable loss.

Use contiguous float32 CUDA tensors and an explicit conversion at the caller.
The adapter owns device/shape/dtype admission, autograd integration and lazy
extension loading. The CUDA implementation owns stream-correct launches and
reductions. Training authorization alone does not permit building this extension:
`execution.cuda_compile_authorized` is a separate flag. There is no silent
fallback from a requested CUDA backend. The primary-environment compilation
failure supplies no kernel-execution or backward-parity evidence. The separately
completed external attempt supplies one-batch agreement only in its own recorded
environment and tolerance; broader numerical coverage remains unestablished.

## Model comparisons and ensembles

Use a per-pixel ANN as a limited RGB baseline, a spatial CNN for local context,
and a bounded patch transformer for image context. Each begins from random
weights. Owned neural training uses AdamW and optional BF16 forward autocast;
the original coursework CNN retains its own Adam recipe. Patch/token bounds matter
because attention memory grows with token count; a model name is not a GPU-memory
estimate. Gradient boosting provides a
small residual-regression baseline with explicit features. Trained weights and
pseudo-label comparison results now exist for both completed research campaigns.

Combine comparable positive depth predictions in log metres with nonnegative
weights, excluding invalid member pixels and renormalizing available weight per
pixel. No available prediction means invalid output. Shared teacher targets can
give models correlated errors; member disagreement is not calibrated confidence.
Weights must be selected using development-only evidence and frozen before a
final held-out comparison. Never optimize weights against evaluation labels.
The separately gated saved-output ensemble operation admits matching completed
prediction runs, retains parent/source identities, combines their arrays in
float64 and emits the same score-compatible artifact protocol. Historical
authorization/configuration identities remain in producer records; changing an
authorization flag does not require regenerating already owned predictions.
Saved-output ensemble and scoring admission reconstruct calibrated RGB ray support
and require it to match each prediction's boolean input mask. Prediction validity
must remain within that mask. File identity sets detect aliases without pairwise
artifact comparisons.

## Hyperparameter planning

A trial plan enumerates bounded configurations; it does not launch training,
score a model, or select a winner. Record search method, seed, trial limit,
parameter domains, stable trial identities, episode allocations and the intended
metric. Grid/random planning requires no optimization-service dependency.
The separately gated tuning handler runs owned neural/boosting child operations
sequentially, scores complete tuning episodes and persists a source-bound research
selection. It records failed trials and does not select when all trials fail.
Before training it requires complete training/tuning episode coverage and unique
frame identities. Plan, dataset and producer hashes remain pinned across trials;
completed child artifacts enter lineage immediately, including before a later
stage failure. Admitted scores must match the requested support threshold and
complete tuning frame count.

Partition by complete episode rather than by neighboring video frames. Restrict
candidate episodes to the frozen public development roster and exclude episode
41. Keep tuning episodes disjoint from training episodes; the untouched final
holdout remains separate from both. The completed expanded campaign assessed
that holdout after selection was frozen. Both referenced
coursework notebooks are present as of 2026-10-11 and remain unexecuted.

Scoring rejects incomplete group coverage and insufficient predicted support;
it computes a pixel mean within each episode and weights episode means equally.
This metric compares against MoGe2 pseudo-targets. The final holdout requires a
separate authorization flag and is never read for tuning selection.

For a future authorized experiment, use the same finite resource budget for
each candidate and name failed/incomplete trials. Select a configuration by a
declared development metric such as masked log-depth MAE, then evaluate the frozen
selection once on the untouched holdout. Training loss alone cannot establish
generalization. Any future optimizer, compiler, boosting library or transformer
dependency beyond the approved environment needs a concrete dependency proposal.
