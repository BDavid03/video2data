# Systems evaluation and the project research agenda

This chapter connects the literature to the decisions the project still needs to make. It separates numerical implementation, independent geometric assessment, achieved behavior, and research selection. Equations marked as derivations are analytical explanations, not newly measured project results. Research cutoff: **18 October 2026**.

## GPU systems are part of the scientific method

### Mixed precision changes numerical computation

Micikevicius et al. describe training with lower-precision arithmetic, higher-precision master parameters and loss scaling to manage limited numeric range. Their evidence concerns the studied models and hardware; it does not establish that every lower-precision implementation preserves a particular optimization trajectory. [Mixed Precision Training, 2017 preprint and ICLR 2018](https://arxiv.org/abs/1710.03740)

**Derived numerical interpretation.** A stored floating-point number approximates a real value. For a normal, rounded operation, the relative error is often modeled as \(\operatorname{fl}(x)=x(1+\delta)\), with \(|\delta|\) bounded by a unit-roundoff term. This model excludes overflow, underflow and operations near singularities. BF16 allocates fewer significand bits than FP32; whether its exponent range prevents overflow says little about the accuracy of a small residual obtained by subtracting two nearby depths.

Several operations in this project deserve higher-precision accumulation:

- Log-depth error can lose a small difference between nearby values.
- A valid-pixel denominator changes with masks; accumulation order affects the numerator.
- Normal estimation normalizes a cross product that becomes ill conditioned for nearly collinear samples.
- Variance calculated as \(E[x^2]-E[x]^2\) can suffer cancellation.
- Scale/shift least squares can be ill conditioned when input depths vary little.
- Quaternion and tangent-vector normalization become sensitive near invalid zero inputs.

A prospective mixed-precision design can use low precision for compatible convolutions/matrix products while retaining FP32 loss and geometric reductions. This does not imply parity with FP32 training: even small gradient changes can alter subsequent updates. The completed eager/BF16 campaign records are empirical outcomes for their exact runs, not a universal precision equivalence claim. [Project optimization history](../../design/learning-buildout.md)

### What compilation actually optimizes

The PyTorch 2 paper describes TorchDynamo extracting graph fragments through Python bytecode transformation and TorchInductor compiling them, using Triton for GPU paths and C++ for CPU paths. Graph capture and kernel lowering are separate from the neural architecture. [Ansel et al., PyTorch 2, ASPLOS 2024](https://docs.pytorch.org/assets/pytorch2-2.pdf)

Triton expresses tiled computations so the compiler can map data and work onto GPU memory/execution hierarchies. FlashAttention reorganizes exact attention with tiling and online normalization to reduce high-bandwidth-memory traffic rather than materializing the complete attention matrix. [Tillet, Kung and Cox, 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), [Dao et al., 2022](https://arxiv.org/abs/2205.14135)

**Derived project implication.** A CNN has no attention operation for FlashAttention to accelerate. A tiny patch transformer may already have a small interaction matrix, leaving launch or projection cost dominant. A Rust implementation can reduce some host overhead while executing slower convolution kernels. An eager GPU path may outperform a compiled path if compilation cannot amortize or if a graph contains unsupported/dynamic operations. Architecture labels and implementation languages are not performance measurements.

For an exact semantic comparison, freeze architecture, weights or initialization, preprocessing, precision, batch order, update count and loss reduction. If a backend changes bilinear interpolation, channel dropout, optimizer decay or valid-mask handling, it is an algorithmic change as well as a systems change. See the [Rust chapter](07-rust-candle-burn.md).

### Roofline and whole pipeline cost

The Roofline model relates attainable throughput to compute capacity, bandwidth and arithmetic intensity. It distinguishes a compute ceiling from data-movement limits. [Williams, Waterman and Patterson, 2009](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S09/handouts/papers/RooflineVyNoYellow.pdf)

**Derived bounds.** For arithmetic work \(F\), transferred bytes \(Q\), effective peak throughput \(P\), and effective bandwidth \(B\),

\[
T\gtrsim\max(F/P,Q/B),
\qquad \text{attainable throughput}\lesssim\min(P,BF/Q).
\]

These are explanatory bounds. Logs, reductions, scheduling, occupancy and synchronization complicate a simple FLOP count. Kernel fusion may save intermediate reads/writes without accelerating the underlying convolutions.

For a whole operation, use

\[
T_{\rm total}=T_{\rm decode}+T_{\rm preprocess}+T_{\rm transfer}
+T_{\rm compile}+T_{\rm model}+T_{\rm geometry}+T_{\rm persist}.
\]

If an optimizable component occupies fraction \(p\) of runtime and is accelerated by \(s\), the derived total speedup is

\[
S=\frac1{(1-p)+p/s}.
\]

A large isolated loss-kernel speedup has little end-to-end effect if that kernel is a small fraction of training. Conversely, host synchronization after every scalar metric can dominate a small GPU workload even if the model kernels are efficient.

Compilation amortizes only when the steady-state saving is positive. With compilation cost \(C\), eager step time \(t_e\) and compiled step time \(t_c<t_e\), the break-even iteration count is

\[
N>C/(t_e-t_c).
\]

Cold startup, warm-cache startup, steady-state inference, forward/backward and full optimizer steps answer different questions. A report must specify which one it measures. A CPU timer around asynchronous launches can measure submission overhead rather than completed GPU work; completion boundaries belong in any future measurement protocol.

### Memory accounting

**Derived parameter-state estimate.** FP32 parameters, gradients and Adam first/second moments require roughly \(16P\) bytes for \(P\) scalar parameters. This excludes activations, saved backward tensors, temporary buffers, allocator fragmentation, convolution workspaces, extra master copies and non-framework device memory. The scratch CNN's source-derived parameter count therefore cannot be turned directly into peak VRAM.

Activation storage scales with feature-map area and channel count. Skip connections retain high-resolution activations until the decoder needs them. Training and inference have different retention requirements; batching, checkpointing and memory format alter the balance.

For a video model, cached per-frame tokens and attention buffers can dominate weights. For FlashSAC, replay can dominate network storage. The project runtime analysis gives a transition-storage expression,

\[
\text{bytes/transition}=2D_ob_o+
4(D_a+D_{\rm objective}+2D_{\rm critic\ context})+2,
\]

before multiplying by capacity and adding simulator/model memory. Its dimensions are recipe dependent. [Policy training memory analysis](../../../track3/features/policy_training/README.md)

The single GPU motivates staged execution and bounded windows. It does not establish a particular backend's memory fit. Existing external environments remain read-only, and generated builds/caches belong on Z: under the workspace agreement.

### What the existing compiler evidence supports

The historical primary TorchInductor and CUDA attempts failed. A separate existing external environment later compiled the custom loss and recorded one-real-batch loss/gradient agreement within its stated tolerances. Its local timing describes that batch and environment only. It does not establish full-training equivalence, general speedup, numerical coverage or compatibility of the primary environment. [Recorded compiler outcomes](../../../track3/features/learning_experiments/README.md#expanded-investigation--completed)

For the masked log-L1 loss, the analytical gradient from the learning chapter is a useful future implementation contract. A CUDA implementation also needs correct stream use, reduction support counts, gradient accumulation, dtype/shape admission and explicit failure on unusable supervision. Parallel atomics and reduction order can change floating-point results while preserving the real-number formula.

## What should be measured

### Teacher agreement has a precise meaning

For admitted pixels \(V_e\) in episode \(e\), the research scorer calculates

\[
E_e=\frac1{|V_e|}\sum_{i\in V_e}
|\log\hat Z_i-\log Z_i^T|,
\qquad E_{\rm macro}=\frac1{|\mathcal E|}\sum_eE_e.
\]

It pools pixel errors within an episode and weights episode MAEs equally, retaining coverage. The relevant source mask and prediction admission also affect support. [Research scoring source](../../../track3/features/learning_experiments/score.py)

**Derived interpretation.** Under identical support and weighting, the triangle inequality gives

\[
E(\hat z,z^*)\le E(\hat z,z^T)+E(z^T,z^*),
\]

\[
E(\hat z,z^*)\ge
\max\{0,E(z^T,z^*)-E(\hat z,z^T)\}.
\]

Without independent information about teacher error \(E(z^T,z^*)\), a small student–teacher distance does not bound physical error tightly. It remains possible for a smoother student to improve on some teacher noise; that possibility requires independent assessment rather than inference from teacher agreement.

### Depth metrics and their invariances

The following definitions are mathematical reporting choices, not claims that all are implemented or that they are official challenge metrics:

\[
\operatorname{AbsRel}=|V|^{-1}\sum_{i\in V}\frac{|\hat Z_i-Z_i|}{Z_i},
\]

\[
\operatorname{RMSE}=\sqrt{|V|^{-1}\sum_{i\in V}(\hat Z_i-Z_i)^2},
\quad \operatorname{RMSE}_{\log}=\sqrt{|V|^{-1}\sum_{i\in V}(\log\hat Z_i-\log Z_i)^2},
\]

\[
\delta_k=|V|^{-1}\sum_{i\in V}
\mathbf1\left[\max(\hat Z_i/Z_i,Z_i/\hat Z_i)<1.25^k\right].
\]

RMSE uses metres and emphasizes large absolute errors. Log errors emphasize ratios. AbsRel weights an absolute error differently at near and far depth. A threshold accuracy can hide errors just beyond its threshold. Publish support and distance ranges beside values.

An assessment that rescales every frame independently answers a relative-shape question and can hide metric/temporal drift. Report any aligned supplementary metric separately from raw metric depth. Use a fixed, declared alignment support and do not allow reference labels to correct production predictions.

### Independent geometric evidence

Different diagnostics expose different failure modes:

| Diagnostic | Residual | Conditions needed for interpretation |
|---|---|---|
| Stereo consistency | Left/right reprojection and disparity disagreement | Correct stream pairing, rectification and visibility |
| Unused-view reprojection | Predicted 3D projected into an independently held view | An unused view, correct calibration and observed correspondences |
| Static landmark stability | World-coordinate scatter through time | Actual static membership and admitted camera pose |
| Object rigidity | Canonicalized point/track residual | Correct identity and material correspondence |
| Temporal geometry | Reprojected neighbor depth versus center depth | Motion compensation and occlusion handling |
| Boundary geometry | Depth discontinuity location and surface side | Clean image/depth boundary correspondence |
| Contact consistency | Object-frame contact drift and penetration | Admitted contact regime and usable surface geometry |
| Scale behavior | Anchor residual, support distribution and drift | Independent metric measurements with uncertainty |

Agreement between two stages that share the same teacher is not fully independent. For example, a camera fit optimized against teacher depth and a surface evaluated against that same depth can jointly preserve a scale bias. Retain measurement lineage when describing independence.

### Coverage and failure are part of the outcome

Report expected rows, available references, valid predictions, rejected measurements, interpolated/completed rows and unsupported rows separately. A pose error on the retained subset must accompany

\[
\operatorname{coverage}=
\frac{\text{valid matched predictions}}{\text{eligible reference measurements}}.
\]

No eligible reference gives an undefined error with zero support. Eligible references with no predictions give zero coverage, not a perfect error. A completion method may improve row completeness while lowering the fraction directly observed; preserve both quantities.

Difficult-event strata should be based on independently defined events such as grasp, lift, release, border truncation, occlusion and reacquisition. A public mocap-validity flag is not necessarily ego-image visibility. Stratum definitions chosen after looking at errors can become another selection mechanism.

## The pinned trajectory and mesh metrics

The authoritative facts here are read from the repository's pinned Submission Kit and its [contract interpretation](../../../reports/contract.md). They are not fresh claims about any later rules change. The local source is [object_pose.py](../../../upstream/v2d_submission_kit/v2d_submission_kit/v2dlb/object_pose.py) and [object_mesh.py](../../../upstream/v2d_submission_kit/v2d_submission_kit/v2dlb/object_mesh.py).

### One initial rigid alignment

Let predicted and reference world object poses be \(\hat O_{kt}\) and \(O^*_{kt}\). The scorer uses object slot zero at frame zero:

\[
A=O^*_{0,0}(\hat O_{0,0})^{-1},\qquad
\hat O'_{kt}=A\hat O_{kt}.
\]

One rigid transform applies to every object and frame. It fits no scale. It removes world gauge but does not repair wrong body frames, inter-object layout, timing or drift. Per-object or per-frame alignment would define a different problem.

### ADD and AUC

For fixed canonical points \(x_j\),

\[
\operatorname{ADD}_{kt}=J^{-1}\sum_j
\|\hat R'_{kt}x_j+\hat t'_{kt}-(R^*_{kt}x_j+t^*_{kt})\|.
\]

The pinned scorer does not match symmetric equivalents. For thresholds \(\tau_j=j/100\) metres, \(j=1,\ldots,9\), let \(a_j\) be the share of supported trajectory frames below \(\tau_j\). It applies trapezoidal integration on nine equally spaced points over a normalized zero-to-one abscissa:

\[
\operatorname{AUC}=\frac18\left(\tfrac12a_1+
\sum_{j=2}^{8}a_j+\tfrac12a_9\right),
\]

then averages bodies and ultimately episodes according to the pinned protocol. This is not a continuous integral over arbitrary user-chosen thresholds.

### Position and rotation success gates

SP-SR requires frame/object-averaged position error at most \(0.1\) m and orientation error at most \(0.5\) rad. MP-SR requires every object to satisfy frame-averaged position error strictly below \(0.03\) m and orientation error strictly below \(30^\circ\); if an object contains multiple bodies, the pinned implementation averages those bodies within that object before applying the gates. The distinction between averaging objects and requiring each object matters in multi-object tasks. A good cup trajectory cannot automatically compensate for a poor second object under the latter gate.

### Relative position error

For a two-object episode,

\[
e_{\rm RPE,t}=100\left\|
\hat R_{0t}^\top(\hat t_{1t}-\hat t_{0t})-
(R^*_{0t})^\top(t^*_{1t}-t^*_{0t})
\right\|,
\]

in centimetres. Object-zero orientation affects this quantity; a plain world-coordinate separation vector would be different. Single-object episodes receive zero by definition in the pinned implementation. That convention affects track averages and does not demonstrate perfect geometric reconstruction.

**Derived sensitivity.** With relative separation \(r=t_1-t_0\), a small object-zero orientation error contributes approximately \(-[\delta\theta]_\times R_0^\top r\). Its magnitude can grow with separation even when both object translations look reasonable. Slot-zero orientation is therefore an important multi-object error source.

### Mean per point error

The pinned MPPE attaches six points \(v_j\) at radius \(0.05\) m on the positive/negative object axes:

\[
\operatorname{MPPE}=100\operatorname{mean}_{t,k,j}
\|\hat t_{kt}+\hat R_{kt}v_j-(t^*_{kt}+R^*_{kt}v_j)\|.
\]

This couples position and orientation through a specified lever arm. It is neither raw orientation error nor the training reward formed from squared/shaped point errors. With pure rotation, each point displacement is bounded by \(2r\sin(\theta/2)\); points on the rotation axis move less.

### Shape scoring has a different alignment

CD-O normalizes mesh scale, registers rotation/translation with multiple starts and compares sampled surfaces using symmetric mean distance in centimetres. A generic unsquared surface-distance form is

\[
\tfrac12\left[
|P|^{-1}\sum_{p\in P}\min_{q\in Q}\|p-q\|+
|Q|^{-1}\sum_{q\in Q}\min_{p\in P}\|q-p\|
\right],
\]

with the pinned implementation owning its exact normalization and registration. Distinguish this from squared Chamfer variants with different units.

Because this scorer fits scale, good CD-O cannot establish correct metric collision geometry or metric object trajectories. Conversely, a watertight collision hull can be mechanically convenient while filling concavities that worsen shape distance. Maintain separate geometric purposes and artifacts.

### Achieved behavior and reconstruction reference

The policy's self-score compares achieved motion against the participant's reconstruction. It can improve while the reconstruction remains biased. The confidential external reference is a separate comparison. Copying the reconstructed reference into a trajectory file would not establish achieved behavior even if its numbers resemble the target.

The pinned episode weighting, complete roster and 20 Hz correspondence matter. Capture PTS, nominal dataset time, reference playback time and control-step indices are distinct. Resampling must preserve their mapping; a constant frame count does not prove synchronization.

## Experimental reasoning with a consumed holdout

### Grouping must survive every derivative artifact

All frames, crops, cameras, teacher outputs, augmentations and derived tracks from one episode inherit its role. Training-only normalization and calibration fitting must not read future assessment groups. A manifest containing several roles is not necessarily leakage if loaders select the admitted subset correctly, but every consuming stage must preserve that separation.

The expanded scratch preparation deliberately contains only training episodes because its training loader fits every supplied row. That is an implementation-specific boundary worth preserving. [Scratch and research dataset contracts](../../../track3/features/learning_experiments/README.md)

### Selection is part of fitting

Cawley and Talbot analyze overfitting of finite-sample model-selection criteria. A larger search can exploit chance features of a small tuning set. [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

**Derived project reasoning.** Choosing architectures, loss weights, synthetic proportions, confidence thresholds, checkpoint epochs, ensembles or frameworks using an assessment set makes those outcomes part of selection. Renaming the same episodes does not restore independence. The completed `43/44` holdout can remain historical evidence for its frozen recipe, but later recipes need an independently fixed assessment protocol and genuinely independent assessment evidence for an unbiased final claim.

Episode `41` remains excluded. This survey does not nominate it as replacement holdout. A future protocol may use independently sourced data or another explicitly fixed eligible design; the choice must precede outcomes and respect task/data constraints.

### Nested development and small sample limits

A prospective grouped development procedure could use inner episode folds among training episodes for early recipe screening, then fixed tuning episodes `39/42` for limited selection. That evaluates robustness of the development process; it does not create new final data. Hyperparameters learned from inner folds must not use their outer fold labels indirectly through global preprocessing.

Whole-episode resampling reflects grouping more honestly than resampling millions of pixels. With only two tuning episodes, uncertainty intervals are limited by support regardless of pixel count. Report both episode values and failures. Multi-seed repeats quantify optimizer variation conditional on the same scenes; they do not quantify unseen-scene diversity.

### Compare matched support and honest budgets

For candidates A and B, report their native support and a matched-support comparison. Native support shows practical coverage; matched support helps isolate error differences on comparable observations. Neither should replace the other.

Record attempted and failed configurations, seeds, search-space size, training exposures, total optimizer updates, teacher cost and full operation runtime. If one family gets more trials or richer pretrained features, state it. Avoid interpreting the selected result as a controlled architecture comparison when data coverage or budgets differ.

The original short-prefix baseline and later sparse-full-clip campaign used different prepared frames. Their scores are not directly interchangeable. A frozen top-two ensemble remains the historical winner even if another candidate happened to score differently on the consumed holdout. [Campaign history](../../../track3/features/learning_experiments/README.md)

## A literature informed research agenda

The following are prospective questions. They prioritize evidence that separates plausible explanations of current limitations. They do not change current producers, selected recipes, thresholds or execution authorization.

### First establish a coherent geometric explanation

**Calibration and clocks.** Can every resized/rectified RGB, mask and depth product be tied to the exact camera rays and physical timestamp mapping used by the consumer? A useful result would isolate grid/timing error before attributing failures to a depth network. Owners: [depth](../../../track3/features/depth/README.md), [stereo depth](../../../track3/features/stereo_depth/README.md), [camera motion](../../../track3/features/camera_motion/README.md).

**Independent metric support.** Which surfaces and depths have reliable correspondence, and how does support vary spatially? Record ambiguity, boundary and occlusion structure rather than summarizing stereo by one valid fraction. Compare measurement-supported geometry with learned completion on real tuning videos.

**Camera versus object motion.** Do static surfaces remain stable in the world while manipulated objects move? Residuals should distinguish poor parallax from moving-foreground contamination. A reduced camera residual is persuasive only when the explanatory support is genuinely static.

**Opening failure.** The episode-39 opening path stopped when an above-table cluster touched the image border. The next research question is whether an observable, sufficiently complete initialization exists under the declared criteria, or whether persistent segmentation/alternative view support is needed. Raising the guard threshold to pass the same data would not explain the failure. Owners: [opening masks](../../../track3/features/object_masks/README.md), [opening experiments](../../../track3/features/opening_experiments/README.md).

### Make depth improvements attributable

**Metric versus shape error.** Does a candidate improve relative structure, scalar calibration, boundaries or temporal behavior? A source-separated objective and metrics can show which component changes. Compare the historical scratch architecture with one change at a time before constructing a broad new model.

**Residual correction.** Can bounded correction to an admitted base preserve trusted geometry while improving independently supported residuals? The correction bound should be motivated by the error range it can repair. Record the fraction saturating the bound.

**Synthetic enrichment.** Do analytic samples improve performance on real tuning episodes at matched training budget, or merely reduce synthetic loss? Distinguish fixed total updates from fixed real-example exposure. Keep exact provenance and retain the real-only recipe as a control.

**Temporal context.** Does motion-compensated context add information beyond an equally sized single-frame model? Compare visibility support and motion regimes, not only an aggregate teacher score. A model using future frames is an offline-video model and should be described accordingly.

Owners: [scratch depth](../../../track3/features/scratch_depth/README.md), [learning experiments](../../../track3/features/learning_experiments/README.md), [synthetic depth](../../../track3/features/synthetic_depth/README.md).

### Preserve interaction meaning across the robot boundary

**Canonical objects.** Are tracking meshes, reconstructed meshes and simulator bodies related by explicit transforms? The spoon convention needs a deliberate frame choice. Shape quality and physical scale should remain separate questions.

**Anatomical hands.** Does metric lifting correct wrist placement while preserving anatomical scale and side? Report unresolved/occluded intervals rather than forcing a plausible hand through every frame. Asset prerequisites must be satisfied before genuine hand inference is claimed.

**Stable contacts.** Can canonical object-side contact aggregation reject transient noisy observations without incorrectly enforcing no-slip during sliding? Compare contact timing, normal consistency, penetration and object-frame drift as different residuals.

**Retargeting reachability.** Which reference errors exceed the bounded residual controller's correction region? Address those in reconstruction or retargeting rather than expecting RL to undo an unreachable reference.

Owners: [object pose](../../../track3/features/object_pose/README.md), [object mesh](../../../track3/features/object_mesh/README.md), [hand motion](../../../track3/features/hand_motion/README.md), [scene assembly](../../../track3/features/scene_assembly/README.md), [retargeting](../../../track3/features/retargeting/README.md).

### Establish achieved behavior before scaling policy training

**Assistance-free realizability.** Can a complete, correctly timed reference lead to actual object motion from frame zero without virtual object assistance? Local reset coverage and high assisted reward do not answer this.

**Objective attribution.** Does contact-aware reward improve achieved motion over a matched motion-only objective with scene contacts and observations held fixed? The existing source protocol handles top-level and curriculum contact weights separately; zeroing contact arrays would confound the comparison. [Prospective policy comparison](../../../track3/features/policy_training/research-protocol.md)

**Resource feasibility.** What world count, replay capacity and update schedule fit the actual device while preserving declared curriculum semantics? Resource choices should be explicit and bounded; they are not inferred from a paper's datacenter environment counts.

Owners: [policy training](../../../track3/features/policy_training/README.md), [policy rollout](../../../track3/features/policy_rollout/README.md), [campaign](../../../track3/features/campaign/README.md).

### Explore Rust around an exact semantic boundary

**Inference transcription.** Can a version-pinned Candle or Burn model represent the existing FP32 forward computation with identical preprocessing, skip ordering, resize coordinates and head conversion? Exported weights need a name/layout manifest and an owning source identity.

**Differentiable resize.** If Rust training is pursued, implement or compose the missing bilinear derivative without changing the mathematical decoder. Replacing it with nearest interpolation belongs to a separate architecture comparison.

**Systems value.** Is there a deployment, host-overhead, portability or custom-kernel benefit large enough to justify a second implementation? A focused Rust executable can be useful even without a training speed advantage; that benefit should be named explicitly.

**Promotion boundary.** Research outputs remain research outputs until consumers, provenance and exact artifact contracts admit them. A new Rust slice would own its request, handler, backend adapters, persistence, configuration and documentation. Stable tensor/provenance contracts can be shared without adding horizontal service layers.

## Decision records worth preserving

A future candidate report should answer the following in connected evidence, not just a best-score table:

| Question | Record |
|---|---|
| What hypothesis changed? | One explicit model, data, loss, geometry or systems difference |
| What stayed matched? | Frames, source roles, teacher, camera, precision, update/exposure budget |
| What was selected and when? | Candidate set, criterion, tuning evidence and immutable freeze |
| What evidence is independent? | Measurement lineage and labels not consumed by the candidate |
| Where did it fail? | Missing support, exceptions, truncations and difficult events |
| What does it cost? | Complete operation time, compilation, memory and artifact storage |
| What can the result claim? | Teacher fidelity, geometric accuracy, realized simulation or deployment behavior |

The strongest next contribution is not necessarily a larger network. It is an attributable improvement in a physically useful quantity, tied to the exact source, data, coordinate and execution conditions that produced it.

## Primary systems references

- Micikevicius et al. **Mixed Precision Training.** 2017 preprint; ICLR 2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).
- Ansel et al. **PyTorch 2 Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation.** ASPLOS 2024. [Primary paper](https://docs.pytorch.org/assets/pytorch2-2.pdf).
- Tillet, Kung and Cox. **Triton An Intermediate Language and Compiler for Tiled Neural Network Computations.** MAPL 2019. [Primary paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf).
- Dao et al. **FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness.** 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- Williams, Waterman and Patterson. **Roofline An Insightful Visual Performance Model for Multicore Architectures.** 2009. [Author manuscript](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S09/handouts/papers/RooflineVyNoYellow.pdf).
- Cawley and Talbot. **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.** JMLR 2010. [Primary paper](https://www.jmlr.org/papers/v11/cawley10a.html).
