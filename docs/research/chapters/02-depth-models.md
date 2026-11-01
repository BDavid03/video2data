# Monocular depth and geometry models: architectures, supervision and task-specific deductions

This chapter connects the depth-estimation literature to the project's calibrated video pipeline, MoGe2 pseudo-labels and original scratch CNN. It covers primary papers available by **18 October 2026**, including Depth Anything 3, DINOv3 and the verified MoGe-3 preprint. A publication's existence, architecture or published benchmark result does not establish local installation, compatibility, runtime or accuracy. The installed teacher remains the pinned MoGe2 baseline; newer models discussed here are research references.

**Evidence labels.** “Paper facts” summarize linked primary sources. “Original derivation” and “Proposed adaptation” identify mathematical synthesis or project-specific hypotheses, rather than author-reported results.

**Repository grounding.** Existing implementation descriptions below come from static source reading, principally the [scratch network](../../../track3/features/scratch_depth/_network.py), [research variants and objectives](../../../track3/features/scratch_depth/_research_network.py), [feature documentation](../../../track3/features/scratch_depth/README.md) and [MoGe installation manifest](../../../configs/perception/moge/README.md). No model, test, notebook, compiler or dependency command accompanied this research. No historical holdout result selects any proposed adaptation here. Episode 41 stays excluded; synthetic pilot samples remain explicitly training-only.

## Representation and gauge determine what the model can claim

### Depth, range, inverse depth and points

**Original derivation.** For pixel \(p=(u,v,1)^\top\), calibrated pinhole intrinsics \(K\), and optical-axis depth \(Z(p)>0\), define

\[
r(p)=K^{-1}p,\qquad P(p)=Z(p)r(p).
\]

For ordinary homogeneous pinhole rays, \(r_z=1\). The Euclidean distance from the camera center is therefore

\[
R(p)=\|P(p)\|_2=Z(p)\|r(p)\|_2.
\]

Axial depth and range coincide on the optical axis and diverge away from it. A teacher target in range cannot be consumed as axial depth without a pixel-dependent conversion. The point cloud may look globally plausible while its peripheral geometry is stretched.

Inverse depth is \(\rho=1/Z\). Its gradients emphasize nearby geometry:

\[
\frac{\partial\rho}{\partial Z}=-Z^{-2}.
\]

This is a mathematical change of error weighting, not a universal improvement. A one-centimeter near-surface error and a one-meter far-surface error can exchange importance when moving from metric-depth MAE to inverse-depth MAE. Choose a representation with the downstream use in mind, and preserve the original metric interpretation in artifact metadata.

### Gauge comparison

| Output representation | Equivalence or remaining ambiguity | Information needed for metric use | Main failure if misinterpreted |
|---|---|---|---|
| Metric axial depth \(Z\) | None by definition | Correct calibration and units | Learned metric scale can still be biased |
| Scale-invariant depth | \(Z\sim aZ\), \(a>0\) | A reliable scale constraint | Correct shape at incorrect scene scale |
| Affine-invariant depth | \(Z\sim aZ+b\) | Independent scale and shift constraints | Additive shift changes reconstructed shape |
| Affine-invariant inverse depth | \(\rho\sim a\rho+b\) | Calibration in inverse-depth space | Treating it as affine depth gives wrong surfaces |
| Scale-and-translation-invariant point map | \(P\sim aP+t\) | Camera/projection recovery plus metric scale | Projection and translation conventions become entangled |
| Canonical inverse depth | Model-specific camera normalization | Correct de-canonicalization and focal convention | Missing width/focal factors create scale error |
| Depth and dense ray map | Depends on ray magnitudes and scene normalization | The published ray convention and external metric constraints where needed | Unit-normalizing an intentionally unnormalized ray changes the meaning of depth |

“Affine point map” in the MoGe discussion means global isotropic scale plus translation, with the centered-camera simplification often restricting translation to the optical axis. It does **not** mean an unrestricted \(3\times3\) affine deformation. Such a deformation could alter angles and relative lengths and is a different gauge entirely. The relevant MoGe convention is specified in [MoGe §3.1](https://arxiv.org/html/2410.19115v1#S3.SS1).

**Original derivation.** If \(\rho'=a\rho+b\), then

\[
Z'=\frac{Z}{a+bZ}.
\]

This is a fractional transformation of depth, not \(aZ+b\). A successful affine-disparity alignment says little about absolute depth unless the fitted parameters and support remain physically valid. An ensemble should convert members to one admitted representation before averaging.

## Foundations: relative geometry, multiscale context and pretrained features

### Eigen, Puhrsch and Fergus: global prediction and scale-invariant error

**Paper facts.** *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network* (2014), arXiv:1406.2283, combines coarse image-wide prediction with local refinement. The coarse network is trained first; the fine network then learns with coarse outputs fixed. The network predicts log depth. With \(d_i=\log\hat Z_i-\log Z_i\), its training objective is

\[
L_\lambda=\frac1n\sum_i d_i^2-
\frac{\lambda}{n^2}\left(\sum_i d_i\right)^2.
\]

The paper uses \(\lambda=0.5\); \(\lambda=1\) removes global log-scale error, whereas \(\lambda=0\) is log-space MSE. Missing labels are excluded. Source locators: §§3.1–3.3, especially Eq. 4. [Primary PDF](https://arxiv.org/pdf/1406.2283)

**Original derivation.** Let \(\mu=n^{-1}\sum_i d_i\). Then

\[
L_\lambda=\operatorname{Var}(d)+(1-\lambda)\mu^2.
\]

The objective is a shape-error variance plus a separately weighted scale error. Adding a constant \(c\) to predicted log depth changes only the mean term. For \(\lambda=1\), every global scale has identical loss. Consequently, low fully scale-invariant training error does not imply a metrically useful prediction. A project that needs distance must retain a metric term or provide a separately constrained scale estimator.

### MiDaS: learn from incompatible depth sources

**Paper facts.** *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer* (2019 preprint; subsequently TPAMI), arXiv:1907.01341, combines heterogeneous supervision using scale-and-shift-invariant fitting, multiscale gradient supervision, auxiliary encoder pretraining and multiobjective dataset mixing. Its disparity/inverse-depth representation is distinct from affine invariance in metric depth. The paper considers scale-and-shift fitting and robust normalized alternatives, including median centering and mean absolute deviation normalization. Its zero-shot protocol excludes evaluation datasets from training. Source locators: §V, training objectives and multiobjective optimization; §IV explains varying stereo-film depth budgets. [Primary full text](https://arxiv.org/html/1907.01341v3)

**Original derivation.** For predictions \(x_i\), labels \(y_i\), and valid weights \(w_i\), affine least-squares alignment solves

\[
\min_{a,b}\sum_iw_i(ax_i+b-y_i)^2.
\]

The normal equations are

\[
\begin{bmatrix}
\sum wx^2&\sum wx\\
\sum wx&\sum w
\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}\sum wxy\\\sum wy\end{bmatrix}.
\]

Their determinant is \((\sum w)^2\operatorname{Var}_w(x)\). Constant or nearly constant predictions make scale and shift unidentifiable or ill-conditioned. Implementing an affine objective without a degeneracy policy can therefore generate unstable targets or gradients precisely on flat-depth frames.

The invariant objective also leaves a learning ambiguity: outputs \(x\) and \(cx+d\) can receive the same aligned loss. This is useful when labels have incompatible gauges, but it prevents a direct metric interpretation of the raw output. Keep separate names for raw relative output, aligned output and metric output.

### DPT: a transformer-to-dense decoder

**Paper facts.** *Vision Transformers for Dense Prediction* (2021), arXiv:2103.13413, introduces Dense Prediction Transformers. Intermediate states of a transformer are reassembled into image-like features at different resolutions, then progressively fused by a convolutional decoder. Token reshaping, readout-token handling and spatial resampling bridge classification-oriented ViTs and dense prediction. The transformer supplies broad context while the decoder restores a pixel grid. DPT is an architecture, not inherently a metric-depth formulation; the task head and supervision determine its output gauge. Source locators: §3 and Figure 1. [Primary PDF](https://arxiv.org/pdf/2103.13413)

**Original derivation.** If a ViT patch size is \(p\), its initial spatial token count is approximately \(N=HW/p^2\). Full attention forms \(N\times N\) relationships; doubling both image dimensions multiplies token count by four and attention-pair count by sixteen. A convolutional decoder can restore output dimensions cheaply relative to this global attention expansion, but interpolation alone cannot recreate information discarded before tokenization. The cost/detail tradeoff is therefore coupled to encoder resolution, not merely the resolution of the final depth map.

### DINOv2 and DINOv3: representation learning is not metric calibration

**Paper facts: DINOv2.** *DINOv2: Learning Robust Visual Features without Supervision* (2023), arXiv:2304.07193, combines image-level DINO self-distillation with masked patch-level iBOT objectives. The teacher is an exponential moving average of the student; separate objective heads and centering support training. The normalized-feature KoLeo regularizer is

\[
L_{\rm KoLeo}=-\frac1n\sum_i\log\min_{j\ne i}\|x_i-x_j\|_2.
\]

The LVD-142M dataset is curated by retrieval and deduplication; a late higher-resolution stage supports dense tasks. Large representations are distilled into smaller ViTs. DINOv2 supplies image features, not metric depth. Source locators: §§3–5 and §6.4. [Primary full text](https://arxiv.org/html/2304.07193v1)

**Paper facts: DINOv3.** *DINOv3* (2025), arXiv:2508.10104, introduces Gram anchoring to counter dense-feature degradation during long self-supervised training, alongside scaled curation/training and later adaptation strategies. It is relevant as an encoder development, rather than a calibrated depth estimator. Its release does not establish compatibility with this project's pinned MoGe2 code or checkpoints. Source locator: abstract and technical report. [Primary paper](https://arxiv.org/abs/2508.10104)

**Original deduction.** A scratch CNN trained from random weights on MoGe2 labels is independent of pretrained *student weights*, but not independent of pretrained *knowledge*. Teacher outputs transmit priors from its encoder, geometry training data and label refinement. This distinction matters for scientific attribution and interpreting generalization. A student reproducing teacher depth on a familiar camera distribution does not demonstrate that a few local episodes suffice to learn universal depth from scratch.

## Monocular geometry and metric-depth model families

### MoGe: optimal alignment before geometry supervision

**Paper facts.** *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision* (2024 preprint; CVPR 2025), arXiv:2410.19115, predicts point maps up to scale and translation. Assuming a centered principal point and square pixels simplifies translation to an optical-axis shift. It recovers camera parameters through

\[
\min_{f,\tau}\sum_i
\left(\frac{fX_i}{Z_i+\tau}-u_i\right)^2+
\left(\frac{fY_i}{Z_i+\tau}-v_i\right)^2.
\]

A robust optimal alignment solver supports inverse-depth-weighted \(L_1\) point supervision. Multiscale spherical neighborhoods add local supervision. The formulation addresses focal-distance ambiguity but does not independently recover metric scale. Source locators: §§3.1–3.3, especially §3.1 Eq. 2. [Primary full text](https://arxiv.org/html/2410.19115v1)

**Original deduction.** Projection-consistent geometry and metric geometry are separate conditions. Multiplying every point by \(s>0\) preserves image projection:

\[
\frac{f(sX)}{sZ}=\frac{fX}{Z}.
\]

Thus camera reprojection error alone cannot fix scene scale. A low reprojection residual should not be used as a certificate that distances are correct. It only assesses compatibility of points and projection under the selected camera assumptions.

### MoGe-2: metric scale and label refinement

**Paper facts.** *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details* (2025), arXiv:2507.02546, keeps affine point-map supervision

\[
L_G=\sum_{i\in M} Z_i^{-1}\|s^*\hat P_i+t^*-P_i\|_1
\]

and adds detached scale supervision

\[
L_s=\|\log\hat s-\operatorname{sg}(\log s^*)\|_2^2.
\]

A two-layer CLS-token MLP exponentiates its scalar output. DINOv2 layers 6, 12, 18 and final are projected, reshaped and summed; a residual convolutional decoder uses upsampling and UV information. A synthetic-trained model filters locally inconsistent real geometry; removed regions are completed by matching synthetic-model log-depth gradients under retained real-depth boundary conditions:

\[
\min_{Z^c}\sum_\Omega\|\nabla\log Z^c-\nabla\log\hat Z_{\rm syn}\|_2^2.
\]

Source locators: §§3.1–3.3, Eqs. 1, 3, 7; Appendix A.1. [Primary full text](https://arxiv.org/html/2507.02546v1)

**Original derivation: why detach a fitted scale target?** If \(s^*\) is computed from a prediction \(P_\theta\), then an unrestricted scale loss can backpropagate through both the predicted scale and the geometric fit. The scale target moves as the geometry branch changes. Detaching the fitted target makes the scale branch chase the current geometric normalization without allowing this loss alone to reshape the point field merely to make scale regression easier. It does not completely eliminate task interaction if encoder parameters remain shared; it removes one explicit gradient path.

**Original derivation: the completion equation.** Let \(x=\log Z^c\), \(g=\nabla\log\hat Z_{\rm syn}\). Minimizing \(\int_\Omega\|\nabla x-g\|^2\) with fixed boundary values yields the Euler–Lagrange equation

\[
\Delta x=\nabla\cdot g
\]

inside the completed region. Boundary constraints anchor the solution; a disconnected component without boundary constraints retains an additive log-depth ambiguity, equivalent to multiplicative scale. This explains why “completion” is not equivalent to replacing all real labels with synthetic-model predictions. The boundary conditions carry metric information.

**Proposed adaptation.** For this project, treat MoGe2 depth as a rich but fallible supervisory source. Preserve teacher masks and camera conventions. Add independent stereo evidence only where admitted by its own reliability criteria. Do not infer teacher certainty from visually sharp boundaries or from a low student-to-teacher loss.

### Depth Anything V2: synthetic precision transferred through real imagery

**Paper facts.** *Depth Anything V2* (2024), arXiv:2406.09414, combines precise synthetic labeled images, a large teacher and student distillation through pseudo-labeled real images. The DINOv2-based teacher uses a DPT decoder. The relative-depth formulation includes scale-and-shift-invariant supervision and gradient matching. Appendix B.7 links the usefulness of gradient supervision to accurate labels and adopts gradient-loss weight 2.0 for its studied setup. Metric variants receive additional metric-label fine-tuning; ordinary relative outputs should not be treated as metric depths. Source locators: §§2–5 and Appendix B.7. [Primary full text](https://arxiv.org/html/2406.09414v1)

**Original deduction.** Synthetic supervision and real pseudo-labeling solve different problems. Synthetic labels can have precise boundaries while covering an incomplete appearance distribution. Real inputs broaden appearance coverage while retaining the teacher's structured errors. Neither property establishes that an arbitrary procedural pilot resembles the real task's objects, camera, contact geometry, lighting or depth distribution. Future synthetic mixture weights must therefore be selected on real tuning episodes, with explicit provenance and no historical holdout feedback.

### Depth Pro: canonical inverse depth and multiple image scales

**Paper facts.** *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second* (2024 preprint; ICLR 2025), arXiv:2410.02073, uses a shared ViT patch encoder over multiple scales and a separate whole-image encoder. Merged patch features enter a DPT-like decoder. The described network uses \(1536^2\) processing with \(384^2\) encoder inputs. It predicts canonical inverse depth \(C\), converted using

\[
Z=\frac{f_{\rm px}}{wC}.
\]

A focal/field-of-view estimator permits operation without supplied intrinsics. Training combines real and synthetic supervision followed by a synthetic-detail stage; canonical inverse-depth and derivative objectives support geometric and boundary learning. Source locators: §§3.1–3.3. [Primary full text](https://arxiv.org/html/2410.02073v1)

**Original derivation.** For fixed \(w\), small multiplicative errors satisfy

\[
\delta\log Z\approx\delta\log f_{\rm px}-\delta\log C.
\]

Focal prediction error and canonical-depth error can compensate or reinforce each other. On a calibrated task, replacing inferred focal length with correct transformed intrinsics is a distinct intervention whose effect should be measured on real tuning data. It is not safe to assume that the two heads' errors are independent or that a checkpoint was trained to accept any alternate preprocessing.

### Metric3Dv2: canonical cameras and joint depth–normal refinement

**Paper facts.** *Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation* (2024), arXiv:2404.15506, uses canonical focal length \(f_c\). Its label-transform option applies

\[
Z_c=(f_c/f)Z,\qquad Z=(f/f_c)\hat Z_c.
\]

An alternative resizes image coordinates by \(f_c/f\) without scaling depth values. A ConvGRU-based refinement module jointly updates depth and normals. Supervision includes log-depth, virtual/pairwise-normal, random-proposal local normalization, available normal labels and depth–normal consistency. Known intrinsics enable canonicalization across its mixed datasets. Source locators: §§III-B–III-D, Eqs. 2–10. [Primary full text](https://arxiv.org/html/2404.15506v1)

**Original derivation.** With backprojected points, a local normal can be estimated by

\[
n_i=\frac{(P_{i+e_x}-P_i)\times(P_{i+e_y}-P_i)}
{\|(P_{i+e_x}-P_i)\times(P_{i+e_y}-P_i)\|_2}.
\]

A three-pixel neighborhood crossing an occlusion boundary combines different surfaces and produces an artificial normal. Normal losses therefore need neighborhood-level support, not just a valid central depth. Tiny cross products also make normalization unstable. A normal term can regularize within-surface shape, but it should not smooth foreground depth toward background depth merely because adjacent pixels exist.

### UniDepthV2: camera conditioning, local shape and uncertainty

**Paper facts.** *UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler* (2025), arXiv:2502.20110, represents output as azimuth, elevation and log axial depth. A camera module constructs angular rays and conditions multiresolution features through cross-attention. Its decoder combines sinusoidal camera embeddings, residual convolution and FPN-like fusion. The output objective uses

\[
L=\|\operatorname{Var}(\epsilon)\|_1+
\lambda^\top(\operatorname{E}\epsilon\odot\operatorname{E}\epsilon),
\]

with depth weight \(\lambda_z=0.15\). Geometric augmentation consistency acts on depth outputs. Edge-guided patches compare independently normalized inverse depths. An uncertainty head regresses detached absolute log-depth error through \(L_1\) supervision. Source locators: §§III-A–III-E, Eqs. 1–6. [Primary full text](https://arxiv.org/html/2502.20110v1)

**Original deduction.** A head trained by \(L_1\) regression to \(|\log\hat Z-\log Z|\) estimates a conditional median of that target under ideal optimization, not a calibrated posterior variance. Turning its output into inverse-variance weights requires a further model and calibration evidence. More generally, “confidence” can mean validity, expected residual, learned likelihood precision, ensemble disagreement or sensor matching quality. These quantities should retain different field names and admission rules.

## Video and multi-view geometry

### Video Depth Anything: match changes rather than constant depth

**Paper facts.** *Video Depth Anything: Consistent Depth Estimation for Super-Long Videos* (2025), arXiv:2501.12375, freezes a Depth Anything V2 encoder and trains a DPT-derived spatial-temporal head. Temporal attention mixes frames with temporal positional embeddings. Relative outputs share a video-level scale and shift. Its aligned temporal objective is

\[
L_{\rm TGM}=\frac1{N-1}\sum_t
\big\||d_{t+1}-d_t|-|g_{t+1}-g_t|\big\|_1,
\]

restricted to sufficiently small ground-truth temporal changes. Long-video inference uses overlapping frames, persistent key frames, alignment and interpolation. Source locators: §§3.1–3.3, Eq. 3. [Primary full text](https://arxiv.org/html/2501.12375v1)

**Original derivation.** A tracked static point changes camera coordinates under motion:

\[
P_{t+1}=RP_t+t,\qquad Z_{t+1}=e_3^\top(RP_t+t).
\]

Therefore equal image identity or correspondence does not imply equal camera-axis depth. Penalizing depth equality after optical-flow warping can oppose correct ego-motion. Pose-compensated point comparison, or supervision of the correct target temporal change, is better founded.

The magnitude-based temporal expression also has a limitation: \(|\Delta d|=|\Delta g|\) permits either temporal sign. Spatial depth supervision and other terms must resolve this ambiguity. A temporal penalty should therefore not be interpreted as independently sufficient evidence of physically correct motion.

### Depth Anything 3: depth plus unnormalized ray maps

**Paper facts.** *Depth Anything 3: Recovering the Visual Space from Any Views* (2025; ICLR 2026), arXiv:2511.10647, accepts single/multiple views and optional cameras. A pretrained ViT first attends within views; later blocks alternate within- and cross-view attention. Camera tokens encode supplied parameters or a learned placeholder. Shared reassembly and separate fusion create a dual-DPT head. Rays contain origin \(t\) and **unnormalized** direction \(d=RK^{-1}p\):

\[
P=t+Zd.
\]

An auxiliary camera head supplies convenient parameters. Training combines depth, ray, reconstructed-point, optional-camera and gradient terms after shared scene normalization. A synthetic-trained teacher refines real geometric supervision. Separate monocular and metric variants require attention to output gauge. Source locators: §§3.1–3.4 and §§4.1–4.4. [Primary full text](https://arxiv.org/html/2511.10647v1)

**Original derivation.** Replacing \(d\) by unit ray \(\bar d=d/\|d\|\) while retaining \(Z\) changes the reconstructed point. To preserve the original point, use ray distance \(\bar Z=Z\|d\|\). This is the depth/range distinction expressed in the world frame. A generic “normalize all vectors” utility can silently break a correct depth–ray representation.

For \(V\) views each containing \(N\) tokens, joint attention can involve \((VN)^2\) token pairs, versus \(VN^2\) within-view pairs. The ratio grows with view count. The existence of arbitrary-view input support does not mean arbitrary-length video inference has bounded memory or causal access. The project must separately specify permitted context, latency, future-frame access and episode boundaries before selecting a multi-view model.

## Verified 2026 development: MoGe-3 sparse volumetric refinement

### Architecture and training distinction

**Paper facts.** *MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement* (2026), arXiv:2607.17967, extends a MoGe2-initialized base with Self-Guided Sparse 3D Refinement. Version-one full text omits “MoGe-3” in its title. It uses

\[
q=(X/Z,Y/Z,\zeta),\quad\zeta=\log Z,\qquad
c_{ij}=(i,j,\operatorname{round}(D\zeta_{ij})).
\]

A sparse 3D U-Net conditioned on image features predicts \(\Delta\zeta\); iterations update depth and voxel connectivity while preserving initial rays. Zero-initialized output and detached warmup precede joint training. Refiner gradients use synthetic samples; the base uses mixed real/synthetic supervision. Global/local geometry, edge-angle and auxiliary losses supervise the system. Default described inference uses three refinements. Source locators: §§3.1–3.3, Eqs. 1–6. [Current abstract](https://arxiv.org/abs/2607.17967), [version-one full text](https://arxiv.org/html/2607.17967v1)

### Original deductions and limits for this task

The factorized update has a simple multiplicative interpretation:

\[
Z^{k+1}=Z^k\exp(\Delta\zeta^k).
\]

Holding \(X/Z\) and \(Y/Z\) fixed restricts the correction to motion along the initial camera ray. This preserves pixel association but cannot independently correct an erroneous ray direction or camera model. A local depth refiner therefore does not replace camera calibration.

Log-depth differences represent ratios:

\[
|\log Z_i-\log Z_j|\le\tau
\iff e^{-\tau}\le Z_i/Z_j\le e^{\tau}.
\]

This gives a useful depth-relative neighborhood criterion, but the voxel lattice \((i,j,\operatorname{round}(D\log Z))\) is not an ordinary uniform Cartesian \((X,Y,Z)\) lattice. Its neighborhoods depend on image location and quantized relative depth. Describing it simply as Euclidean nearest-neighbor geometry would hide a material architectural choice.

Uniform metric rescaling adds a constant to \(\log Z\), preserving continuous log-depth differences. With fixed rounding boundaries, however, arbitrary noninteger shifts of \(D\log Z\) can change quantization membership near bin edges. Continuous scale invariance does not automatically imply exact equality of every discretized connectivity pattern. This is a deduction about finite discretization, not a contradictory benchmark claim.

**Proposed adaptation.** The transferable hypothesis is to reduce foreground/background feature mixing when depth discontinuities are already supported by evidence. A small project could investigate masked or depth-gated local aggregation before considering a new sparse-convolution dependency. Such an adaptation is not MoGe-3 reproduction, and the user's dependency-confirmation rule still applies. No replacement teacher or refiner is selected by this chapter.

## Architecture comparison and project relevance

| Family | Main computational pattern | Geometry/scale strategy | Most relevant transferable question |
|---|---|---|---|
| Eigen | Coarse global stack plus local refinement | Log-depth error separates scale and shape | Does the student need a distinct global-scale mechanism? |
| MiDaS | Mixed-data dense prediction | Invariant disparity supervision | Which labels require alignment, and in which space? |
| DPT / DINO encoders | Global token features plus spatial decoder | Determined by downstream heads | What context is lost at the student's small resolution? |
| MoGe | Point-map prediction and robust alignment | Projection recovery with unresolved metric scale | Are teacher coordinates and camera assumptions preserved? |
| MoGe2 | Dense relative geometry plus global scalar | Detached scale target and refined real labels | Can independent metric anchors expose teacher scale bias? |
| Depth Anything V2 | Large teacher and real-image distillation | Relative depth; metric fine-tuning variants | Does procedural data improve real tuning behavior? |
| Depth Pro | Multi-scale patches plus global image | Canonical inverse depth and focal estimation | Does calibrated camera information remove an error source? |
| Metric3Dv2 | Canonical camera and recurrent depth/normal update | Known-camera canonicalization | Are local normal terms supported on the correct surfaces? |
| UniDepthV2 | Camera-conditioned multiresolution decoder | Angular camera representation and log depth | How should local shape and residual confidence be supervised? |
| Video Depth Anything | Frozen image encoder plus temporal head | Shared video gauge and temporal changes | Does temporal regularization preserve real depth change? |
| Depth Anything 3 | Single/multi-view transformer and dual dense head | Depth–ray scene representation | Is multi-view context admitted, causal and memory-bounded? |
| MoGe-3 | Geometry-conditioned sparse iterative refinement | Log-depth ray-preserving residuals | Can feature mixing across supported discontinuities be reduced? |

The table combines the compact paper facts above with original project questions; it is not a benchmark ranking. Models use different supervision, cameras, gauges and evaluation conventions. Published latency and accuracy values cannot be transplanted into the project's hardware, input sizes or data distribution.

## Existing scratch CNN and mathematically motivated adaptations

### Static implementation facts

The [historical scratch network](../../../track3/features/scratch_depth/_network.py) is an original randomly initialized convolutional encoder–decoder. It uses two \(3\times3\) ReLU convolutions per block, channels \(24,48,96\), three \(2\times2\) max-pooling stages and a 192-channel bottleneck with spatial dropout. Three bilinear upsampling stages concatenate matching encoder features. A \(1\times1\) head produces

\[
\hat Z=\operatorname{softplus}(h)+10^{-3}.
\]

The output is positive optical-axis depth. Expressed as **width by height**, a representative \(128\times96\) input gives encoder resolutions \(128\times96\), \(64\times48\), \(32\times24\), then a \(16\times12\) bottleneck. In NCHW tensor order these are height by width, beginning with \([N,3,96,128]\). The maximum-side-128 preparation setting can produce other aspect-ratio-preserving sizes; \(128\times96\) should not be treated as an invariant dataset shape.

The historical objective is masked log-depth MAE:

\[
L=\frac1{|M|}\sum_{i\in M}|\log\hat Z_i-\log Z_i^T|.
\]

It retains teacher scale; it does not perform per-frame alignment. Invalid targets are replaced before logarithms and then masked. There is no learned uncertainty in this historical model.

The [research variant source](../../../track3/features/scratch_depth/_research_network.py) separately defines direct and residual models with admitted base-depth/context channels, bounded log-depth correction, optional robust log loss, boundary residual differences, geometry-derived normal loss and center-camera-transformed temporal support. Those source definitions must not be conflated with proof that every route has executed or that any particular variant is the frozen selected model. The owning feature documentation records execution history separately.

### Original receptive-field analysis

For convolution/pooling layer \(l\) with kernel \(k_l\), stride \(s_l\), input sampling jump \(j_l\) and receptive field \(r_l\),

\[
r_{l+1}=r_l+(k_l-1)j_l,\qquad j_{l+1}=s_lj_l.
\]

Starting with \(r=1,j=1\), the historical encoder's two convolutions, pooling, two convolutions, pooling, two convolutions, pooling and two bottleneck convolutions yield bottleneck receptive field 68 pixels and jump 8, ignoring boundary padding effects. Decoder mixing broadens output dependence further, but a single bottleneck feature is not automatically an image-wide summary at width 128. This distinguishes the student's inductive bias from a pretrained transformer with global token mixing.

**Proposed adaptation.** If scale prediction requires image-wide evidence, global pooling over bottleneck features can supply it without copying a foundation transformer. A separate scalar head should have a defined gauge rather than merely duplicating an unconstrained bias already available to the dense decoder.

### Original shape–scale factorization

A candidate log-depth parameterization is

\[
\log\hat Z_i=g_\theta(I)_i+h_\phi(I),\qquad
\sum_iw_i g_\theta(I)_i=0.
\]

The centering condition removes the ambiguity \(g\mapsto g+c\), \(h\mapsto h-c\). Without it, the apparent scalar head and dense shape head can exchange scale arbitrarily, making their separate interpretation unreliable.

Teacher-relative shape can supervise \(g\), while admitted metric anchors constrain \(h\). This is a proposal, not a change to the existing model. It becomes useful only if real tuning evidence shows that scale and shape errors differ systematically.

### Original bounded residual analysis

The source research residual expression has the form

\[
\hat Z=Z_{\rm base}\exp(b\tanh r).
\]

Its correction ratio is bounded by

\[
e^{-b}<\hat Z/Z_{\rm base}<e^b.
\]

This limits harmful corrections but also makes sufficiently large base errors uncorrectable. The derivative

\[
\partial\log\hat Z/\partial r=b\operatorname{sech}^2r
\]

saturates for large \(|r|\). A candidate bound trades safety against recoverable error magnitude; it should be selected using real tuning evidence. No historical holdout outcome should determine it.

### Original loss design map

A candidate objective may combine distinct terms:

\[
L=\lambda_sL_{\rm centered\,log}
+\lambda_mL_{\rm metric\,anchors}
+\lambda_gL_{\rm target\,gradient}
+\lambda_nL_{\rm normals}
+\lambda_eL_{\rm augmentation}.
\]

The coefficients are not interchangeable. A fully invariant shape term cannot restore metric scale. A gradient term preserves target discontinuities but can faithfully preserve erroneous teacher edges. A normal term operates on 3D neighborhoods and needs calibrated rays. An augmentation-consistency term discourages transform sensitivity, but cannot establish correctness if both augmented predictions share the same bias.

Copying published weights is unjustified because image resolution, target gauge, number of valid pixels, reduction conventions and dataset mixture differ. For example, summing a gradient term while averaging a pixel term changes its effective weight with image size. Each term needs a stated normalization and support count before tuning comparisons are interpretable.

## Independent stereo anchors, calibration and target preparation

### Original derivation of robust scale anchoring

For admitted stereo depth \(Z_i^S\), teacher or student depth \(Z_i^T\), and reliability weights \(w_i\), fit a pure log-scale correction by

\[
a^*=\arg\min_a\sum_iw_i|
\log Z_i^S-\log Z_i^T-a|.
\]

The solution is a weighted median of log-depth differences. Corrected depth is \(e^{a^*}Z^T\). Its weights should reflect stereo reliability, rather than being determined solely by agreement with the monocular teacher; otherwise an independent signal is filtered into confirming the teacher.

If an affine inverse-depth correction is necessary, fit \(\rho_S=A\rho_T+B\) using the earlier affine equations, and require positive corrected inverse depth over accepted support. Nearly constant disparities cannot reliably identify both parameters. A lower-complexity fallback is preferable to an unstable two-parameter fit.

### Original stereo uncertainty propagation

For rectified stereo baseline \(B\), focal length \(f\) in pixels and disparity \(d\),

\[
Z=fB/d,\qquad
\left|\frac{\partial Z}{\partial d}\right|=fB/d^2,
\qquad\sigma_Z/Z\approx\sigma_d/d.
\]

Small disparities are weak far-depth anchors even when their absolute matching error seems small. Repeated textures, low texture, reflections and occlusions can create structured errors that are not captured by this first-order Gaussian approximation. Retain matching support, occlusion exclusions and calibration provenance; dense output alone does not imply dense reliable truth.

### Original image-transform bookkeeping

For resize and crop represented by

\[
A=\begin{bmatrix}s_x&0&-c_x\\0&s_y&-c_y\\0&0&1\end{bmatrix},
\qquad K'=AK,
\]

focal lengths and principal points change in pixels, while metric scene depth does not. A label-space canonicalization is a separate operation and must not be accidentally applied twice.

Actual resamplers can introduce half-pixel offsets, so store the exact mapping used by the pipeline. Depth targets, validity masks, RGB and stereo calibration must refer to the same pixel grid. At small input sizes, a one-pixel shift is a substantial fraction of a thin object's support.

### Original resolution and boundary analysis

Averaging foreground and background depths produces an intermediate surface that may not exist. For a two-surface pixel footprint with depths \(Z_f,Z_b\) and foreground fraction \(q\), arithmetic depth averaging gives \(qZ_f+(1-q)Z_b\); averaging inverse depth and inverting gives

\[
\left(q/Z_f+(1-q)/Z_b\right)^{-1},
\]

which is different. Neither is automatically the correct visible-surface label. Downsampling policy therefore changes the task. Candidate policies include support-aware nearest sampling, conservative boundary masking or a task-specific robust reduction, all documented as preparation choices.

Subpixel structures cannot be reconstructed reliably from a single low-resolution RGB image merely because the output is upsampled. Higher-resolution teacher detail can improve supervision where the input contains evidence, but cannot create missing input information. Avoid comparing a low-resolution student and high-resolution teacher on unqualified boundary sharpness alone.

## Distillation bias, ensembles and confidence

### Original teacher-bias model

Write teacher log-depth as

\[
y^T=y^\star+b(I)+\epsilon.
\]

Under squared-loss distillation, ideal optimization targets the conditional expectation of teacher outputs. Systematic bias \(b(I)\) does not disappear just by adding correlated frames. Under absolute loss, the analogous target is a conditional median. Restricted student capacity can suppress random or inconsistent teacher detail, but this is not proof of improved scene-depth accuracy.

Teacher agreement is consequently a distillation metric. Independent calibrated stereo, carefully admitted physical constraints or new external measurement can assess complementary properties. These evidence types should remain distinct in the [artifact contracts](../../design/artifact-contracts.md) and owning feature records.

### Original correlation limit for ensembles

For equal-variance member errors with pairwise correlation \(r\),

\[
\operatorname{Var}(\bar e)=\sigma^2\left(r+\frac{1-r}{M}\right).
\]

Increasing member count removes only the independent component. Multiple students trained on the same teacher can share scale and boundary bias. A different information source may therefore contribute more than another architecture of similar supervision.

Average outputs only after unit, gauge, camera and mask admission. Averaging metric depths, log depths and inverse depths yields arithmetic, geometric and harmonic combinations respectively; those choices encode different error assumptions. An inverse-tuning-error weight can also overconcentrate on a small sample. Any ensemble definition must be selected and frozen on real tuning evidence before a fresh unbiased assessment.

### Original confidence taxonomy

| Quantity | What it can mean | What it does not automatically establish |
|---|---|---|
| Validity mask | Output is defined/admitted | Small error |
| Stereo matching quality | Correspondence evidence under a matcher | Correct geometry under every occlusion or reflection |
| Predicted residual magnitude | Typical error under its training target | Calibrated posterior variance |
| Ensemble disagreement | Member sensitivity/disagreement | Detection of shared bias |
| Teacher–student agreement | Successful reproduction | Independent ground truth |
| Geometric reprojection residual | Projection compatibility | Correct global metric scale |

These fields should not be collapsed into one generic confidence channel without defining its semantics and calibration. A proposed downstream policy may use conservative admission thresholds, but the threshold must be tuned against the decision it supports, rather than copied from an unrelated paper.

## Failure modes and bounded research questions

| Failure mode | Mechanism | Research question that can be resolved on real tuning data |
|---|---|---|
| Global metric bias | Learned scene-size/focal prior differs from calibrated task | Does independently admitted scale anchoring help consistently? |
| Boundary smearing | Label mismatch or cross-surface feature aggregation | Does support-aware preparation or target-gradient supervision improve supported edges? |
| Hallucinated relief | Appearance texture is mistaken for geometry | Does local shape supervision reduce planar-surface false structure? |
| Thin-object disappearance | Input undersampling or downsampling mixtures | Which structures remain supported at the chosen input resolution? |
| Temporal breathing | Independent frame gauges or unstable scale | Does clipwise gauge handling improve consistency without erasing real motion? |
| Occlusion contamination | A temporal/normal neighborhood combines different surfaces | Are masks defined on complete neighborhoods and visibility? |
| Far-depth instability | Small disparity and inverse-depth sensitivity | Is metric anchoring limited to an identifiable range? |
| Student copies teacher failure | Shared supervisory bias | Does an independent signal reveal a distinct error component? |
| Synthetic shortcut | Procedural appearance or geometry differs from real scenes | Do mixture benefits repeat across both real tuning episodes? |
| Overfit architecture selection | Few independent episodes despite many pixels | Are changes robust across episodes rather than one pooled score? |

The project has eight training and two tuning episodes in its stated small-data regime. Pixels and nearby frames supply optimization signal but are not independent scenes. Episode-first sampling avoids letting clip length silently define training importance. Two tuning episodes support only limited claims about generalization; a large candidate search can overfit them even if each individual model is small.

The earlier campaign has already consumed its frozen holdout assessment. This chapter does not reopen that holdout or use its outcomes to select loss terms, calibration bounds, ensemble members or synthetic ratios. Future changes need a separately fixed assessment protocol. The procedural 128-sample pilot is training-only evidence with explicit synthetic provenance, not evidence that a particular mixture is optimal.

Useful bounded questions are therefore narrow and mechanistic: whether calibration preserves scale under preprocessing; whether independent anchors identify scale errors; whether label support survives downsampling; whether a loss is normalized consistently; whether a temporal neighborhood is in the center camera; and whether any proposed improvement repeats across the real tuning episodes. These questions fit the existing vertical slice and [system design](../../design/system-overview.md) without assuming that a newly published foundation model should replace the installed baseline.

## Primary-source bibliography and reading locators

| Work | Year / identifier | Primary source | Most useful locator |
|---|---|---|---|
| Eigen, Puhrsch, Fergus, *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network* | 2014; 1406.2283 | [Paper](https://arxiv.org/abs/1406.2283) | §§3.1–3.3; Eq. 4 |
| Ranftl et al., *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer* | 2019 preprint; 1907.01341 | [Full text](https://arxiv.org/html/1907.01341v3) | §IV stereo film ambiguity; §V objectives/mixing |
| Ranftl, Bochkovskiy, Koltun, *Vision Transformers for Dense Prediction* | 2021; 2103.13413 | [Paper](https://arxiv.org/abs/2103.13413) | §3; Figure 1 |
| Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* | 2023; 2304.07193 | [Full text](https://arxiv.org/html/2304.07193v1) | §§3–5; §6.4 |
| Wang et al., *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision* | 2024 preprint / CVPR 2025; 2410.19115 | [Full text](https://arxiv.org/html/2410.19115v1) | §§3.1–3.3 |
| Wang et al., *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details* | 2025; 2507.02546 | [Full text](https://arxiv.org/html/2507.02546v1) | §§3.1–3.3; Appendix A.1 |
| Yang et al., *Depth Anything V2* | 2024; 2406.09414 | [Full text](https://arxiv.org/html/2406.09414v1) | §§2–5; Appendix B.7 |
| Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second* | 2024 preprint / ICLR 2025; 2410.02073 | [Full text](https://arxiv.org/html/2410.02073v1) | §§3.1–3.3 |
| Hu et al., *Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation* | 2024; 2404.15506 | [Full text](https://arxiv.org/html/2404.15506v1) | §§III-B–III-D |
| Piccinelli et al., *UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler* | 2025; 2502.20110 | [Full text](https://arxiv.org/html/2502.20110v1) | §§III-A–III-E |
| Chen et al., *Video Depth Anything: Consistent Depth Estimation for Super-Long Videos* | 2025; 2501.12375 | [Full text](https://arxiv.org/html/2501.12375v1) | §§3.1–3.3; Eq. 3 |
| Lin et al., *Depth Anything 3: Recovering the Visual Space from Any Views* | 2025 / ICLR 2026; 2511.10647 | [Full text](https://arxiv.org/html/2511.10647v1) | §§3.1–3.4; §§4.1–4.4 |
| Siméoni et al., *DINOv3* | 2025; 2508.10104 | [Paper](https://arxiv.org/abs/2508.10104) | Technical report; Gram anchoring |
| Kong et al., *MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement* | 2026; 2607.17967 | [Current abstract](https://arxiv.org/abs/2607.17967); [v1 full text](https://arxiv.org/html/2607.17967v1) | §§3.1–3.3; v1 title omits “MoGe-3” |

Section locators identify the read method descriptions; they are preferable to unverified page numbers across changing manuscript versions. Links to versioned full text make the methodological evidence explicit. The bibliography is a focused technical lineage for this depth slice, not a claim that every depth paper published before the cutoff has been exhausted.
