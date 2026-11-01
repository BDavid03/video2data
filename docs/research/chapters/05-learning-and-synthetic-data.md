# Learning architectures, pseudo-supervision, synthetic enrichment, and experimental inference

This chapter connects primary learning-methodology literature to the repository's small-data depth experiments. It distinguishes source findings, facts established by reading the implementation, and original mathematical reasoning. A cited result from another dataset is not a performance prediction for this project. All proposed changes below are prospective research hypotheses; this document does not change the frozen campaign recipe or authorize execution.

The relevant real-data student campaign used 128 accepted MoGe2 pseudo-target training pairs from eight episodes. The separate 128-sample procedural pilot is training-only synthetic data and did not enter that completed campaign. Episodes 43 and 44 have already supplied holdout outcomes; those outcomes cannot select later architectures, losses, ensemble weights, or synthetic ratios. Episode 41 remains excluded. A fresh unbiased assessment requires an independently fixed protocol.

## Reading the evidence correctly

Three evidence categories recur throughout this chapter:

- **Literature finding:** a result or method actually described by a cited primary paper, subject to that paper's task, data, and assumptions.
- **Implementation fact:** behavior visible in the linked source, without implying that it has been executed or experimentally verified during this research pass.
- **Original analysis or proposal:** an algebraic consequence, project-specific interpretation, or future experiment. These passages are not attributed as empirical findings of the cited authors.

## The actual student and the limits of “from scratch”

Let the real training corpus be

\[
\mathcal D_r=\{(I_j,T_j,M_j,e_j)\}_{j=1}^{128},
\]

where RGB image \(I_j\) has a teacher target \(T_j\), usable-target mask \(M_j\), and episode identifier \(e_j\). The student learns \(f_\theta:I\mapsto\widehat D\) from random initialization. Random student initialization does not erase the external knowledge embodied in MoGe2 labels. This is teacher-supervised learning with an original student, not supervision independent of pretrained models.

Hinton, Vinyals, and Dean describe transferring a larger model's or ensemble's predictions into a smaller model. Their classification formulation uses softened distributions

\[
p_i^{(\tau)}=\frac{\exp(z_i/\tau)}{\sum_k\exp(z_k/\tau)},\qquad
\mathcal L_{KD}=\tau^2D_{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)}).
\]

The temperature reveals relations among class probabilities. Continuous pseudo-depth regression is analogous knowledge transfer, but does not acquire that temperature mechanism merely by being called distillation. [Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015](https://arxiv.org/abs/1503.02531)

**Original analysis.** Write the pseudo-target as

\[
T(I)=Y(I)+b(I)+\epsilon(I),
\]

where \(Y\) is unknown physical truth, \(b\) is structured teacher bias, and \(\epsilon\) is residual noise. Under squared error and sufficient capacity, the population predictor of \(T\) approaches \(\mathbb E[T\mid I]\), not automatically \(\mathbb E[Y\mid I]\). Under absolute error it approaches a conditional median instead. Reducing teacher disagreement therefore demonstrates student fidelity, while independently measured geometry or task outcomes are needed to establish physical accuracy.

Dense labels provide many optimization terms but few independent scenes. Neighboring pixels share surfaces, lighting, camera calibration, and teacher failures; neighboring frames share much of the same content. Treating all valid pixels as independent observations understates statistical uncertainty. Eight training episodes remain eight groups even if they produce millions of supervised pixels.

Zhang and colleagues show that expressive classification networks can fit random labels, cautioning against treating training fit as sufficient evidence of generalization. Their experiment does not establish a quantitative noise law for dense teacher regression. [Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization*, 2017](https://arxiv.org/abs/1611.03530)

**Project implication.** Record both initialization provenance and label provenance. Preserve teacher version, preprocessing, masks, frame lineage, and camera conventions alongside the student checkpoint. Claims such as “trained without pretrained weights” should explicitly refer to student parameters.

## Exact coursework CNN architecture

The implementation is [CourseworkDepthCNN](../../../track3/features/scratch_depth/_network.py), documented in the [scratch-depth feature](../../../track3/features/scratch_depth/README.md). It uses RGB in \([0,1]\), channels-first tensors, double convolutions with ReLU, three max-pooling stages, bilinear interpolation to the exact skip size, concatenating skips, bottleneck channel dropout, and a positive optical-axis-depth head. It contains no GroupNorm or residual blocks.

For the representative \(N\times3\times96\times128\) input and default width \(c=24\):

| Stage | Operation | Output shape, excluding batch notation only where obvious |
|---|---|---|
| Input | RGB, channels first | \(N\times3\times96\times128\) |
| Encoder 1 | Two padded \(3\times3\) convolutions, each followed by ReLU | \(N\times24\times96\times128\) |
| Pool 1 | \(2\times2\) max pooling | \(N\times24\times48\times64\) |
| Encoder 2 | Double convolution | \(N\times48\times48\times64\) |
| Pool 2 | Max pooling | \(N\times48\times24\times32\) |
| Encoder 3 | Double convolution | \(N\times96\times24\times32\) |
| Pool 3 | Max pooling | \(N\times96\times12\times16\) |
| Bottleneck | Double convolution, then Dropout2d with default probability 0.1 | \(N\times192\times12\times16\) |
| Decoder 3 input | Bilinear upsample to encoder 3 size; concatenate 192 and 96 channels | \(N\times288\times24\times32\) |
| Decoder 3 output | Double convolution | \(N\times96\times24\times32\) |
| Decoder 2 input | Upsample and concatenate 96 and 48 channels | \(N\times144\times48\times64\) |
| Decoder 2 output | Double convolution | \(N\times48\times48\times64\) |
| Decoder 1 input | Upsample and concatenate 48 and 24 channels | \(N\times72\times96\times128\) |
| Decoder 1 output | Double convolution | \(N\times24\times96\times128\) |
| Head | Biased \(1\times1\) convolution; softplus plus \(10^{-3}\) | \(N\times1\times96\times128\) |

Interpolation uses `align_corners=False`. The actual implementation supports other dimensions and interpolates to each stored skip's dimensions rather than assuming that every dimension is divisible by eight. This matters because repeated integer pooling can round sizes.

### Parameter count derived from the source

**Original derivation.** A biased \(3\times3\) convolution from \(a\) to \(b\) channels has \(9ab+b\) parameters. A double-convolution block therefore has

\[
P_B(a,b)=9ab+9b^2+2b.
\]

The seven blocks have channel pairs \((3,c),(c,2c),(2c,4c),(4c,8c),(12c,4c),(6c,2c),(3c,c)\). Their contributions are

| Block | Parameter expression |
|---|---|
| Encoder 1 | \(9c^2+29c\) |
| Encoder 2 | \(54c^2+4c\) |
| Encoder 3 | \(216c^2+8c\) |
| Bottleneck | \(864c^2+16c\) |
| Decoder 3 | \(576c^2+8c\) |
| Decoder 2 | \(144c^2+4c\) |
| Decoder 1 | \(36c^2+2c\) |
| Head | \(c+1\) |

Hence

\[
P(c)=1899c^2+72c+1,\qquad P(24)=1,095,553.
\]

Pooling, interpolation, ReLU, and dropout add no trainable parameters. This count is algebraic source analysis, not a runtime inspection. It applies to the RGB coursework model, not the six- or eighteen-channel admitted research variants.

### What U-Net supplies and what it does not establish

U-Net combines a contracting context pathway with an expanding localization pathway and corresponding concatenating skips. The original biomedical segmentation study emphasizes augmentation with scarce labeled images. Its results support the architectural idea, not a guarantee about real video depth with 128 teacher frames. [Ronneberger et al., *U-Net*, 2015](https://arxiv.org/abs/1505.04597)

**Original analysis.** A generic decoder recurrence is

\[
E_{\ell+1}=B_{\ell+1}(\operatorname{Pool}(E_\ell)),\qquad
D_\ell=C_\ell([E_\ell,\operatorname{Up}(D_{\ell+1})]).
\]

The bottleneck can combine wider context while the skips preserve edges. At maximum side 128, a thin obstacle may disappear during pooling; reconstruction cannot reliably recover evidence never retained. Conversely, skips can pass texture directly toward the output, so sharp boundaries need not imply correct geometry. Evaluation should distinguish geometric boundaries from texture copying.

For a convolution, approximate multiply-accumulate work is \(HWk^2C_{in}C_{out}\). Width scaling therefore changes most computation quadratically. Parameter count alone also misses high-resolution activation memory, which is substantial in encoder-decoder skips.

### Residual blocks and GroupNorm as prospective changes

Residual learning parameterizes a block as \(y=x+F(x;\theta)\), with projections when shapes differ. He and colleagues show optimization advantages for much deeper image recognition networks. [He et al., *Deep Residual Learning for Image Recognition*, 2016](https://arxiv.org/abs/1512.03385)

**Original analysis.** The Jacobian \(\partial y/\partial x=I+\partial F/\partial x\) contains an identity path, making an identity-like solution easier to represent. This motivates a prospective residual student, but does not prove that increasing depth improves generalization here. Distinguish residual **feature blocks** from a residual **depth correction head**: these are separate design decisions.

GroupNorm computes within-sample statistics across spatial positions and subsets of channels, independent of batch size:

\[
\mu_g=|S_g|^{-1}\sum_{i\in S_g}x_i,\quad
v_g=|S_g|^{-1}\sum_{i\in S_g}(x_i-\mu_g)^2,\quad
y_i=\gamma_i(x_i-\mu_g)/\sqrt{v_g+\epsilon}+\beta_i.
\]

The original work targets small-batch limitations of BatchNorm. [Wu and He, *Group Normalization*, 2018](https://arxiv.org/abs/1803.08494)

**Project proposal.** A normalization ablation should preserve data, seeds, capacity accounting, optimizer budget, and tuning groups. Group count must divide width. The existing default does not contain normalization, and no paper establishes that adding it is necessarily beneficial to this student.

## The other implemented model families

The [learning-experiment factories](../../../track3/features/learning_experiments/models.py) implement alternatives distinct from CourseworkDepthCNN. Their names must not be interpreted more broadly than their actual inductive biases.

| Family | Actual implementation | Information available at one output pixel |
|---|---|---|
| ANN | Repeated \(1\times1\) convolutions and ReLU, then a \(1\times1\) head | That pixel's RGB only |
| Research CNN | Repeated same-resolution padded \(3\times3\) convolutions and ReLU, then \(1\times1\) head | A finite local RGB neighborhood |
| Transformer | Strided patch convolution, learned positional tokens, transformer encoder, transposed-convolution patch decoder | Globally interacting patch tokens within the admitted token budget |
| Gradient boosting | Additive RGB decision stumps fitted to log-depth residuals | That pixel's RGB only |
| Coursework CNN | Three-scale encoder-decoder with skips | Local detail and multiscale context |

### Pointwise ANN and shallow CNN

**Original analysis of implementation.** The ANN is a shared per-pixel MLP written using \(1\times1\) convolutions. It is not an image-flattening fully connected network. If two pixels have identical RGB, this deterministic network produces identical depths, regardless of object identity, position, or context. Its failures can reveal the importance of contextual information, but do not refute the general usefulness of ANNs.

For width \(w\) and \(L\ge1\) hidden convolution layers, a kernel width \(k\) gives

\[
P=(L-1)k^2w^2+(3k^2+L+1)w+1.
\]

Set \(k=1\) for the ANN and \(k=3\) for the shallow CNN. With unit-stride \(3\times3\) hidden convolutions and no pooling, the nominal receptive-field width is \(1+2L\), excluding padding effects. This makes the shallow CNN different from the multiscale coursework network even at similar parameter counts.

### Patch transformer

ViT tokenizes images into patches and applies transformer blocks; its original strong transfer results rely on large-scale pretraining. Attention is

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V.
\]

[Dosovitskiy et al., *An Image Is Worth 16x16 Words*, 2021](https://arxiv.org/abs/2010.11929)

**Implementation fact.** This repository's transformer starts randomly, embeds patches using a convolution with kernel and stride equal to patch size, adds learned one-dimensional positional parameters, applies encoder blocks, and uses a transposed convolution to reconstruct patch-sized outputs. Images are padded to a whole patch grid and cropped after decoding. Feed-forward width is twice embedding width; transformer dropout is zero in this factory.

**Original analysis.** For image size \(H\times W\), token count is \(N=\lceil H/p\rceil\lceil W/p\rceil\). Conventional attention interactions scale quadratically with \(N\). Smaller patches preserve finer spatial evidence but increase attention work. At \(96\times128\), patch size 8 produces 192 tokens; patch size 16 produces 48. These are geometric examples, not assertions about a selected recipe. A learned position sequence also binds representation to token ordering and padding conventions.

The decoder predicts each patch from its contextual token. Global attention does not by itself remove patch-boundary artifacts or supply dense high-frequency geometry. With eight training episodes, comparing randomly initialized transformers to CNNs is a study of inductive bias under scarce supervision, not a reproduction of large pretrained ViT results.

DINOv2 demonstrates transferable image and pixel features through large-scale curated self-supervised learning and distillation. [Oquab et al., *DINOv2*, 2023](https://arxiv.org/abs/2304.07193) **Project implication:** a frozen DINOv2 representation would answer a different question from the original scratch student. Keep transfer-learning and random-initialization results explicitly separated.

### What RGB stump boosting actually measures

Friedman formulates boosting as stagewise optimization in function space, fitting weak learners to negative loss gradients. [Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)

The [local boosting implementation](../../../track3/features/learning_experiments/boosting.py) uses RGB features and decision stumps. Its structure can be written

\[
F_M(x)=b+\nu\sum_{m=1}^{M}h_m(x),\qquad
h_m(x)=a_m\mathbf1[x_{j_m}\le t_m]+c_m\mathbf1[x_{j_m}>t_m],
\]

with a log-depth target and exponentiation for depth output.

**Original deduction.** Since each stump reads a single color channel, the sum has an additive form \(b+g_R(R)+g_G(G)+g_B(B)\). It does not learn spatial context or general cross-channel interactions. This is a useful restricted baseline, not a verdict on arbitrary boosted trees, engineered geometric features, or learned embeddings. Repeatedly fitting similar colors from correlated frames also does not create independent evidence.

## Loss mathematics, support, and geometry

### Exact existing coursework objective

The [existing loss](../../../track3/features/scratch_depth/_network.py) defines

\[
V_i=\operatorname{bool}(M_i)\land\operatorname{finite}(T_i)\land(T_i>0),
\qquad
\mathcal L_{log1}=\frac{\sum_i V_i|\log\widehat D_i-\log T_i|}{\sum_iV_i}.
\]

It replaces invalid target values with one before logarithms or arithmetic. This order matters: multiplying an already-created NaN by zero does not repair it. A batch without usable labels raises an error. No per-image scale alignment appears in this objective. The positive head is \(\widehat D=\log(1+e^a)+10^{-3}\), so its lower bound is strictly above \(10^{-3}\) metres.

**Original analysis.** If \(\widehat D=qT\), its absolute log error is \(|\log q|\), symmetric for reciprocal multiplicative errors. For small depth error \(\delta\), \(\log(T+\delta)-\log T\approx\delta/T\): log error behaves approximately like relative error locally. It is not the same quantity as metre MAE. Near-zero predictions and targets require the documented positive support; silently applying a different epsilon changes the objective.

The reduction averages valid pixels over the batch. An image with more valid target pixels contributes more terms. This differs from averaging per-image losses and differs again from averaging per-episode scores. For the same reason, changing crop, mask, or synthetic geometry can change training exposure even when sample counts remain fixed.

### Scale invariance is a deliberate information tradeoff

Eigen and colleagues introduce scale-invariant log-depth error in a coarse-to-fine monocular architecture. [Eigen et al., *Depth Map Prediction from a Single Image Using a Multi-Scale Deep Network*, 2014](https://arxiv.org/abs/1406.2283)

**Original derivation.** Let \(r_i=\log\widehat D_i-\log T_i\). Then

\[
D_\lambda=\overline{r^2}-\lambda\bar r^2
=\operatorname{Var}(r)+(1-\lambda)\bar r^2.
\]

At \(\lambda=1\), multiplying all predictions by \(q\) adds \(\log q\) to all residuals and leaves the loss unchanged. At \(\lambda<1\), mean log bias remains penalized. “SILog” is insufficient specification because implementations vary in square roots, scale factors, masks, and \(\lambda\). A purely scale-invariant model can recover relative shape yet fail a metric obstacle-distance threshold.

MiDaS uses scale-and-shift-invariant disparity supervision to combine annotations with incompatible scale and shift; it also introduces robust alignment and multiscale residual-gradient regularization. [Ranftl et al., *Towards Robust Monocular Depth Estimation*, 2020](https://arxiv.org/abs/1907.01341)

**Original derivation.** Least-squares alignment solves

\[
(s,t)=\arg\min_{s,t}\sum_i(sq_i+t-y_i)^2,
\]

\[
\begin{bmatrix}\sum q_i^2&\sum q_i\\\sum q_i&n\end{bmatrix}
\begin{bmatrix}s\\t\end{bmatrix}
=\begin{bmatrix}\sum q_iy_i\\\sum y_i\end{bmatrix}.
\]

A constant prediction makes the system singular. Unconstrained algebra can produce a negative scale; physically appropriate sign restrictions need explicit treatment. Affine invariance in inverse depth is not affine invariance in depth. Using such alignment on metric teacher labels deliberately discards part of their information and would require prospective justification.

### Admitted residual correction and robust loss

The [admitted research network](../../../track3/features/scratch_depth/_research_network.py) accepts six channels for a centre frame or eighteen for three aligned context frames: RGB, log depth, validity, and producer confidence per frame. In residual mode,

\[
\widehat D=B\exp(a\tanh r),
\qquad
e^{-a}<\widehat D/B<e^a,
\]

where \(B\) is admitted base depth and the default correction bound is 0.5. This is multiplicative bounded correction, not an unconstrained new metric reconstruction. The direct mode uses softplus plus \(10^{-3}\). The image-only RGB model and these producer-conditioned variants have different information access and must be reported separately.

The research objective first intersects target and base support. Its optional robust term uses Huber loss on log depths with \(\delta=0.1\):

\[
H_\delta(r)=\begin{cases}\tfrac12r^2,&|r|\le\delta,\\
\delta(|r|-\tfrac12\delta),&|r|>\delta.
\end{cases}
\]

**Original analysis.** Huber transitions from a quadratic basin to a linearly growing tail. It can reduce leverage of extreme pseudo-target residuals compared with squared loss, but cannot distinguish an erroneous teacher label from a rare, important obstacle. Reducing loss does not establish that the right observations were downweighted.

### Boundary matching and valid adjacency

For horizontal and vertical neighbors \((i,j)\), the implemented optional boundary term averages

\[
|r_i-r_j|=| (\log\widehat D_i-\log\widehat D_j)
-(\log T_i-\log T_j)|
\]

only where both endpoints have admitted target/base support. Its contribution is weighted by 0.1. Horizontal and vertical sums are combined according to admitted pair counts.

**Original analysis.** This matches the gradient of prediction error. It is not simply total-variation smoothing \(|\nabla\widehat D|\), and it need not erase a true target boundary. Conversely it can preserve a false teacher boundary. “Boundary loss” here does not imply independently labeled obstacle boundaries or semantic edges. At a masked hole, a difference must not use the invalid endpoint. A future multiscale version would also need validity-aware downsampling so invalid labels do not leak into coarse targets.

### Surface normals and camera conventions

Yin and colleagues use normals derived from sampled 3D point triples as higher-order depth constraints. Their virtual normals are distinct from this repository's local-neighborhood construction. [Yin et al., *Enforcing Geometric Constraints of Virtual Normal for Depth Prediction*, 2019](https://arxiv.org/abs/1907.12209)

**Implementation and original derivation.** For pixel \(p=(u,v,1)^\top\), optical-axis depth back-projects as \(P=D K^{-1}p\). At a local origin,

\[
v_x=P_{u+1,v}-P_{u,v},\quad v_y=P_{u,v+1}-P_{u,v},\quad
n=\frac{v_x\times v_y}{\|v_x\times v_y\|}.
\]

The optional implementation compares \(1-\widehat n^\top n_T\), clamps it to \([0,2]\), and adds it with weight 0.1. Support requires the three participating pixels, producer normal support, and nondegenerate predicted and target cross products. Normalization uses epsilon \(10^{-8}\); squared cross-product magnitude must exceed \(10^{-16}\).

The producer must exclude instance boundaries from normal support: points on two different surfaces do not define the local tangent plane of either surface. The loss consumer's mask intersection is not a replacement for that producer responsibility. Incorrect intrinsics, a wrong pixel convention, or unadjusted cropping can systematically bias normals even with plausible-looking depth. For an image-plane transform \(A\), intrinsics should follow the corresponding \(K'=AK\) convention. Optical-axis depth must not be confused with Euclidean range along a unit ray.

### Temporal consistency requires motion and visibility

The optional temporal objective compares centre prediction log depth with producer-aligned temporal depth, using Huber loss with \(\delta=0.1\), temporal support intersected with base validity, and weight 0.1. Alignment must already express depth in the centre camera's optical grid and optical-axis coordinate.

**Original geometric analysis.** A source point transforms as

\[
P_t=R_{t\leftarrow s}(D_sK_s^{-1}p_s)+t_{t\leftarrow s},\qquad
p_t\sim K_tP_t,\qquad D_{s\to t}=P_{t,z}.
\]

Static-world camera motion is insufficient for independently moving objects. Occlusions require visibility handling, typically including depth ordering; disocclusions have no valid source observation. Comparing the same pixel across frames without warping penalizes legitimate camera motion. Comparing warped Euclidean ranges against optical-axis depth is also wrong. Temporal masks must be inference-available producer outputs, not target-derived visibility that would disappear at deployment.

Temporal agreement can still reflect shared teacher bias or correlated warped estimates. It is useful evidence about consistency under the alignment assumptions, not an independent certificate of physical accuracy. Source timestamps, pose direction, units, producer version, and all support counts should remain attached to the loss record.

## Synthetic data: causal meaning, exposure, and provenance

Tobin and colleagues randomize simulator appearance to transfer from simulated RGB to real object localization and manipulation. The paper supports broad rendering variation as one possible transfer mechanism; it does not prescribe a universal synthetic ratio for monocular depth. [Tobin et al., *Domain Randomization*, 2017](https://arxiv.org/abs/1703.06907)

SceneNet RGB-D supplies large-scale rendered indoor trajectories with geometry and semantic annotations, randomized scene layouts, objects, lighting, textures, and camera trajectories. [McCormac et al., *SceneNet RGB-D*, 2017](https://arxiv.org/abs/1612.05079) Virtual KITTI 2 provides controlled variants of cloned driving sequences with depth, motion, segmentation, and camera information. [Cabon et al., *Virtual KITTI 2*, 2020](https://arxiv.org/abs/2001.10773)

### Why perfect synthetic labels can still produce a biased learner

**Original causal analysis.** Consider geometry \(G\), appearance \(A\), camera \(K\), image formation \(I=R(G,A,K)\), and target depth \(D=Z(G,K)\). A procedural generator may accidentally associate one texture with one depth or one primitive with one distance. A learner can exploit \(A\to D\) correlations that disappear in real video. Randomizing appearance independently of geometry can weaken these shortcuts, but unrealistic geometry or camera distributions can still miss the real task's support.

Pixel-perfect labels mean exactness relative to the generator's geometry and conventions. They do not mean the joint distribution \(P_s(I,D)\) matches \(P_r(I,D)\). A 128-image analytic pilot is not equivalent to millions of photorealistic rendered trajectories, and its value must be established through real tuning evidence.

Preserve provenance fields sufficient to reconstruct the scientific claim: generator revision, random seed, geometry family, material/texture rules, illumination, camera pose and intrinsics, depth representation and units, visibility, invalid-pixel policy, split, and parent scene identifier. Re-rendered views of one procedural scene share a parent and should not be counted as independent scenes.

### Mixture mathematics

A prospective objective is

\[
\mathcal L=(1-\alpha)\mathbb E_r[\ell(f(I),T)]+\alpha\mathbb E_s[\ell(f(I),D)].
\]

**Original analysis.** File counts do not determine \(\alpha\). Sampling frequency, valid pixels, loss reduction, source weights, epochs, and repetition all affect exposure. With source draw probability \(p_s\), sample loss weight \(w_s\), and approximately \(m_s\) admitted pixels per synthetic sample, pooled pixel reduction gives an approximate relative contribution

\[
\alpha_{effective}\approx
\frac{p_sw_sm_s}{p_sw_sm_s+(1-p_s)w_rm_r}.
\]

This approximation assumes comparable gradient scales; differing residual distributions can make actual optimization influence differ further. Track source-specific image draws, unique parents, valid pixels, loss magnitudes, and gradient-related diagnostics if an independently authorized future run collects them.

A joint mixture, synthetic pretraining followed by real-only training, and a curriculum are distinct policies even with equal total image counts. Selection among them must occur using real tuning groups under a predefined budget. Do not choose a ratio by which option retrospectively explains holdout 43/44. Synthetic examples remain training-only; synthetic evaluation is at most a generator-specific diagnostic.

## Ensembles: aggregation space, dependence, and uncertainty

Deep ensembles combine separately trained predictive models. For Gaussian member outputs, their moment-matched total variance is

\[
\bar\mu=M^{-1}\sum_m\mu_m,\qquad
v=M^{-1}\sum_m(\sigma_m^2+\mu_m^2)-\bar\mu^2.
\]

[Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*, 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)

### Arithmetic, geometric, and median aggregation answer different questions

**Original analysis.** An arithmetic depth mean is \(\widehat D_A=\sum_mw_mD_m\). A log-space mean yields the weighted geometric mean

\[
\widehat D_G=\exp(\sum_mw_m\log D_m)=\prod_mD_m^{w_m},\qquad
w_m\ge0,\quad\sum_mw_m=1.
\]

The arithmetic mean naturally corresponds to minimizing expected squared metre error; the geometric mean corresponds to minimizing squared log discrepancy among members. A median is robust to a minority of extreme member values and minimizes summed absolute discrepancy. For a uniquely defined odd-sample median, monotonicity of log preserves the median under exponentiation. For even member counts, software that averages the two central values gives different metre-space and log-space medians.

These are aggregation identities, not evidence that one method improves the task. Mean, median, inverse-tuning-error weights, and learned weights all constitute model choices. They require tuning evidence and must be frozen before independent assessment. Sparse validity also requires an explicit rule for member admission and renormalization; silently dropping a failing member changes the effective predictor.

### Correlation limits the benefit of more members

For equal-variance errors with pairwise correlation \(\rho\),

\[
\operatorname{Var}(\bar\epsilon)=\sigma^2\left[\rho+\frac{1-\rho}{M}\right].
\]

For arbitrary weights and covariance \(\Sigma\), variance is \(w^\top\Sigma w\). Shared teacher labels, frame selection, architecture, and initialization ancestry can create substantial dependence. FP32 and BF16 executions of related training recipes are not automatically independent epistemic samples. Neither are checkpoints along one training trajectory or augmented children of the same parent frame.

Low ensemble disagreement can coexist with a shared false depth estimate. High disagreement can identify sensitivity but does not quantify a calibrated physical error probability by itself. Report member provenance and parent counts, not just ensemble size.

### Heteroscedastic regression

Kendall and Gal distinguish aleatoric from epistemic uncertainty and use input-dependent variance in vision regression. With \(s=\log\sigma^2\), Gaussian negative log likelihood has the form

\[
\ell=\tfrac12e^{-s}(y-\mu)^2+\tfrac12s
\]

up to a constant. [Kendall and Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, 2017](https://arxiv.org/abs/1703.04977)

**Original project analysis.** The variance term downweights difficult residuals while its logarithmic penalty discourages unbounded inflation. With pseudo-target supervision, it describes uncertainty under the teacher-label model; it cannot automatically separate teacher bias, true ambiguity, alignment error, and physical sensor noise. A model can improve likelihood while worsening its mean prediction through variance adjustment. Report mean error, sharpness, and coverage separately.

### Calibration and coverage must name their sampling unit

Conformalized quantile regression combines quantile models with calibration to obtain finite-sample marginal coverage under exchangeability. [Romano et al., *Conformalized Quantile Regression*, 2019](https://arxiv.org/abs/1905.03222)

**Original derivation and project implication.** For fixed prediction function and calibration scores \(s_j=|y_j-\widehat y_j|\), a split-conformal radius uses the order statistic with index \(\lceil(n+1)(1-\alpha)\rceil\), subject to the method's finite-sample convention. Millions of correlated pixels do not furnish millions of exchangeable episode samples. Define whether the intended event covers a random pixel, a whole frame, an episode summary, or all pixels simultaneously. These are different probability statements.

Coverage against teacher labels is teacher-label coverage. Global marginal coverage is not conditional coverage for every distance, material, lighting condition, or obstacle category. A small number of independent groups gives coarse quantiles and wide uncertainty. Reserve calibration groups independently from model and ensemble selection in any future protocol.

## Grouped selection, leakage, and a fresh assessment

Cawley and Talbot show that optimizing a finite-sample selection criterion can overfit that criterion, producing substantial subsequent selection bias. Model selection must be treated as part of the fitting procedure. [Cawley and Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

**Project protocol proposal.** A scientifically independent future comparison would freeze the full pipeline: target generation, masks, normalization, architecture families, optimizer ranges, stopping policy, synthetic candidate ratios, ensemble candidates, metrics, and resource budget. Inner episode groups select among those choices; outer groups assess the entire selection procedure. Any final independent dataset is accessed only after that protocol and selected recipe are fixed. This is a prospective design, not a proposal to recycle the consumed holdout.

Leakage can enter without direct training on holdout targets. Examples include selecting augmentation after observing failure cases, fitting RGB normalization on all episodes, choosing synthetic geometry to explain holdout mistakes, selecting an ensemble on holdout, and using target-derived masks at inference. Even unpublished manual decisions are part of selection.

For episode metric \(E_e\), a macro average is \(|\mathcal E|^{-1}\sum_eE_e\). A pooled pixel metric instead weights episodes by admitted pixel counts. Both can be meaningful but answer different questions. Report them with clear definitions. For paired model comparisons, compute episode-level differences before summarizing; a large improvement on one heavily sampled episode should not silently stand for consistent improvement across environments.

An episode bootstrap, if a future protocol authorizes it, resamples independent episodes rather than individual pixels. With very few episodes it remains unstable. Statistical significance does not repair a biased split or an adaptively reused holdout.

## Optimizers, seeds, and fair research budgets

Adam maintains bias-corrected exponentially weighted gradient and squared-gradient moments. [Kingma and Ba, *Adam*, 2015](https://arxiv.org/abs/1412.6980) AdamW decouples weight decay from adaptive gradient scaling; L2 regularization and weight decay are not equivalent under Adam. [Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, 2019](https://arxiv.org/abs/1711.05101)

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
\theta_{t+1}=(1-\eta_t\lambda)\theta_t-
\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\]

**Implementation distinction.** The coursework [training operation](../../../track3/features/scratch_depth/train.py) uses `torch.optim.Adam` with `weight_decay=request.weight_decay`: this is the coupled L2 form, not AdamW. Other research neural training can use AdamW. The equations above explain the AdamW alternative and do not redefine the coursework recipe.

**Original analysis.** Optimizer names are insufficient experiment specifications. Include learning-rate schedule, decay, parameter exclusions, accumulation, batch size, number of updates, clipping, precision, and checkpoint selection. Switching Adam to AdamW changes the meaning of the decay hyperparameter. No optimizer can manufacture scene diversity absent from the training set.

Dodge and colleagues show why selected performance should be interpreted together with hyperparameter-search budget, proposing expected best validation performance as a function of computational budget. [Dodge et al., *Show Your Work*, 2019](https://arxiv.org/abs/1909.03004)

**Project proposal.** Predeclare seed sets and search budgets. Distinguish configuration selection from seed averaging: choosing the best seed spends selection budget, whereas reporting all predefined seeds measures optimization variability. A family given many more trials has more opportunities to exploit tuning noise. Keep failed or interrupted trials in the ledger, and compare wall time or update budgets alongside trial counts when architectures differ substantially.

Use paired seeds where appropriate, but do not interpret them as paired independent scenes. Teacher initialization, scene generation, sampling, data order, dropout, and student initialization require distinct reproducibility records. Parent-frame and parent-scene identifiers prevent augmentation counts from masquerading as data diversity.

Mixed-precision training uses lower-precision computation with methods such as higher-precision accumulation and loss scaling to manage numerical limitations. [Micikevicius et al., *Mixed Precision Training*, 2018](https://arxiv.org/abs/1710.03740) **Project implication:** loss logarithms, variance reductions, normal normalization, and small support counts warrant deliberate precision choices. Numerical differences can create optimization diversity but do not independently establish better accuracy or calibrated uncertainty. Generic compiler and attention-system details belong to the companion systems discussion rather than altering this learning protocol.

## Prospective research decisions supported by this synthesis

The literature supports a disciplined sequence of questions, not a retrospective replacement recipe:

- Does multiscale context improve real tuning behavior beyond pointwise color and shallow local baselines, under a fixed budget?
- Does a normalization or feature-residual change improve generalization without increasing selection freedom unfairly?
- Is the task metric or shape-relative, and which loss invariances preserve the information actually required downstream?
- Do boundary, normal, and temporal objectives have trustworthy producer support and correct camera physics?
- Does synthetic enrichment help real tuning episodes, and is its effective exposure documented independently of sample counts?
- Do ensemble members exhibit complementary real tuning errors, rather than simply different names or numerical regimes?
- Can any uncertainty statement be calibrated at the appropriate independent sampling unit using labels that justify its claimed meaning?
- What independent future evidence can assess the complete frozen procedure without consulting already consumed holdout outcomes?

The immediate documentary action is to retain these questions and the complete provenance chain. Changing architecture, synthetic ratio, or ensemble on the basis of the completed campaign's holdout would answer a different, adaptively selected question.

## Primary bibliography

| Work | Main role in this chapter |
|---|---|
| [Hinton, Vinyals, and Dean, 2015](https://arxiv.org/abs/1503.02531) | Knowledge distillation and teacher/student distinction |
| [Zhang et al., 2017](https://arxiv.org/abs/1611.03530) | Training fit does not establish generalization |
| [Ronneberger, Fischer, and Brox, 2015](https://arxiv.org/abs/1505.04597) | Encoder-decoder skips and localization |
| [He et al., 2016](https://arxiv.org/abs/1512.03385) | Residual feature learning |
| [Wu and He, 2018](https://arxiv.org/abs/1803.08494) | Batch-independent group normalization |
| [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929) | Patch transformer foundation |
| [Oquab et al., 2023](https://arxiv.org/abs/2304.07193) | Large-scale pretrained visual features |
| [Friedman, 2001](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) | Functional-gradient boosting |
| [Eigen, Puhrsch, and Fergus, 2014](https://arxiv.org/abs/1406.2283) | Scale-invariant depth error |
| [Ranftl et al., 2020](https://arxiv.org/abs/1907.01341) | Mixed-dataset invariances and gradient regularization |
| [Yin et al., 2019](https://arxiv.org/abs/1907.12209) | Higher-order geometric normal constraints |
| [Tobin et al., 2017](https://arxiv.org/abs/1703.06907) | Domain randomization |
| [McCormac et al., 2017](https://arxiv.org/abs/1612.05079) | SceneNet RGB-D |
| [Cabon, Murray, and Humenberger, 2020](https://arxiv.org/abs/2001.10773) | Virtual KITTI 2 |
| [Lakshminarayanan, Pritzel, and Blundell, 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Predictive deep ensembles |
| [Kendall and Gal, 2017](https://arxiv.org/abs/1703.04977) | Heteroscedastic vision regression |
| [Romano, Patterson, and Candès, 2019](https://arxiv.org/abs/1905.03222) | Conformalized quantile coverage |
| [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html) | Model-selection overfitting |
| [Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980) | Adam |
| [Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101) | AdamW |
| [Dodge et al., 2019](https://arxiv.org/abs/1909.03004) | Search-budget reporting |
| [Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740) | Mixed-precision methodology |
