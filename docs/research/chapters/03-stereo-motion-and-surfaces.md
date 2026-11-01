# Calibrated stereo, camera motion, and reconstructable surfaces

## Purpose and evidence boundaries

This chapter connects stereo correspondence, metric scale, camera estimation, moving-object tracking, and surface reconstruction to the project's egocentric video-to-policy task. Its central question is not whether a method creates a plausible three-dimensional visualization. It is whether the resulting coordinates support defensible object trajectories, relative motion, contact locations, and simulation geometry.

Paper descriptions below report verified primary literature. Sections marked **Original derivation** explain mathematics developed here from stated assumptions. Sections marked **Proposed adaptation** describe research designs rather than implemented or established project behavior. The selected papers connect correspondence, metric scale, dynamic geometry and reconstructable surfaces.

Repository connections are the [stereo slice](../../../track3/features/stereo_depth/README.md), [camera-motion slice](../../../track3/features/camera_motion/README.md), [object-pose slice](../../../track3/features/object_pose/README.md), [scene assembly](../../../track3/features/scene_assembly/README.md), and [error budget](../../design/error-budget.md). Their documented implementation and historical evidence take precedence over a proposed architecture in this chapter. The full task remains unfinished; opening-window completion is not full-clip reliability. The research document changes no runtime authorization, model recipe, dependency, or production default. No models, notebooks, compilers, tests, checks, or benchmarks were run to produce it.

Episode 41 remains excluded. Already consumed outcomes from episodes 43/44 must not guide recipe changes, synthetic mixtures, thresholds, or method selection. Future unbiased claims require an independently fixed protocol. Historic repository statements referring to older measurements are not permission to reopen excluded observations.

## The metric camera contract

### Calibration as an observed ruler

Zhang estimates camera intrinsics from multiple orientations of a planar pattern, models radial distortion, and refines a closed-form solution with nonlinear maximum-likelihood optimization. The project's calibration grid is relevant because its known physical spacing can supply metric information independently of object identity. **Paper locator:** the method and nonlinear-refinement discussion in [Zhang, *A Flexible New Technique for Camera Calibration*, 2000](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/); the author's [technical report](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) provides the full derivation and appendices.

**Original derivation.** A calibration is more than a focal length. The useful contract binds image size, lens-distortion convention, optical frame, intrinsics, stereo rigid transform, exposure timing, and preprocessing. A physical baseline measured in millimetres and depth interpreted in metres require an explicit conversion. A scale error in the baseline propagates directly into all stereo depth, even if pixel matching is perfect.

With the convention that \(T_{ab}\) maps frame \(b\) into frame \(a\), projection is \(\tilde u\sim K[I_3\mid0]T_{cw}\widetilde X_w\), where \(K\) is \(3\times3\), \(T_{cw}\) is \(4\times4\), and \(\widetilde X_w=[X_w^T,1]^T\). Throughout the derivations below, the shorthand \(TX\) for a Euclidean point denotes the affine action \(RX+t=[I_3\mid0]T\widetilde X\), not literal multiplication of a \(4\times4\) matrix by a three-vector. Foundational pinhole and rigid-transform conventions belong in the foundations chapter; two operational consequences matter here:

- Optical depth \(Z\) and Euclidean range differ by the ray norm: for \(q=K^{-1}[u,v,1]^T\), \(X_c=Zq\) and \(r=Z\|q\|\).
- If preprocessing maps pixels through \(u'=Au\), its effective intrinsics are \(K'=AK\). A crop changes the principal point; a resize changes focal length and principal point; rectification defines a new virtual camera.

These are artifact requirements, not optional explanatory metadata. A depth array on the wrong grid can look reasonable while back-projecting every point incorrectly. Half-pixel resize conventions must be retained when they are material at the working resolution. For pixel-center scaling, \(c_x'=s_x(c_x+1/2)-1/2\); adding a crop or padding offset changes that expression again.

### Epipolar geometry before rectification

**Original derivation.** Let the right camera coordinate of a left-camera point be

\[
X_R=RX_L+t.
\]

Normalized image rays \(x_L=K_L^{-1}\tilde u_L\), \(x_R=K_R^{-1}\tilde u_R\) and the baseline vector are coplanar. Their scalar triple product vanishes:

\[
x_R^T[t]_\times R x_L=0,
\qquad E=[t]_\times R,
\qquad F=K_R^{-T}EK_L^{-1}.
\]

Thus \(\tilde u_R^TF\tilde u_L=0\) constrains the second observation to a line, reducing correspondence search from two dimensions to one in the ideal calibrated model. This does not identify the correct point on that line. Repetitive texture, lack of texture, transparency, and occlusion remain unresolved.

Rectification rotates both optical coordinate systems into virtual views with parallel epipolar lines. Under the conventional horizontal arrangement, corresponding points share a row. Each image is first undistorted and then resampled through its rectification mapping. Interpolation changes pixel values and introduces invalid borders. A matcher operating on the rectified images must use the rectified projection matrices, not the original image intrinsics.

An observed vertical discrepancy between matched features is a diagnostic, not a complete calibration diagnosis. Possible causes include extrinsic error, imperfect distortion modeling, exposure offset, rolling shutter, feature localization, or incorrect correspondence. A learned matcher tolerating vertical error may improve output while leaving the physical calibration error intact. It should not silently redefine the meaning of the saved projection matrices.

### Triangulation and disparity geometry

**Original derivation.** For rectified cameras sharing focal length \(f\) and separated by baseline \(B>0\),

\[
u_L=fX/Z+c_L,\qquad
u_R=f(X-B)/Z+c_R.
\]

Writing \(d=u_L-u_R\) gives

\[
d_e=d-(c_L-c_R),\qquad Z=\frac{fB}{d_e}.
\]

The familiar \(Z=fB/d\) requires equal rectified principal points or a disparity already corrected for their difference. Sign conventions are consequential: exchanging left and right changes disparity sign and the appropriate baseline direction. A positive-depth gate cannot repair an incorrectly defined transform by itself.

If images shrink by \(s\), disparity and focal length both shrink by \(s\), preserving depth. Upsampling a low-resolution disparity without scaling its magnitude changes metric scale. Interpolating disparity and interpolating depth also differ because inversion is nonlinear:

\[
\frac{fB}{\mathbb E[d]}\ne\mathbb E\!\left[\frac{fB}{d}\right].
\]

This matters near boundaries where interpolation can blend foreground and background disparities into a nonexistent surface.

For unrectified observations, a linear triangulation initializer can stack cross-product equations \(\tilde u\times PX=0\) for each view and solve the homogeneous system. A subsequent estimate should minimize appropriately weighted reprojection error. A low reprojection residual is necessary but not sufficient: a near-parallel pair of rays can meet with tiny image residual while leaving depth poorly determined.

### Uncertainty, synchronization, and observability

**Original derivation.** With \(Z=fB/d_e\),

\[
\frac{\partial Z}{\partial d_e}=-\frac{fB}{d_e^2},
\qquad
\sigma_Z\approx\frac{Z^2}{fB}\sigma_{d_e}.
\]

Under independent small errors,

\[
\left(\frac{\sigma_Z}{Z}\right)^2\approx
\left(\frac{\sigma_f}{f}\right)^2+
\left(\frac{\sigma_B}{B}\right)^2+
\left(\frac{\sigma_{d_e}}{d_e}\right)^2.
\]

For correlated parameters, use \(J\Sigma J^T\). Near-zero disparity makes the first-order approximation unstable; a positivity-constrained or interval description may be more useful. If disparity uncertainty is \([d_- ,d_+]\) with \(0<d_-\), then

\[
Z\in[fB/d_+,fB/d_-].
\]

The interval is asymmetric. Clipping uncertain disparities to a positive minimum produces a finite number but does not establish bounded physical depth.

Timing error can masquerade as disparity. If corresponding image motion is \(\dot u\) and the inter-camera offset is \(\Delta t\), a local approximation is \(\delta d\approx\dot u\Delta t\). The resulting depth error grows approximately as \(Z^2\delta d/(fB)\). Static camera calibration does not remove this effect. Per-row rolling-shutter timing further means one rigid inter-camera transform may not describe all pixels at their exposure times.

**Proposed adaptation.** Future permitted investigations should separate geometric admission, matching ambiguity, and timing diagnostics. A narrow admitted vertical-residual distribution is conditional on its gate; it cannot demonstrate that rejected tails disappeared. This distinction matches the stereo slice's documented pre-admission distributions and rejection counts. New learned matchers should preserve the same camera/grid/provenance contract before their output is compared.

## Stereo model architectures and the evidence they exploit

### Semi-global matching

SGM combines a pixelwise matching cost with an approximation of a global disparity-smoothness objective. Its original mutual-information cost addresses radiometric differences. Multiple one-dimensional path recurrences aggregate evidence, allowing a practical compromise between strictly local matching and expensive global optimization. **Paper locator:** cost, energy, and pathwise aggregation in [Hirschmüller, *Stereo Processing by Semiglobal Matching and Mutual Information*, 2008](https://elib.dlr.de/55367/); the [2005 precursor](https://www.robotic.de/fileadmin/robotic/hirschmu/cvpr05hh.pdf) gives the original conference formulation.

**Original explanatory formulation.** A representative disparity energy is

\[
E(D)=\sum_pC(p,D_p)+\sum_{(p,q)}
\{P_1\mathbf1(|D_p-D_q|=1)+P_2\mathbf1(|D_p-D_q|>1)\}.
\]

For path direction \(r\), the recurrence compares continuation at \(d\), neighboring disparities with penalty \(P_1\), and an arbitrary prior disparity with penalty \(P_2\). Subtracting the previous path minimum prevents values from growing without changing the minimizer. Summing directional costs reduces directional bias but does not solve the full two-dimensional energy exactly.

For this project, SGBM is an existing transparent baseline. It exposes how texture, disparity range, smoothness, and row alignment affect support. It also creates an important selection problem: a low-error surviving subset may be mostly textured background and say little about manipulated object surfaces. Coverage and the geometry of missing regions must accompany residual statistics.

### RAFT-Stereo and iterative correspondence refinement

RAFT-Stereo uses rectified row-wise correlation and multilevel convolutional GRUs to iteratively update disparity. The recurrent mechanism spreads context across scales while repeatedly looking up matching evidence near its current estimate. **Paper locator:** architecture and update-operator discussion in [Lipson, Teed, and Deng, *RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching*, 2021 preprint](https://arxiv.org/abs/2109.07547).

**Original explanatory model.** Let feature vectors be \(f_L(v,u)\), \(f_R(v,u')\). A row-correlation volume stores \(C(v,u,u')=\langle f_L(v,u),f_R(v,u')\rangle\). A recurrent estimator can update

\[
h_{k+1}=\mathcal U(h_k,\operatorname{lookup}(C,u-d_k),c,d_k),
\qquad d_{k+1}=d_k+\Delta(h_{k+1}).
\]

The architecture repeatedly combines unary context and pairwise evidence rather than accepting a single initial match. Iteration count trades runtime against convergence behavior, but more iterations are not a guarantee of better physical geometry. An update can consistently reinforce a wrong repeated-pattern correspondence.

### IGEV-Stereo and encoded geometric context

IGEV-Stereo constructs a geometry encoding volume containing geometry/context information and local matching details. It obtains an initial disparity from that representation and indexes it during iterative refinement. The stated motivation is that raw all-pairs correlation lacks sufficient nonlocal geometric context for ambiguous regions. **Paper locator:** geometry encoding volume, combined volume, and iterative update sections in [Xu et al., *Iterative Geometry Encoding Volume for Stereo Matching*, 2023](https://arxiv.org/abs/2303.06615).

**Proposed adaptation.** The relevant comparison is not simply a leaderboard position. It is whether extra encoded context improves the particular missing-support patterns that matter to object reconstruction while preserving metric consistency. A method that fills a textureless object with a smooth incorrect plane can increase coverage and worsen contact geometry simultaneously.

### FoundationStereo and efficient descendants

FoundationStereo adapts frozen DepthAnythingV2 features with a CNN side branch. Its hybrid volume combines groupwise correlation and unary-feature concatenation. Attentive Hybrid Cost Filtering includes axial-planar convolutions and attention along disparity; an initial disparity is refined with GRUs using filtered-volume and correlation lookups. The paper introduces a curated one-million-pair synthetic dataset. **Paper locators:** §§3.1–3.5, hybrid-volume Eq. (1), initialization Eq. (2), refinement Eqs. (3)–(10), training Eq. (11) in the verified [v2 manuscript](https://arxiv.org/html/2501.09898v2). [Wen et al., *FoundationStereo: Zero-Shot Stereo Matching*, 2025](https://arxiv.org/abs/2501.09898).

Fast-FoundationStereo uses backbone distillation, blockwise architecture search for cost filtering, and structured pruning of refinement. Its authors also introduce automatically pseudo-labeled real stereo data. It is a relevant efficiency candidate, but published speed ratios do not establish RTX 5080 latency. **Paper locator:** the three-part acceleration design and pseudo-labeling pipeline in [Wen, Dewan, and Birchfield, *Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching*, 2025 preprint](https://arxiv.org/abs/2512.11130).

**Original architectural reasoning.** A volume with dimensions proportional to \(HWD\) becomes expensive when image resolution and disparity range grow together. Doubling both image dimensions and retaining the same physical near-depth range approximately doubles the required disparity range, increasing such volume size by roughly eightfold. Feature downsampling and separable filtering reduce cost but can affect thin structures and boundaries. Any practical resolution policy should therefore be recorded as a geometry decision, not merely a speed flag.

**Proposed adaptation.** A learned prior can help disambiguate an image region; it does not turn that region into independently observed geometry. Preserve a distinction between measured stereo support, prior-assisted estimate, inpainted estimate, and unknown. Confidence should combine independent indicators such as left-right consistency, matching ambiguity, occlusion status, and valid temporal reprojection. The current left-right residual-derived field is explicitly uncalibrated matching quality, not a probability or error bound.

## Metric anchoring without silently changing the current design

### Existing per-frame observation contract

The [object tracker](../../../track3/features/object_pose/README.md) documents a specific rule: **observed per-frame anchors supply metric monocular depth; valid direct stereo overrides those pixels; missing per-frame scale may use direct stereo but never an episode median or an interpolated scale**. The stereo slice reports an episode median as a summary. That summary is not authorization for a tracker fallback.

This distinction is essential. An episode summary can be useful for diagnosing drift while being inadmissible as the scale attached to an unsupported frame. The research literature does not override that implementation contract. A future global or temporally regularized calibration would be a separate design change requiring explicit provenance, selection rules, missingness semantics, and an independent assessment protocol.

### Robust scale estimators and their assumptions

**Original derivation.** Given relative optical depths \(m_i>0\) and trusted stereo depths \(z_i>0\) on admitted pixels of one frame, estimate a positive scale by

\[
\hat s=\arg\min_{s>0}\sum_iw_i\rho(sm_i-z_i).
\]

Weighted squared loss yields

\[
\hat s=\frac{\sum_iw_im_iz_i}{\sum_iw_im_i^2}.
\]

This estimator emphasizes large \(m_i\). Under multiplicative error, define \(\ell_i=\log z_i-\log m_i\) and estimate a robust location. Absolute log loss yields a weighted median; squared log loss yields a weighted mean; a Huber log-location interpolates the influence behavior. These are different statistical models, not interchangeable implementations of the same estimator.

The incumbent median of \(z_i/m_i\) should be described exactly as such. For positive ratios, applying log before or after a consistently defined median has closely related ordering behavior, but software's even-sample median interpolation can make the numerical values differ. Documentation should not relabel an incumbent estimator without inspecting its actual operation.

An inverse-depth prediction with an affine ambiguity requires a different model, \(z_i^{-1}=a q_i+b\). Use it only if justified by the model's documented output convention. Giving every model a free affine depth correction can hide shape distortion. A constant additive optical-depth shift does not correspond to a uniform Euclidean rescaling.

### Failure modes of apparently stable anchors

**Original reasoning.** Stable scale does not imply correct geometry. If both methods agree on one tabletop but disagree on object height, a stable background ratio can coexist with wrong object shape. If stereo uncertainty grows with distance while weights ignore it, distant noisy matches can shift the fit. If a repeated pattern produces coherent wrong disparity, robust statistics can preserve the wrong majority.

Anchor support therefore needs spatial coverage, depth-range coverage, mask provenance, and missingness. Excluding hands and objects does not prove the complement is static. Correspondence admission may favor textured regions, introducing selection bias. An uncertainty interval on the fitted scale should not imply uncertainty bounds on unobserved surfaces.

The scale anchor must remain independent of any reserved geometry residuals. Reusing the same pixels to fit scale and to claim improvement creates optimism even without label access. The stereo slice's deterministic support holdback is relevant, but it establishes independence only for the reserved evidence it actually separates. It does not make a consumed episode reusable for unrestricted recipe selection.

### Prospective temporal calibration is a design change

**Proposed adaptation; not current behavior.** One could model log-scale \(a_t\) using observed frame factors and a temporal prior,

\[
\min_{a_t}\sum_{t,i}w_{ti}\rho(a_t-\ell_{ti})+
\lambda\sum_t\left(\frac{a_{t+1}-a_t}{\Delta t}\right)^2.
\]

This trades local evidence against smoothness and fills otherwise unsupported values. That is exactly why it must not be introduced as an innocuous diagnostic. Its filled scales would be inferred rather than observed; abrupt true calibration changes or estimator failures could be smoothed away. A shared global scale is the limiting case of an even stronger assumption. Any future adoption needs explicit observed/inferred flags, influence limits, changed downstream contracts, and frozen development-only selection. No value from such a model is supplied by this documentation change.

## Camera motion and bundle adjustment

### Classical and learned systems

COLMAP's foundational SfM pipeline combines feature matching, geometric verification, initialization, incremental registration, triangulation, filtering, and bundle adjustment. Its sparse camera/landmark reconstruction is distinct from dense multiview stereo. **Paper locator:** §2's pipeline review and the proposed incremental-reconstruction components in [Schönberger and Frahm, *Structure-From-Motion Revisited*, 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html).

DROID-SLAM places recurrent learned updates around a dense bundle-adjustment layer jointly refining camera poses and pixelwise depth. Despite monocular training, its formulation can use stereo or RGB-D observations at inference. **Paper locator:** recurrent update architecture and Dense Bundle Adjustment in [Teed and Deng, *DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras*, 2021](https://arxiv.org/abs/2108.10869).

MegaSaM adds movement predictions, monocular-depth initialization/regularization, and uncertainty-aware global bundle adjustment for casual dynamic videos. Its optional consistent-depth stage fixes camera parameters while optimizing depth and uncertainty. It emphasizes insufficient parallax as an observability problem, rather than assuming optimization alone recovers depth. **Paper locators:** §3.2, local objective Eq. (9), Hessian approximation Eq. (10), §3.3 and Appendix A.3 in the [verified manuscript](https://arxiv.org/html/2412.04463v1). [Li et al., *MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos*, 2024 preprint/CVPR 2025](https://arxiv.org/abs/2412.04463).

DROID-SLAM in the Wild estimates per-pixel uncertainty from multiview feature inconsistency and uses differentiable uncertainty-aware bundle adjustment in dynamic environments. Its author-reported throughput is not an RTX 5080 measurement. **Paper locator:** uncertainty estimation and differentiable BA formulation; [Li et al., *DROID-SLAM in the Wild*, 2026](https://arxiv.org/abs/2603.19076). Its primary arXiv record identifies CVPR 2026.

### Objective, gauge, and residual normalization

**Original derivation.** For static landmarks \(X_j\) observed at image coordinates \(u_{ij}\),

\[
\min_{T_i,X_j,K}\sum_{ij}
\rho\!\left(e_{ij}^T\Sigma_{ij}^{-1}e_{ij}\right),
\qquad e_{ij}=u_{ij}-\pi(K_iT_iX_j).
\]

Calibrated stereo adds a second projection through the known baseline transform. An optical-depth factor can use \(e_z=(T_iX_j)_z-z_{ij}\), normalized by a justified measurement noise. Photometric, reprojection, metre-depth, and temporal residuals cannot simply be added with unit weights: they have different units, support counts, and noise structures.

Fixing one camera pose removes global rigid gauge. Monocular geometry retains scale gauge unless a physical factor removes it. A stereo baseline supplies that factor only if its units and transform are respected. Optimizing focal length freely alongside weak-parallax depth can introduce poorly constrained tradeoffs; already measured calibration should not automatically be relaxed because an image-only method allows it.

Robust loss limits individual residual influence. It does not guarantee the static hypothesis wins when most admitted features lie on a moving foreground. The definition of admitted support and the variable gauge are therefore as important as the optimizer.

### Schur complement and what the linear system reveals

**Original derivation.** Linearizing residuals gives \(e(\theta+\delta)\approx e+J\delta\). With robust weights fixed for one step, normal equations take the form

\[
\begin{bmatrix}H_{cc}&H_{cx}\\H_{xc}&H_{xx}\end{bmatrix}
\begin{bmatrix}\delta c\\\delta x\end{bmatrix}
=-\begin{bmatrix}g_c\\g_x\end{bmatrix}.
\]

Eliminating structure increments,

\[
\delta x=-H_{xx}^{-1}(g_x+H_{xc}\delta c),
\]

produces the reduced camera system

\[
(H_{cc}-H_{cx}H_{xx}^{-1}H_{xc})\delta c
=-g_c+H_{cx}H_{xx}^{-1}g_x.
\]

The Schur complement explains why local landmark structure is computationally useful: independent landmarks create small structure blocks even when the reduced camera graph is connected. Dense pixel-depth parameterization changes block sizes and sparsity, but the principle remains joint conditioning rather than independent pose and depth guesses.

If the system is rank deficient after gauge fixing, damping can produce a numerical step without adding observational information. An inverse damped Hessian is not automatically calibrated physical covariance. Approximate uncertainty must identify the noise model, fixed variables, robust weights, priors, and gauge. Small diagonal entries in a covariance approximation can miss coupled ambiguous directions.

Near-pure rotation illustrates this. For \(t=0\),

\[
u'=\pi(K'R[ZK^{-1}u])=\pi(K'RK^{-1}u),
\]

so image reprojection is independent of \(Z\). No amount of minimizing this reprojection objective recovers triangulation depth. A depth prior can select one explanation, but that selected depth is prior-dependent.

### Implications for the existing joint-refinement design

The [scene-refinement source design](../../../track3/features/scene_assembly/README.md) keeps original gauge, calibration, metric scale, stereo baseline, and mesh origins fixed. Its finite-difference Gauss–Newton implementation is not the same implementation as the differentiable BA layers in these papers. The literature informs factors and observability reasoning; it does not establish that the current solver has their accuracy or runtime.

The existing design also records rejected support and retains original candidates. Objective reduction is a fit statistic. It becomes evidence of downstream improvement only when independent observations and the relevant action-facing metrics support that claim. Contact residuals require actual supported physical-contact evidence; proximity alone is not an admissible contact constraint.

## Pointmap models and multiview geometry transformers

### DUSt3R and MASt3R

DUSt3R regresses dense pointmaps from image pairs with transformer encoders/decoders, without provided camera calibration or poses, then aligns pairwise predictions globally for larger collections. **Paper locator:** pointmap formulation, confidence-weighted regression, and global alignment in [Wang et al., *DUSt3R: Geometric 3D Vision Made Easy*, 2023 preprint/CVPR 2024](https://arxiv.org/abs/2312.14132).

MASt3R adds a dense local-descriptor head and matching objective to the pointmap framework, with fast reciprocal matching to reduce dense correspondence cost. **Paper locator:** dense matching head, matching loss, and reciprocal matching in [Leroy, Cabon, and Revaud, *Grounding Image Matching in 3D with MASt3R*, 2024](https://arxiv.org/abs/2406.09756); [ECCV primary PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09080.pdf).

**Original derivation.** A pointmap \(P_i(u)\in\mathbb R^3\) preserves a dense relationship between image pixels and predicted three-dimensional coordinates. For a physically consistent camera-frame map, \(P_i(u)=Z_i(u)K_i^{-1}\tilde u\). A freely regressed map can deviate from these camera rays. That freedom can improve learned reconstruction but requires an explicit projection-consistency diagnostic when the map is consumed as calibrated geometry.

For paired maps predicted in a shared pair frame, global alignment can seek transforms \(S_{ij}\) and common per-view geometry \(Q_i\), schematically

\[
\min_{Q,S}\sum_{(i,j),u}w_{iju}
\|Q_i(u)-S_{ij}P_i^{(ij)}(u)\|^2.
\]

This expression is explanatory rather than a transcription of a paper's exact optimizer. If \(S\) includes scale, a metric stereo constraint must fix the appropriate gauge before the output is used as metre geometry. Pairwise self-consistency is not independent metric accuracy.

### VGGT and the cost of joint attention

VGGT alternates within-frame and global self-attention over image tokens, with camera/register tokens and heads for cameras, depth, pointmaps, and tracks. It reports that unprojecting its depth with its camera estimates can outperform its direct pointmap branch. **Paper locators:** §3.2 Alternating-Attention, §3.3 prediction heads, and the discussion before §3.2 in [Wang et al., *VGGT: Visual Geometry Grounded Transformer*, 2025, verified v1](https://arxiv.org/html/2503.11651v1).

InfiniteVGGT introduces a causal architecture with a bounded adaptive KV cache and a training-free pruning mechanism, together with long continuous sequences for drift assessment. “Endless” processing should not be interpreted as a proof of bounded geometry error over arbitrary time. **Paper locator:** rolling-memory/pruning design and Long3D benchmark in [Yuan et al., *InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams*, 2026](https://arxiv.org/abs/2601.02281).

**Original complexity reasoning.** With \(F\) frames and \(P\) tokens per frame, naive all-frame attention has \((FP)^2\) pair interactions; independent frame attention has \(FP^2\). Memory-efficient attention can avoid materializing the full attention matrix, but it does not make the mathematical interaction count disappear. Cache pruning bounds retained context, introducing a policy for which past evidence survives.

**Proposed adaptation.** On one RTX 5080, bounded windows and cached intermediate artifacts are plausible research designs. Overlapping windows need explicit alignment and ownership rules. A similarity alignment can hide scale drift if compared only after rescaling each window. Retaining the fixed metric camera/stereo factors and reporting uncorrected inter-window discrepancies is necessary for an honest comparison. No published throughput or memory figure should be substituted for a local measured resource result.

### Geometry through time

MonST3R adapts pointmap prediction to per-timestep geometry using fine-tuning on dynamic, posed, depth-labeled video and video-specific downstream optimization. It avoids requiring every frame's geometry to be one static scene. **Paper locator:** dynamic pointmap adaptation and video optimization in [Zhang et al., *MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion*, 2024 preprint/ICLR 2025](https://arxiv.org/abs/2410.03825).

**Original reasoning.** Time-varying pointmaps are a representation of changing geometry, not persistent object identity. Two points at successive timesteps can be spatially close while belonging to different objects, or far apart while belonging to the same moving surface. Policy reconstruction needs that temporal association in addition to scene geometry. Appearance coherence, object rigidity, and contact hypotheses impose distinct constraints and should not be collapsed into one temporal smoothing term.

## Camera-object motion decomposition and canonical fusion

### Why foreground motion cannot determine the camera alone

**Original derivation.** An observed object pose is

\[
T_{co,t}=T_{cw,t}T_{wo,t}.
\]

Only the product is constrained by an isolated object's camera observations. For any chosen world-frame trajectory \(A_t\), replacing camera and object factors with compensating transforms can preserve their product. Static background, inertial information, calibrated multiview observations, or justified motion priors are needed to separate explanations.

Given depth at frame \(t\), the predicted static correspondence is

\[
\hat u_{t+1}=\pi\!\left(K_{t+1}T_{c_{t+1}c_t}
[Z_t(u)K_t^{-1}\tilde u]\right).
\]

Rigid flow is \(\hat u_{t+1}-u\). A residual against measured flow can indicate independent motion, but also wrong depth, wrong camera pose, occlusion, timing, or calibration. Visibility reasoning must precede interpretation. Reprojection outside the image, behind the camera, or behind a closer surface provides no direct static-consistency evidence.

The existing quiet/border opening guard's failure is a concrete reason not to equate semantic heuristics with static support. A border region can contain a moving foreground; a quiet-looking object can be camera-unstable; optical flow can be small under some motions while depth remains unobservable. The right abstraction is explicit supported static evidence, not a universally trusted image location.

### Object-frame reconstruction

**Original derivation.** A camera point maps into a canonical object frame as

\[
X_o=T_{wo,t}^{-1}T_{wc,t}X_c.
\]

Accumulating these canonical points removes both camera motion and the object's rigid world motion. Fusing directly in world coordinates smears a moving object. Fusing in camera coordinates mixes head motion into shape. An error in either transform blurs the canonical surface; the fusion system should not interpret repeated pose errors as evidence for a thicker object.

A rigid object model assumes one transform explains all its surface points. Articulation requires part transforms and joint structure; deformation requires another model. Human hands and soft objects cannot be made rigid by averaging their depth observations.

### ICP, ambiguity, and what the current tracker actually does

ICP alternates closest-point assignment and rigid registration, converging to a local minimum of its distance objective. Initialization and overlap remain critical. **Paper locator:** iterative closest-point algorithm and convergence discussion in [Besl and McKay, *A Method for Registration of 3-D Shapes*, 1992, primary paper PDF](https://www-evasion.imag.fr/Membres/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf).

**Original derivation.** A point-to-plane refinement can minimize

\[
\sum_iw_i\rho\big(n_i^T(Rp_i+t-q_i)\big)^2.
\]

On one plane, translations parallel to the plane and rotation about its normal are weak or absent constraints. A low residual on a small visible patch therefore does not establish a fully observed six-dimensional pose. Requiring more points from the same plane does not necessarily add the missing independent constraints. The information matrix's directions matter, not just the count of correspondences.

The current tracker uses local trimmed ICP with support, residual, confidence, and motion admission; recovery includes proper axis-permutation seeds and rejects unresolved competing hypotheses. Those seeds do not establish global optimality or a verified continuous symmetry model. Geometry-derived confidence is not calibrated probability. Current visibility support is not a rendered visible fraction.

The project's official metrics are **unsymmetrized** as documented in the object-pose slice. A geometrically indistinguishable rotation can still score as wrong. Do not claim symmetry-invariant scoring or verified symmetry resolution. Texture, handles, and persistent observations may resolve a rotation; otherwise retain the limitation. Prospective symmetry-aware internal hypotheses are a design choice and do not alter the official metric.

### Joint object tracking and reconstruction

BundleSDF couples online pose-graph optimization with a neural object field and a memory-frame pool for tracking and reconstructing an initially segmented unknown rigid object from RGB-D video. **Paper locators:** §3.1 coarse initialization, pose-graph optimization, and neural object field sections in [Wen et al., *BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects*, 2023](https://arxiv.org/abs/2303.14158); [CVPR primary PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf).

**Proposed adaptation.** The valuable architectural connection is feedback between canonical shape and pose, with observed frames retained as evidence. It is not permission to let a reconstruction explain away every inconsistent depth value. A joint system needs held-apart observations, occlusion handling, uncertainty, and a clear distinction between shape completion and measured surface. For scanned project objects, independently reconstructed geometry can be diagnostic without replacing the approved asset frame or measured origin.

## TSDF fusion and what a surface means

### Range fusion and dense tracking

Curless and Levoy integrate registered range data into a volumetric signed-distance representation and extract a surface from accumulated evidence. **Paper locator:** volumetric integration and weighted distance-function formulation in [*A Volumetric Method for Building Complex Models from Range Images*, 1996, author publication page](https://lightfield.stanford.edu/papers/volrange/).

KinectFusion combines dense depth tracking, volumetric fusion, and model prediction. It illustrates how an accumulated surface can support subsequent camera registration. Its depth-sensor and primarily static-scene setting must be reconsidered for dynamic egocentric video. **Paper locator:** system pipeline, tracking, integration, and raycasting in [Newcombe et al., *KinectFusion: Real-Time Dense Surface Mapping and Tracking*, 2011](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/); [primary PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf).

### Projective TSDF update and visibility

**Original derivation.** For a world voxel \(v\), compute \(x_c=T_{cw}v\) and \(u=\pi(Kx_c)\). A projective normalized truncated-distance observation is

\[
\phi(v)=\operatorname{clip}\left(
\frac{Z_{\mathrm{obs}}(u)-x_{c,z}}{\mu},-1,1\right).
\]

Within a justified visible integration band,

\[
F'(v)=\frac{W(v)F(v)+w(v)\phi(v)}{W(v)+w(v)},
\qquad W'(v)=\min(W(v)+w(v),W_{\max}).
\]

Positive values represent space in front of the measured surface under this sign convention. The quantity is projective distance along optical depth, not automatically exact Euclidean signed distance. For a fronto-parallel local surface the approximation is convenient; grazing geometry changes its relation to normal distance.

Weights might account for depth variance, view angle, segmentation uncertainty, and pose uncertainty. However, repeated frames are correlated. Treating one hundred nearly identical biased observations as one hundred independent measurements gives excessive certainty. Saturating weights limits domination but does not calibrate the resulting uncertainty.

Unknown space differs from observed free space. Behind the first observed surface, an RGB-D view does not reveal whether an object is solid, hollow, thin, or backed by another surface. Blindly integrating negative distances far behind the observation invents occupancy. Likewise, hole filling and mesh closing are geometric priors, not evidence that hidden surfaces were seen.

Pose uncertainty creates spatial uncertainty before depth fusion. A rotational error \(\delta\theta\) displaces a point approximately by \(\delta\theta\times X\); its magnitude grows with distance from the rotation center. Small angular camera drift can therefore blur a large object more than local depth noise. A fixed truncation band cannot distinguish all these error sources.

### Surface extraction and simulation requirements

**Original reasoning.** Extracting a zero isosurface turns a volumetric estimate into triangles. Triangle density is not measurement resolution: subdividing a coarse surface increases polygon count without adding evidence. A simulation asset additionally needs consistent orientation, usable topology, object body frame, units, and a deliberate collision representation.

The scene assembly slice already distinguishes visual and collision assets, treats contact as a sampled geometric proximity proxy, and refuses to invent a signed distance for invalid/open collision meshes. Those limitations should remain visible. A reconstructed mesh can be visually closed but still have the wrong thickness, concavity, mass distribution, or support face. These errors affect resting pose, contact response, and policy behavior differently from RGB rendering quality.

## Neural surfaces and appearance representations

### NeuS and signed-distance geometry

NeuS learns an SDF surface and appearance through a volume-rendering formulation designed to reduce surface-reconstruction bias. An eikonal term encourages unit-norm spatial gradients. **Paper locator:** SDF-based rendering construction, bias analysis, and optimization objective in [Wang et al., *NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction*, 2021](https://arxiv.org/abs/2106.10689); [NeurIPS primary PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/e41e164f7485ec4a28741a2d0ea41c74-Paper.pdf).

**Original explanatory derivation.** Let \(f_\theta(x)=0\) define the surface. A field satisfying the local signed-distance property has \(\|\nabla f_\theta(x)\|\approx1\), motivating

\[
L_{\mathrm{eik}}=\mathbb E_x(\|\nabla f_\theta(x)\|-1)^2.
\]

This term regularizes a field; it does not identify the correct hidden topology. Multiple surfaces may explain a finite image set similarly. RGB losses, masks, depth constraints, and normal constraints carry different information. If a learned appearance model can explain view-dependent effects, surface geometry may remain weakly constrained despite excellent image fit.

A scene-specific neural surface optimization also has different resource and evidence semantics from feed-forward depth inference. Its optimization images are fitting evidence. Any claim about improved geometry needs independent observations or measurements rather than only reconstruction loss on those same images.

### Gaussian splatting and why appearance is not collision geometry

3D Gaussian Splatting represents a radiance field using anisotropic Gaussians, jointly optimizes their properties and density, and renders them with visibility-aware splatting. Its principal goal is efficient high-quality novel-view synthesis. **Paper locator:** Gaussian representation, adaptive density control, differentiable rasterization, and optimization in [Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, 2023](https://arxiv.org/abs/2308.04079).

**Original derivation.** A Gaussian spatial kernel is

\[
G(x)=\exp[-\tfrac12(x-\mu)^T\Sigma^{-1}(x-\mu)],
\qquad \Sigma=RSS^TR^T.
\]

Projected splats contribute color through opacity/transmittance composition. Position, covariance, opacity, and view-dependent color can compensate for one another in image formation. Thus different spatial arrangements may explain similar rendered pixels. The Gaussian set does not inherently specify a unique watertight material boundary.

A collision engine needs a surface/occupancy interpretation and scale. Thresholding opacity, connecting centers, or extracting an auxiliary field adds such an interpretation; it is another modeling decision requiring geometric evidence. High PSNR, convincing novel views, or sharp textures do not certify contact normals or object thickness.

**Proposed adaptation.** Keep separate products for appearance, observed geometric support, reconstructed surface, and collision approximation. Their provenance may overlap, but they answer different questions. Appearance reconstruction could support human inspection or visualization without being allowed to overwrite the simulation mesh used for action learning.

## Research priorities and decision boundaries

The literature supports a calibrated metric backbone with learned context and explicit motion decomposition. It does not choose the project's next recipe automatically. A useful future research sequence would first freeze the evidence contract, then compare method families on permitted development data while retaining coverage, missingness, calibration, and computational cost.

The decisive questions are concrete:

- Does a stereo challenger improve independent metric geometry on the same fixed grid, including difficult surfaces, rather than only increasing admitted support?
- Does camera estimation retain static support and expose insufficient parallax, rather than absorbing foreground motion?
- Does object registration distinguish genuinely observed pose from geometric ambiguity and recovery guesses?
- Does fusion reduce independent surface error without smearing motion or inventing hidden structure?
- Do resulting trajectories and collision assets improve the actual downstream behavior under a separately fixed protocol?

On one RTX 5080, staged inference, bounded windows, cached features, sparse geometric refinement, and separate expensive research candidates are plausible resource strategies. Published training scale and benchmark speed do not establish local feasibility. New production dependencies still require confirmation. The currently implemented observed per-frame scale rule, unsymmetrized official metrics, excluded episodes, and immutable consumed holdout outcomes remain binding.

## Primary bibliography

| Paper | Year and identity | Primary source and useful locator |
|---|---|---|
| Zhengyou Zhang, **A Flexible New Technique for Camera Calibration** | 2000; DOI 10.1109/34.888718 | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/); planar homographies, intrinsic recovery, nonlinear refinement |
| Heiko Hirschmüller, **Stereo Processing by Semiglobal Matching and Mutual Information** | 2008; DOI 10.1109/TPAMI.2007.1166 | [DLR author archive](https://elib.dlr.de/55367/); matching cost, smoothness energy, path aggregation |
| Heiko Hirschmüller, **Accurate and Efficient Stereo Processing by Semi-Global Matching and Mutual Information** | 2005; CVPR precursor | [Author PDF](https://www.robotic.de/fileadmin/robotic/hirschmu/cvpr05hh.pdf); original SGM formulation |
| Lahav Lipson, Zachary Teed, Jia Deng, **RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching** | 2021 preprint; arXiv:2109.07547 | [arXiv](https://arxiv.org/abs/2109.07547); architecture and multilevel recurrent update |
| Gangwei Xu et al., **Iterative Geometry Encoding Volume for Stereo Matching** | 2023; arXiv:2303.06615 | [arXiv](https://arxiv.org/abs/2303.06615); geometry encoding volume and iterative refinement |
| Bowen Wen et al., **FoundationStereo: Zero-Shot Stereo Matching** | 2025; arXiv:2501.09898 | [arXiv](https://arxiv.org/abs/2501.09898), [verified v2 HTML](https://arxiv.org/html/2501.09898v2); §§3.1–3.5, Eqs. (1)–(11) |
| Bowen Wen, Shaurya Dewan, Stan Birchfield, **Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching** | 2025 preprint; arXiv:2512.11130 | [arXiv](https://arxiv.org/abs/2512.11130); distillation, architecture search, pruning |
| Johannes L. Schönberger, Jan-Michael Frahm, **Structure-From-Motion Revisited** | 2016; CVPR pp. 4104–4113 | [CVF primary publication](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html); §2 and reconstruction pipeline |
| Zachary Teed, Jia Deng, **DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras** | 2021; arXiv:2108.10869 | [arXiv](https://arxiv.org/abs/2108.10869); update operator and Dense Bundle Adjustment |
| Zhengqi Li et al., **MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos** | 2024 preprint; CVPR 2025; arXiv:2412.04463 | [arXiv](https://arxiv.org/abs/2412.04463), [verified v1 HTML](https://arxiv.org/html/2412.04463v1); §3.2 Eqs. (9)–(10), §3.3, Appendix A.3 |
| Moyang Li, Zihan Zhu, Marc Pollefeys, Daniel Barath, **DROID-SLAM in the Wild** | 2026; arXiv:2603.19076; primary record identifies CVPR 2026 | [arXiv](https://arxiv.org/abs/2603.19076); uncertainty and differentiable BA |
| Shuzhe Wang et al., **DUSt3R: Geometric 3D Vision Made Easy** | 2023 preprint; CVPR 2024; arXiv:2312.14132 | [arXiv](https://arxiv.org/abs/2312.14132); pointmaps, regression confidence, global alignment |
| Vincent Leroy, Yohann Cabon, Jérôme Revaud, **Grounding Image Matching in 3D with MASt3R** | 2024; arXiv:2406.09756 | [arXiv](https://arxiv.org/abs/2406.09756); descriptor head and reciprocal matching |
| Jianyuan Wang et al., **VGGT: Visual Geometry Grounded Transformer** | 2025; arXiv:2503.11651 | [arXiv](https://arxiv.org/abs/2503.11651), [verified v1 HTML](https://arxiv.org/html/2503.11651v1); §§3.2–3.3 |
| Junyi Zhang et al., **MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion** | 2024 preprint; ICLR 2025; arXiv:2410.03825 | [arXiv](https://arxiv.org/abs/2410.03825); dynamic pointmaps and downstream video optimization |
| Shuai Yuan et al., **InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams** | 2026; arXiv:2601.02281 | [arXiv](https://arxiv.org/abs/2601.02281); rolling KV memory and Long3D |
| Paul J. Besl, Neil D. McKay, **A Method for Registration of 3-D Shapes** | 1992; DOI 10.1109/34.121791 | [Primary paper PDF](https://www-evasion.imag.fr/Membres/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf); ICP and local convergence |
| Bowen Wen et al., **BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects** | 2023; arXiv:2303.14158 | [arXiv](https://arxiv.org/abs/2303.14158); §3 tracking/reconstruction coupling |
| Brian Curless, Marc Levoy, **A Volumetric Method for Building Complex Models from Range Images** | 1996; SIGGRAPH | [Author publication page](https://lightfield.stanford.edu/papers/volrange/); weighted signed-distance integration |
| Richard A. Newcombe et al., **KinectFusion: Real-Time Dense Surface Mapping and Tracking** | 2011; ISMAR | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-tracking/); tracking, integration, raycasting |
| Peng Wang et al., **NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction** | 2021; arXiv:2106.10689 | [arXiv](https://arxiv.org/abs/2106.10689); surface rendering and bias analysis |
| Bernhard Kerbl et al., **3D Gaussian Splatting for Real-Time Radiance Field Rendering** | 2023; arXiv:2308.04079 | [arXiv](https://arxiv.org/abs/2308.04079); Gaussian representation, density control, rasterization |
