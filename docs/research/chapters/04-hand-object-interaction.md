# Hand–object reconstruction, tracking, contact, and retargeting

## Purpose, evidence boundaries, and project context

This chapter studies how an egocentric video becomes a coherent sequence of anatomical hands, rigid objects, inferred interactions, and robot reference motions. Its central recommendation is to preserve the distinction between observations, learned geometric predictions, temporal predictions, and physical hypotheses at every interface. A plausible hand mesh is not a measurement of hidden fingertips; a persistent mask is not a material correspondence; proximity is not force; and a feasible robot joint configuration is not evidence that an object will follow the intended trajectory.

The literature cutoff is **2026-10-18**. Links point to primary papers, author project resources, or official release notes. Paper summaries are intentionally compact. Most of the mathematical development is an **independent derivation or project proposal**, explicitly labelled, rather than a reproduction of a particular paper's method. Reported benchmark results are not evidence of this repository's performance.

The project connections are the vertical slices for [object masks](../../../track3/features/object_masks/README.md), [hand motion](../../../track3/features/hand_motion/README.md), [object pose](../../../track3/features/object_pose/README.md), [object meshes](../../../track3/features/object_mesh/README.md), [retargeting](../../../track3/features/retargeting/README.md), and [public diagnostics](../../../track3/features/public_diagnostics/README.md). The chapter does not change their authorization gates, dependencies, or implementation status. In particular, WiLoR and MANO assets remain outstanding prerequisites; anonymous mask identities are not anatomical sides; the supplied scans and missing spoon have different geometry provenance; and the implemented contact proxy does not measure forces.

This research involved read-only source inspection and document editing. It did not execute tests, checks, models, notebooks, compilers, simulations, installations, or diagnostics. Episode 41 remains excluded. The completed campaign's held-out episodes 43 and 44 must not become recipe-selection evidence. Future design comparisons should use admissible training/tuning evidence and an independently fixed assessment protocol where a fresh unbiased assessment is required.

## What the video can identify

### A coupled state rather than independent predictions

**Independent formulation.** Let the reconstruction state be

\[
\mathcal X=\left\{\beta_h,s_o,\theta_{h,t},
{}^wT_{h,t},{}^wT_{o,t},{}^wT_{c,t},K_t,c_t\right\}_{t=1}^T.
\]

Here \(t\) indexes source frames, \(h\) identifies an anatomical hand, \(o\) identifies an object, \(\beta_h\) describes sequence-level hand shape, \(s_o\) is object scale, \(\theta_{h,t}\) is hand articulation, and \(c_t\) describes an interaction state. A transform \({}^aT_b\in SE(3)\) maps coordinates from frame \(b\) into frame \(a\). \(K_t\) is the pinhole camera intrinsic matrix in the actual processing pixel grid. Homogeneous column vectors are assumed throughout.

For canonical object point \(X_o\), the predicted pixel is

\[
u_t=\pi\!\left(K_t\left[R_{co,t}(s_oX_o)+p_{co,t}\right]\right),
\qquad
\pi([x,y,z]^\top)=[x/z,y/z]^\top.
\]

Multiplying both \(s_o\) and \(p_{co,t}\) by the same positive constant does not change this projection. Thus, an object can have an excellent projected silhouette while having incorrect metric size and depth. Contact cannot reliably repair this gauge freedom if hand dimensions and hand depth are also free to change.

Camera-relative and world-relative motion are related by

\[
{}^{c_t}T_{o,t}=({}^{w}T_{c,t})^{-1}{}^wT_{o,t}.
\]

Treating a camera-relative trajectory as world-relative introduces apparent object movement whenever the observer moves. The same distinction applies to hands. Contact between two camera-relative reconstructions can be assessed in their shared frame, but world velocities and gravity-sensitive reasoning require the camera transform as well.

### Observed, inferred, and unsupported quantities

**Project proposal.** A useful state record should distinguish:

| Quantity | Available evidence | Unsupported inference to avoid |
|---|---|---|
| Object mask | Visible image boundaries and model propagation | Complete hidden object geometry |
| Hand side | Anatomical classifier, thumb geometry, temporal identity | Mask index equals left/right |
| Articulation | Visible landmarks and image-conditioned pose prior | All FK joints are observed |
| Metric hand placement | Accepted surface-depth support and camera geometry | Detector confidence is metric confidence |
| Object pose | Correspondences, depth, silhouette, prior geometry | Small residual proves all six pose axes are observed |
| Contact hypothesis | Proximity, compatible geometry, temporal consistency | Measured pressure, friction, or grip force |
| Robot reference | Kinematic optimization and declared constraints | Dynamic execution success |

The [hand-motion contract](../../../track3/features/hand_motion/README.md) already separates inferred articulation, metric wrist support, anatomical parameter conversion, and world validity. Research additions should preserve that separation. A missing camera pose must not invalidate an otherwise useful camera-relative hand observation, and a valid camera-relative observation must not silently become world-valid.

## Detection, segmentation, and point tracking

### Open-vocabulary localization

**Paper finding.** Grounding DINO fuses image and language evidence in a feature enhancer, language-guided query selection, and a cross-modality decoder. It extends a DINO detector to text-conditioned object localization. Its output is text-associated detection evidence, not an interaction state. [Liu et al., Grounding DINO](https://arxiv.org/abs/2303.05499)

**Independent architectural interpretation.** A generic cross-attention operation is

\[
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}(QK^\top/\sqrt d)V.
\]

When queries represent candidate objects and keys/values represent image or language features, the result provides content-dependent evidence aggregation. The existence of attention does not establish grounded identity across frames: the query may select a different visually compatible object after occlusion. Prompt wording, distractors, and repeated object instances remain sources of ambiguity.

**Project proposal.** Use text localization to initialize or reacquire an instance, then preserve the instance's own identity and geometry evidence. Do not assume that repeatedly detecting the same noun phrase yields the same physical object. Reacquisition needs association using appearance, geometry, predicted location, and interaction context.

### SAM 2, SAM 3, and SAM 3.1

**Paper finding.** SAM 2 combines an image encoder, prompt-conditioned mask decoding, memory attention, and streaming memory. Prompts and earlier predictions condition video segmentation. [Ravi et al., SAM 2](https://arxiv.org/abs/2408.00714)

**Paper finding.** SAM 3 adds promptable concept segmentation using text phrases or exemplars. Its image detector and memory-based video tracker share a backbone. A presence head separates concept recognition from localization. [Carion et al., SAM 3](https://arxiv.org/abs/2511.16719)

**Verified release information.** SAM 3.1 was released on March 27, 2026. Object Multiplex groups objects into fixed-capacity buckets for shared processing, reducing redundant per-object computation. The official notes report about sevenfold speedup at 128 objects on one H100 relative to the November 2025 SAM 3 release. This is a conditional release result, not this project's measured runtime. [Official SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

**Project analysis.** Segmentation memory supports temporal consistency of appearance and boundaries, but an identity switch can remain visually coherent. Occlusion can also yield an empty visible mask even though the physical object continues to exist. A downstream tracker must distinguish visible support, propagated support, and absence of support.

Mask erosion can reduce boundary contamination when sampling depth or point queries. It also removes the very thin structures that matter for spoon handles or fingertips. Therefore erosion should be resolution-aware and optional for thin regions, with its sampling effect recorded. A mask-quality score calibrated on large objects need not describe the reliability of a narrow handle.

An instance index such as `hand_0` is a segmentation identity. Assigning it to anatomical left because it lies on the left side of the image is incorrect under crossing hands, mirrored images, camera roll, and a hand entering from the opposite side. The mapping from mask identity to anatomical identity must be explicit and may remain unresolved.

### TAPIR and CoTracker3

**Paper finding.** TAPIR first matches a query to candidate locations independently in each frame, then refines trajectories and query features using local correlations. [Doersch et al., TAPIR](https://arxiv.org/abs/2306.08637)

**Paper finding.** CoTracker3 combines a simplified joint point-tracking model with pseudo-label training on real videos. It provides online and offline variants and addresses the gap between synthetic tracking supervision and real video. [Karaev et al., CoTracker3](https://arxiv.org/abs/2410.11831)

**Independent geometric use.** For canonical surface point \(X_i\), image track \(u_{i,t}\), visibility weight \(v_{i,t}\), and uncertainty \(\Sigma_{i,t}\), define

\[
E_{\rm tracks}=\sum_{i,t}v_{i,t}
\rho\!\left(
\|u_{i,t}-\pi(K_t[R_{co,t}X_i+p_{co,t}])\|^2_{\Sigma_{i,t}^{-1}}
\right).
\]

The robust penalty \(\rho\) limits outlier influence. This formulation requires a persistent material point, not merely a similar pixel pattern. Specular highlights, moving shadows, object boundaries, and fingers crossing an object violate the rigid-point assumption.

**Project proposal.** Initialize object tracks away from hand overlap where possible. Admit a track as rigid evidence only while its appearance, mask membership, and rigid-motion residual remain compatible. Retain occluded predictions as hypotheses with declining support. An occlusion flag does not make a predicted pixel a direct observation. Record whether online or offline tracking produced each result because future-frame access changes latency and failure behavior.

## Parametric hands and reconstruction architectures

### MANO geometry and forward kinematics

**Paper finding.** MANO provides a learned low-dimensional hand representation with shape-dependent joints, pose corrective deformations, and linear blend skinning. Its parameters encode geometric priors rather than contact forces. [Romero, Tzionas and Black, MANO](https://is.mpg.de/ps/en/publications/embodiedhands)

The standard structure can be written

\[
T_P(\beta,\theta)=\bar T+B_S(\beta)+B_P(\theta),
\qquad J(\beta)=\mathcal J(\bar T+B_S(\beta)),
\]

\[
V_i=\sum_kw_{ik}G_k(\theta,J(\beta))
\begin{bmatrix}T_{P,i}\\1\end{bmatrix}.
\]

\(\bar T\) is the template, \(B_S\) the identity-dependent shape displacement, \(B_P\) the pose correction, \(G_k\) a rest-compensated bone transform, and \(w_{ik}\) the skinning weight. Global wrist placement is applied separately.

**Independent explanation.** In a kinematic tree, an articulated transform is a product along the root-to-joint path. The rest compensation ensures that zero articulation preserves the rest template. A skin vertex near a joint receives contributions from adjacent bones, giving a differentiable deformation instead of a rigid discontinuity. Corrective displacements compensate for deformations that simple weighted rigid transforms cannot represent well.

Pose and shape have different semantics. Changing pose rotates an articulated configuration; changing shape can alter finger lengths and palm dimensions. Allowing unconstrained per-frame shape creates a shortcut that explains occlusion or camera errors by changing the person's anatomy. Use one \(\beta_h\) per anatomical hand across a sequence, and record the evidence supporting any departure from that assumption.

The common 21-keypoint hand format is not 21 independently articulated bones. Fingertip landmarks may be selected mesh vertices appended to skeletal joints. Joint order, fingertip definition, wrist origin, units, and orientation frames must be part of the artifact contract. A 21-position array alone cannot recover all anatomical joint-frame rotations: rotation about a bone's axis can leave its endpoints unchanged.

### HaMeR

**Paper finding.** HaMeR uses a ViT-H backbone and a transformer decoder head. A query token cross-attends to image tokens to regress MANO and camera parameters. Training combines available parameter/joint supervision, 2D reprojection, and adversarial plausibility terms. Its paper writes camera projection as \(x=\Pi_K(X+t)\) with fixed intrinsics. Calling its paper's renderer strictly orthographic would be inaccurate. [Pavlakos et al., HaMeR](https://arxiv.org/abs/2312.05251)

**Project analysis.** A regressor resolves ambiguous images using learned pose and shape priors. That makes it useful for initialization and dangerous as a substitute for evidence. A hidden fingertip can be anatomically plausible but placed behind the wrong object surface. Likewise a reasonable crop-camera translation does not establish calibrated metric distance in the full video grid.

Use projected landmarks, accepted surface depth, visible silhouette support, and temporal consistency to assess placement. Retain whether a landmark is observed or inferred. A detector's probability should not be copied into every joint's visibility field because the detector and joint-visibility problems have different targets.

### WiLoR

**Paper finding.** WiLoR combines a fully convolutional hand detector and transformer reconstruction. Detection uses a DarkNet backbone, a PANet-style multiscale neck, and anchor-free heads predicting boxes, side labels, and keypoints. Reconstruction first predicts MANO/camera parameters, then projects estimated geometry into multiscale image features to predict residual corrections. [Potamias et al., WiLoR](https://arxiv.org/abs/2409.12259)

**Independent interpretation.** Image-aligned refinement addresses a weakness of global regression: a single global token can encode a plausible hand without enforcing tight local alignment. Sampling features at predicted projected locations creates a feedback loop between geometry and image evidence. However, if the initialization places a finger on a visually similar neighboring structure, refinement may reinforce that error. Multiscale evidence increases the region of useful context; it does not eliminate correspondence ambiguity.

The local [hand-motion documentation](../../../track3/features/hand_motion/README.md) correctly treats licensed MANO assets, FK conventions, and WiLoR provisioning as prerequisites. This chapter does not establish their availability, deployment rights, or numerical behavior. Those remain separate integration decisions.

## Camera, crop, anatomical side, and orientation conventions

### Weak-perspective approximation and metric placement

**Independent derivation.** Perspective projection is

\[
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
\]

When hand depth variation is small relative to mean depth \(Z_0\), a weak-perspective approximation uses \(s_x=f_x/Z_0\) and \(s_y=f_y/Z_0\). With a square crop of width \(b\) pixels and normalized coordinates spanning \([-1,1]\), a commonly encountered scale conversion is

\[
Z_0\approx\frac{2f}{bs}.
\]

The factor of two depends on normalization. The formula is not portable without the crop transform, padding, resize, focal-length convention, and definition of the predicted translation. Some practical pipelines describe this initialization as weak perspective even when their final renderer uses perspective; distinguish the approximation from the renderer.

Let \(A_{\rm crop}\) map full-grid homogeneous pixels to crop pixels. Then the effective crop intrinsics are \(K_{\rm crop}=A_{\rm crop}K\) for an affine crop/resize without distortion. Any image reflection must be recorded separately because it changes handedness of coordinates. Applying a full-image principal point to a centered crop can introduce a lateral translation error even if depth is correct.

Given metric depth \(Z(u,v)\), unprojection is

\[
X_c=Z(u,v)K^{-1}[u,v,1]^\top.
\]

Depth at a projected fingertip can belong to an occluder. A proposed robust fitting objective is

\[
E_z=\sum_jw_j\rho\!\left(
Z(u_j,v_j)-[R_hJ_j(\theta,\beta)+p_h]_z
\right),
\]

where \(w_j\) represents accepted geometric support, not detector score. Surface-based fitting is preferable when enough mesh samples are visible because it can compare the frontmost projected hand surface rather than sparse uncertain keypoints. Exclude object overlap and ambiguous two-hand overlap where their depth ownership is unresolved.

### Anatomical handedness and reflections

**Independent analysis.** A proper rotation satisfies \(R^\top R=I\) and \(\det R=+1\). A reflection has determinant \(-1\). Therefore mirroring a right-hand reconstruction is not generally equivalent to adding a wrist rotation. It changes coordinate handedness and requires consistent treatment of articulation, surface geometry, triangle winding, camera convention, and anatomical frames.

The project contract distinguishes `mirrored_right_hand` inputs from `anatomical_unmirrored` outputs and records whether conversion has already occurred. A downstream consumer must not mirror those parameters again. Position-only overlays can miss a convention error because the reflected surface may still project plausibly while its local orientation targets are wrong.

Anatomical side should combine classifier evidence, thumb placement, feasible articulation, and temporal identity. Under severe occlusion, an unresolved label is more accurate than a guessed one. Screen-left, mask index, and anatomical-left should remain separate fields.

### Joint orientations and quaternion conventions

**Project analysis.** The upstream retargeting contract uses 21 world positions and absolute anatomical joint orientations. Finger direction alone leaves twist unresolved. MANO FK supplies those orientations in a declared axis convention; an independently built geometric frame must document its additional choices.

For \(R_{wj}=R_{wc}R_{cj}\), a valid camera-relative joint rotation becomes world-valid only if the camera rotation is valid. Reordering quaternion entries is representation conversion; applying a basis transformation is a geometric operation. They must not be conflated. Unit quaternions \(q\) and \(-q\) represent the same rotation, so sign continuity may be useful for interpolation, while component-wise finite differences without handling that equivalence can create artificial jumps.

For an angular comparison, use the relative rotation

\[
\Delta R=R_1^\top R_2,
\qquad
\alpha=\cos^{-1}\!\left(\operatorname{clip}\left(
\frac{\operatorname{tr}(\Delta R)-1}{2},-1,1\right)\right).
\]

This avoids interpreting Euler-angle wraparound as a physical discontinuity. Near \(\pi\), the rotation-log axis itself needs care even when the scalar angle remains meaningful.

## Object geometry, canonical frames, and robust tracking

### FoundationPose and BundleSDF

**Paper finding.** FoundationPose supports novel-object estimation/tracking from CAD geometry or reference images. A neural implicit object representation supports rendering when CAD is unavailable. Learned refinement/scoring compare rendered hypotheses with observations; synthetic training and contrastive learning support generalization. The object field represents signed distance and appearance with multiresolution hash encoding. [Wen et al., FoundationPose](https://arxiv.org/abs/2312.08344)

**Paper finding.** BundleSDF jointly tracks and reconstructs unknown rigid objects from monocular RGB-D video. A Neural Object Field and pose-graph optimization share information through a pool of posed memory frames. Initial segmentation establishes the target; observations accumulate geometry and appearance over time. [Wen et al., BundleSDF](https://arxiv.org/abs/2303.14158)

**Project analysis.** A supplied metric scan reduces unknowns: shape and scale can remain fixed while pose is estimated. Reconstructing that same object from scratch introduces extra freedoms that can absorb pose or calibration errors. The spoon is a distinct case because its reference scan is missing. A generated or generic spoon proxy may support coarse tracking, but its extent, shape, and body frame are hypotheses requiring explicit provenance.

The current observed-fusion design preserves the tracking body frame and marks incomplete observed geometry as unsuitable for collision without further work. An observed mesh, an appearance-completed mesh, and a watertight collision proxy are different artifacts. A completion model can improve plausible hidden shape while decreasing fidelity to the particular object.

### Canonical frame bookkeeping

**Independent derivation.** If mesh coordinates change by rigid transform \(A\),

\[
X'_o=AX_o,\qquad T'_{co}=T_{co}A^{-1}.
\]

Both must change together. Recentering the vertices without compensating pose moves the physical object. Scaling geometry requires consistent translation/extent handling; a similarity transform is not an \(SE(3)\) pose.

Document the object frame's origin, axes, units, geometry hash, and any canonicalization transform. Bounding-box center, vertex centroid, center of mass, grasp frame, and the dataset's body origin need not coincide. A frame convenient for optimization must not silently replace the frame required for exported trajectories.

### Lie-group pose updates and complementary residuals

**Independent derivation.** For a left-multiplied local increment,

\[
T\leftarrow\exp(\widehat{\delta\xi})T,
\qquad \delta\xi\in\mathbb R^6.
\]

The hat operator embeds translational and rotational increments into the Lie algebra \(\mathfrak{se}(3)\). Left and right increments have different coordinate interpretations; Jacobians and covariance must use the selected convention consistently.

A proposed objective is

\[
E_o=\lambda_kE_{\rm tracks}+\lambda_mE_{\rm mask}
+\lambda_dE_{\rm depth}+\lambda_pE_{\rm prior}.
\]

Silhouette residuals constrain the image boundary; depth constrains visible surface distance; tracked correspondences constrain material motion; temporal priors limit implausible jumps. None should dominate merely because its numerical units are larger. Pixel, metre, and radian residuals should be normalized by declared uncertainty or scale before weighting.

For matched 3D points \(X_i,Y_i\) with target surface normal \(n_i\), the point-to-plane residual is

\[
r_i=n_i^\top(RX_i+p-Y_i).
\]

A robust loss suppresses contamination by hand points and mismatches. However, robustification cannot manufacture missing constraints: a planar patch poorly constrains some tangential motion, and a cylindrical patch poorly constrains rotation about its axis.

With linearized residual Jacobian \(J\) and weight matrix \(W\),

\[
H\approx J^\top WJ.
\]

Small eigenvalues of \(H\) indicate weakly observed directions. A low residual and ill-conditioned \(H\) can coexist. Confidence should therefore include support count, support distribution, residual scale, pose observability, and source provenance. Dense neighboring pixels are correlated; counting them as independent measurements overstates confidence.

### Occlusion, symmetry, and unchanged official metrics

**Independent analysis.** If a verified symmetry group \(\mathcal G\) leaves an object's observed geometry/appearance unchanged, hypotheses \(T\) and \(TG\) may be observationally equivalent for \(G\in\mathcal G\). Internal tracking can maintain multiple such hypotheses, retain an unobservable angle, or use a verified quotient representation for association. Approximate geometric symmetry can be broken by texture, handles, contacts, or task-specific frame conventions.

**Project constraint.** Default project diagnostics and official metrics remain **unsymmetrized**. Do not replace their errors by a minimization over symmetries unless an independently verified metric protocol explicitly requires that operation. Internal hypothesis handling does not authorize a changed reported score. A separate, clearly named symmetry-aware diagnostic would need a verified group and declared semantics; it must not replace the headline result.

During occlusion, process priors can propagate a pose distribution, but predictions are not measurements. Reacquisition should compare several hypotheses when symmetry or large movement makes a single extrapolation fragile. Preserve observed/predicted support labels and avoid smoothing across an identity switch.

## Joint reconstruction and contemporary systems

### HOMan and HOLD

**Paper finding.** HOMan jointly fits hands and manipulated objects from RGB videos using detections, hand estimates, segmentation, and geometric/physical constraints. Its learning-free fitting approach supports two-hand interactions without task-specific 3D supervision. [Hasson et al., HOMan](https://arxiv.org/abs/2108.07044)

**Paper finding.** HOLD removes the prescanned-object-template requirement using a compositional articulated implicit representation. Initial hand estimates and object structure-from-motion initialize reconstruction; hand–object constraints refine poses, followed by improved geometry reconstruction. [Fan et al., HOLD](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_HOLD_Category-agnostic_3D_Reconstruction_of_Interacting_Hands_and_Objects_from_CVPR_2024_paper.html)

**Independent interpretation.** Joint inference uses complementary information: an object constrains where fingers can exist, and a plausible grasp constrains object placement. It also couples errors. If object geometry is inaccurate, a strong contact loss can pull a correct finger onto the wrong surface. If the hand initializer is wrong, a flexible object model can deform to accommodate it.

A proposed factorization is

\[
E(\mathcal X)=E_{\rm image}+E_{\rm metric}
+E_{\rm temporal}+E_{\rm anatomical}
+E_{\rm interaction}+E_{\rm collision}.
\]

Use bounded corrections and preserve visible-evidence anchors. Optimize small groups of well-supported variables before opening all state freedoms. Sharing a sequence-level hand shape and object geometry prevents per-frame changes from hiding incorrect pose.

### HOT3D as evidence about egocentric observability

**Paper finding.** HOT3D supplies egocentric multiview recordings, hand/object/camera annotations, and scanned geometry. Its experiments show benefits from multiple viewpoints for hand tracking, object pose, and in-hand reconstruction. These capture conditions differ from an unconstrained RGB-only deployment setting. [Banerjee et al., HOT3D](https://arxiv.org/abs/2411.19167)

**Project implication.** When another calibrated view or stereo geometry already exists, using that evidence is often more defensible than adding another monocular prior. The comparison must retain provenance: a method using extra views, known scans, or metric depth has stronger inputs than an RGB-only method. Architecture tables should report those inputs alongside model names.

### VIDEOMANIP

**Paper finding.** VIDEOMANIP combines MoGe-2, SAM 2, generated object meshes, FoundationPose, HaMeR, retargeting, and contact optimization. The inspected v1 formulation assumes a static camera, estimates/refines mesh scale, and uses depth to align hands and objects. Direct transfer to moving-camera footage therefore requires additional camera treatment. The February 11, 2026 v2 title says “3D”; the v1 HTML says “4D.” [Chen et al., VIDEOMANIP](https://arxiv.org/abs/2602.09013)

**Project analysis.** This is useful evidence for assembling strong pretrained components into a shared coordinate system. It does not establish that independent model outputs are already calibrated. A language-derived object-size estimate is a prior, not a metric measurement. A predicted depth field supplies a common reference, but errors in that reference affect both hands and objects and can make their apparent agreement misleading.

### C2Dex

**Paper finding.** C2Dex uses stable object-side contacts as its shared reconstruction/retargeting representation. Candidate contacts are filtered with surface-normal compatibility, transformed into canonical object coordinates, clustered across locally stable intervals, and represented by cluster medoids. Reconstruction combines contact, penetration, and regularization; retargeting preserves contacts and local Laplacian interaction structure, followed by residual simulation refinement. It is an August 2026 preprint revised September 6, with the record describing submission to RA-L. [Ren et al., C2Dex](https://arxiv.org/abs/2608.07045)

**Independent project implication.** Object-side interaction targets are valuable because the object's surface is shared across human and robot embodiments, while finger lengths and joint spaces differ. The canonical frame removes rigid object motion before temporal aggregation. However, stable contact is a regime assumption: sliding, rolling, and finger gaiting require changing correspondences. The existence of a dominant cluster does not prove that contact was physically present.

## Contact geometry, collision, and mechanics

### Proximity is a proxy

**Independent derivation.** Let \(d(x,\mathcal M_o)\) be unsigned distance to object surface \(\mathcal M_o\). A smooth proximity statistic is

\[
p(x)=\exp\left[-d(x,\mathcal M_o)^2/(2\sigma^2)\right].
\]

\(\sigma\) sets a distance scale; it does not calibrate this number as a physical contact probability. A one-centimetre reconstruction error can reverse whether a fingertip appears touching or separated. Two nearby surfaces can face the same direction and never form a plausible contact.

Normal compatibility can add evidence. For outward unit normals \(n_h,n_o\), approximately opposing surfaces have \(-n_h^\top n_o\) near one. This is insufficient by itself: silhouette overlap and opposing normals can occur on separated surfaces along the viewing ray. Combine distance, visibility, temporal consistency, and uncertainty, while retaining the label “inferred contact hypothesis.”

The current project's contact output is a geometric proxy. It should not be described as measured contact force, tactile sensing, friction estimation, or proof of a stable grasp.

### Signed distance and penetration

**Independent derivation.** Let \(\phi_o\) be a canonical signed distance field, positive outside. At time \(t\),

\[
\phi_{o,t}(x)=\phi_o(R_{o,t}^\top(x-p_{o,t})).
\]

For hand vertices \(V_{h,t,i}\), a proposed penalty is

\[
E_{\rm penetration}=\sum_{t,i}
\left[\max(0,-\phi_{o,t}(V_{h,t,i}))\right]^2.
\]

The sign convention must match the mesh. Open surfaces, inconsistent winding, and thin structures can make inside/outside ambiguous. Vertex-only sampling can miss edge or triangle crossings. A collision approximation should declare whether it uses a watertight proxy, convex decomposition, sampled surface, or an observed incomplete mesh, and which penetrations it can miss.

An unsigned distance cannot distinguish penetration from separation. Penalizing all small distances also destroys intended grasp contact. Use an explicit signed formulation or a declared alternative, with contact intervals controlling attraction and collision terms controlling forbidden overlap.

### Stable contact and sliding regimes

**Independent proposed objective.** Canonicalize a candidate hand contact point:

\[
\bar x_{t,i}=R_{o,t}^{\top}(V_{h,t,i}-p_{o,t}).
\]

For a fixed no-slip contact, temporal consistency suggests

\[
E_{\rm stable}=\sum_{t,i}\omega_{t,i}
\|\bar x_{t+1,i}-\bar x_{t,i}\|^2.
\]

The weight \(\omega_{t,i}\) should be nonzero only for admitted stable intervals. It must vanish across a release or identity discontinuity. Fixed hand-vertex correspondence is especially restrictive during rolling contact, where the touching material points on the hand and object can change.

For sliding, a weaker surface constraint can maintain near-zero normal gap while permitting tangential motion. Let \(n\) be the object normal and \(\Delta x\) relative displacement. The normal and tangential components are

\[
\Delta x_n=nn^\top\Delta x,
\qquad \Delta x_\tau=(I-nn^\top)\Delta x.
\]

Penalizing normal departure without forcing \(\Delta x_\tau=0\) expresses a different interaction model from a stable grasp. Contact-state selection therefore changes the mathematical problem; it is not merely a confidence threshold.

### Forces require a different inverse problem

**Independent mechanics reminder.** Let gap \(g\ge0\), normal force \(f_n\ge0\), and tangential force \(f_\tau\) satisfy

\[
gf_n=0,
\qquad \|f_\tau\|\le\mu f_n.
\]

These express unilateral contact and a friction-cone approximation. Recovering \(f_n\), \(f_\tau\), or \(\mu\) requires additional assumptions or measurements. Even if object acceleration were known, multiple contact-force distributions can generate the same net wrench.

For object mass \(m\), inertia \(I\), angular velocity \(\omega\), and contact moment arms \(r_i\), mechanics would impose

\[
m\ddot p=mg+\sum_i f_i,
\qquad
I\dot\omega+\omega\times I\omega=\sum_i r_i\times f_i.
\]

Unknown mass distribution, camera acceleration, uncertain contact locations, and noisy derivatives make this inverse problem substantially harder than proximity estimation. Collision-free motion can still be dynamically infeasible. The chapter proposes no force estimates or simulation execution.

## Retargeting across embodiments

### What should transfer

**Paper finding.** DexPilot demonstrates vision-based teleoperation of a dexterous arm–hand system. It motivates matching useful hand relationships rather than directly copying human joint angles. [Handa et al., DexPilot](https://arxiv.org/abs/1910.03135)

**Paper finding.** AnyTeleop broadens vision-based teleoperation across different hands, arms, camera configurations, and real/simulated environments. [Qin et al., AnyTeleop](https://arxiv.org/abs/2307.04577)

**Project analysis.** The human and robot need not share finger lengths, joint counts, actuation couplings, or reachable thumb opposition. Joint-angle copying does not preserve fingertip contact. A successful mapping should preserve task-relevant object-relative targets and contact timing while allowing different robot joint configurations.

The upstream [retargeting contract](../../../track3/features/retargeting/README.md) requires declared positional and orientation conventions. A research objective based only on fingertip positions is not a replacement for that full contract. Missing hands also require explicit treatment; a parked template is an unobserved reference, not recovered demonstration evidence.

### Position, vector, and pinch objectives

**Independent formulations.** For robot FK keypoints \(f_j(q)\), target positions \(a_j\), and robot configuration \(q\),

\[
E_{\rm pos}=\sum_jw_j\|f_j(q)-a_j\|^2.
\]

For human points \(h_j\), vector matching uses

\[
E_{\rm vec}=\sum_{(i,j)}w_{ij}
\|(f_i(q)-f_j(q))-\alpha_{ij}(h_i-h_j)\|^2.
\]

Common translation cancels, so wrist placement needs another constraint. A single scale \(\alpha\) captures overall size; finger-specific scales can accommodate proportions but may weaken the task geometry if left unconstrained. The choice of vector pairs matters: wrist-to-tip vectors prioritize reach, while fingertip-to-fingertip vectors encode opposition and pinches.

DexPilot-style objectives emphasize close fingertip relationships to preserve pinches across morphology. An implementation must state its distance projection, activation thresholds, weights, and transition behavior. The generic equation above is an independent vector-retargeting expression, not a transcription of the exact DexPilot objective. Hard switching near a threshold can produce chatter; hysteresis or a smooth transition can reduce that artifact, provided the modification is identified as a project choice.

### Interaction graphs and Laplacian coordinates

**Independent mathematical development.** Construct a graph of corresponding hand keypoints and object points. Let \(V\) contain node coordinates and let normalized weights satisfy \(\sum_jw_{ij}=1\). Define

\[
\delta_i(V)=v_i-\sum_jw_{ij}v_j,
\qquad L=I-W.
\]

An interaction-coordinate objective is

\[
E_{\rm graph}=\|LV_r-LV_h\|_F^2,
\]

where \(V_r\) and \(V_h\) use corresponding node order and compatible frames. Because \(L\mathbf1=0\), a common translation leaves the coordinates unchanged. A common proper rotation rotates the coordinates; matching raw coordinates requires the two interactions to share orientation. Scale changes multiply them. Thus the formulation is translation invariant but not automatically invariant to arbitrary independent rotation or scale.

Object nodes anchor local hand–object relationships that hand-only vectors omit. Fixed object samples should be identified in the canonical frame so changing sample order cannot change the target graph. Topology and weight policy must also be explicit. Rebuilding a nearest-neighbor graph every optimization step can introduce discontinuities; freezing topology preserves the target structure but can retain a poor initial neighborhood. These are design tradeoffs, not universally resolved defaults.

The graph objective alone has a translation nullspace and can permit task-irrelevant solutions. Combine it with wrist placement, object-relative contact targets, and robot limits. Morphological mismatch can make exact coordinate preservation impossible, so report residuals by task-relevant region rather than expecting a zero objective.

### Sequence constraints and time

**Project proposal.** A bounded retargeting problem is

\[
\min_{\{q_t,T_{r,t}\}}
E_{\rm vec}+E_{\rm contact}+E_{\rm graph}
+E_{\rm collision}+E_{\rm evidence}+E_{\rm temporal},
\]

\[
q_{\min}\le q_t\le q_{\max},
\qquad
|q_{t+1}-q_t|\le\dot q_{\max}\Delta t_t.
\]

Hard constraints express known limits more reliably than finite penalties. Optimization feasibility is still kinematic feasibility: torque limits, controller response, friction, and object dynamics remain additional conditions.

Use actual source presentation timestamps. For \(\Delta t_t=t_{t+1}-t_t\), velocity estimates are secants divided by \(\Delta t_t\), not nominal FPS. Acceleration compares adjacent secants over their actual midpoint spacing. Missing observations, duplicated timestamps, and identity changes must break derivative support. Smoothing across a grasp transition can shift contact timing even while reducing average joint jitter.

The current conservative wrist-refinement design changes all joints on a hand by a common bounded translation, preserving FK articulation and bone lengths. An articulated optimizer would be a new capability requiring visible landmark evidence, licensed assets, configuration, and an independently authorized run. A research document does not imply that capability already exists.

## Design decisions supported by the synthesis

| Decision | Rationale | Required evidence before claiming success |
|---|---|---|
| Preserve calibrated pixel-grid and transform provenance | Crop, stereo, and camera errors can masquerade as hand or object motion | Matching grids, clocks, frames, and source identities |
| Keep anatomical side independent of mask index | Segmentation IDs are not anatomical labels | Side evidence and explicit conversion history |
| Use regressors as initializers with uncertainty | Priors can plausibly fill occluded geometry incorrectly | Visible reprojection and accepted metric support |
| Fix sequence-level shape where appropriate | Per-frame shape can absorb pose/calibration errors | Declared identity model and bounded exceptions |
| Preserve supplied scan frames | Recentering changes pose semantics | Recorded mesh-to-body transforms |
| Separate the spoon from exact-scan cases | Missing geometry introduces extra uncertainty | Explicit proxy/reconstruction provenance |
| Maintain weakly observed pose hypotheses internally | Symmetry and occlusion can leave pose directions unresolved | Support/observability diagnostics |
| Keep official/default scores unsymmetrized | Internal ambiguity handling does not redefine the metric | Unchanged metric protocol and separate optional diagnostics |
| Treat contact as a time-dependent hypothesis | Stable grasp, sliding, and release require different objectives | Geometry, interval support, and uncertainty |
| Keep proximity separate from force | Geometry does not identify a force distribution | Additional mechanics assumptions or measured sensing |
| Retarget object-relative interactions | Human and robot morphology differ | Contact/kinematic residuals and declared limits |
| Select recipes on admissible tuning evidence | Consumed holdout cannot support unbiased reselection | Frozen choices and independently fixed assessment protocol |

No table entry is an executed acceptance result. These are proposals for organizing implementation and later authorized evidence collection, consistent with [artifact contracts](../../design/artifact-contracts.md) and the [error budget](../../design/error-budget.md).

## Primary bibliography and version notes

Author lists use “et al.” for the two large SAM collaborations; the linked primary records contain their complete author lists. All arXiv identifiers below identify primary author manuscripts. Conference labels are included where verified; a preprint submission statement is not described as acceptance.

- **Liu, Shilong; Zeng, Zhaoyang; Ren, Tianhe; Li, Feng; Zhang, Hao; Yang, Jie; Li, Chunyuan; Yang, Jianwei; Su, Hang; Zhu, Jun; Zhang, Lei.** *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* arXiv:2303.05499, 2023. [Primary record](https://arxiv.org/abs/2303.05499).
- **Ravi, Nikhila, et al.** *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024. [Primary record](https://arxiv.org/abs/2408.00714).
- **Carion, Nicolas, et al.** *SAM 3: Segment Anything with Concepts.* arXiv:2511.16719, 2025. [Primary record](https://arxiv.org/abs/2511.16719).
- **Meta, facebookresearch/sam3.** *SAM 3.1 — March 27, 2026.* Official release notes; engineering release rather than a separately claimed peer-reviewed paper. [Release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
- **Doersch, Carl; Yang, Yi; Vecerik, Mel; Gokay, Dilara; Gupta, Ankush; Aytar, Yusuf; Carreira, Joao; Zisserman, Andrew.** *TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement.* arXiv:2306.08637, 2023. [Primary record](https://arxiv.org/abs/2306.08637).
- **Karaev, Nikita; Makarov, Iurii; Wang, Jianyuan; Neverova, Natalia; Vedaldi, Andrea; Rupprecht, Christian.** *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos.* arXiv:2410.11831, 2024. [Primary record](https://arxiv.org/abs/2410.11831).
- **Romero, Javier; Tzionas, Dimitrios; Black, Michael J.** *Embodied Hands: Modeling and Capturing Hands and Bodies Together.* ACM Transactions on Graphics 36(6), Article 245, 2017, pp. 245:1–245:17. DOI:10.1145/3130800.3130883. [Author publication record](https://is.mpg.de/ps/en/publications/embodiedhands). The later arXiv deposit is not the original publication year.
- **Pavlakos, Georgios; Shan, Dandan; Radosavovic, Ilija; Kanazawa, Angjoo; Fouhey, David; Malik, Jitendra.** *Reconstructing Hands in 3D with Transformers.* arXiv:2312.05251, 2023; CVPR 2024. [Primary record](https://arxiv.org/abs/2312.05251).
- **Potamias, Rolandos Alexandros; Zhang, Jinglei; Deng, Jiankang; Zafeiriou, Stefanos.** *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild.* arXiv:2409.12259, 2024; CVPR 2025. [Primary record](https://arxiv.org/abs/2409.12259); [conference paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf).
- **Wen, Bowen; Yang, Wei; Kautz, Jan; Birchfield, Stan.** *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects.* arXiv:2312.08344, 2023; CVPR 2024. [Primary record](https://arxiv.org/abs/2312.08344).
- **Wen, Bowen; Tremblay, Jonathan; Blukis, Valts; Tyree, Stephen; Müller, Thomas; Evans, Alex; Fox, Dieter; Kautz, Jan; Birchfield, Stan.** *BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects.* arXiv:2303.14158; CVPR 2023. [Primary record](https://arxiv.org/abs/2303.14158).
- **Hasson, Yana; Varol, Gül; Laptev, Ivan; Schmid, Cordelia.** *Towards unconstrained joint hand-object reconstruction from RGB videos.* arXiv:2108.07044, 2021. [Author preprint](https://arxiv.org/abs/2108.07044); [official author implementation](https://github.com/hassony2/homan). Conference publication: 3DV 2021, pp. 659–668, DOI:10.1109/3DV53792.2021.00075; conference author ordering lists Schmid before Laptev.
- **Fan, Zicong; Parelli, Maria; Kadoglou, Maria Eleni; Chen, Xu; Kocabas, Muhammed; Black, Michael J.; Hilliges, Otmar.** *HOLD: Category-agnostic 3D Reconstruction of Interacting Hands and Objects from Video.* CVPR 2024, pp. 494–504; arXiv:2311.18448. [Conference record](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_HOLD_Category-agnostic_3D_Reconstruction_of_Interacting_Hands_and_Objects_from_CVPR_2024_paper.html).
- **Banerjee, Prithviraj; Shkodrani, Sindi; Moulon, Pierre; Hampali, Shreyas; Han, Shangchen; Zhang, Fan; Zhang, Linguang; Fountain, Jade; Miller, Edward; Basol, Selen; Newcombe, Richard; Wang, Robert; Engel, Jakob Julian; Hodan, Tomas.** *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos.* arXiv:2411.19167, 2024; CVPR 2025. [Primary record](https://arxiv.org/abs/2411.19167). Distinguish this record from the earlier *Introducing HOT3D* manuscript, arXiv:2406.09598.
- **Chen, Hongyi; Dong, Tony; Wu, Tiancheng; Wang, Liquan; Jangir, Yash; Niu, Yaru; Ye, Yufei; Bharadhwaj, Homanga; Erickson, Zackory; Ichnowski, Jeffrey.** *Dexterous Manipulation Policies from RGB Human Videos via 3D Hand-Object Trajectory Reconstruction.* arXiv:2602.09013v2, February 11, 2026. [Versioned record](https://arxiv.org/abs/2602.09013v2). Framework name: VIDEOMANIP. Inspected v1 full text uses “4D” in the title: [v1 HTML](https://arxiv.org/html/2602.09013v1).
- **Ren, Jie; Jiang, Zhehao; Yang, Yinhong; Jia, Haorui; Jiang, Han; Li, Ben; Yao, Yao; Lin, Cheng; Shen, Qiu; Bing, Zhenshan; Long, Xiao-Xiao; Cao, Xun.** *C2Dex: Contact-Consistent Reconstruction and Retargeting for Dexterous Manipulation from Monocular Video.* arXiv:2608.07045v2, September 6, 2026; first submitted August 7. [Versioned record](https://arxiv.org/abs/2608.07045v2); [inspected v1 method text](https://arxiv.org/html/2608.07045v1). Record states submitted to RA-L.
- **Handa, Ankur; Van Wyk, Karl; Yang, Wei; Liang, Jacky; Chao, Yu-Wei; Wan, Qian; Birchfield, Stan; Ratliff, Nathan; Fox, Dieter.** *DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System.* arXiv:1910.03135, 2019; ICRA 2020, pp. 9164–9170. [Primary record](https://arxiv.org/abs/1910.03135); [author project page](https://sites.google.com/view/dex-pilot/home).
- **Qin, Yuzhe; Yang, Wei; Huang, Binghao; Van Wyk, Karl; Su, Hao; Wang, Xiaolong; Chao, Yu-Wei; Fox, Dieter.** *AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System.* arXiv:2307.04577, 2023. [Primary record](https://arxiv.org/abs/2307.04577).
