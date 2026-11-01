# Retargeting, contact mechanics, and reference-conditioned robot policies

## Scope and evidence

This chapter studies the transition from reconstructed human hand–object motion to robot actions that produce achieved object trajectories in simulation. It separates the published algorithms from the executable FlashCHORD source pinned by this project, and separates both from original mathematical deductions. The literature supports design questions and prospective research; it does not establish that this project's unfinished simulator integration trains successfully.

Research date: 2026-10-18. Primary papers and read-only local source were consulted. No simulator, model, notebook, compiler, test, dependency installation, or submission was executed. No held-out campaign outcomes informed this chapter. Existing frozen choices and synthetic-mixture decisions are unchanged.

The local authority for the challenge integration is the [Track 3 contract](../../../reports/contract.md), with detailed [runtime evidence](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md). Runtime statements below refer to `nvidia-isaac/video_to_data` commit `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, present under `upstream/video_to_data-flashchord-aee95f2/`. A published CHORD equation and this pinned runtime can legitimately differ; the executable source defines the behavior of that integration.

**Notation.** \(T\in\mathrm{SE}(3)\) denotes a rigid pose, \(R\in\mathrm{SO}(3)\) a rotation, \(p\in\mathbb R^3\) a position, \(q\) robot joint coordinates, and \(v\) a velocity or spatial twist as specified locally. A bar denotes a reference target; a hat denotes a reconstructed estimate. \([x]_+=\max(x,0)\) acts elementwise. Equations marked *schematic* specify a conceptual model rather than an exact implementation.

## The task is physical reference tracking

The local contract distinguishes reconstructed reference trajectories from submitted trajectories. For the pose leaderboards, submission contains object poses achieved during policy rollout. The floating-hand route observes simulated object poses and current participant-created reference targets. Evaluation uses one world, deterministic actions, a frame-zero reset, assistance off, and failure terminations disabled. The floating Sharpa action has 56 residual coordinates at 20 control steps per second. These facts describe this pinned project contract, not every policy-learning benchmark. [Contract](../../../reports/contract.md)

An original abstraction makes the dependency chain explicit:

\[
I_{1:T}\xrightarrow{\mathcal R}
(\hat\tau,\hat{\mathcal G},\hat{\mathcal C})
\xrightarrow{\mathcal K}\bar q_{1:T}
\xrightarrow{\pi_\theta,\mathcal F}
T^{\mathrm{achieved}}_{1:T}.
\]

Reconstruction \(\mathcal R\) estimates motion, geometry, and contact evidence. Retargeting \(\mathcal K\) converts human motion into an embodiment-specific prior. The policy \(\pi_\theta\) chooses corrective actions. The simulator \(\mathcal F\) determines which object motion those actions actually produce. Improving one intermediate fit cannot establish improvement at the final output.

For example, fingertips can match reconstructed human locations accurately while exerting forces in unsuitable directions. A hand can track an object that is being moved by training assistance without learning to move it independently. A policy can obtain a high local tracking reward after random reference resets while failing when required to execute every preceding contact transition from frame zero.

### An augmented Markov model

The following is an original, schematic formalization:

\[
z_t=\left(q_t,\dot q_t,
\{T^k_t,v^k_t\}_{k=1}^{K},\phi_t,h_t,\xi\right),
\qquad z_{t+1}\sim P_\xi(\cdot\mid z_t,a_t).
\]

Here \(\phi_t\) identifies reference phase, \(h_t\) includes action-filter or delay history, and \(\xi\) contains physical parameters such as mass, friction, and controller gains. A nonstationary reference-tracking task becomes stationary in this augmented state when phase follows a specified transition. Action filtering and delays also belong in state: the same raw action can have different physical consequences when previous processed actions differ.

The actor need not receive every component. A schematic observation is

\[
o_t=[q_t,\dot q_t,\Delta q_t,\Delta T^{\mathrm{hand}}_t,
\{T^k_t,\Delta T^k_t\}_{k=1}^{K},a_{t-1},c_t].
\]

The pinned schema must determine actual ordering, normalization, quaternion convention, contact slots, and dimension. Object count changes the feature inventory. Treating a contact-rich system as Markov in an underspecified feature vector is an assumption, not a consequence of choosing an MLP.

## Closest manipulation-transfer papers

### CHORD: compare contact effects across embodiments

*Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration* (Zhu et al., 2026; arXiv:2607.00033) uses object tracking, motion imitation, and contact guidance. Contact primitives are \(w=[f;p\times f]\), and sampled support is \(\sigma=\max_{\mathrm{col}}(B^\top W)\). Its published tolerance-band reward is

\[
r^k_{\mathrm{cws}}=\exp\left[-\frac{
\|[(1-\beta)\sigma_h-\sigma_r]_+\|_2^2+
\|[\sigma_r-(1+\beta)\sigma_h]_+\|_2^2}{v_{\mathrm{cws}}}\right].
\]

Separate missed/unintended-contact penalties accompany it. Random reference resets, temporary virtual object control, residual actions, and assistance annealing support learning. For noisy reconstructed contacts the paper describes reduced positive support-direction coverage. Its task success definition is specific to its experiments and is not interchangeable with every challenge score. [Primary paper, §§3–4](https://arxiv.org/html/2607.00033v1)

### DexMachina: transfer function through an assistance curriculum

*DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation* (Mandi et al., 2025 preprint; ICML 2026; arXiv:2505.24853) emphasizes tracking object states across robot hands. Virtual object controllers initially move objects toward targets; controller strength decreases as robot contacts assume responsibility. Motion and contact guidance accompany object tracking. Its state observations include object states, joint targets, finger-to-object distances, and normalized contact forces. The reported implementation uses PPO in Genesis, generally 12,000 parallel environments, or 10,000 for one hand, on an L40S or H100. Those operating points do not establish feasibility on the project's smaller GPU. [Primary paper](https://arxiv.org/html/2505.24853v1)

### ManipTrans: learn motion first, interaction correction second

*ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning* (Li et al., CVPR 2025; arXiv:2503.21860) pretrains a general hand-motion imitation policy, then learns a task-specific residual under interaction constraints. Both stages use PPO. The reference includes wrist pose and motion together with MANO keypoint motion. This provides substantially more hand supervision than an isolated initialization pose. The paper's manipulation-transfer controller and the later imitation-learning policies trained on generated DexManipNet trajectories have different action interfaces; results from one should not be attributed to the other. [Primary paper](https://arxiv.org/html/2503.21860v1)

### Human2Sim2Robot: object trajectories can carry the task

*Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration* (Lum et al., 2025; arXiv:2504.12609) extracts an object trajectory and a pre-manipulation hand pose from one RGB-D demonstration. The first defines reward; the second guides initialization. Its object reward is

\[
r_t=\exp[-\alpha d(T^{\mathrm{goal}}_{\tau+t},T^{\mathrm{obj}}_t)].
\]

Its ablations show that inaccurate full hand trajectories can make a residual prior unhelpful; additional hand supervision is not universally beneficial. The demonstrated setting has one object and assumes reliable object tracking and simulation. This motivates distinguishing object-reference quality from hand-reference quality, rather than universally removing hand supervision. [Primary paper, §§III–V](https://arxiv.org/html/2504.12609v1)

### Comparative interpretation

The following is an original synthesis rather than a pooled experimental ranking.

| Method | Where human evidence enters | Main mechanism for the embodiment gap | Project question it informs |
|---|---|---|---|
| CHORD | Object motion, retargeted motion, contact capabilities | Find robot contacts with mechanically related effects | Are reconstructed contact directions trustworthy? |
| DexMachina | Object targets, motion and contact guidance | Gradual transfer of responsibility from object assistance | Does the policy remain effective as assistance reaches zero? |
| ManipTrans | Dynamic hand-motion prior and interaction references | General motion skill plus task-specific correction | Is reusable prior learning affordable and accurate enough? |
| Human2Sim2Robot | Dense object trajectory and initial hand pose | Object-centric exploration with a useful initial condition | Is a poor continuous hand prior restricting the solution? |

Different embodiments, simulators, metrics, observation assumptions, task collections, and compute prevent a universal numerical ordering. A useful comparison identifies which error each method expects its prior to remove and which error its controller must learn to repair.

## Contact mechanics: independent derivations and limits

This section gives independent mechanics and convex-analysis explanations. It clarifies the meaning and limitations of a wrench fingerprint; it is not additional quotation or reproduction of the CHORD formulation.

### Friction cones and normal conventions

Let \(n\) point in the permitted compressive force direction on the object. A unilateral Coulomb contact permits

\[
f=\lambda_n n+f_\parallel,\quad
\lambda_n\ge0,\quad n^\top f_\parallel=0,\quad
\|f_\parallel\|_2\le\mu\lambda_n.
\]

An outward-facing mesh normal generally needs a sign change for this convention. Accidentally retaining the outward sign reverses the direction in which the contact is assumed capable of pushing.

Choose orthonormal tangents \(t_1,t_2\). A polygonal approximation uses

\[
\tilde f_j=n+\mu(\cos\theta_j t_1+\sin\theta_j t_2),
\qquad f_j=\tilde f_j/\|\tilde f_j\|_2.
\]

Normalizing these rays fixes their magnitude for comparing direction sets. It does not estimate actual contact force. Tangent-basis orientation changes the polygon's angular phase; increasing edge count reduces this discretization dependence but increases work.

At offset \(p\) from the object origin,

\[
w=G(p)f,\qquad
G(p)=\begin{bmatrix}I_3\\[p]_\times\end{bmatrix},
\qquad w=\begin{bmatrix}f\\p\times f\end{bmatrix}.
\]

Thus position-only matching misses the contact normal, and normal-only matching misses the lever arm. An underside contact capable of lifting a lid can be spatially close to a side contact incapable of producing the required torque.

### Units, origins, and frames

Force and torque have different physical units. A comparison in \(\mathbb R^6\) requires a declared metric or characteristic length \(r>0\), such as

\[
\bar w=\begin{bmatrix}f\\(p\times f)/r\end{bmatrix}.
\]

Changing mesh units without consistent normalization changes the relative numerical importance of translation and rotation. The chosen object origin matters too: shifting origin changes torque by the moment of force about the shift. Reference and simulated contacts must use a consistent object frame and radius.

For a rotation \(R\) and translation \(p_0\), wrench coordinates transform through the appropriate spatial-force transform. In the common force-then-torque ordering,

\[
f'=Rf,\qquad \tau'=R\tau+p_0\times Rf.
\]

A rigid world transformation preserves physical behavior when every quantity is transformed consistently, but naively comparing coordinates from different frames does not preserve a numerical fingerprint. A fixed dataset-wide frame error and a time-varying camera-pose error have different downstream consequences.

### A bounded support fingerprint is not an unbounded force cone

For normalized primitives \(\bar w_1,\ldots,\bar w_m\), define

\[
\mathcal P=\operatorname{conv}(\{0,\bar w_1,\ldots,\bar w_m\}).
\]

The support function is

\[
h_{\mathcal P}(b)=\max_{w\in\mathcal P}b^\top w
=\max(0,b^\top\bar w_1,\ldots,b^\top\bar w_m).
\]

The final equality follows because a linear function attains its maximum over a finite convex hull at a generating point. It explains both permutation invariance and the nonnegative clamp. Duplicating a primitive does not change the maximum.

By contrast, the physical positive cone generated by the same rays is unbounded. Its support is generally infinite in directions with positive projection onto a feasible ray. A finite maximum over unit primitives must therefore be interpreted as a normalized bounded descriptor, not the complete unbounded set of possible contact forces.

Contact aggregation also changes interpretation. For independently bounded contact sets,

\[
h_{\mathcal P_1+\cdots+\mathcal P_m}(b)
=\sum_i h_{\mathcal P_i}(b).
\]

For a convex hull of alternative primitives, support is a maximum. The maximum does not encode the additive force capacity of several fingers. This helps compare different contact counts, but prevents treating the fingerprint as a full grasp-strength certificate.

### Finite directional coverage and force closure

Sampling \(B\) directions gives a finite descriptor

\[
\sigma(\mathcal P)=(h_{\mathcal P}(b_1),\ldots,h_{\mathcal P}(b_B)).
\]

Agreement on those directions does not prove equality between samples. Narrow deficiencies may remain unseen. For a compact set bounded by radius \(L\), the support function obeys

\[
|h_{\mathcal P}(b)-h_{\mathcal P}(b')|\le L\|b-b'\|_2.
\]

This follows from Cauchy–Schwarz applied to every member of the set. A certified approximation would therefore require directional covering information and a bound on \(L\), rather than simply reporting the number of random bases.

A count of positive sampled supports is a coverage statistic. True force closure concerns whether the origin lies in the interior of a physically admissible wrench set, with conventions and bounds specified. Positive sampled coverage can be useful training guidance without certifying all-direction disturbance rejection or actuator-limited stability.

### Capability does not determine actual motion

Rigid-body dynamics additionally involve inertia, existing velocity, environmental reactions, and actuator limits:

\[
m\ddot x=\sum_i f_i+mg+f_{\mathrm{ext}},
\qquad
I\dot\omega+\omega\times I\omega
=\sum_i p_i\times f_i+\tau_{\mathrm{ext}}.
\]

Matching available wrench directions does not prove that the policy chooses appropriate force magnitudes or contact timing. A support descriptor can remain similar while a controller squeezes excessively, slips, or fails to generate sufficient acceleration. Task tracking and physical rollout remain essential to the interpretation.

## Pinned FlashCHORD: executable details and paper divergence

### Wrench construction

The source [wrench.py](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/objectives/wrench.py) samples normalized six-dimensional Gaussian directions, builds friction-edge rays plus the normal itself, normalizes force directions, divides torque by object radius, and initializes support maxima at zero. Reference and simulator supports share the per-body basis seed.

The generic `sample_wrench_basis` helper accepts a radius argument, but the active objective builder supplies `1.0` there and supplies actual object radius separately to the support kernel. Reading the helper in isolation would incorrectly suggest radius might be applied twice. See [tracking.py](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/objectives/tracking.py).

The pinned [tracking configuration](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/objective/tracking.yaml) uses 512 basis directions, eight friction-cone edges, friction coefficient 0.1, tolerance 0.1, variance 0.1, and seed zero. These are algorithmic fingerprint settings; the force-cone coefficient should not automatically be equated with every simulator material's friction coefficient.

### Mean exponentials versus an exponential of a summed error

The exact live support objective is in [contact.py](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/objectives/contact.py). For one hand/body pair define

\[
A=\{j:\sigma^{\mathrm{ref}}_j>10^{-3}\},
\]

\[
\ell_j=[(1-\beta)\sigma^{\mathrm{ref}}_j-\sigma^{\mathrm{sim}}_j]_+^2
+[\sigma^{\mathrm{sim}}_j-(1+\beta)\sigma^{\mathrm{ref}}_j]_+^2.
\]

The implementation computes

\[
r_{hb}=\frac1{|A|}\sum_{j\in A}
\mathbf1[\sigma^{\mathrm{sim}}_j>10^{-3}]
\exp(-\ell_j/v),
\]

then averages across reference-active bodies and reference-active hands. No reference-active hands gives zero. This differs from the paper's joint exponential in three ways: directional averaging replaces exponentiation after summation; only commanded-active directions enter the denominator; and a currently inactive direction receives zero rather than an ungated positive exponential.

An original consequence is that correctly covered directions retain credit even when another direction is poor. Under a joint exponential, a single large error can suppress the whole score. With all directions active and nonnegative losses, a mean of exponentials and an exponential of a sum are generally very different functions. Changing basis count also affects the unnormalized joint sum differently from a directional average. Reward magnitude and learning behavior must therefore be explained from this source, not inferred from the paper alone.

The missed-contact term averages the fraction of commanded-active directions without current support. The unintended-contact term checks bodies for which the reference requests no contact: per hand it combines a binary fraction of bodies with unwanted contact and mean squared support, then sums over hands. It is not simply the complementary directional tolerance error.

The force-closure implementation counts directions exceeding `min_support` on reference-active hand/body pairs. It retains a reference activity gate, so it is not globally rewarding every hand to grasp every object at all times.

### Objective weights and curriculum composition

The objective configuration gives support matching weight 10, missed contact −1, unintended contact −10, termination −100, action-rate squared magnitude −0.005, and action squared magnitude −0.002. Force closure is disabled. Object and hand tracking fields initially have zero weights in this configuration; the curriculum supplies object-keypoint weight 20, hand-keypoint weight 0.25, and hand-joint weight 0.25. Reading only the objective file would therefore misstate the training reward. [Objective configuration](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/objective/tracking.yaml); [curriculum](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml)

The local runtime audit records a tension between upstream recommendations favoring force closure for noisy egocentric reconstructions and the Sharpa recipe retaining support matching. This chapter documents the tension; it does not change the frozen proposal or select a replacement based on previous holdout results. [Runtime evidence](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md)

## Retargeting and dynamic feasibility

### Kinematic fitting solves only part of the problem

An original schematic retargeting objective is

\[
q_t^\star=\arg\min_{q\in\mathcal Q}
\sum_i w_i\|\mathrm{FK}_i(q)-\hat y_{i,t}\|_2^2
+\lambda_s\|q-q_{t-1}^\star\|_2^2
+\lambda_c E_{\mathrm{collision}}(q).
\]

The feasible set encodes joint limits; geometry and objective design must address differing finger counts, link lengths, axes, and coupling. Human joint angles do not share a universal coordinate interpretation with robot joints. Normalized inter-keypoint vectors can preserve some geometric relationships better than direct angle transfer, but still do not determine feasible contact forces.

A configuration sequence may satisfy joint limits while demanding impossible velocities or accelerations. Dynamic feasibility additionally requires

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=S^\top\tau+J_c(q)^\top\lambda,
\]

with torque bounds, unilateral normal forces, friction constraints, and contact-mode consistency. Complementarity is schematically expressed as

\[
d_i(q)\ge0,\qquad \lambda_{n,i}\ge0,\qquad
d_i(q)\lambda_{n,i}=0.
\]

Compliant simulator contact models may replace exact complementarity with regularized forces. That changes numerical behavior but not the need to reason about penetration, support, friction, and force limits.

Temporal smoothing reduces differentiated noise, but can also shift grasp closure relative to object acceleration. Smoothing wrist, object, and finger trajectories independently can destroy a previously aligned contact event. A prospective design should therefore measure contact-transition timing and relative geometry, rather than assuming a smoother trajectory is always a better physical reference.

### Residual actions and their reachable region

A schematic residual controller uses

\[
p_t^{\mathrm{target}}=\bar p_t+s_p a_t^p,
\qquad q_t^{\mathrm{target}}=\bar q_t+s_q a_t^q,
\]

\[
R_t^{\mathrm{target}}=\bar R_t\operatorname{Exp}(s_R[a_t^R]_\times).
\]

Rotation multiplication side determines the residual frame. This displayed convention is illustrative; use [hand_pose.py](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/runtime/actions/hand_pose.py) for the actual implementation. PD targets are not direct torques: stiffness, damping, solver rate, delays, and contacts mediate their effect.

The pinned Sharpa action uses 0.05 m wrist-translation scale, 0.15 rad wrist-rotation scale, 0.15 rad finger scale, and EMA parameter 0.3. Each hand supplies six wrist and 22 finger coordinates, producing 56 total. [Action configuration](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/action/residual_hand_pose.yaml)

Residual control has a foundational precedent in combining conventional controllers with learned corrections, including contact-rich robot assembly and improvement of imperfect nondifferentiable controllers. [Johannink et al., *Residual Reinforcement Learning for Robot Control*, 2018/2019](https://arxiv.org/abs/1812.03201); [Silver et al., *Residual Policy Learning*, 2018/2019](https://arxiv.org/abs/1812.06298)

An original implication is reachability: bounded residuals define the set of references the learner can correct. A wrist displaced beyond that region cannot be repaired by better optimization alone. Conversely, large residuals broaden exploration and weaken the prior. Action scales must be interpreted together with systematic reconstruction error, collision geometry, controller gains, and filtering.

For the conventional EMA \(\tilde a_t=\alpha a_t+(1-\alpha)\tilde a_{t-1}\), a step settles geometrically and the low-frequency delay grows as \(\alpha\) decreases. The numerical meaning of a named EMA parameter depends on its implementation convention. Never infer lag solely from the label `0.3`; the source specifies which coefficient multiplies current input.

## Policy learning: algorithms versus runtime architectures

### PPO and generalized advantage estimation

PPO alternates sampled interaction and minibatch optimization of a clipped surrogate:

\[
L^{\mathrm{clip}}(\theta)=\mathbb E_t\left[
\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],
\quad \rho_t=\frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|o_t)}.
\]

Clipping discourages excessive beneficial likelihood-ratio changes in this objective; it does not enforce a universal hard trust-region constraint. [Schulman et al., *Proximal Policy Optimization Algorithms*, 2017](https://arxiv.org/abs/1707.06347)

GAE combines temporal-difference residuals:

\[
\delta_t=r_t+\gamma V(z_{t+1})-V(z_t),\qquad
\hat A_t^{\mathrm{GAE}}=\sum_{\ell\ge0}(\gamma\lambda)^\ell\delta_{t+\ell}.
\]

The parameter \(\lambda\) trades variance against sensitivity to value-function error. [Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016](https://arxiv.org/abs/1506.02438)

Original implementation implications include distinguishing true failure termination from fixed-horizon truncation when bootstrapping, ensuring phase and assistance state are available wherever needed by the value function, and avoiding advantages that cross environment resets. These are general mathematical concerns, not claims that the pinned implementation mishandles them.

The pinned [PPO configuration](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/training/reference_ppo.yaml) uses actor and critic hidden dimensions `[1024, 512, 256, 128]`, 24 rollout steps, four minibatches, five epochs, clip 0.1, entropy coefficient 0.001, GAE \(\lambda=0.95\), gradient norm 1, and adaptive learning rate targeting KL 0.005. These are runtime choices, not architectural requirements of PPO.

With \(N\) worlds and \(H\) rollout steps, one collection batch contains \(NH\) transitions. Reducing worlds reduces samples per iteration unless horizon changes. A schedule keyed to iterations then advances after fewer transitions; preserving an iteration count does not preserve the learning protocol.

### SAC foundations and FlashSAC-specific architecture

SAC augments expected reward with entropy:

\[
J(\pi)=\mathbb E_\pi\sum_t\gamma^t
\left[r(z_t,a_t)+\alpha\mathcal H(\pi(\cdot|o_t))\right].
\]

Its off-policy design permits replay reuse. [Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, 2018](https://arxiv.org/abs/1801.01290)

The project's FlashSAC path is a JAX port with runtime-specific extensions. Its base configuration has two actor blocks of width 128, two critic blocks of width 256, expansion factor four, two critics, and 101 distributional atoms. The Sharpa recipe changes actor width to 256 and critic width to 512, uses eight updates per collection, and sets replay capacity 4,194,304. The distributional architecture and these settings must not be attributed wholesale to the foundational SAC paper. [Base configuration](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/training/reference_flash_sac.yaml); [Sharpa recipe](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/experiment/sharpa_flash_sac.yaml)

The runtime audit records discount 0.99, learning rate declining from \(3\times10^{-4}\) to \(1.5\times10^{-4}\), actor updates every two learner steps, target update coefficient 0.01, reward normalization, and a distributional return interval of ±5 in the base settings. Recipes override several values. A reproducible account requires the resolved recipe, not an isolated list of base defaults. [Runtime evidence](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md)

An original concern is nonstationarity from assistance decay: replay collected with a strong virtual controller differs from transitions collected without it. Explicit assistance context can help the critic represent these differences, but it does not make replay age irrelevant or prove that a nominal update ratio is optimal.

### Actor observations and privileged critic context

Asymmetric actor-critic learning allows a critic to use simulator state while an actor uses restricted observations, improving training without requiring all critic inputs at deployment. The foundational paper demonstrates image-based actors with privileged state critics. [Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, 2017](https://arxiv.org/abs/1710.06542)

This project's floating Sharpa actor is state-based. Its inputs include wrist pose, wrist motion, normalized finger coordinates and velocities, object poses, relative reference deltas, raw/processed actions, and contact information. Current-reference targets are used, without the general future-reference window available in the G1-specific route. Critic-only context includes assistance state, phase, and object velocity. This should not be described as an image-only deployment policy. [Observation configuration](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/observation/policy.yaml); [observation implementation](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/envs/observation.py); [runtime evidence](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md)

An original distinction is that using future participant-created references would be a reference-conditioning design decision, whereas using hidden organizer reference poses would be an unavailable-information violation. The local contract says organizer references are confidential. Evaluation input questions that remain open in that contract are not resolved by these papers.

## Assistance, resets, time, and robustness

### Training assistance must disappear from achieved motion

A schematic independent virtual-controller equation is

\[
w_t^{\mathrm{assist}}=\eta_t
\begin{bmatrix}
K_p(\bar x_t-x_t)-K_d\dot x_t\\
K_R\operatorname{Log}(R_t^\top\bar R_t)-K_\omega\omega_t
\end{bmatrix}.
\]

When \(\eta_t>0\), object tracking may be caused partly by an external training wrench. The policy must eventually sustain the behavior at \(\eta_t=0\). Good assisted reward is evidence of performance on the assisted task.

The pinned 250-million-step curriculum has assistance scales `1, .75, .5, .25, .1, .05, .025, .01, 0, 0, 0`, with thresholds specified in environment transitions. Its final stage beginning at 200,007,680 steps sets frame-zero reset probability 0.5. The final threshold is 250,003,456. FlashSAC indexes curriculum by environment steps; PPO's runner indexes its curriculum by iterations. These clocks must be preserved or deliberately re-derived when reducing resources. [Curriculum](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml); [runtime audit](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md)

Random reference resets expose later phases without requiring earlier success. Stabilization with assistance helps initialize contact. The original deduction is a distribution mismatch: the reset distribution can contain later states that the current unassisted policy cannot reach from frame zero. Mixing frame-zero resets addresses that distinction but does not itself establish success.

### Playback speed changes dynamics

Under \(x_\alpha(t)=x(\alpha t)\),

\[
\dot x_\alpha=\alpha\dot x,\qquad
\ddot x_\alpha=\alpha^2\ddot x.
\]

Half-speed motion reduces the acceleration required for the same geometric path by four. Contact durations, slip opportunities, damping effects, and feedback bandwidth relative to motion also change. Playback speed is therefore part of task dynamics, not simply output resampling.

The released floating Sharpa recipe defaults to half speed; this project's contract selects unit speed to align the requested dataset timeline. Reference resampling interpolates positions and joints, uses SLERP for orientations, and nearest-neighbor contacts. Any prospective comparison must hold this timing choice explicit. [Contract](../../../reports/contract.md); [resampling implementation](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/data/resampling.py)

### Domain randomization and consistent physical parameters

Foundational domain-randomization work randomizes rendering properties to transfer visual predictors from simulation. Its results should not be generalized into a guarantee for arbitrary physical parameter randomization. [Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, 2017](https://arxiv.org/abs/1703.06907)

An original prospective objective is

\[
\max_\theta\mathbb E_{\xi\sim p_{\mathrm{train}}}
[J(\pi_\theta;\xi)].
\]

Geometry changes must be coordinated with collision meshes, contact positions, reference frames, inertias, and support heights. A mass increase without sufficient contact friction or actuator capability may make a reference infeasible rather than merely harder. Randomizing mutually inconsistent parameters teaches a mixture of different or impossible tasks.

For this challenge's nominal-world evaluation, robustness is valuable insofar as it improves achieved motion under plausible reconstruction and physical uncertainty. It should not be assumed that broader randomization always improves the nominal metric.

## Whole-body methods and alternative policy families

### Contextual control and recovery

VideoMimic jointly reconstructs humans and environment from ordinary videos and learns contextual humanoid control. Its relevance here is consistent support geometry and embodied interaction with a reconstructed scene. It does not specify the floating-hand architecture used by this challenge. [Allshire et al., *Visual Imitation Enables Contextual Humanoid Control*, 2025](https://arxiv.org/abs/2505.03729)

PHC introduces progressive multiplicative control and failure recovery for large motion collections. It supplies an analogy for scaling tracking capacity and learning recovery, while avatar balance and whole-body imitation differ materially from precise achieved object trajectories. [Luo et al., *Perpetual Humanoid Control for Real-time Simulated Avatars*, 2023](https://arxiv.org/abs/2305.06456)

Original project implication: whole-body methods become directly relevant if the embodiment or observation contract requires locomotion and reachability. They are not automatically necessary when the chosen contract permits floating hands. Additional model components impose data, compute, and integration requirements that need their own justification.

### Behavior cloning requires robot action labels

A schematic cloning objective is

\[
\min_\theta\mathbb E_{(o,a)\sim D}[-\log\pi_\theta(a|o)].
\]

Human video does not directly provide robot torques, PD targets, or valid residual actions. A simulator-grounding stage can generate these labels, but then physical correctness, observation schema, controller gains, filtering, and action normalization become part of dataset provenance. A policy that clones nominal reference commands learns a different target from one cloning successful feedback-corrected actions.

### Diffusion Policy and ACT are contextual alternatives

Diffusion Policy models conditional robot action sequences through denoising diffusion and receding-horizon execution. Its temporal convolutional and transformer formulations address multimodal action distributions after valid demonstrations exist. [Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023](https://arxiv.org/abs/2303.04137)

ACT learns generative action chunks with a transformer from teleoperated robot demonstrations. Its original precision-manipulation results do not establish conversion of arbitrary human video into executable robot actions without grounding. [Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023](https://arxiv.org/abs/2304.13705)

The original synthesis is that both methods may later consume physically grounded trajectories, compress controllers, or study cross-episode generalization. They do not automatically replace reconstruction, retargeting, collision modeling, or the production of valid achieved object motion. Action-chunk horizon also trades temporal consistency against feedback latency, which matters during slip, contact loss, and insertion.

## Resource implications and prospective questions

The pinned runtime documents a 48 GiB operating point, whereas this project has one RTX 5080. Training feasibility on the local GPU remains unestablished by these papers or by source inspection. Replay is device-resident. Its source gives

\[
\text{bytes per transition}=2D_o b_o
+4(D_a+D_{\mathrm{objective}}+2D_{\mathrm{critic\ context}})+2.
\]

Capacity multiplies that cost before simulator buffers, models, optimizer state, batches, and compilation memory. Observation dimension grows with object and contact structure. Float16 replay observations alter storage but do not automatically halve every allocation. [Replay implementation](../../../upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/training/flash_sac/replay.py); [runtime resource audit](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md)

Reducing worlds affects simulator memory, rollout distribution, update opportunities, and curriculum clocks. Reducing replay affects experience age and reuse. Shrinking a network changes capacity. None is a purely mechanical resource substitution with guaranteed equivalent learning. Existing dependencies and read-only external environments remain unchanged by this chapter.

The literature supports the following prospective questions. They are not executed experiments, authorized policy changes, or a revision selected from previous holdout outcomes.

1. Which geometric and timing errors place successful contacts outside the bounded residual region?
2. Are contact normals, contact activity labels, and object-frame conventions reliable enough for support matching?
3. Does averaging directional rewards hide a small set of mechanically decisive deficiencies?
4. How do reference activity gates treat missing versus genuinely absent contact during occlusion?
5. Does frame-zero unassisted execution succeed after training with assisted random-phase resets?
6. Which observed errors come from references, collision decomposition, inertia, controller bandwidth, or learning?
7. What resource reduction preserves explicit curriculum and update semantics on the actual GPU?
8. Would any downstream cloning comparison use successful robot actions with complete provenance and an independently fixed assessment protocol?

An unbiased future assessment requires an independently fixed protocol. The completed campaign's held-out outcomes cannot guide these choices or synthetic enrichment ratios. All proposed mixtures must be selected on real tuning evidence with synthetic provenance explicit.

## Primary bibliography

- Zhu, Xinghao, et al. **Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration.** 2026. arXiv:2607.00033. [Paper](https://arxiv.org/abs/2607.00033). The retrieved v1 header displays 22 June 2026; retain the identifier rather than inferring a date from its numeric prefix.
- Mandi, Zhao, et al. **DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation.** 2025 preprint; ICML 2026. arXiv:2505.24853. [Paper](https://arxiv.org/abs/2505.24853).
- Li, Kailin, et al. **ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning.** CVPR 2025. arXiv:2503.21860. [Paper](https://arxiv.org/abs/2503.21860).
- Lum, Tyler Ga Wei, Olivia Y. Lee, C. Karen Liu, and Jeannette Bohg. **Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration.** 2025. arXiv:2504.12609. [Paper](https://arxiv.org/abs/2504.12609).
- Johannink, Tobias, et al. **Residual Reinforcement Learning for Robot Control.** 2018 preprint; ICRA 2019. arXiv:1812.03201. [Paper](https://arxiv.org/abs/1812.03201).
- Silver, Tom, Kelsey Allen, Josh Tenenbaum, and Leslie Kaelbling. **Residual Policy Learning.** 2018 preprint; 2019 revision. arXiv:1812.06298. [Paper](https://arxiv.org/abs/1812.06298).
- Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. **Proximal Policy Optimization Algorithms.** 2017. arXiv:1707.06347. [Paper](https://arxiv.org/abs/1707.06347).
- Schulman, John, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. **High-Dimensional Continuous Control Using Generalized Advantage Estimation.** 2015 preprint; ICLR 2016. arXiv:1506.02438. [Paper](https://arxiv.org/abs/1506.02438).
- Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel, and Sergey Levine. **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.** ICML 2018. arXiv:1801.01290. [Paper](https://arxiv.org/abs/1801.01290).
- Pinto, Lerrel, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, and Pieter Abbeel. **Asymmetric Actor Critic for Image-Based Robot Learning.** 2017. arXiv:1710.06542. [Paper](https://arxiv.org/abs/1710.06542).
- Tobin, Josh, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.** 2017. arXiv:1703.06907. [Paper](https://arxiv.org/abs/1703.06907).
- Allshire, Arthur, et al. **Visual Imitation Enables Contextual Humanoid Control.** 2025. arXiv:2505.03729. [Paper](https://arxiv.org/abs/2505.03729).
- Luo, Zhengyi, Jinkun Cao, Alexander Winkler, Kris Kitani, and Weipeng Xu. **Perpetual Humanoid Control for Real-time Simulated Avatars.** 2023. arXiv:2305.06456. [Paper](https://arxiv.org/abs/2305.06456).
- Chi, Cheng, et al. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.** RSS 2023; extended revision 2024. arXiv:2303.04137. [Paper](https://arxiv.org/abs/2303.04137).
- Zhao, Tony Z., Vikash Kumar, Sergey Levine, and Chelsea Finn. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.** 2023. arXiv:2304.13705. [Paper](https://arxiv.org/abs/2304.13705).
- NVIDIA Isaac. **video_to_data / FlashCHORD source.** Pinned commit `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, inspected read-only. [Upstream pinned tree](https://github.com/nvidia-isaac/video_to_data/tree/aee95f25ae0943f335d8bb42fa95b6a9acfa266f); [local runtime report](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md).
