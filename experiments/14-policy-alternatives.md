# 14 — End-to-end and hybrid policy alternatives

Status: independent feasibility investigation complete; execution deferred.
Item 13 has no achieved outcomes, so no project policy ranking is established.
No dependencies, checkpoints or datasets were installed/downloaded for this item.

Hypothesis: pretrained policy priors might improve control only if their
observation/action semantics and adaptation data fit the target embodiment.
Larger pretrained capacity does not establish a benefit over explicit geometry.

## Shared data and evaluation boundary

The existing route uses floating Sharpa hands: 56 normalized residual controls
relative to reference wrist/finger trajectories at 20 Hz. Human RGB, estimated
hand joints, pseudo-depth and IK targets are not successful executed robot-action
demonstrations. Eight training episodes and two tuning episodes do not establish
a sufficient adaptation corpus. Episode 41 remains excluded; consumed holdout
43/44 cannot select recipes. Any learned action normalization uses training only.

Every challenger needs a documented transform for coordinates, control mode,
action units, wrist/finger ordering, observation timing and chunk execution.
Its achieved output must still obey the pinned evaluator/Submission Kit contract;
native policy-server output or generated future video is not a submission
trajectory. Keep pretrained/adapted approaches distinct from original models.

## Explicit dispositions

| Approach | Fit, resources, license and disposition |
|---|---|
| OpenVLA / OFT | Investigated; defer executable adaptation. Base OpenVLA predicts seven-dimensional robot actions, unlike 56 residual Sharpa controls. OFT supports continuous/bimanual action fine-tuning but does not supply the project adapter or action corpus. Published OpenVLA LoRA example needs about 27 GB minimum; full tuning is far larger. Code MIT; weights inherit Llama-2 restrictions. Requires separate pinned transformers/timm/tokenizer/attention stack, approved environment and genuinely successful robot actions. |
| π₀, π₀-FAST, π₀.₅ through openpi | Investigated; defer. Requires embodiment-specific state/action transforms, normalization and robot observations/actions. Published GPU estimates: inference >8 GB, LoRA >22.5 GB, full tuning >70 GB, with Ubuntu support. PyTorch path does not currently support LoRA and documents changes to installed transformers files; existing external environments must remain read-only. Code Apache-2.0; individual checkpoint/base-model terms need a frozen inventory. |
| GR00T N1.7 | Investigated; defer. Official main now differs from N1.6: new backbone, expanded 132-dimensional state/action capacity and 40-step horizon. Capacity is not semantic compatibility with Sharpa. Needs embodiment tag/transforms and LeRobot plus modality metadata. Published inference minimum 16 GB; tuning recommendation 40 GB or more, gated backbone access and a separate toolchain. Code Apache-2.0; weights NVIDIA Open Model License. Freeze branch, checkpoint and gated backbone before any proposal. |
| Cosmos Policy, world/action-model route | Investigated; defer. Joint policy/future-state/value training requires robot proprioception, observations, executed actions and appropriate value supervision. Published inference examples use about 6–9 GB, while the smallest listed training example is eight 80 GB H100s for 48 hours and about 185 demonstrations. Neither memory figures nor external success rates establish project feasibility. Code Apache-2.0; each weight/base-model license must be pinned separately. |
| Video/world-model planning without robot adaptation | Reject as a direct substitute for achieved physical trajectories: plausible future pixels do not define feasible 56-dimensional actions or establish contact/collision accuracy. Revisit only with an admitted action/dynamics adapter and a declared achieved-outcome comparison. |
| Reconstruction → existing FlashCHORD | Retain incumbent integration route. It is implemented, feature-owned, embodiment-aware and already accepts scene contacts; achieved performance remains unassessed. No extra policy stack is justified before item 13. |
| Reconstruction → successful-rollout distillation or residual controller | Investigated; defer until successful achieved rollouts supply state/action/outcome records. Reuse existing embodiment and geometry instead of inventing labels. Keep this a separate bounded adaptation track with fixed data/compute and official achieved metrics. |

Published resource figures describe external reference workloads, not measured
fit on this machine. Total adaptation cost must include video decoding,
observation rendering, dataset conversion, action-label production, normalization,
training, compilation, policy inference and simulator rollout. Inference fit alone
does not establish training feasibility.

## Primary evidence and reproducibility limits

- [OpenVLA official README](https://raw.githubusercontent.com/openvla/openvla/main/README.md): action interface, OFT direction, memory and inherited weight licensing.
- [openpi README](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/README.md) and [code license](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/LICENSE): model families, adapters, requirements and PyTorch limitations.
- [GR00T official README](https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/README.md) and [model card](https://huggingface.co/nvidia/GR00T-N1.7-3B): embodiment/data interfaces, hardware and code/model license distinction.
- [Cosmos Policy official recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html), [code license](https://raw.githubusercontent.com/NVlabs/cosmos-policy/main/LICENSE) and [LIBERO model card](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B): robot modalities, joint objectives, training resources and separate weight identity.

These live sources were investigated on 2026-10-18; no exact challenger source
commit, checkpoint digest or full dependency lock was adopted. Do not present
unfrozen live branches as reproducible project candidates. Model/base-model and
training-data terms need separate review for the chosen artifacts; a permissive
code license does not settle all three.

Exact blockers: item-13 achieved evidence and successful action data absent;
new production dependencies need approval; simulation is unauthorized; canonical
Z: runtime output is not writable in this session. Revisit after an immutable
baseline, credible adapter/data manifest, exact license/weight inventory and
bounded resource proposal exist. No project performance or superiority claimed.

Next independent item: 15's reconciliation of already completed optimization.
