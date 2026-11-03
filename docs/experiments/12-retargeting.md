# 12 — Contact-aware retargeting

Status: optional support-aware interval transfer and measured-error envelope
proposal implemented from source, with independent investigation/comparison
design retained; runtime
assessment blocked by items 07–09/11, simulation authorization, an approved policy
environment and writable canonical Z: storage. No operation or check was run.

Hypothesis: preserving object-centric contact effects can improve achieved object
motion over motion-only imitation, provided scene contacts and collision geometry
are credible. This is unestablished for this project.

## Existing implementation and evidence

Reuse `track3/features/scene_assembly/_contacts.py`, retargeting's export/assets/
retarget/support/inspect operations, and policy training/rollout. The scene
already supplies 16 MANO-link contact slots per hand, distinct from 21 joints.
Positions/normals/body IDs describe sampled geometric proximity, not measured
forces or verified contact intervals. Retargeting preserves upstream link order,
metres, scalar-first object quaternions and official object order.

Pinned FlashCHORD `aee95f25` already implements the relevant wrench machinery in
`robotic_grounding/flash_chord/src/flash_chord/objectives/tracking.py` and
`objectives/wrench.py`. Its tracking recipe uses 512 basis directions, eight
friction-cone edges, assumed friction 0.1 and basis seed 0. Do not duplicate it.
Its floating Sharpa route supports two objects; floating wrists remove arm
reachability constraints but not hand joint/velocity, collision or physical
contact constraints. All project retargeting/policy operations remain unrun.

## Approach dispositions

| Approach | Disposition and evidence |
|---|---|
| Pinned FlashCHORD | Retain integration baseline and existing feature-owned implementation. Existing code accepts geometric contacts and computes reference wrench supports; this establishes an integration path, not successful control. Pinned source license assigns Apache-2.0 to code and CC-BY-4.0 to documentation/skills; robot asset rights remain separately unresolved. |
| CHORD | Investigated; reuse the pinned wrench-support implementation instead of importing another stack. The official research describes object-centric force/torque authority and contact ablations. External benchmark improvements do not establish project superiority. |
| SPIDER | Investigated, defer executable integration. Its physics-retargeting pipeline adds dataset conversion, decomposition, IK and simulator optimization. Python 3.12/toolchain installation and simulation need authorization. Its released saved-state replay is explicitly not control-replay dynamics evidence. Source is CC-BY-NC; commercial suitability is unresolved. |
| Motion-only objective | Planned matched ablation, not an implemented or assessed candidate. Keep scene/contact artifacts fixed and alter objective configuration explicitly. |
| Uncertainty-derived randomization | Implemented a disabled measured-error admission/proposal adapter; empirical pose-error bounds retain units, support and hashes, reject out-of-cap errors, and never create a distribution or sampler. No eligible measured input exists. Fixed friction/mass assumptions, solver residuals and member disagreement remain ineligible. |

Primary sources: [CHORD](https://nvidia-isaac.github.io/video_to_data/chord/),
[SPIDER](https://github.com/facebookresearch/spider),
[SPIDER license](https://raw.githubusercontent.com/facebookresearch/spider/main/LICENSE).
Online main branches are research context, not replacements for pinned code.

## Matched comparison and limitations

The exact objective keys and staged resource ceiling are in
[policy training's protocol](../../track3/features/policy_training/research-protocol.md).
Keep reconstruction, object assets, support, reference timing, policy architecture,
seed, replay, training steps and checkpoint rule identical. Compare achieved
trajectories using the pinned official kit; do not compare only retargeted joints.
Track missing contact support, object drops, penetration, contact loss and
recovery. No reference scans or diagnostic pose labels enter production fitting.

Before runtime work: obtain usable upstream scene/hand/mesh evidence; preserve
unrefined artifacts; establish contact visibility and ambiguous intervals; admit
the policy environment/dependencies and canonical storage. Do not claim that
IK, physically feasible motion, uncertainty propagation or better task behaviour
has been demonstrated.

Next independent item: 13's achieved-behaviour protocol. Runtime remains blocked.

## Source buildout disposition

The existing export/assets/retarget/support/inspect chain now optionally consumes
`hand_motion.contacts`, preserving exact frame/PTS timing and the existing pinned
16-link Parquet format. Admission requires explicit independent contact support,
confirmed physical contact, accepted status, sufficient stated confidence and no
ambiguity; rejected overlaps conservatively suppress contact slots. It retains
full diagnostic evidence and per-interval reasons in a copied, hash-tracked
sidecar. No geometric proximity is promoted to physical contact by this adapter.
The hand producer's current unaccepted/unconfirmed hypotheses are diagnostic
only and would suppress reference contacts if transfer were explicitly selected.
That limitation is visible, rather than replaced with a fabricated confidence.

Measured-error proposals support only world-frame signed translation (m) and
rotation-vector (rad) errors, with at least two individually identified independent
geometric measurements per component. The source producer must record every
hashed measurement input; source PTS and object identity must match the scene.
Out-of-cap envelopes require reconstruction review. No mass/friction ranges or
probability models are invented. The artifact remains a proposal with sampling
disabled. No current joint/hand producer supplies calibrated error measurements.

Both optional gates default false, alongside the unchanged false operation flags.
The new paths admit only public training/tuning episodes and refuse diagnostic
label provenance. No test, import, validation command, retarget, simulation,
dependency installation or runtime artifact creation occurred. Exact blockers:
missing independently supported contacts/error measurements, scene/contact hand
lineage compatibility, unapproved policy environment/assets, unauthorized
simulation and inaccessible writable canonical Z: storage. Pinned FlashCHORD's
wrench computation remains the sole wrench engine; CHORD/SPIDER dispositions
above are unchanged. Motion-only/contact objective configuration belongs to
policy training's matched comparison.
