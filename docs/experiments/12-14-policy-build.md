# 12–14 policy source build

Implemented source only, 2026-10-18. No checks, model execution, simulation,
training, dependency operation or runtime artifact creation occurred.

Reused the existing policy vertical slice, pinned FlashCHORD curriculum/objective,
budget arithmetic, reference admission, composition lineage, checkpoint admission
and disabled runtime configuration. The contact-aware recipe retains upstream
values. Motion-only disables and zeroes all five contact terms at top level and
in every copied curriculum stage, including aliased stages and literal weights.
Contact observations, reconstruction, reset probabilities, assistance and motion
terms remain fixed. Complete overrides and resolved jobs enter existing lineage.

The `contact_feasibility` budget is 64 worlds, 128,000 steps, 6,400 warmup,
16,384 replay capacity, 0.125 updates/collection and 5,120-step checkpoints,
with eleven scaled thresholds. These are source arithmetic, not measured fit.
`ComparisonRequest`/`build_comparison` construct the fixed 39/42 roster in
`policy_training/comparison.py`; disabled persistence pins admitted resting and
tracked inspections plus their scenes. Baselines seed 31, then two objective
arms at 31, with conditional 47/73 continuation, total 1,536,256 planned steps.
The declared ceilings remain 1,600,000 steps/two GPU hours including rollout.
No orchestrator executes or meters those ceilings. The final-checkpoint-only
selection option forbids outcome-driven fallback for this protocol.

`hybrid.py` adds the successful-achieved-rollout dataset contract and strict 56D
normalized residual admission, with no seven-to-56-dimensional invention. It
requires pinned robot controls, pre-action observations, timestamps, achieved
poses, success evidence, reference/checkpoint identities and train-only fitted
observation statistics. A tiny original 64-hidden-unit PyTorch MLP uses the
existing learning stack; no fit or FlashSAC export path is claimed. Actual
payload hashing/row admission and an eligible successful corpus are required
before a fitting adapter is justified. Human trajectories/IK/pseudo-depth do
not satisfy robot-action labels. OpenVLA/OFT, openpi, GR00T and Cosmos retain
their explicit provision/data/embodiment blockers from item 14.

Failures/limitations: no successful policy or achieved action corpus exists;
Hydra/list override and checkpoint compatibility remain unrun; assistance-free
behaviour, learning efficacy, memory fit and resource compliance unmeasured.
Canonical Z: output is unwritable here; simulator execution is unauthorized,
policy environment/assets unprovisioned, new dependencies unapproved. No new
43/44 outcome access or selection occurred; 41 is excluded throughout.

Disposition: implement reviewable named recipes, frozen comparison plans and
explicit hybrid contracts; defer all outcome comparisons and production
promotion. Next eligible runtime work requires admitted scene/contacts/assets,
canonical writable storage and separate simulation authorization. No additional
policy stack is justified until the incumbent has achieved evidence.

Source review follow-up: frozen protocol pins now include every policy-training
Python source, the budget and protocol documentation. Tiny-policy materialization
requires the owning canonical/environment/episode authorization and confines
PyTorch/compiler/cache/temp/bytecode paths before its lazy import. No fitting or
FlashSAC export adapter exists; PyTorch availability in the approved policy
interpreter is a separate unresolved prerequisite, never an install request.
