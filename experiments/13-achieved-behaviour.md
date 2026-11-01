# 13 — Achieved robot behaviour

Status: bounded protocol specified; achieved-rollout assessment blocked. No
simulation, scoring, inspection command, test or dependency experiment ran.

Buildout update: implemented the disabled feature-owned
`policy_rollout/achieved_comparison.py` request/handler and `compare-achieved`
command, consuming the policy-training owner's immutable comparison receipt.
The receipt is checked before achieved outcomes; the consumer requires matching
episode/seed/budget/objective/retargeting/checkpoint lineage and the common pinned
tracked reconstruction. It retains every planned attempt, explicit failures and
unassessed extensions, full object-frame denominators, per-object metrics,
matched objective pairs and worst completed attempts. The receipt must predate
the evaluator output. No choices are fitted or promoted by this report.

Extended the existing disabled self-score implementation with per-object
pinned-kit metrics, filled-reference coverage and frame MPPE after one whole-run
alignment. The official mean-object MPPE is retained; per-object RPE is explicitly
undefined. Previously completed self-scores are never overwritten or recomputed
automatically. The offline consumer exports immutable achieved-pose references
for the diagnostic owner and reads no public pose labels.

Event interval attribution has a source contract; the current frozen producer
has no admitted event intervals, so contact-loss/drop/recovery evidence remains
unavailable rather than inferred from position error. Existing-policy and
perception-ablation outcomes remain deferred until admitted checkpoints and
separately frozen controlled variants exist. All new execution flags are false.
No run, scoring, check, import, dependency change or runtime output was produced.

Hypothesis: with reconstruction fixed, contact-aware objectives improve achieved
object trajectories relative to motion-only imitation at matched resources.
Secondary controlled comparisons would attribute perception contributions without
changing geometry, identity, depth and contacts together.

## Reuse and known contract hazards

Reuse `policy_training/` budgets/composition/checkpoint admission,
`policy_rollout/` immutable evaluator output and failure handling, and the pinned
Submission Kit metric implementation. These slices are implemented but unrun;
there are no existing successful policies or achieved rollouts to compare.

The scored poses are achieved simulated object motion, not reconstructed
demonstration poses or IK targets. FlashCHORD uses 56 residual Sharpa actions,
20 Hz control and 100 Hz physics. The project's 20 Hz reference and
`task.motion_speed=1.0` avoid the released half-speed recipe's roster mismatch.
The rollout records row 0 after the first control step, an unresolved organiser
timing interpretation. Runtime `mppe_cm` takes the worst object; the kit's MPPE
averages objects. Use the pinned kit, not a similarly named runtime metric.

## Concrete proposed protocol

The executable boundary and exact staged budgets are documented in
[the feature-owned protocol](../track3/features/policy_training/research-protocol.md).
This is a research proposal, not authorization or a new runtime configuration.

- Episodes 39/42 only; seeds 31 initially, 47/73 only after the feasibility stage.
  Episode 41 is excluded. No outcome from consumed 43/44 selects settings.
- First produce resting and zero-update replay baselines, explicitly labelled
  no optimizer updates. An existing-policy arm is unavailable until an immutable
  checkpoint exists; do not substitute an invented result.
- Then compare motion-only and contact-aware objectives at 128,000 environment
  steps per arm/episode/seed, 64 worlds. One-seed feasibility first; extension
  to three seeds is conditional and fixed in advance. Total campaign ceiling
  1,600,000 environment steps and two GPU hours, whichever comes first. Stop on
  memory/admission failure; no automatic larger run or installation.
- Preserve the same reconstructed reference, assets, support, clocks, reset
  states, replay parameters and assistance schedule. Select the final scheduled
  checkpoint, without choosing a favourable evaluation seed or checkpoint.
- One evaluation world; explicit reset at frame 0; complete duration. No
  simulation-based claim if training stops before assistance reaches zero.
- Primary MPPE with pinned official definitions; secondary AUC, SP-SR, MP-SR,
  RPE. Report full roster denominator, valid coverage, failures and per-object,
  per-episode values. Conditional scores on valid attempts must retain the
  failed-attempt count; do not silently drop failed worlds.
- Record drops, contact loss, recovery and worst failures alongside aggregates.
  Public-label diagnostics are separate from reconstruction self-score and never
  change production outputs. Compare label-free rankings with diagnostic
  rankings only under the frozen protocol; two tuning episodes give weak support.

Run identity must include scene/mesh/contact hashes, processed Parquet,
checkpoint and resolved recipe, budget counters, seeds, simulator version,
evaluator hashes, reference clock, achieved output and failure records. No
uploads or remote logging are proposed.

## Disposition and blockers

Resting, replay, trained baseline, contact ablation, perception ablations and
official achieved scoring are all designed but experimentally unassessed.
An existing-policy baseline is unavailable. Perception ablations follow a stable
item-12 candidate, one variable at a time, under a separately bounded request;
they do not expand this feasibility ceiling.

Exact blockers: simulation is not authorized; policy environment installation
still needs approval; canonical Z: storage is read-only in this session;
scene/contact prerequisites lack independent assessed evidence. The small budget
may be insufficient for learning, and memory fit is unmeasured. Such failure is
a feasibility result, not evidence against contact guidance. Fresh unbiased
assessment needs an independently fixed protocol because 43/44 were consumed.

Next independent item: 14's policy feasibility dispositions.
