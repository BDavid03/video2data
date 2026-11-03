# Proposed contact-objective and achieved-behaviour comparison

Source-reading design only, 2026-10-18. This document grants no execution,
simulation, installation, checks or upload permission. Runtime gates remain in
force. No new generic policy framework or backend is proposed.

## Exact objective boundary

Pinned `aee95f25` already owns contact-wrench support. Read
`configs/objective/tracking.yaml`,
`configs/curriculum/tracking_250m_mixed_reset.yaml`,
`objectives/tracking.py` and `objectives/wrench.py` in its FlashCHORD source.

The released contact-aware arm retains the pinned recipe. For a proposed
motion-only objective, preserve all scene contacts and change these keys:

| Key | Motion-only proposal |
|---|---|
| `objective.contact_wrench_support.enabled` | `false` |
| `objective.contact_wrench_support.weight` | `0.0` |
| `objective.missed_contact.enabled` | `false` |
| `objective.missed_contact.weight` | `0.0` |
| `objective.unintended_contact.enabled` | `false` |
| `objective.unintended_contact.weight` | `0.0` |
| `objective.force_closure.enabled` | `false` |
| `objective.force_closure.weight` | `0.0` |
| `objective.contact_force_l2.enabled` | `false` |
| `objective.contact_force_l2.weight` | `0.0` |

For **every** `curriculum.stages[i].objective_weights`, also set
`contact_wrench_support`, `missed_contact`, `unintended_contact`,
`force_closure` and `contact_force_l2` to zero. The pinned curriculum aliases a
weight mapping across stages; some terms interpolate objective weights while
others are fixed literals. Changing a top-level weight alone can leave contact
rewards active or let a later stage restore them. Preserve all motion,
termination and action-regularization terms and the assistance schedule.

Contact fields also appear in observations. Holding those observations fixed
isolates the objective; it does not establish a policy with no contact inputs.
Any observation ablation is a separate experiment. Zeroing scene arrays instead
would change observations and reference targets, confounding this comparison.

`PolicyRequest.recipe_variant` now names `contact_aware` (the pinned values) or
`motion_only`. `recipes.py` copies all eleven stages independently and explicitly
sets every contact weight, plus each top-level enabled/weight pair. Compose
records these overrides and the complete resolved job; train/checkpoint admission
bind the same configuration. No arbitrary objective override API is added.
Source implementation is unrun and runtime compatibility is unestablished.

## Bounded stages, stopping and clock

Training episodes remain 0/11/12/18/21/23/26/31. This downstream development
protocol uses 39/42 with all derived views together. Episode 41 is forbidden;
43/44 were consumed and supply no fresh final assessment.

1. Prerequisite admission: usable scene/contact/mesh artifacts; approved policy
   environment; writable Z: output/cache; explicit simulation permission.
   Existing compose/inspect/check commands remain prohibited absent permission.
2. `ZERO_UPDATE_BASELINE`: existing `zero_update` budget, 64 worlds and 64 total
   steps, seed 31, resting and replay for each of two development episodes:
   four runs, 256 total environment steps. This saves no learned success claim.
   Runtime may itself fail; no automatic dependency repair follows.
3. Tiny scaled feasibility: proposed complete budget, 64 worlds, 128,000 steps,
   replay warmup 6,400, capacity 16,384, observation float16, updates per
   collection 0.125, checkpoint interval 5,120, no replay persistence. These
   follow the existing budget arithmetic in the new `contact_feasibility` entry.
   Generate all eleven curriculum thresholds from existing `FRACTIONS` with
   downward whole-world rounding; assistance reaches zero around 55% of budget.
   Two objectives × two episodes × seed 31 = 512,000 steps. Memory fit and
   usefulness are unmeasured; no claim of recipe validation follows.
4. Only if the feasibility stage completes without admission/resource failures,
   repeat the same frozen recipe at seeds 47/73: 1,024,000 extra steps. Freeze
   this continuation condition before outcomes. Maximum including baselines is
   1,536,256 steps, below the absolute ceiling 1,600,000 steps and two GPU hours.
   Stop at either ceiling; preserve incomplete runs and assistance status.

The released 10,240,000-step `scaled_template` remains an arithmetic reference,
not this study's budget. No larger run is automatically eligible.

Hold reference/control at 20 Hz, motion speed 1, complete frame range. Assess
the last scheduled checkpoint in one explicit-reset world from frame 0, with
the pinned evaluator and its complete output. Record assistance-free duration;
a tiny nonlearning run can establish only feasibility or failure. Wall-time
ceiling includes compilation/training/rollout; no repeated retries to hide it.

Use pinned-kit MPPE as primary and AUC/SP-SR/MP-SR/RPE as secondary. Preserve
all object/frame denominators and every failure, report valid support separately,
and retain metric units/alignment/point-set identity. Reconstruction self-score
is teacher-like agreement, not independent achieved-task ground truth.

No runtime evidence was produced. See
[item 13](../../../docs/experiments/13-achieved-behaviour.md) for remaining blockers.

## Source build and immutable planning boundary

`ComparisonRequest` requires resting/tracked inspection references for 39/42,
in that order. `build_comparison` returns a source-only plan with all sixteen
attempts and no producer reads. `persist_comparison` additionally admits and
pins both references, their scene producers, source, complete budgets and this
protocol before writing a completed `policy_training.comparison` run with
`protocol.json`. Its new `comparison_authorized` gate is false and canonical
storage is required. It never composes, trains, scores or simulates.

The payload commits baseline seed 31, trained seeds 31/47/73, conditional
extension independent of favourable scores, complete reset and clock semantics,
final-checkpoint-only selection and the aggregate 1,600,000-step/7,200-second
ceilings. The 1,536,256 planned steps include four baseline runs. Aggregate
ceiling enforcement belongs to a future authorized campaign runner; this
planner does not claim to meter GPU execution. Existing per-episode train does
not automatically run this comparison or enforce a multi-run ceiling.

Use `PolicyRequest.final_checkpoint_only=true` for comparison selection. A
failed final scheduled checkpoint is a failed attempt; comparison cannot fall
back to an earlier successful checkpoint. The legacy default selection policy
remains available outside this fixed research protocol.
