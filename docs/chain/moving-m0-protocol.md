# M0 protocol: do D1's partial policies transport the object? (`COMPLETION-PLAN.md` W3; fixed before any number below exists)

## Question

D1 stopped at its wall-clock limit, so it was declared NO-GO on completion and its policies were never rolled out. Two partial
policies exist. M0 scores them once, through the official evaluator, and measures for the first time whether a policy on a
hand reference carries the object along the reference. It answers one question about one task: the shipped `tissue_box_simple`
example. It opens no challenge episode and produces nothing for a submission.

## What was already known when this was written (disclosure)

From `dynamic-feasibility-outcome.md`: S (object held still) mean position 6.889 cm, ADD-AUC 0.4591; Z (zero updates) 7.012 cm,
0.4280; L1 seed 0 7.026 cm, 0.4276; L1 seed 1 6.954 cm, 0.4471; reference travel 17.4 cm; Z moved the box at most 1.5 cm, L1 seed 1
0.72 cm. From `dynamic-scaling-outcome.md`: both D1 seeds timed out at 21.6M and 22.6M transitions with training VOC 0.05.
No transport value has ever been computed for any arm. The author has seen no D1 rollout, because none exists.

## Inputs, fixed now

| Arm | Checkpoint or export | Note |
| --- | --- | --- |
| D1p-seed0 | `.runtime/tmp/dynamic-scaling-seed0-k62pzjmq/training/policy_21580800.safetensors` | the latest policy in that scratch directory |
| D1p-seed1 | `.runtime/tmp/dynamic-scaling-seed1-kzy3z0pr/training/policy_22604800.safetensors` | the latest policy in that scratch directory |
| Z, L1-seed0, L1-seed1 | existing `outputs/dynamic_feasibility/runs/*/trajectory.parquet`, read only | no re-run |
| S | computed: the reference's first pose held for all frames | transport is 0 by construction |

Earlier checkpoints in those directories are never scored, so there is no best-of selection. Each checkpoint's SHA-256 and metadata
(`voc_scale`, steps) go in the record before its evaluation starts, and the policy file is copied beside the record, because
`.runtime/tmp` is scratch.

## Operation

An evaluation-only operation in `track3/features/dynamic_feasibility/`, under `GpuLease`, with the slice's `environment()`. The
training handler and its nonzero-VOC refusal do not change. Each checkpoint runs once with the kit's `evaluate_policy.py` and the
settings D0 used: `evaluation.world_count=1 evaluation.reset_mode=explicit evaluation.start_frame=0`. The evaluator removes
assistance itself (`harness.py:230,438,458`); the record cites what shows it did. Allowance: 5 GPU-minutes in total. A run that
crashes without an export may be retried once, disclosed. A run with an export is never repeated.

## Metrics

On raw world-frame positions, frames `t = 0..389`, with `Da_t = a_t - a_0` and `Dr_t = r_t - r_0`:

- **Transport** `T = sum_t(Da_t . Dr_t) / sum_t(|Dr_t|^2)`. It is displacement gain, the through-origin regression slope, and not a
  temporal tracking score: perfect following gives `T = 1`, a still object gives 0, but `T = 1` does not imply following. The reference
  lifts and returns (z 1.0639 m at frame 0, 1.2354 m at frame 210, 1.0755 m at frame 389), so an object parked part-way earns credit on
  the way up and on the way down.
- **Motion score** `Q = 1 - sum_t(|D15a_t - D15r_t|^2) / sum_t(|D15r_t|^2)`, `t = 0..374`, where `D15x_t = x_(t+15) - x_t` (half a second at
  30 Hz). A still object gives 0 and exact motion gives 1. It is negative when the object's motion adds more error than standing still.
- **Perpendicular error**: mean over frames with `|Dr_t| >= 1 cm` of `|Da_t - (Da_t . u_t) u_t|`, in cm, where `u_t = Dr_t / |Dr_t|`.
- Beside them, without a gate of their own: mean and P90 position error, mean orientation error, ADD-AUC after the kit's alignment
  and raw, largest achieved displacement in cm, the end-frame achieved and reference displacement, and the evaluator's
  `reference_end_fraction`, success rate and completion fields. Reaching the reference's end is not tracking: D0 recorded end fraction
  1.0 with success rate 0.0.

`T`, `Q` and the thresholds below are prospective engineering choices. No recorded distribution of them exists to calibrate against.

## Validity checks (any failure blocks the verdict; the denominator is never reduced)

1. **Metric identities, checked on the real task reference**: `a = r` gives `T = 1`, `Q = 1`, perpendicular 0; `a = r_0` gives
   `T = 0`, `Q = 0`, perpendicular 0; `a = r_0 + 0.5 Dr` gives `T = 0.5`, `Q = 0.75`, perpendicular 0; all within 1e-9.
2. **Reproduction**: recomputed mean position and ADD-AUC of Z, L1-seed0, L1-seed1 and S equal the recorded `metrics.json` and
   `comparator_S.json` values within 1e-9 (cm, AUC), using their seeded surface samples, not `official_metrics.json`'s different ADD
   calculation.
3. **Admission of each new export**: the checks in `measure.score_export`: 390 frames, one world, one body, finite,
   `environment_valid`, frame clock, FPS, reference equal to `task.npz` within 1e-6 m, `eval_script_sha256` equal to the kit evaluator's
   hash. A malformed, nonfinite or incomplete export is **INVALID**: its failure details are kept and it cannot support GO or SIGNAL. No
   `T = 0` is manufactured for it.
4. **Checkpoint identity**: the hashes in the record equal the files at evaluation time; `voc_scale` and steps come from the
   checkpoint's own metadata. The actor loader reads only the policy file, so the older `state_*` file is not needed.
5. **Discrimination controls**, evaluated before either positive verdict is read: for every `j = 0..389`, the counterfactual `a_0 = r_0`,
   `a_t = r_j` for `t >= 1` (jump once and hold). If any such control meets GO's or SIGNAL's metric thresholds, the report says
   METRIC-NONDISCRIMINATING for that verdict and makes no claim that transport was learned.

## Gate (raw values against these thresholds, no rounding and no extra tolerance)

Let `E_S` be `comparator_S.json`'s full-precision `mean_position_cm`, `E` a seed's mean position error, and
`T_base = max(T_Z, T_L1-seed0, T_L1-seed1)`.

- **GO**: both admitted D1p seeds have `T >= 0.50`, `Q >= 0.50` and `E <= 0.5 E_S`. This position limit deliberately differs from D0's
  3 cm limit (`_metrics.py:41`); D0's rule is also reported beside it.
- **SIGNAL**, if not GO: at least one admitted D1p seed has `T >= max(0.20, T_base + 0.10)`, `Q >= 0.10` and `E <= E_S`.
- **NULL**: anything else. NULL means this screen was not met, not that no transport exists.

## What each result does

| Result | Consequence for W3 |
| --- | --- |
| GO | Timed translational following on this task only. M2 runs both arms as planned; M3 remains eligible, subject to M1 and M2 |
| SIGNAL | Learning moves the object but has not finished. M2 runs both arms on a schedule that completes unassisted |
| NULL | The hand reference earned no credit at about 22M transitions. M2 still tests the object-driven reference. D1p counts as M2's hand arm only if M2's object arm runs D1's exact schedule, seeds and transition count; otherwise M2 runs both arms, or says it cannot compare them |

M0 cannot kill W3 alone, because M2 tests a different reference. This is a deliberate reading of the plan's "each step can kill the
route", and it is stated here so the result cannot be read loosely either way. A pass says nothing about a challenge video: this is
the shipped task with a high-quality reference and its own mesh. These metrics see position only, so a tumbling object can pass them;
orientation error is reported beside them for that reason.

## Independent check

If the result is GO or SIGNAL, the lead recomputes `T`, `Q` and perpendicular error from the parquet with a separate script before
believing it (`PROGRAM.md` section 4). The outcome goes in `docs/chain/moving-m0-outcome.md`, whichever way it falls.

**Scope.** No challenge episode is read. Nothing here touches a pack manifest, `AGENTS.md` or a recorded `run.json`. New files only,
under `outputs/dynamic_feasibility/m0/`.
