# Auto-research loop

Status: implemented and authorized on 2026-10-29. Runs in the Linux CUDA
environment inside this checkout (`.runtime/linux-cuda`); the Windows/Z: mount
is no longer used.

A circular pipeline for the scratch-depth research question: **run** a recipe,
**record** its metrics, **visualise** the ledger, **examine** it with an advisor
(Codex `gpt-6-astra`, high reasoning, read-only sandbox) and **improve** by
running the advisor's bounded proposals. Agents do not hand-edit requests
between experiments; the loop proposes, validates and executes them.

```text
scripts/research_loop.py            # minimal CLI; re-executes itself in the Linux env
track3/features/auto_research/
  recipes.py      bounded recipe space, validation, identity, baselines, heuristic mutation
  campaign.py     teacher/dataset registry (cached per data setting) and one-recipe execution
  metrics.py      AbsRel, RMSE-log, SILog, delta1-3, bias, support, gap, convergence, cost
  ledger.py       append-only ledger.jsonl, leaderboard and advisor summary
  visualize.py    dashboard.png (4 panels) and best-samples.png (qualitative)
  advisor.py      codex exec consultation with a JSON output schema and bounds validation
  loop.py         propose -> run -> record -> visualise -> examine; resume and stop rules
  finalize.py     freeze tuning selection (+ ensembles), then one holdout assessment
  report.py       fills templates/V2D_Challenge_Extended_Abstract_Template into main.tex
  runtime.json    execution flags and advisor settings
```

## Operation

```bash
python scripts/research_loop.py                      # loop.json budget: 24 iterations / 8 h / patience 8
python scripts/research_loop.py -n 6 --no-consult    # quick heuristic-only session
python scripts/research_loop.py --finalize --report  # freeze, assess holdout once, render LaTeX
```

Every iteration writes `outputs/auto_research/iterations/<run-id>/` with
`recipe.json`, `outcome.json` and, for consultations, `advisor-prompt.md`,
`advisor-output.json`, `advisor-stdout.log` and `consultation.json`. The
ledger is `outputs/auto_research/ledger.jsonl`; the rolling summary is
`outputs/auto_research/summary.md` with `dashboard.png` and `best-samples.png`.
Child producers (teachers, datasets, training, prediction, scoring, ensembles)
keep their own `run.json` under the owning features' output directories.

## Recipe space

A recipe is `{technique, parameters, training, data, seed, note}`. Techniques:
`ann`, `cnn`, `transformer` (research networks), `gradient_boosting`, and
`scratch` (the fixed coursework CNN with optimisation options). Architecture
bounds come from `learning_experiments/_config.py`; training bounds are epochs
1–100, batch 1–8, learning rate 1e-6–0.1, weight decay 0–1, BF16 flag (plus
channels-last, fused Adam, torch.compile and brightness jitter for scratch);
data bounds are 4–64 teacher samples per episode and prepared side 64–128.
Teachers and datasets are generated once per data setting and reused through
`outputs/auto_research/registry.json`.

## Metrics and selection

The objective metric is equal-episode log-depth MAE on tuning episodes 39/42
against MoGe2 pseudo-targets (the existing `score` operation). **MAE alone does
not select the model** (user direction, 2026-10-29: a low-MAE model that is
blurred, scale-biased, patchy or inconsistent is unusable). Each completed
iteration is first checked against usability gates, then ranked by a composite:

| Gate | Requirement | Why |
|---|---|---|
| support_fraction | ≥ 0.98 | predicts everywhere the teacher is valid |
| delta1 | ≥ 0.70 | most pixels within a 1.25× factor |
| abs mean log bias | ≤ 0.10 | metric scale is right, not just relative shape |
| edge_correlation | ≥ 0.40 | log-depth gradients agree with the teacher: real structure, not blur |
| episode spread | ≤ 0.05 | consistent across tuning episodes |

Composite (lower is better) = log-MAE + SILog + (1 − δ1) + |bias| + 0.5·(1 − edge).
The loop's "best so far", patience, advisor leaderboard, finalize selection and
report all use this rule; `best_by_mae_only` is reported alongside for contrast.
Companion training metrics (final/best loss, epochs to within 5 % of final,
generalisation gap, parameter count, peak GPU memory, wall time) explain and steer.

## Protocol guards

- Holdout 43/44 is read only by `finalize`, after `frozen-selection.json` is
  written from tuning evidence; its hash is re-checked afterwards.
- Episode 41 is excluded by every child operation.
- The advisor runs read-only and may only choose inside the validated space;
  invalid or repeated proposals are logged and replaced by heuristic mutations.
- Failed recipes are recorded outcomes with their error and failed child run.

## Limitations

Two tuning episodes and one seed give weak statistical support. Scores compare
to pseudo-labels, not measured depth. `pdflatex` is not installed; the report
operation writes `main.tex` ready to build.
