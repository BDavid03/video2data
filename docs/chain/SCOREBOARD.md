# Scoreboard

Each lens is scored 0-5 by a fresh-context critic, never by the context that did the work (see
`PROGRAM.md` section 2). A score rises only with new evidence. "Unscored" means no critic has
looked yet; the first context scores every lens.

| Lens | Score | Evidence | Next improvement | Last scored |
| --- | --- | --- | --- | --- |
| Submission readiness | unscored | `SUBMISSION.md`: no deliverable done | | |
| Models | unscored | depth line frozen (iteration 17, tuning 0.0756, holdout 0.0827 vs MoGe2); no mask, mesh, pose or policy model run | | |
| Metrics | unscored | depth metrics and usability gates in `track3/features/auto_research/metrics.py`; no local Track 3 metric | | |
| Visualisations | unscored | `outputs/auto_research/dashboard.png`, `best-samples.png` | | |
| Presentation | unscored | draft `main.tex` from the depth line | | |
| Insights | unscored | ledger of 26 iterations in `outputs/auto_research/summary.md` | | |
| Engineering | unscored | reduction pass 2026-10-29; `track3/runs.py` shared primitives | | |
