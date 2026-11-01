# Scoreboard

Each lens is scored 0-5 by a fresh-context critic, never by the context that did the work (see
`PROGRAM.md` section 2). A score rises only with new evidence. Two critics score each pass, a
Claude sub-agent and Codex; where they differ the lower score is kept.

| Lens | Score | Evidence | Next improvement | Last scored |
| --- | --- | --- | --- | --- |
| Submission readiness | 1 (Claude 1, Codex 1) | `SUBMISSION.md`: no deliverable done; D1 first stage ran (ep 38 depth); D2 runtime installed and smoke-tested; no packed file, no PDF | pack a valid D1: 36 video-measured extents primitives through `pack_reconstruction.py`, hash recorded | 01, 2026-11-01 |
| Models | 1 (Claude 1, Codex 2) | depth line frozen (iteration 17, tuning 0.0756, holdout 0.0827 vs MoGe2 pseudo-labels); no mesh, pose or policy model run | video-measured extents for 3+ scanned objects on public episodes, scored against cube 2.52 and oracle 1.17 cm | 01 |
| Metrics | 2 (Claude 2, Codex 2) | `scripts/track3/local_cdo.py` runs the kit's own CD-O and `budget_mesh`; `scripts/track3/static_pose_baseline.py` runs the kit's pose functions (rerun after a layout bug); depth gates | report pair-weighted CD-O (36-pair weights) beside the per-object mean; a real sampling floor with different seeds per side | 01 |
| Visualisations | 1 (Claude 1, Codex 2) | depth-only figures (`outputs/auto_research/dashboard.png`, `best-samples.png`) | per-object CD-O bars (cube, oracle, video-measured) with primitive-vs-scan overlays | 01 |
| Presentation | 1 (Claude 1, Codex 2) | depth-only draft `main.tex`, unbuilt, describes MAE-only selection, count mismatch, no anonymous option | install Tectonic, build an anonymous two-page Track 3 skeleton PDF whose numbers match the records | 01 |
| Insights | 2 (Claude 2, Codex 3) | oracle-extents analysis (proportions close >50 % of the primitive-to-scan gap); static-objects pose floor; score-weight priority (72 % in four objects) | per-object explanation of where extents do not help (white_pot: box 1.75 vs cylinder 1.49; rack stays ~2 cm) | 01 |
| Engineering | 2 (Claude 2, Codex 2) | idempotent install scripts with per-step logs, checksummed git-lfs; `paths.py` equivalence-checked; recorded smoke and request files | make a CD-O run self-describing (candidate files kept, hashes, versions, command) | 01 |
