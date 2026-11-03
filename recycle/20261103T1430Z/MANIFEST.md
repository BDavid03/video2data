# Recycle 20261103T1430Z (restructure phase 1, `docs/chain/restructure-plan.md`)

Moved here instead of deleted (owner, 2026-11-03: "can move everything thats gonna be deleted into a recycle folder").
Each file keeps its original path below this folder; restore with `git mv recycle/20261103T1430Z/<path> <path>`.
Evidence of non-use: `outputs/chain/codex/20261103T140737Z-inventory-19.answer.md` and the lead's `git grep` re-check.

| Original path | Why |
| --- | --- |
| `.agents/skills/ai-ml-skills/Data Preprocessing.md` | copied generic guidance; not among the four the advisor names; nothing references it |
| `.agents/skills/ai-ml-skills/ML Deployment.md` | same |
| `.agents/skills/ai-ml-skills/Nlp Pipeline.md` | same |
| `.agents/skills/ai-ml-skills/Prompt Engineering.md` | same |
| `.agents/skills/ai-ml-skills/Reinforcement Learning.md` | same |
| `.agents/skills/ai-ml-skills/Time Series Forecasting.md` | same |
| `scripts/wsl/shell-policy.sh` | one-line wrapper for the retired `/mnt/z` mount; nothing references it |
| `scripts/wsl/shell-retarget.sh` | same |
| `scripts/track3/d1_figure.py` | superseded by `docs/report/track3/figures/make_figures.py`; nothing references it |

Also in this phase (not moved): the accidental gitlink `.claude/worktrees/chain-ctx01` was removed from the index only
(`git rm --cached`); the live worktree is untouched, and `.gitignore` now ignores `.claude/worktrees/`.
