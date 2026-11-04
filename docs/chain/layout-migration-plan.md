# Layout migration plan

**Owner, live, 2026-11-04 about 03:27 UTC:** "top level folders hsould be scripts, docs, .claude, src, data, submission, models everything else has to be organised under these". Route: Codex `route-layout-25` (read-only). `AGENTS.md` stays pinned by D2 v5's manifest and keeps its old path names; this file and `CLAUDE.md` are the translation.

## Target

| Now | Becomes | Stage |
| --- | --- | --- |
| `templates/`, `coursework/` | `docs/templates/`, `docs/coursework/` | L1 |
| `track3/` | `src/track3/`, import name unchanged, `PYTHONPATH=src` | L2 |
| `configs/`, `tests/`, `upstream/` | `src/configs/`, `src/tests/`, `src/upstream/` | L2 |
| `outputs/`, the dataset snapshot | `data/outputs/`, `data/datasets--nvidia--video_to_data_challenge/` | L2 |
| `runs/`, `manifests/`, `reports/`, `recycle/` | `data/runs/`, `data/manifests/`, `data/reports/`, `data/recycle/` | L2 |
| `submissions/` | `submission/` | L2 |
| `.agents/` | `.claude/agents-skills/` | L2 |
| `.runtime/`, `.cache/`, `.scratch/` | `data/runtime/`, `data/cache/`, `data/scratch/` | L3 |
| `scripts/`, `docs/`, `.claude/`, `models/` | stay | none |
| Top-level files `AGENTS.md`, `CLAUDE.md`, `README.md`, `LICENSE`, `.gitignore`, `.git` | stay: the owner named folders | none |

## Stages and when

| Stage | When | What |
| --- | --- | --- |
| **L1** | now | Two directories nothing imports |
| **L2** | Script built and tested in a scratch clone while M0b trains. Applied when the GPU is idle after M0b's scoring, about 06:00 UTC | One rerunnable `scripts/layout/migrate.py`, with a single path mapping in `track3/paths.py` so old records still resolve. No run record is rewritten |
| **L3** | After the last pack at 12:00 UTC and its upload | `.runtime`, `.cache`, `.scratch`. Its fallback is to roll back and keep the working environments |
| **Main checkout** | After the chain stops | The owner runs the command at the end of this file. While the driver runs it is left alone |

**Why L3 is last.** Six of the worktree's environments are links into the main checkout, and repairing shared runtime paths could damage the checkout the driver runs from. CPU imports cannot prove a moved CUDA extension still works. Losing the environments before the last pack would risk the entry, so the top level keeps these three hidden folders until then. This defers the owner's instruction for them; it does not exempt them.

## Rules

- **Nothing is deleted.** Tracked files move by `git mv`, untracked by plain rename. Historical `run.json`, ledgers, packed files and pinned JSON keep their bytes.
- **Quiescence.** L2 and L3 apply only with no GPU job, no Codex job and no detached run.
- **Live source-drift refusals stay.** Editing path literals changes source hashes, so v5's frozen-source checks would refuse. A separate, explicit historical verification mode checks frozen sources against their recorded commits and relocated artifacts against unchanged hashes. Current-source drift is reported, not hidden. A failure on a v5 release check blocks L2's commit; drift on older packs is only reported.
- **`check_release.sh` must end READY before and after**, with packs byte-identical and PDF `62967a99...` unchanged. A rise in the report's epoch from a touched file is prevented by freezing the original epoch.

## Done means

Top level holds only the seven folders and the named files; `migrate.py --verify` passes `py_compile`, per-environment CPU imports, old and new record-resolution equivalence, a real-data kit CPU evaluator comparison, an inventory of immutable files, both v5 verifiers, then `check_release.sh` READY.

## Main checkout, for the owner, after the chain stops

```bash
cd /home/ceolwolf/repos/video2data
git fetch origin
git merge --ff-only origin/main
python3 scripts/layout/migrate.py --dry-run
python3 scripts/layout/migrate.py --apply
python3 scripts/layout/migrate.py --verify
```

Stop on any failure. `git status` on the main checkout showed files staged under `.agents/` at the start of this context; they are the owner's, so they will need committing or moving first. Do not run this while the driver or any worker is running.

## State

- L1: done in the commit that added this file; `check_release.sh` READY before and after.
- L2: not started.
- L3: not started.
