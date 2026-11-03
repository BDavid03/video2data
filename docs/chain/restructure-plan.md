# File-system restructure plan (owner to-do 5; context 19, 2026-11-03)

Inventory: `outputs/chain/codex/20261103T140737Z-inventory-19.answer.md` (Codex, read-only). At `c287d5e` git tracks
766 entries (765 files and one accidental gitlink) in 26 slices; nine pack manifests pin 528 paths, 73 of them tracked
files; 491 untracked, non-ignored files are mostly TeX caches and build logs. No whole slice is unused: every older
slice still has an importer, a request, or a document that cites it.

## Rules (owner)

- Nothing is deleted. Everything to be removed moves to `recycle/<UTC stamp>/` under its original path, listed in
  `recycle/<UTC stamp>/MANIFEST.md` with the reason. Tracked files move by `git mv`. Untracked clutter moves by plain
  `mv` into `recycle/<UTC stamp>/untracked/`, which git ignores.
- Files pinned by a pack manifest move only after the 12:00 UTC 4 Nov pack cut-off, or as part of a new pack. Packed
  commits keep reproducing at their URLs either way. Historical `outputs/` run records are never rewritten.
  `.runtime/tmp/reproduce-*` directories are pinned and stay where they are.
- The chain scripts (`scripts/chain/`) and the hand-off directory stay where they are: the driver finds them by path.

## Phases

| Phase | What | When |
| --- | --- | --- |
| 1 | Recycle six unreferenced `.agents/skills/ai-ml-skills/*.md`, the two wrappers `scripts/wsl/shell-{policy,retarget}.sh` that point at the retired mount, and `scripts/track3/d1_figure.py` (superseded by `make_figures.py`). Untrack the `.claude/worktrees` gitlink (index only) and ignore worktrees, `.scratch/`, report caches and build logs | now |
| 2 | `experiments/` -> `docs/experiments/`; `TRACK3-PLAN.md` -> `docs/design/legacy-plan.md`; fix the relative links | now |
| 3 | Implementation scripts beside their owners: `scripts/track3/local_cdo.py` -> `mesh_eval/`, `d1_measure.py` -> `mesh_extents/measure_loop.py`, `depth_opening_frames.sh` -> `depth/`, `static_pose_baseline.py` -> `pose_insurance/`; update imports, source inventories, READMEs | after the cut-off |
| 4 | Request drafts beside their slices (`scripts/track3/requests/**` -> `track3/features/<slice>/requests/`); update `d1_meshes.sh` | after the cut-off |
| 5 | Historical depth-study slices that no deliverable uses (`auto_research`, `depth_student`, `learning_experiments`, `scratch_depth`, `opening_experiments`, `synthetic_depth`) and unrun scaffolding: decide per slice whether it goes to `recycle/` or stays as cited evidence, from a fresh importer check. Owner sign-off on the list first | after the cut-off |

Every phase ends with `py_compile` and a CPU import of every slice, and with `scripts/reproduce/verify.sh` re-checking the pins.
