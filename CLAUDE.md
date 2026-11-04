# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` first: it holds the dated user authorizations (what may be executed,
which episodes are excluded, where artifacts may live). This file covers how the code
is put together and how to run it. The Python code style to follow is
`docs/design/simplification-standard.md`.

## Environment and commands

Work only in this Linux checkout (`/home/ceolwolf/repos/video2data`). The interpreter is
the project venv `data/.runtime/linux-cuda/bin/python` (uv-managed Python 3.13, torch 2.7.1+cu128,
OpenCV, SciPy, matplotlib, onnx/onnxruntime). The system `python3` has no third-party packages.

```bash
bash scripts/linux/install-linux-cuda.sh        # (re)create the venv; idempotent, logs under data/reports/installation/
source scripts/linux/use-linux-cuda.sh          # venv on PATH + MoGe source trees on PYTHONPATH for a manual shell

bash scripts/reproduce/d1_meshes.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d --version v4  # plan only; URL is packed into the file
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.<slice> <operation> --request <file.json>
python3 -m py_compile <file>                     # syntax check (no deps needed)
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -c "import src.features.<slice>.<module>"   # CPU import check
data/.runtime/linux-cuda/bin/python -B scripts/documentation/compile.py          # concatenate docs into docs/compiled/all-documentation.md
```

There is no test runner installed (`src/tests/` predates a no-testing period and pytest is not in
the venv). Verify changes with `py_compile`, a CPU import, and an old-vs-new equivalence script
under `data/.runtime/tmp/` for pure functions. Never run two GPU operations at once:
`src/gpu.py` holds a cooperative lock at `data/.runtime/gpu-stage.lock` (stale locks from dead pids
are cleared by `clear_stale_lock`; a live owner is never touched). Keep all generated files, caches and
temp files inside the checkout (`data/outputs/`, `data/.cache/`, `data/.runtime/`); the dataset snapshot,
`upstream/` and `models/` are immutable inputs.

Collaboration and execution scope follow `docs/chain/PROGRAM.md` and the current task brief. Codex
(`codex` CLI) is called only through `scripts/chain/codex_ask.sh` (`PROGRAM.md` section 8). A Codex
config exists at `~/.codex/config.toml`; reply `/import` to scan it for importable items, then
`/import --yes=<digest>` to apply them.

## Architecture

**Vertical slices.** Retained features live under `src/features/` and own their
request, handler, integrations, persistence, runtime flags, drafts and README.
Their module dispatchers are thin entry points. Shared primitives in `src/paths.py`,
`src/provenance.py`, `src/runs.py` and `src/gpu.py` provide path constraints,
provenance, run ownership, admission and the GPU lease; `src/calibration/`,
`src/metrics/` and `src/data/` hold shared geometry, metrics and input handling.
Do not add horizontal service or repository layers.

**Shipping mesh line.** `src/features/mesh_extents/` measures cached video depth
and builds procedural meshes; `src/features/mesh_generative/` owns the pretrained
shape campaigns and video admission; `src/features/mesh_eval/` scores packed meshes
with the kit. `scripts/reproduce/d1_meshes.sh` rebuilds v2/v3/v4 from recorded measures;
`scripts/reproduce/d1_meshes_v5.sh` repacks the passing G2 production tree.
Mesh scoring and packing use `data/.runtime/kit-env/bin/python`; the project venv orchestrates builds.

**Reinforcement learning in simulation is required** (owner, 2026-11-04): every policy behind a submitted pose is
trained by RL in simulation and rolled out through the kit's evaluator. See `docs/chain/PROGRAM.md` section 0.

**Shipping pose line.** `src/features/pose_insurance/` owns scene preparation,
policy runs, frozen diagnostics and pose packing. `scripts/reproduce/d2_poses_v5.sh`
verifies or packs the frozen v5 sources and artifacts. `scripts/reproduce/d3_report.sh`
builds the report; `scripts/reproduce/verify.sh` checks explicit manifest hash entries.
See `scripts/reproduce/README.md` for commands and source-drift limitations.

**Supporting slices.** `src/features/depth/`, `src/features/stereo_depth/`,
`src/features/object_masks/` and `src/features/camera_motion/` retain perception
and preparation tools. `src/features/policy_training/` stays because the D0 loop
reads only its `budget.json`; compose, train, select and comparison require restoring
the recycled retargeting, scene_assembly and policy_rollout producers before admitting new inputs.
`src/features/mesh_fusion/` and
`src/features/dynamic_feasibility/` retain evidence cited in the abstract.

**Gates before work.** The owning slice checks its runtime flags, episode allowlist
and required interpreter before execution. Heavy execution redirects caches into
this checkout; GPU operations use the shared lease. Episode 41 and holdout episodes
43/44 remain excluded from current work. A documented command grants no execution
permission and does not override a source-drift refusal.

**Run ownership.** Operations write request snapshots, hashed inputs and sources,
provenance, status, timing and artifact inventories under `data/outputs/`. Consumers
admit completed producer records and re-hash the artifacts they use. Preserve
historical run records and ledgers; changed identities require fresh outputs.

**Frozen protocol.** Train episodes 0/11/12/18/21/23/26/31, tuning 39/42, holdout 43/44,
split `data/configs/splits/public-v1.json`, dataset revision pinned in `data/configs/sources.lock.json`.
`episode_partition` refuses any deviation. Depth scores compare to MoGe2 pseudo-labels, not
measured depth; say so in any write-up.

The finished depth study and unrun scaffolding are preserved under
`data/recycle/20261104T010336Z/`, with original paths and reasons in its
[manifest](data/recycle/20261104T010336Z/MANIFEST.md). Restore the required dependency group
to its original paths with Git moves as described there; archived sibling imports
still use their original package names.

Design references: `docs/design/README.md` (feature directory), `docs/design/artifact-contracts.md`,
`docs/design/feature-template.md`. The historical plan is `docs/design/legacy-plan.md`; the research backlog
is `docs/experiments/README.md`.
