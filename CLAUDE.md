# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` first: it holds the dated user authorizations (what may be executed,
which episodes are excluded, where artifacts may live). This file covers how the code
is put together and how to run it. The Python code style to follow is
`docs/design/simplification-standard.md`.

## Environment and commands

Work only in this Linux checkout (`/home/ceolwolf/repos/video2data`). The interpreter is
the project venv `.runtime/linux-cuda/bin/python` (uv-managed Python 3.13, torch 2.7.1+cu128,
OpenCV, SciPy, matplotlib, onnx/onnxruntime). The system `python3` has no third-party packages.

```bash
bash scripts/linux/install-linux-cuda.sh        # (re)create the venv; idempotent, logs under reports/installation/
source scripts/linux/use-linux-cuda.sh          # venv on PATH + MoGe source trees on PYTHONPATH for a manual shell

python scripts/research_loop.py                  # closed research loop (re-execs itself in the venv)
python scripts/research_loop.py -n 6 --no-consult
python scripts/research_loop.py --finalize --report   # freeze on tuning, assess holdout once, export ONNX, write LaTeX

PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.<slice> <operation> --request <file.json>
python3 -m py_compile <file>                     # syntax check (no deps needed)
PYTHONPATH=. .runtime/linux-cuda/bin/python -c "import track3.features.<slice>.<module>"   # CPU import check
python scripts/documentation/compile.py          # concatenate docs into docs/compiled/all-documentation.md
```

There is no test runner installed (`tests/` predates a no-testing period and pytest is not in
the venv). Verify changes with `py_compile`, a CPU import, and an old-vs-new equivalence script
under `.runtime/tmp/` for pure functions. Never run two GPU operations at once: `track3/gpu.py`
holds a cooperative lock at `.runtime/gpu-stage.lock` (stale locks from dead pids are cleared
by `clear_stale_lock`; a live owner is never touched). Keep all generated files, caches and
temp files inside the checkout (`outputs/`, `.cache/`, `.runtime/`); the dataset snapshot,
`upstream/` and `models/` are immutable inputs.

Codex (`codex` CLI, model `gpt-6-astra`) is used as a read-only advisor by the research loop.
A Codex config exists at `~/.codex/config.toml`; reply `/import` to scan it for importable
items, then `/import --yes=<digest>` to apply them.

## Architecture

**Vertical slices.** Every feature lives in `track3/features/<slice>/` and owns its request
dataclass, handler, integrations, persistence, `runtime.json` flags, JSON request drafts and
README. `__main__.py` is a thin `operation --request` dispatcher. Shared code is limited to
stable primitives in `track3/`: `paths.py` (permitted artifact roots from
`configs/project.json`, `project_path`, `output_path`, per-platform `interpreter`),
`provenance.py` (`run_fields`, `input_file`, `read_run`, `sha256_file`, `write_json`),
`gpu.py` (lease and telemetry), `runs.py` (run-record lifecycle `recorded`, `runtime_flag` +
`authorize_episode` gates, `producer`/`artifact`/`checked_entry` admission, `heavy_execution`
cache redirection, `source_hashes`, `save_npz`), `calibration/` (NumPy geometry), `metrics/`, `data/`.
Do not add horizontal service or repository layers.

**Gates before work.** Each handler calls `authorize(<operation>)` against its slice's
`runtime.json`, then `heavy_execution`, which checks the interpreter and redirects every
cache/temp env var into the checkout before importing torch/cv2. A false flag or wrong
interpreter raises before anything is written. Episode 41 is refused everywhere; holdout
episodes 43/44 are refused except by `holdout_predict`-gated paths used only by finalize.

**Run ownership.** An operation creates `outputs/<slice>/<operation>/<run-id>/run.json`
holding the request snapshot, hashed inputs, hashed sources, provenance tag, status
(`running`/`completed`/`failed`), timing and an artifact hash inventory. Consumers admit a
producer by reading its `run.json` (`read_run`) and re-hashing the exact artifacts they use;
only `completed` records may be consumed. Never mutate historical records.

**Depth research chain** (the part the loop drives):

1. `depth.generate` runs MoGe2 (`models/moge/model.pt`, pinned sources in `upstream/`) on
   sparse frames of one episode and writes teacher depth NPZs.
2. `scratch_depth.prepare` turns teacher runs into calibrated RGB/pseudo-depth pairs at a
   chosen side (64–512 px) with a `dataset.json`; `learning_experiments/data.py:PreparedPairs`
   is the shared loader and validates the whole contract.
3. Training/prediction families: `learning_experiments` (ann/cnn/transformer/gradient
   boosting, 128 px cap), `scratch_depth` (the coursework CNN, predicts from video, adapted
   by `scratch_adapter`), `depth_student` (fully convolutional UNet/dilated/coursework,
   SILog + gradient losses, EMA, augmentation, up to 512 px, ONNX export).
4. `learning_experiments.score` is the single scorer (equal-episode log-depth MAE) and
   `ensemble_operations` the single combiner; both accept `depth_student.predict` runs.
5. `auto_research` closes the loop: `recipes.py` (bounded space, identity hash),
   `campaign.py` (teacher/dataset registry, one recipe end to end with 1–3 seeds),
   `metrics.py` (AbsRel, SILog, delta, bias, edge fidelity; usability gates + composite),
   `ledger.py` (`outputs/auto_research/ledger.jsonl`), `visualize.py`, `advisor.py` (Codex
   with a strict JSON schema), `loop.py`, `finalize.py` (freeze on tuning, then holdout and
   ONNX), `report.py` (fills `templates/V2D_Challenge_Extended_Abstract_Template`).
   Selection is by usability composite among gate-passing recipes, never MAE alone.

**Frozen protocol.** Train episodes 0/11/12/18/21/23/26/31, tuning 39/42, holdout 43/44,
split `configs/splits/public-v1.json`, dataset revision pinned in `configs/sources.lock.json`.
`episode_partition` refuses any deviation. Scores compare to MoGe2 pseudo-labels, not
measured depth; say so in any write-up.

Other slices (object masks/mesh/pose, hands, scene assembly, retargeting, policy, campaign,
submission, diagnostics) follow the same request/gate/run.json pattern; most are implemented
but unrun. Design references: `docs/design/README.md` (feature directory),
`artifact-contracts.md`, `feature-template.md`.
