# Track 3 local pipeline

A local entry for the NVIDIA Video-to-Data Challenge, Track 3 (Egocentric),
using the existing RTX 5080. The target is the **full complete version 1**:
metric video reconstruction, robot references, trained policies, achieved
simulator trajectories, reconstructed meshes and accepted submission evidence.

Source implementations now span every major stage, including campaign planning
and submission packaging. A complete reconstruction, simulator rollout and
submission do not yet exist. On 18 October 2026 the user authorized the bounded
12-trial [research learning workflow](track3/features/learning_experiments/README.md),
which completed all 12 trials in the canonical Z: checkout. The selected
transformer scored 0.15815 tuning log-depth MAE against MoGe2 pseudo-targets;
the final holdout was not evaluated. This does not establish complete
reconstruction or submission readiness. See the [delivery ledger](docs/design/version1.md) and
[dated implementation history](reports/implementation-status.md).

The subsequent expanded research campaign completed with sparse sampling across
whole clips, isolated coursework-CNN training, native optimization comparisons
and ensembles. Its frozen eager/BF16 scratch-CNN ensemble scored **0.12396 tuning**
and **0.15216 holdout** log-depth MAE against MoGe2 pseudo-targets. Selection was
frozen before holdout labels were generated. The run recorded 35 completed
experiment outcomes and two failed primary-environment compiler attempts;
these are not 35 separately trained models. See the
[expanded results](track3/features/learning_experiments/README.md#expanded-investigation--completed).
The separate external CUDA attempt completed with real-batch loss/gradient
agreement in its recorded environment; it establishes no general speedup.
The episode-39 opening rerun completed rectification, stereo, scale anchor and
camera stages; clustering failed its border/quiet-opening guard, blocking hull
and TSDF reconstruction. A separate
[synthetic-depth pilot](track3/features/synthetic_depth/README.md) generated 128
training-only samples without mixing them into the measured campaign or claiming
a learning improvement. Research execution flags remain enabled under the user's
expanded instruction. This evidence does not complete version 1.

The canonical checkout retains the
[complete research report](outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/report.md),
[comparison figure](outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/comparison.png)
and [candidate CSV](outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/candidates.csv).
Generated-artifact links resolve in the Z: checkout; artifacts were not copied
into this implementation checkout.

## Continuous research loop (2026-10-29)

Work now runs in this Linux checkout with the `.runtime/linux-cuda` environment
(`scripts/linux/install-linux-cuda.sh`); the Z: mount is retired. The
[auto-research slice](track3/features/auto_research/README.md) closes the loop
between running a depth recipe, recording its metrics, visualising the ledger,
consulting Codex and running the next bounded proposal:

```bash
python scripts/research_loop.py            # loop (re-executes itself in the Linux env)
python scripts/research_loop.py --finalize --report
```

Live state: `outputs/auto_research/summary.md`, `dashboard.png`, `ledger.jsonl`.

## Documentation

To collect the project documentation into one Markdown file, use the
[documentation compiler](scripts/documentation/README.md):
`python scripts/documentation/compile.py`. Its default output is
`docs/compiled/all-documentation.md`. The script is provided unrun and untested.

Start with the [design documentation index](docs/design/README.md). The expanded
reference describes the current source as of 5 October 2026 and distinguishes
implemented behavior, historical evidence, design assumptions and remaining work.

| Topic | Reference |
|---|---|
| Architecture and full feature flow | [System overview](docs/design/system-overview.md) |
| Run ownership, provenance, grids, clocks and artifact schemas | [Artifact contracts](docs/design/artifact-contracts.md) |
| Configuration, operation order, failures and future release procedure | [Operating reference](docs/design/operations.md) |
| Resting fallback, tracked promotion and comparison rules | [Entry tiers](docs/design/entry-tiers.md) |
| Current choices and their consequences | [Decision register](docs/design/decisions.md) |
| Error propagation, thresholds and missing measurements | [Error budget](docs/design/error-budget.md) |
| Research papers, model architectures, mathematical derivations and Rust frameworks | [Research volumes](docs/research/README.md), [complete compendium](docs/research/egocentric-video-to-policy.md) |
| Delivery priorities and unresolved dependencies | [Roadmap](docs/design/roadmap.md), [risks](docs/design/risks-and-open-questions.md) |
| Adding or changing a vertical slice | [Feature authoring guide](docs/design/feature-template.md) |
| Pinned competition rules and source interpretation | [Contract report](reports/contract.md) |

The roadmap replaces the historical milestone schedule in
[TRACK3-PLAN.md](TRACK3-PLAN.md). Competition dates and rules in historical reports
are dated source readings, not fresh external verification.

## Implementation scope

Perception slices cover calibrated monocular/stereo depth, observed metric scale,
opening clusters and full-video masks, camera motion, object geometry/poses, and
anatomical hands. Scene assembly records completion, supports and approximate
contacts. Retargeting, policy composition/training, simulator rollout, public
diagnostics, campaign selection and submission each have their own slice.
See the [feature directory](docs/design/README.md#feature-directory) for every
operation and its detailed README.

The [scratch-depth feature](track3/features/scratch_depth/README.md) implements
an original CNN trained from random initialization, with optional explicitly
selected MoGe2 pseudo-depth supervision. Both referenced notebooks are present
under `coursework/` as of 2026-10-11. Teacher-based pairs have now been prepared
for the separate research slice. The original coursework CNN has now trained in
the expanded campaign, with independent optimization variants and calibrated
predictions. Notebook cells remain unexecuted.

The [learning buildout](docs/design/learning-buildout.md) adds optional PyTorch
training optimizations, a custom CUDA loss, and isolated experimental ANN, CNN,
transformer, ensemble and gradient-boosting components with bounded tuning plans.
The first selected research run completed with compilation and holdout excluded.
The expanded campaign completed those model/ensemble comparisons and frozen
holdout assessment. Both primary compiler attempts failed; those records remain
failures. A separate read-only external environment compiled and executed the
custom CUDA loss with one-batch agreement within recorded tolerance. Runtime
outcomes are recorded separately from broad delivery status.

Optional mask, WiLoR and MANO runtimes/assets, licensed hand files, policy and
submission environments, and the spoon body-frame decision remain outstanding.
A neutral parked-hand template is required even for the resting fallback. Current
request drafts with null producer paths are not executable plans.

The specifically approved MoGe2 installation completed on 25 September 2026:
a repository-local Python 3.13 environment, 69 locked packages, three pinned
source checkouts and `models/moge/model.pt`. See
[installation details](configs/perception/moge/README.md) and
[historical logs](reports/installation/moge-20260925/installation.json).
That approval does not authorize tests, inference, simulation or other packages.

New work uses vertical slices: each feature owns its request, handler,
integrations, persistence, configuration and documentation. Shared code contains
stable technical primitives. All generated state belongs under
`Z:/hugging-face/video2data`; upstream references and external environments remain
read-only.

## Current execution policy

The user has directed **absolutely no testing**. The selected bounded research
workflow permitted teacher inference, preparation, training and tuning scoring,
and completed. The expanded research and compiler experiments are now authorized.
Test suites, smoke runs, unrelated validation commands and
dependency-resolution experiments remain stopped. The commands
below document work performed before that direction; do not run them under the
current policy. The combined runner is disabled by `configs/project.json`.

## Historical check commands

Historical PowerShell invocations from this repository; retained as reference
only and prohibited under the current instruction:

```powershell
. ./scripts/use-local-artifacts.ps1
python -m unittest discover -s tests -v
python -m track3.data.audit --decode
python -m track3.calibration.audit --snapshot datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8 --output reports/calibration --image-check
```

The data audit requires the already installed `pyarrow` and FFmpeg (available here
through `imageio-ffmpeg`). The image calibration check uses installed OpenCV and
NumPy. No package installation is needed on the inspected machine. The library
helpers and tests use the Python standard library where possible.

When explicitly re-enabled by the user, `./scripts/run-foundation.ps1` runs these checks sequentially, stores logs in
`reports/foundation/`, and writes `manifests/runs/foundation.json`. It does not run
GPU inference or simulation. A successful exit covers only its stated checks.

The current `manifests/runs/foundation.json` is a coordination record linking the
historical evidence. A full source/configuration/artifact hash snapshot has not
been generated. The provenance command is implemented; the following is its
reference invocation, not a command to run under the current no-validation policy:

```powershell
python -m track3 provenance --output manifests/runs/foundation.json --label foundation --artifact-dir reports --artifact-dir manifests/dataset --artifact-dir manifests/models
```

When run, the provenance command hashes local source, configurations, tests, scripts, and the listed
artifacts; records installed package versions; and checks the upstream commit and
worktree. Dataset file hashes are recorded separately in `manifests/dataset/`.
These records describe the tested environment; they are not a full dependency lock
or an official evaluator result.

## Inputs and storage

- Dataset: `nvidia/video_to_data_challenge` at `5f68335f3acc802033d1e80728c1633197521de8`.
- Upstream: `release/0.2.0` at `43a3f5b4d479581f2e8bdf5e6ed7ca361d39ec5a`, checked out detached in `upstream/video_to_data/`.
- Submission Kit: downloaded 2026-09-30 to `upstream/v2d_submission_kit/`, archive hash in `configs/sources.lock.json`. Read, never executed.
- Policy runtime source: FlashCHORD at `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, a sparse checkout in `upstream/video_to_data-flashchord-aee95f2/` without its Git LFS assets. Read, never installed.
- All four are read-only inputs to our pipeline. No submodules or upstream install scripts were run.
- All generated files, models, caches, temporary files and new environments belong under `Z:/hugging-face/video2data`. `scripts/use-local-artifacts.ps1` refuses a checkout outside that root before creating directories or changing the environment, then routes common tool caches there for the current process.
- The existing external CUDA environment is reused read-only for authorized bounded experiments, with all outputs here. See [runtime evidence](reports/runtime/README.md) and the [real-batch CUDA record](outputs/scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json).
- `.gitignore` keeps large local data/runtime files out of future Git commits; it does not move them out of this directory.

## Evidence and conventions

- [Data audit](reports/data-audit/README.md): explicit episode discovery, byte/hash verification, full CPU video decoding, public pose validity and assets.
- [Calibration](reports/calibration/README.md): camera transform directions, orientation, timing and provisional stereo geometry.
- [Metric protocol](reports/metric-protocol.md) and [frozen split](configs/splits/public-v1.json): development/validation separation and diagnostic metrics.
- [Official contract](reports/contract.md): what is submitted, how it is scored, and what remains open. Source readings are in `reports/contract-evidence/2026-09-30/`. The first audit, from before publication, is kept as [contract-2026-09-24.md](reports/contract-2026-09-24.md).
- [Model readiness](reports/model-readiness/README.md): candidate models, licenses, version pins and pending prerequisites.

Public labels are diagnostic references only. Evaluation episode task identities
and labels are not inferred. Preserve MP4 presentation timestamps, public nominal
timestamps and capture times as distinct fields. Never turn invalid zero poses into
training targets, and never call a synthetic network smoke a robot-policy result.

New production dependencies require confirmation under the workspace agreement.
