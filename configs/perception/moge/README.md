# MoGe2 installation

Status: **installed on 2026-09-25; bounded teacher inference completed on 2026-10-18**.
The separately authorized research workflow generated 200 teacher frames across
ten training/tuning episodes using the existing installation. Episode 41 and
final holdout 43/44 were excluded from that first run. The subsequent expanded
research campaign enables full-clip sparse teachers and generates holdout targets
only after tuning selection is frozen. See the
[research results](../../../track3/features/learning_experiments/README.md#recorded-result).

The user approved the original bounded installation after instructing “absolutely 0 testing.” All 69 wheel packages, three pinned source checkouts and the single model download completed. No tests, import checks, resolver experiments, model smoke runs or validation commands were run during installation. Logs and command outcomes are in `reports/installation/moge-20260925/installation.json`.

The no-install Python3.14 attempt could not satisfy the upstream-declared `moderngl` dependency from Windows wheels. Python3.13.14 binary-only resolution succeeded. Its complete wheel dependency list is `pylock.python313.toml`:69 packages including a fresh torch2.7.1+cu128/torchvision0.22.1+cu128 pair, numpy2.2.6, moderngl5.12.0 and glcontext3.0.0. It preserves upstream's declared MoGe2 dependencies, including Gradio and visualization packages; a smaller custom inference-only dependency set has not been established.

This candidate uses a new environment at `.runtime/moge-python313` based on the already installed Python3.13.14 interpreter. It does not modify or inherit the external Python3.14 CUDA environment. All new dependency/cache/source/model artifacts remain inside this repository, using `scripts/use-local-artifacts.ps1` before every future operation. Do not run `uv pip sync` against a shared or global interpreter: it synchronizes exactly the selected environment.

The approved installation was executed by `scripts/install-moge.ps1`, using:

```powershell
. ./scripts/use-local-artifacts.ps1
uv venv .runtime/moge-python313 --python C:/Users/db009/AppData/Roaming/uv/python/cpython-3.13.14-windows-x86_64-none/python.exe --no-python-downloads
uv pip sync configs/perception/moge/pylock.python313.toml --python .runtime/moge-python313/Scripts/python.exe --only-binary :all: --no-python-downloads
hf download Ruicheng/moge-2-vitl-normal model.pt --revision cb0e8bbd6b1e243589717c78e750b1ba4c093acf --local-dir models/moge
```

The installer reported success for every command. The lock contains wheel URLs and SHA256 values used by the package installer. The model's advertised SHA256 is `280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01`, size 1,323,815,904 bytes. No separate checksum-validation command or inference command was run. All 69 dependency packages are installed in the new environment; the existing external CUDA environment was not modified or reused for this installation.

Three source trees are separately pinned and **not part of the wheel lock**:

| Source | Commit | Repository-local destination |
|---|---|---|
| `microsoft/MoGe` | `07444410f1e33f402353b99d6ccd26bd31e469e8` | `upstream/moge` |
| `EasternJournalist/utils3d` | `3fab839f0be9931dac7c8488eb0e1600c236e183` | `upstream/moge-utils3d` |
| `EasternJournalist/pipeline` | `866f059d2a05cde05e4a52211ec5051fd5f276d6` | `upstream/moge-pipeline` |

Those exact source trees were checked out without running build/install hooks. `scripts/use-moge.ps1` selects the new environment and sets an explicit PYTHONPATH containing the three source directories. Dot-sourcing it changes the current shell environment only; it does not run Python or a model. No editable pip installation or source build was performed. Their static pyproject metadata, hashes and declared requirements are in `manifests/models/moge-source-pins.json` and adjacent `*.pyproject.toml` files. Runtime correctness has not been checked. The external-reuse constraints/files document the unsuccessful Python3.14 attempt; they were not inputs to this Python3.13 installation.

The lock lists178,821,235 bytes for non-torch wheel candidates. This includes two alternative fonttools wheels and excludes torch/torchvision sizes, which were absent from package metadata; an already-started HTTP HEAD attempt returned403. Therefore the exact combined download/storage requirement remains unknown. The1.324GB model size must not be presented as total disk usage. All size/hash values are metadata observations, not proof of downloaded files.

The installer reported approximately 3.0 GiB for the torch wheel and 7.3 MiB for torchvision. Combined installed disk usage has not been measured. This approval did not include the full GroundingDINO/SAM2/FoundationPose/hand/DROID stack, MANO acquisition, source builds, tests, simulation or model execution. Dependency availability alone does not establish an RTX5080 reconstruction. Read `reports/model-readiness/README.md` for the complete route and blockers.
