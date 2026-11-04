# FoundationPose install

A separate install for a one-off pose pilot (`src/features/foundationpose_pilot.py`). It is **not**
part of the entry. The main environments are `scripts/linux/install-linux-cuda.sh`,
`install-kit-env.sh` and `install-flashchord.sh` (see the [README](../README.md)).

- Programs: `data/.runtime/foundationpose/`
- Caches: `data/.cache/foundationpose/`
- Install logs and checks: `data/outputs/foundationpose/install/` (versions and archive hashes in
  `versions.json`)

Older records inside the install may use the old `.runtime/` and `outputs/` paths.

## Sources

| Project | Commit |
| --- | --- |
| [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | `a1b694b83e633c2cb6115b9063d940a687759392` |
| [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | `e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31` |
| [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) | `253ac4fcea7de5f396371124af597e6cc957bfae` |

Two patches to FoundationPose (`data/.runtime/foundationpose/patches.txt`): the hard-coded `/tmp`
mesh export uses `TMPDIR`, and `Utils.py` delays `wp.init()` while CUDA is hidden. Estimation
settings are unchanged. We skipped upstream's `build_all_conda.sh` because it deletes files.

## Build

Toolchain from micromamba and conda-forge: Python 3.11.17, GCC/G++ 13.4.0, nvcc 12.8.93. PyTorch
2.7.1+cu128, torchvision 0.22.1+cu128. `use-env.sh` sets the sm_120 GPU architecture and hides CUDA
by default.

```bash
export CUDA_VISIBLE_DEVICES=''
export MAMBA_ROOT_PREFIX="$PWD/data/.runtime/foundationpose/mamba"
export TMPDIR="$PWD/data/.runtime/foundationpose/tmp"
data/.runtime/foundationpose/bin/micromamba create -y \
  -p "$PWD/data/.runtime/foundationpose/env" -c conda-forge --strict-channel-priority \
  python=3.11 pip gcc_linux-64=13 gxx_linux-64=13 'cmake<4' ninja eigen \
  boost-cpp=1.85 'pybind11<3' cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
  cuda-cccl=12.8 cuda-version=12.8
source data/.runtime/foundationpose/use-env.sh
```

```bash
export CUDA_VISIBLE_DEVICES=''
export UV_CACHE_DIR="$PWD/data/.cache/foundationpose/uv"
data/.runtime/uv/uv pip install --python data/.runtime/foundationpose/env/bin/python \
  'torch @ https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl' \
  'torchvision @ https://download.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl' \
  --index-url https://pypi.org/simple
bash data/.runtime/foundationpose/install-dependencies.sh
cmake -S "$FP_ROOT/upstream/mycpp" -B "$FP_ROOT/upstream/mycpp/build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build "$FP_ROOT/upstream/mycpp/build" -j2
```

Then the other native extensions and the pilot inputs:

```bash
export CUDA_VISIBLE_DEVICES=''
bash data/.runtime/foundationpose/build-extensions.sh
source data/.runtime/foundationpose/use-env.sh
python data/.runtime/foundationpose/prepare_inputs.py
```

Native additions: `mycpp`, PyTorch3D 0.7.9 and nvdiffrast 0.4.0 only. BundleSDF `mycuda`, grid
encoding and Kaolin are left out; the model-based path guards those imports. Open3D also needed
conda-forge libusb 1.0.30, libcap 2.78 and libudev1 262.1.

## Weights

From the [upstream Google Drive folder](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i).
Upstream publishes no checksums. We computed these SHA-256 values locally; they do not prove the
files are the publisher's.

| Under upstream/weights | Bytes | SHA-256 |
| --- | ---: | --- |
| 2023-10-28-18-33-37/model_best.pth | 68,220,109 | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| 2023-10-28-18-33-37/config.yml | 708 | `28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686` |
| 2024-01-11-20-02-45/model_best.pth | 190,229,389 | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |
| 2024-01-11-20-02-45/config.yml | 778 | `a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f` |

## Licence

FoundationPose uses NVIDIA's noncommercial source code licence. Section 3.3 limits it and its
derivatives to research or evaluation. This is separate from our Apache-2.0 licence. Any result from
it used in the entry would have to be disclosed in the report.

## Run one frame

Needs the GPU and the shared lease. The adapter takes `GpuLease` and never removes someone else's lock.

```bash
CUDA_VISIBLE_DEVICES=0 bash -c '
  source data/.runtime/foundationpose/use-env.sh
  exec python data/.runtime/foundationpose/register_one_frame.py \
    --frame data/outputs/foundationpose/inputs/train12-blue_cup-frame0.npz \
    --glb data/.runtime/foundationpose/inputs/blue_cup.glb \
    --output data/outputs/foundationpose/kill-test-train12-cup
'
```

It writes input hashes, `predictions.npz`, `overlay.png` and `done.json`. No labels are loaded or
scored. Inputs are listed in `data/outputs/foundationpose/inputs/manifest.json`; they come from
cached label-free outputs. Scan GLBs are linked read-only.

## Results

- CPU check: 44 of 44 imports passed; both checkpoints loaded with all keys matched. This shows
  linking and loading, not GPU runs. Install size: 13.364 GiB.
- The first two cup attempts crashed in upstream `compute_crop_window_tf_batch` (float32 K met
  float64 crop points). A protocol amendment allowed one more try with K as float64 and nothing else
  changed. It produced all four predictions and overlays.
- Two-pair check passed: train 12 at 1.8102 cm and 2.3550 degrees; train 31 at 1.2313 cm and
  8.8247 degrees. These are train episodes, not independent validation. Tracking was not benchmarked.
- Four GPU headroom stops needed fresh GPU admission; finished outputs were kept and no successful
  step was repeated. The pass rule was fixed before reading motion capture.

Details: `data/outputs/foundationpose/pilot/summary.md` and
[the pilot outcome](chain/foundationpose-pilot-outcome.md).
