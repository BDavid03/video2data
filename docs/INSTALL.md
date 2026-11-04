# FoundationPose isolated install

**CPU-ready install complete; download and build milestones met.**
Do not start registration until installation and CPU checks complete and a
separate GPU job obtains the lease.

This install is approved by `docs/chain/PROGRAM.md`, section 0, owner answer 4.
The tool-observed install start was **2026-11-04 04:13:05 UTC**; the binding
protocol starts at **04:15 UTC**, with the download milestone at **04:40 UTC**.
The follow-up hard limit for builds, CPU checks and frozen inputs is **05:25 UTC**.
No mocap, reference-pose or label data, model inference, GPU query, training
process or GPU lease is accessed. Read-only scan GLBs are hashed for the frozen inputs.
Every install/build/check command exports `CUDA_VISIBLE_DEVICES=''`.

## Layout and source provenance

Everything executable is under `.runtime/foundationpose/`; the existing project
environments are unchanged. Download caches are under `.cache/foundationpose/`.
Commands and evidence are under `outputs/foundationpose/install/`.
Source archives avoid mutating git commands:

- [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose):
  `a1b694b83e633c2cb6115b9063d940a687759392`, under `sources/`, with an `upstream` symlink.
- [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d):
  `e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31`.
- [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast):
  `253ac4fcea7de5f396371124af597e6cc957bfae`.

Source archive SHA-256 values and installed versions are in
`outputs/foundationpose/install/versions.json`.
Two FoundationPose source patches keep runtime behavior within the job boundary:
the hardcoded `/tmp` mesh export uses `TMPDIR`, and `Utils.py` defers `wp.init()`
while `CUDA_VISIBLE_DEVICES` is empty. GPU-visible behavior is unchanged.
`patches.txt` describes both and `versions.json` hashes the changed files.
No estimation parameters are changed. Upstream's `build_all_conda.sh` was read,
then bypassed because it uses destructive cleanup.

## Toolchain and commands

Micromamba comes from `https://micro.mamba.pm/api/micromamba/linux-64/latest`;
the archive is retained with its checksum in the evidence. All conda packages
come from conda-forge. Requested Python 3.11, GCC/G++ 13, CUDA 12.8; actual
resolved Python **3.11.17**, GCC/G++ **13.4.0**, nvcc **12.8.93**.

```bash
export CUDA_VISIBLE_DEVICES=''
export MAMBA_ROOT_PREFIX="$PWD/.runtime/foundationpose/mamba"
export TMPDIR="$PWD/.runtime/foundationpose/tmp"
.runtime/foundationpose/bin/micromamba create -y \
  -p "$PWD/.runtime/foundationpose/env" -c conda-forge --strict-channel-priority \
  python=3.11 pip gcc_linux-64=13 gxx_linux-64=13 'cmake<4' ninja eigen \
  boost-cpp=1.85 'pybind11<3' cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
  cuda-cccl=12.8 cuda-version=12.8
source .runtime/foundationpose/use-env.sh
```

`use-env.sh` sets compilers, CUDA headers/libraries, an explicit **sm_120**
architecture, two build workers, and caches/temp paths inside the install.
It keeps CUDA hidden by default. No build inspects hardware to select an architecture.
PyTorch is **2.7.1+cu128**, torchvision **0.22.1+cu128**, using official PyTorch
wheel URLs and PyPI for dependencies. The initial PyTorch-index-only attempt
failed on an NVIDIA-hosted NVTX download; its log is retained.

```bash
export CUDA_VISIBLE_DEVICES=''
export UV_CACHE_DIR="$PWD/.cache/foundationpose/uv"
.runtime/uv/uv pip install --python .runtime/foundationpose/env/bin/python \
  'torch @ https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl' \
  'torchvision @ https://download.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl' \
  --index-url https://pypi.org/simple
bash .runtime/foundationpose/install-dependencies.sh
cmake -S "$FP_ROOT/upstream/mycpp" -B "$FP_ROOT/upstream/mycpp/build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build "$FP_ROOT/upstream/mycpp/build" -j2
```

Only `mycpp`, PyTorch3D and nvdiffrast are required native additions.
BundleSDF `mycuda`, grid encoding and Kaolin are omitted: the model-based path
guards those optional imports. No model beyond the two FoundationPose inference
checkpoints is downloaded.

## Weights

The [upstream documented Google Drive folder](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i)
worked. The community Hugging Face mirror was **not used**. The downloader and
all four source URLs are retained in `.runtime/foundationpose/download_weights.py`
and `downloads/weights.json`. Neither the documented folder nor the upstream
README publishes checkpoint checksums; the following are locally computed
SHA-256 values, not independently authenticated publisher checksums.

| Relative to upstream/weights | Bytes | SHA-256 |
| --- | ---: | --- |
| 2023-10-28-18-33-37/model_best.pth | 68,220,109 | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| 2023-10-28-18-33-37/config.yml | 708 | `28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686` |
| 2024-01-11-20-02-45/model_best.pth | 190,229,389 | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |
| 2024-01-11-20-02-45/config.yml | 778 | `a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f` |

## Licence

FoundationPose carries NVIDIA's **noncommercial source code licence**
(the bundled `LICENSE` has no separate title; commonly called NVIDIA Source
Code License). Section 3.3 limits use of the work and derivatives to research
or evaluation. Redistribution must include the licence and retain notices;
derivatives must retain that use limitation. This is separate from the
project's Apache-2.0 licence. The full upstream licence remains beside the
source. Any result used in the entry needs this noncommercial pretrained-model
use disclosed in the abstract by the lead.

## CPU evidence and later frame

The requested adapter is `.runtime/foundationpose/register_one_frame.py`:
`register_frame(rgb, depth, mask, K, textured_glb, debug_dir)`. It loads GLB
geometry with `process=False`, retains UVs, converts PBR base colour to the
renderer material interface, and calls `FoundationPose.register(iteration=5)`.
It rejects upstream's fewer-than-four-point fallback and non-finite results.
Hypotheses are transformed back from centered mesh coordinates to the original
GLB frame. It has passed syntax compilation; registration has **not** run.

The follow-up prepared RGB uint8 / optical-axis metric depth / binary mask /
K NPZs from the unchanged Stage-A frame-0 grids, with holes invalid, plus
read-only links to the GLBs. After the current training run releases its lease,
and only in an authorized GPU job, the exact one-frame command is:

```bash
CUDA_VISIBLE_DEVICES=0 bash -c '
  source .runtime/foundationpose/use-env.sh
  exec python .runtime/foundationpose/register_one_frame.py \
    --frame outputs/foundationpose/inputs/train12-blue_cup-frame0.npz \
    --glb .runtime/foundationpose/inputs/blue_cup.glb \
    --output outputs/foundationpose/kill-test-train12-cup
'
```

This runtime adapter CLI owns the `track3.gpu.GpuLease`; it does not remove
an existing lock. The output contains input hashes, `predictions.npz` (all
original-frame hypotheses and scores), `overlay.png`, and `done.json`.
No label loading/scoring is included. A production slice handler is later work
if the pilot passes; this install does not edit `track3/`.

Still unproved without GPU: sm_120 kernel execution, Warp's CUDA JIT,
nvdiffrast rendering, actual texture rendering, predictor allocations,
finite/plausible registration after depth filtering, runtime and peak VRAM.
The first run could incur Warp compilation. The 10-minute pilot needs its
prepared inputs and a free GPU; those are prerequisites, not verified here.
Final CPU outcomes and disk use are recorded below.


## Clock correction and resumed install

The first brief used the tool's observed start **04:13:05 UTC**: its minute-25
cutoff was **04:38:05 UTC**, which prompted a stop. The binding committed
protocol starts at **04:15 UTC**, so minute 25 is **04:40 UTC**. The lead
established PyTorch completed at **04:38:49 UTC** (observed by this worker at
04:39:46). **The protocol's first milestone was met.** The earlier statement
that the pilot missed its milestone was incorrect under that binding clock.
The original CPU report at 04:37 preceded PyTorch completion and is historical.

The follow-up authorizes continuing dependencies, native builds and CPU checks,
plus label-free frozen input preparation, all with CUDA hidden. Its hard stop
is **05:25 UTC**. Registration remains prohibited in this job.


## Frozen pilot inputs (CPU preparation)

`outputs/foundationpose/inputs/manifest.json` records the four NPZs and separate
RGB PNG, metric depth NPY, K NPY, valid NPY and overlap-resolved mask PNGs.
Every output file and every array in each NPZ has a SHA-256. Source hashes bind
Stage-A run manifests, RGB, prepared stereo frames, preprocessing grids and
read-only scan GLBs. The scan links under `.runtime/foundationpose/inputs/`
point into this checkout's immutable dataset snapshot; the sweeper uses
`mini_sweeper_visual.glb`. No scan is decomposed or changed.

| Train episode | Objects | Grid H×W | Valid masked depth pixels |
| --- | --- | --- | --- |
| 12 | blue_cup / plastic_dish_rack | 576×768 | 3,863 / 16,462 |
| 31 | mini_sweeper / plastic_dish_rack | 576×768 | 695 / 17,072 |

Both grids retain K with fx=fy=435.42908653846155, cx=383.5, cy=287.5.
Checks bind frame index 0, pts 0, ego_cam_a, recorded_camera_a_remap, and the
panel's previously recorded remap-pixel equivalence. The preprocessing and
stereo-grid hash is
`2aa2fde1d2d7b0dffc861a8207b6f7ee38b368d4ecee694228e48a4f78e3ea54`.
The remap source_x/source_y shapes, saved stereo K, preprocessing K and
recorded panel K agree exactly. Masks match their artifact hashes, binary
values, pixel counts and valid-depth counts; the two masks in each episode do
not overlap. Invalid depth remains zero; every valid T1 depth value is
unchanged. RGB PNG hashes match the panel producer's recorded remap.

This preparation consumes cached label-free Stage-A artifacts, not raw videos
or pose/reference/label arrays. It verifies the recorded remap provenance; it
does not independently remap the raw video again. The adapter CLI verifies
the frozen train-12 cup NPZ and GLB hashes before running its one registration.

Executed with CUDA hidden:

```bash
export CUDA_VISIBLE_DEVICES=''
source .runtime/foundationpose/use-env.sh
python .runtime/foundationpose/prepare_inputs.py
```

To build the two remaining native extensions (executed in this follow-up):

```bash
export CUDA_VISIBLE_DEVICES=''
bash .runtime/foundationpose/build-extensions.sh
```

The script compiles PyTorch3D, then nvdiffrast, against the isolated torch and
nvcc with two workers and sm_120. It builds no BundleSDF/mycuda/Kaolin path.
`outputs/foundationpose/install/extensions.log` contains the compiler commands.


## Completed CPU admission

Completed **2026-11-04T04:58:19.265765+00:00**, before the follow-up's **05:25 UTC** hard limit.
Dependencies and all three native additions passed: `mycpp`, **PyTorch3D
0.7.9** and **nvdiffrast 0.4.0**. The first resumed CPU check exposed Open3D's
missing `libusb-1.0.so.0`; conda-forge **libusb 1.0.30**, **libcap 2.78** and
**libudev1 262.1** were installed into this environment. `ldd` then found no
missing Open3D shared libraries. Original failed checks are retained under
`cpu-check-initial.*` and `cpu-check-before-libusb.*`.

The final `cpu-check.json` and `cpu-check.log` report **44/44 imports passed**,
including `estimater`, its model/dataset modules, `Utils`, compiled `mycpp`
(with `cluster_poses`), `pytorch3d._C`, renderer/transforms/structures,
`nvdiffrast.torch` and `_nvdiffrast_c`. Torch CUDA initialization is **false**;
Warp runtime initialization is **false**, with Warp initialization intentionally
deferred for CUDA-hidden imports. This proves importability and binary linking;
it does not prove GPU execution.

Both Drive checkpoints were independently rehashed, then loaded with
`torch.load(map_location="cpu", weights_only=True)`. All model tensors are on
CPU and both `load_state_dict(strict=True)` calls report all keys matched.

| Checkpoint | Format / keys | Model tensors | Parameters |
| --- | --- | ---: | ---: |
| Refiner 2023-10-28-18-33-37 | Flat state dictionary; encodeA, encodeAB, pos_embed, rot_head, trans_head | 134 | 16,830,406 |
| Scorer 2024-01-11-20-02-45 | Outer keys: model, optimizer, global_step, epoch; model keys: encoderA, encoderAB, att, att_cross, linear, pos_embed | 116 | 15,773,121 |

Every exact outer and state-dictionary key is listed in
`outputs/foundationpose/install/checkpoint-keys.json`, alongside the independently
verified checkpoint hashes. The full SHA-256 values remain in the weights table
above. `versions.json` records all **88 conda packages**,
**126 Python distributions**, source pins, patches and
compiled-library SHA-256 values.

The persisted frozen-input check reloaded all four NPZs and compared their
file and array hashes, shape/dtype, invalid holes and valid mask-depth counts;
all **16 artifacts** passed rehashing (`frozen-input-check.json`). Registration
was **not run**. No mocap/reference pose/label arrays were opened.

Disk allocated including install caches and outputs: **13.364 GiB**.
This uses a single `du -sB1` invocation and deduplicates hard links across the
environment and caches; the earlier 17.541 GiB estimate counted shared hard
links across roots more than once. All files are retained.

The exact kill-test command above is ready for a separately authorized GPU job
once the existing training run releases its lease. Only a GPU can establish
CUDA driver and sm_120 execution compatibility, Warp JIT, actual textured
nvdiffrast rendering, predictor allocations, finite/plausible registration,
latency and VRAM. No further installation blocker is known from CPU admission.
