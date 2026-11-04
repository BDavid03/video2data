# Local runtime feasibility, 2026-09-24

The RTX 5080 executes CUDA inference, an optimizer update and exact checkpoint reload in an existing Windows environment. **No simulator is installed in the inspected locations; no simulator stepping, policy update, or rendered recording has been completed.** Official Track 3 simulator/task requirements are unresolved. A generated MLP is only a runtime diagnostic.

## Evidence and reuse

- `gpu-smoke.json`: torch 2.12.1+cu132, CUDA 13.2, device capability 12.0, compiled `sm_120`. Generated 16→32→4 MLP, batch 32, one Adam update, model and optimizer checkpoint restored, predictions identical. Elapsed 4.18 seconds. Sampled device usage peaked at 2,903 MiB from an initial 2,421 MiB; torch peak allocated 67,155,456 bytes, reserved 69,206,016 bytes. Sampling interval is 100 ms plus `nvidia-smi` latency, so these are sampled rather than guaranteed absolute peaks. Windows display and other processes are included.
- `cpu-smoke.json`: same checks passed with system torch 2.12.1+cpu. The CPU and GPU random tensors differ, so losses are not accuracy comparisons.
- `tiny-mlp.pt` and `tiny-mlp-cpu.pt`: generated synthetic checkpoints, hashes recorded in reports. Neither contains a trained robot policy.
- `inventory.json`: read-only machine, Python, Docker and storage observations. Existing CUDA interpreter is `Z:/hugging-face/locals/.venv/Scripts/python.exe`. Its `include-system-site-packages=true` makes it useful for probing, but insufficiently isolated for a final reproducible pipeline. It was not modified.

Default Windows Python is 3.14.5 and CPU-only torch. Existing uv Python installs include 3.12/3.13, but no 3.11. Ubuntu 26.04.1 has Python 3.14.4 without torch/numpy/pip and no gcc/cmake/nvcc/uv on PATH. Two discovered WSL virtual environments have Windows paths in `pyvenv.cfg`; they are not reusable Linux environments. No Isaac packages, known Isaac installation directory, or Isaac Docker image were found within the documented inspection scope.

The machine has 66,156,478,464 bytes RAM. C: had 62.4 GiB free and Z: 1,519.0 GiB. WSL's 914 GiB apparent free filesystem capacity is not backing-store capacity: its VHDX resides on C:. Existing Docker data is `/var/lib/docker` in that VHDX. Docker works for root reads; the ordinary user cannot access the socket. Existing images belong to other projects. No images were pulled, containers started, packages installed, user groups edited, or shared disks moved.

## Reproduce the tested diagnostic

Run from the project root. Obtain the coordinator's exclusive GPU slot before CUDA execution.

```powershell
python scripts/runtime/torch_smoke.py --device cpu --output reports/runtime/cpu-smoke.json --checkpoint reports/runtime/tiny-mlp-cpu.pt
& Z:/hugging-face/locals/.venv/Scripts/python.exe scripts/runtime/torch_smoke.py --device cuda --output reports/runtime/gpu-smoke.json --checkpoint reports/runtime/tiny-mlp.pt
```

The script downloads nothing. A CUDA run requires at least 3 GiB driver-reported free VRAM at admission. This reserves an estimated 1 GiB diagnostic allowance and 2 GiB headroom; device-wide samples remain necessary because Windows memory accounting differs from torch/driver allocation accounting.

## Concrete next runtime, pending contract and installation confirmation

All new environments, downloaded files, caches, intermediate files, checkpoints, logs and reports must reside beneath `Z:/hugging-face/video2data`. `configs/runtime/local.json` defines the layout and a provisional 300 GiB total reserve: 120 GiB runtime/download caches, 80 GiB selected models, 100 GiB outputs/checkpoints. These are planning allowances, not measured package sizes.

The smallest next action needing **no new dependencies** is read-only reuse of the existing native CUDA environment for compatible perception components, with every artifact redirected here. Linux-only custom extensions still need an isolated build/runtime; CUDA MLP success says nothing about those extensions.

If the official Track 3 contract permits it, the documented **candidate** simulation stack is Isaac Sim 5.1.0 + Isaac Lab 2.3.0 + Python 3.11 + torch 2.7.0/torchvision 0.22.0 from cu128. The Lab guide documents this combination, including `isaacsim[all,extscache]==5.1.0`, and a selected RL backend can avoid installing all training frameworks. This is a feasibility candidate, not the verified competition version. [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)

Isaac Sim 5.1 lists Ubuntu 22.04/24.04 or Windows 10/11, 16 GB minimum VRAM and 50 GB minimum SSD storage. Ubuntu 26.04 is outside that listed OS matrix. Hardware meeting minimums does not prove a rendered Track 3 task will fit. Native Windows is therefore a concrete candidate if the official task supports it. [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

Native proposal: create `.runtime/python` and `.runtime/envs/isaac51`; use project-local `UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `TORCH_EXTENSIONS_DIR`, `CUDA_CACHE_PATH`, `TEMP` and `TMP`. Before first simulator startup, configure its user/cache/log locations to `.runtime/isaac/{user,cache,logs}` and audit actual resolved paths. Do not run until those locations are verified, because default Kit caches can write to the user profile. Do not upgrade the existing shared ML environment.

Linux alternative: a **new isolated** Ubuntu 24.04 WSL distro with its VHDX in `.runtime/wsl/Track3-Ubuntu24/`, including Docker/containerd storage inside that project-owned VHDX. This needs explicit installation/setup coordination. Do not point overlay2 at an ordinary `/mnt/z` NTFS folder, migrate shared Ubuntu/Docker, or grant shared Docker permissions as a shortcut. No command to create this distro has been executed or approved. A container route must also specify/pin its image digest and size before approval.

Exact missing components before candidate native simulation: Python 3.11, the stated Isaac Sim packages, pinned Isaac Lab source, cu128 torch/torchvision in its new environment, and one selected RL backend after the task interface is known. Exact resolved package sizes and transitive lock are not yet known; the 50 GB official minimum is not a download estimate. Keep a 120 GiB runtime/cache allowance and resolve package metadata/asset needs before making a final install request.

## Remaining acceptance steps

1. Confirm official embodiment, simulator/Lab revision, action/observation contract and task assets with the contract audit; do not adopt a generic sample as Track 3 compliance.
2. Resolve a pinned package plan and storage mapping; obtain the required confirmation for new production dependencies. No approval is requested for an unspecified install.
3. After the chosen runtime is installed, inspect its actual APIs, then create a bounded headless smoke: one environment, 120 physics steps, finite state checks, one policy training update, saved/reloaded checkpoint, and a 120-frame RGB recording (e.g. 320×240). Verify frame count and nonempty changing images. Record scene/seed, total VRAM/RAM, disk growth and wall time.
4. Run the official task separately. A falling cube/cartpole or this MLP is never an official task success claim.

No simulator script was fabricated against uninstalled APIs. Simulator feasibility remains an actual blocker; the tested torch runtime is useful progress but does not close foundation milestone 02.
