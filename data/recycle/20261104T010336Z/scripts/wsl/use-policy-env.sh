#!/usr/bin/env bash
# Source this file; selecting an environment runs no Python and writes no files.
if [[ "${WSL_DISTRO_NAME:-}" != video2data-policy ]]; then
    printf '%s\n' 'Use only the dedicated video2data-policy distribution.' >&2
    return 1 2>/dev/null || exit 1
fi
track3_root=/mnt/z/hugging-face/video2data
case "${1:-policy}" in
    policy) export VIRTUAL_ENV=/opt/video2data/policy-python311 ;;
    retarget) export VIRTUAL_ENV=/opt/video2data/retarget-python311 ;;
    *) printf '%s\n' 'Environment must be policy or retarget.' >&2; return 1 2>/dev/null || exit 1 ;;
esac
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$track3_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export FLASH_CHORD_CACHE_DIR="$track3_root/.cache/flash-chord"
export WARP_CACHE_PATH="$track3_root/.cache/warp"
export ROBOTIC_GROUNDING_INTERMEDIATE_DIR="$track3_root/.cache/retargeting"
export XDG_CACHE_HOME="$track3_root/.cache/policy"
export XDG_CONFIG_HOME="$track3_root/.cache/policy-config"
export XDG_DATA_HOME="$track3_root/.cache/policy-data"
export TMPDIR="$track3_root/.cache/policy-tmp" TMP="$track3_root/.cache/policy-tmp" TEMP="$track3_root/.cache/policy-tmp"
export UV_CACHE_DIR="$track3_root/.cache/uv-policy" PIP_CACHE_DIR="$track3_root/.cache/pip-policy"
export TORCH_HOME="$track3_root/.cache/torch-policy" HF_HOME="$track3_root/.cache/huggingface-policy"
export TORCH_EXTENSIONS_DIR="$track3_root/.cache/torch-extensions-policy"
export TORCHINDUCTOR_CACHE_DIR="$track3_root/.cache/torch-inductor-policy"
export TRITON_CACHE_DIR="$track3_root/.cache/triton-policy"
export CUDA_CACHE_PATH="$track3_root/.cache/cuda-policy"
export MPLCONFIGDIR="$track3_root/.cache/matplotlib-policy" NUMBA_CACHE_DIR="$track3_root/.cache/numba-policy"
export WANDB_MODE=disabled WANDB_DIR="$track3_root/.cache/wandb-policy"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset track3_root
