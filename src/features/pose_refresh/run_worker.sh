#!/usr/bin/env bash
# Launch the FoundationPose registration worker under the parent's GPU lease (refresh slice).
set -euo pipefail
source data/.runtime/foundationpose/use-env.sh
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:${PYTHONPATH:-}"
CACHE="$PWD/data/outputs/refresh/frame0-fp/registration/cache"
mkdir -p "$CACHE/tmp" "$CACHE/xdg" "$CACHE/warp" "$CACHE/torch-extensions" "$CACHE/matplotlib" "$CACHE/torch"
export TMPDIR="$CACHE/tmp" XDG_CACHE_HOME="$CACHE/xdg" WARP_CACHE_PATH="$CACHE/warp"
export TORCH_EXTENSIONS_DIR="$CACHE/torch-extensions" MPLCONFIGDIR="$CACHE/matplotlib" TORCH_HOME="$CACHE/torch"
exec data/.runtime/foundationpose/env/bin/python -B -m src.features.pose_refresh.frame0_fp_worker "$@"
