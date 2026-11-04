#!/usr/bin/env bash
# CPU composition after per-episode scored adoption; the lead supplies the committed URL.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
exec data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.pack_moving "$@"
