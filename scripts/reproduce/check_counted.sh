#!/usr/bin/env bash
# Verify counted artifacts without rebuilding or repairing anything.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
exec data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.counted "$@"
