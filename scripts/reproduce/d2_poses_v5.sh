#!/usr/bin/env bash
# Verify or reproduce v5 on CPU; --run packs after the lead commits the frozen sources.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
if [[ "${1:-}" == --historical ]]; then export V2D_HISTORICAL=1; shift; fi
exec data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.scored_stereo_pack --version v5 "$@"
