#!/usr/bin/env bash
# Verify or reproduce the v4 CPU composition; --run packs only after a code commit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
exec .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.scored_stereo_pack "$@"
