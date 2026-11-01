#!/usr/bin/env bash
# Headless FlashCHORD smoke test: build the tissue-box Sharpa scene with the null viewer and compile
# every MuJoCo-Warp kernel. Proves the runtime starts on this GPU under WSL; it does not train or
# score anything. First run 2026-11-01 (context 01): exit 0, about 2 minutes of kernel compilation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FC="$ROOT/.runtime/flashchord/video_to_data/robotic_grounding"
LOG="${1:-$ROOT/outputs/flashchord/smoke.log}"
mkdir -p "$(dirname "$LOG")"
export WARP_CACHE_PATH="$ROOT/.cache/warp" PYTHONPATH="$FC/flash_chord/src"
PARQUET="$FC/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave"
"$ROOT/.runtime/flashchord/venv/bin/python" "$FC/flash_chord/scripts/debug/replay.py" \
  "task.parquet='$PARQUET'" viewer.backend=none viewer.realtime=false markers.enabled=false \
  "hydra.run.dir=$(dirname "$LOG")/hydra" > "$LOG" 2>&1
echo "exit $? ; $(grep -m1 '^replay:' "$LOG")"
