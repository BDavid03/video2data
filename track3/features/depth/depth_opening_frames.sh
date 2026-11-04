#!/usr/bin/env bash
# MoGe2 depth on the opening frames 0-19 of each given episode, one GPU run at a time.
#   bash track3/features/depth/depth_opening_frames.sh <split> <episode> [<episode> ...]
# Each request is written under .runtime/tmp/req/ and snapshotted in the run's run.json.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.." || exit 64
split=${1:?split}; shift
mkdir -p .runtime/tmp/req
for ep in "$@"; do
  req=.runtime/tmp/req/depth-$split-$ep.json
  printf '{"source_split": "%s", "episode_index": %d, "start_frame": 0, "stop_frame": 20, "frame_stride": 1, "max_image_side": 768, "num_tokens": 1200, "use_fp16": true, "save_normals": false, "gpu_headroom_mib": 2048, "gpu_admission_allowance_mib": 4096}\n' "$split" "$ep" > "$req"
  echo "== $(date -u +%FT%TZ) episode $ep"
  PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.depth --request "$req" 2>&1
done
echo "== done $(date -u +%FT%TZ)"
