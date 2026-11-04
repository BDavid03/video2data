#!/usr/bin/env bash
# Install the Linux CUDA research environment inside this checkout.
# Authorized by the user on 2026-10-29 ("install linux cuda, don't use the windows mount").
# Everything (uv, interpreter, venv, caches, logs) stays under the project root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RECORD_DIR="$ROOT/data/reports/installation/linux-cuda-20261029"
mkdir -p "$RECORD_DIR" data/.runtime/uv data/.runtime/python data/.runtime/tmp data/.cache/uv
export UV_INSTALL_DIR="$ROOT/data/.runtime/uv" UV_NO_MODIFY_PATH=1
export UV_PYTHON_INSTALL_DIR="$ROOT/data/.runtime/python" UV_CACHE_DIR="$ROOT/data/.cache/uv"
export TMPDIR="$ROOT/data/.runtime/tmp" HF_HOME="$ROOT/data/.cache/huggingface" UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4
PY="$ROOT/data/.runtime/linux-cuda/bin/python"
STATUS="$RECORD_DIR/installation.json"
STEPS=()
record() {  # record <status> [error]
  local steps; steps=$(printf '%s\n' "${STEPS[@]:-}" | python3 -c 'import json,sys;print(json.dumps([l for l in sys.stdin.read().split("\n") if l]))')
  python3 - "$1" "${2:-}" "$steps" > "$STATUS" <<'PY'
import json, sys, datetime
status, error, steps = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
rec = {"schema_version": 1, "authorized_date": "2026-10-29",
       "authorization": "User instructed: install linux cuda, do not use the windows mount.",
       "environment": "data/.runtime/linux-cuda", "python_install_dir": "data/.runtime/python",
       "uv_dir": "data/.runtime/uv", "constraints": "data/configs/perception/moge/linux-cuda-constraints.txt",
       "requirements": "data/configs/perception/moge/linux-cuda-requirements.in",
       "frozen_lock": "data/configs/perception/moge/linux-cuda.lock.txt",
       "status": status, "steps": steps, "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
if error: rec["error"] = error
json.dump(rec, sys.stdout, indent=2); print()
PY
}
step() {  # step <name> <command...>
  local name="$1"; shift
  echo "== $name: $*"
  if "$@" > "$RECORD_DIR/$name.log" 2>&1; then STEPS+=("$name:completed"); else
    STEPS+=("$name:failed"); record installation_failed "step $name failed; see $name.log"; tail -30 "$RECORD_DIR/$name.log"; exit 1; fi
  record installing
}
record installing
if [ ! -x data/.runtime/uv/uv ]; then
  step uv-download bash -c 'curl -LsSf https://astral.sh/uv/install.sh -o data/.runtime/tmp/uv-install.sh && sh data/.runtime/tmp/uv-install.sh'
fi
UV="$ROOT/data/.runtime/uv/uv"
step python-install "$UV" python install 3.13
step venv "$UV" venv data/.runtime/linux-cuda --python 3.13 --allow-existing
# --index (not --extra-index-url) gives the cu128 wheel index priority over PyPI.
step torch "$UV" pip install --python "$PY" --index https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128
step packages "$UV" pip install --python "$PY" --index https://download.pytorch.org/whl/cu128 \
  -r data/configs/perception/moge/linux-cuda-requirements.in \
  -c data/configs/perception/moge/linux-cuda-constraints.txt
step freeze bash -c "'$UV' pip freeze --python '$PY' > data/configs/perception/moge/linux-cuda.lock.txt"
step cuda-check "$PY" -c 'import torch, cv2, numpy, scipy, pyarrow, trimesh, matplotlib; assert "cu128" in torch.__version__, torch.__version__; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda, torch.cuda.get_arch_list())'
record installed
echo "installed: $PY"; cat "$RECORD_DIR/cuda-check.log"
