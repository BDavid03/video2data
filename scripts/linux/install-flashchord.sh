#!/usr/bin/env bash
# Install the FlashCHORD runtime (the Track 3 policy evaluator) in its own environment.
# Approved by the owner on 2026-11-01 ("Isaac Lab/FlashCHORD is approved for install").
# FlashCHORD at aee95f25 runs on Newton + Warp + MuJoCo-Warp (pyproject.toml), not Isaac Sim,
# so this is its README's "host installation (without Docker)" route, pinned by uv.lock.
# Docker is unusable here: the user is not in the docker group (owner action, NEEDS-YOU).
# Everything (git-lfs, interpreter, venv, source clone, caches, logs) stays under the checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SHA=aee95f25ae0943f335d8bb42fa95b6a9acfa266f
LFS_VERSION=3.7.1
BASE="$ROOT/.runtime/flashchord"
SRC="$BASE/video_to_data"
VENV="$BASE/venv"
LOGS="$ROOT/.runtime/flashchord/install-logs"
mkdir -p "$BASE" "$LOGS" .runtime/tools .runtime/tmp .cache/uv
export UV_PYTHON_INSTALL_DIR="$ROOT/.runtime/python" UV_CACHE_DIR="$ROOT/.cache/uv"
export TMPDIR="$ROOT/.runtime/tmp" UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
UV="$ROOT/.runtime/uv/uv"

step() {  # step <name> <command...>; one log per step, stop at the first failure
  local name="$1"; shift
  echo "== $(date -u +%FT%TZ) $name: $*"
  if "$@" > "$LOGS/$name.log" 2>&1; then echo "   ok"; else
    echo "   FAILED, see $LOGS/$name.log"; tail -20 "$LOGS/$name.log"; exit 1; fi
}

# 1. git-lfs release binary, checksum-checked against the release's sha256sums file.
LFS_DIR="$ROOT/.runtime/tools/git-lfs-$LFS_VERSION"
if [ ! -x "$LFS_DIR/git-lfs" ]; then
  step git-lfs bash -c "
    set -e; d=\$(mktemp -d '$ROOT/.runtime/tmp/git-lfs.XXXX'); cd \"\$d\"
    tar=git-lfs-linux-amd64-v$LFS_VERSION.tar.gz
    curl -fsSLO https://github.com/git-lfs/git-lfs/releases/download/v$LFS_VERSION/\$tar
    curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v$LFS_VERSION/sha256sums.asc -o sums
    grep -E \" \\*?\$tar\$\" sums | sha256sum -c -
    tar xzf \$tar; mkdir -p '$LFS_DIR'; cp git-lfs-$LFS_VERSION/git-lfs '$LFS_DIR/'"
fi
export PATH="$LFS_DIR:$PATH"

# 2. Source at the evaluator's revision, with the LFS assets FlashCHORD loads.
if [ ! -d "$SRC/.git" ]; then
  step clone env GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/nvidia-isaac/video_to_data.git "$SRC"
fi
step checkout bash -c "cd '$SRC' && GIT_LFS_SKIP_SMUDGE=1 git checkout --detach $SHA && test \"\$(git rev-parse HEAD)\" = $SHA"
step lfs-pull bash -c "cd '$SRC' && git lfs install --local && git lfs pull --include 'robotic_grounding/flash_chord/**,robotic_grounding/source/**'"

# 3. Python 3.11.15 venv with the locked dependencies (train extra: JAX for PPO/FlashSAC; no dev extra).
step python "$UV" python install 3.11.15
step venv "$UV" venv "$VENV" --python 3.11.15 --allow-existing
step sync bash -c "cd '$SRC/robotic_grounding/flash_chord' && VIRTUAL_ENV='$VENV' '$UV' sync --active --frozen --extra train"
step freeze bash -c "'$UV' pip freeze --python '$VENV/bin/python' > '$LOGS/freeze.txt'"

# 4. Device check: Warp and JAX must both see the GPU.
step device "$VENV/bin/python" -c "
import warp as wp, jax, newton, mujoco, hydra
wp.init(); print('warp', wp.__version__, wp.get_cuda_devices())
print('jax', jax.__version__, jax.devices())
print('newton', newton.__version__, 'mujoco', mujoco.__version__, 'hydra', hydra.__version__)"
cat "$LOGS/device.log"
echo "installed: $VENV (source $SRC at $SHA)"
