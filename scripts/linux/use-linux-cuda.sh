# Source to select the Linux CUDA environment and pinned MoGe source trees.
# This changes the current shell only; it runs no Python.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VIRTUAL_ENV="$ROOT/data/.runtime/linux-cuda"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/upstream/moge:$ROOT/upstream/moge-utils3d:$ROOT/upstream/moge-pipeline"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$ROOT/data/.runtime/tmp" MPLCONFIGDIR="$ROOT/data/.cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
