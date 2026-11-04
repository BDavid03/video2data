#!/usr/bin/env bash
# Adapted from kit/examples/container/run.sh; the unmodified example is shipped alongside.
# Its generic runner cannot preserve our historical absolute task/asset paths or pack CD-O.
set -euo pipefail
SUBMISSION="${SUBMISSION:-/submission}"
KIT_PYTHON="${KIT_PYTHON:-${SUBMISSION}/runtime/kit-env/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-${SUBMISSION}/runtime/eval-env/bin/python}"
RUNTIME_ROOT="${SUBMISSION}/runtime/flashchord"
MANIFEST="${MANIFEST:-${SUBMISSION}/submission.json}"
export SUBMISSION KIT_PYTHON EVAL_PYTHON RUNTIME_ROOT MANIFEST
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl HYDRA_FULL_ERROR=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
command="${1:-}"; shift || true
case "$command" in
    verify)
        exec "$KIT_PYTHON" "$SUBMISSION/verifier_kit/tools/verify_container.py" --manifest "$MANIFEST" "$@" ;;
    reproduce|pack|verify-mesh|verify-copies)
        exec "$KIT_PYTHON" "$SUBMISSION/reproduce.py" "$command" --manifest "$MANIFEST" "$@" ;;
    *) echo 'Usage: run.sh verify|verify-copies|verify-mesh|reproduce|pack [options]' >&2; exit 2 ;;
esac
