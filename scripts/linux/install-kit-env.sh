#!/usr/bin/env bash
# Install the submission kit's requirements.txt (plus scipy for the local CD-O scorer) into its own
# environment, .runtime/kit-env. Approved in docs/chain/PROGRAM.md (owner, 2026-11-01): packing needs it.
# The kit's kaggle package is installed because requirements.txt lists it; the chain never uploads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export UV_PYTHON_INSTALL_DIR="$ROOT/.runtime/python" UV_CACHE_DIR="$ROOT/.cache/uv" TMPDIR="$ROOT/.runtime/tmp"
UV="$ROOT/.runtime/uv/uv"
ENV="$ROOT/.runtime/kit-env"
"$UV" venv "$ENV" --python 3.13 --allow-existing
"$UV" pip install --python "$ENV/bin/python" \
  -r upstream/v2d_submission_kit/v2d_submission_kit/requirements.txt scipy
"$UV" pip freeze --python "$ENV/bin/python"
