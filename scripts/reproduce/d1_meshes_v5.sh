#!/usr/bin/env bash
# Repack only a G2 production tree that passed development and the exact packed veto.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == --historical ]]; then export V2D_HISTORICAL=1; shift; fi
data/.runtime/linux-cuda/bin/python -B - "$@" <<'PY'
import argparse
from src.features.mesh_generative.g2_pack import repack
p=argparse.ArgumentParser(description='Gated D1 v5 repack; default prints the command only')
p.add_argument('commit_url')
p.add_argument('--out',default='data/outputs/mesh_generative/g2-real-pack/track3_meshes_v5.parquet')
p.add_argument('--run',action='store_true')
a=p.parse_args()
print(repack(a.commit_url,a.out,a.run))
PY
