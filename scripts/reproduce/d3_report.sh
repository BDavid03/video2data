#!/usr/bin/env bash
# Regenerate the report's current qualitative figure, then build its PDF on CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN=false
if [[ "${1:-}" == --run ]]; then RUN=true; shift; fi
TEX="${1:-docs/report/track3/main.tex}"
[[ $# -le 1 && "$TEX" != --* ]] || { echo 'usage: d3_report.sh [--run] [main.tex]' >&2; exit 2; }
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR="$ROOT/.runtime/tmp/reproduce-report-matplotlib"
if "$RUN"; then
  WORK="$(mktemp -d "$ROOT/.runtime/tmp/reproduce-d3-XXXXXXXX")"
  echo "Command/output log: $WORK/build.log"
fi
command() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if "$RUN"; then
    { printf '+'; printf ' %q' "$@"; printf '\n'; "$@"; } 2>&1 | tee -a "$WORK/build.log"
  fi
}
command .runtime/linux-cuda/bin/python -B "$(dirname "$TEX")/figures/make_figures.py"
command bash scripts/report/build_pdf.sh "$TEX"
