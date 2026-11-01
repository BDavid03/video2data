#!/usr/bin/env bash
# Build a LaTeX document to PDF with the pinned Tectonic binary, cache kept inside the checkout.
#   bash scripts/report/build_pdf.sh <dir/main.tex>
# Tectonic 0.17.0 (x86_64 musl) from github.com/tectonic-typesetting/tectonic, release asset
# SHA-256 8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7, under .runtime/tools/.
# Tectonic is XeTeX: the template class asks microtype for tracking, which only pdfTeX has, so
# microtype raises one recoverable error and switches tracking off. That error is allowed; any
# other TeX error fails the build. (Drop `\pdfoutput=1` from main.tex: it misleads hyperref.)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEX="$(readlink -f "${1:?main.tex}")"
LOG="${TEX%.tex}.log"
export TECTONIC_CACHE_DIR="$ROOT/.cache/tectonic"
"$ROOT/.runtime/tools/tectonic-0.17.0/tectonic" -X compile -Z continue-on-errors --keep-logs "$TEX"
if grep '^! ' "$LOG" | grep -v 'microtype Error: The tracking feature only works with pdftex'; then
  echo "TeX errors above (see $LOG)" >&2
  exit 1
fi
echo "PDF ${TEX%.tex}.pdf"
