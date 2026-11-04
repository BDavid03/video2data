#!/usr/bin/env bash
# Archive the committed report, regenerate its figures, and build a fresh CPU PDF.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN=false
if [[ "${1:-}" == --run ]]; then RUN=true; shift; fi
TEX="${1:-docs/report/track3/main.tex}"
[[ $# -le 1 && "$TEX" == docs/report/track3/main.tex ]] || {
  echo 'usage: d3_report.sh [--run] [docs/report/track3/main.tex]' >&2; exit 2;
}
COMMIT="$(git rev-parse HEAD)"
export SOURCE_DATE_EPOCH="$(python3 -c "import json;print(json.load(open('scripts/reproduce/release.json'))['report']['epoch'])")"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
WORK="$ROOT/data/.runtime/tmp/reproduce-d3-<new>"
if "$RUN"; then WORK="$(mktemp -d "$ROOT/data/.runtime/tmp/reproduce-d3-XXXXXXXX")"; fi
export MPLCONFIGDIR="$WORK/matplotlib" TECTONIC_CACHE_DIR="$WORK/tectonic"
export XDG_CACHE_HOME="$WORK/cache" TMPDIR="$WORK/tmp" TEMP="$WORK/tmp" TMP="$WORK/tmp"
printf 'Commit: %s\nSOURCE_DATE_EPOCH: %s\nOutputs/logs: %s\n' "$COMMIT" "$SOURCE_DATE_EPOCH" "$WORK"
command() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if "$RUN"; then
    { printf '+'; printf ' %q' "$@"; printf '\n'; "$@"; } 2>&1 | tee -a "$WORK/build.log"
  fi
}
command mkdir -p "$WORK/src" "$MPLCONFIGDIR" "$TECTONIC_CACHE_DIR" "$TMPDIR"
command bash -o pipefail -c 'git ls-files -z docs/report/track3 | tar --null -T - -cf - | tar -x -C "$2"' _ "$COMMIT" "$WORK/src"
command cp -a "$ROOT/data/.cache/tectonic/." "$TECTONIC_CACHE_DIR/"
FIGURES="$WORK/src/docs/report/track3/figures"
# Read recorded inputs from this checkout; write figure PDFs into the archived tree.
# the archive above is taken from the working tree, so the layout rewrite needs no commit first
command data/.runtime/linux-cuda/bin/python -B docs/report/track3/figures/make_figures.py --output-dir "$FIGURES"
command data/.runtime/linux-cuda/bin/python -B docs/report/track3/figures/make_figures.py --qualitative --output-dir "$FIGURES"
command data/.runtime/linux-cuda/bin/python -B docs/report/track3/figures/make_figures.py --pose-offsets --output-dir "$FIGURES"
command data/.runtime/linux-cuda/bin/python -B docs/report/track3/figures/make_figures.py --bars --output-dir "$FIGURES"
# Exact pinned build_pdf.sh engine arguments, with a private writable cache.
command "$ROOT/data/.runtime/tools/tectonic-0.17.0/tectonic" -X compile -Z continue-on-errors --keep-logs "$WORK/src/$TEX"
if "$RUN"; then
  if grep '^! ' "$WORK/src/${TEX%.tex}.log" | grep -v 'microtype Error: The tracking feature only works with pdftex'; then
    echo "Unexpected TeX errors; see $WORK/build.log" >&2; exit 1
  fi
  printf '%s\n' "$COMMIT" > "$WORK/commit.txt"
  printf '%s\n' "$SOURCE_DATE_EPOCH" > "$WORK/source-date-epoch.txt"
  sha256sum docs/report/track3/figures/make_figures.py > "$WORK/figure-generator.sha256"
  sha256sum "$FIGURES/"*.pdf "$WORK/src/${TEX%.tex}.pdf" | tee "$WORK/sha256.txt"
fi
