#!/usr/bin/env bash
# Re-hash only the entries actually recorded in a packed manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ROOT"
if [[ "${1:-}" == --historical ]]; then export V2D_HISTORICAL=1; shift; fi
printf '+ data/.runtime/linux-cuda/bin/python -B - %q\n' "${1:?usage: bash scripts/reproduce/verify.sh manifest.json}"
data/.runtime/linux-cuda/bin/python -B - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root))
from src import historical
from src.paths import current_name
manifest = Path(sys.argv[1]).resolve()
manifest.relative_to(root)
data = json.loads(manifest.read_text())
moved_paths = json.loads((root / "scripts/reproduce/moved-paths.json").read_text())
entries = []

def collect(value):
    if isinstance(value, dict):
        if "sha256" in value:
            name = value.get("path", value.get("file"))
            if name is None:
                raise ValueError("Hash entry has no path or file")
            entries.append((name, value["sha256"]))
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)

collect(data)
packed = data["file"]
packed_entry = (packed["path"], packed["sha256"]) if isinstance(packed, dict) else (packed, data["sha256"])
if not entries or packed_entry not in entries:
    raise ValueError("Manifest must record its packed file and SHA-256")
bad = 0
for name, expected in entries:
    path = (root / current_name(name)).resolve()
    moved = moved_paths.get(name) if not path.exists() else None
    if moved is not None:
        path = (root / current_name(moved)).resolve()
    path.relative_to(root)  # No external paths, including symlinks out of the checkout.
    try:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        detail = actual
    except OSError as error:
        actual = None
        detail = str(error)
    ok = actual is not None and actual == expected
    if not ok and historical.active() and historical.frozen_digest(name) == expected:
        ok, detail = True, "frozen source at " + historical.commit()[:10]
        historical.DRIFT[name] = expected
    bad += not ok
    relocation = f" (moved to {moved})" if moved is not None else ""
    print(f"{'OK' if ok else 'MISMATCH'} {name}{relocation}")
    if not ok:
        print(f"  expected {expected}\n  actual   {detail}")
print(f"SUMMARY {len(entries)} entries: {len(entries) - bad} OK, {bad} MISMATCH")
sys.exit(1 if bad else 0)
PY
