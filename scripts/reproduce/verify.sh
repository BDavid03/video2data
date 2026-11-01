#!/usr/bin/env bash
# Re-hash only the entries actually recorded in a packed manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
printf '+ .runtime/linux-cuda/bin/python -B - %q\n' "${1:?usage: bash scripts/reproduce/verify.sh manifest.json}"
.runtime/linux-cuda/bin/python -B - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd()
manifest = Path(sys.argv[1]).resolve()
manifest.relative_to(root)
data = json.loads(manifest.read_text())
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
if not entries or (data["file"], data["sha256"]) not in entries:
    raise ValueError("Manifest must record its packed file and SHA-256")
bad = 0
for name, expected in entries:
    path = (root / name).resolve()
    path.relative_to(root)  # No external paths, including symlinks out of the checkout.
    try:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        detail = actual
    except OSError as error:
        actual = None
        detail = str(error)
    ok = actual == expected
    bad += not ok
    print(f"{'OK' if ok else 'MISMATCH'} {name}")
    if not ok:
        print(f"  expected {expected}\n  actual   {detail}")
print(f"SUMMARY {len(entries)} entries: {len(entries) - bad} OK, {bad} MISMATCH")
sys.exit(1 if bad else 0)
PY
