#!/usr/bin/env bash
# Rebuild the recorded D1 v2 request; default is a read-only CPU plan.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
printf '+ .runtime/linux-cuda/bin/python -B -'
for arg in "$@"; do printf ' %q' "$arg"; done
printf '\n'
.runtime/linux-cuda/bin/python -B - "$@" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

p = argparse.ArgumentParser(description="D1 v2 build, local CD-O, kit pack and preflight; CPU only")
p.add_argument("commit_url", nargs="?", help="defaults to the recorded (currently placeholder) URL")
p.add_argument("--manifest", type=Path)
p.add_argument("--run", action="store_true", help="execute; otherwise print the plan")
a = p.parse_args()
root = Path.cwd()
manifest = a.manifest or sorted(root.glob("submissions/*-d1-v2/manifest.json"))[-1]
m = json.loads(manifest.read_text())
request = Path("scripts/track3/requests/d1/build-v2-r2.json")
record = json.loads((root / m["build_run"]).read_text())
if json.loads(request.read_text()) != record["request"]:
    raise ValueError("Committed request differs from the manifest's build request")
url = a.commit_url or m["code_commit_url"]
if a.run and a.commit_url is None:
    p.error("--run requires an explicit commit URL")
print(f"Manifest: {manifest}\nRecorded build: {m['build_run']}\nMeasure runs: {len(record['request']['measure_runs'])}", flush=True)
work = Path(tempfile.mkdtemp(prefix="reproduce-d1-", dir=root / ".runtime/tmp")) if a.run else Path(".runtime/tmp/reproduce-d1-<new>")
print(f"Outputs/logs: {work}", flush=True)
commands = []

def command(argv, log):
    argv = list(map(str, argv))
    commands.append(argv)
    print("+ " + shlex.join(argv), flush=True)
    if a.run:
        with (work / log).open("w") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)

build = [".runtime/linux-cuda/bin/python", "-B", "-m", "track3.features.mesh_extents", "build", "--request", request]
command(build, "build.log")
new_run = Path((work / "build.log").read_text().strip().splitlines()[-1]) if a.run else Path("<new-build>/run.json")
command([".runtime/kit-env/bin/python", "-B", "scripts/track3/local_cdo.py", "--candidates", new_run.parent / "objects", "--out", work / "cdo.json"], "score.log")
# Rebase the recorded commands so the checkout can move without editing run IDs.
old_root = str(Path(m["pack_command"][0]).parents[3])
pack = [str(root) + s[len(old_root):] if s.startswith(old_root + "/") else s for s in m["pack_command"]]
packed = work / Path(m["file"]).name
for flag, value in (("--meshes", new_run.parent / "meshes"), ("--commit", url), ("--out", packed)):
    pack[pack.index(flag) + 1] = str(value)
command(pack, "pack.log")
preflight = [str(root) + s[len(old_root):] if s.startswith(old_root + "/") else s for s in m["preflight_command"]]
command(preflight, "preflight.log")
if a.run:
    if "Ready to submit" not in (work / "preflight.log").read_text():
        raise ValueError("Kit preflight did not report Ready to submit")
    with packed.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"file": str(packed.relative_to(root)), "sha256": digest, "build_run": str(new_run),
              "source_manifest": str(manifest), "code_commit_url": url, "commands": commands,
              "pack_command": pack, "preflight_command": preflight,
              "inputs": [{"path": str(request), "sha256": hashlib.sha256(request.read_bytes()).hexdigest()}],
              "preflight_verdict": "Ready to submit", "status": "completed"}
    (work / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(work / "manifest.json")
PY
