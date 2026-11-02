#!/usr/bin/env bash
# Rebuild a recorded D1 mesh request; default is a read-only v2 CPU plan.
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
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

p = argparse.ArgumentParser(description="D1 v2/v3/v4 build, local CD-O, kit pack and preflight; CPU only")
p.add_argument("commit_url", nargs="?", help="defaults to the recorded (currently placeholder) URL")
p.add_argument("--manifest", type=Path)
p.add_argument("--version", choices=("v2", "v3", "v4"), default="v2")
p.add_argument("--run", action="store_true", help="execute; otherwise print the plan")
a = p.parse_args()
root = Path.cwd()
manifests = sorted(root.glob(f"submissions/*-d1-{a.version}/manifest.json"))
manifest = a.manifest or manifests[-1]
m = json.loads(manifest.read_text())
recorded_python = Path(m["pack_command"][0])
recorded_root = recorded_python.parents[3] if recorded_python.is_absolute() else root


def rebase(value):
    path = Path(value)
    return str(root / path.relative_to(recorded_root)) if path.is_absolute() else value


def relative(value):
    path = Path(value)
    return path.relative_to(root).as_posix() if path.is_absolute() else str(value)


request = Path("scripts/track3/requests/d1") / ("build-v2-r2.json" if a.version == "v2" else f"build-{a.version}.json")
build_run = Path(rebase(m["build_run"]))
record = json.loads((root / build_run).read_text())
request_data = json.loads(request.read_text())
if request_data != record["request"]:
    raise ValueError("Committed request differs from the manifest's build request")
url = a.commit_url or m["code_commit_url"]
if a.run and a.commit_url is None:
    p.error("--run requires an explicit commit URL")
print(f"Manifest: {manifest}\nRecorded build: {build_run}\nMeasure runs: {len(request_data['measure_runs'])}", flush=True)
scratch = root / ".runtime/tmp"
if a.run:
    scratch.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix=f"reproduce-d1-{a.version}-", dir=scratch)) if a.run else Path(f".runtime/tmp/reproduce-d1-{a.version}-<new>")
print(f"Outputs/logs: {work}", flush=True)
commands = []

def command(argv, log, env=None):
    argv = list(map(str, argv))
    commands.append(argv)
    print("+ " + shlex.join(argv), flush=True)
    if a.run:
        with (work / log).open("w") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)

build = [".runtime/linux-cuda/bin/python", "-B", "-m", "track3.features.mesh_extents", "build", "--request", request]
command(build, "build.log")
new_run = Path((work / "build.log").read_text().strip().splitlines()[-1]) if a.run else Path("<new-build>/run.json")
new_run = Path(relative(new_run))
score_temp = work / "tmp"
if a.run:
    score_temp.mkdir()
score_env = dict(os.environ, TMPDIR=str(score_temp), TEMP=str(score_temp), TMP=str(score_temp))
command([".runtime/kit-env/bin/python", "-B", "scripts/track3/local_cdo.py", "--candidates", new_run.parent / "objects", "--out", work / "cdo.json"], "score.log", env=score_env)
# Rebase the recorded commands so the checkout can move without editing run IDs.
pack = [rebase(s) for s in m["pack_command"]]
packed = work / f"track3_meshes_{a.version}.parquet"
for flag, value in (("--meshes", new_run.parent / "meshes"), ("--commit", url), ("--out", packed)):
    pack[pack.index(flag) + 1] = str(value)
command(pack, "pack.log")
preflight = [rebase(s) for s in m["preflight_command"]]
command(preflight, "preflight.log")
if a.run:
    if "Ready to submit" not in (work / "preflight.log").read_text():
        raise ValueError("Kit preflight did not report Ready to submit")
    with packed.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"file": str(packed.relative_to(root)), "sha256": digest, "build_run": relative(new_run),
              "source_manifest": relative(manifest), "code_commit_url": url, "commands": [[relative(s) for s in command] for command in commands],
              "pack_command": [relative(s) for s in pack], "preflight_command": [relative(s) for s in preflight],
              "inputs": [{"path": str(request), "sha256": hashlib.sha256(request.read_bytes()).hexdigest()}],
              "preflight_verdict": "Ready to submit", "status": "completed"}
    (work / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(work / "manifest.json")
PY
