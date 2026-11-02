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


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_mesh_identity(candidate, reference):
    """Require the same relative PLY file set and SHA-256 for every rebuilt mesh."""
    old = {path.relative_to(reference).as_posix(): digest(path) for path in reference.rglob("*.ply")}
    new = {path.relative_to(candidate).as_posix(): digest(path) for path in candidate.rglob("*.ply")}
    if not old:
        raise ValueError(f"PLY identity: no recorded meshes at {reference}")
    if old.keys() != new.keys():
        raise ValueError(f"PLY identity: file set differs; missing={sorted(old.keys() - new.keys())}, extra={sorted(new.keys() - old.keys())}")
    for name, sha256 in old.items():
        if new[name] != sha256:
            raise ValueError(f"PLY identity: SHA-256 mismatch: {name}")
    return {"status": "passed", "reference": str(reference), "files": len(old), "sha256_by_file": old}


def check_parquet_identity(candidate, reference, expected_url, recorded_url, recorded_sha256):
    """Require the same Parquet schema and ordered values except for the supplied URL."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if digest(reference) != recorded_sha256:
        raise ValueError(f"Parquet identity: recorded packed SHA-256 mismatch: {reference}")
    old, new = pq.read_table(reference), pq.read_table(candidate)
    if not old.schema.equals(new.schema):
        raise ValueError("Parquet identity: column names/order or types differ")
    if old.column_names.count("code_commit_url") != 1 or len(set(old.column_names)) != len(old.column_names):
        raise ValueError("Parquet identity: recorded columns are missing or ambiguous")
    if old.num_rows != new.num_rows or not old.num_rows:
        raise ValueError(f"Parquet identity: row count differs or is empty ({old.num_rows} vs {new.num_rows})")
    for table, url in ((old, recorded_url), (new, expected_url)):
        column = table["code_commit_url"]
        if column.null_count or not pc.all(pc.equal(column, url)).as_py():
            raise ValueError("Parquet identity: code_commit_url mismatch (must match every row)")
    for name in old.column_names:
        if name != "code_commit_url" and not old[name].equals(new[name]):
            raise ValueError(f"Parquet identity: ordered values differ in column {name}")
    return {"status": "passed", "reference": str(reference), "reference_sha256": recorded_sha256,
            "rows": old.num_rows, "columns": old.column_names, "except_column": "code_commit_url",
            "url_all_rows": expected_url}


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
identity_checks = {}
if a.run and a.version == "v4":
    identity_checks["meshes"] = check_mesh_identity(new_run.parent / "meshes", build_run.parent / "meshes")
    print(f"OK PLY identity: {identity_checks['meshes']['files']} files; same file set and SHA-256", flush=True)
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
if a.run and a.version == "v4":
    identity_checks["parquet"] = check_parquet_identity(packed, root / m["file"], url, m["code_commit_url"], m["sha256"])
    print(f"OK Parquet identity: {identity_checks['parquet']['rows']} rows; only code_commit_url may differ", flush=True)
preflight = [rebase(s) for s in m["preflight_command"]]
command(preflight, "preflight.log")
if a.run:
    if "Ready to submit" not in (work / "preflight.log").read_text():
        raise ValueError("Kit preflight did not report Ready to submit")
    result = {"file": str(packed.relative_to(root)), "sha256": digest(packed), "build_run": relative(new_run),
              "source_manifest": relative(manifest), "code_commit_url": url, "commands": [[relative(s) for s in command] for command in commands],
              "pack_command": [relative(s) for s in pack], "preflight_command": [relative(s) for s in preflight],
              "inputs": [{"path": str(path), "sha256": digest(path)} for path in (request, Path("scripts/reproduce/d1_meshes.sh"))],
              "identity_checks": identity_checks,
              "preflight_verdict": "Ready to submit", "status": "completed"}
    (work / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(work / "manifest.json")
PY
