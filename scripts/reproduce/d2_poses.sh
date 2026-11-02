#!/usr/bin/env bash
# Reproduce a v1 GPU campaign or a v2/v3 CPU composition, then pack and preflight.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1
printf '+ .runtime/linux-cuda/bin/python -B -'
for arg in "$@"; do printf ' %q' "$arg"; done
printf '\n'
.runtime/linux-cuda/bin/python -B - "$@" <<'PY'
import argparse
import csv
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

p = argparse.ArgumentParser(description="D2 v1: GPU campaign; D2 v2/v3: CPU export composition, pack and preflight")
p.add_argument("results_dir", nargs="?")
p.add_argument("commit_url", nargs="?")
p.add_argument("--manifest", type=Path)
p.add_argument("--version", choices=("v1", "v2", "v3"), help="select the recorded version (default: latest)")
p.add_argument("--run", action="store_true", help="execute (v1 requires an authorised GPU job; v2/v3 are CPU only)")
a = p.parse_args()
root = Path.cwd()
manifest = a.manifest or sorted(root.glob(f"submissions/*-d2-{a.version or 'v*'}/manifest.json"))[-1]
m = json.loads(manifest.read_text())
# Older manifests encode their version in the packed filename, not a version key.
version = m.get("version", Path(m["file"]).stem.rsplit("_", 1)[-1])
if version not in ("v1", "v2", "v3"):
    p.error(f"Unsupported manifest version: {version}")
if a.version is not None and version != a.version:
    p.error(f"Manifest version {version} differs from --version {a.version}")
# v2 needs only a URL; retain the original RESULTS_DIR URL interface for v1.
if a.results_dir is not None and a.results_dir.startswith("https://") and a.commit_url is None:
    a.commit_url, a.results_dir = a.results_dir, None
url = a.commit_url or m["code_commit_url"]
if a.run and a.commit_url is None:
    p.error("--run requires an explicit commit URL")
episodes = [int(Path(e["path"]).stem.split("_")[1]) for e in m["exports"]]
if set(episodes) & {41, 43, 44}:
    raise ValueError("Forbidden episode in manifest exports")
if version in ("v2", "v3") and a.results_dir is not None:
    p.error("v2/v3 create a fresh temporary results directory; pass only COMMIT_URL")
results = (root / (a.results_dir or Path(m["results_dir"]))).resolve()
results.relative_to(root)
if version == "v1" and a.run and results.exists() and any(results.iterdir()):
    p.error(f"{results} already holds exports; pass a new, empty RESULTS_DIR so recorded exports stay intact")
prefix = "reproduce-d2-"
work = Path(tempfile.mkdtemp(prefix=prefix, dir=root / ".runtime/tmp")) if a.run else Path(f".runtime/tmp/{prefix}<new>")
if version in ("v2", "v3"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = work / "results"
print(f"Manifest: {manifest}\nVersion: {version}\nResults: {results}\nBudget: {m['budget']}\nEvaluation episodes: {episodes}\nLogs: {work}", flush=True)
commands = []
command_results = []

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_csv_identity(candidate, reference, expected_url, recorded_url, recorded_sha256):
    """Require identical CSV fields and row order except for the explicitly supplied URL."""
    if digest(reference) != recorded_sha256:
        raise ValueError(f"CSV identity: recorded packed SHA-256 mismatch: {reference}")
    with reference.open(newline="") as old_stream, candidate.open(newline="") as new_stream:
        old_rows, new_rows = csv.reader(old_stream, strict=True), csv.reader(new_stream, strict=True)
        header = next(old_rows, None)
        if not header or header.count("code_commit_url") != 1 or len(set(header)) != len(header):
            raise ValueError("CSV identity: recorded header is missing or ambiguous")
        if next(new_rows, None) != header:
            raise ValueError("CSV identity: column names/order differ")
        url_index = header.index("code_commit_url")
        rows = 0
        for rows, (old, new) in enumerate(zip_longest(old_rows, new_rows), start=1):
            if old is None or new is None:
                raise ValueError(f"CSV identity: row count differs at data row {rows}")
            if len(old) != len(header) or len(new) != len(header):
                raise ValueError(f"CSV identity: column count differs at data row {rows}")
            if old[url_index] != recorded_url or new[url_index] != expected_url:
                raise ValueError(f"CSV identity: code_commit_url mismatch at data row {rows}")
            for index, name in enumerate(header):
                if index != url_index and old[index] != new[index]:
                    raise ValueError(f"CSV identity: data row {rows}, column {name} differs")
        if not rows:
            raise ValueError("CSV identity: no data rows")
    same_url = expected_url == recorded_url
    if same_url and digest(candidate) != recorded_sha256:
        raise ValueError("CSV identity: unchanged URL requires the recorded whole-file SHA-256")
    return {"status": "passed", "reference": str(reference), "reference_sha256": recorded_sha256,
            "rows": rows, "columns": header, "except_column": "code_commit_url",
            "url_all_rows": expected_url,
            "whole_file_hash_check": "passed" if same_url else "not required: URL changed"}


def command(argv, log):
    argv = list(map(str, argv))
    commands.append(argv)
    print("+ " + shlex.join(argv), flush=True)
    if a.run:
        with (work / log).open("w") as stream:
            completed = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        command_results.append({"argv": argv, "log": log, "exit_code": completed.returncode})
        (work / "command_results.json").write_text(json.dumps(command_results, indent=2) + "\n")
        completed.check_returncode()

inputs = []
if version == "v1":
    queue = json.loads(Path("track3/features/pose_insurance/queue.json").read_text())
    queue["budgets"]["learned"] = m["budget"]
    queue["evaluation_episodes"] = episodes
    queue_path = work / "queue.json"
    if a.run:
        queue_path.write_text(json.dumps(queue, indent=2) + "\n")
        inputs.append({"path": str(queue_path.relative_to(root)), "sha256": digest(queue_path)})
    # Bind the existing loop destination; its handlers, ledger and lease are unchanged.
    entry = "import sys; from pathlib import Path; from track3.features.pose_insurance import loop; loop.RESULTS = Path(sys.argv.pop(1)); loop.main()"
    for phase in ("tasks", "evaluation"):
        command([".runtime/flashchord/venv/bin/python", "-B", "-c", entry, results, "--phase", phase, "--queue", queue_path], phase + ".log")
else:
    # Verify all sources before copying any; never import the GPU campaign for v2.
    composition = m["composition"]
    if version == "v2":
        if (set(composition["K1a"]) != {1, 28} or len(composition["v1"]) != 18
                or len(episodes) != 20 or len(set(episodes)) != 20
                or set(episodes) != set(composition["K1a"] + composition["v1"])):
            raise ValueError("v2 must compose eighteen v1 exports and K1a episodes 1/28")
    else:
        changed, fallback = composition["S1"], composition["v2"]
        if (len(episodes) != 20 or len(set(episodes)) != 20 or not changed
                or set(changed) & set(fallback) or set(episodes) != set(changed + fallback)
                or set(changed) & {1, 28, 35, 36, 37, 38}):
            raise ValueError("v3 must partition twenty episodes into S1 and unchanged v2 exports")
    for export, episode in zip(m["exports"], episodes):
        source = (root / export["source"]).resolve()
        source.relative_to(root)
        if version == "v2":
            folder = "rebuilt-v2" if episode in composition["K1a"] else "results-v1"
            expected = root / "outputs/pose_insurance" / folder / Path(export["path"]).name
            valid_path = source == expected
        elif episode in composition["S1"]:
            valid_path = (source.is_relative_to(root / "outputs/pose_insurance/run-v3")
                          and source.name == "trajectory.parquet" and source.parent.name == f"episode_{episode:06d}")
        else:
            valid_path = source == root / "outputs/pose_insurance/results-v2" / Path(export["path"]).name
        if not valid_path or digest(source) != export["sha256"]:
            raise ValueError(f"Recorded export source/path hash mismatch: {source}")
        inputs.append({"path": str(source.relative_to(root)), "sha256": export["sha256"]})
        print(f"OK source {source.relative_to(root)} {export['sha256']}", flush=True)
    if a.run:
        (work / "sources.json").write_text(json.dumps(inputs, indent=2) + "\n")
        results.mkdir()
        for source, export in zip(inputs, m["exports"]):
            shutil.copyfile(root / source["path"], results / Path(export["path"]).name)
    print(f"{version}: copy verified exports; GPU regeneration is not executed.", flush=True)
loop_count = len(commands)
old_root = str(Path(m["commands"][0][0]).parents[3])
packed = work / Path(m["file"]).name
for index, recorded in enumerate(m["commands"]):
    argv = [str(root) + s[len(old_root):] if s.startswith(old_root + "/") else s for s in recorded]
    for flag, value in (("--results", results), ("--code-commit", url), ("--out", packed)):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(value)
    command(argv, "pack.log" if index == 0 else "preflight.log")
if a.run:
    if "Ready to submit" not in (work / "preflight.log").read_text():
        raise ValueError("Kit preflight did not report Ready to submit")
    exports = [{"path": str((results / Path(e["path"]).name).relative_to(root)),
                "sha256": digest(results / Path(e["path"]).name)} for e in m["exports"]]
    if version in ("v2", "v3") and [e["sha256"] for e in exports] != [e["sha256"] for e in m["exports"]]:
        raise ValueError("Copied exports differ from the recorded source hashes")
    identity_checks = {}
    if version == "v3":
        identity_checks["csv"] = check_csv_identity(packed, root / m["file"], url, m["code_commit_url"], m["sha256"])
        print(f"OK CSV identity: {identity_checks['csv']['rows']} rows; only code_commit_url may differ", flush=True)
    result = {"file": str(packed.relative_to(root)), "sha256": digest(packed), "results_dir": str(results.relative_to(root)),
              "exports": exports, "budget": m["budget"], "source_manifest": str(manifest), "code_commit_url": url,
              "version": version, "commands": commands[loop_count:], "loop_commands": commands[:loop_count],
              "inputs": inputs, "command_results": command_results,
              "identity_checks": identity_checks,
              "reproduction_script": {"path": "scripts/reproduce/d2_poses.sh", "sha256": digest(Path("scripts/reproduce/d2_poses.sh"))},
              "preflight_verdict": "Ready to submit", "status": "completed"}
    (work / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(work / "manifest.json")
PY
