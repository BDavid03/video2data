#!/usr/bin/env bash
# Reproduce a v1 GPU campaign or a v2 CPU composition, then pack and preflight.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1
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
import shutil
import subprocess
import tempfile

p = argparse.ArgumentParser(description="D2 v1: GPU campaign; D2 v2: CPU export composition, pack and preflight")
p.add_argument("results_dir", nargs="?")
p.add_argument("commit_url", nargs="?")
p.add_argument("--manifest", type=Path)
p.add_argument("--run", action="store_true", help="execute (v1 requires an authorised GPU job; v2 is CPU only)")
a = p.parse_args()
root = Path.cwd()
manifest = a.manifest or sorted(root.glob("submissions/*-d2-v*/manifest.json"))[-1]
m = json.loads(manifest.read_text())
# Older manifests encode their version in the packed filename, not a version key.
version = m.get("version", Path(m["file"]).stem.rsplit("_", 1)[-1])
if version not in ("v1", "v2"):
    p.error(f"Unsupported manifest version: {version}")
# v2 needs only a URL; retain the original RESULTS_DIR URL interface for v1.
if a.results_dir is not None and a.results_dir.startswith("https://") and a.commit_url is None:
    a.commit_url, a.results_dir = a.results_dir, None
url = a.commit_url or m["code_commit_url"]
if a.run and a.commit_url is None:
    p.error("--run requires an explicit commit URL")
episodes = [int(Path(e["path"]).stem.split("_")[1]) for e in m["exports"]]
if set(episodes) & {41, 43, 44}:
    raise ValueError("Forbidden episode in manifest exports")
if version == "v2" and a.results_dir is not None:
    p.error("v2 creates its own fresh temporary results directory; pass only COMMIT_URL")
results = (root / (a.results_dir or Path(m["results_dir"]))).resolve()
results.relative_to(root)
if version == "v1" and a.run and results.exists() and any(results.iterdir()):
    p.error(f"{results} already holds exports; pass a new, empty RESULTS_DIR so recorded exports stay intact")
work = Path(tempfile.mkdtemp(prefix="reproduce-d2-", dir=root / ".runtime/tmp")) if a.run else Path(".runtime/tmp/reproduce-d2-<new>")
if version == "v2":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = work / "results"
print(f"Manifest: {manifest}\nVersion: {version}\nResults: {results}\nBudget: {m['budget']}\nEvaluation episodes: {episodes}\nLogs: {work}", flush=True)
commands = []
command_results = []

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

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
    if (set(composition["K1a"]) != {1, 28} or len(composition["v1"]) != 18
            or len(episodes) != 20 or len(set(episodes)) != 20
            or set(episodes) != set(composition["K1a"] + composition["v1"])):
        raise ValueError("v2 must compose eighteen v1 exports and K1a episodes 1/28")
    for export, episode in zip(m["exports"], episodes):
        source = (root / export["source"]).resolve()
        source.relative_to(root)
        folder = "rebuilt-v2" if episode in composition["K1a"] else "results-v1"
        expected = root / "outputs/pose_insurance" / folder / Path(export["path"]).name
        if source != expected or digest(source) != export["sha256"]:
            raise ValueError(f"Recorded export source/path hash mismatch: {source}")
        inputs.append({"path": str(source.relative_to(root)), "sha256": export["sha256"]})
        print(f"OK source {source.relative_to(root)} {export['sha256']}", flush=True)
    if a.run:
        (work / "sources.json").write_text(json.dumps(inputs, indent=2) + "\n")
        results.mkdir()
        for source, export in zip(inputs, m["exports"]):
            shutil.copyfile(root / source["path"], results / Path(export["path"]).name)
    print("v2: copy verified exports; GPU regeneration is not executed.", flush=True)
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
    if version == "v2" and [e["sha256"] for e in exports] != [e["sha256"] for e in m["exports"]]:
        raise ValueError("Copied exports differ from the recorded source hashes")
    result = {"file": str(packed.relative_to(root)), "sha256": digest(packed), "results_dir": str(results.relative_to(root)),
              "exports": exports, "budget": m["budget"], "source_manifest": str(manifest), "code_commit_url": url,
              "version": version, "commands": commands[loop_count:], "loop_commands": commands[:loop_count],
              "inputs": inputs, "command_results": command_results,
              "preflight_verdict": "Ready to submit", "status": "completed"}
    (work / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(work / "manifest.json")
PY
