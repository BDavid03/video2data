#!/usr/bin/env bash
# Resume the frozen evaluation queue, then pack with the recorded kit commands.
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
from pathlib import Path
import shlex
import subprocess
import tempfile

p = argparse.ArgumentParser(description="D2 evaluation only; tasks then serial GPU evaluation, kit pack and preflight")
p.add_argument("results_dir", nargs="?", type=Path)
p.add_argument("commit_url", nargs="?")
p.add_argument("--manifest", type=Path)
p.add_argument("--run", action="store_true", help="execute only when the GPU schedule permits")
a = p.parse_args()
root = Path.cwd()
manifest = a.manifest or sorted(root.glob("submissions/*-d2-v*/manifest.json"))[-1]
m = json.loads(manifest.read_text())
results = (root / (a.results_dir or Path(m["results_dir"]))).resolve()
results.relative_to(root)
url = a.commit_url or m["code_commit_url"]
if a.run and a.commit_url is None:
    p.error("--run requires RESULTS_DIR and an explicit commit URL")
if a.run and results.exists() and any(results.iterdir()):
    p.error(f"{results} already holds exports; pass a new, empty RESULTS_DIR so recorded exports stay intact")
queue = json.loads(Path("track3/features/pose_insurance/queue.json").read_text())
queue["budgets"]["learned"] = m["budget"]
queue["evaluation_episodes"] = [int(Path(e["path"]).stem.split("_")[1]) for e in m["exports"]]
if set(queue["evaluation_episodes"]) & {41, 43, 44}:
    raise ValueError("Forbidden episode in manifest exports")
print(f"Manifest: {manifest}\nResults: {results}\nBudget: {m['budget']}\nEvaluation episodes: {queue['evaluation_episodes']}", flush=True)
work = Path(tempfile.mkdtemp(prefix="reproduce-d2-", dir=root / ".runtime/tmp")) if a.run else Path(".runtime/tmp/reproduce-d2-<new>")
queue_path = work / "queue.json"
print(f"Frozen queue copy and logs: {work}", flush=True)
if a.run:
    queue_path.write_text(json.dumps(queue, indent=2) + "\n")
commands = []

def command(argv, log):
    argv = list(map(str, argv))
    commands.append(argv)
    print("+ " + shlex.join(argv), flush=True)
    if a.run:
        with (work / log).open("w") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)

# The loop has no --results option. Bind its existing export destination, keeping
# its queue, resume admission, ledger, handlers and GPU lease unchanged.
entry = "import sys; from pathlib import Path; from track3.features.pose_insurance import loop; loop.RESULTS = Path(sys.argv.pop(1)); loop.main()"
for phase in ("tasks", "evaluation"):
    command([".runtime/flashchord/venv/bin/python", "-B", "-c", entry, results, "--phase", phase, "--queue", queue_path], phase + ".log")
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
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    exports = [{"path": str((results / Path(e["path"]).name).relative_to(root)),
                "sha256": digest(results / Path(e["path"]).name)} for e in m["exports"]]
    result = {"file": str(packed.relative_to(root)), "sha256": digest(packed), "results_dir": str(results.relative_to(root)),
              "exports": exports, "budget": m["budget"], "source_manifest": str(manifest), "code_commit_url": url,
              "commands": commands[2:], "loop_commands": commands[:2],
              "inputs": [{"path": str(queue_path.relative_to(root)), "sha256": digest(queue_path)}],
              "preflight_verdict": "Ready to submit", "status": "completed"}
    (work / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(work / "manifest.json")
PY
