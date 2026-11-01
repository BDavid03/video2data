"""Pack a completely reconciled v1 queue with the official kit and write its manifest."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from track3.paths import output_path
from track3.provenance import run_fields, write_json, sha256_file
from track3.runs import ROOT, runtime_flag, heavy_execution, recorded, read_json
from .loop import OUTPUT, RESULTS, identity, jobs_for, reconcile

FEATURE = Path(__file__).parent
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"


def pack():
    runtime = runtime_flag(FEATURE, "pack_authorized", "D2 v1 packing")
    heavy_execution(dict(runtime,python=runtime["kit_python"]))
    queue_path = FEATURE / "queue.json"
    queue = read_json(queue_path)
    completion = read_json(OUTPUT / "evaluation_finished.json")
    expected = {identity(request) for request in jobs_for(queue,"evaluation")}
    selected = completion["selected"]
    if completion["status"] != "completed" or {r["identity"] for r in selected} != expected:
        raise ValueError("Completion marker does not match the complete current queue")
    exports = [reconcile(row) for row in selected]
    if set(RESULTS.iterdir()) != {ROOT/e["path"] for e in exports}:
        raise ValueError("The v1 flat directory has missing or extra files")
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    url = f"https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/{head}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    directory = output_path(ROOT,f"submissions/{stamp}-d2-v1")
    record = run_fields(ROOT,"pose_insurance","pack",{"budget":queue["budgets"]["learned"],"code_commit":url},
                        parents=[ROOT/r["run"] for r in selected],
                        inputs=[queue_path,KIT/"tools/pack_submission.py",KIT/"tools/preflight.py",
                                OUTPUT/"evaluation_finished.json",*[ROOT/e["path"] for e in exports]])
    with recorded(directory,record):
        csv_path = directory / "track3_poses_v1.csv"
        commands = [
            [str(ROOT/".runtime/kit-env/bin/python"),str(KIT/"tools/pack_submission.py"),"--track","track_3",
             "--results",str(RESULTS),"--code-commit",url,"--eval-script",str(KIT/"flash_chord/scripts/evaluate_policy.py"),
             "--skip-crosscheck","--out",str(csv_path)],
            [str(ROOT/".runtime/kit-env/bin/python"),str(KIT/"tools/preflight.py"),"--track","track_3",
             "--results",str(RESULTS),"--skip-kaggle"]]
        for command,name in zip(commands,("pack.log","preflight.log")):
            with (directory/name).open("w") as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        with (KIT/"data/track_3_sample_submission.csv").open() as stream:
            roster = [r["row_id"] for r in csv.DictReader(stream)]
        if [r["row_id"] for r in rows] != roster or len(rows) != 9120:
            raise ValueError("Packed file does not match the 9120-row roster")
        verdict = "Ready to submit." in (directory/"preflight.log").read_text()
        if not verdict:
            raise ValueError("Kit preflight did not report Ready to submit")
        manifest = {"file":str(csv_path.relative_to(ROOT)),"sha256":sha256_file(csv_path),"rows":len(rows),
                    "commands":commands,"budget":queue["budgets"]["learned"],"results_dir":str(RESULTS.relative_to(ROOT)),
                    "exports":exports,"preflight_verdict":"Ready to submit", "head":head,"code_commit_url":url,
                    "source_hashes":record["sources"],
                    "note":"Placeholder repository URL pins the requested current HEAD; v1 working-tree fixes await owner review and commit. No upload."}
        record["manifest"] = manifest
        write_json(directory/"manifest.json",manifest)
    print(directory / "manifest.json")


if __name__ == "__main__":
    pack()
