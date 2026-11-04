"""Pack an admitted pose campaign with the official kit and write its manifest."""
import csv
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import shutil
from src.paths import output_path
from src.provenance import run_fields, write_json, sha256_file, input_file
from src.runs import ROOT, runtime_flag, heavy_execution, recorded, read_json
from .loop import OUTPUT, RESULTS, identity, jobs_for, reconcile, admit_result

FEATURE = Path(__file__).parent
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"


def v2_exports(queue):
    """Combine eighteen immutable v1 files with the two accepted K1a evaluator exports."""
    original = read_json(OUTPUT / "evaluation_finished.json")
    revised = read_json(ROOT / "data/outputs/pose_insurance/v2/evaluation_finished.json")
    if original["status"] != "completed" or revised["status"] != "completed":
        raise ValueError("Both source campaigns must be completed")
    revised_queue = read_json(FEATURE / "queue_v2.json")
    if {row["identity"] for row in revised["selected"]} != {identity(request) for request in jobs_for(revised_queue, "evaluation")}:
        raise ValueError("The v2 completion marker differs from the current rebuild queue")
    if revised_queue["evaluation_episodes"] != [1, 28] or {row["episode_index"] for row in revised["selected"]} != {1, 28}:
        raise ValueError("Only episodes 1 and 28 may replace v1 exports")
    selected = [row for row in original["selected"] if row["episode_index"] not in (1, 28)] + revised["selected"]
    if len(selected) != 20 or {row["episode_index"] for row in selected} != set(queue["evaluation_episodes"]):
        raise ValueError("The combined v2 campaign must contain exactly the twenty scored episodes")
    directory = output_path(ROOT, "data/outputs/pose_insurance/results-v2")
    directory.mkdir(parents=True, exist_ok=True)
    exports = []
    for row in selected:
        _, expected = admit_result(row)
        episode = row["episode_index"]
        source = ROOT / f"data/outputs/pose_insurance/{'rebuilt-v2' if episode in (1,28) else 'results-v1'}/episode_{episode:06d}.parquet"
        if sha256_file(source) != expected:
            raise ValueError(f"Selected flat result differs from producer: {source}")
        destination = directory / source.name
        if not destination.exists() or sha256_file(destination) != expected:
            shutil.copyfile(source, destination)
        if sha256_file(destination) != expected:
            raise ValueError("The v2 copy differs from its admitted source")
        exports.append(dict(path=str(destination.relative_to(ROOT)), sha256=expected,
                            source=str(source.relative_to(ROOT)), unchanged_v1=episode not in (1, 28)))
    if set(directory.iterdir()) != {ROOT / row["path"] for row in exports}:
        raise ValueError("The v2 result directory has missing or extra files")
    return selected, directory, exports


def pack(version="v1"):
    runtime = runtime_flag(FEATURE, "pack_authorized", f"D2 {version} packing")
    heavy_execution(dict(runtime,python=runtime["kit_python"]))
    queue_path = FEATURE / "queue.json"
    queue = read_json(queue_path)
    if version == "v2":
        if read_json(ROOT / "data/outputs/pose_insurance/placement/confirmation.json")["verdict"] != "adopt":
            raise PermissionError("D2 v2 packing requires an adopt confirmation verdict")
        selected, results, exports = v2_exports(queue)
    else:
        completion = read_json(OUTPUT / "evaluation_finished.json")
        expected = {identity(request) for request in jobs_for(queue,"evaluation")}
        selected = completion["selected"]
        if completion["status"] != "completed" or {r["identity"] for r in selected} != expected:
            raise ValueError("Completion marker does not match the complete current queue")
        exports = [reconcile(row) for row in selected]
        results = RESULTS
        if set(results.iterdir()) != {ROOT/e["path"] for e in exports}:
            raise ValueError("The v1 flat directory has missing or extra files")
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    url = f"https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/{head}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    directory = output_path(ROOT,f"submission/{stamp}-d2-{version}")
    extra_inputs = ([FEATURE / "queue_v2.json", FEATURE / "placements_v2.json",
                     ROOT / "data/outputs/pose_insurance/v2/evaluation_finished.json",
                     ROOT / "data/outputs/pose_insurance/v2/active_scene_equivalence.json",
                     ROOT / "data/outputs/pose_insurance/placement/confirmation.json",
                     ROOT / "docs/chain/pose-placement-protocol.md",
                     ROOT / "scripts/reproduce/verify.sh"] if version == "v2" else [])
    record = run_fields(ROOT,"pose_insurance","pack",{"budget":queue["budgets"]["learned"],"code_commit":url},
                        parents=[ROOT/r["run"] for r in selected],
                        inputs=[queue_path,KIT/"tools/pack_submission.py",KIT/"tools/preflight.py",
                                OUTPUT/"evaluation_finished.json",*extra_inputs,*[ROOT/e["path"] for e in exports]])
    with recorded(directory,record):
        csv_path = directory / f"track3_poses_{version}.csv"
        commands = [
            [str(ROOT/"data/.runtime/kit-env/bin/python"),str(KIT/"tools/pack_submission.py"),"--track","track_3",
             "--results",str(results),"--code-commit",url,"--eval-script",str(KIT/"flash_chord/scripts/evaluate_policy.py"),
             "--skip-crosscheck","--out",str(csv_path)],
            [str(ROOT/"data/.runtime/kit-env/bin/python"),str(KIT/"tools/preflight.py"),"--track","track_3",
             "--results",str(results),"--skip-kaggle"]]
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
                    "commands":commands,"budget":queue["budgets"]["learned"],"results_dir":str(results.relative_to(ROOT)),
                    "exports":exports,"preflight_verdict":"Ready to submit", "head":head,"code_commit_url":url,
                    "source_hashes":record["sources"],
                    "note":f"Placeholder repository URL pins the requested current HEAD; {version} working-tree changes await owner review and commit. No upload."}
        if version == "v2":
            verification = ["bash", str(ROOT / "scripts/reproduce/verify.sh"), str(directory / "manifest.json")]
            manifest.update(confirmation=read_json(ROOT / "data/outputs/pose_insurance/placement/confirmation.json"),
                            first_pose_and_drift={str(row["episode_index"]):read_json(ROOT / row["run"])["checks"]
                                                  for row in selected if row["episode_index"] in (1, 28)},
                            unchanged_v1_files=sum(row["unchanged_v1"] for row in exports),
                            equivalence=read_json(ROOT / "data/outputs/pose_insurance/v2/active_scene_equivalence.json"),
                            composition=dict(K1a=[1, 28], v1=[row["episode_index"] for row in selected if row["episode_index"] not in (1, 28)]),
                            verification_command=verification,
                            verification_script=input_file(ROOT, ROOT / "scripts/reproduce/verify.sh"),
                            verification_log=str((directory / "verify.log").relative_to(ROOT)))
        record["manifest"] = manifest
        write_json(directory/"manifest.json",manifest)
        if version == "v2":
            with (directory / "verify.log").open("w") as stream:
                subprocess.run(verification, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            record["verification_summary"] = (directory / "verify.log").read_text().splitlines()[-1]
    print(directory / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    pack(parser.parse_args().version)
