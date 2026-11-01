"""Resumable v1 queue with checked producer-to-flat-export reconciliation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from track3.paths import output_path
from track3.provenance import write_json, read_run, sha256_file, input_file, run_fields
from track3.runs import (ROOT, read_json, runtime_flag, authorize_episode, heavy_execution,
                         recorded, new_run_id)
from .request import Request

FEATURE = Path(__file__).parent
OUTPUT = ROOT / "outputs/pose_insurance/v1"
RESULTS = ROOT / "outputs/pose_insurance/results-v1"


def select_version(version):
    """Keep the accepted K1a campaign separate from immutable v1 outputs."""
    global OUTPUT, RESULTS
    OUTPUT = ROOT / f"outputs/pose_insurance/{version}"
    RESULTS = ROOT / f"outputs/pose_insurance/{'rebuilt-v2' if version == 'v2' else 'results-v1'}"


def snapshot(path):
    if path.exists():
        backup = ROOT / ".runtime/tmp/ctx03/backups" / f"{time.time_ns()}_{path.name}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, backup)


def identity(request):
    paths = [*sorted(FEATURE.glob("*.py")), FEATURE / "runtime.json", FEATURE / "placements.json",
             FEATURE / "diagnostic_sampling.json", ROOT / "outputs/pose_insurance/runtime_inventory.json",
             ROOT / "track3/features/mesh_extents/assignments.json"]
    if request.source_split == "evaluation" and request.episode_index in (1, 28):
        paths.extend([FEATURE / "placements_v2.json", ROOT / "outputs/pose_insurance/placement/confirmation.json"])
    sources = {str(p.relative_to(ROOT)):sha256_file(p) for p in paths}
    return hashlib.sha256(json.dumps({"request":asdict(request),"sources":sources},sort_keys=True).encode()).hexdigest()


def jobs_for(queue, phase):
    if phase == "budgets":
        return [Request(episode_index=10, budget=b) for b in queue["budget_pilots"]]
    if phase == "tuning":
        return [Request(source_split="public",episode_index=e,budget=queue["budgets"]["learned"]) for e in (39,42)]
    return [Request(episode_index=e,budget=queue["budgets"]["learned"]) for e in queue["evaluation_episodes"]]


def admit_result(row):
    """Admit the selected run, its task record, checkpoint and official trajectory."""
    from .handler import inspect_export
    path = ROOT / row["run"]
    record = read_run(ROOT, path, kind="pose_insurance.run")
    task_entry = next(e for e in record["inputs"] if any(part in e["path"] for part in ("/build-v1/", "/build-v2/"))
                      and e["path"].endswith("run.json"))
    if input_file(ROOT, task_entry["path"]) != task_entry:
        raise ValueError("Selected task producer changed")
    task = read_run(ROOT, task_entry["path"], kind="pose_insurance.build")
    trajectory = path.parent / "trajectory.parquet"
    expected = record["artifacts"]["trajectory.parquet"]
    if sha256_file(trajectory) != expected or sha256_file(ROOT / record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("Selected trajectory or checkpoint changed")
    inspect_export(trajectory, task)
    return trajectory, expected


def reconcile(row):
    """Restore a missing or stale flat result from the checked selected producer."""
    source, expected = admit_result(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / f"episode_{row['episode_index']:06d}.parquet"
    if not destination.exists() or sha256_file(destination) != expected:
        snapshot(destination)
        temporary = destination.with_suffix(".copy.tmp")
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != expected:
                raise ValueError("Export changed during copying")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path":str(destination.relative_to(ROOT)),"sha256":expected}


def summarize(rows, jobs, phase):
    selected = {}
    for request in jobs:
        key = identity(request)
        selected[key] = next((r for r in reversed(rows) if r["identity"] == key),
                             dict(identity=key, **asdict(request), status="pending", seconds=0))
    lines = [f"# Pose insurance {OUTPUT.name}: {phase}", "", "| Split | Episode | Steps/worlds/min | Status | Updates/actor | Drift cm/deg | Minutes | Peak MiB | Identity uncertainty / assignment note |", "|---|---:|---|---|---|---|---:|---:|---|"]
    for r in selected.values():
        c = r.get("checks", {})
        motion = f"{c['drift_cm']:.6f}/{c['drift_deg']:.6f}" if c else "—"
        budget = r["budget"]
        uncertainty = "; ".join(f"{s['name']}: {s['identity_confidence']}" for s in r.get("scene",[]))
        notes = "; ".join(dict.fromkeys(s.get("assignment_note","") for s in r.get("scene",[])))
        lines.append(f"| {r['source_split']} | {r['episode_index']} | {budget['environment_steps']}/{budget['worlds']}/{budget['replay_minimum']} | {r['status']} | {r.get('optimizer_updates','—')}/{r.get('actor_adam_steps','—')} | {motion} | {r['seconds']/60:.2f} | {r.get('peak_mib','—')} | {uncertainty}; {notes} |")
    path = OUTPUT / ("summary.md" if phase == "evaluation" else f"{phase}_summary.md")
    snapshot(path)
    path.write_text("\n".join(lines)+"\n")
    if phase == "evaluation":
        checkpoints = {str(r["episode_index"]):str(ROOT/r["checkpoint"]) for r in selected.values() if r["status"] == "completed"}
        snapshot(OUTPUT / "checkpoints.json")
        write_json(OUTPUT / "checkpoints.json", checkpoints)
    return selected


def run_queue(phase, queue_path):
    from .handler import run
    queue = read_json(queue_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = OUTPUT / "ledger.jsonl"
    rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    jobs = jobs_for(queue, phase)
    summarize(rows, jobs, phase)
    selected = []
    for request in jobs:
        key = identity(request)
        prior = next((r for r in reversed(rows) if r["identity"] == key and r["status"] == "completed"),None)
        started = time.perf_counter()
        row = dict(identity=key, **asdict(request))
        print(f"{'resume' if prior else 'start'} {phase} {request.source_split}:{request.episode_index} {key[:12]}",flush=True)
        try:
            if prior:
                admit_result(prior)
                row = dict(prior)
            else:
                path = run(request)
                result = read_json(path)
                row.update(status="completed",run=str(path.relative_to(ROOT)),checkpoint=result["checkpoint"],
                           optimizer_updates=result["optimizer_updates"],actor_adam_steps=result["actor_adam_steps"],
                           checks=result["checks"],scene=result["scene"],
                           peak_mib=result["gpu"]["peak_sampled_total_device_used_mib"])
            if phase == "evaluation":
                row["export"] = reconcile(row)
            if phase == "tuning":
                diagnostic = row.get("diagnostic_run")
                if diagnostic:
                    record = read_run(ROOT, diagnostic, kind="pose_insurance.diagnostic")
                    metrics = record["metrics"]
                else:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
                    command = [str(ROOT / ".runtime/linux-cuda/bin/python"),"-m","track3.features.pose_insurance.diagnostic",
                               "--episode",str(request.episode_index),"--run",row["run"]]
                    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
                    diagnostic = result.stdout.strip().splitlines()[-1]
                    record = read_run(ROOT, diagnostic, kind="pose_insurance.diagnostic")
                    metrics = record["metrics"]
                row.update(diagnostic_run=diagnostic, pose_metrics=metrics)
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
            if hasattr(error,"feature_run_path"):
                row["run"] = error.feature_run_path
                failed = read_json(row["run"])
                for field in ("optimizer_updates","actor_adam_steps","gpu","scene"):
                    if field in failed:
                        row[field] = failed[field]
        if not prior:
            row["seconds"] = time.perf_counter()-started
        with ledger.open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        rows.append(row)
        selected.append(row)
        summarize(rows,jobs,phase)
        print(json.dumps(row),flush=True)
        if row["status"] != "completed" and phase != "budgets":
            raise RuntimeError(f"Stopped at first failed episode: {row['error']}")
    exports = []
    if phase == "evaluation":
        exports = [reconcile(r) for r in selected]
        expected = {ROOT / e["path"] for e in exports}
        if set(RESULTS.iterdir()) != expected:
            raise ValueError("Flat results do not match the complete selected queue")
    marker = OUTPUT / f"{phase}_finished.json"
    snapshot(marker)
    write_json(marker,{"status":"completed", "queue_sha256":sha256_file(ROOT/queue_path),
                       "selected":selected,"exports":exports,"finished_unix":time.time()})


def build_tasks(queue_path):
    from .scene import build
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path = OUTPUT / "tasks.json"
    tasks = read_json(path) if path.exists() else {}
    for request in jobs_for(read_json(queue_path),"evaluation"):
        key = identity(request)
        prior = tasks.get(str(request.episode_index))
        if prior and prior["identity"] == key:
            read_run(ROOT,prior["run"],kind="pose_insurance.build")
            continue
        result = build(request)
        record = read_json(result)
        tasks[str(request.episode_index)] = dict(identity=key,run=str(result.relative_to(ROOT)),scene=record["scene"])
        snapshot(path)
        write_json(path,tasks)
    print(f"Built {len(tasks)} v1 tasks: {path}")


def choose_budget(queue_path):
    queue = read_json(queue_path)
    rows = [json.loads(s) for s in (OUTPUT/"ledger.jsonl").read_text().splitlines()]
    candidates = []
    for request in jobs_for(queue,"budgets"):
        key = identity(request)
        row = next((r for r in reversed(rows) if r["identity"] == key and r["status"] == "completed"),None)
        if row and row["seconds"] <= 300:
            candidates.append(row)
    if not candidates:
        raise RuntimeError("No pilot satisfies the drift and five-minute limits; add a smaller pilot to the queue")
    best = max(candidates,key=lambda r:r["optimizer_updates"])
    queue["budgets"]["learned"] = best["budget"]
    snapshot(ROOT/queue_path)
    write_json(ROOT/queue_path,queue)
    write_json(OUTPUT/"budget_selection.json",{"selected":best,"rule":"largest measured update count among passing pilots taking <=300 seconds"})
    print(json.dumps(best))


def reuse_v2(queue_path):
    """Admit existing passing exports after exact active-scene and budget equivalence."""
    runtime = runtime_flag(FEATURE, "run_authorized", "Reuse evaluated K1a scenes")
    queue = read_json(queue_path)
    jobs = jobs_for(queue, "evaluation")
    for request in jobs:
        authorize_episode(runtime, request, "reuse evaluated K1a scene")
    heavy_execution(runtime)
    proof_path = OUTPUT / "active_scene_equivalence.json"
    proof = read_json(proof_path)
    for entry in proof["inputs"]:
        if input_file(ROOT, entry["path"]) != entry:
            raise ValueError("Active scene proof has a changed input")
    if proof["unchanged_v1_scenes"] != 18 or proof["matched_k1a_scenes"] != 2:
        raise ValueError("The active scene proof must cover eighteen v1 and two K1a scenes")
    record = run_fields(ROOT, "pose_insurance", "reuse", {"queue": str(queue_path)},
                        parents=list(queue["reuse_runs"].values()), inputs=[queue_path, proof_path])
    with recorded(OUTPUT / "reuse" / new_run_id(), record):
        selected = []
        for request in jobs:
            path = queue["reuse_runs"][str(request.episode_index)]
            result = read_run(ROOT, path, kind="pose_insurance.run")
            if result["request"] != asdict(request) or result["request"]["budget"] != queue["budgets"]["learned"]:
                raise ValueError("Reused run differs from the queued request or budget")
            evidence = next(row for row in proof["scenes"] if row["episode"] == request.episode_index)
            if evidence["reference_kind"] != "K1a" or evidence["policy_run"] != input_file(ROOT, path):
                raise ValueError("Scene proof does not identify the reused K1a policy run")
            for entry in (evidence["reference"], evidence["recomputed"]):
                if input_file(ROOT, entry["path"]) != entry:
                    raise ValueError("Proven scene bytes changed")
            row = dict(identity=identity(request), **asdict(request), status="completed", run=path,
                       reused=True, checkpoint=result["checkpoint"], scene=result["scene"],
                       checks=result["checks"], optimizer_updates=result["optimizer_updates"],
                       actor_adam_steps=result["actor_adam_steps"], seconds=0,
                       peak_mib=result["gpu"]["peak_sampled_total_device_used_mib"])
            row["export"] = reconcile(row)
            selected.append(row)
        write_json(OUTPUT / "evaluation_finished.json",
                   dict(status="completed", queue_sha256=sha256_file(ROOT / queue_path), selected=selected,
                        exports=[row["export"] for row in selected], reused=2, trained=0,
                        equivalence=input_file(ROOT, proof_path)))
        write_json(OUTPUT / "reuse_selection.json", selected)
        record["selected"] = selected
    print(json.dumps(dict(reused=2, trained=0, marker=str(OUTPUT / "evaluation_finished.json"))))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=["budgets","choose_budget","tasks","evaluation","tuning","reuse"], required=True)
    parser.add_argument("--queue",default="track3/features/pose_insurance/queue.json")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    if args.version == "v2":
        select_version(args.version)
    if args.version == "v2" and read_json(args.queue)["evaluation_episodes"] != [1, 28]:
        raise ValueError("The v2 queue must contain only the accepted episodes 1 and 28")
    if args.version == "v2" and read_json(FEATURE / "placements_v2.json").get("enabled", True) is not True:
        raise PermissionError("The v2 candidate remains disabled")
    if args.phase == "reuse":
        if args.version != "v2":
            raise ValueError("Export reuse is scoped to v2")
        reuse_v2(args.queue)
    elif args.phase == "tasks":
        build_tasks(args.queue)
    elif args.phase == "choose_budget":
        choose_budget(args.queue)
    else:
        run_queue(args.phase,args.queue)


if __name__ == "__main__":
    main()
