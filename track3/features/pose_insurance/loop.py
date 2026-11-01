"""Resumable sequential pilot/campaign runner with ledger and a compact summary."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from track3.provenance import write_json, read_run, sha256_file
from track3.runs import ROOT, read_json
from .request import Request

OUTPUT = ROOT / "outputs/pose_insurance"


def snapshot(path):
    """Preserve a prior mutable report before replacement."""
    if path.exists():
        backup = ROOT / ".runtime/tmp/ctx03/backups" / f"{time.time_ns()}_{path.name}"
        backup.parent.mkdir(parents=True,exist_ok=True)
        backup.write_bytes(path.read_bytes())


def identity(request):
    sources = {p.name:sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    sources["runtime.json"] = sha256_file(Path(__file__).parent / "runtime.json")
    sources["runtime_inventory.json"] = sha256_file(OUTPUT / "runtime_inventory.json")
    sources["assignments.json"] = sha256_file(ROOT / "track3/features/mesh_extents/assignments.json")
    payload = {"request":asdict(request),"sources":sources}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def summarize(rows):
    latest = {}
    for row in rows:
        key = (row["source_split"], row["episode_index"])
        if not row["contract_only"] or key not in latest:
            latest[key] = row
    lines = ["# Pose insurance runs", "", "Ranked by passed status, then drift. Time includes task building and compilation.", "",
             "| Split | Episode | Contract | Status | Updates | Drift cm / deg | Coverage keys | Minutes | Peak MiB | Run |",
             "|---|---:|---|---|---:|---|---|---:|---:|---|"]
    for r in sorted(latest.values(), key=lambda r:(r["status"]!="completed",r.get("checks",{}).get("drift_cm",1e9),r["episode_index"])):
        c = r.get("checks",{})
        motion = f"{c['drift_cm']:.6f} / {c['drift_deg']:.6f}" if c else "—"
        coverage = f"{c.get('covered_keys',0)}/{c.get('roster_keys',0)}"
        lines.append(
            f"| {r['source_split']} | {r['episode_index']} | {r['contract_only']} | {r['status']} | "
            f"{r.get('optimizer_updates','—')} | {motion} | {coverage} | {r['seconds']/60:.2f} | "
            f"{r.get('peak_mib','—')} | {r.get('run',r.get('error',''))} |")
    snapshot(OUTPUT / "summary.md")
    (OUTPUT / "summary.md").write_text("\n".join(lines)+"\n")
    checkpoints = {}
    for r in latest.values():
        if r["status"] == "completed" and r["source_split"] == "evaluation" and not r["contract_only"]:
            checkpoints[str(r["episode_index"])] = str(ROOT / r["checkpoint"])
    snapshot(OUTPUT / "checkpoints.json")
    write_json(OUTPUT / "checkpoints.json", checkpoints)


def run_queue(phase, queue_path):
    from .handler import run
    import os
    import subprocess
    queue = read_json(queue_path)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    ledger = OUTPUT / "ledger.jsonl"
    rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    jobs = queue["pilots"] if phase == "pilots" else [dict(source_split="evaluation",episode_index=e,budget=queue["budgets"]["learned"]) for e in queue["evaluation_episodes"]]
    for job in jobs:
        job = dict(job)
        job["budget"] = queue["budgets"][job.pop("budget_name")] if "budget_name" in job else job["budget"]
        request = Request(**job)
        key = identity(request)
        prior = next((r for r in reversed(rows) if r["identity"] == key and r["status"] == "completed"),None)
        if prior:
            parent = read_run(ROOT, prior["run"], kind="pose_insurance.run")
            trajectory = ROOT / prior["run"]
            trajectory = trajectory.parent / "trajectory.parquet"
            if sha256_file(trajectory) != parent["artifacts"]["trajectory.parquet"]:
                raise ValueError("Completed export changed")
            print(f"skip {request.source_split}:{request.episode_index} {key[:12]}",flush=True)
            continue
        started = time.perf_counter()
        row = dict(identity=key, **asdict(request))
        print(f"start {request.source_split}:{request.episode_index} {key[:12]}",flush=True)
        try:
            path = run(request)
            result = read_json(path)
            row.update(status="completed",run=str(path.relative_to(ROOT)),checkpoint=result["checkpoint"],
                       optimizer_updates=result["optimizer_updates"],actor_adam_steps=result["actor_adam_steps"],checks=result["checks"],
                       peak_mib=result["gpu"]["peak_sampled_total_device_used_mib"])
            if request.source_split == "public":
                diagnostic_path = OUTPUT / f"diagnostic_{request.episode_index}_{key[:12]}.json"
                subprocess.run([str(ROOT / ".runtime/linux-cuda/bin/python"), "-m",
                                "track3.features.pose_insurance.diagnostic", "--episode",str(request.episode_index),
                                "--trajectory",str(path.parent / "trajectory.parquet"),"--out",str(diagnostic_path)],
                               cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),check=True)
                metrics = read_json(diagnostic_path)
                # Diagnostic labels never enter the policy producer's run or flat export directory.
                write_json(OUTPUT / f"diagnostic_{request.episode_index}_{key[:12]}.json",metrics)
                row["pose_metrics"] = metrics
            elif not request.contract_only:
                destination = OUTPUT / "results" / f"episode_{request.episode_index:06d}.parquet"
                destination.parent.mkdir(exist_ok=True)
                if destination.exists() and destination.read_bytes() != (path.parent / "trajectory.parquet").read_bytes():
                    backup = ROOT / ".runtime/tmp/ctx03/backups" / f"{time.time_ns()}_{destination.name}"
                    backup.parent.mkdir(parents=True,exist_ok=True)
                    backup.write_bytes(destination.read_bytes())
                destination.write_bytes((path.parent / "trajectory.parquet").read_bytes())
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}")
            if hasattr(error,"feature_run_path"):
                row["run"] = error.feature_run_path
        row["seconds"] = time.perf_counter()-started
        with ledger.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        rows.append(row)
        summarize(rows)
        print(json.dumps(row),flush=True)
        if row["status"] != "completed":
            raise RuntimeError(f"Stopped at first failed job: {row['error']}")
    snapshot(OUTPUT / f"{phase}_finished.json")
    write_json(OUTPUT / f"{phase}_finished.json",{"status":"completed","queue_sha256":sha256_file(ROOT / queue_path),
                                                        "finished_unix":time.time()})


def build_tasks(queue_path):
    """Build every scored task on CPU, resuming the immutable task registry."""
    from .scene import build
    queue = read_json(queue_path)
    path = OUTPUT / "tasks.json"
    tasks = read_json(path) if path.exists() else {}
    for episode in queue["evaluation_episodes"]:
        request = Request(episode_index=episode,budget=queue["budgets"]["learned"])
        key = identity(request)
        if str(episode) in tasks and tasks[str(episode)]["identity"] == key:
            read_run(ROOT,tasks[str(episode)]["run"],kind="pose_insurance.build")
            continue
        result = build(request)
        record = read_json(result)
        row = {"identity":key,"run":str(result.relative_to(ROOT)),"scene":record["scene"]}
        tasks[str(episode)] = row
        with (OUTPUT / "task_ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(episode_index=episode,**row))+"\n")
        snapshot(path)
        write_json(path,tasks)
    print(f"Built {len(tasks)} scored tasks: {path}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase",choices=["pilots","evaluation","tasks"],default="pilots")
    parser.add_argument("--queue",default="track3/features/pose_insurance/queue.json")
    args = parser.parse_args()
    if args.phase == "tasks":
        build_tasks(args.queue)
    else:
        run_queue(args.phase,args.queue)


if __name__ == "__main__":
    main()
