"""Resume the fixed D0 queue, retain failures and publish its measured verdict."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from track3.provenance import sha256_file, write_json
from track3.runs import ROOT, now_utc
from . import handler
from ._metrics import extension_steps, scaled_thresholds
from .request import Request


def identity(queue_path, queue, task):
    """Bind resume to exact task, protocol, queue and implementation bytes."""
    paths = [queue_path, handler.FEATURE / "runtime.json", ROOT / queue["protocol"],
             *sorted(handler.FEATURE.glob("*.py")),
             *(ROOT / "track3" / name for name in ("gpu.py", "runs.py", "paths.py", "provenance.py"))]
    inputs = [handler.entry(path) for path in paths] + task["inputs"]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def existing_result(directory, campaign_identity):
    """Re-admit a finished trial, or retain an interrupted attempt without retrying."""
    path = directory / "run.json"
    record = json.loads(path.read_text())
    if record["campaign_identity"] != campaign_identity:
        raise ValueError(f"Trial identity changed: {path}")
    if record["status"] == "running":
        try:
            os.kill(record["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Trial owner {record['pid']} is still alive")
        return {**record, "status": "interrupted", "error": "Owner exited; no protocol retry",
                "gpu_seconds": max(0., time.time() - record.get("gpu_started_epoch", time.time()))}
    for name, digest in record.get("artifacts", {}).items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Trial artifact changed: {directory / name}")
    return record


def trial(request, queue, task, campaign_identity, rows):
    """Execute a missing trial once and append exactly one durable ledger row."""
    key = f"{request.arm}-seed{request.seed}"
    if key in rows:
        return
    directory = handler.OUTPUT / "runs" / key
    if (directory / "run.json").exists():
        record = existing_result(directory, campaign_identity)
    else:
        try:
            path = handler.run(request, queue, task, campaign_identity)
        except Exception as error:
            path = directory / "run.json"
            if not path.exists():
                raise
            print(f"{key}: {type(error).__name__}: {error}", flush=True)
        record = json.loads(path.read_text())
    row = {"trial": key, "arm": request.arm, "seed": request.seed, "status": record["status"],
           "run": str((directory / "run.json").relative_to(ROOT)), "request": asdict(request),
           "metrics": record.get("metrics"), "error": record.get("error"),
           "gpu_seconds": record.get("gpu_seconds", 0.),
           "wall_seconds": record.get("timing", {}).get("wall_seconds"),
           "training_seconds": record.get("training_seconds"),
           "evaluation_seconds": record.get("evaluation_seconds"),
           "peak_gpu_mib": record.get("gpu", {}).get("peak_sampled_total_device_used_mib"),
           "learner_updates": record.get("learner_updates"), "actor_adam_steps": record.get("actor_adam_steps")}
    with (handler.OUTPUT / "ledger.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    rows[key] = row
    print(json.dumps(row, allow_nan=False), flush=True)


def verdict(rows, static):
    """Apply the committed two-seed position/completion rule, without score selection."""
    limit = min(3., .5 * static["mean_position_cm"])
    passing = []
    for arm in ("L1", "L2"):
        outcomes = [rows.get(f"{arm}-seed{seed}") for seed in (0, 1)]
        if all(row and row["status"] == "completed" and row["metrics"]["completed"]
               and row["metrics"]["mean_position_cm"] <= limit for row in outcomes):
            passing.append(arm)
    return {"verdict": "GO" if passing else "NO-GO", "passing_arms": passing,
            "mean_position_limit_cm": limit,
            "rule": "Some learned arm must complete on BOTH seeds with mean position <=3 cm and <=0.5*S"}


def summary(rows, static, campaign, l2):
    """Publish compact per-trial measurements and the exact go/no-go decision."""
    decision = verdict(rows, static)
    used = sum(row["gpu_seconds"] for row in rows.values())
    report = {**decision, "campaign_identity": campaign["identity"], "comparator_S": static,
              "trials": list(rows.values()), "l2_decision": l2, "gpu_seconds": used,
              "note": "Raw world-frame position/orientation. ADD-AUC uses kit initial slot-0 alignment. "
                      "1,000 seeded surface samples of the shipped task mesh; not challenge scores."}
    write_json(handler.OUTPUT / "summary.json", report)
    lines = ["# Dynamic feasibility D0", "", f"**{decision['verdict']}**; both-seed position limit "
             f"{decision['mean_position_limit_cm']:.4f} cm. GPU lease time including compilation: {used / 60:.2f} min.",
             "", report["note"], "", "| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | P90 ori deg | ADD-AUC | Complete | Wall s | Peak MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    def metrics_row(name, seed, metrics, complete, wall, memory):
        values = [f"{metrics[key]:.6f}" for key in ("mean_position_cm", "p90_position_cm", "mean_orientation_deg", "p90_orientation_deg", "add_auc")]
        return "| " + " | ".join([name, str(seed), *values, str(complete), str(wall), str(memory)]) + " |"
    lines.append(metrics_row("S (computed)", "—", static, "reference timeline", 0, 0))
    for row in rows.values():
        if row["metrics"]:
            lines.append(metrics_row(row["arm"], row["seed"], row["metrics"], row["metrics"]["completed"],
                                     round(row["wall_seconds"], 2), row["peak_gpu_mib"]))
        else:
            lines.append(f"| {row['arm']} | {row['seed']} | — | — | — | — | — | {row['status']} | {row['wall_seconds']} | {row['peak_gpu_mib']} |")
    ranking = sorted((row for row in rows.values() if row["metrics"]),
                     key=lambda row: row["metrics"]["mean_position_cm"])
    lines += ["", "Measured ranking by mean position: " + ", ".join(row["trial"] for row in ranking),
              "", "L2 decision: " + json.dumps(l2, sort_keys=True), "", decision["rule"],
              "", "A NO-GO means this bounded policy stage did not justify a dynamic pack by the deadline; it does not establish an impossibility of manipulation."]
    for row in rows.values():
        if row["error"]:
            lines.append(f"\n{row['trial']}: {row['error']}")
    (handler.OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def run(queue_path):
    """Run Z, both L1 seeds and only a qualifying, time-sized L2 pair."""
    handler.environment()
    queue_path = Path(queue_path).resolve()
    queue = json.loads(queue_path.read_text())
    original = json.loads((ROOT / "track3/features/policy_training/budget.json").read_text())
    for arm, name in (("Z", "zero_update"), ("L1", "contact_feasibility")):
        if any(queue[arm].get(key) != value for key, value in original[name].items() if key != "note"):
            raise ValueError(f"{arm} differs from the existing feasibility budget")
    task = handler.load_task(queue)
    handler.prepare_arrays(task)
    campaign_identity, inputs = identity(queue_path, queue, task)
    handler.OUTPUT.mkdir(parents=True, exist_ok=True)
    campaign_path = handler.OUTPUT / "campaign.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign["identity"] != campaign_identity:
            raise ValueError("Campaign inputs or implementation changed; no implicit new campaign")
    else:
        campaign = {"identity": campaign_identity, "created_utc": now_utc(), "pid": os.getpid(),
                    "inputs": inputs, "queue": queue, "task_fps": task["fps"], "task_frames": len(task["reference"])}
        write_json(campaign_path, campaign)
    rows = {}
    ledger_path = handler.OUTPUT / "ledger.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            row = json.loads(line)
            if row["trial"] in rows:
                raise ValueError("Duplicate trial in ledger")
            existing_result(ROOT / row["run"].rsplit("/", 1)[0], campaign_identity)
            rows[row["trial"]] = row
    for path in sorted((handler.OUTPUT / "runs").glob("*/run.json")):
        record = existing_result(path.parent, campaign_identity)
        request = Request(**record["request"])
        trial(request, queue, task, campaign_identity, rows)
    static = handler.measure()
    l2 = {"eligible": False, "reason": "L1 has not completed"}
    for arm in ("Z", "L1"):
        for seed in queue[arm]["seeds"]:
            remaining = queue["total_gpu_seconds"] - sum(row["gpu_seconds"] for row in rows.values())
            if remaining <= queue["evaluation_reserve_seconds"]:
                l2 = {"eligible": False, "reason": "75-minute stop budget exhausted"}
                break
            request = Request(arm, seed, queue[arm], min(queue["initial_trial_seconds"], remaining))
            trial(request, queue, task, campaign_identity, rows)
            summary(rows, static, campaign, l2)
    z = rows.get("Z-seed0", {}).get("metrics")
    learned = [rows.get(f"L1-seed{seed}", {}) for seed in (0, 1)]
    valid = [row for row in learned if (row.get("metrics") or {}).get("completed")]
    if z and z["completed"] and valid:
        improvement = 1 - min(row["metrics"]["mean_position_cm"] for row in valid) / z["mean_position_cm"]
        l2 = {"eligible": improvement >= queue["L2"]["minimum_relative_improvement"],
              "relative_improvement_over_Z": improvement}
        decision_path = handler.OUTPUT / "l2_decision.json"
        if decision_path.exists():
            l2 = json.loads(decision_path.read_text())
        elif l2["eligible"]:
            remaining = queue["total_gpu_seconds"] - sum(row["gpu_seconds"] for row in rows.values())
            window = min(queue["L2"]["maximum_pair_seconds"], remaining)
            rate = max(row["training_seconds"] / row["request"]["budget"]["total_environment_steps"] for row in valid)
            reserve = max(queue["evaluation_reserve_seconds"], 2 * max(row["evaluation_seconds"] for row in valid))
            steps = extension_steps(rate, window, reserve, queue["L2"]["time_safety_factor"], 64)
            budget = dict(queue["L1"], total_environment_steps=steps)
            budget["curriculum_thresholds"] = scaled_thresholds(queue["L1"]["curriculum_thresholds"], 128000, steps, 64)
            budget["checkpoint_interval"] = max(64, round(steps / 25 / 64) * 64)
            l2.update(pair_seconds=window, per_seed_seconds=window / 2, training_seconds_per_step=rate,
                      evaluation_reserve_seconds=reserve, budget=budget)
            if steps <= 128000:
                l2.update(eligible=False, reason="No larger budget fits the remaining time")
        write_json(decision_path, l2)
        if l2["eligible"]:
            for seed in queue["L2"]["seeds"]:
                trial(Request("L2", seed, l2["budget"], l2["per_seed_seconds"]), queue, task, campaign_identity, rows)
                summary(rows, static, campaign, l2)
    else:
        l2 = {"eligible": False, "reason": "No completed Z and learned L1 comparison"}
        write_json(handler.OUTPUT / "l2_decision.json", l2)
    report = summary(rows, static, campaign, l2)
    write_json(handler.OUTPUT / "finished.json", {"identity": campaign_identity, "finished_utc": now_utc(),
                                                "verdict": report["verdict"], "gpu_seconds": report["gpu_seconds"]})
    print(f"D0 finished: {report['verdict']}; {report['gpu_seconds'] / 60:.2f} GPU-minutes", flush=True)

