"""Run D0 and D1 queues through shared trial admission and ledger persistence."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from track3.provenance import sha256_file, write_json
from track3.runs import ROOT, now_utc
from . import handler
from ._metrics import extension_steps, scaled_thresholds, verdict
from ._render import d0_summary, d1_summary
from ._sizing import choose_budget
from .request import Request


def scaling_plan(queue):
    """Retain retired probe values in identity only, in their original plan shape."""
    plan = {key: value for key, value in queue.items() if key not in ("chosen", "history")}
    plan["probe"] = {**queue.get("history", {}).get("probe", {}), **queue["probe"]}
    return plan


def identity(queue_path, queue, task):
    """Bind resume to exact task, protocol, queue and implementation bytes."""
    scaling = "D1" in queue or "base_queue" in queue
    paths = [handler.FEATURE / "runtime.json", ROOT / queue["protocol"],
             *sorted(handler.FEATURE.glob("*.py")),
             *(ROOT / "track3" / name for name in ("gpu.py", "runs.py", "paths.py", "provenance.py"))]
    if scaling:
        paths += [ROOT / queue["base_queue"], *(ROOT / path for path in queue["training_runs"])]
    else:
        paths.insert(0, queue_path)
    inputs = [handler.entry(path) for path in paths] + task["inputs"]
    payload = ({"plan": scaling_plan(queue), "inputs": inputs}
               if scaling else inputs)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), inputs


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


def trial(request, queue, task, campaign_identity, rows, output=None):
    """Execute or re-admit one trial and append its historical campaign row shape."""
    output = handler.OUTPUT if output is None else output
    scaling = request.arm == "D1"
    key = f"seed{request.seed}" if scaling else f"{request.arm}-seed{request.seed}"
    if key in rows:
        return
    directory = output / "runs" / key
    path = directory / "run.json"
    if not path.exists():
        try:
            handler.run(request, queue, task, campaign_identity)
        except Exception as error:
            if not path.exists():
                raise
            print(f"{key}: {type(error).__name__}: {error}", flush=True)
    record = existing_result(directory, campaign_identity)
    row = {"status": record["status"], "seed": request.seed,
           "run": str(path.relative_to(ROOT)), "metrics": record.get("metrics"),
           "error": record.get("error"), "gpu_seconds": record.get("gpu_seconds", 0.),
           "wall_seconds": record.get("timing", {}).get("wall_seconds"),
           "peak_gpu_mib": record.get("gpu", {}).get("peak_sampled_total_device_used_mib"),
           "learner_updates": record.get("learner_updates"), "actor_adam_steps": record.get("actor_adam_steps")}
    if scaling:
        row.update(event="trial", key=key)
    else:
        row.update(trial=key, arm=request.arm, request=asdict(request),
                   training_seconds=record.get("training_seconds"),
                   evaluation_seconds=record.get("evaluation_seconds"))
    fields = ("event key status run gpu_seconds wall_seconds peak_gpu_mib error seed metrics learner_updates actor_adam_steps"
              if scaling else "trial arm seed status run request metrics error gpu_seconds wall_seconds training_seconds "
              "evaluation_seconds peak_gpu_mib learner_updates actor_adam_steps")
    row = {name: row[name] for name in fields.split()}
    with (output / "ledger.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    rows[key] = row
    print(json.dumps(row, allow_nan=False), flush=True)


def summary(rows, static, campaign, l2, output=None):
    """Persist a D0 report without changing its historical serialization."""
    report, markdown = d0_summary(rows, static, campaign, l2)
    output = handler.OUTPUT if output is None else output
    write_json(output / "summary.json", report)
    (output / "summary.md").write_text(markdown)
    return report


def run_d0(queue_path):
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


def scaling_summary(rows, selection=None, error=None, output=None, queue_path=None):
    """Render saved D1 events with the unchanged D0 comparators and failed probes."""
    output = handler.SCALING_OUTPUT if output is None else output
    queue = json.loads(Path(queue_path or handler.SCALING_QUEUE).read_text())["D1"]
    baseline = json.loads((handler.OUTPUT / "summary.json").read_text())
    prior = [json.loads(line) for line in (ROOT / queue["prior_sizing_ledger"]).read_text().splitlines()]
    prior_seconds = json.loads((ROOT / queue["prior_sizing_failure"]).read_text())["gpu_seconds"]
    report, markdown = d1_summary(rows, baseline, prior, prior_seconds, selection, error)
    write_json(output / "summary.json", report)
    (output / "summary.md").write_text(markdown)
    return report


def run_scaling(queue_path):
    """Run D1 through the shared trial loop, retaining any historical frozen choice."""
    handler.scaling_runtime("run_authorized")
    handler.environment()
    queue_path = Path(queue_path).resolve()
    document = json.loads(queue_path.read_text())
    queue = document["D1"]
    base = json.loads((ROOT / queue["base_queue"]).read_text())
    task = handler.load_task(base)
    handler.prepare_arrays(task)
    output = handler.SCALING_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    plan = scaling_plan(queue)
    digest, inputs = identity(queue_path, queue, task)
    path = output / "campaign.json"
    if path.exists():
        campaign = json.loads(path.read_text())
        if campaign["identity"] != digest:
            raise ValueError("D1 plan or source identity changed; historical records are render-only")
    else:
        campaign = dict(identity=digest, created_utc=now_utc(), plan=plan, inputs=inputs)
        write_json(path, campaign)
    rows = {}
    ledger = output / "ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row["event"] == "budget_frozen":
                continue
            if row["key"] in rows:
                raise ValueError("Duplicate D1 operation in ledger")
            existing_result((ROOT / row["run"]).parent, digest)
            rows[row["key"]] = row
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if queue["chosen"] != selection["chosen"]:
            raise ValueError("Queue and frozen sizing selection differ")
    else:
        if queue["chosen"] is None:
            measurements = [json.loads((ROOT / path).read_text()) for path in queue["training_runs"]]
            budget, best = choose_budget(measurements, base["L1"], queue["probe"]["maximum_peak_mib"],
                                         queue["probe"]["training_seconds_for_sizing"])
            queue["chosen"] = dict(written_utc=now_utc(), budget=budget, selected_probe=best,
                                   rule="Measured training seconds/transition * 1.1; collection-grid floor")
            write_json(queue_path, document)
        selection = dict(event="budget_frozen", key="selection", chosen=queue["chosen"],
                         queue_sha256=sha256_file(queue_path))
        write_json(selection_path, selection)
    if not ledger.exists() or not any(json.loads(line)["key"] == selection["key"]
                                     for line in ledger.read_text().splitlines()):
        with ledger.open("a") as stream:
            stream.write(json.dumps(selection, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    error = None
    try:
        for seed in queue["seeds"]:
            prior_seconds = json.loads((ROOT / queue["prior_sizing_failure"]).read_text())["gpu_seconds"]
            remaining = queue["total_gpu_seconds"] - prior_seconds - sum(r["gpu_seconds"] for r in rows.values())
            if remaining <= queue["evaluation_reserve_seconds"]:
                error = "65-minute stop budget exhausted before the next seed"
                break
            request = Request("D1", seed, selection["chosen"]["budget"], min(queue["maximum_seed_seconds"], remaining))
            trial(request, {**base, **queue, "queue_path": str(queue_path.relative_to(ROOT))}, task, digest, rows, output)
            scaling_summary(rows, selection, queue_path=queue_path)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        report = scaling_summary(rows, selection, error, queue_path=queue_path)
        write_json(output / "finished.json", dict(finished_utc=now_utc(), identity=digest,
                   verdict=report["verdict"], error=error, gpu_seconds=report["gpu_seconds"]))
        print(f"D1 finished: {report['verdict']}; {report['gpu_seconds']/60:.2f} GPU-minutes", flush=True)


def rerender(queue_path, output):
    """Read historical ledgers without re-admitting obsolete source hashes or executing trials."""
    from track3.paths import output_path
    output = output_path(ROOT, output)
    if output.is_relative_to(handler.OUTPUT):
        raise ValueError("Historical campaign directories are read-only during rendering")
    output.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(queue_path).read_text())
    source = handler.SCALING_OUTPUT if "D1" in document else handler.OUTPUT
    events = [json.loads(line) for line in (source / "ledger.jsonl").read_text().splitlines()]
    if "D1" in document:
        rows = {row["key"]: row for row in events if row["event"] != "budget_frozen"}
        selection = json.loads((source / "selection.json").read_text())
        error = json.loads((source / "finished.json").read_text()).get("error")
        return scaling_summary(rows, selection, error, output, queue_path)
    rows = {row["trial"]: row for row in events}
    return summary(rows, json.loads((source / "comparator_S.json").read_text()),
                   json.loads((source / "campaign.json").read_text()),
                   json.loads((source / "l2_decision.json").read_text()), output)


def run(queue_path):
    """Dispatch either campaign queue to the shared execution and ledger machinery."""
    document = json.loads(Path(queue_path).read_text())
    return run_scaling(queue_path) if "D1" in document else run_d0(queue_path)
