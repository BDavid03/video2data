"""Freeze a throughput-sized D1 budget, then execute seeds zero and one once."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from track3.features.dynamic_feasibility import handler as d0
from track3.features.dynamic_feasibility.loop import existing_result
from track3.provenance import sha256_file, write_json
from track3.runs import ROOT, now_utc
from . import handler
from ._sizing import choose_budget


def prior_usage(queue):
    """Charge the preserved observation-shape setup failure to the same protocol caps."""
    return json.loads((ROOT / queue["prior_sizing_failure"]).read_text())["gpu_seconds"]


def append_event(event):
    """Persist one queue event before proceeding to its dependent work."""
    with (handler.OUTPUT / "ledger.jsonl").open("a") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def initialize(queue, task):
    """Bind the campaign to immutable sources and snapshot all pre-existing D0 files."""
    plan = {key: value for key, value in queue.items() if key != "chosen"}
    paths = [ROOT / plan["protocol"], handler.FEATURE / "runtime.json",
             *sorted(handler.FEATURE.rglob("*.py")), *sorted(d0.FEATURE.glob("*.py")),
             *(ROOT / "track3" / name for name in ("gpu.py", "runs.py", "paths.py", "provenance.py"))]
    inputs = [d0.entry(path) for path in paths] + task["inputs"]
    digest = hashlib.sha256(json.dumps({"plan": plan, "inputs": inputs}, sort_keys=True).encode()).hexdigest()
    path = handler.OUTPUT / "campaign.json"
    if path.exists():
        campaign = json.loads(path.read_text())
        if campaign["identity"] != digest:
            raise ValueError("D1 plan or source identity changed")
        return campaign
    d0_paths = [p for p in d0.OUTPUT.rglob("*") if p.is_file()
                and not p.is_relative_to(d0.OUTPUT / "d1") and p.name != "d1.log"]
    d0_paths += [p for p in d0.FEATURE.glob("*") if p.is_file()]
    campaign = {"identity": digest, "created_utc": now_utc(), "plan": plan, "inputs": inputs,
                "d0_files_before": [d0.entry(p) for p in sorted(d0_paths)]}
    write_json(path, campaign)
    return campaign


def operation(key, kind, identity, action, rows):
    """Resume one immutable operation or execute it once and retain its failure."""
    if key in rows:
        return rows[key]
    directory = handler.OUTPUT / ("probes" if kind == "probe" else "runs") / key
    path = directory / "run.json"
    if not path.exists():
        try:
            action()
        except Exception as error:
            if not path.exists():
                raise
            print(f"{key}: {type(error).__name__}: {error}", flush=True)
    record = existing_result(directory, identity)
    row = {"event": kind, "key": key, "status": record["status"], "run": str(path.relative_to(ROOT)),
           "gpu_seconds": record.get("gpu_seconds", 0.), "wall_seconds": record.get("timing", {}).get("wall_seconds"),
           "peak_gpu_mib": record.get("gpu", {}).get("peak_sampled_total_device_used_mib"),
           "error": record.get("error")}
    if kind == "probe":
        row.update(world_count=record["request"]["world_count"],
                   steps_per_second=record.get("throughput", {}).get("steps_per_second", 0.))
    else:
        row.update(seed=record["request"]["seed"], metrics=record.get("metrics"),
                   learner_updates=record.get("learner_updates"), actor_adam_steps=record.get("actor_adam_steps"))
    append_event(row)
    rows[key] = row
    print(json.dumps(row, allow_nan=False), flush=True)
    return row


def sizing(queue, identity, rows):
    """Spend at most five GPU minutes on the declared worlds, then eligible midpoints."""
    specification = queue["probe"]
    worlds = specification["world_counts"]
    for index, count in enumerate(worlds):
        remaining = specification["maximum_gpu_seconds"] - prior_usage(queue) - sum(row["gpu_seconds"] for row in rows.values() if row["event"] == "probe")
        if remaining <= 0:
            break
        key = f"worlds-{count}"
        allowance = remaining / (len(worlds) - index)
        operation(key, "probe", identity, lambda: handler.probe(count, allowance, identity), rows)
    for count in specification["between_if_failed"]:
        lower, upper = (256, 512) if count == 384 else (512, 1024)
        lo, hi = rows.get(f"worlds-{lower}"), rows.get(f"worlds-{upper}")
        if not lo or not hi or lo["status"] != "completed" or hi["status"] == "completed":
            continue
        remaining = specification["maximum_gpu_seconds"] - prior_usage(queue) - sum(row["gpu_seconds"] for row in rows.values() if row["event"] == "probe")
        if remaining <= 10:
            break
        operation(f"worlds-{count}", "probe", identity, lambda: handler.probe(count, remaining, identity), rows)


def freeze_choice(document, base, rows):
    """Write the chosen budget into the queue and ledger before training starts."""
    queue = document["D1"]
    path = handler.OUTPUT / "selection.json"
    if path.exists():
        selection = json.loads(path.read_text())
        if queue["chosen"] != selection["chosen"]:
            raise ValueError("Queue and frozen sizing selection differ")
    else:
        if queue["chosen"] is None:
            budget, best = choose_budget(list(rows.values()), base["L1"], queue["probe"]["maximum_peak_mib"],
                                         queue["probe"]["training_seconds_for_sizing"])
            queue["chosen"] = {"written_utc": now_utc(), "budget": budget, "selected_probe": best,
                               "rule": "Highest steps/s with peak <=13824 MiB; transitions=floor(steps/s*1350/worlds)*worlds"}
            write_json(handler.QUEUE, document)
        selection = {"event": "budget_frozen", "key": "selection", "chosen": queue["chosen"],
                     "queue_sha256": sha256_file(handler.QUEUE)}
        write_json(path, selection)
    ledger = handler.OUTPUT / "ledger.jsonl"
    if not ledger.exists() or not any(json.loads(line)["key"] == "selection" for line in ledger.read_text().splitlines()):
        append_event(selection)
    return selection


def summary(rows, selection=None, error=None):
    """Report sizing and both-seed outcomes beside the unchanged S and Z comparators."""
    baseline = json.loads((d0.OUTPUT / "summary.json").read_text())
    static = baseline["comparator_S"]
    zero = next(row for row in baseline["trials"] if row["arm"] == "Z")
    queue = json.loads(handler.QUEUE.read_text())["D1"]
    prior_probes = [json.loads(line) for line in (ROOT / queue["prior_sizing_ledger"]).read_text().splitlines()]
    trials = [row for row in rows.values() if row["event"] == "trial"]
    limit = min(3., .5 * static["mean_position_cm"])
    go = all(any(row["seed"] == seed and row["status"] == "completed" and row["metrics"]["completed"]
                 and row["metrics"]["mean_position_cm"] <= limit for row in trials) for seed in (0, 1))
    report = {"verdict": "GO" if go else "NO-GO", "position_limit_cm": limit,
              "probes": [*prior_probes, *[row for row in rows.values() if row["event"] == "probe"]], "selection": selection,
              "trials": trials, "comparator_S": static, "comparator_Z": zero,
              "gpu_seconds": prior_usage(queue) + sum(row["gpu_seconds"] for row in rows.values()), "error": error,
              "setup_correction": "Flattened simulator observation replaced by the existing WarpRLEnv.reset view before any D1 training; all failed sizing time retained and charged"}
    for row in trials:
        if row["metrics"]:
            row["improvement_over_Z"] = 1 - row["metrics"]["mean_position_cm"] / zero["metrics"]["mean_position_cm"]
    write_json(handler.OUTPUT / "summary.json", report)
    lines = ["# Dynamic scaling D1", "", f"**{report['verdict']}**; total GPU time {report['gpu_seconds']/60:.2f} minutes.",
             "", "| Probe worlds | Steps/s | Peak device MiB | Status |", "|---:|---:|---:|---|"]
    for row in report["probes"]:
        lines.append(f"| {row['world_count']} | {row['steps_per_second']:.3f} | {row['peak_gpu_mib']} | {row['status']} |")
    if selection:
        chosen = selection["chosen"]
        lines += ["", f"Budget frozen at **{chosen['written_utc']}**: "
                  f"{chosen['budget']['world_count']} worlds, {chosen['budget']['total_environment_steps']:,} transitions."]
    lines += ["", "Position/orientation are raw world-frame errors. ADD-AUC uses D0's seeded task surface and kit initial alignment.",
              "", "| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | P90 ori deg | ADD-AUC | Complete | Wall s | Peak MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    measurements = [("S, computed", "—", static, "reference", "—", "—"),
                    ("Z", 0, zero["metrics"], True, zero["wall_seconds"], zero["peak_gpu_mib"])]
    for row in trials:
        if row["metrics"]:
            measurements.append(("D1", row["seed"], row["metrics"], row["metrics"]["completed"], row["wall_seconds"], row["peak_gpu_mib"]))
        else:
            lines.append(f"| D1 | {row['seed']} | — | — | — | — | — | {row['status']} | {row['wall_seconds']} | {row['peak_gpu_mib']} |")
    for arm, seed, metrics, complete, wall, memory in measurements:
        values = [f"{metrics[key]:.6f}" for key in ("mean_position_cm", "p90_position_cm", "mean_orientation_deg", "p90_orientation_deg", "add_auc")]
        lines.append("| " + " | ".join([arm, str(seed), *values, str(complete), str(wall), str(memory)]) + " |")
    for row in trials:
        if row["metrics"]:
            lines.append(f"\nSeed {row['seed']} improvement over Z: {100*row['improvement_over_Z']:.4f}%.")
        if row["error"]:
            lines.append(f"\nSeed {row['seed']}: {row['error']}")
    lines += ["", f"GO requires BOTH seeds to complete with mean position <= {limit:.4f} cm."]
    if error:
        lines.append(f"\n{error}")
    (handler.OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def run():
    """Size once without tracking scores, freeze the queue, and run both seeds in order."""
    handler.environment("probe_authorized")
    document = json.loads(handler.QUEUE.read_text())
    queue = document["D1"]
    base = json.loads((ROOT / queue["base_queue"]).read_text())
    task = d0.load_task(base)
    handler.OUTPUT.mkdir(parents=True, exist_ok=True)
    campaign = initialize(queue, task)
    rows = {}
    ledger = handler.OUTPUT / "ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row["event"] != "budget_frozen":
                if row["key"] in rows:
                    raise ValueError("Duplicate D1 operation in ledger")
                existing_result((ROOT / row["run"]).parent, campaign["identity"])
                rows[row["key"]] = row
    selection = None
    error = None
    try:
        if queue["chosen"] is None:
            sizing(queue, campaign["identity"], rows)
        selection = freeze_choice(document, base, rows)
        summary(rows, selection)
        for seed in queue["seeds"]:
            remaining = queue["total_gpu_seconds"] - prior_usage(queue) - sum(row["gpu_seconds"] for row in rows.values())
            if remaining <= queue["evaluation_reserve_seconds"]:
                error = "65-minute stop budget exhausted before the next seed"
                break
            request = handler.Request(seed, selection["chosen"]["budget"], min(queue["maximum_seed_seconds"], remaining))
            operation(f"seed{seed}", "trial", campaign["identity"],
                      lambda: handler.run(request, task, campaign["identity"]), rows)
            summary(rows, selection)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        report = summary(rows, selection, error)
        write_json(handler.OUTPUT / "finished.json", {"finished_utc": now_utc(), "identity": campaign["identity"],
                                                    "verdict": report["verdict"], "error": error,
                                                    "gpu_seconds": report["gpu_seconds"]})
        print(f"D1 finished: {report['verdict']}; {report['gpu_seconds']/60:.2f} GPU-minutes", flush=True)
