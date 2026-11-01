"""Resume the fixed CPU placement queue and write a ranked, gated report."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import time
from track3.paths import output_path
from track3.provenance import input_file, read_run, sha256_file, write_json
from track3.runs import (ROOT, read_json, producer, artifact, source_hashes, runtime_flag,
                         authorize_episode, heavy_execution)
from .placement_diagnostic import (FEATURE, PROTOCOL, OUTPUT, ROSTER, PlacementRequest,
                                   diagnose, label_paths, KIT, D1)


def identity(request):
    """Hash trial settings, code, protocol and exact upstream input identities."""
    entry = read_json(FEATURE / "placements_train.json")["episodes"][str(request.episode_index)]
    measure = producer(entry["measure_run"], "mesh_extents.measure", request)
    depth = producer(measure[1]["request"]["depth_run"], "depth.generate", request)
    dataset = next(ROOT.glob("datasets--*/snapshots/*/track_3/public"))
    meshes = []
    for name in ROSTER[request.episode_index]:
        path = dataset / f"mesh/{name}/{name}.glb"
        meshes.append(path if path.exists() else path.with_name(f"{name}_visual.glb"))
    paths = [PROTOCOL, FEATURE / "placements_train.json", FEATURE / "runtime.json",
             FEATURE / "diagnostic_sampling.json", FEATURE.parent / "mesh_extents/assignments.json",
             ROOT / D1, measure[0], depth[0], KIT / "metric_code/track_3/AUC.py",
             ROOT / "scripts/track3/static_pose_baseline.py", *meshes,
             *label_paths(request.episode_index, "K2" in request.candidate)]
    payload = dict(request=asdict(request), inputs=[input_file(ROOT, path) for path in paths],
                   sources=source_hashes(FEATURE))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resume_result(row):
    """Re-admit a completed trial and every hashed input before resuming."""
    record = read_run(ROOT, row["run"], kind="pose_insurance.placement_diagnostic")
    for entry in record["inputs"]:
        if input_file(ROOT, entry["path"]) != entry:
            raise ValueError(f"Placement input changed: {entry['path']}")
    path = artifact((ROOT / row["run"], record), "metrics.json", {"inputs": []})
    return read_json(path)


def run_queue(queue_path):
    """Execute all missing identities through the diagnostic handler, then summarize."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    runtime = runtime_flag(FEATURE, "placement_diagnostic_authorized", "Placement queue")
    queue = read_json(queue_path)
    if queue["episodes"] != list(ROSTER) or queue["candidates"] != ["v1", "K1a", "K1b", "K2", "K1+K2"]:
        raise ValueError("The queue must retain Amendment 1's eight episodes and fixed candidates")
    for episode in queue["episodes"]:
        authorize_episode(runtime, PlacementRequest(episode), "placement queue")
    heavy_execution(dict(runtime, python=runtime["diagnostic_python"]))
    directory = output_path(ROOT, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "ledger.jsonl"
    # A crash during the last append can leave a torn final line; earlier rows remain usable.
    rows = []
    for line in ledger.read_text().splitlines() if ledger.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    selected, results, executed, skipped = [], {}, 0, 0
    for candidate in queue["candidates"]:
        if candidate == "K1+K2":
            best_k1 = max(("K1a", "K1b"), key=lambda name: sum(r["metrics"]["AUC"] for r in results[name]))
        results[candidate] = []
        for episode in queue["episodes"]:
            request = PlacementRequest(episode, candidate, best_k1 if candidate == "K1+K2" else None)
            key = identity(request)
            prior = next((r for r in reversed(rows) if r["identity"] == key and r["status"] == "completed"), None)
            if prior:
                result = resume_result(prior)
                row = prior
                skipped += 1
            else:
                if executed >= queue["max_trials"]:
                    raise RuntimeError("Trial budget exhausted; resume with a sufficient queue budget")
                started = time.perf_counter()
                row = dict(identity=key, candidate=candidate, episode=episode, k1=request.k1)
                try:
                    path = diagnose(request)
                    row.update(status="completed", run=str(path.relative_to(ROOT)))
                    result = resume_result(row)
                    row["metrics"] = result["metrics"]
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    if hasattr(error, "feature_run_path"):
                        row["run"] = error.feature_run_path
                    raise
                finally:
                    row["seconds"] = time.perf_counter() - started
                    with ledger.open("a") as stream:
                        stream.write("\n" + json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                rows.append(row)
                executed += 1
            selected.append(row)
            results[candidate].append(result)
            print(json.dumps(dict(candidate=candidate, episode=episode, resumed=prior is not None,
                                  metrics=result["metrics"])), flush=True)
    from .placement_report import summarize
    report = summarize(results, directory, best_k1, executed, skipped)
    write_json(directory / "finished.json", dict(status="completed", queue_sha256=sha256_file(ROOT / queue_path),
                                                selected=selected, report=report, executed=executed, skipped=skipped))
    print(directory / "summary.md", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--queue", default="track3/features/pose_insurance/placement_queue.json")
    args = parser.parse_args()
    run_queue(args.queue)


if __name__ == "__main__":
    main()
