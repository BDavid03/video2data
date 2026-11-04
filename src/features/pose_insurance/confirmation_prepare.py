"""Prepare Amendment 2 depth and clusters sequentially through their owning handlers."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
import time
from src.provenance import input_file, write_json
from src.runs import (ROOT, read_json, runtime_flag, authorize_episode, producer,
                         artifact, source_hashes)
from src.features.depth.generate import GenerateDepthRequest, generate_depth
from src.features.mesh_extents.measure import MeasureRequest, measure
from .placement_diagnostic import FEATURE, OUTPUT


def prepare(queue_path):
    """Resume the two preparation stages per episode without reading public object poses."""
    queue = read_json(queue_path)
    if queue["purpose"] != "confirmation" or queue["episodes"] != [2, 27]:
        raise ValueError("Confirmation preparation is fixed to public 2 and 27")
    for item in queue["preparation"]:
        for slice_name, operation, request_type in (("depth", "generate", GenerateDepthRequest),
                                                     ("mesh_extents", "measure", MeasureRequest)):
            request = request_type.from_file(item[f"{slice_name if slice_name == 'depth' else 'measure'}_request"])
            runtime = runtime_flag(FEATURE.parent / slice_name, f"{operation}_authorized", "Confirmation preparation")
            authorize_episode(runtime, request, "confirmation preparation")
    directory = ROOT / OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    ledger = directory / "confirmation_prepare_ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    selected = {}
    for item in queue["preparation"]:
        episode = item["episode"]
        selected[str(episode)] = {}
        for stage, slice_name, request, handler in (
            ("depth", "depth", GenerateDepthRequest.from_file(item["depth_request"]), generate_depth),
            ("measure", "mesh_extents", MeasureRequest.from_file(item["measure_request"]), measure),
        ):
            if stage == "measure":
                request = replace(request, depth_run=selected[str(episode)]["depth"])
            payload = dict(request=asdict(request), sources=source_hashes(FEATURE.parent / slice_name),
                           runtime=input_file(ROOT, FEATURE.parent / slice_name / "runtime.json"))
            key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            prior = next((row for row in reversed(rows) if row["identity"] == key), None)
            if prior:
                run = prior["run"]
                parent = producer(run, f"{slice_name}.{('generate' if stage == 'depth' else 'measure')}", request)
                if stage == "measure":
                    artifact(parent, "overlay_000000.png", {"inputs": []})
                else:
                    frame = next(row for row in parent[1]["frames"] if row["frame_index"] == 0)
                    artifact(parent, frame["artifact"], {"inputs": []}, frame["sha256"])
            else:
                if stage == "depth":
                    locks = [ROOT / "data/.runtime/gpu-stage.lock", ROOT.parents[2] / "data/.runtime/gpu-stage.lock"]
                    if any(path.exists() for path in locks):
                        raise RuntimeError("GPU lease already exists; no owner was touched")
                started = time.perf_counter()
                run = str(handler(request).relative_to(ROOT))
                row = dict(identity=key, episode=episode, stage=stage, run=run,
                           seconds=time.perf_counter() - started)
                with ledger.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
            selected[str(episode)][stage] = run
            write_json(directory / "confirmation_preparation.json", selected)
            print(json.dumps(dict(episode=episode, stage=stage, resumed=prior is not None, run=run)), flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--queue", default="src/features/pose_insurance/placement_confirmation_queue.json")
    args = parser.parse_args()
    prepare(args.queue)


if __name__ == "__main__":
    main()
