"""Persist a gated float64 geometric ensemble of owned prediction artifacts.

No model, checkpoint, GPU or teacher validity is used. Weights must be frozen
independently of holdout results. Importing this module does not import NumPy.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import PurePosixPath

from track3.paths import project_path
from track3.provenance import inherited_provenance, read_run, sha256_file, write_json
from ._common import ROOT, authorize, heavy_execution, number, operation_run, read_json


KINDS = {"learning_experiments.predict", "learning_experiments.boost_predict",
         "learning_experiments.ensemble", "depth_student.predict"}


@dataclass(frozen=True)
class EnsembleRequest:
    prediction_runs: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if (type(self.prediction_runs) is not list or not 1 <= len(self.prediction_runs) <= 8
                or any(type(path) is not str or not path.strip() for path in self.prediction_runs)
                or type(self.weights) is not list or len(self.weights) != len(self.prediction_runs)):
            raise ValueError("Supply 1–8 explicit prediction run paths and matching weights")
        weights = [number(value, "ensemble weight", 0, 1000) for value in self.weights]
        if not any(weights):
            raise ValueError("At least one ensemble weight must be positive")


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _hash(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _relative(path):
    return path.relative_to(ROOT.resolve()).as_posix()


def _artifact(directory, name):
    if (type(name) is not str or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Artifact paths must be native confined run-relative POSIX paths")
    path = project_path(ROOT, directory / name)
    if path != directory / name or not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("Artifact is missing, redirected or outside its owning run")
    return path


def _digest(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _identity(row):
    if type(row) is not dict:
        raise ValueError("Frame identities must be objects")
    for name in ("episode_index", "frame_index"):
        if type(row.get(name)) is not int or row[name] < 0:
            raise ValueError("Frame identity needs nonnegative episode/frame integers")
    if type(row.get("video_pts")) is not int:
        raise ValueError("Frame PTS must be an integer")
    seconds = row.get("video_pts_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds):
        raise ValueError("Frame timestamp must be finite")
    _hash(row.get("sample_sha256"))
    return row["episode_index"], row["frame_index"]


def _ancestry(path, records, hashes, active):
    """Bind artifacts and completed ancestry; retain historical source identities.

    Authorization/request JSON may legitimately change between owned operations.
    Their old hashes are evidence in the immutable run record, not a requirement
    to restore old authorization flags before combining already saved arrays.
    """
    if path in active:
        raise ValueError("Prediction ancestry contains a cycle")
    if path in records:
        return records[path]
    if path.name != "run.json":
        raise ValueError("Expected an explicit owning run.json")
    digest = sha256_file(path)
    record = read_run(ROOT, path)
    active.add(path)
    inputs = record.get("inputs")
    if type(inputs) is not list or any(type(entry) is not dict for entry in inputs):
        raise ValueError("Every ancestor needs an explicit input hash inventory")
    seen = set()
    for entry in inputs:
        if type(entry.get("path")) is not str:
            raise ValueError("Ancestry input path is missing")
        source = project_path(ROOT, entry["path"])
        expected = _hash(entry.get("sha256"))
        seen.add(source)
        if ((source.is_relative_to(ROOT / "track3/features") and source.suffix == ".json")
                or source == ROOT / "configs/perception/moge/local.json"):
            continue
        if source in hashes and hashes[source] != expected:
            raise ValueError("Ancestry contains conflicting input identities")
        if source not in hashes and sha256_file(source) != expected:
            raise ValueError("Ancestry artifact or pinned input changed")
        hashes[source] = expected
        if source.name == "run.json":
            _ancestry(source, records, hashes, active)
    if record["kind"] in KINDS:
        feature, operation = record["kind"].split(".")
        run_id = record.get("run_id")
        if (type(run_id) is not str or not run_id or run_id in (".", "..")
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                       for c in run_id)
                or path != ROOT.resolve() / "outputs" / feature / operation / run_id / "run.json"):
            raise ValueError("Prediction metadata is not in its canonical owning run")
        sources, inventory = record.get("sources"), record.get("source_hashes")
        if type(sources) is not list or not sources or type(inventory) is not dict or not inventory:
            raise ValueError("Prediction producer needs its source identity inventories")
        source_map = {}
        for entry in sources:
            if type(entry) is not dict or type(entry.get("path")) is not str:
                raise ValueError("Malformed prediction source inventory")
            source = project_path(ROOT, entry["path"])
            key = _relative(source)
            expected = _hash(entry.get("sha256"))
            if key in source_map or inventory.get(key) != expected:
                raise ValueError("Prediction source identity inventories disagree")
            source_map[key] = expected
        if source_map != inventory:
            raise ValueError("Prediction source inventories do not match exactly")
        if operation == "ensemble":
            members = record.get("prediction_members")
            if type(members) is not list or not 1 <= len(members) <= 8:
                raise ValueError("Nested ensemble must declare its complete member ancestry")
            member_paths = set()
            for member in members:
                if type(member) is not dict or type(member.get("path")) is not str:
                    raise ValueError("Malformed ensemble member identity")
                member_path = project_path(ROOT, member["path"])
                if (member_path in member_paths or member_path not in seen
                        or member_path not in records or records[member_path]["kind"] not in KINDS
                        or hashes.get(member_path) != _hash(member.get("sha256"))
                        or records[member_path].get("provenance") != member.get("provenance")):
                    raise ValueError("Nested ensemble member is not bound to complete ancestry")
                member_paths.add(member_path)
    if sha256_file(path) != digest or (path in hashes and hashes[path] != digest):
        raise ValueError("Run metadata changed during ancestry admission")
    hashes[path] = digest
    records[path] = record
    active.remove(path)
    return record


def _unchanged(hashes):
    for path, digest in hashes.items():
        if project_path(ROOT, path) != path or sha256_file(path) != digest:
            raise ValueError("Admitted source, metadata or input bytes changed")


def ensemble_predictions(request: EnsembleRequest):
    """Return the completed owning run.json after separate ensemble admission."""
    authorize("ensemble")
    authorize("model")
    authorize("predict")
    # Lists are mutable even in a frozen dataclass; snapshot and re-admit them.
    request = EnsembleRequest(list(request.prediction_runs), list(request.weights))
    weights = [number(value, "ensemble weight", 0, 1000) for value in request.weights]
    run_paths = [project_path(ROOT, path) for path in request.prediction_runs]
    if len(set(run_paths)) != len(run_paths):
        raise ValueError("Duplicate or aliased prediction run paths are refused")
    for index, path in enumerate(run_paths):
        if any(path.samefile(other) for other in run_paths[:index]):
            raise ValueError("Prediction run files cannot alias")
    records, hashes = {}, {}
    producers = [_ancestry(path, records, hashes, set()) for path in run_paths]
    if any(producer["kind"] not in KINDS for producer in producers):
        raise ValueError("Only completed research prediction or ensemble runs may be combined")
    first = producers[0]
    group = first.get("group")
    if group not in ("train", "tuning", "holdout"):
        raise ValueError("Prediction group must be train, tuning or holdout")
    if group == "holdout":
        authorize("holdout_predict")
    for producer in producers:
        for name in ("group", "dataset_sha256", "data_contract", "episode_partition"):
            if name not in producer or not _same(producer[name], first.get(name)):
                raise ValueError("Members must share dataset, data contract, episode partition and group")
    dataset_hash = _hash(first["dataset_sha256"])
    datasets = set()
    for producer in producers:
        candidates = {project_path(ROOT, entry["path"]) for entry in producer["inputs"]
                      if project_path(ROOT, entry["path"]).name == "dataset.json"
                      and entry["sha256"] == dataset_hash}
        if len(candidates) != 1:
            raise ValueError("Each prediction must bind one explicit matching dataset.json input")
        datasets.update(candidates)
    if len(datasets) != 1:
        raise ValueError("Members must bind the same explicit dataset path")
    dataset_path = datasets.pop()
    # All execution flags precede cache creation and any third-party import.
    heavy_execution("ensemble")
    import cv2
    import numpy as np
    from .data import PreparedPairs, pinhole_input_validity

    pairs = PreparedPairs(dataset_path, first["episode_partition"], group, np)
    if pairs.dataset_sha256 != dataset_hash or not _same(pairs.data_contract, first["data_contract"]):
        raise ValueError("Member data contract differs from its owned prepared dataset")
    expected_input_valid = pinhole_input_validity(pairs.data_contract, cv2, np)
    expected = {(row["episode_index"], row["frame_index"]): row for row in pairs.rows}
    maps, paths, all_artifacts, file_ids = [], [], [], set()
    for producer, run_path in zip(producers, run_paths):
        frames, inventory = producer.get("frames"), producer.get("artifacts")
        if type(frames) is not list or not frames or type(inventory) is not dict:
            raise ValueError("Prediction needs nonempty frame and owned artifact inventories")
        counts = [producer[name] for name in ("completed_frames", "complete_frames") if name in producer]
        if not counts or any(type(count) is not int or count != len(frames) for count in counts):
            raise ValueError("Prediction completed-frame count is inconsistent")
        frame_map, path_map = {}, {}
        for frame in frames:
            key = _identity(frame)
            row = expected.get(key)
            if key in frame_map or row is None or frame.get("status") != "completed":
                raise ValueError("Frames must be unique, completed and within the declared dataset group")
            if (frame["sample_sha256"] != row["sha256"]
                    or any(not _same(frame[name], row[name]) for name in ("video_pts", "video_pts_seconds"))):
                raise ValueError("Frame timestamps or source sample hashes differ from the dataset")
            digest = _hash(frame.get("sha256"))
            path = _artifact(run_path.parent, frame.get("artifact"))
            stat = path.stat()
            file_id = (stat.st_dev, stat.st_ino)
            if (path.suffix != ".npz" or inventory.get(frame["artifact"]) != digest
                    or file_id in file_ids
                    or sha256_file(path) != digest):
                raise ValueError("Prediction NPZ is aliased or differs from its owning hash inventory")
            file_ids.add(file_id)
            frame_map[key], path_map[key] = frame, path
            all_artifacts.append(path)
            hashes[path] = digest
        if maps and set(frame_map) != set(maps[0]):
            raise ValueError("Members must have EXACT matching unique frame sets")
        maps.append(frame_map)
        paths.append(path_map)
    contract = pairs.data_contract
    width, height = contract["geometry"]["output_size_width_height"]
    if width * height > 65536:
        raise ValueError("Source predictions exceed 65536 pixels")
    minimum, maximum = contract["minimum_depth_m"], contract["maximum_depth_m"]
    _unchanged(hashes)
    members = [{"path": _relative(path), "sha256": hashes[path],
                "kind": producer["kind"], "provenance": producer["provenance"], "weight": weight}
               for path, producer, weight in zip(run_paths, producers, weights)]
    with operation_run("ensemble", request, inputs=[dataset_path, *all_artifacts],
                       parents=list(records)) as (directory, record):
        if record["provenance"] != inherited_provenance(records.values()):
            raise ValueError("Output provenance differs from its admitted ancestry")
        record.update(technique="ensemble", model_config={"weights": weights},
                      prediction_members=members, dataset=_relative(dataset_path),
                      dataset_sha256=dataset_hash, data_contract=contract,
                      episode_partition=first["episode_partition"], group=group,
                      total_group_samples=len(pairs), selected_samples=len(maps[0]),
                      sample_selection="Exact shared member frame set in first-member order",
                      completed_frames=0, frames=[], model_execution=False,
                      prediction_executed=False, training_executed=False, ensemble_executed=True,
                      targets_used_for_prediction=False, final_holdout_used_for_selection=False,
                      aggregation="Float64 masked weighted natural-log mean, exponentiated per pixel",
                      disagreement="Weighted population log-depth standard deviation; NaN with fewer than two positive-weight supported members; not calibrated uncertainty",
                      member_support="member_count counts supported positive-weight direct parents, not independent models; effective_member_count is squared total weight over summed squared admitted weights",
                      validity="Shared RGB input_valid intersected with finite positive bounded member support")
        (directory / "frames").mkdir()
        for key, source_frame in maps[0].items():
            numerator = np.zeros((height, width), dtype=np.float64)
            denominator = np.zeros_like(numerator)
            log_square_sum = np.zeros_like(numerator)
            weight_square_sum = np.zeros_like(numerator)
            member_count = np.zeros((height, width), dtype=np.uint8)
            shared_mask = shared_k = None
            for frame_map, path_map, weight in zip(maps, paths, weights):
                path, frame = path_map[key], frame_map[key]
                with path.open("rb") as stream:
                    if _digest(stream) != frame["sha256"]:
                        raise ValueError("Prediction artifact changed before decoding")
                    with np.load(stream, allow_pickle=False) as arrays:
                        required = {"depth_m", "valid", "input_valid", "input_intrinsics_pixels"}
                        if not required.issubset(arrays.files):
                            raise ValueError("Prediction requires depth, valid, input_valid and intrinsics")
                        depth, valid, input_valid, intrinsics = (
                            arrays[name].copy() for name in
                            ("depth_m", "valid", "input_valid", "input_intrinsics_pixels"))
                    if _digest(stream) != frame["sha256"]:
                        raise ValueError("Prediction artifact changed during decoding")
                if (depth.shape != (height, width) or depth.dtype.kind != "f"
                        or valid.shape != depth.shape or valid.dtype != np.dtype("bool")
                        or input_valid.shape != depth.shape or input_valid.dtype != np.dtype("bool")
                        or intrinsics.shape != (3, 3) or intrinsics.dtype.kind != "f"
                        or not np.isfinite(intrinsics).all()
                        or not np.allclose(intrinsics, contract["geometry"]["input_intrinsics_pixels"],
                                           rtol=1e-5, atol=1e-5)):
                    raise ValueError("Prediction array types, bounded shape or camera geometry differ")
                if not np.array_equal(input_valid, expected_input_valid) or np.any(valid & ~input_valid):
                    raise ValueError("Member support differs from calibrated RGB input rays")
                if shared_mask is None:
                    shared_mask, shared_k = input_valid, intrinsics
                elif (not np.array_equal(shared_mask, input_valid)
                      or shared_k.dtype != intrinsics.dtype or not np.array_equal(shared_k, intrinsics)):
                    raise ValueError("Every member must share exact input_valid and typed intrinsics arrays")
                # Scalar bounds must retain their precision for float32 members.
                depth64 = depth.astype(np.float64)
                support = (valid & input_valid & np.isfinite(depth64) & (depth64 > 0)
                           & (depth64 >= minimum) & (depth64 <= maximum))
                contribution = support.astype(np.float64) * weight
                safe = np.where(support, depth64, 1.0)
                log_depth = np.log(safe)
                numerator += log_depth * contribution
                log_square_sum += log_depth**2 * contribution
                weight_square_sum += support * weight**2
                member_count += (support & (weight > 0)).astype(np.uint8)
                denominator += contribution
            valid = denominator > 0
            safe_denominator = np.where(valid, denominator, 1.0)
            log_mean = numerator / safe_denominator
            depth = np.exp(log_mean)
            valid &= (shared_mask & np.isfinite(depth) & (depth > 0)
                      & (depth >= minimum) & (depth <= maximum))
            depth = np.where(valid, depth, np.nan)
            log_depth_std = np.where(valid & (member_count >= 2),
                np.sqrt(np.maximum(0.0, log_square_sum / safe_denominator - log_mean**2)), np.nan)
            effective_member_count = np.where(valid,
                denominator**2 / np.where(weight_square_sum > 0, weight_square_sum, 1.0), 0.0)
            frame = {name: source_frame[name] for name in
                     ("episode_index", "frame_index", "video_pts", "video_pts_seconds", "sample_sha256")}
            frame["status"] = "running"
            record["frames"].append(frame)
            destination = directory / "frames" / f"episode_{key[0]:06d}_frame_{key[1]:06d}.npz"
            temporary = destination.with_suffix(".npz.tmp")
            try:
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, depth_m=depth, valid=valid, input_valid=shared_mask,
                                        input_intrinsics_pixels=shared_k, member_count=member_count,
                                        effective_member_count=effective_member_count, log_depth_std=log_depth_std)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            frame.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                         sha256=sha256_file(destination), valid_pixels=int(valid.sum()))
            record["completed_frames"] += 1
            write_json(directory / "run.json", record)
        _unchanged(hashes)
    return directory / "run.json"
