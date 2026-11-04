"""Owned research boosting operations; execution remains explicitly gated.

Training requires boost_train, model and train authorization; prediction requires
boost_predict, model and predict, plus holdout_predict for the holdout group.
NumPy/OpenCV are imported only after those gates and heavy_execution. No torch,
GPU, compiler, installation or automatic model selection is used here.

Uniform reservoir sampling traverses admitted training pairs in producer order,
then valid pixels in row-major order. Memory retains at most 2048 RGB/log-depth
rows. The reservoir does not guarantee equal episode weights or coverage of
every training episode; exact candidate/used coverage is recorded. Prediction
uses all RGB pixels, with geometric input-ray support independent of labels.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
import random
from uuid import uuid4

from track3.paths import project_path
from track3.provenance import inherited_provenance, read_run, sha256_file, write_json
from ._common import (FEATURE, ROOT, artifact, authorize, episode_partition,
                      heavy_execution, integer, operation_run, read_json)
from ._config import parameters as normalize_parameters
from .boosting import (FEATURE_SCHEMA, MAX_ROWS, TARGET_TRANSFORM, BoostModel,
                       BoostRequest, DecisionStump, fit, predict)

__all__ = ["BoostTrainRequest", "BoostPredictRequest", "train_boost", "predict_boost"]

_TRAIN = (0, 11, 12, 18, 21, 23, 26, 31)
_TUNING = (39, 42)
_HOLDOUT = (43, 44)
_SOURCES = ("boosting.py", "_config.py", "data.py", "boosting_operations.py")
_KIND = "learning_experiments.boost_checkpoint"
_DATA_FIELDS = ("dataset_revision", "split_id", "target_kind", "geometry",
                "max_image_side", "calibration_sha256", "minimum_depth_m", "maximum_depth_m")
_SAMPLING = "seeded_uniform_reservoir_producer_order_row_major_valid_pixels"


@dataclass(frozen=True)
class BoostTrainRequest:
    dataset: str | None = None
    parameters: dict = field(default_factory=dict)
    seed: int = 31
    train_episodes: tuple[int, ...] = _TRAIN
    tuning_episodes: tuple[int, ...] = _TUNING
    holdout_episodes: tuple[int, ...] = _HOLDOUT

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        _optional_path(self.dataset, "dataset")
        normalize_parameters("gradient_boosting", self.parameters)
        integer(self.seed, "seed", 0, 2**32 - 1)


@dataclass(frozen=True)
class BoostPredictRequest:
    dataset: str | None = None
    checkpoint: str | None = None
    group: str = "tuning"
    max_samples: int = 128

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        _optional_path(self.dataset, "dataset")
        _optional_path(self.checkpoint, "checkpoint")
        if self.group not in ("train", "tuning", "holdout"):
            raise ValueError("group must be train, tuning or holdout")
        integer(self.max_samples, "max_samples", 1, 10000)


def _optional_path(value, name):
    if value is not None and (type(value) is not str or not value):
        raise ValueError(f"{name} must be an explicit project path")


def _plain(value):
    """Serialize finite JSON metadata without invoking arbitrary object hooks."""
    def visit(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child)
            return
        raise ValueError("Require finite serialized JSON metadata")
    visit(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _same(left, right):
    return json.dumps(_plain(left), sort_keys=True) == json.dumps(_plain(right), sort_keys=True)


def _digest(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Require a lowercase SHA-256 digest")
    return value


def _relative(path):
    return path.relative_to(ROOT.resolve()).as_posix()


def _source_identity(record):
    hashes = record.get("source_hashes")
    if type(hashes) is not dict:
        raise ValueError("Missing training source hash inventory")
    sources = []
    for filename in _SOURCES:
        path = project_path(ROOT, FEATURE / filename)
        row = {"path": _relative(path), "sha256": sha256_file(path)}
        if hashes.get(row["path"]) != row["sha256"]:
            raise ValueError("Boosting implementation differs from its training sources")
        sources.append(row)
    return sources


def _sample_rows(pairs, seed, np):
    rng = random.Random(seed)
    reservoir, candidates = [], 0
    episodes = sorted({row["episode_index"] for row in pairs.rows})
    seen = {episode: 0 for episode in episodes}
    for index in range(len(pairs)):
        sample = pairs[index]
        episode = sample["row"]["episode_index"]
        rgb = sample["rgb"].reshape(-1, 3)
        depths = sample["depth_m"].reshape(-1)
        usable = (sample["valid"].reshape(-1) & np.isfinite(depths)
                  & (depths >= 0.001) & (depths <= 1000.0))
        for pixel in np.flatnonzero(usable):
            candidates += 1
            seen[episode] += 1
            slot = candidates - 1 if candidates <= MAX_ROWS else rng.randrange(candidates)
            if slot < MAX_ROWS:
                row = ([float(value) / 255.0 for value in rgb[pixel]],
                       math.log(float(depths[pixel])), episode)
                if candidates <= MAX_ROWS:
                    reservoir.append(row)
                else:
                    reservoir[slot] = row
    if not reservoir:
        raise ValueError("No valid training labels in boosting's [0.001,1000] metre bounds")
    used = {episode: 0 for episode in episodes}
    for _, _, episode in reservoir:
        used[episode] += 1
    sampling = {
        "method": _SAMPLING, "seed": seed, "capacity": MAX_ROWS,
        "candidate_rows": candidates, "used_rows": len(reservoir),
        "target_depth_bounds_m": [0.001, 1000.0],
        "training_pair_count": len(pairs), "declared_train_episodes": episodes,
        "candidate_rows_by_episode": {str(e): seen[e] for e in episodes},
        "used_rows_by_episode": {str(e): used[e] for e in episodes},
        "candidate_episode_coverage": [e for e in episodes if seen[e]],
        "used_episode_coverage": [e for e in episodes if used[e]],
        "equal_episode_weighting": False, "tuning_or_holdout_used": False,
    }
    return reservoir, sampling


def train_boost(request: BoostTrainRequest):
    """Fit only admitted training labels and return an owned checkpoint.json."""
    authorize("boost_train")
    authorize("model")
    authorize("train")
    if request.dataset is None:
        raise ValueError("Supply a completed prepared dataset.json")
    config = normalize_parameters("gradient_boosting", request.parameters)
    partition = episode_partition(request.train_episodes, request.tuning_episodes, request.holdout_episodes)
    heavy_execution("boost_train")
    import numpy as np
    from .data import PreparedPairs

    pairs = PreparedPairs(project_path(ROOT, request.dataset), partition, "train", np)
    with operation_run("boost_train", request, inputs=pairs.input_paths,
                       parents=[pairs.producer_path]) as (directory, record):
        record.update(technique="gradient_boosting", model_config=config, parameters=config,
                      episode_partition=partition, data_contract=pairs.data_contract,
                      dataset_sha256=pairs.dataset_sha256, training_samples=len(pairs),
                      training_executed=False, prediction_executed=False,
                      initialization="training_log_depth_mean_no_pretrained_weights",
                      selection_note="Training episodes only; no tuning or holdout fit or selection")
        record["source_identity"] = _source_identity(record)
        rows, sampling = _sample_rows(pairs, request.seed, np)
        record.update(sampling=sampling, training_rows=len(rows))
        model = fit(BoostRequest(**config, train_episodes=tuple(partition["train_episodes"]),
                                tuning_episodes=tuple(partition["tuning_episodes"]),
                                holdout_episodes=tuple(partition["holdout_episodes"])),
                    [row[0] for row in rows], [row[1] for row in rows],
                    episodes=[row[2] for row in rows])
        record.update(training_executed=True, trained_estimators=len(model.stumps))
        state = json.loads(json.dumps(asdict(model), allow_nan=False))
        payload = {
            "schema_version": 1, "kind": _KIND, "research_only": True,
            "pretrained_weights": False, "training_run": _relative(directory / "run.json"),
            "training_run_id": record["run_id"], "training_request": _plain(record["request"]),
            "provenance": record["provenance"], "technique": "gradient_boosting",
            "model_config": config, "parameters": config, "episode_partition": partition,
            "data_contract": pairs.data_contract, "dataset_sha256": pairs.dataset_sha256,
            "training_rows": len(rows), "trained_estimators": len(model.stumps),
            "sampling": sampling, "source_identity": record["source_identity"], "model_state": state,
        }
        destination = artifact(directory / "checkpoint.json")
        write_json(destination, payload)
        record.update(checkpoint="checkpoint.json", checkpoint_sha256=sha256_file(destination))
        record["artifacts"]["checkpoint.json"] = record["checkpoint_sha256"]
    return destination


def _admit_sampling(record, request, partition):
    sampling = _plain(record.get("sampling"))
    if type(sampling) is not dict:
        raise ValueError("Missing reservoir sampling provenance")
    rows = integer(record.get("training_rows"), "training_rows", 1, MAX_ROWS)
    candidates = sampling.get("candidate_rows")
    if type(candidates) is not int or candidates < rows or rows != min(candidates, MAX_ROWS):
        raise ValueError("Reservoir counters do not describe a completed bounded sample")
    samples = record.get("training_samples")
    if type(samples) is not int or samples < 1 or candidates > samples * 65536:
        raise ValueError("Candidate rows exceed the admitted training pixel budget")
    episodes = partition["train_episodes"]
    fixed = {"method": _SAMPLING, "seed": request["seed"], "capacity": MAX_ROWS,
             "used_rows": rows, "training_pair_count": samples, "declared_train_episodes": episodes,
             "target_depth_bounds_m": [0.001, 1000.0], "equal_episode_weighting": False,
             "tuning_or_holdout_used": False}
    if any(not _same(sampling.get(key), value) for key, value in fixed.items()):
        raise ValueError("Sampling metadata differs from the training request")
    mappings = []
    for name, total in (("candidate_rows_by_episode", candidates), ("used_rows_by_episode", rows)):
        mapping = sampling.get(name)
        if (type(mapping) is not dict or set(mapping) != {str(e) for e in episodes}
                or any(type(v) is not int or v < 0 for v in mapping.values())
                or sum(mapping.values()) != total):
            raise ValueError("Sampling episode counts do not sum to the recorded total")
        mappings.append(mapping)
    if any(mappings[1][str(e)] > mappings[0][str(e)] for e in episodes):
        raise ValueError("Used episode rows exceed their candidate counts")
    for name, mapping in zip(("candidate_episode_coverage", "used_episode_coverage"), mappings):
        if not _same(sampling.get(name), [e for e in episodes if mapping[str(e)]]):
            raise ValueError("Sampling episode coverage differs from its exact counts")
    return sampling


def _admit_state(state, config, partition, rows):
    state = _plain(state)
    fields = {"intercept", "learning_rate", "stumps", "train_episodes", "tuning_episodes",
              "holdout_episodes", "training_rows", "feature_schema", "target_transform",
              "scratch_checkpoint_admitted", "submission_candidate"}
    if type(state) is not dict or set(state) != fields:
        raise ValueError("Unknown or incomplete BoostModel state")
    expected = {"learning_rate": config["learning_rate"], "training_rows": rows,
                "feature_schema": list(FEATURE_SCHEMA), "target_transform": TARGET_TRANSFORM,
                "scratch_checkpoint_admitted": False, "submission_candidate": False,
                **{f"{group}_episodes": partition[f"{group}_episodes"]
                   for group in ("train", "tuning", "holdout")}}
    if any(not _same(state[key], value) for key, value in expected.items()):
        raise ValueError("Boost model state differs from the training contract")
    intercept = state["intercept"]
    if type(intercept) not in (int, float) or not math.isfinite(intercept):
        raise ValueError("Boost intercept must be finite")
    stumps = state["stumps"]
    if type(stumps) is not list or len(stumps) != config["estimators"]:
        raise ValueError("Boost checkpoint is not a fully fitted estimator sequence")
    for stump in stumps:
        if (type(stump) is not dict or set(stump) != {"feature", "threshold", "left", "right"}
                or (stump["feature"] is not None
                    and (type(stump["feature"]) is not int or stump["feature"] not in (0, 1, 2)))
                or any(type(stump[k]) not in (int, float) or not math.isfinite(stump[k])
                       for k in ("threshold", "left", "right"))
                or not 0 <= stump["threshold"] <= 1):
            raise ValueError("Require finite RGB decision stumps with bounded thresholds")
    return BoostModel(intercept=intercept, learning_rate=config["learning_rate"],
                      stumps=tuple(DecisionStump(**stump) for stump in stumps),
                      train_episodes=tuple(partition["train_episodes"]),
                      tuning_episodes=tuple(partition["tuning_episodes"]),
                      holdout_episodes=tuple(partition["holdout_episodes"]), training_rows=rows)


def _load_checkpoint(path):
    """Read same hashed JSON bytes and bind state to the completed owning run."""
    path = project_path(ROOT, path)
    if path.name != "checkpoint.json":
        raise ValueError("Require a direct owned checkpoint.json")
    directory = path.parent
    run_path = directory / "run.json"
    record = read_run(ROOT, run_path, kind="learning_experiments.boost_train")
    run_id = record.get("run_id")
    if (type(run_id) is not str or not run_id or run_id in (".", "..")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in run_id)):
        raise ValueError("Invalid boosting training run ID")
    expected_directory = ROOT.resolve() / "outputs/learning_experiments/boost_train" / run_id
    if directory != expected_directory or expected_directory.resolve() != expected_directory:
        raise ValueError("Boost checkpoint escapes its exact owning run directory")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not 1 <= len(raw) <= 1024 * 1024:
        raise ValueError("Boost checkpoint exceeds the bounded JSON size")
    digest = hashlib.sha256(raw).hexdigest()
    artifacts = record.get("artifacts")
    if (record.get("checkpoint") != "checkpoint.json" or type(artifacts) is not dict
            or artifacts.get("checkpoint.json") != digest or record.get("checkpoint_sha256") != digest):
        raise ValueError("Boost checkpoint bytes differ from their owning artifact inventory")
    payload = _plain(json.loads(raw.decode("utf-8-sig")))
    request = _plain(record.get("request"))
    if type(request) is not dict:
        raise ValueError("Missing serialized boosting request")
    train_request = BoostTrainRequest(**request)
    config = normalize_parameters("gradient_boosting", train_request.parameters)
    partition = episode_partition(train_request.train_episodes, train_request.tuning_episodes,
                                  train_request.holdout_episodes)
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("technique") != "gradient_boosting" or record.get("research_only") is not True
            or record.get("training_executed") is not True
            or record.get("submission_candidate") is not False
            or record.get("scratch_checkpoint_admitted") is not False
            or not _same(record.get("model_config"), config) or not _same(record.get("parameters"), config)
            or not _same(record.get("episode_partition"), partition)
            or type(record.get("trained_estimators")) is not int
            or record["trained_estimators"] != config["estimators"]):
        raise ValueError("Completed boosting run differs from its request or estimator counters")
    sampling = _admit_sampling(record, request, partition)
    sources = _source_identity(record)
    if not _same(record.get("source_identity"), sources):
        raise ValueError("Recorded boosting source identity differs from actual source bytes")
    contract = _plain(record.get("data_contract"))
    if type(contract) is not dict or any(key not in contract for key in _DATA_FIELDS):
        raise ValueError("Incomplete boosting data contract")
    if contract["target_kind"] != "moge2_pseudo_depth":
        raise ValueError("Boosting requires the prepared MoGe2 pseudo-depth target")
    _digest(contract["calibration_sha256"])
    side = integer(contract["max_image_side"], "max_image_side", 64, 256)
    low, high = contract["minimum_depth_m"], contract["maximum_depth_m"]
    if (type(low) not in (int, float) or type(high) not in (int, float)
            or not math.isfinite(low) or not math.isfinite(high) or not 0 < low < high):
        raise ValueError("Training depth bounds must be positive and strictly ordered")
    dataset_digest = _digest(record.get("dataset_sha256"))
    if train_request.dataset is None:
        raise ValueError("Training checkpoint requires its explicit dataset input")
    dataset_path = project_path(ROOT, train_request.dataset)
    inputs = record.get("inputs")
    if type(inputs) is not list or any(type(entry) is not dict for entry in inputs):
        raise ValueError("Training checkpoint requires its input hash inventory")
    matches = [entry for entry in inputs if type(entry.get("path")) is str
               and project_path(ROOT, entry["path"]) == dataset_path]
    if len(matches) != 1 or matches[0].get("sha256") != dataset_digest:
        raise ValueError("Training dataset is not uniquely bound to the input inventory")
    dataset_raw = dataset_path.read_bytes()
    if hashlib.sha256(dataset_raw).hexdigest() != dataset_digest:
        raise ValueError("Training dataset manifest bytes changed")
    document = _plain(json.loads(dataset_raw.decode("utf-8-sig")))
    if (type(document) is not dict or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or document.get("kind") != "scratch_depth_training_pairs"):
        raise ValueError("Training dataset must be a prepared-pairs manifest")
    inherited_provenance([document])
    if (document["provenance"] != record["provenance"]
            or any(not _same(document.get(key), value) for key, value in contract.items())
            or contract["dataset_revision"] != partition["dataset_revision"]
            or contract["split_id"] != partition["split_id"]):
        raise ValueError("Training dataset and checkpoint contracts differ")
    dataset_rows = document.get("samples")
    if type(dataset_rows) is not list or any(type(row) is not dict for row in dataset_rows):
        raise ValueError("Training dataset lacks its sample inventory")
    declared = set(partition["train_episodes"] + partition["tuning_episodes"] + partition["holdout_episodes"])
    seen_samples, sample_counts = set(), {e: 0 for e in partition["train_episodes"]}
    for row in dataset_rows:
        episode, frame = row.get("episode_index"), row.get("frame_index")
        if (type(episode) is not int or episode not in declared or type(frame) is not int
                or frame < 0 or (episode, frame) in seen_samples):
            raise ValueError("Dataset contains duplicate frames or undeclared episodes")
        seen_samples.add((episode, frame))
        if episode in sample_counts:
            sample_counts[episode] += 1
    if sum(sample_counts.values()) != record["training_samples"] or any(v == 0 for v in sample_counts.values()):
        raise ValueError("Training sample counter differs from the source dataset")
    geometry = contract["geometry"]
    if type(geometry) is not dict:
        raise ValueError("Missing admitted training geometry")
    size = geometry.get("output_size_width_height")
    if type(size) is not list or len(size) != 2:
        raise ValueError("Missing admitted training output size")
    width, height = [integer(v, "training image dimension", 1, side) for v in size]
    intrinsics = geometry.get("input_intrinsics_pixels")
    if (type(intrinsics) is not list or len(intrinsics) != 3
            or any(type(row) is not list or len(row) != 3 for row in intrinsics)
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for row in intrinsics for v in row)
            or intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0
            or intrinsics[0][1] != 0 or intrinsics[1][0] != 0
            or intrinsics[2] != [0, 0, 1]):
        raise ValueError("Require finite canonical positive-focal pinhole intrinsics")
    for episode, count in sample_counts.items():
        if sampling["candidate_rows_by_episode"][str(episode)] > count * width * height:
            raise ValueError("Candidate episode rows exceed the source sample pixel budget")
    expected = {
        "schema_version": 1, "kind": _KIND, "research_only": True, "pretrained_weights": False,
        "training_run": _relative(run_path), "training_run_id": run_id, "training_request": request,
        "provenance": record["provenance"], "technique": "gradient_boosting", "model_config": config,
        "parameters": config, "episode_partition": partition, "data_contract": contract,
        "dataset_sha256": dataset_digest, "training_rows": record["training_rows"],
        "trained_estimators": config["estimators"], "sampling": sampling, "source_identity": sources,
    }
    if type(payload) is not dict or set(payload) != set(expected) | {"model_state"}:
        raise ValueError("Unknown or incomplete boosting checkpoint payload")
    if any(not _same(payload[key], value) for key, value in expected.items()):
        raise ValueError("Boost checkpoint metadata differs from its completed owner")
    model = _admit_state(payload["model_state"], config, partition, record["training_rows"])
    return model, payload, run_path, digest


def predict_boost(request: BoostPredictRequest):
    """Predict RGB pixels in bounded chunks and return the completed run.json."""
    authorize("boost_predict")
    authorize("model")
    authorize("predict")
    if request.group == "holdout":
        authorize("holdout_predict")
    if request.dataset is None or request.checkpoint is None:
        raise ValueError("Supply a prepared dataset and an owned boosting checkpoint")
    heavy_execution("boost_predict")
    import numpy as np
    import cv2
    from .data import PreparedPairs, pinhole_input_validity

    checkpoint_path = project_path(ROOT, request.checkpoint)
    model, payload, training_path, checkpoint_digest = _load_checkpoint(checkpoint_path)
    pairs = PreparedPairs(project_path(ROOT, request.dataset), payload["episode_partition"], request.group, np)
    if not _same(pairs.data_contract, payload["data_contract"]):
        raise ValueError("Prediction geometry, calibration and depth contract differ from training")
    input_valid = pinhole_input_validity(pairs.data_contract, cv2, np)
    width, height = pairs.data_contract["geometry"]["output_size_width_height"]
    if input_valid.shape != (height, width) or input_valid.dtype != np.dtype("bool"):
        raise ValueError("Geometric input validity must be a boolean mask on the admitted grid")
    selected = range(min(len(pairs), request.max_samples))
    with operation_run("boost_predict", request, inputs=[checkpoint_path, *pairs.input_paths],
                       parents=[training_path, pairs.producer_path]) as (directory, record):
        record.update(technique="gradient_boosting", model_config=payload["model_config"],
                      episode_partition=payload["episode_partition"], data_contract=pairs.data_contract,
                      dataset_sha256=pairs.dataset_sha256, group=request.group,
                      checkpoint_sha256=checkpoint_digest, checkpoint_training_rows=model.training_rows,
                      checkpoint_trained_estimators=len(model.stumps), training_executed=False,
                      prediction_executed=True, targets_available_in_pairs=True,
                      targets_used_for_prediction=False, total_group_samples=len(pairs),
                      selected_samples=len(selected), completed_frames=0, frames=[],
                      sample_selection="First max_samples in producer order, no random subsampling",
                      validity="Geometric RGB ray support intersected with finite bounded predicted depth",
                      accuracy_note="No score or final-holdout selection is performed")
        artifact(directory / "frames").mkdir()
        for index in selected:
            sample = pairs[index]
            row = sample["row"]
            frame = {"episode_index": row["episode_index"], "frame_index": row["frame_index"],
                     "video_pts": row["video_pts"], "video_pts_seconds": row["video_pts_seconds"],
                     "sample_sha256": row["sha256"], "status": "running"}
            record["frames"].append(frame)
            rgb = sample["rgb"].reshape(-1, 3)
            depth64 = np.empty(len(rgb), dtype=np.float64)
            for start in range(0, len(rgb), MAX_ROWS):
                features = (rgb[start:start + MAX_ROWS].astype(np.float64) / 255.0).tolist()
                depth64[start:start + len(features)] = predict(
                    model, features, episodes=[row["episode_index"]] * len(features))
            depth64 = depth64.reshape(height, width)
            low, high = pairs.data_contract["minimum_depth_m"], pairs.data_contract["maximum_depth_m"]
            # Reject float32 overflow before conversion, then recheck rounding.
            valid = (input_valid & np.isfinite(depth64) & (depth64 > 0)
                     & (depth64 >= low) & (depth64 <= high) & (depth64 <= np.finfo(np.float32).max))
            depth = np.where(valid, depth64, np.nan).astype(np.float32)
            # Compare stored values in float64 so scalar bounds are not rounded
            # to float32 along with the predictions (e.g. 1.00000001 -> 1.0).
            stored_depth64 = depth.astype(np.float64)
            valid &= (np.isfinite(stored_depth64) & (stored_depth64 > 0)
                      & (stored_depth64 >= low) & (stored_depth64 <= high))
            depth = np.where(valid, depth, np.nan).astype(np.float32)
            destination = artifact(directory / "frames" / (
                f"episode_{row['episode_index']:06d}_frame_{row['frame_index']:06d}.npz"))
            temporary = artifact(destination.parent / f".{destination.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, depth_m=depth, valid=valid, input_valid=input_valid,
                                        input_intrinsics_pixels=sample["input_intrinsics_pixels"])
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            frame.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                         sha256=sha256_file(destination), valid_pixels=int(valid.sum()))
            record["artifacts"][frame["artifact"]] = frame["sha256"]
            record["completed_frames"] += 1
            write_json(artifact(directory / "run.json"), record)
    return directory / "run.json"
