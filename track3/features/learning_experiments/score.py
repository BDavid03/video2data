"""Disabled-by-default accuracy diagnostic for complete research predictions.

Future execution requires score and model authorization before NumPy import or
cache creation, and holdout_predict authorization for the holdout group. No
model is constructed or executed. Completed predict/boost_predict/ensemble producers
must cover exactly every prepared row in the declared group, including every
selected episode; a first-max_samples prefix cannot be scored as a full group.

The metric is absolute natural-log depth error on the intersection of bounded,
finite prediction support and usable prepared pseudo-target labels. Pixel errors
are summed in float64 within each episode, then episode MAEs receive equal
weight. Coverage is recorded per sample, per episode and overall. Zero sample
support or coverage below minimum_prediction_fraction in any episode refuses a score.
This is a research comparison to MoGe2 estimated-metre pseudo-labels whose
accuracy is not established, not an official evaluation or holdout selection.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import PurePosixPath

from track3.paths import project_path
from track3.provenance import read_run, sha256_file
from ._common import ROOT, authorize, heavy_execution, operation_run, read_json


@dataclass(frozen=True)
class ScoreRequest:
    dataset: str | None = None
    prediction_run: str | None = None
    group: str = "tuning"
    minimum_prediction_fraction: float = 0.9

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        for name in ("dataset", "prediction_run"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be an explicit local artifact path")
        if self.group not in ("train", "tuning", "holdout"):
            raise ValueError("group must be train, tuning or holdout")
        if (type(self.minimum_prediction_fraction) not in (int, float)
                or not math.isfinite(self.minimum_prediction_fraction)
                or not 0 < self.minimum_prediction_fraction <= 1):
            raise ValueError("minimum_prediction_fraction must be finite in (0, 1]")


def _key(row):
    if not isinstance(row, dict):
        raise ValueError("Prediction frames must be manifest objects")
    episode, frame = row.get("episode_index"), row.get("frame_index")
    if type(episode) is not int or type(frame) is not int or min(episode, frame) < 0:
        raise ValueError("Prediction frame identity must use nonnegative episode/frame integers")
    return episode, frame


def _prediction_path(directory, row):
    name = row.get("artifact")
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Prediction artifacts require confined relative POSIX paths")
    path = project_path(ROOT, directory / name)
    if not path.is_relative_to(directory) or path != directory / name:
        raise ValueError("Prediction artifact is redirected or escapes its owning run")
    if path.suffix != ".npz" or not path.is_file():
        raise FileNotFoundError(f"Required actual prediction NPZ artifact is missing: {path}")
    return path


def _digest(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def score_predictions(request: ScoreRequest):
    """Persist an owning score run only after separately authorized execution."""
    authorize("score")
    if request.group == "holdout":
        authorize("holdout_predict")
    if request.dataset is None or request.prediction_run is None:
        raise ValueError("Supply actual prepared dataset.json and completed prediction run.json")
    heavy_execution("score")
    import cv2
    import numpy as np
    from .data import PreparedPairs, pinhole_input_validity

    prediction_path = project_path(ROOT, request.prediction_run)
    if prediction_path.name != "run.json":
        raise ValueError("Expected the owning prediction run.json")
    prediction_hash = sha256_file(prediction_path)
    producer = read_run(ROOT, prediction_path)
    if (producer.get("kind") not in ("learning_experiments.predict", "learning_experiments.boost_predict",
                                    "learning_experiments.ensemble", "depth_student.predict")
            or producer.get("group") != request.group):
        raise ValueError("Expected a completed research prediction producer for the requested group")
    partition = producer.get("episode_partition")
    pairs = PreparedPairs(project_path(ROOT, request.dataset), partition, request.group, np)
    if (producer.get("dataset_sha256") != pairs.dataset_sha256
            or producer.get("data_contract") != pairs.data_contract):
        raise ValueError("Prediction dataset hash or public data contract differs from prepared pairs")
    expected_input_valid = pinhole_input_validity(pairs.data_contract, cv2, np)
    inputs = producer.get("inputs")
    if not isinstance(inputs, list) or any(not isinstance(entry, dict) for entry in inputs):
        raise ValueError("Prediction producer has no input hash inventory")
    dataset_entries = [entry for entry in inputs if isinstance(entry.get("path"), str)
                       and project_path(ROOT, entry["path"]) == pairs.dataset_path]
    if not dataset_entries or any(entry.get("sha256") != pairs.dataset_sha256 for entry in dataset_entries):
        raise ValueError("Prediction input inventory does not bind the supplied dataset path/hash")
    frames = producer.get("frames")
    artifacts = producer.get("artifacts")
    if not isinstance(frames, list) or not isinstance(artifacts, dict):
        raise ValueError("Prediction producer requires frame and artifact inventories")
    expected = {_key(row): row for row in pairs.rows}
    frame_map, paths, file_ids = {}, {}, set()
    for frame in frames:
        key = _key(frame)
        if key in frame_map or key not in expected or frame.get("status") != "completed":
            raise ValueError("Prediction frames are duplicated, incomplete or outside the dataset group")
        row = expected[key]
        if (frame.get("sample_sha256") != row["sha256"]
                or any(field not in frame or field not in row or frame[field] != row[field]
                       for field in ("video_pts", "video_pts_seconds"))):
            raise ValueError("Prediction frame sample hash or timestamps differ from prepared data")
        digest = frame.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or artifacts.get(frame.get("artifact")) != digest):
            raise ValueError("Prediction frame hash is not bound to its producer artifacts")
        path = _prediction_path(prediction_path.parent, frame)
        stat = path.stat()
        file_id = (stat.st_dev, stat.st_ino)
        if file_id in file_ids or sha256_file(path) != digest:
            raise ValueError("Prediction artifact is aliased or differs from its recorded hash")
        file_ids.add(file_id)
        frame_map[key], paths[key] = frame, path
    if set(frame_map) != set(expected):
        raise ValueError("Scoring requires EXACT complete dataset-group frame coverage, not a max_samples prefix")
    # The current producer spells this completed_frames; accept complete_frames
    # from the public requested contract, rejecting contradictory declarations.
    counts = [producer[name] for name in ("completed_frames", "complete_frames") if name in producer]
    if not counts or any(type(count) is not int or count != len(expected) for count in counts):
        raise ValueError("Prediction completed-frame count differs from complete dataset coverage")
    if sha256_file(prediction_path) != prediction_hash:
        raise ValueError("Prediction run changed during admission")

    with operation_run("score", request,
                       inputs=[*pairs.input_paths, *(paths[_key(row)] for row in pairs.rows)],
                       parents=[prediction_path, pairs.producer_path]) as (directory, record):
        record.update(dataset=pairs.dataset_path.relative_to(ROOT).as_posix(),
                      dataset_sha256=pairs.dataset_sha256,
                      prediction_run=prediction_path.relative_to(ROOT).as_posix(),
                      prediction_run_sha256=prediction_hash, group=request.group,
                      episode_partition=partition, data_contract=pairs.data_contract,
                      metric="log_depth_mae", metric_units="dimensionless natural-log depth ratio",
                      target_units=pairs.document["target_units"],
                      teacher_accuracy_warning="MoGe2 pseudo-label accuracy is not established; targets are not measured ground truth",
                      aggregation="Pixel-weighted MAE per episode, then equal-episode mean",
                      minimum_prediction_fraction=request.minimum_prediction_fraction, score_value=None,
                      episode_metrics=[], sample_metrics=[], support_fraction=None,
                      complete_frames=len(expected), final_holdout_used_for_selection=False,
                      validation_run=True, official_evaluation=False, tests_run=False,
                      model_execution=False, prediction_executed=False, training_executed=False)
        accumulators = {episode: {"target_pixels": 0, "supported_pixels": 0,
                                  "absolute_log_error_sum": 0.0, "frames": 0}
                        for episode in partition[f"{request.group}_episodes"]}
        zero_samples = []
        for index in range(len(pairs)):
            sample = pairs[index]
            row, target = sample["row"], sample["depth_m"]
            key = _key(row)
            frame = frame_map[key]
            path = _prediction_path(prediction_path.parent, frame)
            if path != paths[key]:
                raise ValueError("Prediction sample path changed after admission")
            with path.open("rb") as stream:
                if _digest(stream) != frame["sha256"]:
                    raise ValueError("Prediction artifact changed after admission")
                with np.load(stream, allow_pickle=False) as arrays:
                    if not {"depth_m", "valid", "input_valid", "input_intrinsics_pixels"}.issubset(arrays.files):
                        raise ValueError("Prediction artifact lacks required public arrays")
                    depth = arrays["depth_m"].copy()
                    valid = arrays["valid"].copy()
                    input_valid = arrays["input_valid"].copy()
                    intrinsics = arrays["input_intrinsics_pixels"].copy()
                if _digest(stream) != frame["sha256"]:
                    raise ValueError("Prediction artifact changed during decoding")
            if (depth.shape != target.shape or depth.dtype.kind != "f"
                    or valid.shape != target.shape or valid.dtype != np.dtype("bool")
                    or input_valid.shape != target.shape or input_valid.dtype != np.dtype("bool")
                    or intrinsics.shape != (3, 3) or intrinsics.dtype.kind != "f"
                    or not np.isfinite(intrinsics).all()
                    or not np.allclose(intrinsics, sample["input_intrinsics_pixels"], rtol=1e-5, atol=1e-5)):
                raise ValueError("Prediction dtype, shape or optical geometry differs from prepared targets")
            if not np.array_equal(input_valid, expected_input_valid) or np.any(valid & ~input_valid):
                raise ValueError("Prediction support differs from calibrated RGB input rays")
            if np.any(valid & (~np.isfinite(depth) | (depth <= 0))):
                raise ValueError("Prediction declares nonfinite or nonpositive depth valid")
            target_valid = sample["valid"] & np.isfinite(target) & (target > 0)
            # Promote stored depths before comparing Python scalar bounds.
            depth64 = depth.astype(np.float64)
            supported = (target_valid & valid & np.isfinite(depth64)
                         & (depth64 >= pairs.data_contract["minimum_depth_m"])
                         & (depth64 <= pairs.data_contract["maximum_depth_m"]))
            target_count, supported_count = int(target_valid.sum()), int(supported.sum())
            errors = np.abs(np.log(depth[supported].astype(np.float64))
                            - np.log(target[supported].astype(np.float64)))
            error_sum = float(errors.sum(dtype=np.float64))
            if not math.isfinite(error_sum):
                raise ValueError("Nonfinite absolute log-depth error")
            fraction = supported_count / target_count if target_count else 0.0
            record["sample_metrics"].append({
                "episode_index": key[0], "frame_index": key[1],
                "target_pixels": target_count, "supported_pixels": supported_count,
                "missing_supported_pixels": target_count - supported_count,
                "support_fraction": fraction, "absolute_log_error_sum": error_sum,
                "log_depth_mae": error_sum / supported_count if supported_count else None})
            if not supported_count:
                zero_samples.append(key)
            totals = accumulators[key[0]]
            totals["target_pixels"] += target_count
            totals["supported_pixels"] += supported_count
            totals["absolute_log_error_sum"] += error_sum
            totals["frames"] += 1
        for episode, totals in sorted(accumulators.items()):
            supported_count, target_count = totals["supported_pixels"], totals["target_pixels"]
            record["episode_metrics"].append({
                "episode_index": episode, **totals,
                "missing_supported_pixels": target_count - supported_count,
                "support_fraction": supported_count / target_count if target_count else 0.0,
                "log_depth_mae": totals["absolute_log_error_sum"] / supported_count if supported_count else None})
        target_count = sum(t["target_pixels"] for t in accumulators.values())
        supported_count = sum(t["supported_pixels"] for t in accumulators.values())
        record.update(target_pixels=target_count, supported_pixels=supported_count,
                      missing_supported_pixels=target_count - supported_count,
                      support_fraction=supported_count / target_count if target_count else 0.0)
        if (zero_samples or not supported_count
                or record["support_fraction"] < request.minimum_prediction_fraction
                or any(metric["support_fraction"] < request.minimum_prediction_fraction
                       for metric in record["episode_metrics"])):
            raise ValueError("Insufficient prediction support; coverage is recorded and no score is admitted")
        if sha256_file(prediction_path) != prediction_hash:
            raise ValueError("Prediction run changed during scoring")
        score = math.fsum(metric["log_depth_mae"] for metric in record["episode_metrics"]) / len(accumulators)
        if not math.isfinite(score):
            raise ValueError("Equal-episode log-depth MAE is nonfinite")
        record["score_value"] = float(score)
    return directory / "run.json"
