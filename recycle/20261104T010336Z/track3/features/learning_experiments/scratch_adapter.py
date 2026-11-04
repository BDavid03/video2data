"""Adapt completed original scratch-CNN predictions to the research scoring format.

No network or checkpoint is loaded. The original scratch training and prediction
producers remain explicit parents; no research checkpoint is manufactured.
Only predictions on the exact prepared optical grid can be copied. Training
must cover exactly the declared training episodes, excluding tuning and holdout.
Third-party imports and writes require the existing research prediction gates.
"""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import PurePosixPath

from track3.paths import project_path
from track3.provenance import input_file, read_run, sha256_file, write_json
from ._common import ROOT, authorize, episode_partition, heavy_execution, operation_run, read_json


@dataclass(frozen=True)
class ScratchAdapterRequest:
    dataset: str | None = None
    prediction_runs: list[str] = field(default_factory=list)
    group: str = "tuning"
    train_episodes: tuple[int, ...] = (0, 11, 12, 18, 21, 23, 26, 31)
    tuning_episodes: tuple[int, ...] = (39, 42)
    holdout_episodes: tuple[int, ...] = (43, 44)

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.dataset is not None and (not isinstance(self.dataset, str) or not self.dataset.strip()):
            raise ValueError("dataset must name an explicit prepared dataset.json")
        if (not isinstance(self.prediction_runs, list) or not 1 <= len(self.prediction_runs) <= 12
                or any(not isinstance(path, str) or not path.strip() for path in self.prediction_runs)):
            raise ValueError("Supply 1–12 completed original scratch prediction run paths")
        if self.group not in ("train", "tuning", "holdout"):
            raise ValueError("group must be train, tuning or holdout")


def _owned_run(path, operation):
    record = read_run(ROOT, path, kind=f"scratch_depth.{operation}")
    if (path.name != "run.json" or path.parent.parent != ROOT / "outputs/scratch_depth" / operation
            or path.parent.name != record.get("run_id")
            or record.get("feature") != "scratch_depth" or record.get("operation") != operation):
        raise ValueError("Scratch producer must remain in its canonical owning run directory")
    return record


def _artifact(directory, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Scratch artifact requires a confined relative POSIX path")
    path = project_path(ROOT, directory / name)
    if path != directory / name or not path.is_relative_to(directory) or path.suffix != ".npz":
        raise ValueError("Scratch artifact is redirected or outside its owning directory")
    return path


def _digest(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def adapt_scratch_predictions(request: ScratchAdapterRequest):
    """Return a standard completed research prediction run for score/ensemble use."""
    authorize("predict")
    authorize("model")
    if request.group == "holdout":
        authorize("holdout_predict")
    if request.dataset is None:
        raise ValueError("Supply the prepared scoring dataset.json")
    partition = episode_partition(request.train_episodes, request.tuning_episodes, request.holdout_episodes)
    heavy_execution("predict")
    import cv2
    import numpy as np
    from .data import PreparedPairs, pinhole_input_validity

    pairs = PreparedPairs(project_path(ROOT, request.dataset), partition, request.group, np)
    expected_input_valid = pinhole_input_validity(pairs.data_contract, cv2, np)
    expected = {(row["episode_index"], row["frame_index"]): row for row in pairs.rows}
    teacher_videos = {teacher["episode_index"]: teacher["video_sha256"]
                      for teacher in pairs.document["teachers"]}
    paths = [project_path(ROOT, name) for name in request.prediction_runs]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate scratch prediction producers are refused")
    identities = {pairs.dataset_path: input_file(ROOT, pairs.dataset_path),
                  pairs.producer_path: input_file(ROOT, pairs.producer_path)}
    parents, selected = {pairs.producer_path}, {}
    checkpoint_identity = None
    training_path = training = training_pairs = None
    for path in paths:
        identities[path] = input_file(ROOT, path)
        producer = _owned_run(path, "predict")
        parents.add(path)
        episode = producer.get("episode_index")
        if type(episode) is not int or episode not in partition[f"{request.group}_episodes"]:
            raise ValueError("Scratch prediction episode is outside the requested group")
        source_training = project_path(ROOT, producer["training_run"])
        checkpoint = project_path(ROOT, producer["checkpoint"])
        identity = (source_training, checkpoint, producer.get("checkpoint_sha256"))
        if checkpoint_identity is None:
            checkpoint_identity = identity
            training_path = source_training
            identities[training_path] = input_file(ROOT, training_path)
            training = _owned_run(training_path, "train")
            parents.add(training_path)
            training_pairs = PreparedPairs(project_path(ROOT, training["dataset"]), partition, "train", np)
            if (training_pairs.data_contract != pairs.data_contract
                    or training.get("dataset_sha256") != training_pairs.dataset_sha256
                    or project_path(ROOT, training["request"]["dataset"]) != training_pairs.dataset_path
                    or training.get("training_samples") != len(training_pairs.document["samples"])
                    or training.get("data_contract") != pairs.data_contract
                    or any(row["episode_index"] not in partition["train_episodes"]
                           for row in training_pairs.document["samples"])):
                raise ValueError("Scratch training must use only the declared training episodes and scoring grid")
            parents.add(training_pairs.producer_path)
            identities[training_pairs.dataset_path] = input_file(ROOT, training_pairs.dataset_path)
            identities[training_pairs.producer_path] = input_file(ROOT, training_pairs.producer_path)
            checkpoint_name = checkpoint.relative_to(training_path.parent).as_posix()
            entries = [entry for entry in training.get("checkpoints", [])
                       if entry.get("artifact") == checkpoint_name and entry.get("sha256") == identity[2]]
            if (checkpoint.parent != training_path.parent / "checkpoints" or len(entries) != 1
                    or training.get("artifacts", {}).get(checkpoint_name) != identity[2]
                    or sha256_file(checkpoint) != identity[2]):
                raise ValueError("Scratch checkpoint is not owned by its declared completed training run")
            identities[checkpoint] = input_file(ROOT, checkpoint)
        elif identity != checkpoint_identity:
            raise ValueError("Every adapted episode must use the same original scratch checkpoint")
        contract = pairs.data_contract
        if (producer.get("training_run_sha256") != identities[training_path]["sha256"]
                or producer.get("model_source") != training.get("model_source")
                or producer.get("checkpoint_epoch") != entries[0]["epoch"]
                or producer.get("checkpoint_optimizer_steps") != entries[0]["optimizer_steps"]
                or producer.get("source_video_sha256") != teacher_videos.get(episode)
                or producer.get("split") != "public" or producer.get("camera") != "ego_cam_a"
                or producer.get("preprocessing") != contract["geometry"]
                or any(producer.get(key) != contract[key] for key in
                       ("dataset_revision", "target_kind", "minimum_depth_m", "maximum_depth_m"))):
            raise ValueError("Scratch prediction lineage or optical/depth contract differs from scoring data")
        frames = producer.get("frames")
        if (not isinstance(frames, list) or type(producer.get("completed_frames")) is not int
                or producer["completed_frames"] != len(frames)
                or any(not isinstance(frame, dict) or frame.get("status") != "completed" for frame in frames)):
            raise ValueError("Scratch prediction requires a complete frame inventory")
        seen = set()
        for frame in frames:
            index = frame.get("frame_index")
            if type(index) is not int or index < 0 or index in seen:
                raise ValueError("Scratch frame indices must be unique nonnegative integers")
            seen.add(index)
            key = (episode, index)
            if key not in expected:
                continue
            row = expected[key]
            if key in selected or any(frame.get(name) != row[name] for name in ("video_pts", "video_pts_seconds")):
                raise ValueError("Scratch frame timestamps or identities differ from prepared samples")
            if frame.get("artifact") != f"frames/frame_{index:06d}.npz":
                raise ValueError("Scratch frames must use their canonical owning artifact names")
            artifact = _artifact(path.parent, frame.get("artifact"))
            digest = frame.get("sha256")
            if (producer.get("artifacts", {}).get(frame["artifact"]) != digest
                    or sha256_file(artifact) != digest):
                raise ValueError("Scratch frame differs from its owning prediction artifact inventory")
            identities[artifact] = input_file(ROOT, artifact)
            selected[key] = (artifact, digest)
    if set(selected) != set(expected):
        raise ValueError("Scratch predictions must cover every prepared row of the requested group")

    def unchanged():
        if any(input_file(ROOT, path) != entry for path, entry in identities.items()):
            raise ValueError("Scratch source lineage changed during adaptation")

    unchanged()
    with operation_run("predict", request, inputs=[*pairs.input_paths, *identities],
                       parents=sorted(parents)) as (directory, record):
        record.update(technique="coursework_scratch_cnn", model_config=training["model_config"],
                      episode_partition=partition, data_contract=pairs.data_contract,
                      dataset=pairs.dataset_path.relative_to(ROOT).as_posix(), dataset_sha256=pairs.dataset_sha256,
                      group=request.group, source_prediction_runs=[identities[path] for path in paths],
                      scratch_training_run=identities[training_path],
                      scratch_checkpoint=identities[checkpoint_identity[1]],
                      checkpoint_sha256=checkpoint_identity[2],
                      checkpoint_epoch=entries[0]["epoch"],
                      checkpoint_optimizer_steps=entries[0]["optimizer_steps"],
                      training_dataset_sha256=training_pairs.dataset_sha256,
                      total_group_samples=len(pairs), selected_samples=len(pairs), completed_frames=0, frames=[],
                      sample_selection="Exact prepared group rows matched by episode, frame and video timestamps",
                      adapter="Completed original scratch prediction arrays; no research checkpoint conversion",
                      model_execution=False, prediction_executed=False, training_executed=False,
                      source_prediction_executed=True, targets_used_for_prediction=False,
                      final_holdout_used_for_selection=False, scratch_checkpoint_admitted=True)
        (directory / "frames").mkdir()
        for row in pairs.rows:
            key = (row["episode_index"], row["frame_index"])
            source, digest = selected[key]
            with source.open("rb") as stream:
                if _digest(stream) != digest:
                    raise ValueError("Scratch prediction changed before decoding")
                with np.load(stream, allow_pickle=False) as arrays:
                    depth, valid, input_valid, intrinsics = (
                        arrays[name].copy() for name in
                        ("depth_m", "valid", "input_valid", "input_intrinsics_pixels"))
                if _digest(stream) != digest:
                    raise ValueError("Scratch prediction changed during decoding")
            width, height = pairs.data_contract["geometry"]["output_size_width_height"]
            if (depth.shape != (height, width) or depth.dtype.kind != "f"
                    or valid.shape != depth.shape or valid.dtype != np.dtype("bool")
                    or input_valid.shape != depth.shape or input_valid.dtype != np.dtype("bool")
                    or intrinsics.shape != (3, 3) or intrinsics.dtype.kind != "f"
                    or not np.isfinite(intrinsics).all()
                    or not np.allclose(intrinsics, pairs.data_contract["geometry"]["input_intrinsics_pixels"],
                                       rtol=1e-5, atol=1e-5)
                    or not np.array_equal(input_valid, expected_input_valid)
                    or np.any(valid & (~input_valid | ~np.isfinite(depth) | (depth <= 0)))):
                raise ValueError("Scratch prediction arrays differ from calibrated prepared RGB geometry")
            frame = {name: row[name] for name in
                     ("episode_index", "frame_index", "video_pts", "video_pts_seconds")}
            frame.update(sample_sha256=row["sha256"], status="running", source_prediction=identities[source])
            record["frames"].append(frame)
            destination = directory / "frames" / f"episode_{key[0]:06d}_frame_{key[1]:06d}.npz"
            temporary = destination.with_suffix(".npz.tmp")
            try:
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, depth_m=depth, valid=valid, input_valid=input_valid,
                                        input_intrinsics_pixels=intrinsics)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            frame.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                         sha256=sha256_file(destination), valid_pixels=int(valid.sum()))
            record["completed_frames"] += 1
            write_json(directory / "run.json", record)
        unchanged()
    return directory / "run.json"
