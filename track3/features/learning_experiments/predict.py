"""Persist research predictions from an owned trained checkpoint and paired RGB."""

from dataclasses import dataclass
import os

from track3.gpu import GpuLease
from track3.paths import project_path
from track3.provenance import sha256_file, write_json
from ._common import ROOT, authorize, heavy_execution, integer, operation_run, read_json
from .train import require_model_geometry


@dataclass(frozen=True)
class PredictRequest:
    dataset: str | None = None
    checkpoint: str | None = None
    group: str = "tuning"
    max_samples: int = 128

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        for name, value in (("dataset", self.dataset), ("checkpoint", self.checkpoint)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be an explicit local path")
        if self.group not in ("train", "tuning", "holdout"):
            raise ValueError("group must be train, tuning or holdout")
        integer(self.max_samples, "max_samples", 1, 10000)


def predict_model(request: PredictRequest):
    runtime = authorize("predict")
    authorize("model")
    if request.group == "holdout":
        authorize("holdout_predict")
    if request.dataset is None or request.checkpoint is None:
        raise ValueError("Supply a prepared dataset and an owned trained research checkpoint")
    heavy_execution("predict")
    import cv2
    import numpy as np
    import torch
    from .data import PreparedPairs, pinhole_input_validity
    from .checkpoints import load_checkpoint
    from .models import ModelRequest, create_model

    checkpoint_path = project_path(ROOT, request.checkpoint)
    payload, training, entry = load_checkpoint(checkpoint_path, torch)
    pairs = PreparedPairs(project_path(ROOT, request.dataset), payload["episode_partition"], request.group, np)
    if pairs.data_contract != payload["data_contract"]:
        raise ValueError("Prediction geometry, targets and calibration differ from training")
    require_model_geometry(payload["model_config"], payload["technique"], pairs.document["geometry"])
    input_valid = pinhole_input_validity(pairs.data_contract, cv2, np)
    training_path = checkpoint_path.parent.parent / "run.json"
    selected = list(range(min(len(pairs), request.max_samples)))
    with operation_run("predict", request, inputs=[checkpoint_path, *pairs.input_paths],
                       parents=[training_path, pairs.producer_path]) as (directory, record):
        record.update(technique=payload["technique"], model_config=payload["model_config"],
                      episode_partition=payload["episode_partition"], data_contract=pairs.data_contract,
                      dataset_sha256=pairs.dataset_sha256, group=request.group,
                      checkpoint_sha256=sha256_file(checkpoint_path), checkpoint_epoch=entry["epoch"],
                      checkpoint_optimizer_steps=entry["optimizer_steps"],
                      prediction_executed=True, training_executed=False,
                      targets_available_in_pairs=True, targets_used_for_prediction=False,
                      total_group_samples=len(pairs), selected_samples=len(selected),
                      sample_selection="First max_samples in producer order, no random subsampling",
                      completed_frames=0, frames=[],
                      accuracy_note="No score or final-holdout selection is performed")
        record["validity"] = "In-bounds calibrated RGB ray and finite prediction inside training target bounds"
        (directory / "frames").mkdir()
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                         stage="learning_experiments_predict")
        model = image = prediction = None
        try:
            with lease:
                try:
                    torch.cuda.set_device(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    partition = payload["episode_partition"]
                    model = create_model(ModelRequest(
                        technique=payload["technique"], seed=training["request"]["seed"],
                        parameters=payload["model_config"],
                        train_episodes=tuple(partition["train_episodes"]),
                        tuning_episodes=tuple(partition["tuning_episodes"]),
                        holdout_episodes=tuple(partition["holdout_episodes"])), operation="predict")
                    model.load_state_dict(payload["state_dict"], strict=True)
                    payload = None
                    model = model.to("cuda:0").eval()
                    for index in selected:
                        sample = pairs[index]
                        row = sample["row"]
                        frame = {"episode_index": row["episode_index"], "frame_index": row["frame_index"],
                                 "video_pts": row["video_pts"], "video_pts_seconds": row["video_pts_seconds"],
                                 "sample_sha256": row["sha256"], "status": "running"}
                        record["frames"].append(frame)
                        lease.require_headroom()
                        image = torch.from_numpy(sample["rgb"]).permute(2, 0, 1).unsqueeze(0)
                        image = image.to("cuda:0", dtype=torch.float32) / 255.0
                        with torch.inference_mode():
                            prediction = model(image, episodes=[row["episode_index"]], operation="predict")
                        depth = prediction[0, 0].float().cpu().numpy()
                        # Compare the stored float32 values without rounding contract bounds to float32.
                        depth_for_bounds = depth.astype(np.float64)
                        valid = (input_valid & np.isfinite(depth_for_bounds) & (depth_for_bounds > 0)
                                 & (depth_for_bounds >= pairs.data_contract["minimum_depth_m"])
                                 & (depth_for_bounds <= pairs.data_contract["maximum_depth_m"]))
                        destination = directory / "frames" / (
                            f"episode_{row['episode_index']:06d}_frame_{row['frame_index']:06d}.npz")
                        temporary = destination.with_suffix(".npz.tmp")
                        try:
                            with temporary.open("xb") as stream:
                                np.savez_compressed(stream, depth_m=np.where(valid, depth, np.nan).astype(np.float32),
                                                    valid=valid, input_valid=input_valid,
                                                    input_intrinsics_pixels=sample["input_intrinsics_pixels"])
                            os.replace(temporary, destination)
                        finally:
                            temporary.unlink(missing_ok=True)
                        frame.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                                     sha256=sha256_file(destination), valid_pixels=int(valid.sum()))
                        record["completed_frames"] += 1
                        write_json(directory / "run.json", record)
                        image = prediction = None
                        lease.require_headroom()
                    torch.cuda.synchronize(0)
                finally:
                    if torch.cuda.is_initialized():
                        record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                        model = image = prediction = None
                        torch.cuda.empty_cache()
        finally:
            record["gpu"] = lease.report()
    return directory / "run.json"
