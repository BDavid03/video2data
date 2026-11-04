"""Predict every row of a prepared dataset group with a student checkpoint; NPZ per frame.

Output arrays match learning_experiments.predict (depth_m, valid, input_valid,
input_intrinsics_pixels) so the shared score and ensemble operations admit them.
Optional horizontal-flip test-time augmentation averages in log space.
"""

from dataclasses import dataclass
import os

from track3.gpu import GpuLease
from track3.paths import project_path
from track3.provenance import sha256_file, write_json
from ._common import ROOT, authorize, heavy_execution, operation_run, read_json


@dataclass(frozen=True)
class PredictStudentRequest:
    dataset: str | None = None
    checkpoint: str | None = None
    group: str = "tuning"
    flip_tta: bool = False

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.group not in ("train", "tuning", "holdout"):
            raise ValueError("group must be train, tuning or holdout")
        if type(self.flip_tta) is not bool:
            raise ValueError("flip_tta must be boolean")


def load_checkpoint(path, torch):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "depth_student.checkpoint" or payload.get("pretrained_weights") is not False:
        raise ValueError("Expected an owned depth_student checkpoint without pretrained weights")
    return payload


def predict_student(request: PredictStudentRequest):
    runtime = authorize("predict")
    if request.group == "holdout":
        authorize("holdout_predict")
    if request.dataset is None or request.checkpoint is None:
        raise ValueError("Supply a prepared dataset.json and a depth_student checkpoint")
    heavy_execution("predict")
    import cv2
    import numpy as np
    import torch
    from track3.features.learning_experiments.data import PreparedPairs, pinhole_input_validity
    from ._models import build

    checkpoint_path = project_path(ROOT, request.checkpoint)
    payload = load_checkpoint(checkpoint_path, torch)
    pairs = PreparedPairs(project_path(ROOT, request.dataset), payload["episode_partition"], request.group, np)
    if pairs.data_contract != payload["data_contract"]:
        raise ValueError("Prediction geometry, targets and calibration differ from training")
    input_valid = pinhole_input_validity(pairs.data_contract, cv2, np)
    training_path = checkpoint_path.parent.parent / "run.json"
    with operation_run("predict", request, inputs=[checkpoint_path, *pairs.input_paths],
                       parents=[training_path, pairs.producer_path]) as (directory, record):
        record.update(arch=payload["arch"], architecture=payload["architecture"], episode_partition=payload["episode_partition"],
                      data_contract=pairs.data_contract, dataset_sha256=pairs.dataset_sha256, group=request.group,
                      checkpoint_sha256=sha256_file(checkpoint_path), weights_are_ema=payload.get("weights_are_ema"),
                      prediction_executed=True, training_executed=False, targets_used_for_prediction=False,
                      total_group_samples=len(pairs), completed_frames=0, frames=[], flip_tta=request.flip_tta)
        (directory / "frames").mkdir()
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"], stage="depth_student_predict")
        with lease:
            torch.cuda.set_device(0)
            model = build(payload["arch"], payload["architecture"])
            model.load_state_dict(payload["state_dict"], strict=True)
            model = model.cuda().eval()
            for index in range(len(pairs)):
                sample = pairs[index]
                row = sample["row"]
                frame = {"episode_index": row["episode_index"], "frame_index": row["frame_index"], "video_pts": row["video_pts"],
                         "video_pts_seconds": row["video_pts_seconds"], "sample_sha256": row["sha256"], "status": "running"}
                record["frames"].append(frame)
                lease.require_headroom()
                image = torch.from_numpy(sample["rgb"]).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0
                with torch.inference_mode():
                    log_depth = model(image).float().log()
                    if request.flip_tta:
                        log_depth = 0.5 * (log_depth + model(image.flip(-1)).float().log().flip(-1))
                depth = log_depth.exp()[0, 0].cpu().numpy()
                depth64 = depth.astype(np.float64)
                valid = (input_valid & np.isfinite(depth64) & (depth64 > 0)
                         & (depth64 >= pairs.data_contract["minimum_depth_m"]) & (depth64 <= pairs.data_contract["maximum_depth_m"]))
                destination = directory / "frames" / f"episode_{row['episode_index']:06d}_frame_{row['frame_index']:06d}.npz"
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, depth_m=np.where(valid, depth, np.nan).astype(np.float32), valid=valid,
                                        input_valid=input_valid, input_intrinsics_pixels=sample["input_intrinsics_pixels"])
                os.replace(temporary, destination)
                frame.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                             sha256=sha256_file(destination), valid_pixels=int(valid.sum()))
                record["completed_frames"] += 1
            torch.cuda.synchronize(0)
            write_json(directory / "run.json", record)
            model = None
            torch.cuda.empty_cache()
        record["gpu"] = lease.report()
    return directory / "run.json"
