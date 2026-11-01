"""Generate calibrated per-frame depth using a locally trained scratch checkpoint."""

from dataclasses import dataclass

from track3.gpu import GpuLease
from track3.provenance import read_run, sha256_file, write_json
from ._common import (ROOT, authorize, local_path, open_video, operation_run,
                      pinhole, positive_integer, read_json, relative, rgb_frame, save_npz, select_episode)
from ._checkpoint import owned_checkpoint, require_finite_state, require_payload


@dataclass(frozen=True)
class PredictDepthRequest:
    checkpoint: str | None = None
    episode_index: int = 39
    start_frame: int = 0
    stop_frame: int | None = None
    frame_stride: int = 1

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.checkpoint is not None and (not isinstance(self.checkpoint, str) or not self.checkpoint.strip()):
            raise ValueError("checkpoint must name an explicit local trained checkpoint")
        positive_integer(self.episode_index, "episode_index", 0)
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded from scratch-depth execution")
        positive_integer(self.start_frame, "start_frame", 0)
        positive_integer(self.frame_stride, "frame_stride")
        if self.stop_frame is not None:
            positive_integer(self.stop_frame, "stop_frame", self.start_frame + 1)


def predict_depth(request: PredictDepthRequest):
    runtime = authorize("predict")
    if request.checkpoint is None:
        raise ValueError("Supply a checkpoint from a completed scratch-depth training operation")
    import cv2
    import numpy as np
    import torch
    from ._network import CourseworkDepthCNN

    checkpoint_path = local_path(request.checkpoint)
    training_directory = checkpoint_path.parent.parent
    training = read_run(ROOT, training_directory / "run.json", kind="scratch_depth.train")
    if (training.get("feature") != "scratch_depth" or training.get("operation") != "train"
            or training.get("status") != "completed"):
        raise ValueError("Checkpoint requires a completed scratch training run")
    entry, checkpoint_hash = owned_checkpoint(checkpoint_path, training)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require_payload(checkpoint, training, entry, training_directory)
    require_finite_state(checkpoint["state_dict"], torch)
    inputs = select_episode(request.episode_index)
    if (checkpoint["dataset_revision"] != inputs["revision"] or checkpoint["split_id"] != inputs["split_id"]
            or checkpoint["calibration_sha256"] != sha256_file(inputs["calibration"])):
        raise ValueError("Checkpoint and prediction inputs use different dataset/calibration versions")
    stop = inputs["clock"]["decoded_frame_count"] if request.stop_frame is None else request.stop_frame
    if not request.start_frame < stop <= inputs["clock"]["decoded_frame_count"]:
        raise ValueError("Requested frames are outside the inventoried RGB video")
    maps = pinhole(inputs["camera"], checkpoint["max_image_side"], cv2, np)
    x, y, input_valid, k, geometry = maps
    if geometry != checkpoint["geometry"]:
        raise ValueError("Prediction preprocessing differs from the training camera geometry")
    with operation_run("predict", request, inputs=[checkpoint_path, inputs["video"], inputs["calibration"]],
                       parents=[training_directory / "run.json"]) as (directory, record):
        (directory / "frames").mkdir()
        frames = [{"frame_index": i, "status": "not_processed",
                   "video_pts": inputs["clock"]["pts_integer"][i],
                   "video_pts_seconds": inputs["clock"]["pts_seconds"][i]}
                  for i in range(request.start_frame, stop, request.frame_stride)]
        selected = {frame["frame_index"]: frame for frame in frames}
        record.update(
            split="public", episode_index=request.episode_index, camera="ego_cam_a",
            dataset_revision=inputs["revision"], checkpoint=relative(checkpoint_path), checkpoint_sha256=checkpoint_hash,
            checkpoint_schema_version=checkpoint["schema_version"], checkpoint_epoch=checkpoint["trained_epochs"],
            checkpoint_optimizer_steps=checkpoint["optimizer_steps"], model_source=checkpoint["model_source"],
            training_optimization=checkpoint.get("optimization"),
            training_optimization_sources=checkpoint.get("optimization_sources"),
            training_run=relative(training_directory / "run.json"), training_run_sha256=sha256_file(training_directory / "run.json"),
            source_video=relative(inputs["video"]), source_video_sha256=sha256_file(inputs["video"]),
            source_time_base_and_frame_rate=inputs["clock"]["time_base_and_frame_rate"],
            preprocessing=geometry, frames=frames, completed_frames=0,
            depth_units="metres inherited from MoGe2 pseudo-targets; metric accuracy not established",
            depth_kind="per-frame monocular CNN prediction", target_kind=checkpoint["target_kind"],
            minimum_depth_m=checkpoint["minimum_depth_m"], maximum_depth_m=checkpoint["maximum_depth_m"],
            validity="In-bounds input rays and finite prediction inside the training target range; no learned confidence",
            invalid_depth_encoding="NaN with explicit valid=false", world_T_camera=None,
            interpolation="No temporal interpolation", object_and_hand_poses_read=False,
            other_clocks="Parquet and capture_time clocks are not read or inferred",
        )
        save_npz(directory / "preprocessing.npz", np, source_x=x, source_y=y,
                 input_valid=input_valid, input_intrinsics_pixels=k.astype(np.float32))
        record["preprocessing_artifact_sha256"] = sha256_file(directory / "preprocessing.npz")
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                         stage="scratch_depth_predict")
        capture = model = image = prediction = current = None
        try:
            capture = open_video(inputs, cv2)
            with lease:
                try:
                    torch.cuda.set_device(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    model = CourseworkDepthCNN(**checkpoint["model_config"])
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model = model.to("cuda:0").eval()
                    lease.require_headroom()
                    for index in range(stop):
                        if index not in selected:
                            if not capture.grab():
                                raise RuntimeError(f"RGB decode stopped at frame {index}")
                            continue
                        current = selected[index]
                        current["status"] = "running"
                        rgb = rgb_frame(capture, inputs, maps, cv2)
                        lease.require_headroom()
                        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
                        image = image.to("cuda:0", dtype=torch.float32) / 255.0
                        with torch.inference_mode():
                            prediction = model(image)
                        depth = prediction[0, 0].float().cpu().numpy()
                        depth_for_bounds = depth.astype(np.float64)
                        valid = (input_valid & np.isfinite(depth_for_bounds) & (depth_for_bounds > 0)
                                 & (depth_for_bounds >= checkpoint["minimum_depth_m"])
                                 & (depth_for_bounds <= checkpoint["maximum_depth_m"]))
                        artifact = directory / "frames" / f"frame_{index:06d}.npz"
                        save_npz(artifact, np, depth_m=np.where(valid, depth, np.nan).astype(np.float32),
                                 valid=valid, input_valid=input_valid, input_intrinsics_pixels=k.astype(np.float32))
                        current.update(status="completed", artifact=artifact.relative_to(directory).as_posix(),
                                       sha256=sha256_file(artifact), valid_pixels=int(valid.sum()), total_pixels=int(valid.size))
                        record["completed_frames"] += 1
                        write_json(directory / "run.json", record)
                        current = None
                        image = prediction = None
                        lease.require_headroom()
                    torch.cuda.synchronize(0)
                finally:
                    if torch.cuda.is_initialized():
                        record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                        model = image = prediction = None
                        torch.cuda.empty_cache()
        except BaseException as error:
            if current is not None:
                current.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if capture is not None:
                capture.release()
            record["gpu"] = lease.report()
    return directory / "run.json"
