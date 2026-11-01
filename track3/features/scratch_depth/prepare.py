"""Prepare aligned RGB/pseudo-depth pairs from explicitly selected local MoGe runs."""

from dataclasses import dataclass, field
import math

from track3.provenance import input_file, read_run, sha256_file, write_json
from ._common import (ROOT, authorize, local_path, open_video, operation_run, pinhole,
                      positive_integer, read_json, relative, rgb_frame, save_npz,
                      select_episode, within)


@dataclass(frozen=True)
class PrepareDepthDataRequest:
    teacher_runs: list[str] = field(default_factory=list)
    frame_stride: int = 5
    max_image_side: int = 384
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 20.0

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if not isinstance(self.teacher_runs, list) or any(not isinstance(p, str) for p in self.teacher_runs):
            raise ValueError("teacher_runs must be an explicit list of repository-local run.json paths")
        positive_integer(self.frame_stride, "frame_stride")
        positive_integer(self.max_image_side, "max_image_side", 64, 512)
        if not (math.isfinite(self.minimum_depth_m) and math.isfinite(self.maximum_depth_m)
                and 0 < self.minimum_depth_m < self.maximum_depth_m):
            raise ValueError("Depth bounds must be positive, finite and ordered")


def prepare_depth_data(request: PrepareDepthDataRequest):
    authorize("prepare")
    if not request.teacher_runs:
        raise ValueError("Supply completed local MoGe run.json paths; no teacher output currently exists")
    import cv2
    import numpy as np

    with operation_run("prepare", request, parents=request.teacher_runs) as (directory, record):
        (directory / "samples").mkdir()
        record["frames"] = []
        samples, teachers, rejected, seen = [], [], [], set()
        dataset = {
            "schema_version": 1, "kind": "scratch_depth_training_pairs", "status": "building",
            "source_split": "public", "subset": "development", "camera": "ego_cam_a",
            "target_kind": "moge2_pseudo_depth", "target_units": "estimated metres",
            "target_accuracy": "Not established; pseudo-labels are not measured ground truth",
            "max_image_side": request.max_image_side,
            "minimum_depth_m": request.minimum_depth_m, "maximum_depth_m": request.maximum_depth_m,
            "samples": samples, "teachers": teachers, "rejected_samples": rejected,
            "provenance": record["provenance"], "producer_run": relative(directory / "run.json"),
        }
        for teacher_name in request.teacher_runs:
            teacher_path = local_path(teacher_name)
            teacher = read_run(ROOT, teacher_path, kind="depth.generate")
            if (teacher.get("feature") != "depth" or teacher.get("status") != "completed"
                    or teacher.get("split") != "public" or teacher.get("camera") != "ego_cam_a"):
                raise ValueError("Only completed public RGB MoGe depth runs can supply pseudo-targets")
            inputs = select_episode(teacher["episode_index"])
            record["inputs"] += [input_file(ROOT, inputs[key]) for key in ("video", "calibration")]
            video_hash = sha256_file(inputs["video"])
            calibration_hash = sha256_file(inputs["calibration"])
            if (teacher["dataset_revision"] != inputs["revision"]
                    or local_path(teacher["source_video"]) != inputs["video"]
                    or teacher["input_hashes"]["video_sha256"] != video_hash
                    or teacher["input_hashes"]["calibration_sha256"] != calibration_hash):
                raise ValueError("Teacher inputs differ from the selected pinned RGB/calibration")
            maps = pinhole(inputs["camera"], request.max_image_side, cv2, np)
            x, y, input_valid, k, geometry = maps
            if teachers and geometry != dataset["geometry"]:
                raise ValueError("All training pairs must use the same canonical RGB camera geometry")
            dataset.update(dataset_revision=inputs["revision"], split_id=inputs["split_id"],
                           geometry=geometry, calibration_sha256=calibration_hash)
            frame_rows = {}
            for row in teacher["frames"]:
                index = row["frame_index"]
                positive_integer(index, "teacher frame_index", 0, inputs["clock"]["decoded_frame_count"] - 1)
                if index in frame_rows or row["status"] != "completed":
                    raise ValueError("Teacher frames must be unique and completed")
                if (row["video_pts"] != inputs["clock"]["pts_integer"][index]
                        or row["video_pts_seconds"] != inputs["clock"]["pts_seconds"][index]):
                    raise ValueError("Teacher frame timestamps differ from the pinned video inventory")
                frame_rows[index] = row
            selected = {i: row for i, row in frame_rows.items() if i % request.frame_stride == 0}
            if not selected:
                raise ValueError("Selected teacher run has no frames at the requested stride")
            teachers.append({"run": relative(teacher_path), "sha256": sha256_file(teacher_path),
                             "episode_index": teacher["episode_index"], "model": teacher["model"],
                             "video_sha256": video_hash, "calibration_sha256": calibration_hash})
            capture = open_video(inputs, cv2)
            try:
                for index in range(max(selected) + 1):
                    if index not in selected:
                        if not capture.grab():
                            raise RuntimeError(f"RGB decode stopped before frame {index}")
                        continue
                    key = (teacher["episode_index"], index)
                    if key in seen:
                        raise ValueError(f"Duplicate episode/frame across teacher runs: {key}")
                    seen.add(key)
                    frame_record = {"episode_index": key[0], "frame_index": index, "status": "running",
                                    "video_pts": selected[index]["video_pts"]}
                    record["frames"].append(frame_record)
                    rgb = rgb_frame(capture, inputs, maps, cv2)
                    source = within(teacher_path.parent, selected[index]["artifact"])
                    if sha256_file(source) != selected[index]["sha256"]:
                        raise ValueError("Teacher depth artifact bytes no longer match its manifest")
                    record["inputs"].append(input_file(ROOT, source))
                    with np.load(source, allow_pickle=False) as arrays:
                        depth = arrays["depth_m"].copy()
                        valid = arrays["valid"].astype(bool)
                        teacher_k = arrays["input_intrinsics_pixels"].astype(np.float64)
                    recorded_k = np.asarray(teacher["preprocessing"]["input_pinhole_intrinsics_pixels"])
                    teacher_size = teacher["preprocessing"]["output_size_width_height"]
                    if (depth.shape != (teacher_size[1], teacher_size[0]) or valid.shape != depth.shape
                            or teacher_k.shape != (3, 3) or not np.isfinite(teacher_k).all()
                            or not np.allclose(teacher_k, recorded_k, rtol=1e-5, atol=1e-5)):
                        raise ValueError("Teacher arrays and camera geometry disagree")
                    # Reproject pixel rays between pinhole grids, keeping z-depth unchanged.
                    # Nearest-neighbour labels avoid blending foreground/background or NaNs.
                    u, v = np.meshgrid(np.arange(input_valid.shape[1]), np.arange(input_valid.shape[0]))
                    tx = ((u - k[0, 2]) / k[0, 0] * teacher_k[0, 0] + teacher_k[0, 2]).astype(np.float32)
                    ty = ((v - k[1, 2]) / k[1, 1] * teacher_k[1, 1] + teacher_k[1, 2]).astype(np.float32)
                    safe = np.where(valid & np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
                    target = cv2.remap(safe, tx, ty, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    target_valid = cv2.remap(valid.astype(np.uint8), tx, ty, cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT).astype(bool)
                    target_valid &= (input_valid & np.isfinite(target)
                                     & (target >= request.minimum_depth_m) & (target <= request.maximum_depth_m))
                    if int(target_valid.sum()) < 64:
                        rejected.append({"episode_index": key[0], "frame_index": index,
                                         "reason": "fewer than 64 valid pseudo-target pixels"})
                        frame_record.update(status="completed", accepted=False,
                                            reason="fewer than 64 valid pseudo-target pixels")
                        continue
                    destination = directory / "samples" / f"episode_{key[0]:06d}_frame_{index:06d}.npz"
                    save_npz(destination, np, rgb=rgb, depth_m=np.where(target_valid, target, np.nan),
                             valid=target_valid, input_intrinsics_pixels=k.astype(np.float32))
                    samples.append({"episode_index": key[0], "frame_index": index,
                                    "video_pts": selected[index]["video_pts"],
                                    "video_pts_seconds": selected[index]["video_pts_seconds"],
                                    "artifact": destination.relative_to(directory).as_posix(),
                                    "sha256": sha256_file(destination), "valid_pixels": int(target_valid.sum()),
                                    "teacher_artifact": relative(source), "teacher_sha256": selected[index]["sha256"]})
                    record.update(prepared_samples=len(samples), rejected_samples=len(rejected))
                    frame_record.update(status="completed", accepted=True,
                                        artifact=destination.relative_to(directory).as_posix(),
                                        sha256=samples[-1]["sha256"])
                    write_json(directory / "run.json", record)
            finally:
                capture.release()
        if not samples:
            raise ValueError("No usable training pairs were prepared")
        dataset["status"] = "completed"
        write_json(directory / "dataset.json", dataset)
        record.update(dataset="dataset.json", dataset_sha256=sha256_file(directory / "dataset.json"),
                      target_kind=dataset["target_kind"], prepared_samples=len(samples), rejected_samples=len(rejected))
    return directory / "dataset.json"
