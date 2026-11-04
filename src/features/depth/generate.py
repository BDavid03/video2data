"""Depth request -> local RGB/MoGe2 processing -> explicit per-frame artifacts.

This slice owns its input selection, preprocessing, model adapter, and persistence.
Heavy libraries are loaded only inside an explicitly authorized generation call.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

from src.paths import output_path, project_path
from src.provenance import environment, run_fields, sha256_file, write_json
from src.gpu import GpuLease
from src import runs
from src.runs import ROOT, read_json as _json, save_npz


def _local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class GenerateDepthRequest:
    episode_index: int = 39
    source_split: str = "public"
    start_frame: int = 0
    stop_frame: int | None = None
    max_image_side: int = 768
    num_tokens: int = 1200
    use_fp16: bool = True
    save_normals: bool = False
    gpu_headroom_mib: int = 2048
    gpu_admission_allowance_mib: int = 4096
    frame_stride: int = 1

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation"):
            raise ValueError("source_split must be public or evaluation")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded from depth execution")
        for name in ("episode_index", "start_frame", "frame_stride", "max_image_side", "num_tokens",
                     "gpu_headroom_mib", "gpu_admission_allowance_mib"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("episode_index", "start_frame") else 1):
                raise ValueError(f"Invalid depth request field: {name}")
        if self.stop_frame is not None and (
            type(self.stop_frame) is not int or self.stop_frame <= self.start_frame
        ):
            raise ValueError("stop_frame is exclusive and must follow start_frame")
        if type(self.use_fp16) is not bool or type(self.save_normals) is not bool:
            raise ValueError("Depth precision and normal-output flags must be booleans")
        if not 64 <= self.max_image_side <= 1024 or not 1200 <= self.num_tokens <= 2500:
            raise ValueError("This bounded stage supports image sides 64..1024 and 1200..2500 tokens")


def _select_input(request, sources):
    """Consume explicit inventories and container PTS, never public pose labels."""
    dataset = sources["dataset"]
    episodes = _json("data/manifests/dataset/episodes.json")
    videos = _json("data/reports/data-audit/videos.json")
    split = _json("data/configs/splits/public-v1.json")
    if any(value != dataset["revision"] for value in
           (episodes["revision"], videos["revision"], split["dataset_revision"])):
        raise ValueError("Depth input inventories refer to different dataset revisions")
    if request.source_split == "public":
        if request.episode_index not in (split["development_episode_indices"] +
                                         split["validation_episode_indices"]):
            raise ValueError("Public episode is outside the frozen split")
        if request.episode_index in split["validation_episode_indices"]:
            runtime = _json(Path(__file__).with_name("runtime.json"))
            if runtime["execution"].get("validation_authorized") is not True:
                raise PermissionError("Public validation depth remains separately disabled")
    matches = [item for item in episodes["episodes"]
               if item["split"] == request.source_split and item["episode_index"] == request.episode_index]
    if len(matches) != 1 or matches[0]["issues"]:
        raise ValueError("Depth needs exactly one explicitly inventoried episode in the named split")
    episode = matches[0]
    relative = episode["videos"]["ego_cam_a"]
    clocks = [item for item in videos["videos"] if item["path"] == relative]
    if videos["status"] != "completed" or len(clocks) != 1 or clocks[0]["status"] != "ok":
        raise ValueError("No completed RGB container-timestamp record is available")
    clock = clocks[0]
    count = clock["decoded_frame_count"]
    if len(clock["pts_seconds"]) != count or len(clock["pts_integer"]) != count:
        raise ValueError("RGB timestamp inventory is incomplete")
    stop = count if request.stop_frame is None else request.stop_frame
    if not request.start_frame < stop <= count:
        raise ValueError("Requested frame interval is outside the inventoried episode")
    snapshot = _local_path(Path(dataset["cache_path"]) / "snapshots" / dataset["revision"])
    logical_video = snapshot / relative
    _local_path(logical_video)
    if not logical_video.is_file():
        raise FileNotFoundError(logical_video)
    return episode, clock, snapshot, logical_video, stop


def _configure_local_runtime(local):
    """Interpreter, caches and the pinned MoGe source trees, before any heavy import."""
    runs.heavy_execution(local)
    for relative in reversed(local["source_paths"]):
        path = _local_path(relative)
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))


def _source_snapshot(local):
    """Record source bytes used by the operation, without Git or test commands."""
    entries = []
    roots = [Path(__file__).parent, *(_local_path(path) for path in local["source_paths"])]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if (path.is_file() and path.suffix in (".py", ".json", ".toml") and
                    ".git" not in path.parts and "__pycache__" not in path.parts):
                _local_path(path)
                entries.append({"path": path.relative_to(ROOT).as_posix(),
                                "sha256": sha256_file(path)})
    return entries


def _pinhole_maps(camera, max_side, cv2, np):
    """One remap: factory distorted pixels -> centered square-pixel pinhole."""
    source_width, source_height = camera["resolution"]
    source_k = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    distortion = np.asarray(camera["distortion_coefficients"], dtype=np.float64)
    if camera["distortion_model"] != "Perspective" or distortion.shape != (14,):
        raise ValueError("This feature requires the released 14-coefficient Perspective calibration")
    if (source_k.shape != (3, 3) or not np.isfinite(source_k).all() or
            not np.isfinite(distortion).all() or source_k[0, 0] <= 0 or
            source_k[1, 1] <= 0 or source_width <= 0 or source_height <= 0):
        raise ValueError("Invalid source camera parameters")
    ratio = min(1.0, max_side / max(source_width, source_height))
    width, height = max(1, round(source_width * ratio)), max(1, round(source_height * ratio))
    focal = float(source_k[0, 0]) * width / source_width
    target_k = np.array([[focal, 0, (width - 1) / 2],
                         [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_k, distortion, np.eye(3), target_k, (width, height), cv2.CV_32FC1)
    usable = (np.isfinite(map_x) & np.isfinite(map_y) &
              (map_x >= 0) & (map_x <= source_width - 1) &
              (map_y >= 0) & (map_y <= source_height - 1))
    fov_x = math.degrees(2 * math.atan(width / (2 * focal)))
    metadata = {
        "camera": "ego_cam_a", "camera_frame": "OpenCV optical x-right y-down z-forward",
        "source_size_width_height": [source_width, source_height],
        "output_size_width_height": [width, height],
        "factory_intrinsics_pixels": source_k.tolist(),
        "factory_distortion_coefficients": distortion.tolist(),
        "input_pinhole_intrinsics_pixels": target_k.tolist(),
        "output_distortion": "zero; factory distortion removed once",
        "additional_rotation_degrees": 0, "optical_frame_rotation": "identity",
        "mapping": "Direct remap using factory K,D and centered square-pixel output K",
        "principal_point": "Pixel-center coordinates; centered at ((W-1)/2,(H-1)/2)",
        "fov_x_degrees_supplied_to_model": fov_x,
        "geometry_status": "Derived from provisional factory calibration; not measured depth accuracy",
    }
    return map_x, map_y, usable, target_k, fov_x, metadata


def _frame_arrays(prediction, input_mask, input_k, save_normals, np):
    required = ("depth", "mask", "intrinsics")
    if any(name not in prediction for name in required):
        raise ValueError("MoGe2 did not supply depth, mask and intrinsics")
    depth = prediction["depth"].detach().float().cpu().numpy()
    model_mask = prediction["mask"].detach().bool().cpu().numpy()
    normalized_k = prediction["intrinsics"].detach().float().cpu().numpy()
    if depth.shape != input_mask.shape or model_mask.shape != depth.shape:
        raise ValueError("MoGe2 maps do not match the preprocessed frame")
    if (normalized_k.shape != (3, 3) or not np.isfinite(normalized_k).all() or
            normalized_k[0, 0] <= 0 or normalized_k[1, 1] <= 0):
        raise ValueError("MoGe2 camera intrinsics are missing or nonfinite")
    height, width = depth.shape
    pixel_map = np.array([[width, 0, -0.5], [0, height, -0.5], [0, 0, 1]], dtype=np.float32)
    valid = model_mask & input_mask & np.isfinite(depth) & (depth > 0)
    arrays = {
        "depth_m": np.where(valid, depth, np.nan).astype(np.float32),
        "valid": valid, "model_mask": model_mask, "input_valid": input_mask,
        "model_intrinsics_normalized": normalized_k.astype(np.float32),
        "model_intrinsics_pixels": (pixel_map @ normalized_k).astype(np.float32),
        "input_intrinsics_pixels": input_k.astype(np.float32),
    }
    if save_normals:
        if "normal" not in prediction:
            raise ValueError("Normal output was requested but the model did not return it")
        normals = prediction["normal"].detach().float().cpu().numpy()
        if normals.shape != (*depth.shape, 3):
            raise ValueError("MoGe2 normals do not match the depth map")
        normals_valid = valid & np.isfinite(normals).all(axis=-1)
        arrays["normal_camera"] = np.where(normals_valid[..., None], normals, np.nan).astype(np.float32)
        arrays["normal_valid"] = normals_valid
    return arrays


def generate_depth(request: GenerateDepthRequest) -> Path:
    """Run this vertical slice only after explicit model-run authorization."""
    local = _json("data/configs/perception/moge/local.json")
    if local["execution"]["model_execution_authorized"] is not True:
        raise PermissionError("Depth implementation only is authorized; model execution remains disabled")
    runtime = _json(Path(__file__).with_name("runtime.json"))
    if runtime["execution"].get("generate_authorized") is not True:
        raise PermissionError("Depth generation remains disabled")
    if request.source_split == "evaluation" and runtime["execution"].get("evaluation_authorized") is not True:
        raise PermissionError("Evaluation depth remains separately disabled")
    if request.episode_index not in runtime["authorized_episode_indices"].get(request.source_split, []):
        raise PermissionError("Depth requires explicit authorization for this split and episode")
    sources = _json("data/configs/sources.lock.json")
    episode, clock, snapshot, video_path, stop = _select_input(request, sources)
    policy = _json("data/configs/calibration/ego.json")
    if policy["dataset_revision"] != sources["dataset"]["revision"]:
        raise ValueError("Orientation policy and source dataset revisions differ")
    if policy["orientation"]["additional_pixel_rotation_deg"] != 0:
        raise ValueError("This depth slice expects the released upright decoded RGB images")
    checkpoint = _local_path(local["model"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    calibration_path = snapshot / "track_3/public/meta/camera_calibration.json"
    calibration = _json(calibration_path)
    camera = calibration["cameras"]["ego_cam_a"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    contract = run_fields(ROOT, "depth", "generate", asdict(request), inputs=[
        video_path, calibration_path, checkpoint, local["package_lock"], local["source_pins"],
        "data/configs/perception/moge/local.json", Path(__file__).with_name("runtime.json"),
        "data/configs/sources.lock.json", "data/configs/splits/public-v1.json", "data/configs/calibration/ego.json",
        "data/manifests/dataset/episodes.json", "data/reports/data-audit/videos.json"], allowed=("video_only",))
    directory = output_path(ROOT, f"data/outputs/depth/{request.source_split}/episode_{request.episode_index:06d}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frames").mkdir()
    manifest_path = directory / "run.json"
    frames = [{"frame_index": index, "status": "not_processed",
               "video_pts": clock["pts_integer"][index], "video_pts_seconds": clock["pts_seconds"][index]}
              for index in range(request.start_frame, stop, request.frame_stride)]
    record = {
        "schema_version": 1, "feature": "depth", "run_id": run_id, "status": "running",
        **contract, "split": request.source_split, "episode_index": request.episode_index,
        "task": episode["task"], "camera": "ego_cam_a",
        "dataset_revision": sources["dataset"]["revision"],
        "source_video": video_path.relative_to(ROOT).as_posix(),
        "source_timestamps": "data/reports/data-audit/videos.json",
        "source_time_base_and_frame_rate": clock["time_base_and_frame_rate"],
        "model": local["model"], "source_pins": _json(local["source_pins"]),
        "depth_units": "metres estimated by the model; metric accuracy not established",
        "depth_kind": "per-frame monocular prediction", "seed": 31, "confidence": None,
        "confidence_note": "MoGe2 infer exposes a binary mask, not a calibrated confidence score",
        "invalid_depth_encoding": "NaN with explicit valid=false", "interpolation": "none",
        "world_T_camera": None, "object_and_hand_poses_read": False,
        "other_clocks": "Parquet timestamp and capture_time are not read or inferred by this RGB feature",
        "official_evaluation": False, "frames": frames, "completed_frames": 0,
        "selected_frames": len(frames), "resolved_stop_frame": stop,
        "tests_run": False,
    }
    write_json(manifest_path, record)
    started = time.perf_counter()
    capture = model = torch = None
    lease = GpuLease(ROOT, request.gpu_headroom_mib, request.gpu_admission_allowance_mib)
    current = None
    try:
        _configure_local_runtime(local)
        import cv2
        import numpy as np
        import torch
        from moge.model.v2 import MoGeModel

        record["environment"] = environment()
        record["input_hashes"] = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "video_sha256": sha256_file(video_path),
            "calibration_sha256": sha256_file(calibration_path),
            "handler_sha256": sha256_file(Path(__file__)),
            "package_lock_sha256": sha256_file(_local_path(local["package_lock"])),
        }
        source_files = _source_snapshot(local)
        write_json(directory / "source-files.json", {"files": source_files})
        record["sources"] += [entry for entry in source_files if entry not in record["sources"]]
        map_x, map_y, input_mask, input_k, fov_x, preprocessing = _pinhole_maps(
            camera, request.max_image_side, cv2, np)
        record["preprocessing"] = preprocessing
        save_npz(directory / "preprocessing.npz", np, source_x=map_x, source_y=map_y,
                 input_valid=input_mask, input_intrinsics_pixels=input_k)
        record["preprocessing_artifact_sha256"] = sha256_file(directory / "preprocessing.npz")
        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the selected RGB stream")
        record["decoder_autorotation_disable_accepted"] = bool(
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0))
        if clock.get("rotation_metadata_lines"):
            raise ValueError("Rotation-tagged streams require a separate explicit orientation decision")
        lease.__enter__()
        try:
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(31)
            model = MoGeModel.from_pretrained(str(checkpoint), local_files_only=True).to("cuda:0").eval()
            lease.require_headroom()
            source_width, source_height = camera["resolution"]
            for index in range(stop):
                current = None
                if index < request.start_frame or (index - request.start_frame) % request.frame_stride:
                    if not capture.grab():
                        raise RuntimeError(f"RGB decode stopped at skipped frame {index}")
                    continue
                current = frames[(index - request.start_frame) // request.frame_stride]
                decoded, frame = capture.read()
                if not decoded:
                    raise RuntimeError(f"RGB decode stopped at frame {index}")
                current["status"] = "running"
                frame_started = time.perf_counter()
                if frame.shape[:2] != (source_height, source_width):
                    raise ValueError("RGB dimensions do not match the factory calibration")
                lease.require_headroom()
                undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(np.ascontiguousarray(rgb)).to("cuda:0", dtype=torch.float32)
                image = image.permute(2, 0, 1) / 255.0
                with torch.inference_mode():
                    prediction = model.infer(image, num_tokens=request.num_tokens, fov_x=fov_x,
                                             force_projection=True, apply_mask=False,
                                             use_fp16=request.use_fp16)
                arrays = _frame_arrays(prediction, input_mask, input_k, request.save_normals, np)
                del prediction, image
                artifact = directory / "frames" / f"frame_{index:06d}.npz"
                save_npz(artifact, np, **arrays)
                current.update(status="completed", artifact=artifact.relative_to(directory).as_posix(),
                               sha256=sha256_file(artifact), valid_pixels=int(arrays["valid"].sum()),
                               total_pixels=int(arrays["valid"].size),
                               elapsed_seconds=time.perf_counter() - frame_started)
                record["completed_frames"] += 1
                write_json(manifest_path, record)
                print(f"Depth frame {index}: {current['valid_pixels']}/{current['total_pixels']} valid pixels",
                      flush=True)
                current = None
                lease.require_headroom()
            torch.cuda.synchronize(0)
        finally:
            try:
                if torch.cuda.is_initialized():
                    record["gpu"] = {
                        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
                        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    }
                    prediction = image = model = None
                    torch.cuda.empty_cache()
            finally:
                lease.__exit__(None, None, None)
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        if current is not None:
            current.update(status="failed", error=record["error"])
        raise
    finally:
        if capture is not None:
            capture.release()
        record["gpu"] = {**record.get("gpu", {}), **lease.report()}
        model = None
        record["elapsed_seconds"] = time.perf_counter() - started
        record["timing"] = {"wall_seconds": record["elapsed_seconds"]}
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["output_bytes_excluding_manifest"] = sum(
            path.stat().st_size for path in directory.rglob("*") if path.is_file() and path != manifest_path)
        write_json(manifest_path, record)
    return manifest_path
