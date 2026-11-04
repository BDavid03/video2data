"""Explicit offline ViPE/DA3/VDA result ingestion; never launches a model."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import shutil
from uuid import uuid4

from src.paths import output_path, project_path
from src.provenance import input_file, read_run, run_fields, sha256_file, write_json
from src.features.camera_motion.challenger import normalize_optical_pose
from .generate import ROOT, save_npz


@dataclass(frozen=True)
class IngestChallengerRequest:
    manifest: str
    calibration_depth_run: str

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig")))


def ingest_challenger(request):
    runtime_path = ROOT / "src/features/depth/runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8-sig"))
    if runtime["execution"].get("challenger_ingest_authorized") is not True:
        raise PermissionError("Offline challenger ingestion remains disabled")
    # Check the canonical output root before reading challenger data or importing NumPy.
    output_path(ROOT, "data/outputs/depth_challengers")
    manifest_path = project_path(ROOT, request.manifest)
    source = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if source.get("schema_version") != 1 or source.get("approach") not in ("vipe", "da3", "vda"):
        raise ValueError("Unsupported challenger manifest; no fallback is available")
    episode = source.get("episode_index")
    if (source.get("split") != "public" or type(episode) is not int or
            episode not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)):
        raise PermissionError("Challengers are restricted to frozen training/tuning episodes")
    if source.get("provenance") != "video_only" or source.get("status") != "completed":
        raise ValueError("A completed explicitly video-only export is required")
    parent_path = project_path(ROOT, request.calibration_depth_run)
    parent = read_run(ROOT, parent_path, kind="depth.generate", allowed=("video_only",))
    for key, expected in (("split", "public"), ("episode_index", episode),
                          ("dataset_revision", parent["dataset_revision"]),
                          ("grid", parent["preprocessing_artifact_sha256"]),
                          ("source_video_sha256", parent["input_hashes"]["video_sha256"])):
        if source.get(key) != expected:
            raise ValueError(f"Challenger {key} differs from the calibrated RGB producer")
    if parent["split"] != "public" or parent["episode_index"] != episode:
        raise ValueError("Calibration producer belongs to a different episode")
    if source.get("camera") != "ego_cam_a" or source.get("camera_frame") != "OpenCV optical x-right y-down z-forward":
        raise ValueError("Unsupported optical coordinate frame")
    if source.get("source_time_base_and_frame_rate") != parent["source_time_base_and_frame_rate"]:
        raise ValueError("Challenger source clock differs from the calibrated producer")
    depth_kind = source.get("depth_kind")
    if depth_kind not in ("metric_optical_z_m", "relative_optical_z"):
        raise ValueError("Only declared optical-z depth is supported; scale is never inferred")
    record = run_fields(ROOT, "depth", "ingest_challenger", asdict(request),
                        parents=[parent_path], inputs=[manifest_path, runtime_path], allowed=("video_only",))

    def receipt(entry):
        if not isinstance(entry, dict) or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError("Every source, weight, license and array needs a SHA256 receipt")
        path = project_path(ROOT, entry["path"])
        actual = input_file(ROOT, path)
        if actual["sha256"] != entry["sha256"]:
            raise ValueError(f"Receipt mismatch: {entry['path']}")
        if actual not in record["inputs"]:
            record["inputs"].append(actual)
        return path

    provisioning = source["provisioning"]
    if not re.fullmatch(r"[0-9a-f]{40}", provisioning.get("source_commit", "")):
        raise ValueError("An immutable source commit is required")
    receipt(provisioning["source_archive"])
    for name in ("weights", "licenses"):
        if not isinstance(provisioning.get(name), list) or not provisioning[name]:
            raise ValueError(f"Explicit {name} receipts are required")
        for entry in provisioning[name]:
            receipt(entry)
    if not isinstance(provisioning.get("license_review"), str) or not provisioning["license_review"].strip():
        raise ValueError("Record component and checkpoint license terms and use restrictions")
    grid_path = parent_path.parent / "preprocessing.npz"
    receipt({"path": str(grid_path), "sha256": source["grid"]})
    frames = source.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 384:
        raise ValueError("The offline adapter admits 1..384 explicitly selected frames")
    known = {row["frame_index"]: row for row in parent["frames"] if row["status"] == "completed"}
    indices = [row["frame_index"] for row in frames]
    if any(type(index) is not int for index in indices) or indices != sorted(set(indices)):
        raise ValueError("Frame indices must be unique and increasing")
    paths = []
    for frame in frames:
        reference = known.get(frame["frame_index"])
        if reference is None or any(frame.get(key) != reference[key] for key in ("video_pts", "video_pts_seconds")):
            raise ValueError("Challenger frame/PTS differs from the calibrated producer")
        paths.append(receipt(frame))
    import numpy as np

    with np.load(grid_path, allow_pickle=False) as grid:
        input_valid = grid["input_valid"].astype(bool)
        intrinsics = grid["input_intrinsics_pixels"].copy()
    if input_valid.ndim != 2 or max(input_valid.shape) > 518:
        raise ValueError("Offline challenger ingestion requires a bounded grid of maximum side 518")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"data/outputs/depth_challengers/public/episode_{episode:06d}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frames").mkdir()
    record.update(run_id=run_id, split="public", episode_index=episode,
                  dataset_revision=parent["dataset_revision"], grid=source["grid"],
                  preprocessing_artifact_sha256=source["grid"], preprocessing=parent["preprocessing"],
                  approach=source["approach"], provisioning=provisioning, depth_kind=depth_kind,
                  metric_accuracy_established=False, scale_fit=None, frames=[], artifacts={},
                  confidence_semantics="producer score, uncalibrated; never converted to a probability",
                  pose_world=source.get("pose_world"), promotion="incumbent retained; comparison required")
    write_json(directory / "run.json", record)
    try:
        shutil.copyfile(grid_path, directory / "preprocessing.npz")
        record["artifacts"]["preprocessing.npz"] = sha256_file(directory / "preprocessing.npz")
        for frame, path in zip(frames, paths):
            with np.load(path, allow_pickle=False) as arrays:
                depth = np.asarray(arrays["depth"], dtype=np.float32)
                valid = arrays["valid"]
                if depth.shape != input_valid.shape or valid.shape != depth.shape or valid.dtype != np.bool_:
                    raise ValueError("Depth and boolean invalid support must preserve the calibrated grid")
                if not np.array_equal(arrays["intrinsics_pixels"], intrinsics):
                    raise ValueError("Supplied calibration was changed; resizing or calibration fitting is unsupported")
                valid = valid & input_valid & np.isfinite(depth) & (depth > 0)
                key = "depth_m" if depth_kind == "metric_optical_z_m" else "depth_relative"
                output = {key: np.where(valid, depth, np.nan), "valid": valid,
                          "intrinsics_pixels": intrinsics, "frame_index": frame["frame_index"],
                          "pts": frame["video_pts"]}
                if "confidence" in arrays:
                    confidence = np.asarray(arrays["confidence"], dtype=np.float32)
                    if confidence.shape != depth.shape or not np.isfinite(confidence[valid]).all():
                        raise ValueError("Confidence must be finite on valid depth support")
                    output["confidence"] = np.where(valid, confidence, np.nan)
                if "extrinsic" in arrays:
                    if source["approach"] == "vda" or not source.get("pose_world"):
                        raise ValueError("VDA supplies no camera motion; other poses require a named world gauge")
                    if source.get("extrinsic_direction") not in ("camera_to_world", "world_to_camera"):
                        raise ValueError("Unsupported extrinsic direction")
                    units = source.get("pose_translation_units")
                    if units not in ("m", "relative") or (units == "m" and depth_kind != "metric_optical_z_m"):
                        raise ValueError("Pose units must be explicit and consistent with the declared depth scale")
                    pose_key = "world_T_camA" if units == "m" else "world_T_camA_relative"
                    pose_valid = arrays["pose_valid"]
                    if pose_valid.shape != () or pose_valid.dtype != np.bool_:
                        raise ValueError("pose_valid must be an explicit boolean scalar")
                    output["pose_valid"] = pose_valid.copy()
                    output[pose_key] = (normalize_optical_pose(arrays["extrinsic"], source.get("extrinsic_direction"), np)
                                        if bool(pose_valid) else np.full((4, 4), np.nan))
                artifact = f"frames/frame_{frame['frame_index']:06d}.npz"
                save_npz(directory / artifact, np, **output)
                record["frames"].append({"frame_index": frame["frame_index"], "video_pts": frame["video_pts"],
                                         "video_pts_seconds": frame["video_pts_seconds"], "status": "completed",
                                         "artifact": artifact, "sha256": sha256_file(directory / artifact),
                                         "valid_pixels": int(valid.sum())})
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "run.json", record)
    return directory / "run.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    print(ingest_challenger(IngestChallengerRequest.from_file(parser.parse_args().request)))


if __name__ == "__main__":
    main()
