"""Discover released episodes from files; never expand metadata split ranges."""

import json
import math
import re
from pathlib import Path

EPISODE_RE = re.compile(r"episode_(\d+)\.(mp4|parquet)$")
CAMERAS = ("ego_cam_a", "ego_cam_b", "ego_cam_c")
QUATERNION_NORM_TOLERANCE = 1e-3


def _empty(split, index, issues=()):
    return {"split": split, "episode_index": index, "videos": {}, "trajectory": None, "issues": list(issues)}


def _episode_files(root, pattern):
    """(episode index, path) for files under root whose names match the release pattern."""
    for path in sorted(root.glob(f"**/{pattern}")):
        if match := EPISODE_RE.fullmatch(path.name):
            yield int(match[1]), path


def discover_episodes(snapshot):
    """Return released episodes with snapshot-relative paths and explicit issues.

    Evaluation records contain videos only; their labels and tasks are neither accessed nor inferred.
    """
    snapshot = Path(snapshot)
    metadata_path = snapshot / "track_3/public/meta/episodes_metadata.jsonl"
    metadata = {}
    if metadata_path.is_file():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                index = int(row["episode_index"])
                if index in metadata:
                    raise ValueError(f"Duplicate public episode metadata: {index}")
                metadata[index] = row
    records = []
    for split in ("public", "evaluation"):
        public = split == "public"
        by_id = {}
        for index, path in _episode_files(snapshot / "track_3" / split / "videos", "episode_*.mp4"):
            camera = path.parent.name.removeprefix("observation.images.")
            record = by_id.setdefault(index, _empty(split, index))
            if camera in record["videos"]:
                raise ValueError(f"Duplicate camera {camera} for {split} episode {index}")
            record["videos"][camera] = path.relative_to(snapshot).as_posix()
            if camera not in CAMERAS:
                record["issues"].append(f"unknown_camera:{camera}")
        if public:
            for index, path in _episode_files(snapshot / "track_3/public/data", "episode_*.parquet"):
                record = by_id.setdefault(index, _empty(split, index))
                if record["trajectory"]:
                    raise ValueError(f"Duplicate trajectory for public episode {index}")
                record["trajectory"] = path.relative_to(snapshot).as_posix()
            for index in metadata:
                by_id.setdefault(index, _empty(split, index, ["metadata_without_files"]))
        for index, record in sorted(by_id.items()):
            record["task"] = metadata.get(index, {}).get("task") if public else None
            if public and index not in metadata:
                record["issues"].append("missing_public_metadata")
            if public and not record["trajectory"]:
                record["issues"].append("missing_public_trajectory")
            record["issues"] += [f"missing_camera:{camera}" for camera in CAMERAS if camera not in record["videos"]]
            paths = [*record["videos"].values(), *([record["trajectory"]] if record["trajectory"] else [])]
            record["issues"] += [f"missing_path:{relative}" for relative in paths if not (snapshot / relative).is_file()]
            records.append(record)
    return records


def validate_pose(pose, visible, quaternion_tolerance=QUATERNION_NORM_TOLERANCE):
    """Target validity for an xyz + wxyz pose; invalid poses are reported, never normalised."""
    if not isinstance(visible, bool):
        return {"valid": False, "reason": "invalid_visibility"}
    if not visible:
        return {"valid": False, "reason": "not_visible"}
    if pose is None or len(pose) != 7:
        return {"valid": False, "reason": "invalid_shape"}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in pose):
        return {"valid": False, "reason": "nonfinite_or_nonnumeric"}
    if all(v == 0 for v in pose):
        return {"valid": False, "reason": "zero_pose"}
    norm = math.sqrt(sum(v * v for v in pose[3:]))
    if abs(norm - 1) > quaternion_tolerance:
        return {"valid": False, "reason": "invalid_quaternion_norm", "quaternion_norm": norm}
    return {"valid": True, "reason": "valid", "quaternion_norm": norm}


def iter_public_frames(snapshot, episode):
    """Yield public diagnostic rows with per-object validity; frame index equals row index."""
    if episode.get("split") != "public":
        raise ValueError("Public trajectory adapter refuses evaluation episodes")
    relative = episode.get("trajectory")
    if not relative:
        raise FileNotFoundError("Public episode has no trajectory")
    # HF snapshots symlink into sibling blobs, so validate the logical path rather than the resolved one.
    logical = Path(relative)
    if logical.is_absolute() or ".." in logical.parts or logical.parts[:3] != ("track_3", "public", "data"):
        raise ValueError("Trajectory must be a public snapshot-relative path")
    import pyarrow.parquet as pq

    table = pq.read_table((Path(snapshot).resolve() / relative).resolve())
    if (table.schema.metadata or {}).get(b"pose_convention") != b"world_T_object":
        raise ValueError("Public trajectory must explicitly declare world_T_object")
    for frame_index, raw in enumerate(table.to_pylist()):
        if raw["frame_index"] != frame_index or raw["episode_index"] != episode["episode_index"]:
            raise ValueError("Public trajectory episode/frame index mismatch")
        objects = []
        for obj in raw["observation.objects"]:
            state = validate_pose(obj["pose"], obj["visible"])
            objects.append({"name": obj["name"], "pose_xyz_wxyz": obj["pose"], "visible": obj["visible"],
                            "valid": state["valid"], "invalid_reason": None if state["valid"] else state["reason"]})
        yield {"episode_index": raw["episode_index"], "frame_index": frame_index,
               "timestamp_seconds": raw["timestamp"], "capture_time_seconds": raw["capture_time"],
               "pose_convention": "world_T_object", "label_use": "public_diagnostic", "objects": objects}
