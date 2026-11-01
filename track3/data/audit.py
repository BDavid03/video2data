"""Read-only, CPU-only audit of a pinned local Track 3 dataset snapshot."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .episodes import CAMERAS, discover_episodes, validate_pose
from track3.provenance import sha256_file, write_json

REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
LIMITATIONS = ["Container PTS equality cannot establish physical camera exposure synchronization.",
               "Decoded frame checksum equality is only a duplicate candidate, not proof of sensor frame duplication.",
               "Evaluation labels and task identities were not accessed or inferred.",
               "Nominal video timing and public capture timing must remain separate."]


def verify_file(path, expected):
    result = {"expected_size": expected["size"], "algorithm": "sha256" if expected.get("lfs_sha256") else "git-blob-sha1"}
    result["expected_hash"] = expected.get("lfs_sha256") or expected["blob_id"]
    if not path.is_file():
        return dict(result, status="missing")
    size = path.stat().st_size
    digest = hashlib.sha256() if result["algorithm"] == "sha256" else hashlib.sha1()
    if result["algorithm"] == "git-blob-sha1":
        digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    result.update(actual_size=size, actual_hash=digest.hexdigest())
    result["status"] = "ok" if size == expected["size"] and result["actual_hash"] == result["expected_hash"] else "mismatch"
    return result


def timestamp_summary(values):
    if not values:
        return {"count": 0}
    finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    if not finite:
        return {"count": len(values), "all_finite": False}
    deltas = [b - a for a, b in zip(values, values[1:])]
    return {"count": len(values), "all_finite": True, "first": values[0], "last": values[-1],
            "duplicate_count": sum(d == 0 for d in deltas), "backwards_count": sum(d < 0 for d in deltas),
            "delta_min": min(deltas) if deltas else None, "delta_max": max(deltas) if deltas else None,
            "delta_median": statistics.median(deltas) if deltas else None}


def _asset_evidence(snapshot, name, issues):
    """Mesh and URDF paths for one object, appending missing-asset issues."""
    urdf = snapshot / "track_3/public/urdf" / f"{name}_rigid.urdf"
    meshes = sorted(p.relative_to(snapshot).as_posix() for p in (snapshot / "track_3/public/mesh" / name).glob("*.glb"))
    asset = {"meshes": meshes, "urdf": urdf.relative_to(snapshot).as_posix(), "urdf_exists": urdf.is_file(), "mesh_references": []}
    if not meshes or not urdf.is_file():
        issues.append(f"missing_asset:{name}")
    if urdf.is_file():
        for node in ET.parse(urdf).iter("mesh"):
            relative = node.attrib.get("filename", "")
            exists = (urdf.parent / relative).is_file()
            asset["mesh_references"].append({"filename": relative, "exists": exists})
            if not exists:
                issues.append(f"missing_urdf_mesh:{name}:{relative}")
    return asset


def audit_trajectory(snapshot, episode):
    import pyarrow.parquet as pq

    table = pq.read_table(snapshot / episode["trajectory"])
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    objects = json.loads(metadata.get("objects", "[]"))
    rows = table.to_pylist()
    issues = []
    if metadata.get("pose_convention") != "world_T_object":
        issues.append("invalid_or_missing_pose_convention")
    validity = Counter()
    object_counts = {name: Counter() for name in objects}
    invalid_visible = []
    for frame, row in enumerate(rows):
        if row["frame_index"] != frame:
            issues.append(f"noncontiguous_frame_index:{frame}:{row['frame_index']}")
        if row["episode_index"] != episode["episode_index"]:
            issues.append(f"wrong_episode_index:{frame}")
        names = [obj["name"] for obj in row["observation.objects"]]
        if sorted(names) != sorted(objects) or len(names) != len(set(names)):
            issues.append(f"object_membership_mismatch:{frame}")
        for obj in row["observation.objects"]:
            state = validate_pose(obj["pose"], obj["visible"])
            validity[state["reason"]] += 1
            counts = object_counts.setdefault(obj["name"], Counter())
            counts[state["reason"]] += 1
            if all(v == 0 for v in obj["pose"]):
                counts["all_zero_pose"] += 1
            if obj["visible"] and not state["valid"]:
                invalid_visible.append({"frame_index": frame, "object": obj["name"], "reason": state["reason"]})
    assets = {name: _asset_evidence(snapshot, name, issues) for name in objects}
    if invalid_visible:
        issues.append(f"invalid_visible_poses:{len(invalid_visible)}")
    series = {column: [r[column] for r in rows] for column in ("timestamp", "capture_time")}
    summaries = {column: timestamp_summary(values) for column, values in series.items()}
    for column, timing in summaries.items():
        if not timing.get("all_finite", False) or timing.get("duplicate_count") or timing.get("backwards_count"):
            issues.append(f"invalid_{column}")
    return {"episode_index": episode["episode_index"], "task": episode["task"], "path": episode["trajectory"],
            "row_count": len(rows), "metadata": metadata, "objects": objects, "assets": assets,
            "validity_counts": dict(validity), "per_object_counts": {k: dict(v) for k, v in object_counts.items()},
            "invalid_visible_poses": invalid_visible, "issues": issues,
            "timestamps": summaries["timestamp"], "capture_times": summaries["capture_time"],
            "timestamp_seconds": series["timestamp"], "capture_time_seconds": series["capture_time"]}


def find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    if importlib.util.find_spec("imageio_ffmpeg"):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    return None


def decode_video(executable, path):
    started = time.monotonic()
    command = [executable, "-hide_banner", "-nostdin", "-nostats", "-v", "info", "-xerror",
               "-hwaccel", "none", "-threads", "2", "-noautorotate", "-i", str(path), "-map", "0:v:0",
               "-an", "-sn", "-vf", "showinfo", "-fps_mode", "passthrough", "-f", "null", "-"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_seconds": time.monotonic() - started}
    frames = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+).*?\bs:(\d+)x(\d+).*?\bchecksum:([0-9A-F]+)", process.stderr)
    pts = [float(f[2]) for f in frames]
    checksums = [f[5] for f in frames]
    dimensions = sorted({(int(f[3]), int(f[4])) for f in frames})
    header = process.stderr.split("Stream mapping:")[0]
    errors = [line for line in process.stderr.splitlines() if any(token in line.lower() for token in ("error", "invalid", "corrupt", "failed"))]
    config = re.findall(r"config in time_base:\s*([^,]+), frame_rate:\s*(\S+)", process.stderr)
    return {"status": "ok" if process.returncode == 0 and frames else "decode_failed", "returncode": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 4), "decoded_frame_count": len(frames),
            "dimensions_width_height": dimensions, "time_base_and_frame_rate": config[:1],
            "timestamps": timestamp_summary(pts), "pts_seconds": pts, "pts_integer": [int(f[1]) for f in frames],
            "frame_checksums_adler32": checksums, "adjacent_identical_decoded_frame_checksums": sum(a == b for a, b in zip(checksums, checksums[1:])),
            "decoder_frame_indices_contiguous": [int(f[0]) for f in frames] == list(range(len(frames))),
            "container_header": header, "rotation_metadata_lines": [line.strip() for line in header.splitlines() if "rotation" in line.lower() or "displaymatrix" in line.lower()],
            "errors": errors, "autorotation": False, "hardware_acceleration": "none"}


def audit_failures(summary):
    """Return all hard failures, including incomplete requested capabilities."""
    failures = []
    if not summary["track3_file_count"] or any(k != "ok" for k in summary["integrity_counts"]):
        failures.append("file_integrity")
    if summary["episode_issues"]:
        failures.append("episode_discovery")
    if summary["trajectory_status"] != "audited" or summary["trajectory_issues"]:
        failures.append("public_trajectories")
    if summary["video_requested"]:
        if summary["video_status"] != "completed" or summary["video_count"] != summary["expected_video_count"] or summary["video_issues"]:
            failures.append("video_decode")
        if len(summary["pairing"]) != sum(summary["episode_counts"].values()) or any(
            not p.get("frame_counts_equal", False) or not p.get("container_pts_equal", False)
            or not p.get("public_rows_match_video_frames", True) for p in summary["pairing"]
        ):
            failures.append("frame_pairing")
    return failures


def _decode_videos(executable, snapshot, episodes, report_dir, revision):
    """Decode every released stream, checkpointing videos.json after each one."""
    videos = []
    for episode in episodes:
        for camera, relative in sorted(episode["videos"].items()):
            record = {"split": episode["split"], "episode_index": episode["episode_index"], "camera": camera, "path": relative}
            record.update(decode_video(executable, snapshot / relative))
            videos.append(record)
            print(f"decode {len(videos)}/120 {episode['split']} {episode['episode_index']} {camera}: {record['status']} {record.get('decoded_frame_count', 0)} frames", flush=True)
            write_json(report_dir / "videos.json", {"schema_version": 1, "revision": revision, "decoder": executable, "status": "in_progress", "videos": videos})
    return videos


def _pairing(episodes, videos, trajectories):
    """Cross-camera frame/PTS agreement for episodes with all three streams decoded."""
    pairing = []
    for episode in episodes:
        streams = [v for v in videos if v["split"] == episode["split"] and v["episode_index"] == episode["episode_index"]]
        if {v["camera"] for v in streams} != set(CAMERAS) or len(streams) != 3 or any(v["status"] != "ok" for v in streams):
            continue
        counts = [v["decoded_frame_count"] for v in streams]
        entry = {"split": episode["split"], "episode_index": episode["episode_index"], "frame_counts_equal": len(set(counts)) == 1,
                 "container_pts_equal": all(v["pts_integer"] == streams[0]["pts_integer"] and v["time_base_and_frame_rate"] == streams[0]["time_base_and_frame_rate"] for v in streams),
                 "physical_exposure_synchronization": "unverified", "comparison_tolerance_seconds": 0.0}
        trajectory = next((t for t in trajectories if t["episode_index"] == episode["episode_index"]), None) if episode["split"] == "public" else None
        if trajectory and "row_count" in trajectory:
            entry["public_rows_match_video_frames"] = all(c == trajectory["row_count"] for c in counts)
            if entry["public_rows_match_video_frames"]:
                entry["max_video_pts_minus_public_timestamp_seconds"] = max(abs(a-b) for a,b in zip(streams[0]["pts_seconds"], trajectory["timestamp_seconds"]))
        pairing.append(entry)
    return pairing


def _video_issue(video):
    timestamps = video.get("timestamps", {})
    return (video["status"] != "ok" or video.get("errors") or not video.get("decoder_frame_indices_contiguous")
            or not timestamps.get("all_finite") or timestamps.get("duplicate_count") or timestamps.get("backwards_count"))


def audit(cache_root, revision, manifest_dir, report_dir, decode=False):
    started = time.monotonic()
    cache_root = Path(cache_root).resolve()
    snapshot = cache_root / "snapshots" / revision
    tree_path = cache_root / "trees" / f"{revision}.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))["files"]
    for destination in (Path(manifest_dir).resolve(), Path(report_dir).resolve()):
        if destination == cache_root or cache_root in destination.parents:
            raise ValueError("Audit output must not be inside the immutable dataset cache")
    manifest_dir, report_dir = Path(manifest_dir), Path(report_dir)
    inventory = [dict(path=relative, **verify_file(snapshot / relative, expected))
                 for relative, expected in sorted(tree.items()) if relative.startswith("track_3/")]
    write_json(manifest_dir / "inventory.json", {"schema_version": 1, "revision": revision, "tree_sha256": sha256_file(tree_path), "files": inventory})
    episodes = discover_episodes(snapshot)
    write_json(manifest_dir / "episodes.json", {"schema_version": 1, "revision": revision, "episodes": episodes, "discovery": "actual filenames plus explicit public episode metadata; no split-range expansion"})
    parquet_available = importlib.util.find_spec("pyarrow") is not None
    trajectory_status = "audited" if parquet_available else "blocked_missing_pyarrow"
    trajectories = []
    for episode in episodes if parquet_available else ():
        if episode["split"] == "public" and episode["trajectory"]:
            try:
                trajectories.append(audit_trajectory(snapshot, episode))
            except Exception as error:
                trajectories.append({"episode_index": episode["episode_index"], "issues": [f"trajectory_read_failed:{type(error).__name__}:{error}"]})
    write_json(report_dir / "trajectories.json", {"schema_version": 1, "revision": revision, "status": trajectory_status, "episodes": trajectories})
    executable = find_ffmpeg() if decode else None
    videos = _decode_videos(executable, snapshot, episodes, report_dir, revision) if executable else []
    video_status = "completed" if executable else ("blocked_missing_ffmpeg" if decode else "not_requested")
    write_json(report_dir / "videos.json", {"schema_version": 1, "revision": revision, "decoder": executable, "status": video_status, "videos": videos})
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "revision": revision,
               "snapshot": str(snapshot), "input_immutable": True, "python": sys.version, "elapsed_seconds": round(time.monotonic() - started, 3),
               "integrity_counts": dict(Counter(r["status"] for r in inventory)), "track3_file_count": len(inventory),
               "verified_bytes": sum(r.get("actual_size", 0) for r in inventory), "episode_counts": dict(Counter(e["split"] for e in episodes)),
               "public_episode_indices": [e["episode_index"] for e in episodes if e["split"] == "public"],
               "episode_issues": [e for e in episodes if e["issues"]], "trajectory_count": len(trajectories),
               "trajectory_status": trajectory_status,
               "trajectory_issues": [{"episode_index": t["episode_index"], "issues": t["issues"]} for t in trajectories if t["issues"]],
               "public_pose_validity_counts": dict(sum((Counter(t.get("validity_counts", {})) for t in trajectories), Counter())),
               "video_status_counts": dict(Counter(v["status"] for v in videos)), "decoded_frame_count": sum(v.get("decoded_frame_count", 0) for v in videos),
               "video_requested": decode, "video_status": video_status,
               "video_count": len(videos), "expected_video_count": sum(len(e["videos"]) for e in episodes),
               "video_issues": [{"path": v["path"], "status": v["status"], "timestamps": v.get("timestamps"), "errors": v.get("errors", [])}
                                for v in videos if _video_issue(v)],
               "pairing": _pairing(episodes, videos, trajectories), "limitations": LIMITATIONS}
    summary["failures"] = audit_failures(summary)
    write_json(report_dir / "audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pairing", "limitations")}, indent=2))
    return summary


def main(argv=None):
    from track3.paths import output_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="datasets--nvidia--video_to_data_challenge")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--manifest-dir", default="manifests/dataset")
    parser.add_argument("--report-dir", default="reports/data-audit")
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    manifest_dir = output_path(project_root, args.manifest_dir)
    report_dir = output_path(project_root, args.report_dir)
    result = audit(args.cache_root, args.revision, manifest_dir, report_dir, args.decode)
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
