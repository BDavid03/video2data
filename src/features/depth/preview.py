"""Bounded calibrated RGB previews from saved depth maps; no model or hand dependency."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from uuid import uuid4

from src.paths import output_path, project_path
from src.provenance import input_file, read_run, run_fields, sha256_file, write_json
from .generate import ROOT, _configure_local_runtime


@dataclass(frozen=True)
class PreviewDepthRequest:
    depth_run: str
    source_split: str = "public"
    episode_index: int = 39
    frame_indices: list[int] = field(default_factory=lambda: [0])
    maximum_decode_frames: int = 4096

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig")))

    def __post_init__(self):
        if (self.source_split != "public" or type(self.episode_index) is not int or
                self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)):
            raise PermissionError("RGB preview is restricted to frozen training/tuning episodes")
        if type(self.maximum_decode_frames) is not int or not 1 <= self.maximum_decode_frames <= 32768:
            raise ValueError("Declare a decode ceiling between 1 and 32768 source frames")
        if (not isinstance(self.frame_indices, list) or not 1 <= len(self.frame_indices) <= 64 or
                any(type(index) is not int or index < 0 for index in self.frame_indices) or
                self.frame_indices != sorted(set(self.frame_indices)) or
                self.frame_indices[-1] >= self.maximum_decode_frames):
            raise ValueError("Name 1..64 unique increasing frames inside the decode ceiling")


def preview(request):
    runtime_path = ROOT / "src/features/depth/runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8-sig"))
    if runtime["execution"].get("preview_authorized") is not True:
        raise PermissionError("Calibrated RGB preview remains disabled")
    # Refuse the implementation checkout before creating caches or loading a decoder.
    output_path(ROOT, "data/outputs/depth_preview")
    parent_path = project_path(ROOT, request.depth_run)
    parent = read_run(ROOT, parent_path, kind="depth.generate", allowed=("video_only",))
    if parent.get("split") != request.source_split or parent.get("episode_index") != request.episode_index:
        raise ValueError("RGB preview and depth producer belong to different episode partitions")
    lock_path = ROOT / "data/configs/sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))["dataset"]
    clock_path = ROOT / "data/reports/data-audit/videos.json"
    inventory = json.loads(clock_path.read_text(encoding="utf-8-sig"))
    clocks = [row for row in inventory["videos"] if row["split"] == request.source_split and
              row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if (parent.get("dataset_revision") != lock["revision"] or inventory.get("status") != "completed" or
            inventory.get("revision") != lock["revision"] or len(clocks) != 1 or clocks[0]["status"] != "ok"):
        raise ValueError("RGB preview requires the matching completed source clock inventory")
    clock = clocks[0]
    count = clock["decoded_frame_count"]
    if (type(count) is not int or count <= 0 or len(clock["pts_integer"]) != count or
            len(clock["pts_seconds"]) != count or request.frame_indices[-1] >= count or
            clock.get("rotation_metadata_lines") or
            parent["source_time_base_and_frame_rate"] != clock["time_base_and_frame_rate"]):
        raise ValueError("RGB preview requires an unchanged complete source clock and orientation")
    for index in request.frame_indices:
        rows = [row for row in parent["frames"] if row["frame_index"] == index]
        if (len(rows) != 1 or rows[0]["status"] != "completed" or
                rows[0]["video_pts"] != clock["pts_integer"][index] or
                rows[0]["video_pts_seconds"] != clock["pts_seconds"][index]):
            raise ValueError("Selected preview frame has no matching completed depth/PTS outcome")
    video = project_path(ROOT, parent["source_video"])
    expected = project_path(ROOT, f"{lock['cache_path']}/snapshots/{lock['revision']}/{clock['path']}")
    video_entry = input_file(ROOT, video)
    if video != expected or video_entry["sha256"] != parent["input_hashes"]["video_sha256"]:
        raise ValueError("RGB preview source bytes differ from the calibrated producer")
    grid = parent["preprocessing_artifact_sha256"]
    grid_path = project_path(ROOT, parent_path.parent / "preprocessing.npz")
    grid_entry = input_file(ROOT, grid_path)
    if grid_entry["sha256"] != grid:
        raise ValueError("RGB preview preprocessing bytes differ from the calibrated producer")
    record = run_fields(ROOT, "depth", "preview", asdict(request), parents=[parent_path],
                        inputs=[runtime_path, lock_path, clock_path, grid_path, video], allowed=("video_only",))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"data/outputs/depth_preview/public/episode_{request.episode_index:06d}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=lock["revision"], grid=grid, frames=[], artifacts={},
                  review_only=True, quality_approved=False, selected_frames=request.frame_indices,
                  scope="Calibrated RGB only; no hands, depth-quality or geometric approval")
    write_json(directory / "run.json", record)
    capture = None
    try:
        _configure_local_runtime({"source_paths": []})
        import cv2
        import numpy as np

        with np.load(grid_path, allow_pickle=False) as arrays:
            valid = arrays["input_valid"]
            mx, my = arrays["source_x"], arrays["source_y"]
            k = arrays["input_intrinsics_pixels"]
            if (valid.ndim != 2 or valid.dtype != np.bool_ or mx.shape != valid.shape or
                    my.shape != valid.shape or k.shape != (3, 3) or not np.isfinite(k).all() or
                    min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]) or
                    not np.isfinite(mx[valid]).all() or not np.isfinite(my[valid]).all()):
                raise ValueError("Invalid saved pinhole preprocessing maps")
            mx, my = (np.where(valid, value, -1).astype(np.float32) for value in (mx, my))
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError("Could not open calibrated preview RGB")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        selected = set(request.frame_indices)
        for index in range(request.frame_indices[-1] + 1):
            if index not in selected:
                if not capture.grab():
                    raise RuntimeError(f"RGB preview decode stopped at frame {index}")
                continue
            row = {"frame_index": index, "video_pts": clock["pts_integer"][index],
                   "video_pts_seconds": clock["pts_seconds"][index], "status": "running"}
            record["frames"].append(row)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"RGB preview decode stopped at frame {index}")
            if list(bgr.shape[1::-1]) != parent["preprocessing"]["source_size_width_height"]:
                raise ValueError("RGB dimensions differ from the saved preprocessing maps")
            rgb = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rgb[~valid] = 0
            filename = f"frame_{index:06d}_rgb.png"
            if not cv2.imwrite(str(directory / filename), rgb):
                raise RuntimeError("Could not write calibrated RGB preview")
            digest = sha256_file(directory / filename)
            record["artifacts"][filename] = digest
            row.update(status="completed", rgb={"file": filename, "sha256": digest})
            write_json(directory / "run.json", record)
        write_json(directory / "preview.json", {"review_only": True, "quality_approved": False,
                   "scope": record["scope"], "depth_run": input_file(ROOT, parent_path), "frames": record["frames"]})
        record["artifacts"]["preview.json"] = sha256_file(directory / "preview.json")
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        for row in record["frames"]:
            if row["status"] == "running":
                row.update(status=record["status"], error=record["error"])
        raise
    finally:
        if capture is not None:
            capture.release()
        write_json(directory / "run.json", record)
    return directory / "run.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    print(preview(PreviewDepthRequest.from_file(parser.parse_args().request)))


if __name__ == "__main__":
    main()
