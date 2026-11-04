"""Selected-frame visual review of saved hand projections; no model invocation."""

from dataclasses import dataclass, field
from html import escape
import json

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, arrays, authorize, operation_run, producer, read_json,
                      require_episode, require_input, rgb_clock, rgb_video)


@dataclass(frozen=True)
class PreviewHandsRequest:
    source_split: str = "public"
    episode_index: int = 39
    hand_run: str | None = None
    depth_run: str | None = None
    frame_indices: list[int] = field(default_factory=lambda: [0])

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        require_episode(self, "preview")
        if (not isinstance(self.frame_indices, list) or not 1 <= len(self.frame_indices) <= 64
                or any(type(index) is not int or index < 0 for index in self.frame_indices)
                or self.frame_indices != sorted(set(self.frame_indices))):
            raise ValueError("Name between 1 and 64 unique ascending frame indices")


def _overlay(image, joints, valid, intrinsics, input_valid, cv2, np):
    canvas = image.copy()
    height, width = input_valid.shape
    counts = []
    for side, color in enumerate(((255, 180, 40), (70, 150, 255))):
        if not valid[side]:
            counts.append(0)
            continue
        points = joints[side]
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("Observed hand joints must be finite MANO21 camera points")
        pixels = points @ intrinsics.T
        forward = pixels[:, 2] > 0
        uv = np.full((21, 2), np.nan)
        uv[forward] = pixels[forward, :2] / pixels[forward, 2:3]
        inside = (forward & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                  & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
        coordinates = np.zeros((21, 2), dtype=np.int32)
        coordinates[inside] = np.rint(uv[inside]).astype(np.int32)
        selected = np.flatnonzero(inside)
        inside[selected] &= input_valid[coordinates[selected, 1], coordinates[selected, 0]]
        for start in (1, 5, 9, 13, 17):
            chain = [0, start, start + 1, start + 2, start + 3]
            for a, b in zip(chain, chain[1:]):
                if inside[a] and inside[b]:
                    cv2.line(canvas, tuple(map(int, coordinates[a])), tuple(map(int, coordinates[b])), color, 2, cv2.LINE_AA)
        for index in np.flatnonzero(inside):
            location = tuple(map(int, coordinates[index]))
            cv2.circle(canvas, location, 3, color, -1, cv2.LINE_AA)
            if index in (0, 4, 8, 12, 16, 20):
                cv2.putText(canvas, str(index), location, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        counts.append(int(inside.sum()))
    return canvas, counts


def _remap_grid(preprocessing, np):
    k, input_valid = preprocessing["input_intrinsics_pixels"], preprocessing["input_valid"]
    mx, my = preprocessing["source_x"], preprocessing["source_y"]
    if (input_valid.ndim != 2 or input_valid.dtype != np.dtype("bool") or
            mx.shape != input_valid.shape or my.shape != input_valid.shape or k.shape != (3, 3) or
            not np.isfinite(k).all() or not np.isfinite(mx[input_valid]).all() or not np.isfinite(my[input_valid]).all() or
            min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("Invalid preview pinhole grid")
    return (k, input_valid, np.where(input_valid, mx, -1).astype(np.float32),
            np.where(input_valid, my, -1).astype(np.float32))


def _page(frames):
    sections = []
    for row in frames:
        details = escape(json.dumps({key: row[key] for key in ("valid", "world_valid", "projected_joints", "outcomes")}, indent=2))
        sections.append(f'<section><h2>Frame {row["frame_index"]} · PTS {row["video_pts"]}</h2>'
                        f'<div class="images"><img src="{row["rgb"]["file"]}" alt="Calibrated RGB">'
                        f'<img src="{row["overlay"]["file"]}" alt="Projected left and right hand joints"></div>'
                        f'<details><summary>Validity and fit outcomes</summary><pre>{details}</pre></details></section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Hand projection review</title>'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>body{font:16px system-ui;margin:24px;background:#141820;color:#e6eaf0}'
            '.images{display:flex;flex-wrap:wrap;gap:12px}img{width:min(100%,720px);height:auto}'
            'section{margin:32px 0}pre{white-space:pre-wrap}</style><body><h1>Hand projection review</h1>'
            '<p>Left: cyan. Right: orange. Wrist and fingertip numbers follow MANO21. '
            'Only valid camera estimates are drawn. Selected frames only; this report grants no quality approval.</p>'
            + ''.join(sections) + '</body></html>')


def preview(request):
    authorize(request, "preview")
    hands = producer(request.hand_run, "hand_motion.lift", request)
    depth = producer(request.depth_run, "depth.generate", request)
    require_input(hands, depth)
    grid = depth[1]["preprocessing_artifact_sha256"]
    if hands[1].get("grid") != grid or hands[1].get("joint_order") != "MANO21" or hands[1].get("units") != "metres":
        raise ValueError("Preview requires lifted MANO21 hands on the selected metric depth grid")
    with operation_run(request, [hands, depth], "preview") as (directory, record):
        import cv2
        import numpy as np

        clock = rgb_clock(request, record)
        count = clock["decoded_frame_count"]
        data = arrays(hands, "hands.npz", record, np)
        if (data["sides"].tolist() != ["left", "right"] or
                not np.array_equal(data["frame_index"], np.arange(count)) or
                not np.array_equal(data["pts"], np.asarray(clock["pts_integer"], dtype=np.int64)) or
                data["joints_camA"].shape != (count, 2, 21, 3) or
                any(data[key].shape != (count, 2) or data[key].dtype != np.dtype("bool")
                    for key in ("valid", "world_valid")) or (data["world_valid"] & ~data["valid"]).any()):
            raise ValueError("Hand preview arrays differ from the complete RGB clock or validity contract")
        if request.frame_indices[-1] >= count:
            raise ValueError("Preview frame is outside the RGB clip")
        k, input_valid, mx, my = _remap_grid(arrays(depth, "preprocessing.npz", record, np, grid), np)
        video = rgb_video(depth, clock, record)
        record.update(grid=grid, review_only=True, quality_approved=False, selected_frames=request.frame_indices)
        report = {"hand_run": input_file(ROOT, hands[0]), "depth_run": input_file(ROOT, depth[0]),
                  "review_only": True, "quality_approved": False, "frames": record["frames"],
                  "scope": "Selected camera-space projections; no automatic quality, contact or world-orientation assessment"}
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise RuntimeError("Could not open preview RGB")
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            selected = set(request.frame_indices)
            for index in range(request.frame_indices[-1] + 1):
                if index not in selected:
                    if not capture.grab():
                        raise RuntimeError(f"Preview RGB decode stopped at frame {index}")
                    continue
                row = {"frame_index": index, "video_pts": int(data["pts"][index]), "status": "running"}
                record["frames"].append(row)
                outcomes = [item for item in hands[1]["frames"] if item["frame_index"] == index]
                if len(outcomes) != 1 or outcomes[0]["status"] != "completed" or outcomes[0]["video_pts"] != row["video_pts"]:
                    raise ValueError("Hand preview outcome does not match the selected frame")
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Preview RGB decode stopped at frame {index}")
                if list(bgr.shape[1::-1]) != depth[1]["preprocessing"]["source_size_width_height"]:
                    raise ValueError("Preview RGB dimensions differ from depth preprocessing")
                rgb = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                rgb[~input_valid] = 0
                overlay, projected = _overlay(rgb, data["joints_camA"][index], data["valid"][index], k, input_valid, cv2, np)
                row.update(valid=data["valid"][index].tolist(), world_valid=data["world_valid"][index].tolist(),
                           projected_joints=projected, outcomes=outcomes[0]["hands"])
                for label, image in (("rgb", rgb), ("overlay", overlay)):
                    filename = f"frame_{index:06d}_{label}.png"
                    if not cv2.imwrite(str(directory / filename), image):
                        raise RuntimeError("Could not write hand preview image")
                    digest = sha256_file(directory / filename)
                    record["artifacts"][filename] = digest
                    row[label] = {"file": filename, "sha256": digest}
                row["status"] = "completed"
                write_json(directory / "run.json", record)
            write_json(directory / "preview.json", report)
            (directory / "preview.html").write_text(_page(report["frames"]), encoding="utf-8")
            for filename in ("preview.json", "preview.html"):
                record["artifacts"][filename] = sha256_file(directory / filename)
        finally:
            capture.release()
    return directory / "run.json"
