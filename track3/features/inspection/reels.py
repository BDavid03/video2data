"""Bounded offline manual-event reels over admitted saved RGB, never estimation."""

from bisect import bisect_right
from datetime import datetime, timezone
from fractions import Fraction
import time
from uuid import uuid4

from track3.paths import output_path, project_path
from track3.provenance import read_run, run_fields, sha256_file, write_json
from .operations import ROOT, EPISODES, _artifact, _confine_caches, _gate, _json, _verified


def _plan(document, frames, time_base, request):
    """Map each manual event window to CFR samples holding the latest exact source frame/PTS."""
    pts = document["clock"]["pts_integer"]
    plan, count = [], 0
    for event in document["manual_events"]:
        start, stop = event["start_frame"], event["stop_frame"]
        indices = list(range(start, stop))
        if not indices or any(i not in frames for i in indices):
            raise ValueError("Reel windows must retain every source frame")
        if stop >= len(pts):
            raise ValueError("Reel window needs a following source PTS to bound its last-frame duration")
        duration = (pts[stop] - pts[start]) * time_base
        samples = -(-duration.numerator * request["fps"] // duration.denominator)
        count += samples
        if count > request["max_encoded_frames"]:
            raise ValueError("Declared event windows exceed the total encoded-frame budget")
        window_pts = [pts[i] for i in indices]
        mapping = []
        for k in range(samples):
            source_tick = Fraction(pts[start]) + Fraction(k, request["fps"]) / time_base
            index = indices[max(0, bisect_right(window_pts, source_tick) - 1)]
            mapping.append({"encoded_frame": k, "encoded_seconds": k / request["fps"],
                            "frame_index": index, "source_pts": pts[index]})
        plan.append({"event": event, "mapping": mapping, "source_duration_seconds": float(duration)})
    if not plan:
        raise ValueError("Supply at least one explicit manual image-review event; no semantic event is inferred")
    return plan


def _canvas(frame, sample, labels, side, images, record, cv2, np):
    """Side-by-side A/B panels for one sample; each source image is verified and resized once per reel run."""
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    panel_width = side // len(labels)
    for panel, label in enumerate(labels):
        channel = frame["candidates"][label].get("preview_run", {})
        if channel.get("status") != "available":
            cv2.putText(canvas, f"{label} MISSING SOURCE", (panel * panel_width + 4, side // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
            continue
        image_kind = "overlay" if "overlay" in channel["artifacts"] else "rgb"
        entry = channel["artifacts"][image_kind]
        sample.setdefault("image_channels", {})[label] = image_kind
        key = (entry["path"], entry["sha256"])
        if key not in images:
            asset = _verified(entry, "Reel source image changed since inspection export")
            if entry not in record["inputs"]:
                record["inputs"].append(entry)
            image = cv2.imread(str(asset))
            if image is None:
                raise ValueError("Could not decode admitted reel source image")
            factor = min(panel_width / image.shape[1], (side - 64) / image.shape[0])
            width, height = max(1, int(image.shape[1] * factor)), max(1, int(image.shape[0] * factor))
            images[key] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = images[key]
        height, width = image.shape[:2]
        x, y = panel * panel_width + (panel_width - width) // 2, 64 + (side - 64 - height) // 2
        canvas[y:y + height, x:x + width] = image
    return canvas


def encode(value):
    request = _json(value)
    if set(request) != {"inspection_run", "max_side", "max_encoded_frames", "fps"}:
        raise ValueError("Declare inspection run, max_side, max_encoded_frames and fps")
    for key, low, high in (("max_side", 128, 1024), ("max_encoded_frames", 1, 4096), ("fps", 1, 60)):
        if type(request[key]) is not int or not low <= request[key] <= high:
            raise ValueError(f"Reel {key} exceeds its bounded budget")
    # Gate before opening a source path; reject reserved episode directories first.
    if _json("track3/features/inspection/runtime.json").get("encode_authorized") is not True:
        raise PermissionError("Inspection reel encoding remains disabled")
    path = project_path(ROOT, request["inspection_run"])
    base = ROOT / "outputs/inspection/public"
    if not path.is_relative_to(base):
        raise ValueError("Reels consume feature-owned inspection exports only")
    parts = path.relative_to(base).parts
    if len(parts) != 3 or parts[0] not in {f"episode_{i:06d}" for i in EPISODES}:
        raise PermissionError("Reel path must identify an admitted episode before opening")
    source = read_run(ROOT, path, kind="inspection.export", allowed=("video_only",))
    _gate("encode", source["episode_index"])
    document_entry = _artifact(path, "inspection.json", source["artifacts"]["inspection.json"])
    document = _json(document_entry["path"])
    time_base = Fraction(document["source_time_base"])
    if time_base <= 0:
        raise ValueError("Reels require a positive source time base")
    frames = {row["frame_index"]: row for row in document["frames"]}
    plan = _plan(document, frames, time_base, request)
    record = run_fields(ROOT, "inspection", "encode", request, parents=[path], allowed=("video_only",),
                        inputs=[document_entry["path"], "track3/features/inspection/runtime.json"])
    record.update(split="public", episode_index=source["episode_index"], dataset_revision=source["dataset_revision"],
                  artifacts={}, review_only=True, quality_approved=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/inspection/public/episode_{source['episode_index']:06d}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record["started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "run.json", record)
    try:
        _confine_caches()
        import cv2
        import numpy as np

        labels = [row["label"] for row in document["candidates"]]
        side = request["max_side"] // 2 * 2
        images = {}
        for slot, item in enumerate(plan):
            target = directory / f"event_{slot:03d}.mp4"
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), request["fps"], (side, side))
            try:
                if not writer.isOpened():
                    raise RuntimeError("Existing OpenCV build cannot open mp4v; no codec is installed or substituted")
                for sample in item["mapping"]:
                    frame = frames[sample["frame_index"]]
                    canvas = _canvas(frame, sample, labels, side, images, record, cv2, np)
                    text = f"{item['event']['kind']} frame {frame['frame_index']} PTS {frame['source_pts']} A/B shared source clock"
                    cv2.putText(canvas, text, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
                    cv2.putText(canvas, "Manual event; missing frames retained; CFR resampling", (4, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
                    writer.write(canvas)
            finally:
                writer.release()
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError("Encoder produced no bytes")
            item["artifact"] = target.name
            record["artifacts"][target.name] = sha256_file(target)
        manifest = {"schema_version": 1, "status": "encoded_unreviewed", "windows": plan,
                    "fps": request["fps"], "source_time_base": str(time_base),
                    "semantics": "CFR sample-and-hold; each exact source frame/PTS mapping retained; no interpolation",
                    "limitations": "OpenCV writer has no per-frame success signal; files were not decoded or assessed"}
        write_json(directory / "reel-manifest.json", manifest)
        record["artifacts"]["reel-manifest.json"] = sha256_file(directory / "reel-manifest.json")
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        write_json(directory / "run.json", record)
    return directory / "run.json"
