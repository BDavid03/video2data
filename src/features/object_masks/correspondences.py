"""Bounded image-only identity support; a lost query is never silently restarted."""

from dataclasses import asdict, dataclass
import math
import time

from src.paths import output_path, project_path
from src.provenance import input_file, run_fields, sha256_file, write_json
from src.runs import (ROOT, artifact, dataset_revision, new_run_id, producer, read_json, recorded,
                         require_input, runtime_flag, same_revision)
from .opening_clusters import FEATURE
from .generate import _episode


@dataclass(frozen=True)
class CorrespondencesRequest:
    source_split: str = "public"
    episode_index: int = 39
    depth_run: str | None = None
    masks_run: str | None = None
    corrections: str | None = None
    corrections_sha256: str | None = None
    start_frame: int = 0
    frame_count: int = 32
    points_per_instance: int = 128
    minimum_support: int = 6
    forward_backward_limit_px: float = 1.0
    geometric_limit_px: float = 2.0
    maximum_displacement_px: float = 40.0

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split != "public" or self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42):
            raise PermissionError("Correspondences admit public training/tuning only; 41/43/44 excluded")
        for key in ("start_frame", "frame_count", "points_per_instance", "minimum_support"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key == "start_frame" else 1):
                raise ValueError(f"Invalid correspondence bound: {key}")
        if not 1 <= self.frame_count <= 384 or not 6 <= self.minimum_support <= self.points_per_instance <= 256:
            raise ValueError("Bound to 384 frames and 6–256 queries per instance")
        for key in ("forward_backward_limit_px", "geometric_limit_px", "maximum_displacement_px"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid correspondence threshold: {key}")
        if (self.corrections is None) != (self.corrections_sha256 is None):
            raise ValueError("Manual corrections need an explicit file and SHA256")


def _exclusive(masks, slot, xy, np):
    height, width = masks.shape[1:]
    x, y = np.rint(xy).astype(int).T
    inside = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    result = np.zeros(len(xy), dtype=bool)
    at = np.flatnonzero(inside)
    result[at] = masks[slot, y[at], x[at]] & (masks[:, y[at], x[at]].sum(axis=0) == 1)
    return result


def _advance(previous, current, tracks, masks, instances, request, cv2, np):
    """LK, finite/bounds, round-trip, exclusive-mask and rigid-image consensus gates."""
    live = [track for track in tracks if track["geometric_accepted"]]
    if not live:
        return []
    old = np.asarray([track["xy"] for track in live], dtype=np.float32)
    kwargs = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, old[:, None], None, **kwargs)
    if new is None or status is None:
        return [dict(track, xy=None, visible=False, geometric_accepted=False,
                     rejection_reason="forward_flow_failed") for track in live]
    new = new.reshape(-1, 2)
    finite = np.isfinite(new).all(axis=1)
    safe = np.where(finite[:, None], new, old).astype(np.float32)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, safe[:, None], None, **kwargs)
    roundtrip = np.full(len(old), np.inf)
    if back is not None and back_status is not None:
        good = back_status.ravel().astype(bool) & np.isfinite(back.reshape(-1, 2)).all(axis=1)
        roundtrip[good] = np.linalg.norm(back.reshape(-1, 2)[good] - old[good], axis=1)
    keep = status.ravel().astype(bool) & finite
    reasons = [None if value else "flow_failed_or_nonfinite" for value in keep]
    for selector, reason in ((roundtrip > request.forward_backward_limit_px, "round_trip"),
                             (np.linalg.norm(safe - old, axis=1) > request.maximum_displacement_px, "displacement")):
        for index in np.flatnonzero(keep & selector):
            reasons[index] = reason
        keep &= ~selector
    for slot, name in enumerate(instances):
        members = np.asarray([track["roster_id"] == name for track in live])
        exclusive = _exclusive(masks, slot, safe, np)
        for index in np.flatnonzero(members & keep & ~exclusive):
            reasons[index] = "outside_or_ambiguous_mask"
        keep[members] &= exclusive[members]
        selected = np.flatnonzero(members & keep)
        accepted = np.zeros(len(selected), dtype=bool)
        if len(selected) >= request.minimum_support:
            transform, inliers = cv2.estimateAffinePartial2D(
                old[selected], safe[selected], method=cv2.RANSAC,
                ransacReprojThreshold=request.geometric_limit_px, maxIters=500, confidence=0.99)
            if transform is not None and np.isfinite(transform).all() and inliers is not None:
                accepted = inliers.ravel().astype(bool)
                predicted = old[selected] @ transform[:, :2].T + transform[:, 2]
                accepted &= np.linalg.norm(predicted - safe[selected], axis=1) <= request.geometric_limit_px
        if accepted.sum() < request.minimum_support:
            accepted[:] = False
        for index in selected[~accepted]:
            keep[index], reasons[index] = False, "insufficient_rigid_image_consensus"
    return [dict(track, xy=safe[index].astype(float).tolist() if finite[index] else None,
                 visible=bool(keep[index]), geometric_accepted=bool(keep[index]),
                 rejection_reason=reasons[index], forward_backward_error_px=
                 float(roundtrip[index]) if np.isfinite(roundtrip[index]) else None)
            for index, track in enumerate(live)]


CORRECTION_FIELDS = {"frame_index", "video_pts", "rgb_sha256", "operator", "reason", "revision", "roster_id", "points_xy"}


def _corrections(request, masks_parent, grid, instances, stop):
    """Admit the optional hash-pinned manual image corrections; returns (input entry, {(frame, roster_id): row})."""
    if not request.corrections:
        return None, {}
    path = project_path(ROOT, request.corrections)
    entry = input_file(ROOT, path)
    if entry["sha256"] != request.corrections_sha256:
        raise ValueError("Manual correction hash differs")
    correction = read_json(path)
    if set(correction) != {"schema_version", "provenance", "masks_run_sha256", "grid", "frames"} or correction["schema_version"] != 1 or correction["provenance"] != "video_only":
        raise ValueError("Corrections accept image-only schema1; diagnostic label inputs are forbidden")
    if correction["masks_run_sha256"] != input_file(ROOT, masks_parent[0])["sha256"] or correction["grid"] != grid:
        raise ValueError("Corrections differ from the exact mask parent or grid")
    rows = {}
    for row in correction["frames"]:
        if set(row) != CORRECTION_FIELDS:
            raise ValueError("Unexpected manual correction fields")
        index, name = row["frame_index"], row["roster_id"]
        if type(index) is not int or not request.start_frame <= index < stop or name not in instances or (index, name) in rows:
            raise ValueError("Invalid or duplicate correction identity")
        if not all(isinstance(row[key], str) and row[key].strip() for key in ("operator", "reason", "revision")):
            raise ValueError("Corrections require operator, reason and revision")
        rows[index, name] = row
    return entry, rows


def _seed_points(current, masks, slot, correction, pts, rgb_hash, request, cv2, np):
    """Manual correction points, or corners inside the slot's exclusive mask region."""
    if correction:
        if correction["video_pts"] != pts or correction["rgb_sha256"] != rgb_hash:
            raise ValueError("Manual correction differs from exact source RGB/PTS")
        points = np.asarray(correction["points_xy"], dtype=np.float32)
    else:
        region = masks[slot] & (masks.sum(axis=0) == 1)
        corners = cv2.goodFeaturesToTrack(current, maxCorners=request.points_per_instance,
                                          qualityLevel=0.01, minDistance=5, mask=region.astype(np.uint8) * 255)
        points = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2))
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) > request.points_per_instance or not np.isfinite(points).all():
        raise ValueError("Invalid or over-budget query points")
    if len(points) and (not _exclusive(masks, slot, points, np).all() or
                        len(np.unique(np.rint(points), axis=0)) != len(points)):
        raise ValueError("Seed points must occupy unique pixels within an exclusive source mask")
    return points


def _load_masks(masks_parent, row, index, pts, valid, slots, record, np):
    if row is None or row["status"] != "completed" or row["video_pts"] != pts:
        raise ValueError("Mask frame absent or source PTS differs")
    if masks_parent[1]["artifacts"].get(row["artifact"]) != row["sha256"]:
        raise ValueError("Mask frame hash differs from its producer artifact inventory")
    with np.load(artifact(masks_parent[0], row["artifact"], record, row["sha256"]), allow_pickle=False) as data:
        masks = data["masks"].copy()
        if int(data["frame_index"]) != index or int(data["pts"]) != pts:
            raise ValueError("Mask array frame/PTS mismatch")
    if masks.dtype != np.dtype("bool") or masks.shape != (slots, *valid.shape) or (masks & ~valid).any():
        raise ValueError("Mask arrays differ from the admitted grid")
    return masks


def _identities(tracks, instances, segments, gaps, minimum_support):
    """Per-roster support for this frame; a newly unsupported identity opens a discontinuity."""
    identities = []
    for name in instances:
        supported = sum(track["roster_id"] == name and track["geometric_accepted"] for track in tracks) >= minimum_support
        if not supported and gaps[name] == 0:
            segments[name] += 1
        gaps[name] = 0 if supported else gaps[name] + 1
        identities.append(dict(roster_id=name, visibility="observed" if supported else "unknown",
                               identity_supported=supported, gap_frames=gaps[name], discontinuity_id=segments[name],
                               rejection_reason=None if supported else "complete_loss_or_insufficient_geometric_support"))
    return identities


def correspondences(request: CorrespondencesRequest):
    runtime = runtime_flag(FEATURE, "correspondences_authorized", "Identity correspondence execution")
    if request.episode_index not in runtime["authorized_episode_indices"].get("public", []):
        raise PermissionError("Correspondences require explicit episode authorization")
    depth = producer(request.depth_run, "depth.generate", request)
    masks_parent = producer(request.masks_run, "object_masks.generate", request)
    require_input(masks_parent, depth)
    revision = same_revision((depth, masks_parent), dataset_revision())
    grid = depth[1]["preprocessing_artifact_sha256"]
    if masks_parent[1].get("grid") != grid:
        raise ValueError("Correspondence parents use different grids")
    record = run_fields(ROOT, "object_masks", "correspondences", asdict(request),
                        parents=[depth[0], masks_parent[0]], allowed=("video_only",),
                        inputs=[FEATURE / "runtime.json", "data/reports/data-audit/videos.json",
                                "data/reports/data-audit/trajectories.json", "data/configs/splits/public-v1.json"])
    record["dataset_revision"] = revision
    names, clock = _episode(request, record)
    instances = [*names, "hand_0", "hand_1"]
    if masks_parent[1]["instances"] != instances:
        raise ValueError("Mask roster order differs")
    if len(instances) > 16:
        raise ValueError("Bound correspondence work to sixteen roster slots")
    stop = request.start_frame + request.frame_count
    if stop > clock["decoded_frame_count"]:
        raise ValueError("Requested correspondence window exceeds the source clock")
    pts = clock["pts_integer"]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Correspondence PTS must be strictly increasing")
    entry, correction_rows = _corrections(request, masks_parent, grid, instances, stop)
    if entry is not None:
        record["inputs"].append(entry)
    run_id = new_run_id()
    record.update(run_id=run_id, schema_version=1, split="public", episode_index=request.episode_index,
                  grid=grid, instances=instances, frames=[], artifacts={}, method="bounded_lk_image_consensus",
                  identity_policy="incumbent_window_seed_then_no_restart_without_image_correction",
                  geometry_scope="2d_similarity_consensus_not_metric_rigidity", visibility_policy="unsupported_is_unknown",
                  query_namespace=run_id, maximum_wall_seconds=1800)
    started = time.perf_counter()
    with recorded(output_path(ROOT, f"data/outputs/object_masks/public/episode_{request.episode_index:06d}/{run_id}"),
                  record) as (directory, record):
        import cv2
        import numpy as np

        with np.load(artifact(depth[0], "preprocessing.npz", record, grid), allow_pickle=False) as data:
            valid = data["input_valid"].copy()
        if valid.ndim != 2 or valid.dtype != np.dtype("bool"):
            raise ValueError("Invalid source grid validity")
        rows = {row["frame_index"]: row for row in masks_parent[1]["frames"]}
        if len(rows) != len(masks_parent[1]["frames"]):
            raise ValueError("Duplicate mask frames")
        tracks, previous, serial = [], None, 0
        segments, gaps = dict.fromkeys(instances, 0), dict.fromkeys(instances, 0)
        for index in range(request.start_frame, stop):
            if time.perf_counter() - started > record["maximum_wall_seconds"]:
                raise RuntimeError("Correspondence 1800-second wall budget exhausted; no automatic retry")
            masks = _load_masks(masks_parent, rows.get(index), index, pts[index], valid, len(instances), record, np)
            rgb_name = f"input_frames/{index:06d}.jpg"
            rgb_hash = masks_parent[1]["artifacts"][rgb_name]
            current = cv2.imread(str(artifact(masks_parent[0], rgb_name, record, rgb_hash)), cv2.IMREAD_GRAYSCALE)
            if current is None or current.shape != valid.shape:
                raise ValueError("Prepared RGB cannot be decoded on the admitted grid")
            tracks = _advance(previous, current, tracks, masks, instances, request, cv2, np) if previous is not None else []
            for slot, name in enumerate(instances):
                correction = correction_rows.get((index, name))
                if index != request.start_frame and correction is None:
                    continue
                points = _seed_points(current, masks, slot, correction, pts[index], rgb_hash, request, cv2, np)
                if correction:
                    segments[name] += 1
                    tracks = [track for track in tracks if track["roster_id"] != name]
                if len(points) < request.minimum_support:
                    continue
                for point in points:
                    serial += 1
                    tracks.append(dict(query_id=f"q{serial:06d}", roster_id=name, origin_frame_index=index,
                                       origin_video_pts=pts[index],
                                       origin_xy=point.astype(float).tolist(), xy=point.astype(float).tolist(),
                                       visible=True, geometric_accepted=True, rejection_reason=None,
                                       discontinuity_id=segments[name], support_kind="manual_image_seed" if correction else "incumbent_mask_seed"))
            document = dict(schema_version=1, frame_index=index, video_pts=pts[index], grid=grid,
                            rgb_sha256=rgb_hash, tracks=tracks,
                            identities=_identities(tracks, instances, segments, gaps, request.minimum_support),
                            corrections=[value for (frame, _), value in correction_rows.items() if frame == index])
            name = f"frames/frame_{index:06d}.json"
            write_json(directory / name, document)
            digest = sha256_file(directory / name)
            record["artifacts"][name] = digest
            record["frames"].append(dict(frame_index=index, video_pts=pts[index], status="completed", artifact=name, sha256=digest))
            previous = current
            write_json(directory / "run.json", record)
    return directory / "run.json"
