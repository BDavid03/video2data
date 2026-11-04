"""CPU frame-to-map camera tracking, metric table plane and explicit world gauge."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from src.paths import output_path
from src.runs import save_npz as _save
from src.provenance import input_file, read_run, run_fields, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
# A feature becomes a new landmark only when no existing landmark projects this close to it.
NEW_LANDMARK_RADIUS_PX = 8


def _path(value):
    path = (ROOT / Path(value)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Camera-motion inputs must stay inside the repository")
    return path


def _json(value):
    return json.loads(_path(value).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class EstimateCameraRequest:
    source_split: str = "public"
    episode_index: int = 39
    depth_run: str | None = None
    stereo_run: str | None = None
    anchor_run: str | None = None
    masks_run: str | None = None
    opening_frame_count: int | None = 20
    slot0_point_camA_m: list[float] | None = None
    slot0_point_source: str | None = None
    feature_budget: int = 4000
    map_budget: int = 20000
    minimum_inliers: int = 12
    mask_dilation: int = 7
    matching_ratio: float = 0.75
    reprojection_limit_px: float = 3.0
    plane_distance_m: float = 0.01
    plane_iterations: int = 500
    plane_minimum_points: int = 200
    plane_sample_budget: int = 12000
    plane_max_tilt_degrees: float = 65.0
    lower_image_fraction: float = 0.5
    seed: int = 31

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation"):
            raise ValueError("source_split must be public or evaluation")
        for name in ("episode_index", "feature_budget", "map_budget", "minimum_inliers", "mask_dilation",
                     "plane_iterations", "plane_minimum_points", "plane_sample_budget", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name in ("episode_index", "seed") else 1):
                raise ValueError(f"Invalid camera request field: {name}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if self.opening_frame_count is not None and (type(self.opening_frame_count) is not int or self.opening_frame_count < 1):
            raise ValueError("opening_frame_count must be positive or null for the full clip")
        if self.minimum_inliers < 6 or self.map_budget < self.feature_budget or self.plane_minimum_points < 3:
            raise ValueError("Invalid feature, inlier or plane point budget")
        if self.plane_sample_budget < self.plane_minimum_points:
            raise ValueError("Plane sampling budget is below the minimum support")
        limits = (self.matching_ratio, self.reprojection_limit_px, self.plane_distance_m,
                  self.plane_max_tilt_degrees, self.lower_image_fraction)
        if (not all(math.isfinite(value) for value in limits) or not 0 < self.matching_ratio < 1 or
                self.reprojection_limit_px <= 0 or self.plane_distance_m <= 0 or
                not 0 < self.plane_max_tilt_degrees < 90 or not 0 < self.lower_image_fraction < 1):
            raise ValueError("Invalid camera-motion thresholds")
        if self.slot0_point_camA_m is not None and (len(self.slot0_point_camA_m) != 3 or
                not all(math.isfinite(value) for value in self.slot0_point_camA_m) or self.slot0_point_camA_m[2] <= 0):
            raise ValueError("slot0_point_camA_m must be a finite point in front of the first camera")


def _authorize(request):
    runtime = _json(FEATURE / "runtime.json")
    if runtime["execution"].get("estimate_authorized") is not True:
        raise PermissionError("Camera-motion estimation remains disabled")
    if request.source_split == "evaluation" and runtime["execution"].get("evaluation_authorized") is not True:
        raise PermissionError("Evaluation camera motion remains disabled")
    split = _json("data/configs/splits/public-v1.json")
    if request.source_split == "public":
        if request.episode_index in split["validation_episode_indices"]:
            if runtime["execution"].get("validation_authorized") is not True:
                raise PermissionError("Public validation camera motion remains disabled")
        elif request.episode_index not in split["development_episode_indices"]:
            raise ValueError("Episode is outside the frozen public split")
    if request.episode_index not in runtime["authorized_episode_indices"].get(request.source_split, []):
        raise PermissionError("Camera motion requires authorization for the named split and episode")


def _producer(value, kind, request):
    if value is None:
        raise ValueError(f"Supply an explicit completed {kind} run.json")
    path = _path(value)
    record = read_run(ROOT, path, kind=kind, allowed=("video_only",))
    if record.get("split") != request.source_split or record.get("episode_index") != request.episode_index:
        raise ValueError("Camera input belongs to a different split or episode")
    return path, record


def _artifact(producer, name, digest, record):
    path = _path(producer.parent / name)
    if not path.is_relative_to(producer.parent):
        raise ValueError("Artifact escapes its run directory")
    entry = input_file(ROOT, path)
    if entry["sha256"] != digest:
        raise ValueError(f"Input artifact differs from its producer: {path}")
    if entry not in record["inputs"]:
        record["inputs"].append(entry)
    return path


def _frame(parent, index, pts, record, np):
    path, source = parent
    rows = [row for row in source["frames"] if row["frame_index"] == index]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["video_pts"] != pts:
        raise ValueError("Missing, incomplete or mismatched camera input frame")
    path = _artifact(path, rows[0]["artifact"], rows[0]["sha256"], record)
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _points(depth, k, np):
    v, u = np.indices(depth.shape)
    return np.stack(((u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth), axis=-1)


def _plane(points, request, np):
    if len(points) < request.plane_minimum_points:
        raise ValueError("no_support_plane: insufficient static points in the lower image")
    rng = np.random.default_rng(request.seed)
    if len(points) > request.plane_sample_budget:
        points = points[rng.choice(len(points), request.plane_sample_budget, replace=False)]
    best = None
    up = np.array([0., -1., 0.])
    minimum_cosine = math.cos(math.radians(request.plane_max_tilt_degrees))
    for _ in range(request.plane_iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal @ up < 0:
            normal = -normal
        if normal @ up < minimum_cosine:
            continue
        mask = abs((points - a) @ normal) <= request.plane_distance_m
        if best is None or int(mask.sum()) > int(best.sum()):
            best = mask
    if best is None or int(best.sum()) < request.plane_minimum_points:
        raise ValueError("no_support_plane: no dominant near-horizontal plane")
    center = np.median(points[best], axis=0)
    _, _, vt = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = vt[-1]
    if normal @ up < 0:
        normal = -normal
    if normal @ up < minimum_cosine:
        raise ValueError("no_support_plane: refined plane exceeds the tilt limit")
    offset = -float(np.median(points[best] @ normal))
    inliers = abs(points @ normal + offset) <= request.plane_distance_m
    if int(inliers.sum()) < request.plane_minimum_points:
        raise ValueError("no_support_plane: refined plane lost support")
    return normal, offset, points[inliers]


def _world(normal, offset, point, np):
    origin = point - (point @ normal + offset) * normal
    heading = np.array([0., 0., 1.])
    heading -= (heading @ normal) * normal
    if np.linalg.norm(heading) < 1e-6:
        raise ValueError("no_support_plane: viewing direction cannot define table heading")
    heading /= np.linalg.norm(heading)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack((heading, np.cross(normal, heading), normal))
    matrix[:3, 3] = -matrix[:3, :3] @ origin
    return matrix


def _inputs(request):
    """Admit the producer runs, RGB clock and video that one tracking run consumes."""
    # The scorer removes one rigid transform, so the origin is free; a slot-0 point needs its source.
    if request.slot0_point_camA_m is not None and not request.slot0_point_source:
        raise ValueError("A supplied slot-0 point needs its label-free source description")
    if request.opening_frame_count is None and request.masks_run is None:
        raise ValueError("Full-clip tracking requires masks; unmasked tracking is opening-only")
    depth = _producer(request.depth_run, "depth.generate", request)
    stereo = _producer(request.stereo_run, "stereo_depth.generate", request)
    anchor = _producer(request.anchor_run, "stereo_depth.anchor", request)
    masks = _producer(request.masks_run, None, request) if request.masks_run else None
    if masks and masks[1]["kind"] not in ("object_masks.generate", "object_masks.opening_clusters"):
        raise ValueError("Expected an object mask producer")
    parents = [depth, stereo, anchor] + ([masks] if masks else [])
    source = _json("data/configs/sources.lock.json")["dataset"]
    revision = source["revision"]
    inventory = _json("data/reports/data-audit/videos.json")
    rows = [row for row in inventory["videos"] if row["split"] == request.source_split
            and row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if inventory["status"] != "completed" or inventory["revision"] != revision or len(rows) != 1 or rows[0]["status"] != "ok":
        raise ValueError("Camera motion requires a completed RGB clock inventory")
    clock = rows[0]
    count = clock["decoded_frame_count"]
    if (clock.get("rotation_metadata_lines") or clock["time_base_and_frame_rate"] != [["1/10240", "20/1"]]
            or len(clock["pts_integer"]) != count or len(clock["pts_seconds"]) != count):
        raise ValueError("Unsupported RGB clock or orientation metadata")
    if any(parent["dataset_revision"] != revision for _, parent in parents):
        raise ValueError("Camera input revisions differ")
    stop = count if request.opening_frame_count is None else request.opening_frame_count
    if stop > count:
        raise ValueError("Opening interval exceeds the clip")
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(parent.get("grid") != grid for _, parent in parents[1:]):
        raise ValueError("Camera inputs use different colour pixel grids")
    # The anchor's scale must belong to this depth run and this stereo run.
    for named, expected in ((anchor[1]["request"].get("depth_run"), depth[0]),
                            (anchor[1]["request"].get("stereo_run"), stereo[0]),
                            (stereo[1]["request"].get("depth_run"), depth[0])):
        if named is None or _path(named) != _path(expected):
            raise ValueError("Anchor, stereo and depth runs were not built from one another")
    video = _path(Path(source["cache_path"]) / "snapshots" / revision / clock["path"])
    if _path(depth[1]["source_video"]) != video:
        raise ValueError("Depth input names a different RGB video")
    return dict(depth=depth, stereo=stereo, anchor=anchor, masks=masks, parents=parents, revision=revision,
                clock=clock, count=count, stop=stop, grid=grid, video=video)


def _metric_depth(mono, metric, scale_row, pts, shape, np):
    """Anchored monocular depth where the anchor observed a scale, overwritten by valid stereo depth."""
    z = np.full(shape, np.nan, dtype=np.float64)
    if scale_row and scale_row["video_pts"] == pts and scale_row["scale_status"] == "observed":
        scale = scale_row["scale"]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid metric scale")
        valid = mono["valid"].astype(bool) & np.isfinite(mono["depth_m"]) & (mono["depth_m"] > 0)
        z[valid] = mono["depth_m"][valid] * scale
    valid = metric["rgb_valid"].astype(bool) & np.isfinite(metric["rgb_depth_m"]) & (metric["rgb_depth_m"] > 0)
    z[valid] = metric["rgb_depth_m"][valid]
    return z


def _define_world(xyz, depth_valid, request, scale_source, grid, record, directory, np):
    """Fit the lower-image table plane on frame 0, fix the world gauge and write world.json."""
    lower = np.indices(xyz.shape[:2])[0] >= round(xyz.shape[0] * (1 - request.lower_image_fraction))
    normal, offset, plane_points = _plane(xyz[depth_valid & lower], request, np)
    if request.slot0_point_camA_m is not None:
        origin_point = np.asarray(request.slot0_point_camA_m, dtype=np.float64)
        origin_rule = "Table projection of the explicitly supplied first-frame slot-0 point"
    else:
        # Where the principal ray of the first camera meets the table.
        if normal[2] >= -1e-6 or offset <= 0:
            raise ValueError("no_support_plane: the first principal ray does not meet the table")
        origin_point = np.array([0., 0., -offset / normal[2]])
        origin_rule = "Point where the principal ray of the first camera meets the table"
    world = _world(normal, offset, origin_point, np)
    plane_world = plane_points @ world[:3, :3].T + world[:3, 3]
    definition = {"up_axis": "+z", "support_plane": {"normal": [0, 0, 1], "offset_m": 0.0,
                  "inlier_count": len(plane_points), "extent_xy_m": [plane_world[:, :2].min(axis=0).tolist(),
                                                                     plane_world[:, :2].max(axis=0).tolist()]},
                  "origin_rule": origin_rule,
                  "origin_source": request.slot0_point_source,
                  "heading_rule": "First camera viewing direction projected onto the table",
                  "scale_source": scale_source,
                  "camera0_plane": {"normal": normal.tolist(), "offset_m": offset},
                  "world_T_camA0": world.tolist(), "table_level_assumed": True,
                  "grid": grid, "provenance": record["provenance"]}
    write_json(directory / "world.json", definition)
    record["artifacts"]["world.json"] = sha256_file(directory / "world.json")
    return world, normal, offset


def _features(keypoints, z, depth_valid, k, np):
    """Keypoint pixels, whether each has valid static depth, and its camera-frame point."""
    pixels = np.asarray([point.pt for point in keypoints], dtype=np.float64)
    nearest = np.rint(pixels).astype(np.int64)
    good = ((nearest[:, 0] >= 0) & (nearest[:, 0] < z.shape[1]) &
            (nearest[:, 1] >= 0) & (nearest[:, 1] < z.shape[0]))
    nearest[:, 0] = np.clip(nearest[:, 0], 0, z.shape[1] - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, z.shape[0] - 1)
    good &= depth_valid[nearest[:, 1], nearest[:, 0]]
    feature_z = z[nearest[:, 1], nearest[:, 0]]
    camera_points = np.column_stack(((pixels[:, 0] - k[0, 2]) * feature_z / k[0, 0],
                                     (pixels[:, 1] - k[1, 2]) * feature_z / k[1, 1], feature_z))
    return pixels, good, camera_points


def _solve_pose(descriptors, pixels, good, map_points, map_descriptors, matcher, k, request, cv2, np):
    """Ratio-tested one-to-one map matches, EPnP RANSAC and LM refinement.

    Returns (world_T_camA or None, inlier count, median residual px, matched query indices).
    """
    candidates = [pair[0] for pair in matcher.knnMatch(descriptors, map_descriptors, k=2)
                  if len(pair) == 2 and good[pair[0].queryIdx]
                  and pair[0].distance < request.matching_ratio * pair[1].distance]
    matches, used = [], set()
    for match in sorted(candidates, key=lambda item: item.distance):
        if match.trainIdx not in used:
            matches.append(match)
            used.add(match.trainIdx)
    matched_queries = {match.queryIdx for match in matches}
    if len(matches) < request.minimum_inliers:
        return None, None, None, matched_queries
    object_points = np.asarray([map_points[match.trainIdx] for match in matches])
    image_points = np.asarray([pixels[match.queryIdx] for match in matches])
    solved, rv, tv, inliers = cv2.solvePnPRansac(object_points, image_points, k, None,
        iterationsCount=200, reprojectionError=request.reprojection_limit_px,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not solved or inliers is None or len(inliers) < request.minimum_inliers:
        return None, None, None, matched_queries
    ids = inliers.ravel()
    rv, tv = cv2.solvePnPRefineLM(object_points[ids], image_points[ids], k, None, rv, tv)
    rotation = cv2.Rodrigues(rv)[0]
    projected = cv2.projectPoints(object_points[ids], rv, tv, k, None)[0].reshape(-1, 2)
    residual = float(np.median(np.linalg.norm(projected - image_points[ids], axis=1)))
    if (not math.isfinite(residual) or residual > request.reprojection_limit_px or
            not ((object_points[ids] @ rotation.T + tv.ravel())[:, 2] > 0).all()):
        return None, len(ids), residual, matched_queries
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3], pose[:3, 3] = rotation.T, -rotation.T @ tv.ravel()
    return pose, len(ids), residual, matched_queries


def _unoccupied(map_points, pose, pixels, shape, k, np):
    """Features with no existing landmark projecting into their own or a neighbouring grid cell.

    Re-observed landmarks that merely failed the ratio test must not be added again:
    duplicates defeat the ratio test and exhaust the map budget.
    """
    seen = (map_points - pose[:3, 3]) @ pose[:3, :3]
    front = seen[:, 2] > 1e-6
    cells_w = shape[1] // NEW_LANDMARK_RADIUS_PX + 3
    cells_h = shape[0] // NEW_LANDMARK_RADIUS_PX + 3
    occupied = np.zeros((cells_h, cells_w), dtype=bool)
    px = seen[front, 0] / seen[front, 2] * k[0, 0] + k[0, 2]
    py = seen[front, 1] / seen[front, 2] * k[1, 1] + k[1, 2]
    inside = (px >= 0) & (px < shape[1]) & (py >= 0) & (py < shape[0])
    cu = (px[inside] // NEW_LANDMARK_RADIUS_PX).astype(np.int64) + 1
    cv = (py[inside] // NEW_LANDMARK_RADIUS_PX).astype(np.int64) + 1
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            occupied[cv + dv, cu + du] = True
    fu = np.clip(pixels[:, 0] // NEW_LANDMARK_RADIUS_PX, 0, cells_w - 3).astype(np.int64) + 1
    fv = np.clip(pixels[:, 1] // NEW_LANDMARK_RADIUS_PX, 0, cells_h - 3).astype(np.int64) + 1
    return ~occupied[fv, fu]


def estimate(request: EstimateCameraRequest):
    _authorize(request)
    inputs = _inputs(request)
    depth, stereo, anchor, masks = inputs["depth"], inputs["stereo"], inputs["anchor"], inputs["masks"]
    clock, count, grid, video = inputs["clock"], inputs["count"], inputs["grid"], inputs["video"]
    record = run_fields(ROOT, "camera_motion", "estimate", asdict(request), parents=[path for path, _ in inputs["parents"]],
                        allowed=("video_only",), inputs=[video, "data/reports/data-audit/videos.json",
                        "data/configs/sources.lock.json", "data/configs/splits/public-v1.json", "data/configs/calibration/ego.json",
                        "data/configs/perception/moge/pylock.python313.toml", FEATURE / "runtime.json"])
    video_entry = next(entry for entry in record["inputs"] if entry["path"] == video.relative_to(ROOT).as_posix())
    if video_entry["sha256"] != depth[1]["input_hashes"]["video_sha256"]:
        raise ValueError("RGB video bytes differ from depth input")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"data/outputs/camera_motion/{request.source_split}/episode_{request.episode_index:06d}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=inputs["revision"], grid=grid, frames=[], artifacts={}, gpu=None,
                  opening_frame_count=request.opening_frame_count)
    started, capture, arrays = time.perf_counter(), None, None
    write_json(directory / "run.json", record)
    try:
        import cv2
        import numpy as np

        cv2.setRNGSeed(request.seed)
        grid_file = _artifact(depth[0], "preprocessing.npz", grid, record)
        with np.load(grid_file, allow_pickle=False) as loaded:
            map_x, map_y = loaded["source_x"].copy(), loaded["source_y"].copy()
            input_valid, k = loaded["input_valid"].astype(bool), loaded["input_intrinsics_pixels"].astype(np.float64)
        anchor_file = _artifact(anchor[0], "anchor.json", anchor[1]["artifacts"]["anchor.json"], record)
        scales = {row["frame_index"]: row for row in _json(anchor_file)["frames"]}
        arrays = {"world_T_camA": np.full((count, 4, 4), np.nan, dtype=np.float64),
                  "valid": np.zeros(count, dtype=bool), "observed": np.zeros(count, dtype=bool),
                  "inliers": np.zeros(count, dtype=np.int32), "reprojection_px": np.full(count, np.nan, dtype=np.float32),
                  "frame_index": np.arange(count, dtype=np.int64), "pts": np.asarray(clock["pts_integer"], dtype=np.int64)}
        detector = cv2.ORB.create(nfeatures=request.feature_budget)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError("Could not open RGB video")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        map_points = map_descriptors = world = None
        for index in range(inputs["stop"]):
            pts = clock["pts_integer"][index]
            row = {"frame_index": index, "video_pts": pts, "status": "running"}
            record["frames"].append(row)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at frame {index}")
            if list(image.shape[1::-1]) != depth[1]["preprocessing"]["source_size_width_height"]:
                raise ValueError("RGB dimensions differ from the depth preprocessing")
            image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mono, metric = _frame(depth, index, pts, record, np), _frame(stereo, index, pts, record, np)
            if (mono["depth_m"].shape != input_valid.shape or metric["rgb_depth_m"].shape != input_valid.shape or
                    not np.allclose(k, mono["input_intrinsics_pixels"], atol=1e-5) or
                    not np.allclose(k, metric["rgb_intrinsics_pixels"], atol=1e-5)):
                raise ValueError("Camera depth dimensions or intrinsics disagree")
            z = _metric_depth(mono, metric, scales.get(index), pts, input_valid.shape, np)
            static = input_valid.copy()
            if masks:
                layers = _frame(masks, index, pts, record, np)["masks"].astype(bool)
                if layers.ndim != 3 or layers.shape[1:] != z.shape:
                    raise ValueError("Camera masks differ from the pixel grid")
                excluded = cv2.dilate(layers.any(axis=0).astype(np.uint8),
                                      np.ones((request.mask_dilation, request.mask_dilation), dtype=np.uint8))
                static &= excluded == 0
            xyz = _points(z, k, np)
            depth_valid = static & np.isfinite(xyz).all(axis=-1) & (z > 0)
            if index == 0:
                world, normal, offset = _define_world(xyz, depth_valid, request, str(anchor[0].relative_to(ROOT).as_posix()),
                                                      grid, record, directory, np)
                if not masks:
                    static &= np.isfinite(z) & (abs(xyz @ normal + offset) <= request.plane_distance_m)
            keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                                               (static.astype(np.uint8) * 255))
            if descriptors is None or len(keypoints) < request.minimum_inliers:
                row.update(status="completed", pose_status="insufficient_depth")
                if index == 0:
                    raise ValueError("insufficient_depth: cannot initialise a static landmark map")
                write_json(directory / "run.json", record)
                continue
            pixels, good, camera_points = _features(keypoints, z, depth_valid, k, np)
            matched_queries = set()
            if index == 0:
                if int(good.sum()) < request.minimum_inliers:
                    raise ValueError("insufficient_depth: too few initial metric landmarks")
                pose, inlier_count, residual = world, int(good.sum()), 0.0
            else:
                pose, inlier_count, residual, matched_queries = _solve_pose(
                    descriptors, pixels, good, map_points, map_descriptors, matcher, k, request, cv2, np)
            if pose is None:
                row.update(status="completed", pose_status="tracking_lost")
            else:
                arrays["world_T_camA"][index] = pose
                arrays["valid"][index] = arrays["observed"][index] = True
                arrays["inliers"][index], arrays["reprojection_px"][index] = inlier_count, residual
                row.update(status="completed", pose_status="observed", inliers=inlier_count, reprojection_px=residual)
                world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
                if not masks:
                    good &= abs(world_points[:, 2]) <= request.plane_distance_m
                good[list(matched_queries)] = False
                if map_points is not None:
                    good &= _unoccupied(map_points, pose, pixels, z.shape, k, np)
                ids = np.flatnonzero(good)
                room = request.map_budget - (0 if map_points is None else len(map_points))
                ids = ids[:max(0, room)]
                if len(ids):
                    map_points = world_points[ids] if map_points is None else np.concatenate((map_points, world_points[ids]))
                    map_descriptors = descriptors[ids] if map_descriptors is None else np.concatenate((map_descriptors, descriptors[ids]))
                if index == 0 and (map_points is None or len(map_points) < request.minimum_inliers):
                    raise ValueError("insufficient_depth: too few static landmarks survived world placement")
            _save(directory / "trajectory.npz", np, **arrays)
            record["artifacts"]["trajectory.npz"] = sha256_file(directory / "trajectory.npz")
            write_json(directory / "run.json", record)
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        for row in record["frames"]:
            if row["status"] == "running":
                row.update(status=record["status"], error=record["error"])
        raise
    finally:
        if capture is not None:
            capture.release()
        if arrays is not None:
            _save(directory / "trajectory.npz", np, **arrays)
            record["artifacts"]["trajectory.npz"] = sha256_file(directory / "trajectory.npz")
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        write_json(directory / "run.json", record)
    return directory / "run.json"
