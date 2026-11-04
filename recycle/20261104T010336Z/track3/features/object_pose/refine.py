"""Disabled bounded SE(3) refinement from explicitly admitted image evidence."""

from dataclasses import dataclass
import math

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, arrays, artifact, authorize, frame_arrays, local_path, operation_run,
                      producer, read_json, require_input, require_rigid, save_npz)

EPISODES = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)


@dataclass(frozen=True)
class PoseRefinementRequest:
    source_split: str = "public"
    episode_index: int = 39
    pose_run: str | None = None
    camera_run: str | None = None
    correspondence_run: str | None = None
    depth_run: str | None = None
    observations_file: str | None = None
    observations_sha256: str | None = None
    window_frames: int = 16
    maximum_frames: int = 512
    maximum_points_per_frame: int = 128
    maximum_nfev: int = 40
    minimum_points: int = 6
    maximum_translation_m: float = 0.01
    maximum_rotation_rad: float = 0.05
    rapid_translation_m: float = 0.05
    rapid_rotation_rad: float = 0.25
    temporal_weight: float = 0.1
    correction_weight: float = 0.1
    maximum_gap_seconds: float = 0.15

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split != "public" or self.episode_index not in EPISODES:
            raise PermissionError("Pose refinement is restricted to training/tuning; 41/43/44 excluded")
        for name, ceiling in (("window_frames", 32), ("maximum_frames", 512),
                              ("maximum_points_per_frame", 256), ("maximum_nfev", 80),
                              ("minimum_points", 256)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"Invalid bounded refinement field: {name}")
        if not 6 <= self.minimum_points <= self.maximum_points_per_frame:
            raise ValueError("At least six supported body correspondences are required")
        for name in ("maximum_translation_m", "maximum_rotation_rad", "rapid_translation_m",
                     "rapid_rotation_rad", "temporal_weight", "correction_weight", "maximum_gap_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid refinement limit: {name}")
        if not self.observations_file or not self.observations_sha256:
            raise ValueError("Supply a separately admitted correspondence receipt and SHA256")


def _check_body_origin(origin, track, point, slot, expected, result, depth, record, np):
    """A body point must be the pinned stereo backprojection of its accepted query's origin pixel."""
    if (origin.get("depth_run_sha256") != expected[3]["sha256"] or
            origin.get("pose_run_sha256") != expected[0]["sha256"] or
            origin.get("observed") is not True or origin.get("metric_valid") is not True):
        raise ValueError("Body point lacks pinned metric origin evidence")
    origin_index = origin["frame_index"]
    if (type(origin_index) is not int or not 0 <= origin_index < len(result["pts"])
            or origin_index != track["origin_frame_index"]
            or not result["valid"][origin_index, slot]
            or origin["video_pts"] != int(result["pts"][origin_index])):
        raise ValueError("Body-point origin lacks a valid measured pose clock")
    measured = frame_arrays(depth, origin_index, origin["video_pts"], record, np)
    u, v = origin["pixel_xy"]
    h, w = measured["rgb_depth_m"].shape
    if (type(u) is not int or type(v) is not int or not 0 <= u < w or not 0 <= v < h
            or not measured["rgb_valid"][v, u]):
        raise ValueError("Body origin lacks valid direct stereo support")
    if not np.array_equal(np.rint(track["origin_xy"]).astype(int), [u, v]):
        raise ValueError("Body point does not originate at its accepted identity query")
    origin_z = float(measured["rgb_depth_m"][v, u])
    if not math.isfinite(origin_z) or origin_z <= 0:
        raise ValueError("Body origin lacks positive metric depth")
    camera_point = np.linalg.solve(measured["rgb_intrinsics_pixels"], [u, v, 1]) * origin_z
    object_from_camera = np.linalg.inv(result["camA_T_object"][origin_index, slot])
    body_point = object_from_camera[:3, :3] @ camera_point + object_from_camera[:3, 3]
    if not np.allclose(body_point, point, atol=1e-6, rtol=0):
        raise ValueError("Body point differs from its pinned measured backprojection")


def _observed_depths(row, keep, pixels, k, index, depth, record, np):
    """Receipt depths (NaN for null), each matching same-pixel direct stereo depth."""
    current_depth = frame_arrays(depth, index, row["video_pts"], record, np)
    if not np.allclose(k, current_depth["rgb_intrinsics_pixels"], atol=1e-6, rtol=0):
        raise ValueError("Receipt intrinsics differ from the selected metric grid")
    z = np.asarray([np.nan if value is None else value for value in row["depth_m"]], dtype=float)
    if (np.isinf(z).any() or (z[np.isfinite(z)] <= 0).any()):
        raise ValueError("Depth observations must be positive metres or null")
    h, w = current_depth["rgb_depth_m"].shape
    for j in np.flatnonzero(keep & np.isfinite(z)):
        u, v = np.rint(pixels[j]).astype(int)
        if (not 0 <= u < w or not 0 <= v < h or not current_depth["rgb_valid"][v, u]
                or not np.isclose(z[j], current_depth["rgb_depth_m"][v, u], atol=1e-6, rtol=0)):
            raise ValueError("Depth residual lacks same-pixel direct metric evidence")
    return z


def admit_refinement_observations(request, parents, result, motion, record, np):
    """Admit a pinned receipt's measured body correspondences, keyed (frame, slot), without fitting.

    ``parents`` is ordered compose poses, camera motion, image tracks, direct stereo.
    """
    poses, camera, tracks, depth = parents
    entry = input_file(ROOT, local_path(request.observations_file))
    if entry["sha256"] != request.observations_sha256:
        raise ValueError("Correspondence receipt digest differs")
    receipt = read_json(request.observations_file)
    if (receipt.get("schema_version") != 1 or receipt.get("provenance") != "video_only" or
            receipt.get("body_point_origin") != "measured_depth_backprojection" or
            receipt.get("grid") != poses[1]["grid"] or
            receipt.get("mesh_body_frames") != poses[1]["mesh_body_frames"] or
            receipt.get("split") != request.source_split or
            receipt.get("episode_index") != request.episode_index):
        raise ValueError("Receipt lacks admitted image-only body geometry and exact body frames")
    expected = [input_file(ROOT, parent[0]) for parent in parents]
    if receipt.get("parents") != expected:
        raise ValueError("Receipt must pin selected pose, camera, correspondence and metric-depth manifests in order")
    record["inputs"].append(entry)
    names = result["object_names"].tolist()
    observations = {}
    for row in receipt.get("observations", []):
        index, name = row["frame_index"], row["object_name"]
        if type(index) is not int or not 0 <= index < len(result["pts"]) or name not in names:
            raise ValueError("Receipt refers to an unknown frame/object")
        slot = names.index(name)
        if row["video_pts"] != int(result["pts"][index]) or (index, slot) in observations:
            raise ValueError("Duplicate or mismatched correspondence clock")
        if not result["world_valid"][index, slot] or not motion["valid"][index]:
            continue
        frame = [item for item in tracks[1]["frames"] if item["frame_index"] == index]
        if len(frame) != 1 or frame[0]["video_pts"] != row["video_pts"] or frame[0]["status"] != "completed":
            raise ValueError("Missing completed identity evidence at receipt clock")
        identity = read_json(artifact(tracks[0], frame[0]["artifact"], record, frame[0]["sha256"]))
        accepted = {item["query_id"]: item for item in identity["tracks"]
                    if item["roster_id"] == name and item["visible"] and item["geometric_accepted"]}
        points = np.asarray(row["object_points_m"], dtype=float)
        pixels = np.asarray(row["image_points_px"], dtype=float)
        k = np.asarray(row["intrinsics_pixels"], dtype=float)
        visible = np.asarray(row["visible"])
        supported = np.asarray(row["supported"])
        n = len(points)
        if (n > request.maximum_points_per_frame or points.shape != (n, 3) or pixels.shape != (n, 2)
                or visible.shape != (n,) or supported.shape != (n,) or visible.dtype.kind != "b"
                or supported.dtype.kind != "b" or k.shape != (3, 3) or len(row["query_ids"]) != n
                or len(set(row["query_ids"])) != n
                or len(row["depth_m"]) != n or len(row["origin_evidence"]) != n
                or not np.isfinite(points).all() or not np.isfinite(pixels).all() or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Malformed calibrated correspondence arrays")
        keep = visible & supported
        for j in np.flatnonzero(keep):
            track = accepted.get(row["query_ids"][j])
            if track is None or not np.allclose(track["xy"], pixels[j], atol=1e-4, rtol=0):
                raise ValueError("Body correspondence lacks an accepted same-image identity track")
            _check_body_origin(row["origin_evidence"][j], track, points[j], slot, expected, result, depth, record, np)
        z = _observed_depths(row, keep, pixels, k, index, depth, record, np)
        sigmas = (row["pixel_sigma"], row["depth_sigma_m"])
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in sigmas):
            raise ValueError("Residual scales must be positive and finite")
        if keep.sum() < request.minimum_points or np.linalg.matrix_rank(points[keep] - points[keep].mean(0)) < 2:
            continue
        observations[index, slot] = (points[keep], pixels[keep], z[keep], k, *sigmas)
    return receipt, observations


def _delta(vector, np, Rotation):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    transform[:3, 3] = vector[3:]
    return transform


def _image_residual(x, observed, cameras, baseline, np, Rotation):
    """Pixel, depth and in-front residuals of corrected world poses for one window."""
    values = []
    for j, (points, pixels, z, k, sigma_px, sigma_z) in enumerate(observed):
        pose = cameras[j] @ baseline[j] @ _delta(x[j], np, Rotation)
        xyz = points @ pose[:3, :3].T + pose[:3, 3]
        projected = xyz @ k.T
        xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
        valid_z = np.isfinite(z)
        values += [((xy - pixels) / sigma_px).ravel(), (xyz[valid_z, 2] - z[valid_z]) / sigma_z,
                   np.minimum(xyz[:, 2] - 1e-6, 0) / sigma_z]
    return np.concatenate(values)


def _residual(flat, observed, cameras, baseline, dt, scales, request, np, Rotation):
    """Image residuals plus correction-size and velocity-change penalties."""
    x = flat.reshape(-1, 6)
    values = [_image_residual(x, observed, cameras, baseline, np, Rotation),
              (request.correction_weight * x / scales).ravel()]
    if len(x) > 1:
        values.append((request.temporal_weight * np.diff(x / scales, axis=0) / dt).ravel())
    return np.concatenate(values)


def _robust_cost(residual, np):
    return float(np.sum(2 * (np.sqrt(1 + residual * residual) - 1)))


def _segments(observations, result, seconds, slot, request, np, Rotation):
    """Split one object's supported frames into bounded windows of consecutive, non-rapid motion."""
    segments, segment = [], []
    for index in range(len(seconds)):
        eligible = (index, slot) in observations and not result["reinitialised"][index, slot]
        if segment and eligible:
            previous = segment[-1]
            a, b = result["world_T_object"][[previous, index], slot]
            eligible_edge = (index == previous + 1 and seconds[index] - seconds[previous] <= request.maximum_gap_seconds
                             and np.linalg.norm(b[:3, 3] - a[:3, 3]) <= request.rapid_translation_m
                             and Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() <= request.rapid_rotation_rad)
            if not eligible_edge:
                segments.append(segment)
                segment = []
        if not eligible:
            if segment:
                segments.append(segment)
                segment = []
            continue
        segment.append(index)
        if len(segment) == request.window_frames:
            segments.append(segment)
            segment = []
    if segment:
        segments.append(segment)
    return segments


def _refine_window(indices, slot, observations, result, motion, seconds, request, np, least_squares, Rotation):
    """Fit bounded corrections for one window; returns its report and (world, camera_T_world, x) updates."""
    baseline = result["unrefined_world_T_object"][indices, slot]
    cameras = np.linalg.inv(motion["world_T_camA"][indices])
    observed = [observations[index, slot] for index in indices]
    scales = np.asarray([request.maximum_rotation_rad] * 3 + [request.maximum_translation_m] * 3)
    dt = np.diff(seconds[indices])[:, None]
    zero = np.zeros((len(indices), 6))
    before = _image_residual(zero, observed, cameras, baseline, np, Rotation)
    fitted = least_squares(_residual, zero.ravel(), bounds=(-np.tile(scales, len(indices)), np.tile(scales, len(indices))),
                           loss="soft_l1", max_nfev=request.maximum_nfev,
                           args=(observed, cameras, baseline, dt, scales, request, np, Rotation))
    x = fitted.x.reshape(-1, 6)
    after = _image_residual(x, observed, cameras, baseline, np, Rotation)
    # A budget exhaustion is retained as a failure; no partial iterate is promoted.
    accepted = bool(fitted.success and np.isfinite(x).all() and _robust_cost(after, np) < _robust_cost(before, np)
                    and (np.linalg.norm(x[:, :3], axis=1) <= request.maximum_rotation_rad).all()
                    and (np.linalg.norm(x[:, 3:], axis=1) <= request.maximum_translation_m).all())
    report = {"object_name": result["object_names"][slot].item(), "frame_indices": indices,
              "accepted": accepted, "solver_status": int(fitted.status), "nfev": int(fitted.nfev),
              "before_robust_image_cost": _robust_cost(before, np), "after_robust_image_cost": _robust_cost(after, np),
              "residual_count": len(before)}
    updates = [(baseline[j] @ _delta(x[j], np, Rotation), cameras[j], x[j]) for j in range(len(indices))] if accepted else []
    return report, updates


def _admit_poses(result, motion, request, np):
    """Composed pose/camera arrays must share one bounded clock, shapes and composition."""
    if (len(result["pts"]) > request.maximum_frames or
            not np.array_equal(result["pts"], motion["pts"]) or
            not np.array_equal(result["frame_index"], motion["frame_index"]) or
            not np.array_equal(result["frame_index"], np.arange(len(result["pts"])))):
        raise ValueError("Pose/camera clock mismatch or frame budget exceeded")
    count, bodies = len(result["pts"]), len(result["object_names"])
    if (result["world_valid"].shape != (count, bodies) or result["valid"].shape != (count, bodies)
            or result["world_T_object"].shape != (count, bodies, 4, 4)
            or result["camA_T_object"].shape != (count, bodies, 4, 4)
            or result["reinitialised"].shape != (count, bodies)
            or motion["valid"].shape != (count,)
            or motion["world_T_camA"].shape != (count, 4, 4)
            or any(result[key].dtype.kind != "b" for key in ("valid", "world_valid", "reinitialised"))
            or motion["valid"].dtype.kind != "b"):
        raise ValueError("Malformed pose/camera arrays")
    require_rigid(result["world_T_object"][result["world_valid"]], np)
    require_rigid(result["camA_T_object"][result["valid"]], np)
    require_rigid(motion["world_T_camA"][motion["valid"]], np)
    if not np.array_equal(result["world_valid"], result["valid"] & motion["valid"][:, None]):
        raise ValueError("Composed pose validity differs from its camera and object observations")
    for index in range(count):
        keep = result["world_valid"][index]
        if keep.any() and not np.allclose(result["world_T_object"][index, keep],
                motion["world_T_camA"][index] @ result["camA_T_object"][index, keep], atol=1e-6, rtol=0):
            raise ValueError("Composed world poses differ from the selected camera trajectory")
    return count, bodies


def refine(request):
    authorize("refine", request)
    poses = producer(request.pose_run, "object_pose.compose", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    tracks = producer(request.correspondence_run, "object_masks.correspondences", request)
    depth = producer(request.depth_run, "stereo_depth.generate", request)
    require_input(poses, camera)
    source_depth = producer(tracks[1]["request"]["depth_run"], "depth.generate", request)
    require_input(tracks, source_depth)
    require_input(depth, source_depth)
    parents = [poses, camera, tracks, depth]
    if any(parent[1].get("grid", poses[1]["grid"]) != poses[1]["grid"] for parent in parents):
        raise ValueError("Refinement producers differ in calibrated image grid")
    with operation_run("refine", request, parents) as (directory, record):
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        result = arrays(poses, "poses.npz", record, np)
        motion = arrays(camera, "trajectory.npz", record, np)
        count, bodies = _admit_poses(result, motion, request, np)
        receipt, observations = admit_refinement_observations(request, parents, result, motion, record, np)
        opening_summary = read_json(artifact(poses[0], "opening.json", record, poses[1]["artifacts"]["opening.json"]))
        alternatives = receipt["symmetry_alternatives"]
        if not isinstance(alternatives, dict) or set(alternatives) != set(result["object_names"].tolist()):
            raise ValueError("Declare symmetry alternatives for every object, including empty lists")
        ambiguous = {item["name"] for item in opening_summary["objects"] if item.get("rotation_ambiguous")}
        for name, transforms in alternatives.items():
            if transforms:
                require_rigid(np.asarray(transforms, dtype=float), np)
                ambiguous.add(name)
        # PTS units must be explicit; frame indices never stand in for elapsed time.
        numerator, denominator = receipt["time_base"]
        if type(numerator) is not int or type(denominator) is not int or min(numerator, denominator) <= 0:
            raise ValueError("An exact positive source PTS time base is required")
        seconds = result["pts"].astype(float) * numerator / denominator
        if (np.diff(seconds) <= 0).any():
            raise ValueError("Refinement requires increasing source PTS")
        source_clock = {row["frame_index"]: row for row in depth[1]["frames"]}
        for i, frame_index in enumerate(result["frame_index"]):
            clock = source_clock.get(int(frame_index))
            if (clock is None or clock["video_pts"] != int(result["pts"][i])
                    or not math.isclose(seconds[i], clock["video_pts_seconds"], abs_tol=1e-8, rel_tol=0)):
                raise ValueError("Receipt time base differs from the pinned source clock")
        for key in ("world_T_object", "camA_T_object", "valid", "world_valid"):
            result["unrefined_" + key] = result[key].copy()
        corrections = np.zeros((count, bodies, 6), dtype=float)
        supported = np.zeros((count, bodies), dtype=bool)
        windows = []
        for slot in range(bodies):
            if result["object_names"][slot] in ambiguous:
                windows.append({"object_name": str(result["object_names"][slot]), "accepted": False,
                                "reason": "unresolved_symmetry_alternatives_retained"})
                continue
            for indices in _segments(observations, result, seconds, slot, request, np, Rotation):
                report, updates = _refine_window(indices, slot, observations, result, motion, seconds,
                                                 request, np, least_squares, Rotation)
                windows.append(report)
                for index, (world, camera_t_world, correction) in zip(indices, updates):
                    result["world_T_object"][index, slot] = world
                    result["camA_T_object"][index, slot] = camera_t_world @ result["world_T_object"][index, slot]
                    corrections[index, slot] = correction
                    supported[index, slot] = True
        result.update(correction_object_se3=corrections, refinement_supported=supported)
        record.update(grid=poses[1]["grid"], object_names=result["object_names"].tolist(),
                      mesh_body_frames=poses[1]["mesh_body_frames"], method="bounded_correspondence_refinement",
                      interpolation="none", promotion="unassessed_research_candidate")
        save_npz(directory / "poses.npz", np, **result)
        # Preserve opening values; the opening summary is not silently recomputed from corrected poses.
        write_json(directory / "opening.json", opening_summary)
        write_json(directory / "refinement.json", {"schema_version": 1, "windows": windows,
                   "symmetry_alternatives": receipt.get("symmetry_alternatives", []),
                   "symmetry_policy": "retained_not_resolved_by_smoothing", "opening_summary": "unrefined",
                   "unsupported_frames": int((~supported).sum()), "production_promoted": False})
        record["artifacts"] = {name: sha256_file(directory / name)
                               for name in ("poses.npz", "opening.json", "refinement.json")}
        record["frames"] = [{"frame_index": int(result["frame_index"][i]), "video_pts": int(result["pts"][i]),
                             "status": "completed", "artifact": "poses.npz", "sha256": record["artifacts"]["poses.npz"]}
                            for i in range(count)]
    return directory / "run.json"
