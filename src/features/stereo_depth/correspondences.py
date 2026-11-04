"""Sparse reciprocal image correspondences independent of candidate depth values."""

import time

from src.provenance import sha256_file, write_json
from ._common import (artifact, authorize, operation_run, producer, save_npz, select_episode, selected_indices,
                      stereo_captures)
from .support import PARTITION


def _reciprocal(images, detector, matcher, rect, np):
    """ORB matches passing the 0.75 ratio test in both directions; returns pairs, endpoints, eligible count."""
    keypoints, descriptors = zip(*(detector.detectAndCompute(image, rect[f"{side}_valid"].astype(np.uint8) * 255)
                                   for image, side in zip(images, ("left", "right"))))
    pairs = []
    if all(descriptor is not None and len(descriptor) >= 2 for descriptor in descriptors):
        forward = matcher.knnMatch(descriptors[0], descriptors[1], k=2)
        backward = matcher.knnMatch(descriptors[1], descriptors[0], k=2)
        reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
                   if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance}
        pairs = [pair[0] for pair in forward if len(pair) == 2
                 and pair[0].distance < .75 * pair[1].distance
                 and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    left = np.array([keypoints[0][pair.queryIdx].pt for pair in pairs], dtype=np.float64).reshape(-1, 2)
    right = np.array([keypoints[1][pair.trainIdx].pt for pair in pairs], dtype=np.float64).reshape(-1, 2)
    return pairs, left, right, len(keypoints[0])


def _admit(left, right, rect, k, grid_valid, projections, request, cv2, np):
    """Triangulate into camera A and apply the ordered admission filters, counting each rejection."""
    homogeneous = cv2.triangulatePoints(rect["P1"], rect["P2"], left.T, right.T).T
    points = homogeneous[:, :3] / np.where(abs(homogeneous[:, 3:4]) > 1e-12, homogeneous[:, 3:4], np.nan)
    points_a = points @ rect["R1"] @ rect["A_T_C"][:3, :3].T + rect["A_T_C"][:3, 3]
    rgb = points_a @ k.T
    projected = rgb[:, :2] / np.where(rgb[:, 2:3] > 0, rgb[:, 2:3], np.nan)
    finite = np.isfinite(projected).all(axis=1) & np.isfinite(points_a).all(axis=1)
    uv = np.rint(np.where(np.isfinite(projected), projected, -1)).astype(np.int64)
    height, width = grid_valid.shape
    inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    vertical_error = abs(left[:, 1] - right[:, 1])
    vertical_summary = {"denominator": len(left), "median": float(np.median(vertical_error)),
                        "p90": float(np.quantile(vertical_error, .9)),
                        "maximum": float(np.max(vertical_error)), "units": "px",
                        "support": "all_reciprocal_matches_before_geometric_admission"}
    keep = inside.copy()
    counts = {"nonfinite_or_outside_A_grid": int((~inside).sum())}
    depth_ok = ((points[:, 2] > 0) & (points_a[:, 2] >= request.minimum_depth_m)
                & (points_a[:, 2] <= request.maximum_depth_m))
    for name, condition in (("depth_range_or_behind", depth_ok),
                            ("vertical_mismatch", vertical_error <= request.vertical_tolerance_px)):
        before = int(keep.sum())
        keep &= condition
        counts[name] = before - int(keep.sum())
    before = int(keep.sum())
    keep[inside] &= grid_valid[uv[inside, 1], uv[inside, 0]]
    counts["invalid_A_grid"] = before - int(keep.sum())
    before = int(keep.sum())
    for side, projection in zip((left, right), projections):
        predicted = np.column_stack((points_a, np.ones(len(points_a)))) @ projection.T
        predicted = predicted[:, :2] / np.where(predicted[:, 2:3] > 0, predicted[:, 2:3], np.nan)
        keep &= np.linalg.norm(predicted - side, axis=1) <= request.reprojection_tolerance_px
    counts["triangulation_reprojection"] = before - int(keep.sum())
    return points_a, uv, keep, counts, vertical_summary


def _unique_pixels(keep, pairs, uv, np):
    """One query per A-grid pixel (best descriptor distance first) prevents fit/evidence collisions."""
    seen, chosen = set(), []
    for j in sorted(np.flatnonzero(keep), key=lambda j: (pairs[j].distance, pairs[j].queryIdx)):
        identity = f"a:{uv[j, 1]}:{uv[j, 0]}"
        if identity not in seen:
            seen.add(identity)
            chosen.append(j)
    return np.asarray(chosen, dtype=np.int64)


def correspondences(request):
    authorize("correspondences", request)
    selection = select_episode(request)
    rect_path, rect_record = producer(request.rectification_run, "stereo_depth.rectify", request)
    depth_path, depth_record = producer(request.depth_run, "depth.generate", request)
    with operation_run("correspondences", request, selection, [rect_path, depth_path]) as (directory, record):
        import cv2
        import numpy as np

        if any(source["dataset_revision"] != selection["revision"] for source in (rect_record, depth_record)):
            raise ValueError("Correspondence producer revisions differ")
        rect_file = artifact(rect_path, "rectification.npz", record, rect_record["artifacts"]["rectification.npz"])
        grid = depth_record["preprocessing_artifact_sha256"]
        grid_file = artifact(depth_path, "preprocessing.npz", record, grid)
        with np.load(rect_file, allow_pickle=False) as arrays:
            rect = {name: arrays[name].copy() for name in arrays.files}
        with np.load(grid_file, allow_pickle=False) as arrays:
            k, grid_valid = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].astype(bool)
        if max(grid_valid.shape) > request.max_image_side or max(rect["map_left_x"].shape) > request.max_image_side:
            raise ValueError("Inputs exceed the declared research resolution")
        # X_rect_left = R1 C_T_A X_A. P2 already contains the pair translation.
        rect_left_from_a = np.eye(4)
        rect_left_from_a[:3] = rect["R1"] @ np.linalg.inv(rect["A_T_C"])[:3]
        projections = [rect[name] @ rect_left_from_a for name in ("P1", "P2")]
        record.update(grid=grid, evidence_contract="independent-image-correspondences-v1",
                      holdback_partition=PARTITION, query_identity="a:{row}:{column}",
                      depth_values_used=False, visibility="reciprocal-descriptor-and-positive-triangulation-proxy",
                      limitations=["Reciprocal matching is not proof of unoccluded surface visibility.",
                                   "Sparse textured support cannot establish full-surface accuracy."])
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        started = time.monotonic()
        selected = set(selected_indices(request, selection["stop"]))
        (directory / "frames").mkdir()
        with stereo_captures(selection, record, cv2) as captures:
            for index in range(selection["stop"]):
                if time.monotonic() - started > 1200:
                    raise TimeoutError("Sparse correspondence CPU wall ceiling reached")
                images = []
                for capture, side in zip(captures, ("left", "right")):
                    ok, image = capture.read()
                    if not ok or tuple(image.shape[1::-1]) != tuple(rect["source_size"]):
                        raise ValueError("Stereo source decode/shape failure")
                    if index in selected:
                        images.append(cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                      rect[f"map_{side}_x"], rect[f"map_{side}_y"], cv2.INTER_LINEAR))
                if not images:
                    continue
                pts = selection["clocks"]["ego_cam_a"]["pts_integer"][index]
                row = {"frame_index": index, "video_pts": pts, "status": "running"}
                record["frames"].append(row)
                pairs, left, right, eligible = _reciprocal(images, detector, matcher, rect, np)
                points_a, uv = np.empty((0, 3)), np.empty((0, 2), dtype=np.int64)
                keep, counts, vertical_summary = np.zeros(len(pairs), dtype=bool), {}, None
                if pairs:
                    points_a, uv, keep, counts, vertical_summary = _admit(
                        left, right, rect, k, grid_valid, projections, request, cv2, np)
                rejection_counts = {"no_reciprocal_ratio_match": eligible - len(pairs), **counts}
                chosen = _unique_pixels(keep, pairs, uv, np)
                rejection_counts["duplicate_A_grid_identity"] = int(keep.sum()) - len(chosen)
                destination = directory / "frames" / f"frame_{index:06d}.npz"
                save_npz(destination, np, query_ids=np.asarray([f"a:{uv[j, 1]}:{uv[j, 0]}" for j in chosen], dtype="U64"),
                         rgb_uv=uv[chosen], stereo_depth_m=points_a[chosen, 2],
                         left_uv=left[chosen], right_uv=right[chosen],
                         left_projection=projections[0], right_projection=projections[1],
                         input_intrinsics_pixels=k, frame_index=np.int64(index), pts=np.int64(pts))
                row.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                           sha256=sha256_file(destination), eligible_queries=eligible,
                           reciprocal_queries=len(pairs), admitted_queries=len(chosen),
                           rejected_queries=eligible - len(chosen), rejection_counts=rejection_counts,
                           reciprocal_vertical_residual_px=vertical_summary)
                write_json(directory / "run.json", record)
    return directory / "run.json"
