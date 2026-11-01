"""CPU SGBM with reverse consistency and nearest-surface colour transfer."""

from track3.provenance import sha256_file, write_json
from ._common import (ROOT, artifact, authorize, operation_run, producer, save_npz, select_episode,
                      stereo_captures)


def _transfer(depth, valid, rect, rgb_k, shape, np, quality):
    """Z-buffer rectified-left depth into the colour grid; quality follows the winning source point."""
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    p = rect["P1"]
    points = np.column_stack(((u - p[0, 2]) * z / p[0, 0], (v - p[1, 2]) * z / p[1, 1], z))
    # Row vectors: undo rectification with R1, then map camera C into A.
    points = points @ rect["R1"] @ rect["A_T_C"][:3, :3].T + rect["A_T_C"][:3, 3]
    positive = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    samples = quality[v, u][positive]
    points = points[positive]
    projected = points @ rgb_k.T
    uv = projected[:, :2] / projected[:, 2:3]
    height, width = shape
    inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    pixels = np.rint(uv[inside]).astype(np.int64)
    bins = pixels[:, 1] * width + pixels[:, 0]
    result = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(result, bins, points[inside, 2])
    result = result.reshape(shape)
    mask = np.isfinite(result)
    transferred = np.where(mask, result, np.nan).astype(np.float32)
    # Quality comes from the exact nearest source point (first row-major on ties),
    # never a max/mean of other surfaces projecting into the same pixel.
    depths = points[inside, 2]
    nearest = np.full(height * width, np.inf)
    np.minimum.at(nearest, bins, depths)
    front = depths == nearest[bins]
    indices = np.arange(len(depths), dtype=np.int64)
    winners = np.full(height * width, len(depths), dtype=np.int64)
    np.minimum.at(winners, bins[front], indices[front])
    transferred_quality = np.full(height * width, np.nan, dtype=np.float32)
    admitted = winners < len(depths)
    transferred_quality[admitted] = samples[inside][winners[admitted]]
    return transferred, mask, transferred_quality.reshape(shape)


def _match(images, matcher, reverse, right_min, focal, baseline, rect, request, np):
    """Left disparity, metric depth, validity and left-right residual/quality for one rectified pair."""
    disparity = matcher.compute(*images).astype(np.float32) / 16
    right_disparity = reverse.compute(images[1], images[0]).astype(np.float32) / 16
    vv, uu = np.indices(disparity.shape)
    corresponding = np.rint(uu - disparity).astype(np.int64)
    inside = (corresponding >= 0) & (corresponding < disparity.shape[1])
    safe_u = np.clip(corresponding, 0, disparity.shape[1] - 1)
    reverse_sample = right_disparity[vv, safe_u]
    lr_residual = abs(disparity + reverse_sample)
    valid = (rect["left_valid"] & inside & rect["right_valid"][vv, safe_u]
             & (disparity > 0) & (disparity >= request.minimum_disparity)
             & (reverse_sample >= right_min)
             & (lr_residual <= request.left_right_tolerance_px))
    depth = focal * baseline / np.where(valid, disparity, 1)
    valid &= np.isfinite(depth) & (depth >= request.minimum_depth_m) & (depth <= request.maximum_depth_m)
    quality = np.clip(1 - lr_residual / request.left_right_tolerance_px, 0, 1).astype(np.float32)
    return disparity, depth, valid, lr_residual, quality


def generate(request):
    authorize("generate", request)
    selection = select_episode(request)
    rect_path, rect_record = producer(request.rectification_run, "stereo_depth.rectify", request)
    depth_path, depth_record = producer(request.depth_run, "depth.generate", request)
    with operation_run("generate", request, selection, [rect_path, depth_path]) as (directory, record):
        import cv2
        import numpy as np

        if any(source["dataset_revision"] != selection["revision"] for source in (rect_record, depth_record)):
            raise ValueError("Producer dataset revisions differ")
        rect_file = artifact(rect_path, "rectification.npz", record, rect_record["artifacts"]["rectification.npz"])
        grid = depth_record["preprocessing_artifact_sha256"]
        grid_file = artifact(depth_path, "preprocessing.npz", record, grid)
        with np.load(rect_file, allow_pickle=False) as arrays:
            rect = {name: arrays[name].copy() for name in arrays.files}
        with np.load(grid_file, allow_pickle=False) as arrays:
            rgb_k, rgb_input_valid = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].astype(bool)
        if rgb_k.shape != (3, 3) or not np.isfinite(rgb_k).all() or rgb_input_valid.ndim != 2:
            raise ValueError("Invalid colour pinhole grid")
        record.update(grid=grid, grid_path=grid_file.relative_to(ROOT).as_posix(),
                      confidence_field="rgb_match_confidence",
                      confidence_semantics="uncalibrated_matching_quality_not_probability",
                      confidence_origin="linear_left_right_disparity_residual",
                      confidence_transfer="same_nearest_visible_point_as_rgb_depth; first_row_major_source_on_exact_ties",
                      confidence_note="1-abs(left_disparity+sampled_right_disparity)/LR_tolerance, clipped [0,1]; not calibrated accuracy or visibility probability")
        focal, baseline = float(rect["P1"][0, 0]), -float(rect["P2"][0, 3] / rect["P2"][0, 0])
        if request.minimum_disparity + request.num_disparities >= rect["map_left_x"].shape[1]:
            raise ValueError("Disparity search is wider than the rectified image")
        parameters = dict(numDisparities=request.num_disparities, blockSize=request.block_size,
                          P1=8 * request.block_size**2, P2=32 * request.block_size**2,
                          uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                          disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        matcher = cv2.StereoSGBM.create(minDisparity=request.minimum_disparity, **parameters)
        right_min = -request.minimum_disparity - request.num_disparities + 1
        reverse = cv2.StereoSGBM.create(minDisparity=right_min, **parameters)
        (directory / "frames").mkdir()
        clock = selection["clocks"]["ego_cam_a"]
        with stereo_captures(selection, record, cv2) as captures:
            for index in range(selection["stop"]):
                row = None
                if index >= request.start_frame:
                    row = {"frame_index": index, "video_pts": clock["pts_integer"][index],
                           "video_pts_seconds": clock["pts_seconds"][index], "status": "running"}
                    record["frames"].append(row)
                images = []
                for capture, side in zip(captures, ("left", "right")):
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(f"Stereo decode stopped at frame {index}")
                    if tuple(image.shape[1::-1]) != tuple(rect["source_size"]):
                        raise ValueError("Stereo source dimensions differ from calibration")
                    if row is not None:
                        images.append(cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                      rect[f"map_{side}_x"], rect[f"map_{side}_y"], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT))
                if row is None:
                    continue
                disparity, depth, valid, lr_residual, quality = _match(
                    images, matcher, reverse, right_min, focal, baseline, rect, request, np)
                rgb_depth, rgb_valid, rgb_quality = _transfer(depth, valid, rect, rgb_k, rgb_input_valid.shape, np,
                                                             quality)
                rgb_valid &= rgb_input_valid & (rgb_depth >= request.minimum_depth_m) & (rgb_depth <= request.maximum_depth_m)
                destination = directory / "frames" / f"frame_{index:06d}.npz"
                save_npz(destination, np, disparity_px=np.where(valid, disparity, np.nan).astype(np.float32),
                         depth_m=np.where(valid, depth, np.nan).astype(np.float32), valid=valid,
                         rgb_depth_m=np.where(rgb_valid, rgb_depth, np.nan).astype(np.float32), rgb_valid=rgb_valid,
                         left_right_residual_px=np.where(valid, lr_residual, np.nan).astype(np.float32),
                         rgb_match_confidence=np.where(rgb_valid, rgb_quality, np.nan).astype(np.float32),
                         rgb_intrinsics_pixels=rgb_k.astype(np.float64), frame_index=np.int64(index),
                         pts=np.int64(row["video_pts"]))
                row.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                           sha256=sha256_file(destination), valid_pixels=int(valid.sum()), rgb_valid_pixels=int(rgb_valid.sum()))
                record["artifacts"][row["artifact"]] = row["sha256"]
                write_json(directory / "run.json", record)
    return directory / "run.json"
