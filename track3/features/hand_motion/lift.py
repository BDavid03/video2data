"""Fit camera-space hand surfaces to metric depth, then lift measured joints."""

import math

from track3.provenance import sha256_file
from ._common import (arrays, artifact, authorize, frame_arrays, operation_run, producer,
                      read_json, require_input, require_rotations, save_npz)


def _clock(hands, motion, np):
    count = len(motion["frame_index"])
    if count == 0:
        raise ValueError("Empty hand/camera clock")
    for source in (hands, motion):
        for key in ("frame_index", "pts"):
            value = source[key]
            if value.shape != (count,) or value.dtype != np.dtype("int64"):
                raise ValueError("Hand/camera clocks must be one-dimensional int64 arrays")
            if not np.array_equal(value, motion[key]):
                raise ValueError("Hand and camera frame-index/PTS grids differ")
        if not np.array_equal(source["frame_index"], np.arange(count)) or (np.diff(source["pts"]) <= 0).any():
            raise ValueError("Hand lifting requires the full ordered RGB clip clock")
    return count


def _kinematics(hands, source, count, np):
    if (source.get("joint_order") != "MANO21" or source.get("units") != "metres" or
            source.get("joint_frame_convention") != "MANO" or
            source.get("output_parameter_form") != "anatomical_unmirrored" or
            source.get("orientation_source") != "mano_fk" or not source.get("hand_model")):
        raise ValueError("Hand lifting requires anatomical, metric MANO forward kinematics")
    form = source.get("input_parameter_form")
    if (form not in ("anatomical_unmirrored", "mirrored_right_hand") or
            type(source.get("unmirroring_applied")) is not bool or
            source["unmirroring_applied"] != (form == "mirrored_right_hand")):
        raise ValueError("Kinematics must declare one consistent wrist/finger un-mirroring")
    if hands["sides"].tolist() != ["left", "right"]:
        raise ValueError("Hand side order must be left, right")
    shapes = {"valid": (count, 2), "joints_camA": (count, 2, 21, 3),
              "joint_rotations_camA": (count, 2, 21, 3, 3),
              "wrist_orient_camA": (count, 2, 3), "finger_pose": (count, 2, 45),
              "betas": (2, 10), "score": (count, 2), "mask_index": (count, 2)}
    if any(hands[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("Invalid kinematics array dimensions")
    surface = hands["surface_points_camA"]
    if surface.ndim != 4 or surface.shape[:2] != (count, 2) or surface.shape[2] == 0 or surface.shape[3] != 3:
        raise ValueError("Kinematics requires camera-space metric surface samples T x 2 x V x 3")
    if hands["valid"].dtype != np.dtype("bool") or hands["mask_index"].dtype.kind not in "iu":
        raise ValueError("Kinematics validity must be boolean and mask indices integral")
    valid = hands["valid"]
    for name in ("joints_camA", "surface_points_camA", "wrist_orient_camA", "finger_pose", "score"):
        if not np.isfinite(hands[name][valid]).all():
            raise ValueError(f"Non-finite observed hand field: {name}")
    if ((hands["score"][valid] < 0) | (hands["score"][valid] > 1)).any():
        raise ValueError("Observed detector scores must be in [0, 1]")
    for side in range(2):
        if valid[:, side].any() and not np.isfinite(hands["betas"][side]).all():
            raise ValueError("An observed hand requires a finite episode shape vector")
    if (hands["joints_camA"][valid, 0, 2] <= 0).any():
        raise ValueError("An observed wrist must be in front of the camera")
    require_rotations(hands["joint_rotations_camA"][valid], np)


def _pixels(points, k, shape, np):
    height, width = shape
    front = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[front] = (points[front] @ k.T)[:, :2] / points[front, 2, None]
    inside = front & (projected[:, 0] >= 0) & (projected[:, 0] < width)
    inside &= (projected[:, 1] >= 0) & (projected[:, 1] < height)
    ids = np.flatnonzero(inside)
    pixels = np.floor(projected[ids] + 0.5).astype(np.int64)
    keep = (pixels[:, 0] < width) & (pixels[:, 1] < height)
    return ids[keep], pixels[keep]


def _residuals(surface, mask, depth, k, np):
    ids, pixels = _pixels(surface, k, depth.shape, np)
    # A pixel contributes once; retain only its closest projected surface sample.
    order = np.argsort(surface[ids, 2], kind="stable")
    ids, pixels = ids[order], pixels[order]
    _, first = np.unique(pixels[:, 1] * depth.shape[1] + pixels[:, 0], return_index=True)
    ids, pixels = ids[first], pixels[first]
    measured = depth[pixels[:, 1], pixels[:, 0]]
    keep = mask[pixels[:, 1], pixels[:, 0]] & np.isfinite(measured) & (measured > 0)
    return measured[keep] - surface[ids[keep], 2]


def _inliers(residual, request, np):
    center = np.median(residual)
    deviation = np.median(abs(residual - center))
    return residual[abs(residual - center) <= max(request.outlier_floor_m, 3 * 1.4826 * deviation)]


def _supported(surface, mask, depth, k, request, np):
    """Depth residuals after outlier rejection; short arrays mean insufficient support."""
    residual = _residuals(surface, mask, depth, k, np)
    return _inliers(residual, request, np) if len(residual) >= request.minimum_depth_support else residual


def _fit(joints, surface, mask, depth, k, request, np):
    ray = joints[0] / joints[0, 2]
    shift = 0.0
    for _ in range(request.fit_iterations):
        residual = _supported(surface + ray * shift, mask, depth, k, request, np)
        if len(residual) < request.minimum_depth_support:
            return None, "insufficient_depth", len(residual), None, shift
        delta = float(np.median(residual))
        shift += delta
        if abs(shift) > request.maximum_depth_shift_m:
            return None, "depth_shift_limit", len(residual), None, shift
        if abs(delta) < 1e-5:
            break
    fitted = joints + ray * shift
    if (fitted[:, 2] <= 1e-6).any():
        return None, "behind_camera", 0, None, shift
    # Reproject after the final update so support and residual describe the output.
    residual = _supported(surface + ray * shift, mask, depth, k, request, np)
    if len(residual) < request.minimum_depth_support:
        return None, "insufficient_depth", len(residual), None, shift
    error = float(np.median(abs(residual)))
    if error > request.maximum_depth_residual_m:
        return None, "depth_residual_limit", len(residual), error, shift
    return fitted, "observed", len(residual), error, shift


def _metric_depth(mono, stereo, scale, k, grid_valid, np):
    shape = grid_valid.shape
    for source, depth_key, valid_key, intrinsic_key in (
            (mono, "depth_m", "valid", "input_intrinsics_pixels"),
            (stereo, "rgb_depth_m", "rgb_valid", "rgb_intrinsics_pixels")):
        if (source[depth_key].shape != shape or source[valid_key].shape != shape or
                source[valid_key].dtype != np.dtype("bool") or
                source[intrinsic_key].shape != (3, 3) or
                not np.allclose(source[intrinsic_key], k, atol=1e-5, rtol=0)):
            raise ValueError("Hand depth inputs disagree with the calibrated pixel grid")
    depth = np.full(shape, np.nan, dtype=np.float64)
    if scale is not None:
        valid = grid_valid & mono["valid"] & np.isfinite(mono["depth_m"]) & (mono["depth_m"] > 0)
        depth[valid] = mono["depth_m"][valid] * scale
    valid = grid_valid & stereo["rgb_valid"] & np.isfinite(stereo["rgb_depth_m"]) & (stereo["rgb_depth_m"] > 0)
    depth[valid] = stereo["rgb_depth_m"][valid]
    return depth


def _parents(request):
    hand = producer(request.kinematics_run, "hand_motion.kinematics", request)
    masks = producer(request.masks_run, "object_masks.generate", request)
    depth = producer(request.depth_run, "depth.generate", request)
    stereo = producer(request.stereo_run, "stereo_depth.generate", request)
    anchor = producer(request.anchor_run, "stereo_depth.anchor", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(parent[1].get("grid") != grid for parent in (hand, masks, stereo, anchor, camera)):
        raise ValueError("Hand inputs use different colour pixel grids")
    for source, inputs in ((hand, (masks, depth)), (masks, (depth,)), (stereo, (depth,)),
                           (anchor, (depth, stereo)), (camera, (depth, stereo, anchor))):
        for parent in inputs:
            require_input(source, parent)
    instances = masks[1]["instances"]
    if (not isinstance(instances, list) or any(not isinstance(name, str) for name in instances) or
            len(set(instances)) != len(instances) or not {"hand_0", "hand_1"}.issubset(instances)):
        raise ValueError("Mask producer must identify hand_0 and hand_1 layers")
    return (hand, masks, depth, stereo, anchor, camera), grid


def _check_trajectory(motion, count, np):
    if (motion["valid"].shape != (count,) or motion["valid"].dtype != np.dtype("bool") or
            motion["world_T_camA"].shape != (count, 4, 4)):
        raise ValueError("Invalid camera trajectory dimensions or validity type")
    transforms = motion["world_T_camA"][motion["valid"]]
    if not np.isfinite(transforms).all() or not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Camera trajectory contains invalid observed rigid transforms")
    require_rotations(transforms[:, :3, :3], np)


def _anchor_scales(document, hands, count, grid):
    """Per-frame observed metric scale (None where unobserved) from the anchor document."""
    if document.get("grid") != grid:
        raise ValueError("Anchor document uses a different pixel grid")
    scales = {}
    for row in document["frames"]:
        index = row["frame_index"]
        if (type(index) is not int or index < 0 or index >= count or index in scales or
                row["video_pts"] != int(hands["pts"][index]) or row["status"] != "completed"):
            raise ValueError("Invalid anchor frame identity or status")
        scale = row["scale"] if row["scale_status"] == "observed" else None
        if scale is not None and (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
            raise ValueError("Observed depth scale must be finite and positive")
        if row["scale_status"] == "observed" and scale is None:
            raise ValueError("Observed depth scale is missing")
        scales[index] = scale
    return scales


def _empty_result(hands, count, np):
    result = {"sides": np.asarray(["left", "right"]),
              "valid": np.zeros((count, 2), dtype=bool), "world_valid": np.zeros((count, 2), dtype=bool),
              "joints_world": np.full((count, 2, 21, 3), np.nan, dtype=np.float64),
              "joint_rotations_world": np.full((count, 2, 21, 3, 3), np.nan, dtype=np.float64),
              "joints_camA": np.full((count, 2, 21, 3), np.nan, dtype=np.float64),
              "wrist_orient_camA": np.full((count, 2, 3), np.nan, dtype=np.float32),
              "finger_pose": np.full((count, 2, 45), np.nan, dtype=np.float32),
              "betas": hands["betas"].astype(np.float32),
              "score": np.where(hands["valid"], hands["score"], np.nan).astype(np.float32),
              "in_frame_fraction": np.full((count, 2), np.nan, dtype=np.float32),
              "depth_support": np.zeros((count, 2), dtype=np.int32),
              "depth_residual_m": np.full((count, 2), np.nan, dtype=np.float32),
              "depth_shift_m": np.full((count, 2), np.nan, dtype=np.float32),
              "frame_index": hands["frame_index"].copy(), "pts": hands["pts"].copy()}
    result["betas"][~hands["valid"].any(axis=0)] = np.nan
    return result


def _lift_side(index, side, detail, hands, motion, layers, hand_layers, metric, k, shape, result, request, np):
    """Fit one observed hand to metric depth and lift it to world; outcome goes in ``detail``."""
    layer = int(hands["mask_index"][index, side])
    if not layers["present"][layer]:
        detail["status"] = "mask_absent"
        return
    other_layer = next(value for value in hand_layers if value != layer)
    fit_mask = layers["masks"][layer].copy()
    if layers["present"][other_layer]:
        fit_mask &= ~layers["masks"][other_layer]
    fitted, status, support, error, shift = _fit(
        hands["joints_camA"][index, side], hands["surface_points_camA"][index, side],
        fit_mask, metric, k, request, np)
    result["depth_support"][index, side] = support
    result["depth_shift_m"][index, side] = shift
    if error is not None:
        result["depth_residual_m"][index, side] = error
    detail.update(status=status, depth_support=support, depth_residual_m=error, depth_shift_m=shift)
    if fitted is None:
        return
    visible, _ = _pixels(fitted, k, shape, np)
    fraction = len(visible) / 21
    result["in_frame_fraction"][index, side] = fraction
    detail["in_frame_fraction"] = fraction
    if fraction < request.minimum_in_frame_fraction:
        detail["status"] = "out_of_frame"
        return
    result["valid"][index, side] = True
    result["joints_camA"][index, side] = fitted
    for key in ("wrist_orient_camA", "finger_pose"):
        result[key][index, side] = hands[key][index, side]
    if not motion["valid"][index]:
        detail["status"] = "camera_invalid"
        return
    pose = motion["world_T_camA"][index]
    result["world_valid"][index, side] = True
    result["joints_world"][index, side] = fitted @ pose[:3, :3].T + pose[:3, 3]
    result["joint_rotations_world"][index, side] = pose[:3, :3] @ hands["joint_rotations_camA"][index, side]
    detail["world_valid"] = True


def lift(request):
    authorize(request)
    parents, grid = _parents(request)
    hand, masks, depth, stereo, anchor, camera = parents
    instances = masks[1]["instances"]
    hand_layers = {instances.index("hand_0"), instances.index("hand_1")}
    with operation_run(request, parents) as (directory, record):
        import numpy as np

        hands = arrays(hand, "kinematics.npz", record, np)
        motion = arrays(camera, "trajectory.npz", record, np)
        count = _clock(hands, motion, np)
        _kinematics(hands, hand[1], count, np)
        assigned = hands["mask_index"]
        both_observed = hands["valid"].all(axis=1)
        if (not np.isin(assigned[hands["valid"]], list(hand_layers)).all() or
                (assigned[both_observed, 0] == assigned[both_observed, 1]).any()):
            raise ValueError("Observed sides must map to distinct named hand-mask layers")
        _check_trajectory(motion, count, np)
        preprocessing = arrays(depth, "preprocessing.npz", record, np, grid)
        k, grid_valid = preprocessing["input_intrinsics_pixels"], preprocessing["input_valid"]
        if (grid_valid.ndim != 2 or grid_valid.dtype != np.dtype("bool") or k.shape != (3, 3) or
                not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or
                not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)):
            raise ValueError("Invalid hand pinhole grid or intrinsics")
        world = read_json(artifact(camera, "world.json", record))
        if world.get("up_axis") != "+z" or world.get("grid") != grid:
            raise ValueError("Hand lifting requires the selected camera's z-up world")
        scales = _anchor_scales(read_json(artifact(anchor, "anchor.json", record)), hands, count, grid)
        result = _empty_result(hands, count, np)
        record.update(grid=grid, orientation_source="mano_fk", joint_order="MANO21", units="metres",
                      joint_frame_convention="MANO", output_parameter_form="anatomical_unmirrored",
                      input_parameter_form=hand[1]["input_parameter_form"],
                      unmirroring_applied=hand[1]["unmirroring_applied"], hand_model=hand[1]["hand_model"],
                      depth_method="wrist_ray_surface_median", interpolation="none")
        for index in range(count):
            pts = int(hands["pts"][index])
            row = {"frame_index": index, "video_pts": pts, "status": "running", "hands": []}
            record["frames"].append(row)
            candidates = hands["valid"][index] & (hands["score"][index] >= request.minimum_score)
            metric = layers = None
            if candidates.any():
                mono = frame_arrays(depth, index, pts, record, np)
                stereo_frame = frame_arrays(stereo, index, pts, record, np)
                metric = _metric_depth(mono, stereo_frame, scales.get(index), k, grid_valid, np)
                layers = frame_arrays(masks, index, pts, record, np)
                if (layers["masks"].shape != (len(instances), *grid_valid.shape) or
                        layers["present"].shape != (len(instances),) or
                        layers["masks"].dtype != np.dtype("bool") or layers["present"].dtype != np.dtype("bool")):
                    raise ValueError("Invalid hand-mask arrays")
            for side, name in enumerate(("left", "right")):
                detail = {"side": name, "status": "not_detected", "world_valid": False}
                row["hands"].append(detail)
                if not hands["valid"][index, side]:
                    continue
                if not candidates[side]:
                    detail["status"] = "low_score"
                    continue
                _lift_side(index, side, detail, hands, motion, layers, hand_layers, metric, k,
                           grid_valid.shape, result, request, np)
            row["status"] = "completed"
        save_npz(directory / "hands.npz", np, **result)
        record["artifacts"]["hands.npz"] = sha256_file(directory / "hands.npz")
        record["valid_hand_frames"] = result["valid"].sum(axis=0).tolist()
        record["world_valid_hand_frames"] = result["world_valid"].sum(axis=0).tolist()
        if not hands["valid"].any():
            raise ValueError("no_hands: no hand was detected in the clip")
        if not result["valid"].any():
            raise ValueError("no_valid_hands: no hand passed depth and visibility requirements")
    return directory / "run.json"
