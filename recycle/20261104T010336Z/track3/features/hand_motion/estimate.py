"""Calibrated per-frame hand regression with explicit anatomical assignment."""

from itertools import product

from track3.gpu import GpuLease
from track3.provenance import sha256_file
from ._common import (ROOT, arrays, authorize, frame_arrays, operation_run, producer,
                      require_input, rgb_clock, rgb_video, save_npz)
from ._wilor import detections, pipeline, provisioned, regress


def _virtual_grid(preprocessing, cv2, np):
    k = preprocessing["input_intrinsics_pixels"]
    valid = preprocessing["input_valid"]
    if (valid.ndim != 2 or valid.dtype != np.dtype("bool") or k.shape != (3, 3) or
            not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("Invalid hand-estimation pinhole calibration")
    height, width = valid.shape
    focal = float(min(k[0, 0], k[1, 1]))
    virtual = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    v, u = np.indices(valid.shape)
    pixels = np.stack((u, v, np.ones(valid.shape)), axis=-1) @ (k @ np.linalg.inv(virtual)).T
    mx = (pixels[..., 0] / pixels[..., 2]).astype(np.float32)
    my = (pixels[..., 1] / pixels[..., 2]).astype(np.float32)
    support = cv2.remap(valid.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) != 0
    return virtual, mx, my, support


def _match_masks(boxes, masks, request, np):
    height, width = masks.shape[1:]
    iou = np.zeros((len(boxes), 2), dtype=np.float64)
    for index, box in enumerate(boxes):
        x0, y0 = np.floor(box[:2]).astype(int)
        x1, y1 = np.ceil(box[2:4]).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        area = (x1 - x0) * (y1 - y0)
        for layer in range(2):
            intersection = int(masks[layer, y0:y1, x0:x1].sum())
            union = area + int(masks[layer].sum()) - intersection
            iou[index, layer] = intersection / union if union else 0
    options = [None, *range(len(boxes))]
    candidates = []
    for pair in product(options, repeat=2):
        chosen = [index for index in pair if index is not None]
        if len(set(chosen)) != len(chosen):
            continue
        if any(index is not None and iou[index, layer] < request.minimum_mask_iou for layer, index in enumerate(pair)):
            continue
        score = sum(iou[index, layer] * float(boxes[index, 4]) for layer, index in enumerate(pair) if index is not None)
        candidates.append((score, pair))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    alternatives = [pair for score, pair in candidates if best_score - score < request.assignment_margin]
    return [(index, layer, float(iou[index, layer])) for layer, index in enumerate(best)
            if index is not None and all(pair[layer] == index for pair in alternatives)]


def _assign(matches, boxes, previous, index, shape, request, np):
    options, candidates = [None, *range(len(matches))], []
    diagonal = float(np.hypot(*shape))
    for pair in product(options, repeat=2):
        selected = [value for value in pair if value is not None]
        if len(set(selected)) != len(selected):
            continue
        score, allowed = 0.0, True
        for side, choice in enumerate(pair):
            if choice is None:
                continue
            detection, layer, overlap = matches[choice]
            box = boxes[detection]
            history = previous[side]
            continuing = history is not None and index - history["frame"] <= request.maximum_track_gap
            center = (box[:2] + box[2:4]) / 2
            if continuing:
                displacement = float(np.linalg.norm(center - history["center"]) / diagonal)
                if displacement > request.maximum_center_step * (index - history["frame"]):
                    allowed = False
                    break
                penalty = displacement + (0.1 if history["layer"] != layer else 0)
                if int(box[5]) != side:
                    penalty += request.handedness_penalty
            elif int(box[5]) != side:
                allowed = False
                break
            else:
                penalty = 0.0
            score += float(box[4]) * overlap - penalty
        if allowed:
            candidates.append((score, pair))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    near = [pair for score, pair in candidates if best_score - score < request.assignment_margin]
    # Ambiguity is per side; no uncertain handedness is sent to the regressor.
    return [choice if all(pair[side] == choice for pair in near) else None for side, choice in enumerate(best)]


def _empty_result(count, pts, np):
    return {"sides": np.asarray(["left", "right"]), "frame_index": np.arange(count, dtype=np.int64), "pts": pts,
            "valid": np.zeros((count, 2), dtype=bool), "score": np.full((count, 2), np.nan, dtype=np.float32),
            "mask_index": np.full((count, 2), -1, dtype=np.int64),
            "wrist_orient_camA": np.full((count, 2, 3), np.nan, dtype=np.float32),
            "finger_pose": np.full((count, 2, 45), np.nan, dtype=np.float32),
            "translation_camA_m": np.full((count, 2, 3), np.nan, dtype=np.float32),
            "betas": np.full((2, 10), np.nan, dtype=np.float32),
            "betas_per_frame": np.full((count, 2, 10), np.nan, dtype=np.float32),
            "boxes_virtual_xyxy": np.full((count, 2, 4), np.nan, dtype=np.float32)}


def _observe(model, lease, bgr, virtual_masks, index, row, previous, result, hand_layers, request, np):
    """Detect, assign and regress the hands of one frame into ``result``."""
    lease.require_headroom()
    boxes = detections(model, bgr, request, np)
    matches = _match_masks(boxes, virtual_masks, request, np)
    assignment = _assign(matches, boxes, previous, index, virtual_masks.shape[1:], request, np)
    for side, choice in enumerate(assignment):
        detail = {"side": ("left", "right")[side], "status": "unassigned"}
        row["hands"].append(detail)
        if choice is None:
            continue
        detection, layer, overlap = matches[choice]
        box = boxes[detection]
        values = regress(model, bgr, box, side, request, np)
        lease.require_headroom()
        for key in ("wrist_orient_camA", "finger_pose", "translation_camA_m"):
            result[key][index, side] = values[key]
        result["betas_per_frame"][index, side] = values["betas"]
        result["valid"][index, side] = True
        result["score"][index, side] = box[4]
        result["mask_index"][index, side] = hand_layers[layer]
        result["boxes_virtual_xyxy"][index, side] = box[:4]
        previous[side] = {"frame": index, "center": (box[:2] + box[2:4]) / 2, "layer": layer}
        detail.update(status="observed", detector_score=float(box[4]), mask_iou=overlap,
                      detector_side=int(box[5]), assigned_side=side, mask_index=hand_layers[layer])
    row.update(status="completed", observed_hands=int(result["valid"][index].sum()))


def estimate(request):
    authorize(request, "estimate")
    masks = producer(request.masks_run, "object_masks.generate", request)
    depth = producer(request.depth_run, "depth.generate", request)
    require_input(masks, depth)
    grid = depth[1]["preprocessing_artifact_sha256"]
    if masks[1].get("grid") != grid:
        raise ValueError("Hand masks and depth use different pixel grids")
    instances = masks[1]["instances"]
    if len(set(instances)) != len(instances) or not {"hand_0", "hand_1"}.issubset(instances):
        raise ValueError("Hand estimation requires uniquely named hand_0 and hand_1 masks")
    hand_layers = [instances.index("hand_0"), instances.index("hand_1")]
    with operation_run(request, [masks, depth], "estimate") as (directory, record):
        prepared = provisioned(record)
        import cv2
        import numpy as np

        clock = rgb_clock(request, record)
        count = clock["decoded_frame_count"]
        pts = np.asarray(clock["pts_integer"], dtype=np.int64)
        if (np.diff(pts) <= 0).any():
            raise ValueError("RGB PTS must be strictly increasing")
        preprocessing = arrays(depth, "preprocessing.npz", record, np, grid)
        k_virtual, vx, vy, virtual_valid = _virtual_grid(preprocessing, cv2, np)
        source_x, source_y = preprocessing["source_x"], preprocessing["source_y"]
        if source_x.shape != vx.shape or source_y.shape != vx.shape:
            raise ValueError("Depth remap dimensions differ from the calibrated grid")
        video = rgb_video(depth, clock, record)
        result = _empty_result(count, pts, np)
        record.update(grid=grid, hand_model=record["backend"], output_parameter_form="anatomical_unmirrored",
                      translation_convention="mano_model_origin_camA_m", pose_representation="axis_angle_full_flat_mean",
                      virtual_intrinsics_pixels=k_virtual.tolist(), input_color="BGR", interpolation="none",
                      score_source="detector_confidence", shape_policy="median_over_accepted_frames")
        save_npz(directory / "inference_grid.npz", np, source_x=vx, source_y=vy, valid=virtual_valid,
                 input_intrinsics_pixels=preprocessing["input_intrinsics_pixels"], virtual_intrinsics_pixels=k_virtual)
        record["artifacts"]["inference_grid.npz"] = sha256_file(directory / "inference_grid.npz")
        previous, capture, lease = [None, None], None, None
        try:
            with GpuLease(ROOT, request.gpu_headroom_mib, request.gpu_admission_mib, stage="hand_motion.estimate") as lease:
                with pipeline(record, prepared, k_virtual[0, 0], vx.shape) as model:
                    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
                    if not capture.isOpened():
                        raise RuntimeError("Could not open hand-estimation RGB")
                    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
                    for index in range(count):
                        row = {"frame_index": index, "video_pts": int(pts[index]), "status": "running", "hands": []}
                        record["frames"].append(row)
                        ok, bgr = capture.read()
                        if not ok:
                            raise RuntimeError(f"Hand RGB decode stopped at frame {index}")
                        if list(bgr.shape[1::-1]) != depth[1]["preprocessing"]["source_size_width_height"]:
                            raise ValueError("Hand RGB dimensions differ from depth preprocessing")
                        bgr = cv2.remap(bgr, source_x, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        bgr = cv2.remap(bgr, vx, vy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        bgr[~virtual_valid] = 0
                        mask_data = frame_arrays(masks, index, int(pts[index]), record, np)
                        if (mask_data["masks"].shape != (len(instances), *vx.shape) or
                                mask_data["present"].shape != (len(instances),) or
                                mask_data["masks"].dtype != np.dtype("bool") or mask_data["present"].dtype != np.dtype("bool")):
                            raise ValueError("Invalid hand mask arrays")
                        virtual_masks = np.stack([cv2.remap(
                            mask_data["masks"][layer].astype(np.uint8), vx, vy, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT) != 0 for layer in hand_layers])
                        virtual_masks &= virtual_valid & mask_data["present"][hand_layers, None, None]
                        if not virtual_masks.any():
                            row.update(status="completed", observed_hands=0, outcome="no_hand_masks")
                            continue
                        _observe(model, lease, bgr, virtual_masks, index, row, previous, result, hand_layers, request, np)
            if not result["valid"].any():
                raise ValueError("no_hands: no confident anatomically assigned hand estimates")
        finally:
            if capture is not None:
                capture.release()
            if lease is not None:
                record["gpu"] = lease.report()
            for side in range(2):
                selected = result["valid"][:, side]
                if selected.any():
                    result["betas"][side] = np.median(result["betas_per_frame"][selected, side], axis=0)
            save_npz(directory / "estimates.npz", np, **result)
            digest = sha256_file(directory / "estimates.npz")
            record["artifacts"]["estimates.npz"] = digest
            record["valid_hand_frames"] = result["valid"].sum(axis=0).tolist()
            for row in record["frames"]:
                row.update(artifact="estimates.npz", sha256=digest)
    return directory / "run.json"
