"""Anatomical MANO kinematics in the isolated, explicitly provisioned CPU runtime."""

from track3.provenance import sha256_file
from ._common import arrays, authorize, operation_run, producer, require_input, require_rotations, save_npz
from ._mano import ManoKinematics, load_runtime


def _parameters(data, source, masks, np):
    form = source.get("output_parameter_form")
    if (form not in ("anatomical_unmirrored", "mirrored_right_hand") or
            source.get("translation_convention") != "mano_model_origin_camA_m" or
            source.get("pose_representation") != "axis_angle_full_flat_mean" or
            not source.get("hand_model") or data["sides"].tolist() != ["left", "right"]):
        raise ValueError("Estimator must declare MANO pose, translation, model and side conventions")
    count = len(data["frame_index"])
    if count == 0:
        raise ValueError("Empty hand-estimate clock")
    for key in ("frame_index", "pts"):
        if data[key].shape != (count,) or data[key].dtype != np.dtype("int64"):
            raise ValueError("Hand-estimate clocks must be one-dimensional int64 arrays")
    if not np.array_equal(data["frame_index"], np.arange(count)) or (np.diff(data["pts"]) <= 0).any():
        raise ValueError("Kinematics requires the full ordered RGB clip clock")
    shapes = {"valid": (count, 2), "score": (count, 2), "mask_index": (count, 2),
              "wrist_orient_camA": (count, 2, 3), "finger_pose": (count, 2, 45),
              "translation_camA_m": (count, 2, 3), "betas": (2, 10)}
    if any(data[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Invalid hand-estimate parameter dimensions")
    valid = data["valid"]
    if valid.dtype != np.dtype("bool") or data["mask_index"].dtype.kind not in "iu":
        raise ValueError("Hand-estimate validity must be boolean and mask indices integral")
    for key in ("score", "wrist_orient_camA", "finger_pose", "translation_camA_m"):
        if not np.isfinite(data[key][valid]).all():
            raise ValueError(f"Non-finite observed MANO parameters: {key}")
    if ((data["score"][valid] < 0) | (data["score"][valid] > 1)).any():
        raise ValueError("Observed hand scores must lie in [0, 1]")
    if (data["translation_camA_m"][valid, 2] <= 0).any():
        raise ValueError("Observed MANO origin must lie in front of the camera")
    instances = masks[1]["instances"]
    if len(set(instances)) != len(instances) or not {"hand_0", "hand_1"}.issubset(instances):
        raise ValueError("Mask producer must have unique hand_0 and hand_1 layers")
    hand_layers = [instances.index("hand_0"), instances.index("hand_1")]
    both = valid.all(axis=1)
    if (not np.isin(data["mask_index"][valid], hand_layers).all() or
            (data["mask_index"][both, 0] == data["mask_index"][both, 1]).any()):
        raise ValueError("Observed hand sides must map to distinct named hand layers")
    for side in range(2):
        if valid[:, side].any() and not np.isfinite(data["betas"][side]).all():
            raise ValueError("Observed side needs one finite episode shape vector")
    return count, form


def _initial_result(data, count, form, np):
    """Copy the estimate, un-mirror if declared, and mark every side invalid until its batch succeeds."""
    result = {key: data[key].copy() for key in (
        "sides", "frame_index", "pts", "valid", "score", "mask_index", "betas",
        "wrist_orient_camA", "finger_pose")}
    result.update(joints_camA=np.full((count, 2, 21, 3), np.nan, dtype=np.float64),
                  joint_rotations_camA=np.full((count, 2, 21, 3, 3), np.nan, dtype=np.float64),
                  surface_points_camA=np.full((count, 2, 778, 3), np.nan, dtype=np.float64))
    if form == "mirrored_right_hand":
        # A reflection of the x axis negates y/z in an axial rotation vector.
        result["wrist_orient_camA"][:, 0] *= np.array([1, -1, -1])
        result["finger_pose"][:, 0] = (result["finger_pose"][:, 0].reshape(count, 15, 3)
                                       * np.array([1, -1, -1])).reshape(count, 45)
    for key in ("wrist_orient_camA", "finger_pose", "score"):
        result[key] = result[key].astype(np.float32)
        result[key][~data["valid"]] = np.nan
    result["betas"] = result["betas"].astype(np.float32)
    result["betas"][~data["valid"].any(axis=0)] = np.nan
    result["mask_index"] = result["mask_index"].astype(np.int64)
    result["mask_index"][~data["valid"]] = -1
    result["valid"][:] = False
    return result


def kinematics(request):
    authorize(request, "kinematics")
    estimate = producer(request.estimate_run, "hand_motion.estimate", request)
    masks = producer(request.masks_run, "object_masks.generate", request)
    depth = producer(request.depth_run, "depth.generate", request)
    require_input(estimate, masks)
    require_input(estimate, depth)
    require_input(masks, depth)
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(parent[1].get("grid") != grid for parent in (estimate, masks)):
        raise ValueError("Hand-estimate and mask pixel grids differ from depth")
    with operation_run(request, [estimate, masks, depth], "kinematics") as (directory, record):
        runtime = load_runtime(record)
        import numpy as np

        data = arrays(estimate, "estimates.npz", record, np)
        count, form = _parameters(data, estimate[1], masks, np)
        result = _initial_result(data, count, form, np)
        record.update(grid=grid, hand_model=estimate[1]["hand_model"], orientation_source="mano_fk",
                      joint_order="MANO21", joint_frame_convention="MANO", units="metres",
                      input_parameter_form=form, output_parameter_form="anatomical_unmirrored",
                      unmirroring_applied=form == "mirrored_right_hand", interpolation="none",
                      translation_convention="mano_model_origin_camA_m", center_idx=None, flat_hand_mean=True)
        record["frames"] = [{"frame_index": i, "video_pts": int(data["pts"][i]), "status": "running",
                             "sides_completed": []} for i in range(count)]
        try:
            for side, name in enumerate(("left", "right")):
                indices = np.flatnonzero(data["valid"][:, side])
                if not len(indices):
                    continue
                model = ManoKinematics(runtime, name)
                for start in range(0, len(indices), request.batch_size):
                    ids = indices[start:start + request.batch_size]
                    joints, rotations, surface = model.forward(
                        result["wrist_orient_camA"][ids, side], result["finger_pose"][ids, side],
                        result["betas"][side], data["translation_camA_m"][ids, side], np)
                    if (joints.shape != (len(ids), 21, 3) or rotations.shape != (len(ids), 21, 3, 3) or
                            surface.shape != (len(ids), 778, 3) or not np.isfinite(joints).all() or
                            not np.isfinite(rotations).all() or not np.isfinite(surface).all()):
                        raise ValueError("MANO backend produced malformed joints, rotations or surface")
                    require_rotations(rotations, np)
                    if (joints[:, 0, 2] <= 0).any():
                        raise ValueError("MANO backend produced an invalid wrist depth")
                    result["joints_camA"][ids, side] = joints
                    result["joint_rotations_camA"][ids, side] = rotations
                    result["surface_points_camA"][ids, side] = surface
                    result["valid"][ids, side] = True
                    for index in ids:
                        record["frames"][int(index)]["sides_completed"].append(name)
                del model
            for row in record["frames"]:
                row["status"] = "completed"
            if not result["valid"].any():
                raise ValueError("no_hands: no observed hand parameters to evaluate")
        finally:
            for key in ("wrist_orient_camA", "finger_pose", "score"):
                result[key][~result["valid"]] = np.nan
            save_npz(directory / "kinematics.npz", np, **result)
            digest = sha256_file(directory / "kinematics.npz")
            record["artifacts"]["kinematics.npz"] = digest
            record["valid_hand_frames"] = result["valid"].sum(axis=0).tolist()
            for row in record["frames"]:
                row.update(artifact="kinematics.npz", sha256=digest)
    return directory / "run.json"
