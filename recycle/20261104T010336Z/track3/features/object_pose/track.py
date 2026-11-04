"""Measured geometric object tracking with explicit loss and reacquisition."""

from itertools import permutations, product
import math

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, artifact, arrays, authorize, frame_arrays, local_path,
                      operation_run, producer, read_json, require_input, require_rigid, save_npz)
from .initialise import _color, _icp, _load_models, _project, _sample, _score


def _angle(a, b, np):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def _recovery_rotations(base, np):
    # The 24 proper signed axis permutations cover SO(3) coarsely; ICP refines them.
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.eye(3)[:, order] * np.asarray(signs)
            if np.linalg.det(rotation) > 0:
                yield rotation @ base


def _surface_support(model, pose, view, request, np):
    points = model["samples"] @ pose[:3, :3].T + pose[:3, 3]
    ids, pixels = _project(points, view, 1e-6, np)
    if not len(ids):
        return 0.0
    width = view["mask"].shape[1]
    order = np.argsort(points[ids, 2], kind="stable")
    ids, pixels = ids[order], pixels[order]
    _, first = np.unique(pixels[:, 1] * width + pixels[:, 0], return_index=True)
    ids, pixels = ids[first], pixels[first]
    u, v = pixels.T
    supported = view["mask"][v, u] & np.isfinite(view["depth"][v, u])
    supported &= abs(view["depth"][v, u] - points[ids, 2]) <= request.residual_limit_m
    return float(supported.mean())


def _register(model, cloud, view, seed, recovery, request, np):
    initial = [seed.copy()]
    if recovery:
        for rotation in _recovery_rotations(seed[:3, :3], np):
            candidate = np.eye(4)
            candidate[:3, :3] = rotation
            candidate[:3, 3] = np.median(cloud, axis=0) - np.median(model["samples"] @ rotation.T, axis=0)
            initial.append(candidate)
    candidates = []
    for pose in initial:
        refined, residual, distances = _icp(model, cloud, pose, request, np)
        if not np.isfinite(refined).all() or not math.isfinite(residual) or residual > request.residual_limit_m:
            continue
        support = _surface_support(model, refined, view, request, np)
        confidence = float(np.mean(distances <= request.residual_limit_m) * math.exp(-residual / request.residual_limit_m))
        floor = request.recovery_surface_support if recovery else request.minimum_surface_support
        if support < floor or confidence < request.minimum_confidence:
            continue
        if not recovery and (np.linalg.norm(refined[:3, 3] - seed[:3, 3]) > request.maximum_translation_step_m or
                             _angle(refined[:3, :3], seed[:3, :3], np) > request.maximum_rotation_step_rad):
            continue
        candidates.append({"pose": refined, "residual": residual, "color": _color(model, refined, [view], request, np),
                           "surface_support": support, "confidence": confidence})
    if not candidates:
        return None, "registration_lost"
    _score(candidates, request)
    best = min(candidates, key=lambda item: item["cost"])
    # Recovery has no trustworthy temporal orientation; reject unresolved alternatives.
    if recovery:
        for other in candidates:
            separated = (_angle(best["pose"][:3, :3], other["pose"][:3, :3], np) > math.radians(15) or
                         np.linalg.norm(best["pose"][:3, 3] - other["pose"][:3, 3]) > request.residual_limit_m)
            close_geometry = abs(other["residual"] - best["residual"]) <= request.ambiguity_distance_m
            close_color = (best["color"] is None or other["color"] is None or
                           other["color"] <= best["color"] + request.ambiguity_color_margin)
            if separated and close_geometry and close_color:
                return None, "recovery_ambiguous"
    return best, "reinitialised" if recovery else "observed"


def _depth(mono, stereo, scale, k, grid_valid, np):
    z = np.full(grid_valid.shape, np.nan, dtype=np.float64)
    for source, value, valid, intrinsics, factor in (
            (mono, "depth_m", "valid", "input_intrinsics_pixels", scale),
            (stereo, "rgb_depth_m", "rgb_valid", "rgb_intrinsics_pixels", 1.0)):
        if (source[value].shape != z.shape or source[valid].shape != z.shape or
                source[valid].dtype != np.dtype("bool") or source[intrinsics].shape != (3, 3) or
                not np.allclose(source[intrinsics], k, atol=1e-5, rtol=0)):
            raise ValueError("Tracking depth differs from the calibrated pixel grid")
        if factor is not None:
            keep = grid_valid & source[valid] & np.isfinite(source[value]) & (source[value] > 0)
            z[keep] = source[value][keep] * factor
    return z


def _scales(anchor, grid, pts, record, np):
    """Observed per-frame metric scale (or None) from the anchor run, keyed by frame index."""
    scale_doc = read_json(artifact(anchor[0], "anchor.json", record, anchor[1]["artifacts"]["anchor.json"]))
    if scale_doc.get("grid") != grid:
        raise ValueError("Scale document uses a different pixel grid")
    scales = {}
    for row in scale_doc["frames"]:
        index = row["frame_index"]
        if (type(index) is not int or not 0 <= index < len(pts) or index in scales or
                row["video_pts"] != int(pts[index]) or row["status"] != "completed"):
            raise ValueError("Invalid tracking scale frame identity")
        value = row["scale"] if row["scale_status"] == "observed" else None
        if row["scale_status"] == "observed" and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError("Invalid observed tracking depth scale")
        scales[index] = value
    return scales


def _track_frame(row, index, rgb, z, layers, present, k, grid_valid, models, names, result, motion, previous,
                 request, np, cv2):
    """Register every present object from its hand-free, unshared, metric mask pixels at one frame."""
    bodies = len(names)
    layers = layers & present[:, None, None]
    hand_mask = cv2.dilate(layers[-2:].any(axis=0).astype(np.uint8),
                           np.ones((request.hand_mask_dilation, request.hand_mask_dilation), dtype=np.uint8)) != 0
    v, u = np.indices(z.shape)
    rays = np.stack((u, v, np.ones(z.shape)), axis=-1) @ np.linalg.inv(k).T
    xyz = rays * z[..., None]
    depth_valid = grid_valid & np.isfinite(xyz).all(axis=-1) & (z > 0)
    for slot, name in enumerate(names):
        detail = {"name": name, "status": "mask_absent"}
        row["objects"].append(detail)
        if not present[slot]:
            continue
        # Overlapping object identities are not independent observations.
        object_mask = layers[slot] & ~hand_mask & ~(layers[:bodies].sum(axis=0) > 1)
        cloud = _sample(xyz[object_mask & depth_valid], request.point_budget, np)
        detail["support_points"] = len(cloud)
        if len(cloud) < request.minimum_points:
            detail["status"] = "insufficient_depth"
            continue
        last = previous[slot]
        seed = result["camA_T_object"][last, slot].copy()
        compensated = bool(motion["valid"][last] and motion["valid"][index])
        if compensated:
            seed = np.linalg.inv(motion["world_T_camA"][index]) @ motion["world_T_camA"][last] @ seed
        recovery = last != index - 1 or not compensated
        view = {"camera": np.eye(4), "k": k, "rgb": rgb, "depth": z, "mask": object_mask}
        fit, status = _register(models[name], cloud, view, seed, recovery, request, np)
        if fit is None and not recovery:
            recovery = True
            fit, status = _register(models[name], cloud, view, seed, True, request, np)
        detail["status"] = status
        if fit is None:
            continue
        result["valid"][index, slot] = True
        result["camA_T_object"][index, slot] = fit["pose"]
        result["confidence"][index, slot] = fit["confidence"]
        result["confidence_valid"][index, slot] = True
        result["surface_support"][index, slot] = fit["surface_support"]
        result["registration_residual_m"][index, slot] = fit["residual"]
        result["reinitialised"][index, slot] = recovery
        detail.update(residual_m=fit["residual"], surface_support=fit["surface_support"],
                      confidence=fit["confidence"], color_residual=fit["color"])
        previous[slot] = index


def track(request):
    authorize("track", request)
    opening_parent = producer(request.pose_run, "object_pose.initialise", request)
    mesh = producer(request.mesh_run, "object_mesh.resolve", request)
    masks = producer(request.masks_run, "object_masks.generate", request)
    depth = producer(request.depth_run, "depth.generate", request)
    stereo = producer(request.stereo_run, "stereo_depth.generate", request)
    anchor = producer(request.anchor_run, "stereo_depth.anchor", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    parents = [opening_parent, mesh, masks, depth, stereo, anchor, camera]
    if opening_parent[1].get("pending_objects") or mesh[1].get("pending_objects"):
        raise ValueError("Tracking requires the complete opening and all object meshes")
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(source.get("grid") != grid for _, source in (opening_parent, masks, stereo, anchor, camera)):
        raise ValueError("Tracking inputs use different colour pixel grids")
    for source, inputs in ((opening_parent, (mesh, depth, anchor, camera)), (masks, (depth,)),
                           (stereo, (depth,)), (anchor, (depth, stereo)), (camera, (depth, stereo, anchor))):
        for parent in inputs:
            require_input(source, parent)
    with operation_run("track", request, parents) as (directory, record):
        import cv2
        import numpy as np
        import trimesh

        models = _load_models(mesh, record, request, np, trimesh)
        # A reconstructed object's opening orientation defines its body frame,
        # not a constraint on the moving object's rotations.
        for model in models.values():
            model["prior"] = None
        result = arrays(opening_parent, "poses.npz", record, np)
        motion = arrays(camera, "trajectory.npz", record, np)
        opening = read_json(artifact(opening_parent[0], "opening.json", record, opening_parent[1]["artifacts"]["opening.json"]))
        names = result["object_names"].tolist()
        instances = masks[1]["instances"]
        if (not names or len(set(names)) != len(names) or names != mesh[1]["object_names"] or
                set(models) != set(names) or names != [item["name"] for item in opening["objects"]] or
                instances != [*names, "hand_0", "hand_1"]):
            raise ValueError("Tracking requires named object masks in exact object-slot order, followed by both hand layers")
        body_frames = [models[name]["record"] for name in names]
        if body_frames != opening_parent[1]["mesh_body_frames"]:
            raise ValueError("Tracking mesh body frames differ from the opening")
        count, bodies = len(result["frame_index"]), len(names)
        stop = opening["stop_frame"]
        if type(stop) is not int or not 0 < stop <= count or opening.get("grid") != grid:
            raise ValueError("Invalid opening interval or pixel grid")
        for key in ("frame_index", "pts"):
            if (result[key].shape != (count,) or result[key].dtype != np.dtype("int64") or
                    motion[key].dtype != np.dtype("int64") or not np.array_equal(result[key], motion[key])):
                raise ValueError("Tracking and camera clocks differ")
        if not np.array_equal(result["frame_index"], np.arange(count)) or (np.diff(result["pts"]) <= 0).any():
            raise ValueError("Tracking requires the full ordered RGB clip clock")
        if (result["valid"].shape != (count, bodies) or result["valid"].dtype != np.dtype("bool") or
                result["camA_T_object"].shape != (count, bodies, 4, 4) or
                motion["valid"].shape != (count,) or motion["valid"].dtype != np.dtype("bool") or
                motion["world_T_camA"].shape != (count, 4, 4)):
            raise ValueError("Invalid tracking pose/validity dimensions")
        require_rigid(result["camA_T_object"][result["valid"]], np)
        require_rigid(motion["world_T_camA"][motion["valid"]], np)
        if not result["valid"][:stop].any(axis=0).all():
            raise ValueError("Every object needs an observed opening pose")
        result["valid"][stop:] = False
        result["camA_T_object"][~result["valid"]] = np.nan
        result["world_valid"] = np.zeros((count, bodies), dtype=bool)
        result["world_T_object"] = np.full((count, bodies, 4, 4), np.nan, dtype=np.float64)
        for key in ("confidence", "visible_fraction", "surface_support", "registration_residual_m"):
            result[key] = np.full((count, bodies), np.nan, dtype=np.float32)
        for key in ("confidence_valid", "visible_fraction_valid", "reinitialised"):
            result[key] = np.zeros((count, bodies), dtype=bool)
        grid_path = artifact(depth[0], "preprocessing.npz", record, grid)
        with np.load(grid_path, allow_pickle=False) as data:
            k, mx, my = data["input_intrinsics_pixels"].copy(), data["source_x"].copy(), data["source_y"].copy()
            grid_valid = data["input_valid"].copy()
        if (grid_valid.ndim != 2 or grid_valid.dtype != np.dtype("bool") or mx.shape != grid_valid.shape or
                my.shape != grid_valid.shape or k.shape != (3, 3) or not np.isfinite(k).all() or
                k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Invalid tracking pinhole grid")
        scales = _scales(anchor, grid, result["pts"], record, np)
        video = local_path(depth[1]["source_video"])
        entry = input_file(ROOT, video)
        if entry["sha256"] != depth[1]["input_hashes"]["video_sha256"]:
            raise ValueError("Tracking RGB differs from depth input")
        record["inputs"].append(entry)
        record.update(grid=grid, object_names=names, mesh_body_frames=body_frames, pending_objects=[],
                      interpolation="none", opening_quality=opening["opening_quality"],
                      confidence_note="Geometric inlier share times exp(-mean residual / residual limit); not calibrated",
                      visibility_note="No triangle renderer: visible_fraction remains NaN/invalid; surface_support is sampled pixel coverage")
        previous = [int(np.flatnonzero(result["valid"][:stop, slot])[-1]) for slot in range(bodies)]
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
        try:
            if not capture.isOpened():
                raise RuntimeError("Could not open tracking RGB")
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            for index in range(count):
                ok, rgb = capture.read()
                if not ok:
                    raise RuntimeError(f"Tracking RGB decode stopped at frame {index}")
                if list(rgb.shape[1::-1]) != depth[1]["preprocessing"]["source_size_width_height"]:
                    raise ValueError("Tracking RGB dimensions differ from depth preprocessing")
                row = {"frame_index": index, "video_pts": int(result["pts"][index]), "status": "running", "objects": []}
                record["frames"].append(row)
                if index < stop:
                    row.update(status="completed", source="opening", observed_objects=int(result["valid"][index].sum()))
                    continue
                rgb = cv2.cvtColor(cv2.remap(rgb, mx, my, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                mono = frame_arrays(depth, index, row["video_pts"], record, np)
                stereo_frame = frame_arrays(stereo, index, row["video_pts"], record, np)
                z = _depth(mono, stereo_frame, scales.get(index), k, grid_valid, np)
                layer_data = frame_arrays(masks, index, row["video_pts"], record, np)
                layers, present = layer_data["masks"], layer_data["present"]
                if (layers.shape != (bodies + 2, *z.shape) or present.shape != (bodies + 2,) or
                        layers.dtype != np.dtype("bool") or present.dtype != np.dtype("bool")):
                    raise ValueError("Invalid tracking mask arrays")
                _track_frame(row, index, rgb, z, layers, present, k, grid_valid, models, names, result, motion,
                             previous, request, np, cv2)
                row.update(status="completed", observed_objects=int(result["valid"][index].sum()))
        finally:
            capture.release()
            save_npz(directory / "poses.npz", np, **result)
            record["artifacts"]["poses.npz"] = sha256_file(directory / "poses.npz")
            opening["provenance"] = record["provenance"]
            write_json(directory / "opening.json", opening)
            record["artifacts"]["opening.json"] = sha256_file(directory / "opening.json")
            for row in record["frames"]:
                if row["status"] == "completed":
                    row.update(artifact="poses.npz", sha256=record["artifacts"]["poses.npz"])
            record["observed_frames_per_object"] = result["valid"].sum(axis=0).tolist()
            record["lost_frames_per_object"] = (~result["valid"]).sum(axis=0).tolist()
    return directory / "run.json"
