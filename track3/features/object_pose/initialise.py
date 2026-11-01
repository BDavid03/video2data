"""Opening registration with one-way ICP, joint assignment and colour tie-breaking."""

import math

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, artifact, arrays, authorize, average, frame_arrays, local_path,
                      operation_run, producer, read_json, require_rigid, save_npz)


def _sample(points, budget, np):
    if len(points) <= budget:
        return points
    return points[np.linspace(0, len(points) - 1, budget, dtype=np.int64)]


def _within_prior(model, transform, np):
    prior = model.get("prior")
    if prior is None:
        return True
    cosine = (np.trace(prior[:3, :3].T @ transform[:3, :3]) - 1) / 2
    return cosine >= -1e-8


def _icp(model, observed, initial, request, np, fixed_rotation=False):
    transform = initial.copy()
    points = _sample(observed, request.point_budget, np)
    for _ in range(request.icp_iterations):
        body = (points - transform[:3, 3]) @ transform[:3, :3]
        distance, nearest = model["tree"].query(body)
        limit = max(request.minimum_points, math.ceil(len(points) * request.trim_fraction))
        keep = np.argsort(distance)[:limit]
        source, target = model["samples"][nearest[keep]], points[keep]
        a, b = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - a).T @ (target - b))
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(vt.T @ u.T)
        rotation = initial[:3, :3] if fixed_rotation else vt.T @ correction @ u.T
        updated = np.eye(4, dtype=np.float64)
        updated[:3, :3], updated[:3, 3] = rotation, b - rotation @ a
        if not _within_prior(model, updated, np):
            break
        change = np.linalg.norm(updated - transform)
        transform = updated
        if change < 1e-7:
            break
    distance = model["tree"].query((points - transform[:3, 3]) @ transform[:3, :3])[0]
    return transform, float(np.mean(distance)), distance


def _project(points, view, near, np):
    """Indices and rounded pixels of camera-frame points beyond ``near`` that land inside the view."""
    ids = np.flatnonzero(points[:, 2] > near)
    projected = points[ids] @ view["k"].T
    uv = projected[:, :2] / projected[:, 2:3]
    height, width = view["mask"].shape
    inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    return ids[inside], np.rint(uv[inside]).astype(np.int64)


def _color(model, transform, views, request, np):
    if model["colors"] is None:
        return None
    vertices = model["vertices"] @ transform[:3, :3].T + transform[:3, 3]
    errors = []
    for view in views:
        pose = view["camera"]
        points = (vertices - pose[:3, 3]) @ pose[:3, :3]
        ids, pixels = _project(points, view, 0, np)
        if not len(ids):
            continue
        u, v = pixels.T
        visible = (view["mask"][v, u] & np.isfinite(view["depth"][v, u])
                   & (abs(points[ids, 2] - view["depth"][v, u]) <= request.residual_limit_m))
        if visible.any():
            errors.extend(np.linalg.norm(view["rgb"][v[visible], u[visible]].astype(float) / 255
                                         - model["colors"][ids[visible]], axis=1).tolist())
    return float(np.median(errors)) if len(errors) >= request.minimum_points else None


def _score(candidates, request):
    worst = max((item["color"] for item in candidates if item["color"] is not None), default=0.0)
    for item in candidates:
        item["cost"] = item["residual"] + request.color_weight_m * (
            item["color"] if item["color"] is not None else worst)


def _color_sweep(model, cloud, views, best, axes, request, np):
    from scipy.spatial.transform import Rotation

    ambiguous = []
    world_axes = [best["pose"][:3, :3] @ axis for axis in axes]
    decided, refined = False, False
    for axis in world_axes:
        base = best["pose"].copy()
        center = base[:3, :3] @ model["mesh"].centroid + base[:3, 3]
        candidates = []
        for degrees in range(-180, 180):
            pose = base.copy()
            pose[:3, :3] = Rotation.from_rotvec(axis * np.radians(degrees)).as_matrix() @ base[:3, :3]
            pose[:3, 3] = center - pose[:3, :3] @ model["mesh"].centroid
            if not _within_prior(model, pose, np):
                continue
            distance = model["tree"].query((cloud - pose[:3, 3]) @ pose[:3, :3])[0]
            residual = float(distance.mean())
            if abs(residual - best["residual"]) <= request.ambiguity_distance_m:
                candidates.append({"pose": pose, "residual": residual,
                                   "color": _color(model, pose, views, request, np), "angle": degrees})
        measured = [item for item in candidates if item["color"] is not None]
        if not measured:
            ambiguous.append(axis)
            continue
        winner = min(measured, key=lambda item: item["color"])
        refined = True
        alternatives = [item for item in candidates
                        if abs((item["angle"] - winner["angle"] + 180) % 360 - 180) > 2]
        unresolved = any(item["color"] is None or
                         item["color"] <= winner["color"] + request.ambiguity_color_margin
                         for item in alternatives)
        if unresolved:
            ambiguous.append(axis)
        else:
            decided = True
        best = winner
    best.update(rotation_ambiguous=bool(ambiguous), color_rotation_decided=decided,
                color_rotation_refined=refined,
                ambiguity_axes_object=[(best["pose"][:3, :3].T @ axis).tolist() for axis in ambiguous])
    return best


def _fit(model, cloud, views, request, np):
    from scipy.spatial.transform import Rotation

    if len(cloud) < request.minimum_points:
        return None
    candidates, seeds = [], []
    for stable in ([] if model.get("prior") is not None else model["stable"]):
        for angle in np.linspace(0, 2 * np.pi, request.yaw_steps, endpoint=False):
            rotation = Rotation.from_rotvec([0, 0, angle]).as_matrix() @ stable[:3, :3]
            initial = np.eye(4, dtype=np.float64)
            initial[:3, :3] = rotation
            rotated = model["samples"] @ rotation.T
            initial[:3, 3] = np.median(cloud, axis=0) - np.median(rotated, axis=0)
            initial[2, 3] = -float((model["vertices"] @ rotation.T)[:, 2].min())
            seeds.append(initial)
    if model.get("prior") is not None:
        seeds.append(model["prior"].copy())
    for initial in seeds:
        transform, residual, _ = _icp(model, cloud, initial, request, np)
        if _within_prior(model, transform, np):
            candidates.append({"pose": transform, "residual": residual,
                               "color": _color(model, transform, views, request, np)})
    if not candidates:
        raise ValueError("No stable resting orientation is available for an object mesh")
    _score(candidates, request)
    best = min(candidates, key=lambda item: item["cost"])
    axes = []
    for other in candidates:
        delta = Rotation.from_matrix(best["pose"][:3, :3].T @ other["pose"][:3, :3]).as_rotvec()
        angle = np.linalg.norm(delta)
        if angle > np.radians(15) and abs(best["residual"] - other["residual"]) <= request.ambiguity_distance_m:
            axis = delta / angle
            if all(abs(np.dot(axis, old)) < 0.98 for old in axes):
                axes.append(axis)
    best = _color_sweep(model, cloud, views, best, axes, request, np)
    _score([*candidates, best], request)
    return best


def _load_models(mesh_parent, record, request, np, trimesh):
    from scipy.spatial import cKDTree

    path = artifact(mesh_parent[0], "mesh.json", record, mesh_parent[1]["artifacts"]["mesh.json"])
    document, result = read_json(path), {}
    for item in document["objects"]:
        mesh_path = artifact(mesh_parent[0], item["file"], record, item["sha256"])
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or
                not np.isfinite(mesh.vertices).all() or item["name"] in result):
            raise ValueError("Expected one finite triangle mesh per named object")
        samples, _ = trimesh.sample.sample_surface(mesh, request.mesh_sample_count, seed=request.seed)
        stable, _ = trimesh.poses.compute_stable_poses(
            mesh, center_mass=mesh.convex_hull.center_mass, sigma=0, n_samples=1, seed=request.seed)
        prior = None
        if item["source"] == "reconstructed":
            if (item.get("source_split") != request.source_split or item.get("source_episode") != request.episode_index
                    or item.get("camera_run") != input_file(ROOT, local_path(request.camera_run))):
                raise ValueError("Reconstructed body frame must use this episode and camera run")
            prior = np.asarray(item["world_T_object"], dtype=np.float64)
            require_rigid(prior, np)
        ids = np.linspace(0, len(mesh.vertices) - 1, min(len(mesh.vertices), request.mesh_sample_count), dtype=np.int64)
        colors = None
        if mesh.visual.kind in ("texture", "vertex", "face"):
            visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
            colors = np.asarray(visual.vertex_colors)[ids, :3].astype(np.float64) / 255
        result[item["name"]] = {"mesh": mesh, "samples": samples, "tree": cKDTree(samples),
                                 "stable": stable, "vertices": np.asarray(mesh.vertices)[ids],
                                 "colors": colors, "record": item, "prior": prior}
    return result


def _opening_views(request, grid, depth, masks, anchor, motion, stop, cluster_count, record, np, cv2):
    """Decode opening frames into per-cluster sampled world clouds and colour views."""
    with np.load(artifact(depth[0], "preprocessing.npz", record, grid), allow_pickle=False) as data:
        k, mx, my = data["input_intrinsics_pixels"].copy(), data["source_x"].copy(), data["source_y"].copy()
    anchor_path = artifact(anchor[0], "anchor.json", record, anchor[1]["artifacts"]["anchor.json"])
    scale_rows = {row["frame_index"]: row for row in read_json(anchor_path)["frames"]}
    video = local_path(depth[1]["source_video"])
    video_entry = input_file(ROOT, video)
    if video_entry["sha256"] != depth[1]["input_hashes"]["video_sha256"]:
        raise ValueError("Opening RGB differs from the depth producer's input")
    record["inputs"].append(video_entry)
    clouds = [[] for _ in range(cluster_count)]
    views = [[] for _ in range(cluster_count)]
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open opening RGB")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        for index in range(stop):
            ok, rgb = capture.read()
            if not ok:
                raise RuntimeError(f"Opening RGB decode stopped at frame {index}")
            if list(rgb.shape[1::-1]) != depth[1]["preprocessing"]["source_size_width_height"]:
                raise ValueError("Opening RGB dimensions differ from depth preprocessing")
            rgb = cv2.cvtColor(cv2.remap(rgb, mx, my, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            pts = int(motion["pts"][index])
            if not motion["valid"][index]:
                raise ValueError("Opening camera pose is missing")
            pose = motion["world_T_camA"][index]
            require_rigid(pose, np)
            mono = frame_arrays(depth, index, pts, record, np)
            mask_data = frame_arrays(masks, index, pts, record, np)
            scale = scale_rows.get(index)
            if scale is None or scale["scale_status"] != "observed" or scale["video_pts"] != pts:
                raise ValueError("Opening metric scale is missing")
            if not math.isfinite(scale["scale"]) or scale["scale"] <= 0:
                raise ValueError("Invalid metric scale")
            z = mono["depth_m"].astype(np.float64) * scale["scale"]
            layers = mask_data["masks"].astype(bool)
            if (z.shape != mx.shape or layers.shape != (cluster_count, *z.shape) or
                    not np.allclose(k, mono["input_intrinsics_pixels"], atol=1e-5)):
                raise ValueError("Opening products disagree on grid or cluster count")
            valid = mono["valid"].astype(bool) & np.isfinite(z) & (z > 0)
            v, u = np.indices(z.shape)
            xyz = np.stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1)
            xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
            groups = [xyz[layer & valid] for layer in layers]
            if any(len(points) < request.minimum_points for points in groups):
                raise ValueError("Opening cluster has insufficient metric support")
            for cluster in range(cluster_count):
                clouds[cluster].append(_sample(groups[cluster], request.point_budget, np))
                views[cluster].append({"index": index, "camera": pose, "k": k, "rgb": rgb,
                                       "depth": z, "mask": layers[cluster] & valid})
    finally:
        capture.release()
    return clouds, views


def _assign(models, names, active, fused, views, request, np):
    """Fit scans to opening clusters jointly; returns fits keyed (slot, cluster) and slot->cluster."""
    from scipy.optimize import linear_sum_assignment

    cluster_count = len(fused)
    fits, assignment = {}, {}
    spoon = names.index("wooden_spoon") if "wooden_spoon" in names else None
    scan_slots = [slot for slot in active if slot != spoon]
    if spoon is not None and cluster_count != len(names):
        raise ValueError("opening_objects_inseparable: spoon assignment needs one distinct leftover cluster")
    if cluster_count >= len(names):
        for slot in scan_slots:
            for cluster in range(cluster_count):
                fits[slot, cluster] = _fit(models[names[slot]], fused[cluster], views[cluster], request, np)
        _score(list(fits.values()), request)
        if scan_slots:
            costs = np.asarray([[fits[slot, cluster]["cost"] if
                                 (spoon is None or fits[slot, cluster]["residual"] <= request.residual_limit_m)
                                 else np.inf for cluster in range(cluster_count)] for slot in scan_slots])
            slots, clusters = linear_sum_assignment(costs)
            assignment = {scan_slots[slot]: int(cluster) for slot, cluster in zip(slots, clusters)}
        if spoon is not None:
            leftover = set(range(cluster_count)) - set(assignment.values())
            if len(leftover) != 1:
                raise ValueError("opening_objects_inseparable: no unique spoon cluster")
            cluster = leftover.pop()
            if any(fits[slot, cluster]["residual"] <= request.residual_limit_m for slot in scan_slots):
                raise ValueError("opening_objects_inseparable: a listed scan also explains the leftover cluster")
            assignment[spoon] = cluster
            if spoon in active:
                fits[spoon, cluster] = _fit(models["wooden_spoon"], fused[cluster], views[cluster], request, np)
    elif cluster_count == 1:
        remaining = fused[0].copy()
        for slot in sorted(range(len(names)), key=lambda i: -float(np.prod(models[names[i]]["mesh"].extents))):
            model = models[names[slot]]
            fit = _fit(model, remaining, views[0], request, np)
            if fit is None:
                raise ValueError("opening_objects_inseparable: too few remaining points")
            distance = model["tree"].query((remaining - fit["pose"][:3, 3]) @ fit["pose"][:3, :3])[0]
            explained = distance <= request.residual_limit_m
            if int(explained.sum()) < request.minimum_points:
                raise ValueError("opening_objects_inseparable: a scan explains too few points")
            fit["pose"], fit["residual"], _ = _icp(model, remaining[explained], fit["pose"], request, np,
                                                   fixed_rotation=fit["color_rotation_refined"])
            fits[slot, 0], assignment[slot] = fit, 0
            remaining = remaining[~explained]
    else:
        raise ValueError("opening_objects_inseparable: unsupported merged cluster layout")
    return fits, assignment


def _empty_poses(names, count, pts, np):
    bodies = len(names)
    return {"object_names": np.asarray(names), "frame_index": np.arange(count, dtype=np.int64),
            "pts": pts.astype(np.int64),
            "camA_T_object": np.full((count, bodies, 4, 4), np.nan, dtype=np.float64),
            "world_T_object": np.full((count, bodies, 4, 4), np.nan, dtype=np.float64),
            "valid": np.zeros((count, bodies), dtype=bool), "world_valid": np.zeros((count, bodies), dtype=bool),
            "confidence": np.full((count, bodies), np.nan, dtype=np.float32),
            "confidence_valid": np.zeros((count, bodies), dtype=bool),
            "visible_fraction": np.full((count, bodies), np.nan, dtype=np.float32),
            "visible_fraction_valid": np.zeros((count, bodies), dtype=bool),
            "reinitialised": np.zeros((count, bodies), dtype=bool)}


def initialise(request):
    authorize("initialise", request)
    mesh = producer(request.mesh_run, "object_mesh.resolve", request)
    masks = producer(request.masks_run, "object_masks.opening_clusters", request)
    depth = producer(request.depth_run, "depth.generate", request)
    anchor = producer(request.anchor_run, "stereo_depth.anchor", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    parents = [mesh, masks, depth, anchor, camera]
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(source.get("grid") != grid for _, source in (masks, anchor, camera)):
        raise ValueError("Object-pose inputs use different colour pixel grids")
    with operation_run("initialise", request, parents) as (directory, record):
        import cv2
        import numpy as np
        import trimesh

        record["grid"] = grid
        models = _load_models(mesh, record, request, np, trimesh)
        names = masks[1]["object_names"]
        pending = [name for name in names if name not in models]
        if (names != mesh[1]["object_names"] or not set(models).issubset(names) or len(names) != len(set(names)) or
                pending not in ([], ["wooden_spoon"]) or pending != mesh[1].get("pending_objects", [])):
            raise ValueError("Only the first-pass wooden_spoon mesh may be unresolved")
        record["object_names"] = names
        record["mesh_body_frames"] = [models[name]["record"] for name in names if name in models]
        record["pending_objects"] = pending
        record["initialisation_pass"] = "scans_only" if pending else "complete"
        motion = arrays(camera, "trajectory.npz", record, np)
        count = len(motion["frame_index"])
        stop = masks[1]["opening_stop_frame"]
        if (not 0 < stop <= count or not np.array_equal(motion["frame_index"], np.arange(count)) or
                motion["world_T_camA"].shape != (count, 4, 4)):
            raise ValueError("Invalid opening interval or camera frame grid")
        cluster_count = len(masks[1]["instances"])
        if cluster_count == 0:
            raise ValueError("opening_objects_inseparable: no opening clusters")
        clouds, views = _opening_views(request, grid, depth, masks, anchor, motion, stop, cluster_count, record, np, cv2)
        fused = [_sample(np.concatenate(group), request.point_budget, np) for group in clouds]
        active = [slot for slot, name in enumerate(names) if name in models]
        fits, assignment = _assign(models, names, active, fused, views, request, np)
        degraded = any(fits[slot, assignment[slot]]["residual"] > request.residual_limit_m for slot in active)
        if degraded and not request.allow_degraded_opening:
            raise ValueError("opening_objects_inseparable: registration exceeds the residual limit")
        result = _empty_poses(names, count, motion["pts"], np)
        opening = {"method": "geometric", "opening_quality": "degraded" if degraded else "measured", "grid": grid,
                   "start_frame": 0, "stop_frame": stop, "objects": [],
                   "object_assignments": {name: assignment[i] for i, name in enumerate(names)},
                   "pending_objects": pending, "initialisation_pass": record["initialisation_pass"],
                   "next_step": "reconstruct pending spoon, resolve all meshes, initialise again" if pending else None,
                   "provenance": record["provenance"]}
        for index in range(stop):
            record["frames"].append({"frame_index": index, "video_pts": int(result["pts"][index]), "status": "running"})
            for slot in active:
                cluster = assignment[slot]
                cloud = clouds[cluster][index]
                fit = fits[slot, cluster]
                if cluster_count < len(names):
                    distances = np.asarray([models[names[s]]["tree"].query(
                        (cloud - fits[s, 0]["pose"][:3, 3]) @ fits[s, 0]["pose"][:3, :3])[0] for s in range(len(names))])
                    cloud = cloud[(np.argmin(distances, axis=0) == slot) & (distances[slot] <= request.residual_limit_m)]
                if len(cloud) < request.minimum_points:
                    continue
                pose, residual, _ = _icp(models[names[slot]], cloud, fit["pose"], request, np,
                                         fixed_rotation=fit["color_rotation_refined"])
                if residual <= request.residual_limit_m or request.allow_degraded_opening:
                    if residual > request.residual_limit_m:
                        opening["opening_quality"] = "degraded"
                    result["camA_T_object"][index, slot] = np.linalg.inv(motion["world_T_camA"][index]) @ pose
                    result["valid"][index, slot] = True
            record["frames"][-1].update(observed_objects=int(result["valid"][index].sum()))
        save_npz(directory / "poses.npz", np, **result)
        record["artifacts"]["poses.npz"] = sha256_file(directory / "poses.npz")
        for row in record["frames"]:
            row.update(status="completed", artifact="poses.npz", sha256=record["artifacts"]["poses.npz"])
        for slot in active:
            name = names[slot]
            ids = np.flatnonzero(result["valid"][:, slot])
            if not len(ids):
                raise ValueError(f"Object {name} was never registered in an opening frame")
            mean, spread_m, spread_degrees = average(motion["world_T_camA"][ids] @ result["camA_T_object"][ids, slot], np)
            fit = fits[slot, assignment[slot]]
            opening["objects"].append({"name": name, "world_T_object": mean.tolist(),
                "observed_frames": ids.tolist(), "spread_m": spread_m, "spread_degrees": spread_degrees,
                "registration_residual_m": fit["residual"], "color_residual": fit["color"],
                "rotation_ambiguous": fit["rotation_ambiguous"], "ambiguity_axes_object": fit["ambiguity_axes_object"],
                "color_rotation_refined": fit["color_rotation_refined"], "color_sweep_step_degrees": 1,
                "method": "geometric"})
        write_json(directory / "opening.json", opening)
        record["artifacts"]["opening.json"] = sha256_file(directory / "opening.json")
        record["confidence_note"] = "Geometric registration has no tracker score or rendered visibility fraction"
        record["opening_quality"] = opening["opening_quality"]
    return directory / "run.json"
