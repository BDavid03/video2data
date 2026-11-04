"""Opening-depth hulls or observed TSDF surfaces, without scan-shape substitution."""

from functools import partial
import math

from track3.provenance import input_file, sha256_file, write_json
from ._common import (FEATURE, ROOT, artifact, arrays, authorize, frame_arrays, mesh_metadata,
                      operation_run, producer, read_json)
from ._fusion import OpeningVolume
from ._points import OpeningPoints


def _body_frame(hull, name, settings, np, trimesh):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = hull.centroid
    if name != "wooden_spoon":
        return pose, {"origin": "surface_centroid", "axes": "world_axes", "provisional": True}
    if not settings.get("convention_review_note"):
        raise ValueError("Spoon frame requires a recorded human review of the frame-convention report")
    if (settings["origin"] not in ("surface_centroid", "base_center") or
            settings["x_axis"] != "longest_principal_axis_bowl_to_handle" or
            settings["z_axis"] != "shortest_principal_axis_towards_world_up"):
        raise ValueError("Unsupported explicit provisional spoon frame rule")
    surface, _ = trimesh.sample.sample_surface(hull, 4000, seed=31)
    _, vectors = np.linalg.eigh(np.cov((surface - hull.centroid).T))
    x, z = vectors[:, -1], vectors[:, 0]
    if z[2] < 0:
        z = -z
    axis = (surface - hull.centroid) @ x
    residual = surface - hull.centroid - axis[:, None] * x
    low = np.linalg.norm(residual[axis <= np.quantile(axis, 0.2)], axis=1)
    high = np.linalg.norm(residual[axis >= np.quantile(axis, 0.8)], axis=1)
    if np.median(high) > np.median(low):
        x = -x
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    pose[:3, :3] = np.column_stack((x, y, z))
    if settings["origin"] == "base_center":
        body_vertices = (hull.vertices - pose[:3, 3]) @ pose[:3, :3]
        base = np.array([(body_vertices[:, 0].min() + body_vertices[:, 0].max()) / 2,
                         (body_vertices[:, 1].min() + body_vertices[:, 1].max()) / 2, body_vertices[:, 2].min()])
        pose[:3, 3] += pose[:3, :3] @ base
    return pose, {**settings, "provisional": True, "sign_rule": "The narrower end is the handle",
                  "end_widths_m": [float(np.median(low)), float(np.median(high))],
                  "sign_ambiguous": bool(abs(np.median(low) - np.median(high)) < 0.002)}


def _parents(request):
    masks = producer(request.masks_run, "object_masks.opening_clusters", request)
    depth = producer(request.depth_run, "depth.generate", request)
    anchor = producer(request.anchor_run, "stereo_depth.anchor", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    pose = producer(request.pose_run, None, request) if request.pose_run else None
    if pose and pose[1]["kind"] not in ("object_pose.initialise", "object_pose.compose"):
        raise ValueError("Hull object assignment requires a geometric opening pose run")
    if pose and any(input_file(ROOT, parent[0]) not in pose[1]["inputs"]
                    for parent in (masks, depth, anchor, camera)):
        raise ValueError("Hull assignments must consume the same masks, depth, anchor and camera runs")
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(source.get("grid") != grid for _, source in [masks, anchor, camera] + ([pose] if pose else [])):
        raise ValueError("Hull inputs use different colour pixel grids")
    return masks, depth, anchor, camera, pose, grid


def _assignments(request, record, masks, pose):
    """Selected object names, their opening-cluster indices and the cluster count."""
    names = request.objects or record["object_names"]
    if not set(names).issubset(record["object_names"]) or masks[1]["object_names"] != record["object_names"]:
        raise ValueError("Hull objects must belong to the selected episode")
    assignments = dict(request.cluster_assignments)
    if pose:
        opening = read_json(artifact(pose[0], "opening.json", record, pose[1]["artifacts"]["opening.json"]))
        if not request.objects and opening.get("pending_objects"):
            names = opening["pending_objects"]
            if names != ["wooden_spoon"] or "wooden_spoon" not in record["object_names"]:
                raise ValueError("Only the episode's spoon may be pending reconstruction")
        if assignments and assignments != opening["object_assignments"]:
            raise ValueError("Request and opening pose disagree on cluster assignments")
        assignments = opening["object_assignments"]
    clusters = len(masks[1]["instances"])
    if any(name not in assignments or type(assignments[name]) is not int or not 0 <= assignments[name] < clusters for name in names):
        raise ValueError("Explicit object-to-opening-cluster assignments are required")
    if len(set(assignments.values())) != len(assignments):
        raise ValueError("insufficient_views: merged objects do not have separable opening clusters")
    return names, assignments, clusters


def _frame(depth, masks, index, pts, scale, *, k, layer_shape, record, np):
    """Metric depth, cluster layers and valid-depth mask for one opening frame."""
    mono = frame_arrays(depth, index, pts, record, np)
    layers = frame_arrays(masks, index, pts, record, np)["masks"].astype(bool)
    z = mono["depth_m"].astype(np.float64) * scale
    if z.shape != layer_shape[1:] or layers.shape != layer_shape or not np.allclose(k, mono["input_intrinsics_pixels"], atol=1e-5):
        raise ValueError("Hull products differ from the declared pixel grid")
    return z, layers, mono["valid"].astype(bool) & np.isfinite(z) & (z > 0)


def _accumulate(request, record, directory, names, assignments, stop, motion, scales, frame, k, np):
    clouds = {name: OpeningPoints(request.voxel_size_m, request.maximum_point_voxels, np) for name in names}
    record["point_fusion"] = {}
    for index in range(stop):
        pts = int(motion["pts"][index])
        record["frames"].append({"frame_index": index, "video_pts": pts, "status": "running"})
        if not motion["valid"][index]:
            raise ValueError("insufficient_views: missing opening camera measurement")
        transform = motion["world_T_camA"][index]
        if (not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]) or
                not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
            raise ValueError("Invalid observed camera transform")
        scale = scales.get(index)
        if (scale is None or scale["scale_status"] != "observed" or scale["video_pts"] != pts or
                not math.isfinite(scale["scale"]) or scale["scale"] <= 0):
            raise ValueError("insufficient_views: missing observed scale")
        z, layers, valid = frame(index, pts, scale["scale"])
        if any(np.count_nonzero(layer & valid) < request.minimum_points for layer in layers):
            raise ValueError("insufficient_views: an opening cluster has too few metric points")
        for name in names:
            clouds[name].add_depth(z, layers[assignments[name]] & valid, k, transform)
            record["point_fusion"][name] = clouds[name].summary()
        write_json(directory / "run.json", record)
    return clouds


def _closed_hull(cloud, name, settings, np, trimesh):
    """Convex hull of the points plus their table footprint, with its provisional body frame."""
    footprint = cloud.copy()
    footprint[:, 2] = 0.0
    cloud = np.concatenate((cloud, footprint))
    if np.linalg.matrix_rank(cloud - cloud.mean(axis=0)) < 3:
        raise ValueError(f"insufficient_views: {name} does not define a three-dimensional hull")
    hull = trimesh.convex.convex_hull(cloud)
    if not hull.is_watertight or not hull.is_winding_consistent or hull.volume <= 0:
        raise ValueError("Opening points did not produce a closed outward-oriented hull")
    world_t_body, rule = _body_frame(hull, name, settings, np, trimesh)
    rule.update(bottom_completion="Each fused point contributes its table-world (x, y, 0) footprint",
                principal_axes_source="Surface of the hull after table-footprint completion")
    return hull, None, world_t_body, rule


def _observed_surface(cloud, layer, request, stop, motion, scales, frame, k, np, trimesh):
    """Open TSDF surface re-reading each opening frame (bounded memory); centroid origin, world axes."""
    volume = OpeningVolume(cloud, request, np)
    for index in range(stop):
        z, layers, valid = frame(index, int(motion["pts"][index]), scales[index]["scale"])
        volume.integrate(index, z, layers[layer], valid, k, motion["world_T_camA"][index])
    surface, fusion = volume.surface(trimesh)
    world_t_body = np.eye(4, dtype=np.float64)
    world_t_body[:3, 3] = surface.centroid
    return surface, fusion, world_t_body, {
        "origin": "observed_surface_centroid", "axes": "world_axes", "provisional": True,
        "bottom_completion": "None; unobserved surfaces remain open",
        "use": "CD-O reconstruction only; not a collision or spoon-frame replacement"}


def reconstruct(request):
    authorize("reconstruct", request)
    if request.method not in ("opening_hull", "opening_tsdf"):
        raise ValueError("Moving observed surfaces require the disabled fuse operation")
    masks, depth, anchor, camera, pose, grid = _parents(request)
    parents = [masks, depth, anchor, camera] + ([pose] if pose else [])
    with operation_run("reconstruct", request, parents) as (directory, record):
        import numpy as np
        import trimesh

        record.update(grid=grid, method=request.method)
        names, assignments, clusters = _assignments(request, record, masks, pose)
        motion = arrays(camera, "trajectory.npz", record, np)
        count = len(motion["frame_index"])
        stop = masks[1]["opening_stop_frame"]
        if not 0 < stop <= count or not np.array_equal(motion["frame_index"], np.arange(count)):
            raise ValueError("Invalid hull opening frame grid")
        if request.method == "opening_tsdf" and stop < request.minimum_fusion_views:
            raise ValueError("insufficient_views: opening interval is shorter than minimum_fusion_views")
        anchor_path = artifact(anchor[0], "anchor.json", record, anchor[1]["artifacts"]["anchor.json"])
        scales = {row["frame_index"]: row for row in read_json(anchor_path)["frames"]}
        with np.load(artifact(depth[0], "preprocessing.npz", record, grid), allow_pickle=False) as data:
            k, shape = data["input_intrinsics_pixels"].copy(), data["input_valid"].shape
        frame = partial(_frame, depth, masks, k=k, layer_shape=(clusters, *shape), record=record, np=np)
        clouds = _accumulate(request, record, directory, names, assignments, stop, motion, scales, frame, k, np)
        objects = []
        settings = read_json(FEATURE / "sources.json")["spoon_frame"]
        for name in names:
            accumulator = clouds.pop(name)
            cloud, point_fusion = accumulator.centroids(), accumulator.summary()
            del accumulator
            if len(cloud) < request.minimum_points:
                raise ValueError(f"insufficient_views: {name} has too few fused measured points")
            if request.method == "opening_hull":
                hull, fusion, world_t_body, rule = _closed_hull(cloud, name, settings, np, trimesh)
            else:
                hull, fusion, world_t_body, rule = _observed_surface(
                    cloud, assignments[name], request, stop, motion, scales, frame, k, np, trimesh)
            hull.apply_transform(np.linalg.inv(world_t_body))
            destination = directory / f"{name}.glb"
            with destination.open("xb") as stream:
                stream.write(hull.export(file_type="glb"))
            item = {"name": name, "file": destination.name, "sha256": sha256_file(destination),
                    "source": "reconstructed", "body_frame": "provisional_rule", "frame_rule": rule,
                    "world_T_object": world_t_body.tolist(), "method": request.method,
                    "fusion": fusion, "point_fusion": point_fusion,
                    "source_split": request.source_split, "source_episode": request.episode_index,
                    "camera_run": input_file(ROOT, camera[0]),
                    "source_frames": list(range(stop)), **mesh_metadata(hull, np)}
            objects.append(item)
            record["artifacts"][destination.name] = item["sha256"]
        write_json(directory / "mesh.json", {"objects": objects, "units": "metres", "provenance": record["provenance"]})
        record["artifacts"]["mesh.json"] = sha256_file(directory / "mesh.json")
        for row in record["frames"]:
            row.update(status="completed", artifact="mesh.json", sha256=record["artifacts"]["mesh.json"])
    return directory / "run.json"
