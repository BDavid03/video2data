"""Disabled, bounded moving-object observed fusion; no geometry completion."""

from dataclasses import dataclass
from functools import partial
import math

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, ObjectMeshRequest, arrays, artifact, authorize, frame_arrays,
                      mesh_metadata, operation_run, producer, read_json, require_input)
from ._fusion import OpeningVolume
from ._points import OpeningPoints


@dataclass(frozen=True)
class MovingFusionRequest(ObjectMeshRequest):
    method: str = "moving_tsdf"
    frame_start: int = 0
    frame_stop: int = 600
    frame_stride: int = 5
    minimum_pose_confidence: float = 0.5
    minimum_surface_support: float = 0.5
    maximum_pose_residual_m: float = 0.03

    def __post_init__(self):
        super().__post_init__()
        if self.source_split != "public" or self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42):
            raise PermissionError("Moving fusion admits frozen training/tuning episodes only; excludes 41/43/44")
        if self.method != "moving_tsdf" or self.scans_only or self.reconstruction_run or self.cluster_assignments:
            raise ValueError("Moving fusion uses named masks and measured trajectories only")
        if (any(type(v) is not int for v in (self.frame_start, self.frame_stop, self.frame_stride)) or
                not 0 <= self.frame_start < self.frame_stop or self.frame_stride < 1 or
                len(range(self.frame_start, self.frame_stop, self.frame_stride)) > 120):
            raise ValueError("Declare an ordered interval of at most 120 sampled frames")
        for value in (self.minimum_pose_confidence, self.minimum_surface_support):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Pose/support thresholds must lie in [0, 1]")
        if not math.isfinite(self.maximum_pose_residual_m) or self.maximum_pose_residual_m <= 0:
            raise ValueError("Declare a finite positive pose residual limit")


def _rigid(matrix, np):
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError("Fusion requires proper rigid measured camera/object transforms")


def _parents(request):
    pose = producer(request.pose_run, "object_pose.track", request)
    masks = producer(request.masks_run, "object_masks.generate", request)
    depth = producer(request.depth_run, "depth.generate", request)
    anchor = producer(request.anchor_run, "stereo_depth.anchor", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    for source, dependencies in ((pose, [masks, depth, anchor, camera]), (masks, [depth]),
                                 (anchor, [depth]), (camera, [depth, anchor])):
        for parent in dependencies:
            require_input(source, parent)
    grid = depth[1]["preprocessing_artifact_sha256"]
    if any(source.get("grid") != grid for _, source in (pose, masks, anchor, camera)):
        raise ValueError("Moving fusion parents differ in pixel grid")
    if pose[1].get("interpolation") != "none" or pose[1].get("pending_objects"):
        raise ValueError("Fusion requires complete measured tracking without interpolation")
    return pose, masks, depth, anchor, camera, grid


def _tracking(pose, masks, camera, record, np):
    """Object trajectory and camera motion on one validated full frame clock."""
    trajectory = arrays(pose, "poses.npz", record, np)
    motion = arrays(camera, "trajectory.npz", record, np)
    names = trajectory["object_names"].tolist()
    bodies, count = len(names), len(trajectory["frame_index"])
    if (names != pose[1]["object_names"] or names != record["object_names"] or
            masks[1]["instances"] != [*names, "hand_0", "hand_1"]):
        raise ValueError("Fusion requires exact named object order followed by both hand layers")
    for key in ("frame_index", "pts"):
        if (trajectory[key].shape != (count,) or trajectory[key].dtype != np.dtype("int64") or
                not np.array_equal(trajectory[key], motion[key])):
            raise ValueError("Fusion camera/object clocks differ")
    if not np.array_equal(trajectory["frame_index"], np.arange(count)) or (np.diff(trajectory["pts"]) <= 0).any():
        raise ValueError("Fusion requires a full strictly ordered frame clock")
    for key in ("valid", "confidence_valid", "confidence", "surface_support", "registration_residual_m"):
        if trajectory[key].shape != (count, bodies):
            raise ValueError("Fusion pose support shape differs")
    if (trajectory["valid"].dtype != np.dtype("bool") or trajectory["confidence_valid"].dtype != np.dtype("bool") or
            trajectory["camA_T_object"].shape != (count, bodies, 4, 4) or
            motion["valid"].shape != (count,) or motion["valid"].dtype != np.dtype("bool") or
            motion["world_T_camA"].shape != (count, 4, 4)):
        raise ValueError("Invalid fusion transform or validity arrays")
    return trajectory, motion, names


def _pinhole(depth, grid, record, np):
    with np.load(artifact(depth[0], "preprocessing.npz", record, grid), allow_pickle=False) as data:
        k, grid_valid = data["input_intrinsics_pixels"].copy(), data["input_valid"].copy()
    if (grid_valid.ndim != 2 or grid_valid.dtype != np.dtype("bool") or k.shape != (3, 3) or
            not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose([k[0, 1], k[1, 0]], 0, atol=1e-8, rtol=0)):
        raise ValueError("Invalid fusion pinhole grid")
    return k, grid_valid


def _scales(anchor, grid, count, pts, record):
    document = read_json(artifact(anchor[0], "anchor.json", record, anchor[1]["artifacts"]["anchor.json"]))
    if document.get("grid") != grid:
        raise ValueError("Fusion anchor grid differs")
    scales = {}
    for row in document["frames"]:
        index = row["frame_index"]
        if type(index) is not int or not 0 <= index < count or index in scales or row["video_pts"] != int(pts[index]):
            raise ValueError("Invalid or duplicate fusion anchor timestamp")
        scales[index] = row
    return scales


def _observation(depth, masks, index, pts, scale, slot, *, k, grid_valid, bodies, record, np):
    """Metric depth, the unoccluded single-identity object mask and valid depth for one frame."""
    mono = frame_arrays(depth, index, pts, record, np)
    layers = frame_arrays(masks, index, pts, record, np)
    z = mono["depth_m"].astype(np.float64) * scale
    if (z.shape != grid_valid.shape or mono["valid"].shape != z.shape or
            mono["valid"].dtype != np.dtype("bool") or
            not np.allclose(mono["input_intrinsics_pixels"], k, atol=1e-5, rtol=0) or
            layers["masks"].shape != (bodies + 2, *z.shape) or
            layers["masks"].dtype != np.dtype("bool") or
            layers["present"].shape != (bodies + 2,) or layers["present"].dtype != np.dtype("bool")):
        raise ValueError("Fusion depth/masks differ from the declared pixel grid")
    visible = layers["masks"] & layers["present"][:, None, None]
    mask = visible[slot] & ~visible[-2:].any(axis=0) & (visible[:bodies].sum(axis=0) == 1)
    return z, mask, grid_valid & mono["valid"] & np.isfinite(z) & (z > 0)


def _pose_supported(trajectory, motion, index, slot, request, np):
    return (motion["valid"][index] and trajectory["valid"][index, slot] and
            trajectory["confidence_valid"][index, slot] and
            np.isfinite(trajectory["confidence"][index, slot]) and
            trajectory["confidence"][index, slot] >= request.minimum_pose_confidence and
            np.isfinite(trajectory["surface_support"][index, slot]) and
            trajectory["surface_support"][index, slot] >= request.minimum_surface_support and
            np.isfinite(trajectory["registration_residual_m"][index, slot]) and
            0 <= trajectory["registration_residual_m"][index, slot] <= request.maximum_pose_residual_m)


def _admit(name, slot, frames, trajectory, motion, scales, observe, k, reports, request, np):
    """Accumulate object-frame points from supported frames; returns points and admitted views."""
    points = OpeningPoints(request.voxel_size_m, request.maximum_point_voxels, np)
    admitted = []
    for index in frames:
        detail = {"frame_index": index, "video_pts": int(trajectory["pts"][index]), "status": "unsupported"}
        reports.append(detail)
        scale = scales.get(index)
        if not _pose_supported(trajectory, motion, index, slot, request, np):
            detail["reason"] = "camera_or_measured_pose_support_absent"
            continue
        if (scale is None or scale["status"] != "completed" or scale["scale_status"] != "observed" or
                not math.isfinite(scale["scale"]) or scale["scale"] <= 0):
            detail["reason"] = "observed_metric_scale_absent"
            continue
        world_t_camera = motion["world_T_camA"][index]
        camera_t_object = trajectory["camA_T_object"][index, slot]
        _rigid(world_t_camera, np)
        _rigid(camera_t_object, np)
        world_t_object = world_t_camera @ camera_t_object
        object_t_camera = np.linalg.inv(world_t_object) @ world_t_camera
        z, mask, valid = observe(index, detail["video_pts"], scale["scale"], slot)
        detail["support_pixels"] = int((mask & valid).sum())
        if detail["support_pixels"] < request.minimum_points:
            detail.update(reason="insufficient_unoccluded_metric_pixels")
            continue
        points.add_depth(z, mask & valid, k, object_t_camera)
        admitted.append((index, object_t_camera, world_t_object))
        detail.update(status="observed", reason="accepted")
    if len(admitted) < request.minimum_fusion_views:
        raise ValueError(f"insufficient_views: {name} lacks supported moving observations")
    return points, admitted


def fuse(request):
    if not isinstance(request, MovingFusionRequest):
        raise TypeError("Moving fusion requires an explicitly bounded MovingFusionRequest")
    authorize("fuse", request)
    pose, masks, depth, anchor, camera, grid = _parents(request)
    with operation_run("fuse", request, [pose, masks, depth, anchor, camera]) as (directory, record):
        import numpy as np
        import trimesh

        trajectory, motion, names = _tracking(pose, masks, camera, record, np)
        count = len(trajectory["frame_index"])
        frames = list(range(request.frame_start, min(request.frame_stop, count), request.frame_stride))
        if len(frames) < request.minimum_fusion_views:
            raise ValueError("insufficient_views: sampled interval is too short")
        body_frames = pose[1]["mesh_body_frames"]
        if [item["name"] for item in body_frames] != names:
            raise ValueError("Fusion requires recorded body origins in exact object order")
        k, grid_valid = _pinhole(depth, grid, record, np)
        scales = _scales(anchor, grid, count, trajectory["pts"], record)
        selected_names = request.objects or names
        if not set(selected_names).issubset(names):
            raise ValueError("Unknown fusion object")
        record.update(grid=grid, object_names=selected_names, method=request.method,
                      mesh_body_frames=[body_frames[names.index(name)] for name in selected_names],
                      interpolation="none", completion=False,
                      confidence_note="Pose confidence and view counts are support diagnostics, not calibrated uncertainty")
        observe = partial(_observation, depth, masks, k=k, grid_valid=grid_valid, bodies=len(names), record=record, np=np)
        objects = []
        record["observation_reports"] = {}
        for name in selected_names:
            slot = names.index(name)
            reports = record["observation_reports"][name] = []
            points, admitted = _admit(name, slot, frames, trajectory, motion, scales, observe, k, reports, request, np)
            cloud = points.centroids()
            if len(cloud) < request.minimum_points:
                raise ValueError(f"insufficient_views: {name} has too few observed voxels")
            volume = OpeningVolume(cloud, request, np)
            for index, object_t_camera, _ in admitted:
                z, mask, valid = observe(index, int(trajectory["pts"][index]), scales[index]["scale"], slot)
                volume.integrate(index, z, mask, valid, k, object_t_camera)
            surface, fusion = volume.surface(trimesh)
            fusion["grid_origin_object_m"] = fusion.pop("grid_origin_world_m")
            fusion.update(coordinate_frame="inherited_object_body", independent_view_coverage="unmeasured",
                          completed_surface=False, support_is_uncertainty=False)
            # Measurement counts live beside the mesh; they do not imply independent viewpoints.
            support_path = directory / f"{name}_support.npz"
            with support_path.open("xb") as stream:
                np.savez_compressed(stream, voxel_cells=points.cells, point_counts=points.counts,
                                    centroids_object_m=cloud, tsdf_weight=volume.weight.reshape(volume.shape),
                                    tsdf_sum=volume.total.reshape(volume.shape), grid_origin_object_m=volume.origin,
                                    voxel_size_m=np.asarray(request.voxel_size_m))
            destination = directory / f"{name}.glb"
            with destination.open("xb") as stream:
                stream.write(surface.export(file_type="glb"))
            item = {"name": name, "file": destination.name, "sha256": sha256_file(destination),
                    "source": "reconstructed", "method": request.method,
                    "body_frame": body_frames[slot]["body_frame"], "inherited_body_frame": body_frames[slot],
                    "frame_rule": {"origin": "unchanged_from_tracking_mesh", "axes": "unchanged_from_tracking_mesh",
                                   "completion": "none", "collision_ready": False},
                    "world_T_object": admitted[0][2].tolist(), "reference_frame_index": admitted[0][0],
                    "source_split": request.source_split, "source_episode": request.episode_index,
                    "source_frames": [index for index, _, _ in admitted], "fusion": fusion,
                    "point_fusion": points.summary(), "support_file": support_path.name,
                    "support_sha256": sha256_file(support_path), "observation_report": reports,
                    "camera_run": input_file(ROOT, camera[0]), "pose_run": input_file(ROOT, pose[0]),
                    **mesh_metadata(surface, np)}
            objects.append(item)
            record["artifacts"][destination.name] = item["sha256"]
            record["artifacts"][support_path.name] = item["support_sha256"]
            del volume, points, cloud, surface
        write_json(directory / "mesh.json", {"objects": objects, "units": "metres", "provenance": record["provenance"],
                                             "completion": False, "collision_ready": False})
        record["artifacts"]["mesh.json"] = sha256_file(directory / "mesh.json")
        record["frames"] = [{"frame_index": index, "video_pts": int(trajectory["pts"][index]), "status": "completed",
                             "artifact": "mesh.json", "sha256": record["artifacts"]["mesh.json"]} for index in frames]
    return directory / "run.json"
