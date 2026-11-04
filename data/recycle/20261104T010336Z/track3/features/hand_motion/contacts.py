"""Observed-surface contact hypotheses, with explicit physical-contact abstention."""

from dataclasses import dataclass
import math

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, arrays, artifact, authorize, is_number, load_npz, operation_run, producer, read_json,
                      require_input, require_rotations, save_npz)
from .refine import derivatives, research_episode, source_seconds

TIPS = [4, 8, 12, 16, 20]


@dataclass(frozen=True)
class ContactHandsRequest:
    source_split: str = "public"
    episode_index: int = 39
    hand_run: str | None = None
    pose_run: str | None = None
    mesh_run: str | None = None
    maximum_frames: int = 384
    maximum_surface_points: int = 1000000
    maximum_gap_seconds: float = 0.1
    proximity_m: float = 0.015
    approach_speed_m_s: float = 0.03
    comotion_speed_m_s: float = 0.03
    lift_speed_m_s: float = 0.02

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        research_episode(self)
        for key, ceiling in (("maximum_frames", 384), ("maximum_surface_points", 1000000)):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= ceiling:
                raise ValueError(f"Invalid contact resource ceiling: {key}")
        for key in ("maximum_gap_seconds", "proximity_m", "approach_speed_m_s", "comotion_speed_m_s", "lift_speed_m_s"):
            if not is_number(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid contact bound: {key}")


def _intervals(data, side, name, hypotheses, phases, reasons, cues, seconds, gap):
    rows = []
    start = 0
    count = len(phases)
    for stop in range(1, count + 1):
        contiguous = (stop < count and data["frame_index"][stop] == data["frame_index"][stop - 1] + 1 and
                      seconds[stop] - seconds[stop - 1] <= gap)
        if contiguous and (phases[stop], hypotheses[stop], reasons[stop]) == (phases[start], hypotheses[start], reasons[start]):
            continue
        rows.append({"side": side, "object_name": name, "start_frame": int(data["frame_index"][start]),
                     "end_frame": int(data["frame_index"][stop - 1]) + 1,
                     "start_pts": int(data["pts"][start]), "end_pts": int(data["pts"][stop - 1]),
                     "phase": phases[start], "candidate_phase": hypotheses[start],
                     "ambiguity": list(reasons[start]), "accepted": False, "physical_contact_confirmed": False,
                     "support": {"frames": stop - start, "joint_landmarks_observed": False,
                                 "cues": sorted(set(cue for row in cues[start:stop] for cue in row))}})
        start = stop
    return rows


def _surface(mesh, pose, slot, name, item, record, request, np):
    """Admit one object's observed voxel support; return its points, voxel size and lineage."""
    if item.get("method") != "moving_tsdf" or item.get("frame_rule", {}).get("completion") != "none":
        raise ValueError("Contact support must come from observed moving-object fusion")
    path = artifact(mesh[0], item["support_file"], record, item["support_sha256"])
    support = load_npz(path, np)
    points = support["centroids_object_m"]
    voxel = float(np.asarray(support["voxel_size_m"]).reshape(()))
    if (points.ndim != 2 or points.shape[1] != 3 or not 0 < len(points) <= request.maximum_surface_points or
            not np.isfinite(points).all() or not math.isfinite(voxel) or voxel <= 0 or
            support["point_counts"].shape != (len(points),) or (support["point_counts"] <= 0).any()):
        raise ValueError("Invalid observed-surface support points")
    frames = pose[1].get("mesh_body_frames", [])
    if not isinstance(frames, list) or slot >= len(frames) or item.get("inherited_body_frame") != frames[slot]:
        raise ValueError("Observed surface and pose body frames differ")
    lineage = {"object_name": name,
               "support_file": {"path": path.relative_to(ROOT).as_posix(), "sha256": item["support_sha256"]},
               "distance_semantics": "unsigned nearest observed voxel centroid; not signed collision distance",
               "voxel_size_m": voxel, "coverage": "partial_observed_surface"}
    return points, voxel, lineage


def _hypothesis(d, previous, dt, tip_speed, object_velocity, object_moving, voxel, request, np):
    """Multi-cue phase hypothesis for one frame; never a verified physical event."""
    reach = request.proximity_m + math.sqrt(3) * voxel
    tip_near = (d <= reach)[TIPS]
    stable = tip_speed <= request.comotion_speed_m_s
    why = ["fk_joints_inferred", "partial_surface", "no_observed_touch_or_force"]
    evidence = ["metric_wrist_depth", "observed_voxel_patch", "pts_object_relative_motion"]
    candidate = "unknown"
    radial_speed = (float(np.min(d)) - float(np.min(previous))) / dt
    if radial_speed < -request.approach_speed_m_s:
        candidate = "approach"
    if tip_near.any() and (tip_near & stable).any():
        candidate = "contact"
    if tip_near[0] and tip_near[1:].any() and (tip_near & stable).sum() >= 2:
        candidate = "grasp"
        why.append("opposition_and_force_unobserved")
    if candidate in ("contact", "grasp") and object_moving and object_velocity[2] > request.lift_speed_m_s:
        candidate = "lift"
        evidence.append("object_upward_motion")
    if (previous[TIPS] <= reach).any() and not tip_near.any() and radial_speed > request.approach_speed_m_s:
        candidate = "release"
    if candidate == "release" and object_moving and np.linalg.norm(object_velocity) <= request.lift_speed_m_s:
        candidate = "support_transition"
        why.append("supporting_surface_unobserved")
    return candidate, why, evidence


def contacts(request):
    research_episode(request)
    authorize(request, "contacts")
    hand = producer(request.hand_run, "hand_motion.refine", request)
    pose = producer(request.pose_run, "object_pose.compose", request)
    mesh = producer(request.mesh_run, "object_mesh.fuse", request)
    track_path = mesh[1]["request"].get("pose_run")
    if not track_path:
        raise ValueError("Observed fusion must identify its exact tracking parent")
    track = producer(track_path, "object_pose.track", request)
    require_input(mesh, track)
    require_input(pose, track)
    if hand[1].get("grid") != pose[1].get("grid") or mesh[1].get("grid") != pose[1].get("grid"):
        raise ValueError("Hand, object and observed surface grids differ")
    # Both world trajectories must consume exactly the same immutable camera run.
    lift = producer(hand[1]["request"].get("hand_run"), "hand_motion.lift", request)
    require_input(hand, lift)
    camera = producer(lift[1]["request"].get("camera_run"), "camera_motion.estimate", request)
    require_input(lift, camera)
    require_input(pose, camera)
    with operation_run(request, [hand, pose, mesh, track, lift, camera], "contacts") as (directory, record):
        import numpy as np
        from scipy.spatial import cKDTree

        hands = arrays(hand, "hands.npz", record, np)
        objects = arrays(pose, "poses.npz", record, np)
        source_count = len(objects["frame_index"])
        ids = hands["frame_index"]
        if (ids < 0).any() or (ids >= source_count).any() or not np.array_equal(objects["frame_index"][ids], ids):
            raise ValueError("Contact source frames are outside the complete object clock")
        objects = {key: value[ids].copy() if key != "object_names" and value.ndim and value.shape[0] == source_count
                   else value for key, value in objects.items()}
        seconds = source_seconds(request, hands, record, np)
        count = len(seconds)
        if count > request.maximum_frames:
            raise ValueError("Contact processing exceeds declared frame budget")
        if any(not np.array_equal(hands[key], objects[key]) for key in ("pts", "frame_index")):
            raise ValueError("Contact hand and object clocks differ")
        names = objects["object_names"].tolist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate contact object identities")
        mesh_doc = read_json(artifact(mesh, "mesh.json", record))
        entries = {row["name"]: row for row in mesh_doc["objects"]}
        if set(entries) != set(names):
            raise ValueError("Contact surface and pose object rosters differ")
        valid = objects["world_valid"]
        poses = objects["world_T_object"]
        if valid.shape != (count, len(names)) or valid.dtype != np.dtype("bool") or poses.shape != (count, len(names), 4, 4):
            raise ValueError("Malformed contact object trajectories")
        if not np.isfinite(poses[valid]).all() or not np.allclose(poses[valid, 3, :], [0, 0, 0, 1]):
            raise ValueError("Contact objects require valid homogeneous world poses")
        require_rotations(poses[valid, :3, :3], np)
        shape = (count, 2, len(names), 21)
        distances = np.full(shape, np.nan, dtype=np.float64)
        local_points = np.full((*shape, 3), np.nan, dtype=np.float64)
        geometry_valid = np.zeros((count, 2, len(names)), dtype=bool)
        sliding_speed = np.full(shape, np.nan, dtype=np.float64)
        intervals = []
        surface_lineage = []
        for slot, name in enumerate(names):
            points, voxel, lineage = _surface(mesh, pose, slot, name, entries[name], record, request, np)
            tree = cKDTree(points)
            surface_lineage.append(lineage)
            object_velocity, _, object_vvalid, _ = derivatives(poses[:, slot, :3, 3], valid[:, slot], seconds,
                                                               hands["frame_index"], request.maximum_gap_seconds, np)
            for side, side_name in enumerate(("left", "right")):
                keep = valid[:, slot] & hands["world_valid"][:, side] & hands["wrist_depth_supported"][:, side]
                geometry_valid[:, side, slot] = keep
                for index in np.flatnonzero(keep):
                    local = (hands["joints_world"][index, side] - poses[index, slot, :3, 3]) @ poses[index, slot, :3, :3]
                    local_points[index, side, slot] = local
                    distances[index, side, slot] = tree.query(local, k=1)[0]
                local_velocity, _, lvvalid, _ = derivatives(local_points[:, side, slot],
                    np.broadcast_to(keep[:, None], (count, 21)), seconds, hands["frame_index"], request.maximum_gap_seconds, np)
                sliding_speed[:, side, slot] = np.linalg.norm(local_velocity, axis=-1)
                hypotheses, phases, reasons, cues = [], [], [], []
                for index in range(count):
                    candidate, phase, why, evidence = "unknown", "unknown", ["missing_geometry_or_wrist_support"], []
                    if keep[index] and lvvalid[index].all():
                        candidate, why, evidence = _hypothesis(
                            distances[index, side, slot], distances[index - 1, side, slot],
                            seconds[index] - seconds[index - 1],
                            np.linalg.norm(local_velocity[index, TIPS], axis=-1),
                            object_velocity[index], object_vvalid[index], voxel, request, np)
                        phase = candidate
                    hypotheses.append(candidate)
                    phases.append(phase)
                    reasons.append(tuple(why))
                    cues.append(evidence)
                intervals.extend(_intervals(hands, side_name, name, hypotheses, phases, reasons, cues,
                                            seconds, request.maximum_gap_seconds))
        save_npz(directory / "contacts.npz", np, frame_index=hands["frame_index"], pts=hands["pts"],
                 object_names=objects["object_names"], distance_to_observed_voxel_m=distances,
                 hand_joints_object_m=local_points, geometry_valid=geometry_valid,
                 sliding_speed_object_m_s=sliding_speed,
                 physical_contact_confirmed=np.zeros_like(geometry_valid),
                 penetration_m=np.full_like(distances, np.nan))
        document = {"schema_version": 1, "source_split": request.source_split, "episode_index": request.episode_index,
                    "frame_index": hands["frame_index"].tolist(), "pts": hands["pts"].tolist(), "time_base": record["time_base"],
                    "hand_run": input_file(ROOT, hand[0]), "pose_run": input_file(ROOT, pose[0]),
                    "mesh_run": input_file(ROOT, mesh[0]), "intervals": intervals, "surfaces": surface_lineage,
                    "interval_bounds": "end_frame exclusive; end_pts is final included source sample",
                    "semantics": "Phases are inferred multi-cue hypotheses, never verified physical events; accepted remains false. Missing geometry or motion leaves phase unknown.",
                    "penetration": "unavailable: partial observed voxel support is not a closed signed collision surface"}
        write_json(directory / "contacts.json", document)
        record.update(grid=hand[1]["grid"], units="metres", method="observed_voxel_motion_hypotheses",
                      contact_semantics="unconfirmed_hypotheses_only", object_names=names)
        record["artifacts"] = {name: sha256_file(directory / name) for name in ("contacts.npz", "contacts.json")}
        record["frames"] = [{"frame_index": int(frame), "video_pts": int(pts), "status": "completed"}
                            for frame, pts in zip(hands["frame_index"], hands["pts"])]
    return directory / "run.json"
