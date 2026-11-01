"""Resting and full scene bundles with explicit fills, seating and triangle contacts."""

import shutil

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, arrays, artifact, authorize, clock_row, operation_run, producer,
                      read_json, require_input, save_npz, template_producer)
from ._geometry import closest, seating_offset
from ._contacts import LINKS, contract as contact_contract, queries as contact_queries, reduce_links
from ._hands import blend as blend_hand, contract as hand_completion_contract

POSE_KINDS = ("object_pose.initialise", "object_pose.compose", "object_pose.refine")
HAND_FIELDS = {"joint_order": "MANO21", "joint_frame_convention": "MANO", "units": "metres",
               "output_parameter_form": "anatomical_unmirrored", "orientation_source": "mano_fk"}
# Scene asset paths keep the historical absolute prefix that consumers resolve via track3.paths.
ASSET_PREFIX = "/mnt/z/hugging-face/video2data/"


def _segments(mask, np):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _rotations(rotations, np):
    if (not np.isfinite(rotations).all() or not np.allclose(
            np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-5) or
            not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError("Invalid observed rotation matrices")


def _average(poses, np, weights=None):
    from scipy.spatial.transform import Rotation

    result = np.eye(4)
    result[:3, 3] = np.median(poses[:, :3, 3], axis=0) if weights is None else np.average(poses[:, :3, 3], axis=0, weights=weights)
    result[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean(weights=weights).as_matrix()
    return result


def _bridge_objects(poses, valid, request, np):
    from scipy.spatial.transform import Rotation, Slerp

    filled = np.zeros(valid.shape, dtype=bool)
    for slot in range(valid.shape[1]):
        for start, end in _segments(~valid[:, slot], np):
            if start == 0 or end == len(poses) or end - start > request.maximum_gap_frames:
                raise ValueError(f"Object slot {slot} has an unbridgeable gap [{start}, {end})")
            a, b = poses[start - 1, slot], poses[end, slot]
            fraction = np.arange(1, end - start + 1) / (end - start + 1)
            poses[start:end, slot] = np.eye(4)
            poses[start:end, slot, :3, 3] = (1 - fraction)[:, None] * a[:3, 3] + fraction[:, None] * b[:3, 3]
            poses[start:end, slot, :3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(fraction).as_matrix()
            filled[start:end, slot] = True
    return filled


def _parking(poses, meshes, template, request, np):
    if (template.get("source") is None or template.get("joint_order") != "MANO21" or
            template.get("joint_frame_convention") != "anatomical_unmirrored" or
            template.get("sides") != ["left", "right"] or template.get("palms") != "down"):
        raise ValueError("parked_template_missing: supply a sourced, palms-down MANO21 template")
    joints = np.asarray(template["joints_relative_to_wrist_m"], dtype=np.float64)
    rotations = np.asarray(template["joint_rotations_world"], dtype=np.float64)
    surface = np.asarray(template["surface_points_relative_to_wrist_m"], dtype=np.float64)
    if joints.shape != (2, 21, 3) or rotations.shape != (2, 21, 3, 3) or not np.isfinite(joints).all():
        raise ValueError("Invalid parked-hand template dimensions or joint positions")
    if (surface.shape != (2, 778, 3) or not np.isfinite(surface).all() or
            not np.allclose(joints[:, 0], 0, atol=1e-8, rtol=0)):
        raise ValueError("Parked-hand geometry must be finite and relative to a zero wrist")
    _rotations(rotations, np)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for frame in poses:
        for pose, mesh in zip(frame, meshes):
            vertices = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
            low, high = np.minimum(low, vertices.min(axis=0)), np.maximum(high, vertices.max(axis=0))
    for side in range(2):
        radius = float(max(np.max(np.linalg.norm(joints[side], axis=1)),
                           np.max(np.linalg.norm(surface[side], axis=1))))
        if not 0.05 < radius < 0.4:
            raise ValueError("Parked-hand surface bounds are incompatible with metres")
        clearance = request.parking_clearance_m
        y = high[1] + radius + clearance if side == 0 else low[1] - radius - clearance
        joints[side] += np.array([low[0] - radius - clearance, y, max(0., high[2]) + radius + clearance])
    return joints, rotations


def _complete_hands(joints, rotations, valid, parked, parked_rotations, request, np):
    source = np.zeros(valid.shape, dtype=np.uint8)
    for side in range(2):
        for start, end in _segments(~valid[:, side], np):
            if start > 0 and end < len(joints) and end - start <= request.maximum_gap_frames:
                a, b = joints[start - 1, side].copy(), joints[end, side].copy()
                ra, rb = rotations[start - 1, side].copy(), rotations[end, side].copy()
                for index in range(start, end):
                    fraction = (index - start + 1) / (end - start + 1)
                    joints[index, side], rotations[index, side] = blend_hand(a, b, ra, rb, fraction, np)
                    source[index, side] = 1
            else:
                transit = min(request.parking_transition_frames, max(1, (end - start) // 2))
                for index in range(start, end):
                    joints[index, side], rotations[index, side] = parked[side], parked_rotations[side]
                    if start > 0 and index - start < transit:
                        fraction = (index - start + 1) / transit
                        joints[index, side], rotations[index, side] = blend_hand(
                            joints[start - 1, side], parked[side], rotations[start - 1, side],
                            parked_rotations[side], fraction, np)
                    elif end < len(joints) and end - index <= transit:
                        fraction = (transit - (end - index) + 1) / (transit + 1)
                        joints[index, side], rotations[index, side] = blend_hand(
                            parked[side], joints[end, side], parked_rotations[side],
                            rotations[end, side], fraction, np)
                    source[index, side] = 2
    return source


def _support_order(names, supports):
    if any(name not in names or target not in ["table", *names] or name == target for name, target in supports.items()):
        raise ValueError("Invalid declared support relation")
    order, remaining = [], set(names)
    while remaining:
        ready = [name for name in names if name in remaining and supports.get(name, "table") not in remaining]
        if not ready:
            raise ValueError("Object support relations contain a cycle")
        order.extend(ready)
        remaining.difference_update(ready)
    return [names.index(name) for name in order]


def _surface_sequence(joints, poses, mesh, np):
    rotation, translation = poses[:, :3, :3], poses[:, :3, 3]
    points_per_hand = joints.shape[2]
    point_count = 2 * points_per_hand
    body = (joints.reshape(len(poses), point_count, 3) - translation[:, None]) @ rotation
    points, inverse = np.unique(body.reshape(-1, 3), axis=0, return_inverse=True)
    distance, position, normal, inside = closest(points, mesh.triangles, np)
    position = position[inverse].reshape(-1, point_count, 3) @ np.swapaxes(rotation, -1, -2) + translation[:, None]
    normal = normal[inverse].reshape(-1, point_count, 3) @ np.swapaxes(rotation, -1, -2)
    return (distance[inverse].reshape(-1, 2, points_per_hand), position.reshape(-1, 2, points_per_hand, 3),
            normal.reshape(-1, 2, points_per_hand, 3), inside[inverse].reshape(-1, 2, points_per_hand))


def _admit(request):
    """Resolve every producer and check lineage/grids before a run directory exists."""
    parked = template_producer(request.parked_hand_run)
    pose = producer(request.pose_run, None, request)
    if pose[1]["kind"] not in POSE_KINDS:
        raise ValueError("Scene assembly requires an opening or composed object-pose run")
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    mesh = producer(request.mesh_run, "object_mesh.resolve", request)
    hand = producer(request.hand_run, None, request) if request.mode == "full" else None
    if hand and hand[1]["kind"] not in ("hand_motion.lift", "hand_motion.refine"):
        raise ValueError("Scene assembly requires admitted lifted or refined hand motion")
    if hand and hand[1]["kind"] == "hand_motion.refine":
        lifted = producer(hand[1]["request"]["hand_run"], "hand_motion.lift", request)
        if (input_file(ROOT, lifted[0]) not in hand[1]["inputs"] or
                input_file(ROOT, camera[0]) not in lifted[1]["inputs"] or
                input_file(ROOT, camera[0]) not in hand[1]["inputs"]):
            raise ValueError("Refined hands must preserve exact lifted-hand and camera lineage")
    bound_pose = pose
    if pose[1]["kind"] == "object_pose.refine":
        bound_pose = producer(pose[1]["request"]["pose_run"], "object_pose.compose", request)
        require_input(pose, bound_pose)
        if request.mode == "resting":
            raise ValueError("Refined pose opening summaries remain original; select compose for resting mode")
    if hand and any(hand[1].get(key) != value for key, value in HAND_FIELDS.items()):
        raise ValueError("Scene completion requires metric MANO21 hands with anatomical MANO frames")
    for parent in [camera] + ([hand] if hand else []):
        require_input(bound_pose, parent)
    if pose[1]["kind"] == "object_pose.refine":
        require_input(pose, camera)
    if pose[1].get("pending_objects"):
        raise ValueError("First-pass spoon assignments are not complete opening poses")
    if pose[1]["grid"] != camera[1]["grid"] or (hand and hand[1]["grid"] != camera[1]["grid"]):
        raise ValueError("Scene inputs use different colour pixel grids")
    return pose, camera, mesh, parked, hand


def _clock(request, record):
    clock = clock_row(request, record["dataset_revision"], "A completed clip clock inventory is required")
    pts = clock["pts_integer"]
    if (clock["time_base_and_frame_rate"] != [["1/10240", "20/1"]] or len(pts) != clock["decoded_frame_count"]
            or any(right - left != 512 for left, right in zip(pts, pts[1:]))):
        raise ValueError("Scene frame clock must be the exact 20 Hz RGB clock")
    return clock


def _collision_mesh(path, name, np, trimesh):
    """Load a mesh, merge vertices by position and drop degenerate faces; require a closed positive volume."""
    asset = trimesh.load_mesh(path, process=False)
    if not isinstance(asset, trimesh.Trimesh) or not np.isfinite(asset.vertices).all():
        raise ValueError(f"invalid_collision_mesh: {name}")
    vertices, inverse = np.unique(np.asarray(asset.vertices), axis=0, return_inverse=True)
    faces = inverse[np.asarray(asset.faces)]
    triangles = vertices[faces]
    keep = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > 2e-12
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[keep], process=False)
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError(f"invalid_collision_mesh: {name}")
    return mesh


def _template(parked_parent, directory, record):
    """Admit the parked-hand template and keep a copy inside the scene bundle."""
    path, source = parked_parent
    template_path = artifact(path, "parked_hand.json", record, source["artifacts"]["parked_hand.json"])
    artifact(path, "model_rest.npz", record, source["artifacts"]["model_rest.npz"])
    template = read_json(template_path)
    origin = template.get("source")
    if (template.get("scope") != "model_asset_template" or template.get("observed") is not False or
            template.get("units") != "metres" or template.get("provenance") != "video_only" or
            not isinstance(origin, dict) or origin.get("backend") != source.get("backend") or
            origin.get("model_rest_sha256") != source["artifacts"].get("model_rest.npz")):
        raise ValueError("Parked-hand template differs from its source convention or backend")
    shutil.copyfile(template_path, directory / "parked_hand.json")
    record["artifacts"]["parked_hand.json"] = sha256_file(directory / "parked_hand.json")
    record["parked_hand_template"] = {"run": input_file(ROOT, path), "artifact": input_file(ROOT, template_path),
                                      "geometry": "neutral MANO surface and joints; not observed"}
    return template


def _observed_poses(pose_parent, clock, names, offset, record, request, np):
    count = clock["decoded_frame_count"]
    data = arrays(pose_parent, "poses.npz", record, np)
    if (not np.array_equal(data["frame_index"], np.arange(count)) or
            not np.array_equal(data["pts"], clock["pts_integer"]) or data["object_names"].tolist() != names):
        raise ValueError("Object pose frame or name grid differs from the clip")
    valid = data["world_valid"].astype(bool)
    poses = data["world_T_object"].astype(np.float64)
    if valid.shape != (count, len(names)) or poses.shape != (count, len(names), 4, 4) or not np.isfinite(poses[valid]).all():
        raise ValueError("Malformed observed object poses")
    _rotations(poses[valid, :3, :3], np)
    poses[~valid] = np.nan
    poses[..., 2, 3] += offset
    filled = _bridge_objects(poses, valid, request, np)
    return poses, filled, np.ones(valid.shape, dtype=np.uint8), data["confidence"]


def _observed_hands(hand_parent, clock, offset, parked, parked_rotations, record, request, np):
    count = clock["decoded_frame_count"]
    hand = arrays(hand_parent, "hands.npz", record, np)
    if (hand["sides"].tolist() != ["left", "right"] or not np.array_equal(hand["frame_index"], np.arange(count)) or
            not np.array_equal(hand["pts"], clock["pts_integer"])):
        raise ValueError("Hand identity/frame grid differs from the scene")
    if (any(hand[key].shape != (count, 2) or hand[key].dtype != np.dtype("bool") for key in ("valid", "world_valid"))
            or (hand["world_valid"] & ~hand["valid"]).any()):
        raise ValueError("Hand world validity must be boolean and a subset of camera validity")
    valid = hand["world_valid"]
    joints, rotations = hand["joints_world"].astype(np.float64), hand["joint_rotations_world"].astype(np.float64)
    if joints.shape != (count, 2, 21, 3) or rotations.shape != (count, 2, 21, 3, 3) or not np.isfinite(joints[valid]).all():
        raise ValueError("Malformed observed hand arrays")
    _rotations(rotations[valid], np)
    joints[~valid], rotations[~valid] = np.nan, np.nan
    joints[..., 2] += offset
    source = _complete_hands(joints, rotations, valid, parked, parked_rotations, request, np)
    return joints, rotations, source


def _rest_and_smooth(poses, states, filled, joints, mean_poses, confidence, meshes, request, np):
    """Hold hand-free quiet segments at their mean pose; confidence-smooth the moving frames."""
    from scipy.spatial.transform import Rotation

    count = len(poses)
    for slot, mesh in enumerate(meshes):
        translations = np.linalg.norm(np.diff(poses[:, slot, :3, 3], axis=0), axis=1)
        orientation = Rotation.from_matrix(poses[:, slot, :3, :3])
        quiet = (translations <= request.rest_translation_m) & ((orientation[:-1].inv() * orientation[1:]).magnitude() <= request.rest_rotation_rad)
        candidate = np.ones(count, dtype=bool)
        candidate[1:] &= quiet
        candidate[:-1] &= quiet
        candidate &= ~filled[:, slot]
        for start, end in _segments(candidate, np):
            distance, _, _, inside = _surface_sequence(joints[start:end], poses[start:end, slot], mesh, np)
            candidate[start:end] &= ~inside.any(axis=(1, 2)) & (distance.min(axis=(1, 2)) > request.hand_near_m)
        for start, end in _segments(candidate, np):
            if end - start >= request.minimum_rest_frames:
                poses[start:end, slot] = mean_poses[slot] if start == 0 else _average(poses[start:end, slot], np)
                states[start:end, slot] = 0
        original = poses[:, slot].copy()
        for start, end in _segments(states[:, slot] == 1, np):
            for index in range(start, end):
                low, high = max(start, index - request.smoothing_radius), min(end, index + request.smoothing_radius + 1)
                weights = np.asarray(confidence[low:high, slot], dtype=np.float64)
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
                poses[index, slot] = _average(original[low:high], np, weights)


def _seat(poses, states, meshes, names, request, np):
    """Lower/raise each resting segment onto the table or its declared supporting object."""
    seating = []
    for slot in _support_order(names, request.supports):
        support = request.supports.get(names[slot], "table")
        for start, end in _segments(states[:, slot] == 0, np):
            transform = poses[start, slot]
            vertices = meshes[slot].vertices @ transform[:3, :3].T + transform[:3, 3]
            triangles = None
            if support != "table":
                other = names.index(support)
                if not np.allclose(poses[start:end, other], poses[start, other], atol=1e-6):
                    raise ValueError("A resting object's declared support moves during its resting segment")
                support_pose = poses[start, other]
                triangles = meshes[other].triangles @ support_pose[:3, :3].T + support_pose[:3, 3]
            correction = seating_offset(vertices, triangles, np)
            if abs(correction) > request.seating_limit_m:
                raise ValueError(f"Seating correction for {names[slot]} exceeds the configured limit")
            poses[start:end, slot, 2, 3] += correction
            seating.append({"object": names[slot], "start_frame": int(start), "stop_frame": int(end),
                            "support": support, "correction_m": correction})
    return seating


def _contacts(joints, poses, states, meshes, request, np):
    """Nearest object surface per hand link within the contact distance, plus joint penetration depth."""
    count = len(poses)
    contacts = np.zeros((count, 2, 16, 3), dtype=np.float32)
    normals = np.zeros_like(contacts)
    bodies = np.zeros((count, 2, 16), dtype=np.int32)
    penetration = np.zeros((count, 2, 21), dtype=np.float32)
    minimum = np.full((count, 2, 16), np.inf)
    resting = request.mode == "resting"
    query_points, link_groups = contact_queries(joints[:1] if resting else joints, np)
    for slot, mesh in enumerate(meshes):
        if resting:
            distance, _, _, inside = _surface_sequence(query_points, poses[:1, slot], mesh, np)
            if inside.any() or (distance <= request.contact_distance_m).any():
                raise ValueError("Parked hands contact a resting object")
            penetration[:] = np.maximum(penetration, np.where(inside[:, :, :21], distance[:, :, :21], 0))
            continue
        for start, end in sorted(_segments(states[:, slot] == 0, np) + _segments(states[:, slot] == 1, np)):
            distance, point, normal, inside = _surface_sequence(query_points[start:end], poses[start:end, slot], mesh, np)
            penetration[start:end] = np.maximum(penetration[start:end], np.where(inside[:, :, :21], distance[:, :, :21], 0))
            distance, point, normal = reduce_links(distance, point, normal, link_groups, np)
            take = (distance < minimum[start:end]) & (distance <= request.contact_distance_m)
            contacts[start:end][take] = point[take]
            normals[start:end][take] = normal[take]
            bodies[start:end][take] = slot + 1
            minimum[start:end][take] = distance[take]
    return contacts, normals, bodies, penetration


def _copy_assets(directory, names, asset_paths, collision_paths, items, request, record):
    assets = directory / "assets"
    assets.mkdir()
    objects = []
    for name, path, collision_path, item in zip(names, asset_paths, collision_paths, items):
        destination = assets / f"{name}.glb"
        shutil.copyfile(path, destination)
        record["artifacts"][f"assets/{name}.glb"] = sha256_file(destination)
        collision_destination = assets / f"{name}_collision.glb"
        shutil.copyfile(collision_path, collision_destination)
        collision_hash = sha256_file(collision_destination)
        record["artifacts"][f"assets/{name}_collision.glb"] = collision_hash
        objects.append({**item, "file": ASSET_PREFIX + destination.relative_to(ROOT).as_posix(),
                        "collision_file": ASSET_PREFIX + collision_destination.relative_to(ROOT).as_posix(),
                        "collision_sha256": collision_hash,
                        "collision_geometry_rule": "Merge vertices by position and remove degenerate faces in memory",
                        "mass_kg": request.mass_kg[name], "mass_source": "request estimate"})
    return objects


def _scene_inputs(pose_parent, camera, mesh_parent, record, request, np, trimesh):
    """Admit opening poses, the table world and collision meshes in one consistent slot order."""
    opening = read_json(artifact(pose_parent[0], "opening.json", record, pose_parent[1]["artifacts"]["opening.json"]))
    world = read_json(artifact(camera[0], "world.json", record, camera[1]["artifacts"]["world.json"]))
    mesh_doc = read_json(artifact(mesh_parent[0], "mesh.json", record, mesh_parent[1]["artifacts"]["mesh.json"]))
    names = [item["name"] for item in mesh_doc["objects"]]
    if names != [item["name"] for item in opening["objects"]] or names != pose_parent[1]["object_names"]:
        raise ValueError("Scene object slot order differs between inputs")
    identity = [(item["name"], item["sha256"], item.get("collision_sha256")) for item in mesh_doc["objects"]]
    if [(item["name"], item["sha256"], item.get("collision_sha256"))
            for item in pose_parent[1].get("mesh_body_frames", [])] != identity:
        raise ValueError("Pose mesh hashes differ from the selected resolve run")
    if set(request.mass_kg) != set(names):
        raise ValueError("Supply an explicit mass estimate in kilograms for every scene object")
    if (world["up_axis"] != "+z" or not np.allclose(world["support_plane"]["normal"], [0, 0, 1]) or
            opening["grid"] != camera[1]["grid"]):
        raise ValueError("Scene requires a common z-up table world")
    offset = float(world["support_plane"]["offset_m"])
    meshes, asset_paths, collision_paths = [], [], []
    for item in mesh_doc["objects"]:
        path = artifact(mesh_parent[0], item["file"], record, item["sha256"])
        collision_path = artifact(mesh_parent[0], item["collision_file"], record, item["collision_sha256"]) if item.get("collision_file") else path
        meshes.append(_collision_mesh(collision_path, item["name"], np, trimesh))
        asset_paths.append(path)
        collision_paths.append(collision_path)
    mean_poses = np.asarray([item["world_T_object"] for item in opening["objects"]], dtype=np.float64)
    if mean_poses.shape != (len(names), 4, 4) or not np.isfinite(mean_poses).all() or not np.allclose(mean_poses[:, 3], [0, 0, 0, 1]):
        raise ValueError("An opening pose is missing or malformed")
    _rotations(mean_poses[:, :3, :3], np)
    mean_poses[:, 2, 3] += offset
    return opening, world, mesh_doc, names, offset, meshes, asset_paths, collision_paths, mean_poses


def assemble(request):
    authorize(request)
    pose_parent, camera, mesh_parent, parked_parent, hand_parent = _admit(request)
    parents = [pose_parent, camera, mesh_parent, parked_parent] + ([hand_parent] if hand_parent else [])
    with operation_run(request, parents) as (directory, record):
        import numpy as np
        import trimesh

        clock = _clock(request, record)
        count = clock["decoded_frame_count"]
        (opening, world, mesh_doc, names, offset, meshes, asset_paths, collision_paths,
         mean_poses) = _scene_inputs(pose_parent, camera, mesh_parent, record, request, np, trimesh)
        template = _template(parked_parent, directory, record)
        if request.mode == "resting":
            poses = np.repeat(mean_poses[None], count, axis=0)
            filled = np.zeros((count, len(names)), dtype=bool)
            states = np.zeros((count, len(names)), dtype=np.uint8)
            parked, parked_rotations = _parking(poses[:1], meshes, template, request, np)
            joints = np.repeat(parked[None], count, axis=0)
            rotations = np.repeat(parked_rotations[None], count, axis=0)
            hand_source = np.full((count, 2), 2, dtype=np.uint8)
        else:
            poses, filled, states, confidence = _observed_poses(pose_parent, clock, names, offset, record, request, np)
            parked, parked_rotations = _parking(poses, meshes, template, request, np)
            joints, rotations, hand_source = _observed_hands(hand_parent, clock, offset, parked, parked_rotations,
                                                             record, request, np)
            _rest_and_smooth(poses, states, filled, joints, mean_poses, confidence, meshes, request, np)
        seating = _seat(poses, states, meshes, names, request, np)
        if request.mode == "resting":
            # Re-park against the seated poses.
            parked, parked_rotations = _parking(poses[:1], meshes, template, request, np)
            joints[:] = parked
            rotations[:] = parked_rotations
        contacts, normals, bodies, penetration = _contacts(joints, poses, states, meshes, request, np)
        if not np.isfinite(poses).all() or not np.isfinite(joints).all() or not np.isfinite(rotations).all():
            raise ValueError("Scene completion left nonfinite values")

        objects = _copy_assets(directory, names, asset_paths, collision_paths, mesh_doc["objects"], request, record)
        hand_completion = hand_completion_contract() if request.mode == "full" else {"method": "constant_parked_template"}
        scene = {"fps": 20.0, "frame_count": count, "objects": objects, "mode": request.mode,
                 "parked_hand_template": record["parked_hand_template"],
                 "contact_layout": contact_contract(), "hand_completion": hand_completion,
                 "opening_quality": opening.get("opening_quality", "measured"), "provenance": record["provenance"],
                 "world": {**world, "support_plane": {**world["support_plane"], "offset_m": 0.0},
                           "assembly_z_shift_m": offset}, "grid": camera[1]["grid"]}
        save_npz(directory / "trajectories.npz", np, world_T_object=poses, object_state=states, object_filled=filled,
                 object_names=np.asarray(names), frame_index=np.arange(count, dtype=np.int64),
                 pts=np.asarray(clock["pts_integer"], dtype=np.int64), hand_joints_world=joints,
                 hand_joint_rotations_world=rotations, hand_source=hand_source, contact_position=contacts,
                 contact_normal=normals, contact_body=bodies,
                 contact_link_names=np.asarray([name for name, _ in LINKS]))
        report = {"seating": seating, "object_filled_share": filled.mean(axis=0).tolist(),
                  "hand_filled_share": (hand_source != 0).mean(axis=0).tolist(),
                  "hand_completion": hand_completion,
                  "hand_penetration_depth_m": penetration.tolist(),
                  "rotation_ambiguous": {item["name"]: item["rotation_ambiguous"] for item in opening["objects"]},
                  "contact_layout": contact_contract()}
        write_json(directory / "scene.json", scene)
        write_json(directory / "report.json", report)
        record.update(grid=camera[1]["grid"], object_names=names, opening_quality=scene["opening_quality"])
        for name in ("scene.json", "trajectories.npz", "report.json"):
            record["artifacts"][name] = sha256_file(directory / name)
        record["frames"] = [{"frame_index": index, "video_pts": clock["pts_integer"][index],
                             "status": "completed", "artifact": "trajectories.npz",
                             "sha256": record["artifacts"]["trajectories.npz"]} for index in range(count)]
    return directory / "run.json"
