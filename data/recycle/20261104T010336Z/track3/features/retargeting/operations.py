"""Scene export, rigid assets and unchanged upstream floating-hand retargeting."""

import math
import shutil
from xml.etree import ElementTree as ET

from track3.provenance import input_file, write_json
from ._common import (ROOT, ROSTER, artifact, authorize, bundle_root, linux_path, local_path,
                      operation_run, partition, producer, read_json, run_upstream, runtime_source)
from ._evidence import contact_admission, randomization_proposal


def _scene(request, parent=None):
    value = request.scene_run if parent is None else parent[1]["scene_run"]["path"]
    scene = producer(value, "scene_assembly.assemble", request)
    if parent and input_file(ROOT, scene[0]) != parent[1]["scene_run"]:
        raise ValueError("Scene run changed after retargeting began")
    if request.scene_run and local_path(request.scene_run) != scene[0]:
        raise ValueError("The request selects a different scene")
    return scene


def _identity(scene, request, record):
    document = read_json(artifact(scene, "scene.json", record))
    names = [item["name"] for item in document["objects"]]
    if request.source_split == "evaluation":
        expected = read_json(ROSTER)["episodes"][str(request.episode_index)]
    else:
        inventory = read_json("reports/data-audit/trajectories.json")
        rows = [row for row in inventory["episodes"] if row["episode_index"] == request.episode_index]
        if len(rows) != 1 or inventory["revision"] != record["dataset_revision"]:
            raise ValueError("Missing pinned public object inventory")
        expected = rows[0]["objects"]
    clock = read_json("reports/data-audit/videos.json")
    rows = [row for row in clock["videos"] if row["split"] == request.source_split and
            row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if (len(rows) != 1 or clock["status"] != "completed" or clock["revision"] != record["dataset_revision"] or
            rows[0]["status"] != "ok" or rows[0]["time_base_and_frame_rate"] != [["1/10240", "20/1"]]):
        raise ValueError("Missing exact 20 Hz clip clock")
    if names != expected or document["fps"] != 20.0 or document["frame_count"] != rows[0]["decoded_frame_count"]:
        raise ValueError("Scene object order, frame count or rate differs from the episode")
    pts = rows[0]["pts_integer"]
    if len(pts) != document["frame_count"] or any(b - a != 512 for a, b in zip(pts, pts[1:])):
        raise ValueError("The upstream fixed-rate reference requires an exact 512-tick source PTS grid")
    record.update(scene_run=input_file(ROOT, scene[0]), object_names=names, frame_count=document["frame_count"],
                  scene_mode=document["mode"], opening_quality=document.get("opening_quality", "measured"))
    return document, rows[0]["pts_integer"]


def _schema(record):
    runtime_source("source/robotic_grounding/robotic_grounding/retarget/data_logger.py", record)
    from robotic_grounding.retarget.data_logger import ManoSharpaData
    return ManoSharpaData


def _contact_names(document, record):
    runtime_source("source/robotic_grounding/robotic_grounding/retarget/params.py", record)
    from robotic_grounding.retarget.params import MANO_HAND_LINKS

    expected = [{"name": name, "joint_indices": list(indices)} for name, indices in MANO_HAND_LINKS.items()]
    layout = document.get("contact_layout", {})
    if (len(expected) != 16 or layout.get("schema_version") != 1 or layout.get("joint_order") != "MANO21" or
            layout.get("body_ids") != "one_based_object_slot_zero_absent" or layout.get("links") != expected):
        raise ValueError("Scene contacts must declare the pinned upstream 16-link order and joint groups")
    record["contact_layout"] = layout
    return [item["name"] for item in expected]


def _write_row(row, destination, record):
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row], schema=_schema(record)._schema), destination, compression="zstd")


def _read_row(path):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path)
    if file.metadata.num_rows != 1:
        raise ValueError("Exactly one motion row is required")
    row = file.read().to_pylist()[0]
    for parent in path.parents:
        if parent.name.startswith(("sequence_id=", "robot_name=")):
            key, value = parent.name.split("=", 1)
            if row.get(key) not in (None, value):
                raise ValueError("Parquet identity differs from its partition")
            row[key] = value
    return row


def _copy_bundle(parent, directory, request, record):
    for name in ("contact-admission.json", "scene-randomization-proposal.json"):
        if name in parent[1]["artifacts"]:
            shutil.copyfile(artifact(parent, name, record), directory / name)
            record.setdefault("research_sidecars", []).append(name)
    for name in parent[1]["artifacts"]:
        if not name.startswith("assets/"):
            continue
        source = artifact(parent, name, record)
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".parquet":
            row = _read_row(source)
            for field in ("object_mesh_paths", "object_urdf_paths"):
                row[field] = [linux_path(ROOT, directory / local_path(value).relative_to(parent[0].parent))
                              for value in row[field]]
            _write_row(row, destination, record)
        else:
            shutil.copyfile(source, destination)
    for stage in ("loaded", "processed"):
        files = list(partition(directory, stage, request).glob("*.parquet"))
        if files:
            if len(files) != 1:
                raise ValueError("A stage contains more than one motion file")
            record[stage + "_parquet"] = files[0].relative_to(directory).as_posix()


INERTIA_KEYS = (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))


def _require_rotations(rotations, message):
    import numpy as np
    if (not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-5) or
            not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError(message)


def _scene_arrays(scene, record, pts, contact_names):
    """Load the scene trajectories and validate grid, shapes, rigidity and contact-slot conventions."""
    import numpy as np
    with np.load(artifact(scene, "trajectories.npz", record), allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    count, names = record["frame_count"], record["object_names"]
    if (not np.array_equal(data["frame_index"], np.arange(count)) or not np.array_equal(data["pts"], pts)
            or data["object_names"].tolist() != names):
        raise ValueError("Scene arrays differ from the declared frame/name grid")
    if "contact_link_names" not in data or data["contact_link_names"].tolist() != contact_names:
        raise ValueError("Scene contact-array slots differ from their link metadata")
    shape = {"world_T_object": (count, len(names), 4, 4), "hand_joints_world": (count, 2, 21, 3),
             "hand_joint_rotations_world": (count, 2, 21, 3, 3), "contact_position": (count, 2, 16, 3),
             "contact_normal": (count, 2, 16, 3), "contact_body": (count, 2, 16)}
    for key, expected in shape.items():
        if data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"Incomplete scene array: {key}")
    objects = data["world_T_object"]
    if not np.allclose(objects[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Scene poses must be finite rigid transforms")
    _require_rotations(objects[..., :3, :3], "Scene poses must be finite rigid transforms")
    _require_rotations(data["hand_joint_rotations_world"], "Invalid MANO joint rotations")
    ids = data["contact_body"]
    if ((ids < 0) | (ids > len(names)) | (ids != np.floor(ids))).any():
        raise ValueError("Contact IDs must be integers, zero or one-based object slots")
    active = ids > 0
    if active.any() and not np.allclose(np.linalg.norm(data["contact_normal"][active], axis=-1), 1, atol=1e-4):
        raise ValueError("Active contact normals must be unit length")
    if np.any(data["contact_position"][~active] != 0) or np.any(data["contact_normal"][~active] != 0):
        raise ValueError("Absent contacts must have zero position and normal fields")
    return data


def _admit_contacts(contacts, scene, data, request, pts, directory, record):
    """Zero every contact slot not covered by an admitted research contact interval."""
    import numpy as np
    evidence = read_json(artifact(contacts, "contacts.json", record))
    for key in ("hand_run", "pose_run", "mesh_run"):
        entry = evidence.get(key)
        if not isinstance(entry, dict) or entry not in scene[1]["inputs"]:
            raise ValueError(f"Contact evidence must use the exact scene producer: {key}")
    admitted, report = contact_admission(evidence, request, data["frame_index"].tolist(), pts, record["object_names"])
    ids = data["contact_body"]
    mask = np.zeros(ids.shape, dtype=bool)
    for frame, side, body in admitted:
        mask[frame, side] |= ids[frame, side] == body
    report.update(source_run=input_file(ROOT, contacts[0]), hand_run=evidence.get("hand_run"),
                  original_active_slots=int((ids > 0).sum()), admitted_active_slots=int(mask.sum()))
    for key in ("contact_body", "contact_position", "contact_normal"):
        data[key][~mask] = 0
    write_json(directory / "contact-admission.json", report)
    record.setdefault("research_sidecars", []).append("contact-admission.json")


def _propose_randomization(uncertainty, request, pts, directory, record):
    evidence = read_json(artifact(uncertainty, "scene-errors.json", record))
    proposal = randomization_proposal(evidence, request)
    for component in evidence["components"]:
        if component["object_name"] not in record["object_names"]:
            raise ValueError("Measured scene-error object is absent from scene")
        for support in component["support"]:
            frame = support["frame_index"]
            if frame >= record["frame_count"] or support["pts"] != pts[frame]:
                raise ValueError("Measured scene-error support is outside the scene clock")
            source = support["source_artifact"]
            if input_file(ROOT, local_path(source["path"])) != source or source not in uncertainty[1]["inputs"]:
                raise ValueError("Measured error source must match its hashed producer input")
    proposal.update(source_run=input_file(ROOT, uncertainty[0]), time_base="1/10240")
    write_json(directory / "scene-randomization-proposal.json", proposal)
    record.setdefault("research_sidecars", []).append("scene-randomization-proposal.json")


def _wxyz(rotations):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(*rotations.shape[:-2], 4)


def _motion_row(data, request, scene, directory, contact_names, record):
    """One upstream ManoSharpaData row; optional MANO parameters pass through, never inferred."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    count, names = record["frame_count"], record["object_names"]
    objects, rotations = data["world_T_object"], data["hand_joint_rotations_world"]
    processed = bundle_root(directory) / "processed"
    logger = _schema(record)
    row = {name: ([] if temporal or kind is list or getattr(kind, "__origin__", None) is list else None)
           for name, _, kind, temporal in logger._field_specs}
    row.update(sequence_id=f"episode_{request.episode_index:06d}", robot_name="sharpa_wave", fps=20.0,
               raw_motion_file=linux_path(ROOT, scene[0].parent / "trajectories.npz"),
               mano_flat_hand_mean=True, mano_center_idx=None, mano_to_robot_scale=request.mano_to_robot_scale,
               mano_link_names=contact_names,
               object_name="_".join(names), safe_object_name="_".join(names),
               object_body_names=names, safe_object_body_names=names,
               object_mesh_paths=[linux_path(ROOT, processed / f"{name}.obj") for name in names],
               object_urdf_paths=[linux_path(ROOT, processed / f"{name}_rigid.urdf") for name in names],
               object_mesh_radius=[], object_articulation=np.zeros(count).tolist(),
               object_body_position=objects[..., :3, 3].tolist(),
               object_body_wxyz=_wxyz(objects[..., :3, :3]).tolist(),
               object_root_position=objects[:, 0, :3, 3].tolist(),
               object_root_axis_angle=Rotation.from_matrix(objects[:, 0, :3, :3]).as_rotvec().tolist())
    for side_index, side in enumerate(("left", "right")):
        row[f"mano_{side}_trans"] = data["hand_joints_world"][:, side_index, 0].tolist()
        row[f"mano_{side}_joints"] = data["hand_joints_world"][:, side_index].tolist()
        row[f"mano_{side}_joints_wxyz"] = _wxyz(rotations[:, side_index]).tolist()
        for suffix in ("global_orient", "finger_pose", "betas"):
            key = f"mano_{side}_{suffix}"
            if key in data:
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"Nonfinite optional MANO parameters: {key}")
                row[key] = data[key].tolist()
        for suffix, key in (("positions", "contact_position"), ("normals", "contact_normal"), ("part_ids", "contact_body")):
            row[f"mano_{side}_object_contact_{suffix}"] = data[key][:, side_index].tolist()
    return row


def export(request):
    authorize("export", request)
    scene = _scene(request)
    contacts = producer(request.contact_run, "hand_motion.contacts", request) if request.contact_run else None
    uncertainty = producer(request.uncertainty_run, None, request) if request.uncertainty_run else None
    extras = [parent for parent in (contacts, uncertainty) if parent]
    if extras and any(parent[1].get("provenance") != "video_only" for parent in [scene, *extras]):
        raise ValueError("Research contact/uncertainty transfer requires video-only evidence")
    with operation_run("export", request, [scene, *extras]) as (directory, record):
        document, pts = _identity(scene, request, record)
        contact_names = _contact_names(document, record)
        data = _scene_arrays(scene, record, pts, contact_names)
        if contacts:
            _admit_contacts(contacts, scene, data, request, pts, directory, record)
        if uncertainty:
            _propose_randomization(uncertainty, request, pts, directory, record)
        row = _motion_row(data, request, scene, directory, contact_names, record)
        path = partition(directory, "loaded", request) / "motion.parquet"
        _write_row(row, path, record)
        record.update(loaded_parquet=path.relative_to(directory).as_posix(),
                      mano_parameters="Optional model parameters pass through when present; absent parameters remain empty, never inferred from joints")
    return directory / "run.json"


def _clean_mesh(path):
    """Load a finite triangle mesh, merge duplicate vertices and drop degenerate faces."""
    import numpy as np
    import trimesh
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Object asset is not a finite triangle mesh")
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    faces = inverse[mesh.faces]
    triangles = vertices[faces]
    keep = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > 2e-12
    return trimesh.Trimesh(vertices=vertices, faces=faces[keep], process=False)


def _write_urdf(path, name, mass, center, inertia):
    robot = ET.Element("robot", name=name)
    ET.SubElement(ET.SubElement(robot, "mujoco"), "compiler", balanceinertia="true", discardvisual="false")
    link = ET.SubElement(robot, "link", name="object")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "inertia", **{INERTIA_KEYS[i][j]: str(inertia[i, j]) for i in range(3) for j in range(i, 3)})
    for role, filename in (("visual", f"{name}.obj"), ("collision", f"{name}_collision.obj")):
        element = ET.SubElement(link, role)
        ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(element, "geometry"), "mesh", filename=filename, scale="1 1 1")
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def assets(request):
    authorize("assets", request)
    parent = producer(request.input_run, "retargeting.export", request)
    scene = _scene(request, parent)
    with operation_run("assets", request, [parent, scene]) as (directory, record):
        import numpy as np

        document, _ = _identity(scene, request, record)
        _copy_bundle(parent, directory, request, record)
        destination = bundle_root(directory) / "processed"
        destination.mkdir(parents=True, exist_ok=True)
        radii, metadata = [], []
        for item in document["objects"]:
            name = item["name"]
            meshes = []
            for field in ("file", "collision_file"):
                relative = local_path(item.get(field, item["file"])).relative_to(scene[0].parent).as_posix()
                meshes.append(_clean_mesh(artifact(scene, relative, record)))
            visual, collision = meshes
            if not collision.is_watertight or not collision.is_winding_consistent or collision.volume <= 0:
                raise ValueError("Collision mass properties require closed outward geometry")
            mass = float(item["mass_kg"])
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("A positive scene mass is required")
            collision.density = mass / collision.volume
            inertia, center = collision.moment_inertia, collision.center_mass
            if not np.isfinite(inertia).all() or (np.linalg.eigvalsh(inertia) <= 0).any():
                raise ValueError("Invalid mesh-derived inertia")
            visual.export(destination / f"{name}.obj", file_type="obj", include_texture=False)
            collision.export(destination / f"{name}_collision.obj", file_type="obj", include_texture=False)
            _write_urdf(destination / f"{name}_rigid.urdf", name, mass, center, inertia)
            radii.append(float(np.linalg.norm(visual.vertices, axis=1).max()))
            metadata.append({"name": name, "mass_kg": mass, "center_mass": center.tolist(), "inertia": inertia.tolist()})
        path = directory / record["loaded_parquet"]
        row = _read_row(path)
        row["object_mesh_radius"] = radii
        _write_row(row, path, record)
        write_json(directory / "asset-mass.json", metadata)
    return directory / "run.json"


def retarget(request):
    authorize("retarget", request)
    parent = producer(request.input_run, "retargeting.assets", request)
    scene = _scene(request, parent)
    with operation_run("retarget", request, [parent, scene]) as (directory, record):
        _identity(scene, request, record)
        _copy_bundle(parent, directory, request, record)
        root = bundle_root(directory)
        arguments = ["--input_dir", root / "loaded", "--output_dir", root / "processed", "--save",
                     "--device", "cpu", "--sequence_id", f"episode_{request.episode_index:06d}"]
        for key in ("mano_to_robot_scale", "smooth_reference_window", "smooth_reference_polyorder", "surface_margin",
                    "surface_granularity", "adaptive_offset_alpha", "adaptive_offset_cap", "adaptive_offset_smooth_window", "hand_object_offset"):
            arguments.extend(["--" + key, getattr(request, key)])
        for key in ("smooth_reference", "surface_project", "adaptive_offset"):
            if getattr(request, key):
                arguments.append("--" + key)
        loaded = directory / record["loaded_parquet"]
        row = _read_row(loaded)
        row["mano_to_robot_scale"] = request.mano_to_robot_scale
        _write_row(row, loaded, record)
        run_upstream("scripts/retarget/ego_recon_to_sharpa.py", arguments, directory, record)
        files = list(partition(directory, "processed", request).glob("*.parquet"))
        if len(files) != 1:
            raise ValueError("Retargeter did not write exactly one processed motion")
        record["processed_parquet"] = files[0].relative_to(directory).as_posix()
    return directory / "run.json"


def _support_shapes(path):
    import numpy as np
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError("Could not open support stage")
    result = []
    for prim in stage.Traverse():
        if prim.GetTypeName() not in ("Cylinder", "Cube"):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        matrix = np.array([[transform[i][j] for j in range(4)] for i in range(4)], dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Support shape has a nonfinite transform")
        if (np.abs(matrix[:3, :3] - np.eye(3)) > 1e-8).any():
            raise ValueError("Support shapes must have translation only")
        if prim.GetTypeName() == "Cylinder" and prim.GetAttribute("axis").Get() not in (None, "Z"):
            raise ValueError("Support cylinders must use the Z axis")
        translate = prim.GetAttribute("xformOp:translate").Get()
        position = [0.0, 0.0, 0.0] if translate is None else list(map(float, translate))
        if len(position) != 3 or (np.abs(matrix[3, :3] - position) > 1e-8).any():
            raise ValueError("Inherited support translations are ignored by upstream and are refused")
        if prim.GetTypeName() == "Cylinder":
            dimensions = [float(prim.GetAttribute(key).Get()) for key in ("radius", "height")]
        else:
            size = prim.GetAttribute("size").Get()
            dimensions = [float(size if size is not None else 2.0)]
        if not all(math.isfinite(value) and value > 0 for value in dimensions):
            raise ValueError("Support dimensions must be finite and positive")
        result.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(), "position": position, "dimensions": dimensions})
    return result


def support(request):
    authorize("support", request)
    parent = producer(request.input_run, "retargeting.retarget", request)
    scene = _scene(request, parent)
    with operation_run("support", request, [parent, scene]) as (directory, record):
        _identity(scene, request, record)
        _copy_bundle(parent, directory, request, record)
        path = bundle_root(directory) / "reconstructed_stage" / f"episode_{request.episode_index:06d}_support.usda"
        path.parent.mkdir(parents=True)
        run_upstream("scripts/reconstruct_support_surfaces.py", ["--input_dir", bundle_root(directory) / "processed",
                     "--dataset", "ego_recon", "--sequence_id", f"episode_{request.episode_index:06d}",
                     "--robot_name", "sharpa_wave", "--ground_threshold", "0.05", "--output", path], directory, record)
        shapes = _support_shapes(path)
        record.update(support_file=path.relative_to(directory).as_posix(), scene_support=bool(shapes))
        write_json(directory / "support.json", {"shapes": shapes, "scene.support": bool(shapes)})
    return directory / "run.json"


def _check(findings, name, passed, detail=None, blocking=True):
    findings.append({"check": name, "passed": bool(passed), "blocking": bool(blocking and not passed), "detail": detail})


def _inspection_shapes(count, objects):
    shapes = {"object_body_position": (count, objects, 3), "object_body_wxyz": (count, objects, 4),
              "object_root_position": (count, 3), "object_root_axis_angle": (count, 3), "object_articulation": (count,)}
    for side in ("left", "right"):
        shapes.update({f"mano_{side}_trans": (count, 3), f"mano_{side}_joints": (count, 21, 3),
                       f"mano_{side}_joints_wxyz": (count, 21, 4),
                       f"mano_{side}_object_contact_positions": (count, 16, 3),
                       f"mano_{side}_object_contact_normals": (count, 16, 3),
                       f"mano_{side}_object_contact_part_ids": (count, 16),
                       f"robot_{side}_wrist_position": (count, 3), f"robot_{side}_wrist_wxyz": (count, 4),
                       f"robot_{side}_finger_joints": (count, 22), f"robot_{side}_frames": (count, 67, 7),
                       f"robot_{side}_frame_task_errors": (count, 11),
                       f"robot_{side}_num_optimization_iterations": (count,)})
    return shapes


def _check_arrays(findings, row, count, names, record):
    """Shape/finiteness of every numeric field; returns the arrays that passed."""
    import numpy as np
    values = {}
    for key, expected in _inspection_shapes(count, len(names)).items():
        try:
            value = np.asarray(row.get(key), dtype=np.float64)
            valid = value.shape == expected and np.isfinite(value).all()
            _check(findings, key, valid, {"expected_shape": expected, "actual_shape": value.shape})
            if valid:
                values[key] = value
                if key.endswith("wxyz"):
                    _check(findings, key + "_unit", np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-4, rtol=0))
        except (ValueError, TypeError) as error:
            _check(findings, key, False, str(error))
    for key, _, _, temporal in _schema(record)._field_specs:
        value = row.get(key)
        if temporal and value:
            try:
                _check(findings, key + "_frame_count", len(value) == count and np.isfinite(np.asarray(value, dtype=float)).all())
            except (ValueError, TypeError) as error:
                _check(findings, key + "_frame_count", False, str(error))
    radii = np.asarray(row.get("object_mesh_radius", []), dtype=float)
    _check(findings, "mesh_radii", radii.shape == (len(names),) and np.isfinite(radii).all() and (radii > 0).all())
    if "object_articulation" in values:
        _check(findings, "rigid_articulation", (values["object_articulation"] == 0).all())
    return values


def _admit_urdf(path, parent, record):
    """One free-root rigid link with positive mass/inertia and hashed visual/collision meshes."""
    import numpy as np
    robot = ET.parse(path).getroot()
    links = robot.findall("link")
    if len(links) != 1 or robot.findall("joint"):
        raise ValueError("Expected one free-root rigid link and no internal joints")
    link = links[0]
    mass = float(link.find("inertial/mass").attrib["value"])
    inertia = link.find("inertial/inertia").attrib
    tensor = np.array([[float(inertia[key]) for key in keys] for keys in INERTIA_KEYS])
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(tensor).all() or (np.linalg.eigvalsh(tensor) <= 0).any():
        raise ValueError("Mass and inertia must be finite and positive")
    if not link.findall("collision/geometry/mesh"):
        raise ValueError("A collision mesh is required")
    for node in link.findall(".//geometry/mesh"):
        mesh = (path.parent / node.attrib["filename"]).resolve()
        artifact(parent, mesh.relative_to(parent[0].parent).as_posix(), record)


def _check_assets(findings, row, names, parent, record):
    for field in ("object_mesh_paths", "object_urdf_paths"):
        paths = row.get(field, [])
        _check(findings, field + "_count", len(paths) == len(names))
        for value in paths:
            try:
                relative = local_path(value).relative_to(parent[0].parent).as_posix()
                path = artifact(parent, relative, record)
                _check(findings, field + "_exists", path.is_file() and not path.stem.endswith("_cm"), value)
                if field == "object_urdf_paths":
                    _admit_urdf(path, parent, record)
                    _check(findings, "rigid_urdf", True, value)
            except (ValueError, OSError, KeyError, AttributeError, TypeError, ET.ParseError) as error:
                _check(findings, field + "_asset", False, f"{value}: {error}")


def _check_robot(findings, row, values, names, request):
    """Robot name tables, frame quaternions and contact slots; returns IK errors above the threshold."""
    import numpy as np
    errors = []
    for side in ("left", "right"):
        for suffix, expected in (("finger_joint_names", 22), ("frame_names", 67), ("frame_task_names", 11)):
            items = row.get(f"{side}_robot_{suffix}", [])
            _check(findings, f"{side}_{suffix}", len(items) == expected and len(set(items)) == expected and all(items))
        key = f"robot_{side}_frames"
        if key in values:
            _check(findings, key + "_quaternions", np.allclose(np.linalg.norm(values[key][..., 3:], axis=-1), 1, atol=1e-4, rtol=0))
        ids = values.get(f"mano_{side}_object_contact_part_ids")
        normals = values.get(f"mano_{side}_object_contact_normals")
        positions = values.get(f"mano_{side}_object_contact_positions")
        if ids is not None:
            _check(findings, side + "_contact_ids", ((ids == np.floor(ids)) & (ids >= 0) & (ids <= len(names))).all())
            if normals is not None:
                _check(findings, side + "_contact_normals", np.allclose(np.linalg.norm(normals[ids > 0], axis=-1), 1, atol=1e-4, rtol=0))
                _check(findings, side + "_absent_contact_normals", (normals[ids == 0] == 0).all())
            if positions is not None:
                _check(findings, side + "_absent_contact_positions", (positions[ids == 0] == 0).all())
        key = f"robot_{side}_frame_task_errors"
        if key in values:
            task_names = row.get(f"{side}_robot_frame_task_names", [])
            for frame, task in np.argwhere(values[key] > request.ik_position_error_m):
                errors.append({"side": side, "frame": int(frame), "task": task_names[task] if len(task_names) > task else int(task),
                               "position_error_m": float(values[key][frame, task])})
    return errors


def inspect(request):
    authorize("inspect", request)
    parent = producer(request.input_run, "retargeting.support", request)
    scene = _scene(request, parent)
    with operation_run("inspect", request, [parent, scene]) as (directory, record):
        document, _ = _identity(scene, request, record)
        contact_names = _contact_names(document, record)
        findings = []
        source = artifact(parent, parent[1]["processed_parquet"], record)
        _check(findings, "one_partition_file", list(source.parent.glob("*.parquet")) == [source])
        row = _read_row(source)
        count, names = record["frame_count"], record["object_names"]
        _check(findings, "one_row", True)
        _check(findings, "partition_identity", row.get("sequence_id") == f"episode_{request.episode_index:06d}" and row.get("robot_name") == "sharpa_wave")
        _check(findings, "fps", row.get("fps") == 20.0)
        _check(findings, "object_order", row.get("object_body_names") == names)
        _check(findings, "mano_contact_link_order", row.get("mano_link_names") == contact_names)
        values = _check_arrays(findings, row, count, names, record)
        _check_assets(findings, row, names, parent, record)
        errors = _check_robot(findings, row, values, names, request)
        _check(findings, "ik_position_errors", not errors, errors, blocking=False)
        support_path = artifact(parent, parent[1]["support_file"], record)
        expected = bundle_root(parent[0].parent) / "reconstructed_stage" / f"episode_{request.episode_index:06d}_support.usda"
        _check(findings, "support_search_path", support_path == expected)
        try:
            shapes = _support_shapes(support_path)
            scene_support = bool(shapes)
            _check(findings, "support_shapes", True, {"shapes": shapes, "scene.support": scene_support})
            _check(findings, "support_record", parent[1]["scene_support"] == scene_support)
        except (ValueError, RuntimeError) as error:
            scene_support = False
            _check(findings, "support_shapes", False, str(error))
        report = {"findings": findings, "blocking": any(item["blocking"] for item in findings),
                  "frame_count": count, "fps": 20.0, "object_names": names, "scene.support": scene_support,
                  "scene_run": record["scene_run"], "scene_mode": record["scene_mode"],
                  "processed_parquet": input_file(ROOT, source), "support_file": input_file(ROOT, support_path),
                  "bundle_run": input_file(ROOT, parent[0]), "ik_position_errors": errors,
                  "provenance": record["provenance"]}
        report["research_sidecars"] = {name: input_file(ROOT, artifact(parent, name, record))
                                      for name in parent[1].get("research_sidecars", [])}
        write_json(directory / "inspect.json", report)
        record.update(blocking=report["blocking"], scene_support=scene_support, processed_parquet=linux_path(ROOT, source))
    return directory / "run.json"
