"""Build a static task from admitted depth and mesh-extents producers."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from src.paths import output_path
from src.provenance import run_fields, write_json, sha256_file
from src.runs import (ROOT, artifact, producer, read_json, new_run_id, recorded, runtime_flag,
                         authorize_episode, heavy_execution)
from ._scene import parked_robot, camera_points, box_properties
from ._placement import place_boxes

FEATURE = Path(__file__).parent
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"
FC = ROOT / "data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord"
D1 = "data/outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff/run.json"
TEMPLATE = ROOT / "data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave"


def placement_v2(request):
    """Admit K1a only for the two physically accepted scenes in the lead decision."""
    if request.source_split != "evaluation" or request.episode_index not in (1, 28):
        return None
    config = read_json(FEATURE / "placements_v2.json")
    if config.get("enabled", True) is not True:
        return None
    if read_json(config["confirmation"])["verdict"] != "adopt":
        raise PermissionError("K1a scene construction requires an adopt confirmation verdict")
    return config["episodes"][str(request.episode_index)]


def episode_layout(request):
    if request.source_split == "evaluation":
        import csv
        names = read_json(KIT / "data/track_3_evaluation_objects.json")["episodes"][str(request.episode_index)]
        with (KIT / "data/track_3_sample_submission.csv").open() as stream:
            roster = [r["row_id"] for r in csv.DictReader(stream)]
        prefix = f"track_3/e{request.episode_index:06d}/"
        frames = sorted({int(s.split("/f")[1].split("/")[0]) for s in roster if s.startswith(prefix)})
        return names, frames
    # Public metadata exposes names and length; no object poses enter task construction.
    dataset = next(ROOT.glob("datasets--*/snapshots/*/track_3/public"))
    info = pq.read_table(dataset / f"data/chunk-000/episode_{request.episode_index:06d}.parquet", columns=["frame_index"])
    assignment = read_json("src/features/mesh_extents/assignments.json")["episodes"][f"public:{request.episode_index}"]
    # The two authorised tuning videos have one object each, named by their video assignments.
    return list(assignment["objects"]), list(range(info.num_rows))


def select_measure(request, d1, placements=()):
    if placements:
        paths = {p["measure_run"] for p in placements}
        if len(paths) != 1:
            raise ValueError("Placement overlays must use one consistent measure run per episode")
        return producer(paths.pop(), "mesh_extents.measure", request)
    for value in d1[1]["request"]["measure_runs"]:
        row = read_json(value)
        if (row["split"], row["episode_index"]) == (request.source_split, request.episode_index):
            return producer(value, "mesh_extents.measure", request)
    # Unassigned episodes were deliberately omitted from D1's pooling.
    folder = ROOT / f"data/outputs/mesh_extents/{request.source_split}/episode_{request.episode_index:06d}"
    candidates = [(p, read_json(p)) for p in sorted(folder.glob("*/run.json"))]
    candidates = [(p,r) for p,r in candidates if r["status"] == "completed" and r.get("frames")]
    if not candidates:
        raise ValueError("No completed mesh-extents measurement")
    return producer(candidates[-1][0], "mesh_extents.measure", request)


def build(request):
    runtime = runtime_flag(FEATURE, "build_authorized", "Insurance task build")
    authorize_episode(runtime, request, "insurance task build")
    heavy_execution(runtime)
    placement = placement_v2(request)
    d1 = producer(D1, "mesh_extents.build")
    placement_path = FEATURE / "placements.json"
    placements = [p for p in read_json(placement_path)["placements"]
                  if (p["source_split"], p["episode_index"]) == (request.source_split, request.episode_index)]
    measure = select_measure(request, d1, placements)
    if placement and measure[0] != ROOT / placement["measure_run"]:
        raise ValueError("K1a cluster ids must use their reviewed measure run")
    by_name = {p["slot_name"]:p for p in placements}
    assignment_path = "src/features/mesh_extents/assignments.json"
    assignment = read_json(assignment_path)["episodes"][f"{request.source_split}:{request.episode_index}"]
    depth = producer(measure[1]["request"]["depth_run"], "depth.generate", request)
    record = run_fields(ROOT, "pose_insurance", "build", asdict(request), parents=[p[0] for p in (d1, measure, depth)],
                        inputs=[assignment_path, placement_path, FEATURE / "runtime.json", KIT / "data/track_3_sample_submission.csv",
                                KIT / "data/track_3_evaluation_objects.json"], allowed=("video_only",))
    record.update(split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=depth[1]["dataset_revision"])
    version = "v2" if placement else "v1"
    if placement:
        from src.runs import record_input
        from src.provenance import input_file
        for path in (FEATURE / "placements_v2.json", "data/outputs/pose_insurance/placement/confirmation.json",
                     "docs/chain/pose-placement-protocol.md"):
            record_input(input_file(ROOT, path), record)
        record.update(placement_candidate="K1a", placement_confirmation="Amendment 2; one measurable episode")
    directory = output_path(ROOT, f"data/outputs/pose_insurance/build-{version}/{new_run_id()}/episode_{request.episode_index:06d}")
    with recorded(directory, record):
        objects = read_json(artifact(d1, "objects.json", record))["objects"]
        names, roster = episode_layout(request)
        count = max(roster) + 1
        first = measure[1]["frames"][0]
        labels = np.load(artifact(measure, f"labels_{first['frame_index']:06d}.npz", record))["label"]
        frames = {r["frame_index"]:r for r in depth[1]["frames"] if r["status"] == "completed"}
        frame = frames.get(0, frames[min(frames)])
        with np.load(artifact(depth[0], frame["artifact"], record, frame["sha256"])) as a:
            z = a["depth_m"].astype(float)
        with np.load(artifact(depth[0], "preprocessing.npz", record, depth[1]["preprocessing_artifact_sha256"])) as a:
            points = camera_points(z, a["input_intrinsics_pixels"])
        normal = np.asarray(first["plane"]["normal"], float)
        positions, quaternions, meshes, urdfs, radii, scene = [], [], [], [], [], []
        if placements:
            artifact(measure, "overlay_000000.png", record)
        assignment_note = assignment.get("note", "")
        slot_metadata, clusters, proportions, measurements = [], [], [], []
        for slot, name in enumerate(names):
            shape = objects[name]
            measured = measure[1].get("objects", {}).get(name, {}).get("median_m", shape["extents_m"])
            proportions.append(shape["extents_m"])
            measurements.append(measured)
            entry = by_name.get(name)
            ids = entry["cluster_ids"] if entry else assignment["objects"].get(name)
            confidence = entry["confidence"] if entry else (
                "uncertain" if any(word in assignment_note.lower() for word in ("arbitrary", "not known")) else "reviewed")
            note = entry["note"] if entry else "Reviewed mesh_extents assignment"
            if placement:
                note = placement["note"]
            if entry is None and (ids is None or first["frame_index"] != 0):
                raise ValueError(f"{name}: record an explicit placement before building this slot")
            clusters.append(points[np.isin(labels, [i+1 for i in ids])] if ids is not None else None)
            if ids is None:
                note += ("; K1a merged centroid and PCA yaw" if placement else
                         "; default 0.5 m slot spacing with identity orientation")
            slot_metadata.append((entry, ids, confidence, note))
        merged = points[np.isin(labels, [i+1 for i in placement["merged_cluster_ids"]])] if placement else None
        geometry = place_boxes(proportions, measurements, clusters, normal,
                               [entry is not None for entry, _, _, _ in slot_metadata],
                               merged=merged, k1="K1a" if placement else None,
                               stacked_slot=placement["stacked_slot"] if placement else 1)
        for slot, name in enumerate(names):
            shape = objects[name]
            entry, ids, confidence, note = slot_metadata[slot]
            dims, pos, quat = (geometry[key][slot] for key in ("dimensions", "before", "quaternions"))
            mesh = directory / f"{name}.obj"
            source_mesh = artifact(d1, shape["file"], record, shape["sha256"])
            obj = trimesh.load(source_mesh, force="mesh")
            obj.apply_scale(dims / np.asarray(shape["extents_m"]))
            obj.export(mesh)
            urdf = directory / f"{name}.urdf"
            mass = .1
            inertia, radius = box_properties(dims, mass)
            urdf.write_text(f'''<robot name="{name}">
  <link name="{name}">
    <inertial><mass value="{mass}"/>
      <inertia ixx="{inertia[0]}" iyy="{inertia[1]}" izz="{inertia[2]}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><mesh filename="{mesh}"/></geometry></visual>
    <collision><geometry><box size="{" ".join(map(str,dims))}"/></geometry></collision>
  </link>
</robot>
''')
            positions.append(pos.tolist())
            quaternions.append(quat.tolist())
            meshes.append(str(mesh))
            urdfs.append(str(urdf))
            radii.append(radius)
            scene.append({"slot":slot,"name":name,"dimensions_m":dims.tolist(),"position":pos.tolist(),
                          "quaternion_wxyz":quat.tolist(),"clusters":ids,"placement":note,
                          "position_before_clearance":pos.tolist(), "identity_confidence":confidence,
                          "assignment_note":assignment_note,
                          "placement_overlay":entry["overlay"] if entry else None,
                          "placement_source":"placements.json" if entry else "mesh_extents.assignments.json"})
            if placement:
                scene[-1].update(placement_source="placements_v2.json", merged_cluster_ids=placement["merged_cluster_ids"],
                                 placement_candidate="K1a", stacked_slot=placement["stacked_slot"])
        for row, new, shift in zip(scene, geometry["positions"], geometry["shifts"]):
            row["collision_displacement_cm"] = float(shift)
            row["position"] = new.tolist()
        positions = geometry["positions"].tolist()
        template_path = next(TEMPLATE.glob("*.parquet"))
        template = pq.read_table(template_path).to_pylist()[0]
        task = parked_robot(template, count)
        task.update(fps=20., object_name=f"episode_{request.episode_index:06d}", object_body_names=names,
                    object_body_position=np.repeat([positions], count, axis=0).tolist(),
                    object_body_wxyz=np.repeat([quaternions], count, axis=0).tolist(),
                    object_mesh_paths=meshes, object_urdf_paths=urdfs, object_mesh_radius=radii,
                    object_articulation=np.zeros(count).tolist(), sequence_id=f"episode_{request.episode_index:06d}",
                    robot_name="sharpa_wave")
        pq.write_table(pa.Table.from_pylist([task]), directory / "task.parquet")
        record.update(roster_frames=roster, task_frames=count, object_names=names,
                      template={"path":str(template_path),"sha256":sha256_file(template_path)},
                      scale="MoGe-2 monocular metric prediction, unanchored and uncalibrated; no absolute error bound",
                      scene=scene)
        write_json(directory / "initial_scene.json", {k:record[k] for k in ("scene","scale","object_names","task_frames")})
    return directory / "run.json"
