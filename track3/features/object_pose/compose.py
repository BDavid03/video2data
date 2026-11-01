"""Compose measured camera-relative poses with measured camera motion only."""

from track3.provenance import sha256_file, write_json
from ._common import (artifact, arrays, authorize, average, operation_run, producer,
                      read_json, require_input, require_rigid, save_npz)


def compose(request):
    authorize("compose", request)
    poses = producer(request.pose_run, None, request)
    if poses[1]["kind"] not in ("object_pose.initialise", "object_pose.track"):
        raise ValueError("Compose requires an initialisation or camera-relative tracking run")
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    require_input(poses, camera)
    if poses[1].get("pending_objects"):
        raise ValueError("Reconstruct the pending spoon and initialise again before composition")
    hand = producer(request.hand_run, None, request) if request.hand_run else None
    if hand:
        if hand[1]["kind"] not in ("hand_motion.lift", "hand_motion.refine"):
            raise ValueError("Composition requires admitted lifted or refined hand motion")
        if hand[1]["kind"] == "hand_motion.refine":
            baseline_hand = producer(hand[1]["request"]["hand_run"], "hand_motion.lift", request)
            require_input(hand, baseline_hand)
            require_input(baseline_hand, camera)
        require_input(hand, camera)
        if hand[1]["grid"] != camera[1]["grid"]:
            raise ValueError("Hand and camera runs use different colour pixel grids")
    parents = [poses, camera] + ([hand] if hand else [])
    mask_kind = "object_masks.generate" if poses[1]["kind"] == "object_pose.track" else "object_masks.opening_clusters"
    for field, kind in (("masks_run", mask_kind), ("depth_run", "depth.generate"),
                        ("anchor_run", "stereo_depth.anchor"), ("stereo_run", "stereo_depth.generate")):
        value = poses[1]["request"].get(field)
        if value:
            parent = producer(value, kind, request)
            require_input(poses, parent)
            parents.append(parent)
    if poses[1]["grid"] != camera[1]["grid"]:
        raise ValueError("Pose and camera runs use different colour pixel grids")
    with operation_run("compose", request, parents) as (directory, record):
        import numpy as np

        result = arrays(poses, "poses.npz", record, np)
        motion = arrays(camera, "trajectory.npz", record, np)
        if (not np.array_equal(result["frame_index"], motion["frame_index"]) or
                not np.array_equal(result["pts"], motion["pts"])):
            raise ValueError("Pose and camera frame-index/PTS grids differ")
        count, bodies = len(result["frame_index"]), len(result["object_names"])
        if (result["valid"].shape != (count, bodies) or result["camA_T_object"].shape != (count, bodies, 4, 4) or
                motion["valid"].shape != (count,) or motion["world_T_camA"].shape != (count, 4, 4)):
            raise ValueError("Invalid pose or camera array dimensions")
        valid = result["valid"].astype(bool)
        camera_valid = motion["valid"].astype(bool)
        require_rigid(result["camA_T_object"][valid], np)
        require_rigid(motion["world_T_camA"][camera_valid], np)
        result["camA_T_object"][~valid] = np.nan
        result["world_valid"] = valid & camera_valid[:, None]
        result["world_T_object"] = np.full((count, bodies, 4, 4), np.nan, dtype=np.float64)
        for index in range(count):
            selected = result["world_valid"][index]
            if selected.any():
                result["world_T_object"][index, selected] = (
                    motion["world_T_camA"][index] @ result["camA_T_object"][index, selected])
            record["frames"].append({"frame_index": int(result["frame_index"][index]),
                                     "video_pts": int(result["pts"][index]), "status": "running",
                                     "observed_objects": int(selected.sum())})
        opening_path = artifact(poses[0], "opening.json", record, poses[1]["artifacts"]["opening.json"])
        opening = read_json(opening_path)
        if [item["name"] for item in opening["objects"]] != result["object_names"].tolist():
            raise ValueError("Opening object order differs from the pose arrays")
        for slot, item in enumerate(opening["objects"]):
            indices = np.asarray(item["observed_frames"], dtype=np.int64)
            if ((indices < 0) | (indices >= count)).any():
                raise ValueError("Opening frame identity is outside the clip")
            indices = indices[result["world_valid"][indices, slot]]
            if not len(indices):
                raise ValueError(f"No composable opening pose for {item['name']}")
            mean, spread_m, spread_degrees = average(result["world_T_object"][indices, slot], np)
            item.update(world_T_object=mean.tolist(), observed_frames=indices.tolist(),
                        spread_m=spread_m, spread_degrees=spread_degrees)
        record.update(grid=poses[1]["grid"], object_names=result["object_names"].tolist(),
                      mesh_body_frames=poses[1]["mesh_body_frames"], interpolation="none")
        opening["provenance"] = record["provenance"]
        save_npz(directory / "poses.npz", np, **result)
        write_json(directory / "opening.json", opening)
        record["artifacts"] = {name: sha256_file(directory / name) for name in ("poses.npz", "opening.json")}
        for row in record["frames"]:
            row.update(status="completed", artifact="poses.npz", sha256=record["artifacts"]["poses.npz"])
    return directory / "run.json"
