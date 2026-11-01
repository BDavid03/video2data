"""Official policy evaluation, strict trajectory inspection and pinned self-scores."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from track3.gpu import GpuLease
from track3.provenance import input_file, sha256_file, write_json
from ._common import (LOCK, ROOT, KIT, artifact, artifact_pin, authorize, checked_entry, kit_file, linux_path,
                      local_path, operation_run, producer, read_json)

POSE_SUFFIXES = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")


def _clock(request, record):
    if request.source_split == "evaluation":
        names = read_json(kit_file("data/track_3_evaluation_objects.json", record))["episodes"][str(request.episode_index)]
        path = kit_file("data/track_3_sample_submission.csv", record)
        frames = set()
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                match = re.fullmatch(r"track_3/e(\d+)/w000/f(\d+)/o(\d+)", row["row_id"])
                if match and int(match[1]) == request.episode_index:
                    frames.add(int(match[2]))
        if not frames or frames != set(range(max(frames) + 1)):
            raise ValueError("Invalid episode roster clock")
        return names, len(frames)
    for name in ("videos", "trajectories"):
        record["inputs"].append(input_file(ROOT, f"reports/data-audit/{name}.json"))
    inventory = read_json("reports/data-audit/trajectories.json")
    videos = read_json("reports/data-audit/videos.json")
    objects = [row for row in inventory["episodes"] if row["episode_index"] == request.episode_index]
    clips = [row for row in videos["videos"] if row["split"] == "public" and row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if (len(objects) != 1 or len(clips) != 1 or videos["status"] != "completed" or clips[0]["status"] != "ok" or
            any(item["revision"] != record["dataset_revision"] for item in (inventory, videos)) or
            clips[0]["time_base_and_frame_rate"] != [["1/10240", "20/1"]]):
        raise ValueError("Missing pinned public video clock and object order")
    return objects[0]["objects"], clips[0]["decoded_frame_count"]


def _checkpoint(training, value, record):
    if value is None:
        raise ValueError("Name the exact checkpoint")
    path = local_path(value)
    matches = [item for item in training[1]["checkpoints"] if training[0].parent / item["file"] == path]
    if len(matches) != 1:
        raise ValueError("Checkpoint is not owned by the named training run")
    pin = artifact_pin(training, matches[0]["file"], record)
    if pin["sha256"] != matches[0]["sha256"]:
        raise ValueError("Checkpoint changed")
    from track3.features.policy_training.checkpoints import admit_checkpoint

    record["sources"].append(input_file(ROOT, "track3/features/policy_training/checkpoints.py"))
    expected = read_json(artifact(training, "resolved-config.json", record))
    budget = read_json(artifact(training, "budget.json", record))
    reference = checked_entry(training[1]["reference"], record)
    admitted = admit_checkpoint(path, root=ROOT, expected_config=expected, reference=reference,
                                frame_count=training[1]["frame_count"], budget=budget,
                                budget_name=training[1]["request"]["budget"], policy_kind=training[1]["policy_kind"])
    if any(matches[0].get(key) != value for key, value in admitted.items()):
        raise ValueError("Checkpoint metadata differs from its owning training inventory")
    record["checkpoint_admission"] = admitted
    record.update(checkpoint=pin, training_run=input_file(ROOT, training[0]),
                  policy_kind=training[1]["policy_kind"], reference=training[1]["reference"])
    return path, matches[0]


def _runtime(config, record):
    installation = read_json(config["installation_record"])
    directory = Path(config["checkout"]) / "robotic_grounding/flash_chord/src/flash_chord/evaluation"
    digest = hashlib.sha256()
    lock = read_json(LOCK)
    for name in sorted(("__init__.py", "harness.py", "metrics.py", "rollout.py")):
        path = directory / name
        reference = local_path(lock["policy_runtime"]["path"]) / "robotic_grounding/flash_chord/src/flash_chord/evaluation" / name
        content = path.read_bytes()
        if content != reference.read_bytes().replace(b"\r\n", b"\n") or b"\r" in content:
            raise ValueError("Installed evaluator differs from the pinned LF source")
        digest.update(name.encode()); digest.update(b"\0"); digest.update(content)
        record["sources"].append(input_file(ROOT, reference))
    if digest.hexdigest() != installation["eval_code_sha256"]:
        raise ValueError("Installed evaluator changed since provisioning")
    manifest = read_json(KIT / "KIT_MANIFEST.json")
    for name in manifest["files"]:
        reference = kit_file(name, record)
        if (Path(config["kit"]) / name).read_bytes() != reference.read_bytes():
            raise ValueError("Installed kit differs from its manifest")
    record["eval_code_sha256"] = digest.hexdigest()
    return Path(config["kit"]) / "flash_chord/scripts/evaluate_policy.py"


def rollout(request):
    config = authorize("rollout", request)
    training = producer(request.training_run, "policy_training.train", request)
    resting = producer(request.resting_inspection, "policy_rollout.inspect", request) if request.resting_inspection else None
    if training[1]["policy_kind"] != "resting" and resting is None:
        raise ValueError("Produce and inspect the resting fallback first")
    with operation_run("rollout", request, [training] + ([resting] if resting else [])) as (directory, record):
        checkpoint, selected = _checkpoint(training, request.checkpoint, record)
        if resting:
            report = read_json(artifact(resting, "inspect.json", record))
            if report["blocking"] or report["policy_kind"] != "resting":
                raise ValueError("The fallback is not an inspected valid resting rollout")
        record["fallback_ladder"] = {
            "selected_checkpoint": record["checkpoint"],
            "earlier_checkpoints": [artifact_pin(training, item["file"], record)
                                    for item in reversed(training[1]["checkpoints"])
                                    if item["environment_steps"] < selected["environment_steps"]],
            "resting_inspection": input_file(ROOT, resting[0]) if resting else None,
            "execution": "Each retry requires a separately authorized rollout request; no automatic launch"}
        script = _runtime(config, record)
        names, count = _clock(request, record)
        overrides = {"evaluation.checkpoint": linux_path(ROOT, checkpoint), "evaluation.world_count": 1,
                     "evaluation.reset_mode": "explicit", "evaluation.start_frame": 0,
                     "logging.mode": "disabled", "hydra.run.dir": linux_path(ROOT, directory / "hydra"),
                     "evaluation.object_trajectories_output": linux_path(ROOT, directory / f"episode_{request.episode_index:06d}.parquet"),
                     "evaluation.metrics_output": linux_path(ROOT, directory / f"episode_{request.episode_index:06d}.json")}
        if request.source_root:
            overrides["evaluation.source_root"] = linux_path(ROOT, request.source_root)
        arguments = [sys.executable, str(script)] + [key + "=" + json.dumps(value) for key, value in overrides.items()]
        write_json(directory / "overrides.json", overrides)
        record.update(command=arguments, object_names=names, frame_count=count,
                      eval_script_sha256=sha256_file(script), metrics_scope="Tracking against reconstructed reference, not leaderboard scores")
        environment = {**os.environ, "FLASH_CHORD_CACHE_DIR": str(ROOT / ".cache/flash-chord"), "WANDB_MODE": "disabled",
                       "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(ROOT / ".cache/policy-tmp"),
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        lease = GpuLease(ROOT, request.headroom_mib, request.admission_allowance_mib, stage="policy_rollout.rollout")
        try:
            with lease, (directory / "evaluation.log").open("x", encoding="utf-8") as stream:
                subprocess.run(arguments, cwd=Path(config["checkout"]) / "robotic_grounding/flash_chord",
                               env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                lease.require_headroom()
        finally:
            record["gpu"] = lease.report()
        import pyarrow.parquet as pq
        path = directory / f"episode_{request.episode_index:06d}.parquet"
        if not all(pq.ParquetFile(path).read(columns=["environment_valid"]).column(0).to_pylist()):
            record["failure_reason"] = "invalid_world"
            raise ValueError("invalid_world: use the next recorded fallback rung")
        record["tracking_metrics"] = read_json(directory / f"episode_{request.episode_index:06d}.json")
    return directory / "run.json"


def _inspect_file(path, request, source, record):
    import numpy as np
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path).read()
    metadata = {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()}
    rows = table.to_pylist()
    names, count = _clock(request, record)
    checks = []  # (name, passed) in report order
    expected = {"schema": "flash_chord_object_trajectories_v1", "position_unit": "meter", "quaternion_order": "xyzw",
                "row_order": "step,environment_id,object_body_id", "episode_index": str(request.episode_index),
                "step_count": str(count), "world_count": "1", "object_body_count": str(len(names)), "pose_width": "7", "start_frame": "0"}
    for key, value in expected.items():
        checks.append(("metadata." + key, metadata.get("flash_chord." + key) == value))
    try:
        fps = float(metadata.get("flash_chord.control_fps", "nan"))
    except ValueError:
        fps = float("nan")
    checks.append(("control_fps", fps == 20.0))
    required = {"environment_id", "step", "reference_frame", "time_s", "object_id", "object_body_id", "object_body_name", "environment_valid"}
    required.update(prefix + suffix for prefix in ("achieved_", "reference_") for suffix in POSE_SUFFIXES)
    checks.append(("required_columns", required <= set(table.column_names)))
    if required <= set(table.column_names):
        keys = [(row["step"], row["environment_id"], row["object_body_id"]) for row in rows]
        checks += [
            ("complete_unique_grid", len(keys) == count * len(names) and
             set(keys) == {(t, 0, b) for t in range(count) for b in range(len(names))}),
            ("official_body_order", all(type(row["object_body_id"]) is int and 0 <= row["object_body_id"] < len(names) and
             row["object_body_name"] == names[row["object_body_id"]] and row["object_id"] == row["object_body_id"] for row in rows)),
            ("reference_clock", all(row["reference_frame"] == row["step"] and isinstance(row["time_s"], (int, float)) and
             abs(row["time_s"] - row["step"] / 20.0) < 1e-7 for row in rows)),
            ("every_world_valid", bool(rows) and all(row["environment_valid"] is True for row in rows))]
        for prefix in ("achieved_", "reference_"):
            poses = np.asarray([[row[prefix + suffix] for suffix in POSE_SUFFIXES] for row in rows], dtype=float).reshape(-1, 7)
            checks += [(prefix + "finite_metres", bool(rows) and np.isfinite(poses).all() and (np.abs(poses[:, :3]) <= 5).all()),
                       (prefix + "unit_quaternion", bool(rows) and np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4))]
    config = read_json("configs/runtime/policy-env.json")
    installed = read_json(config["installation_record"])
    allowlists = read_json(kit_file("config/tracks.json", record))
    expected_hashes = {"checkpoint_sha256": source["checkpoint"]["sha256"], "eval_code_sha256": installed["eval_code_sha256"],
                       "eval_script_sha256": sha256_file(kit_file("flash_chord/scripts/evaluate_policy.py", record))}
    for key, value in expected_hashes.items():
        actual = metadata.get("flash_chord." + key, "")
        checks.append((key, bool(re.fullmatch("[0-9a-f]{64}", actual)) and actual == value))
        accepted = allowlists.get("accepted_" + key, [])
        if accepted:
            checks.append((key + "_allowlist", actual in accepted))
    try:
        reference_matches = local_path(metadata["flash_chord.reference_parquet"]) == local_path(source["reference"]["path"])
    except (KeyError, ValueError):
        reference_matches = False
    checks.append(("reference_lineage", reference_matches))
    findings = [{"check": name, "blocking": not bool(passed)} for name, passed in checks]
    return {"findings": findings, "blocking": any(item["blocking"] for item in findings),
            "frame_count": count, "object_names": names, "metadata": metadata}


def inspect(request):
    authorize("inspect", request)
    parent = producer(request.rollout_run, "policy_rollout.rollout", request)
    training = producer(parent[1]["training_run"]["path"], "policy_training.train", request)
    with operation_run("inspect", request, [parent, training]) as (directory, record):
        checked_entry(parent[1]["training_run"], record)
        _checkpoint(training, parent[1]["checkpoint"]["path"], record)
        if record["checkpoint"] != parent[1]["checkpoint"] or record["reference"] != parent[1]["reference"]:
            raise ValueError("Rollout training lineage changed")
        pose_file = artifact_pin(parent, f"episode_{request.episode_index:06d}.parquet", record)
        report = _inspect_file(local_path(pose_file["path"]), request, parent[1], record)
        report.update(rollout_run=input_file(ROOT, parent[0]), pose_file=pose_file,
                      training_run=record["training_run"], checkpoint=record["checkpoint"], policy_kind=record["policy_kind"],
                      tracking_metrics=read_json(artifact(parent, f"episode_{request.episode_index:06d}.json", record)),
                      metrics_scope="Tracking against reconstructed reference, not leaderboard scores")
        write_json(directory / "inspect.json", report)
        record.update(blocking=report["blocking"], rollout_run=report["rollout_run"], fallback_ladder=parent[1]["fallback_ladder"])
        if report["blocking"]:
            raise ValueError("Rollout inspection failed; see inspect.json and the recorded fallback ladder")
    return directory / "run.json"


def _module(name, record):
    spec = importlib.util.spec_from_file_location("_track3_rollout_" + name, kit_file("v2dlb/" + name + ".py", record))
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _poses(matrices):
    """Rigid 4x4 transforms -> [x, y, z, qx, qy, qz, qw] via the symmetric-K eigenvector method."""
    import numpy as np
    r = matrices[..., :3, :3]
    if (not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5).all()):
        raise ValueError("Scene contains a non-rigid pose")
    rows = [[r[..., 0, 0]-r[..., 1, 1]-r[..., 2, 2], r[..., 1, 0]+r[..., 0, 1], r[..., 2, 0]+r[..., 0, 2], r[..., 2, 1]-r[..., 1, 2]],
            [r[..., 1, 0]+r[..., 0, 1], r[..., 1, 1]-r[..., 0, 0]-r[..., 2, 2], r[..., 2, 1]+r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0]],
            [r[..., 2, 0]+r[..., 0, 2], r[..., 2, 1]+r[..., 1, 2], r[..., 2, 2]-r[..., 0, 0]-r[..., 1, 1], r[..., 1, 0]-r[..., 0, 1]],
            [r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1], np.trace(r, axis1=-2, axis2=-1)]]
    k = np.stack([np.stack(row, axis=-1) for row in rows], axis=-2) / 3
    result = np.empty(matrices.shape[:-2] + (7,))
    result[..., :3] = matrices[..., :3, 3]
    result[..., 3:] = np.linalg.eigh(k)[1][..., :, -1]
    return result


def _finite_or_none(values):
    import numpy as np
    return {name: value if np.isfinite(value) else None for name, value in values.items()}


def _scene_reference(scene, record, count, names):
    """Admit the pinned tracked scene trajectory; return reference poses and fill counts."""
    import numpy as np
    with np.load(artifact(scene, "trajectories.npz", record), allow_pickle=False) as data:
        transforms = data["world_T_object"]
        if (transforms.shape != (count, len(names), 4, 4) or not np.isfinite(transforms).all() or
                not np.array_equal(data["frame_index"], np.arange(count)) or data["object_names"].tolist() != names or
                data["object_filled"].shape != (count, len(names)) or
                not np.allclose(transforms[..., 3, :], [0, 0, 0, 1])):
            raise ValueError("Invalid pinned scene trajectory")
        return (_poses(transforms)[:, None], int(np.count_nonzero(data["object_filled"])),
                np.count_nonzero(data["object_filled"], axis=0).tolist())


def _per_object(metric, aligned, reference, points, names, count, filled_by_object):
    """Per-object metrics and per-frame MPPE on the once-aligned attempt.

    Realigning an event or frame would erase the accumulated error that event attribution is meant to show.
    """
    import numpy as np
    per_object, frame_mppe = {}, {}
    for index, name in enumerate(names):
        selected = slice(index, index + 1)
        ids, object_points = np.asarray([index]), points[selected]
        per_object[name] = _finite_or_none({
            key: float(metric.episode_metric(key, aligned[:, :, selected], reference[:, :, selected],
                                             ids, object_points, align_initial_pose=False))
            for key in ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")})
        per_object[name].update(rpe_cm=None, rpe_scope="Defined on object pairs; see episode value",
                                expected_poses=count, achieved_poses=count,
                                filled_reference_poses=int(filled_by_object[index]))
        frame_mppe[name] = [float(metric.episode_metric(
            "mppe_cm", aligned[t:t + 1, :, selected], reference[t:t + 1, :, selected],
            ids, object_points, align_initial_pose=False)) for t in range(count)]
    return per_object, frame_mppe


def self_score(request):
    authorize("self-score", request)
    inspection = producer(request.inspection_run, "policy_rollout.inspect", request)
    scene = producer(request.scene_run, "scene_assembly.assemble", request)
    with operation_run("self-score", request, [inspection, scene]) as (directory, record):
        import numpy as np
        import pyarrow.parquet as pq
        from track3.features.campaign.selections import read_pin_history, tracked_scene_pin

        pins, pin_log = read_pin_history(ROOT, "outputs/campaign/pins.jsonl")
        if pin_log is not None:
            record["inputs"].append(pin_log)
        record["sources"].append(input_file(ROOT, "track3/features/campaign/selections.py"))
        selected = tracked_scene_pin(pins, request.source_split, request.episode_index, read_json)
        scene_pin = input_file(ROOT, scene[0])
        if selected is None or {"path": selected["run"], "sha256": selected["sha256"]} != scene_pin:
            raise ValueError("Self-score requires the current pinned tracked scene")
        if scene[1]["request"].get("mode") != "full":
            raise ValueError("Tracked scene pin differs from its producer request")
        record["comparison_scene_pin"] = selected
        report = read_json(artifact(inspection, "inspect.json", record))
        if report["blocking"]:
            raise ValueError("Cannot self-score an invalid rollout")
        checked_entry(report["rollout_run"], record)
        checked_entry(report["checkpoint"], record)
        document = read_json(artifact(scene, "scene.json", record))
        names, count = _clock(request, record)
        if document["mode"] != "full" or document["frame_count"] != count or [row["name"] for row in document["objects"]] != names:
            raise ValueError("Self-score requires a tracked full scene on the same grid")
        reference, filled, filled_by_object = _scene_reference(scene, record, count, names)
        rows = pq.ParquetFile(checked_entry(report["pose_file"], record)).read().to_pylist()
        achieved = np.empty_like(reference)
        for row in rows:
            achieved[row["step"], 0, row["object_body_id"]] = [row["achieved_" + suffix] for suffix in POSE_SUFFIXES]
        vertices_module = _module("vertices", record)
        points = np.asarray([vertices_module.sample_vertices(
            checked_entry({"path": local_path(item["file"]).relative_to(ROOT).as_posix(), "sha256": item["sha256"]}, record),
            count=500, seed=0) for item in document["objects"]])
        metric = _module("object_pose", record)
        values = _finite_or_none({
            name: float(metric.episode_metric(name, achieved, reference, np.arange(len(names)), points, align_initial_pose=True))
            for name in ("add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm")})
        aligned = metric._warp_to_reference_start(achieved, reference)
        per_object, frame_mppe = _per_object(metric, aligned, reference, points, names, count, filled_by_object)
        extent = np.asarray(document["world"]["support_plane"]["extent_xy_m"], dtype=float)
        if extent.shape != (2, 2) or not np.isfinite(extent).all() or np.any(extent[0] >= extent[1]):
            raise ValueError("Invalid table inlier extent")
        final = aligned[-1, 0, :, :3]
        # Origins may sit at the object base; allow 2 cm of settling below the table.
        outside = bool(np.any(final[:, :2] < extent[0]) or np.any(final[:, :2] > extent[1]) or np.any(final[:, 2] < -0.02))
        write_json(directory / "self_score.json", {"inspection_run": input_file(ROOT, inspection[0]),
            "rollout_run": report["rollout_run"], "scene_run": scene_pin, "policy_kind": report["policy_kind"],
            "metrics": values, "left_working_area": outside, "align_initial_pose": True,
            "per_object": per_object, "frame_mppe_cm": frame_mppe,
            "coverage": {"expected_poses": count * len(names), "achieved_poses": count * len(names),
                         "frame_count": count, "object_names": names},
            "filled_reference_poses": filled, "scope": "Agreement with pinned reconstruction; not a leaderboard estimate"})
    return directory / "run.json"
