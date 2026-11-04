"""Public campaign selection metadata contract; no operation or runtime imports."""

import hashlib
import json
import re

from track3.paths import project_path


def _valid_pin(entry):
    return (isinstance(entry, dict) and entry.get("split") in ("public", "evaluation") and
            type(entry.get("episode_index")) is int and entry["episode_index"] >= 0 and entry["episode_index"] != 41 and
            all(isinstance(entry.get(key), str) and entry[key] for key in ("artifact_name", "run_id", "run")) and
            isinstance(entry.get("reason"), str) and entry["reason"].strip() != "" and
            isinstance(entry.get("sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is not None and
            ("role" not in entry or (isinstance(entry["role"], str) and entry["role"] != "")) and
            entry.get("tier") in (None, "resting", "tracked") and
            (entry.get("tier") is None or entry["artifact_name"] == "policy_rollout.inspect"))


def read_pin_history(root, path):
    """Parse and hash one byte snapshot of the append-only pin log; returns (entries, log identity)."""
    path = project_path(root, path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return [], None
    base = project_path(root, ".")
    history = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        try:
            entry = json.loads(line)
        except ValueError as error:
            raise ValueError(f"Malformed campaign pin at line {number}") from error
        if not _valid_pin(entry):
            raise ValueError(f"Malformed campaign pin at line {number}")
        run = project_path(root, entry["run"])
        relative = run.relative_to(base)
        if run.name != "run.json" or relative.parts[0] not in ("outputs", "runs"):
            raise ValueError(f"Campaign pin escapes producer storage at line {number}")
        history.append({**entry, "run": relative.as_posix()})
    return history, {"path": path.relative_to(base).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def tracked_scene_pin(history, split, episode, load_record):
    """A later resting-scene pin does not replace the selected full scene."""
    for entry in reversed(history):
        if (entry["split"], entry["episode_index"], entry["artifact_name"]) != (split, episode, "scene_assembly.assemble"):
            continue
        role = entry.get("role")
        if role is None:
            record = load_record(entry["run"])
            role = run_role(record, {}) if record else "unknown"
        if role not in ("resting", "tracked"):
            raise ValueError("Cannot classify a scene pin; restore its producer metadata before selecting a full scene")
        if role == "tracked":
            return entry
    return None


def _camera_role(record, records):
    for entry in record["inputs"]:
        parent = records.get(entry["path"], {})
        if parent.get("kind") == "camera_motion.estimate":
            return "tracked" if parent["request"].get("opening_frame_count", 20) is None else "resting"
    return "unknown"


def _parent_records(record, records):
    return [records.get(item["path"], {}) for item in record["inputs"]]


def run_role(record, records):
    """Separate repeat passes using producer evidence, including older records."""
    kind, request = record["kind"], record.get("request", {})
    if kind == "camera_motion.estimate":
        return "tracked" if request.get("opening_frame_count", 20) is None else "resting"
    if kind == "object_masks.opening_clusters":
        return _camera_role(record, records)
    if kind == "object_mesh.resolve":
        if request.get("scans_only") or record.get("pending_objects"):
            return "scans"
        for parent in _parent_records(record, records):
            if parent.get("kind") == "object_mesh.reconstruct":
                return "complete:" + _camera_role(parent, records)
        return "complete"
    if kind == "object_pose.initialise":
        return ("scans:" if record.get("pending_objects") else "complete:") + _camera_role(record, records)
    if kind == "object_mesh.reconstruct":
        parents = _parent_records(record, records)
        spoon = request.get("objects") == ["wooden_spoon"] or (not request.get("objects") and any(
            item.get("kind") == "object_pose.initialise" and item.get("pending_objects") for item in parents))
        return ("spoon:" if spoon else "submission:") + _camera_role(record, records)
    if kind == "object_pose.compose":
        parents = _parent_records(record, records)
        if any(item.get("kind") == "object_pose.track" for item in parents):
            return "tracked"
        if any(item.get("kind") == "object_pose.initialise" for item in parents):
            return "resting"
        return "unknown"
    if kind == "scene_assembly.assemble":
        return {"resting": "resting", "full": "tracked"}.get(request.get("mode"), "unknown")
    if kind.startswith("retargeting."):
        return {"resting": "resting", "full": "tracked"}.get(record.get("scene_mode"), "unknown")
    if kind.startswith("policy_"):
        policy = record.get("policy_kind", request.get("policy_kind"))
        if kind == "policy_rollout.self_score":
            parents = _parent_records(record, records)
            policy = next((item.get("policy_kind") for item in parents
                           if item.get("kind") == "policy_rollout.inspect"), None)
        return {"resting": "resting", "trained": "tracked", "replay": "replay"}.get(policy, "unknown")
    return "shared"


def role_pins(history, records):
    selected = {}
    for pin in history:
        source = records.get(pin["run"])
        role = pin.get("role") or (run_role(source, records) if source else "unknown")
        selected[(pin["split"], pin["episode_index"], pin["artifact_name"], role)] = pin
    return selected


def entry_pins(history):
    """Artifact selection without a tier does not erase an entry promotion."""
    selected = {}
    for pin in history:
        if pin["artifact_name"] == "policy_rollout.inspect" and pin.get("tier") in ("resting", "tracked"):
            selected[(pin["split"], pin["episode_index"], pin["artifact_name"])] = pin
    return selected


def inspection_tier(record):
    if record.get("kind") != "policy_rollout.inspect" or record.get("blocking") is not False:
        return None
    return {"resting": "resting", "trained": "tracked"}.get(record.get("policy_kind"))
