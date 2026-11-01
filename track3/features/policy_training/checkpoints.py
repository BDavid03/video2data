"""Public checkpoint-admission contract; standard-library metadata reads only."""

from copy import deepcopy
import hashlib
import json
import re

from track3.paths import project_path


def _constant(value):
    raise ValueError(f"Non-finite JSON value in checkpoint metadata: {value}")


def _json(value):
    result = json.loads(value, parse_constant=_constant)
    if not isinstance(result, dict):
        raise ValueError("Checkpoint metadata must contain JSON objects")
    return result


def read_policy_metadata(path):
    """Read a bounded safetensors header, without importing a tensor runtime."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        size = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < size < 64 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        header = stream.read(size)
        if len(header) != size:
            raise ValueError("Truncated safetensors header")
    raw = _json(header).get("__metadata__")
    if not isinstance(raw, dict) or any(not isinstance(value, str) for value in raw.values()):
        raise ValueError("Safetensors metadata must be a string mapping")
    if (raw.get("format") != "learner_state" or raw.get("version") != "1" or
            raw.get("user.algorithm") != "flash_sac" or raw.get("user.checkpoint_kind") != "policy"):
        raise ValueError("Expected a FlashCHORD actor policy checkpoint")
    return {key[5:]: value for key, value in raw.items() if key.startswith("user.")}


def _configuration(value, root):
    config = deepcopy(value)
    # train_flash_sac.py resolves this path before serializing the configuration.
    config["task"]["parquet"] = project_path(root, config["task"]["parquet"]).as_posix()
    return config


def require_same_composition(composed, training, root):
    """Only output_dir may differ between the two resolved job configurations."""
    previous, current = (_configuration(value, root) for value in (composed, training))
    previous.pop("output_dir", None)
    current.pop("output_dir", None)
    # Hydra's --cfg job output omits its own hydra.run.dir namespace.
    if previous != current:
        raise ValueError("Resolved training recipe changed after composition")


def _counter(metadata, key):
    value = metadata.get(key, "")
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"Invalid checkpoint counter: {key}")
    return int(value)


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def admit_checkpoint(path, *, root, expected_config, reference, frame_count, budget, budget_name, policy_kind):
    """Bind saved configuration/counters to the owning training request; tensors are not decoded."""
    metadata = read_policy_metadata(path)
    saved = _configuration(_json(metadata["resolved_config_json"]), root)
    expected = _configuration(expected_config, root)
    loaded_reference = saved.pop("reference", None)
    if saved != expected:
        raise ValueError("Saved policy configuration differs from the resolved training recipe")
    if (expected["task"]["motion_speed"] != 1.0 or expected["sim"]["fps"] != 20.0 or
            expected["task"]["source_frame_playback"] is not False or
            expected["task"]["motion_start_frame"] != 0 or expected["task"]["motion_end_frame"] != -1 or
            expected["logging"]["mode"] != "disabled" or
            project_path(root, expected["task"]["parquet"]) != project_path(root, reference)):
        raise ValueError("Policy configuration does not preserve the exact offline episode reference")
    if (not isinstance(loaded_reference, dict) or
            project_path(root, loaded_reference.get("parquet", "")) != project_path(root, reference) or
            type(loaded_reference.get("num_frames")) is not int or loaded_reference["num_frames"] != frame_count or
            loaded_reference.get("playback_fps") != 20.0 or
            loaded_reference.get("object_articulations") != {"version": 1, "entries": []}):
        raise ValueError("Checkpoint reference metadata differs from the complete rigid-object episode")
    steps, updates = (_counter(metadata, key) for key in ("environment_steps", "global_update_step"))
    if (not 0 < steps <= budget["total_environment_steps"] or steps % budget["world_count"] or
            path.name != f"policy_{steps}.safetensors"):
        raise ValueError("Checkpoint filename/counter lies outside the whole-collection training budget")
    if budget_name == "zero_update" and updates != 0:
        raise ValueError("A zero-update policy performed optimizer updates")
    if policy_kind == "trained" and steps == budget["total_environment_steps"] and updates == 0:
        raise ValueError("The final trained policy contains no optimizer updates")
    return {"environment_steps": steps, "global_update_step": updates,
            "resolved_config_sha256": hashlib.sha256(metadata["resolved_config_json"].encode()).hexdigest(),
            "policy_schema_sha256": _canonical_sha256(_json(metadata["policy_schema_json"])),
            "critic_schema_sha256": _canonical_sha256(_json(metadata["critic_schema_json"])),
            "admission": "Exact configuration, reference and counter metadata; tensors not decoded"}
