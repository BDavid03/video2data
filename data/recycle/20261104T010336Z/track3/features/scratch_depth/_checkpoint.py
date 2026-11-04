"""Scratch-model checkpoint ownership and inference admission, without model imports."""

import math

from track3.provenance import sha256_file
from ._common import ARCHITECTURE, FEATURE, local_path, positive_integer, relative, within

DATA_FIELDS = ("dataset_revision", "split_id", "target_kind", "geometry", "max_image_side",
               "calibration_sha256", "minimum_depth_m", "maximum_depth_m")
MODEL_CONFIG = {"base_channels": 24, "dropout": 0.1}


def _require_optimization(payload, training):
    """Optional schema-2 extension; historical eager checkpoints stay readable.

    Keep admission here independent of the optimization module being admitted:
    prediction hashes sources but never imports that module or the CUDA adapter.
    Missing metadata is allowed only for the original eager training recipe.
    """
    fields = ("bf16_autocast", "channels_last", "torch_compile", "fused_adam")
    request = training["request"]
    options = {name: request.get(name, False) for name in fields}
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Checkpoint optimization flags must be explicit booleans")
    options["loss_backend"] = request.get("loss_backend", "torch")
    if options["loss_backend"] not in ("torch", "cuda"):
        raise ValueError("Checkpoint has an unknown loss backend")
    metadata = ("optimization", "optimization_sources")
    present = any(key in record for record in (payload, training) for key in metadata)
    used = any(options[name] for name in fields) or options["loss_backend"] == "cuda"
    if not present:
        if used:
            raise ValueError("Optimized checkpoints require optimization source identity")
        return
    if any(record.get("optimization") != options for record in (payload, training)):
        raise ValueError("Checkpoint optimization options differ from the training request")
    sources = []
    for filename in ("_optimization.py", "train.py"):
        definition = {"path": relative(FEATURE / filename), "sha256": sha256_file(FEATURE / filename)}
        sources.append(definition)
    if options["loss_backend"] == "cuda":
        paths = [FEATURE / "_cuda_loss.py", *sorted(
            path for path in FEATURE.rglob("*")
            if path.is_file() and path.suffix in {".cpp", ".cu"})]
        definitions = [{"path": relative(path), "sha256": sha256_file(path)} for path in paths]
        sources.extend(definitions)
    if any(record.get("optimization_sources") != sources for record in (payload, training)):
        raise ValueError("Scratch optimization sources changed; use the checkpoint's original implementation")
    inventory = training.get("source_hashes", {})
    if any(inventory.get(source["path"]) != source["sha256"] for source in sources):
        raise ValueError("Training source inventory does not identify the optimization implementation")


def owned_checkpoint(path, training):
    directory = path.parent.parent
    if path.parent.name != "checkpoints":
        raise ValueError("Scratch checkpoints must belong to their training run's checkpoints directory")
    digest = sha256_file(path)
    matches = [row for row in training.get("checkpoints", [])
               if within(directory, row["artifact"]) == path and row.get("sha256") == digest]
    if len(matches) != 1 or training.get("artifacts", {}).get(path.relative_to(directory).as_posix()) != digest:
        raise ValueError("Checkpoint path/hash is not uniquely owned by the completed training run")
    return matches[0], digest


def require_payload(payload, training, entry, training_directory):
    if (not isinstance(payload, dict) or payload.get("schema_version") != 2 or
            payload.get("architecture") != ARCHITECTURE or payload.get("model_config") != MODEL_CONFIG or
            training.get("model_config") != MODEL_CONFIG or payload.get("pretrained_weights") is not False or
            payload.get("provenance") != training["provenance"] or
            payload.get("training_run_id") != training["run_id"] or
            local_path(payload.get("training_run", "")) != training_directory / "run.json" or
            payload.get("training_request") != training["request"] or
            payload.get("dataset_sha256") != training.get("dataset_sha256") or
            payload.get("coursework_hashes") != training.get("coursework_hashes")):
        raise ValueError("Checkpoint model, request or provenance differs from its owning training run")
    definition = {"path": relative(FEATURE / "_network.py"), "sha256": sha256_file(FEATURE / "_network.py")}
    if payload.get("model_source") != definition or training.get("model_source") != definition:
        raise ValueError("Scratch network source changed; use the checkpoint's original implementation")
    if training.get("source_hashes", {}).get(definition["path"]) != definition["sha256"]:
        raise ValueError("Training source inventory does not identify this network definition")
    _require_optimization(payload, training)
    contract = training.get("data_contract", {})
    if any(key not in payload or payload[key] != contract.get(key) for key in DATA_FIELDS):
        raise ValueError("Checkpoint preprocessing or target contract differs from training")
    if payload["target_kind"] != "moge2_pseudo_depth":
        raise ValueError("Unknown scratch-depth target source")
    low, high = payload["minimum_depth_m"], payload["maximum_depth_m"]
    if (type(low) not in (int, float) or type(high) not in (int, float) or
            not math.isfinite(low) or not math.isfinite(high) or not 0 < low < high):
        raise ValueError("Checkpoint depth bounds must be finite, positive and ordered")
    positive_integer(payload["max_image_side"], "checkpoint max_image_side", 64, 512)
    epochs, samples, batch = (training["request"]["epochs"], training["training_samples"], training["request"]["batch_size"])
    for name, value in (("epochs", epochs), ("training_samples", samples), ("batch_size", batch)):
        positive_integer(value, name)
    epoch = payload.get("trained_epochs")
    positive_integer(epoch, "checkpoint epoch", 1, epochs)
    per_epoch = (samples + batch - 1) // batch
    expected_steps = epoch * per_epoch
    if (entry.get("epoch") != epoch or entry.get("optimizer_steps") != expected_steps or
            type(payload.get("optimizer_steps")) is not int or payload["optimizer_steps"] != expected_steps or
            training.get("optimizer_steps") != epochs * per_epoch or len(training.get("epochs", [])) != epochs):
        raise ValueError("Checkpoint counters disagree with the complete fixed-epoch training request")
    if not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
        raise ValueError("Checkpoint has no model state")


def require_finite_state(state, torch):
    if not state or any(not isinstance(name, str) or not torch.is_tensor(value) or
                        not bool(torch.isfinite(value).all()) for name, value in state.items()):
        raise ValueError("Scratch model state must contain finite named tensors")
