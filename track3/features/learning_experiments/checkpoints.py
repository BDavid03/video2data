"""Research-only neural checkpoint persistence and admission.

PyTorch is always supplied by the authorized caller, never imported here. Pass
the ORIGINAL model's state_dict(), not a compiled wrapper's state. These files
contain inference weights and provenance only: no optimizer, gradients, resume
state, scratch-depth admission, or submission admission. The operation context
owns run.json persistence and completion; save_checkpoint updates its record.
training_request is copied from that context's JSON-normalized record.request.
Source identity binds architecture, configuration, training and label filtering
(models.py, _config.py, train.py, data.py); all four must exist at future runtime.

Admission hashes the dataset manifest when request.dataset is present. Individual
sample-byte admission remains the data adapter's responsibility. Hashes establish
local consistency, not authenticity of an untrusted producer or pickle sandboxing.
"""

import hashlib
import json
import math
import os
from pathlib import PurePosixPath
from uuid import uuid4

from track3.paths import project_path
from track3.provenance import inherited_provenance, read_run, sha256_file
from ._common import FEATURE, ROOT, artifact, authorize, episode_partition, integer, number
from ._config import parameters

__all__ = ["save_checkpoint", "load_checkpoint"]

KIND = "learning_experiments.neural_checkpoint"
DATA_FIELDS = (
    "dataset_revision", "split_id", "target_kind", "geometry", "max_image_side",
    "calibration_sha256", "minimum_depth_m", "maximum_depth_m",
)
SOURCE_FILES = ("models.py", "_config.py", "train.py", "data.py")
MAX_SAMPLES = 1_000_000
MAX_STATE_ELEMENTS = 16_000_000
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


def _plain(value):
    """Detach JSON metadata, rejecting objects, nonfinite numbers and odd keys."""
    def visit(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child)
            return
        raise ValueError("Checkpoint metadata must be finite serialized JSON")
    visit(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(value):
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Require a lowercase SHA-256 digest")
    return value


def _agrees(left, right):
    """Serialized equality keeps booleans distinct from numeric metadata."""
    return json.dumps(_plain(left), sort_keys=True) == json.dumps(_plain(right), sort_keys=True)


def _relative(path):
    return path.relative_to(ROOT.resolve()).as_posix()


def _owner(directory, record):
    """A run can own only its direct, fixed feature output directory."""
    if type(record) is not dict:
        raise ValueError("Require a training record object")
    run_id = record.get("run_id")
    if (type(run_id) is not str or not run_id or run_id in (".", "..")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                   for c in run_id)):
        raise ValueError("Invalid training run ID")
    expected = ROOT.resolve() / "outputs/learning_experiments/train" / run_id
    if directory != expected.resolve() or directory != expected:
        raise ValueError("Checkpoint must belong to the canonical owning training directory")
    if record.get("kind") != "learning_experiments.train":
        raise ValueError("Only learning_experiments.train may own neural checkpoints")
    return directory / "run.json"


def _sources(record):
    hashes = record.get("source_hashes")
    if type(hashes) is not dict:
        raise ValueError("Training record needs its source_hashes inventory")
    identities = []
    for filename in SOURCE_FILES:
        path = project_path(ROOT, FEATURE / filename)
        identity = {"path": _relative(path), "sha256": sha256_file(path)}
        if hashes.get(identity["path"]) != identity["sha256"]:
            raise ValueError("Current neural sources differ from the training source inventory")
        identities.append(identity)
    return identities


def _contract(record, request, partition, config):
    contract = _plain(record.get("data_contract"))
    if type(contract) is not dict or any(key not in contract for key in DATA_FIELDS):
        raise ValueError("Require the complete preprocessing and depth-target contract")
    if (contract["dataset_revision"] != partition["dataset_revision"]
            or contract["split_id"] != partition["split_id"]
            or contract["target_kind"] != "moge2_pseudo_depth"):
        raise ValueError("Training data must use the frozen public pseudo-depth contract")
    _digest(contract["calibration_sha256"])
    side = integer(contract["max_image_side"], "max_image_side", 64, 256)
    low, high = contract["minimum_depth_m"], contract["maximum_depth_m"]
    if (type(low) not in (int, float) or type(high) not in (int, float)
            or not math.isfinite(low) or not math.isfinite(high) or not 0 < low < high):
        raise ValueError("Depth bounds must be positive and strictly ordered")
    geometry = contract["geometry"]
    if type(geometry) is not dict or not geometry:
        raise ValueError("Require explicit camera geometry")
    size = geometry.get("output_size_width_height")
    if type(size) is not list or len(size) != 2:
        raise ValueError("Geometry requires output_size_width_height")
    width, height = [integer(v, "image dimension", 1, side) for v in size]
    if width * height > 65536:
        raise ValueError("Training geometry exceeds the neural pixel budget")
    intrinsics = geometry.get("input_intrinsics_pixels")
    if (type(intrinsics) is not list or len(intrinsics) != 3
            or any(type(row) is not list or len(row) != 3 for row in intrinsics)
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for row in intrinsics for v in row)
            or intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0
            or intrinsics[0][1] != 0 or intrinsics[1][0] != 0
            or intrinsics[2] != [0, 0, 1]):
        raise ValueError("Require finite positive-focal-length pinhole intrinsics")
    if record["technique"] == "transformer":
        patch = config["patch_size"]
        if ((width + patch - 1) // patch) * ((height + patch - 1) // patch) > config["max_tokens"]:
            raise ValueError("Checkpoint geometry exceeds the transformer token budget")
    # Compare complete structures, including additional future contract fields.
    for key, value in contract.items():
        if key in request and not _agrees(request[key], value):
            raise ValueError(f"Training request and data contract disagree on {key}")
    return contract


def _dataset(record, request, contract, partition, samples):
    digest = _digest(record.get("dataset_sha256"))
    value = request.get("dataset")
    if value is None:
        raise ValueError("A trained checkpoint requires its explicit prepared dataset input")
    if type(value) is not str or not value:
        raise ValueError("Training dataset must be an explicit project input path")
    path = project_path(ROOT, value)
    inputs = record.get("inputs")
    if type(inputs) is not list or any(type(entry) is not dict for entry in inputs):
        raise ValueError("Training checkpoint requires its input hash inventory")
    matches = [entry for entry in inputs if type(entry.get("path")) is str
               and project_path(ROOT, entry["path"]) == path]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise ValueError("Training dataset is not uniquely bound to the input inventory")
    # Hash and parse the very same bytes, rather than trusting a producer's hash.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Dataset manifest bytes differ from the training record")
    document = _plain(json.loads(raw.decode("utf-8-sig")))
    if (type(document) is not dict or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or document.get("kind") != "scratch_depth_training_pairs"
            or document.get("status") != "completed"):
        raise ValueError("Require a completed dataset manifest")
    inherited_provenance([document])
    if (document["provenance"] != record["provenance"]
            or any(not _agrees(document.get(key), value) for key, value in contract.items())):
        raise ValueError("Dataset provenance or complete data contract differs from training")
    rows = document.get("samples")
    if type(rows) is not list or not rows:
        raise ValueError("Dataset manifest has no samples")
    allowed = set(partition["train_episodes"] + partition["tuning_episodes"]
                  + partition["holdout_episodes"])
    seen, counts = set(), {episode: 0 for episode in partition["train_episodes"]}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("Dataset manifest samples must be objects")
        episode, frame = row.get("episode_index"), row.get("frame_index")
        if (type(episode) is not int or episode not in allowed
                or type(frame) is not int or frame < 0 or (episode, frame) in seen):
            raise ValueError("Dataset manifest contains duplicate frames or undeclared episodes")
        seen.add((episode, frame))
        if episode in counts:
            counts[episode] += 1
    if sum(counts.values()) != samples or any(count == 0 for count in counts.values()):
        raise ValueError("Recorded training sample count differs from the dataset partition")
    return digest


def _metadata(record, *, completed):
    if type(record) is not dict:
        raise ValueError("Require a training record object")
    if record.get("status") != ("completed" if completed else "running"):
        raise ValueError("Training run has the wrong lifecycle status")
    if (record.get("research_only") is not True or record.get("submission_candidate") is not False
            or record.get("scratch_checkpoint_admitted") is not False
            or record.get("training_executed") is not True):
        raise ValueError("Checkpoint requires an owned research-only training operation")
    inherited_provenance(provenance=record.get("provenance"))
    request = _plain(record.get("request"))
    if type(request) is not dict:
        raise ValueError("Require the serialized training request")
    technique = record.get("technique")
    if technique not in ("ann", "cnn", "transformer") or request.get("technique") != technique:
        raise ValueError("Checkpoint technique must agree with the neural training request")
    values = request.get("model_parameters")
    if type(values) is not dict:
        raise ValueError("Require explicit architecture parameters in the request")
    config = parameters(technique, values)
    if not _agrees(record.get("model_config"), config):
        raise ValueError("Model config must contain normalized architecture parameters only")
    integer(request.get("seed"), "seed", 0, 2**32 - 1)
    epochs = integer(request.get("epochs"), "epochs", 1, 100)
    batch = integer(request.get("batch_size"), "batch_size", 1, 8)
    number(request.get("learning_rate"), "learning_rate", 0.000001, 0.1)
    number(request.get("weight_decay"), "weight_decay", 0, 1)
    if type(request.get("bf16_autocast")) is not bool:
        raise ValueError("Training request must record its explicit BF16 choice")
    if not _agrees(record.get("optimization"), {"bf16_autocast": request["bf16_autocast"]}):
        raise ValueError("Training optimization differs from the request")
    partition = episode_partition(request.get("train_episodes"), request.get("tuning_episodes"),
                                  request.get("holdout_episodes"))
    if not _agrees(record.get("episode_partition"), partition):
        raise ValueError("Checkpoint partition differs from the complete training request")
    samples = integer(record.get("training_samples"), "training_samples", 1, MAX_SAMPLES)
    per_epoch = (samples + batch - 1) // batch
    history = record.get("epochs")
    if type(history) is not list or not 1 <= len(history) <= epochs:
        raise ValueError("Require a nonempty bounded epoch history")
    _plain(history)
    for index, row in enumerate(history, 1):
        if (type(row) is not dict or type(row.get("epoch")) is not int
                or row["epoch"] != index or type(row.get("optimizer_steps")) is not int
                or row["optimizer_steps"] != index * per_epoch):
            raise ValueError("Epoch history disagrees with fixed complete-batch counters")
    steps = integer(record.get("optimizer_steps"), "optimizer_steps", 1, epochs * per_epoch)
    if steps != len(history) * per_epoch or (completed and len(history) != epochs):
        raise ValueError("Training run does not contain all claimed completed epochs")
    contract = _contract(record, request, partition, config)
    return {
        "training_request": request, "provenance": record["provenance"],
        "technique": technique, "model_config": config, "episode_partition": partition,
        "optimization": _plain(record["optimization"]),
        "data_contract": contract,
        "dataset_sha256": _dataset(record, request, contract, partition, samples),
        "source_identity": _sources(record),
    }, per_epoch, len(history)


def _inventory(record, directory, per_epoch, last_epoch):
    entries, artifacts = record.get("checkpoints"), record.get("artifacts")
    if type(entries) is not list or type(artifacts) is not dict:
        raise ValueError("Require checkpoint and artifact inventories")
    paths, epochs = set(), set()
    for index, entry in enumerate(entries, 1):
        if type(entry) is not dict:
            raise ValueError("Invalid checkpoint inventory entry")
        epoch = integer(entry.get("epoch"), "inventory epoch", 1, last_epoch)
        if epoch != index:
            raise ValueError("Checkpoint inventory must contain consecutive epochs in order")
        relative = entry.get("artifact")
        expected = f"checkpoints/epoch_{epoch:04d}.pt"
        if type(relative) is not str or relative != expected:
            raise ValueError("Checkpoint inventory must use canonical run-relative paths")
        path = project_path(ROOT, directory / PurePosixPath(relative))
        if path.parent != directory / "checkpoints" or path.name != PurePosixPath(relative).name:
            raise ValueError("Checkpoint path escapes its owning checkpoints directory")
        digest = _digest(entry.get("sha256"))
        if (path in paths or epoch in epochs or artifacts.get(relative) != digest
                or type(entry.get("optimizer_steps")) is not int
                or entry["optimizer_steps"] != epoch * per_epoch):
            raise ValueError("Checkpoint ownership, artifact hash or counters are inconsistent")
        paths.add(path)
        epochs.add(epoch)
    if entries and record.get("checkpoint") != entries[-1]["artifact"]:
        raise ValueError("Training record checkpoint pointer differs from its inventory")
    return entries


def _state(state, torch, *, copy):
    if not isinstance(state, dict) or not 1 <= len(state) <= 4096:
        raise ValueError("Require a nonempty bounded original model state_dict")
    result, elements = {}, 0
    for name, tensor in state.items():
        if (type(name) is not str or not name or "_orig_mod" in name.split(".")
                or not torch.is_tensor(tensor) or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.is_meta or not tensor.is_floating_point()
                or tensor.numel() == 0 or tensor.ndim > 8):
            raise ValueError("State must contain original, dense floating named tensors")
        elements += tensor.numel()
        if elements > MAX_STATE_ELEMENTS:
            raise ValueError("Checkpoint exceeds the bounded model-state budget")
        if not copy and (tensor.device.type != "cpu" or tensor.requires_grad or tensor.grad is not None):
            raise ValueError("Loaded checkpoint tensors must be CPU weights without gradients")
        value = tensor.detach().cpu().clone() if copy else tensor
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Checkpoint tensors must all be finite")
        result[name] = value
    return result


def save_checkpoint(directory, record, state, *, epoch, optimizer_steps, torch):
    """Atomically save one completed epoch and update the caller's run record.

    The running record must already contain that epoch's history and counters.
    The operation context persists/finalizes run.json and its artifact inventory.
    """
    authorize("model")
    authorize("train")
    directory = artifact(directory)
    run = _owner(directory, record)
    metadata, per_epoch, last_epoch = _metadata(record, completed=False)
    integer(epoch, "epoch", 1, last_epoch)
    integer(optimizer_steps, "optimizer_steps", 1, last_epoch * per_epoch)
    if epoch != last_epoch or optimizer_steps != epoch * per_epoch:
        raise ValueError("Save must describe the latest fully trained epoch")
    entries = _inventory(record, directory, per_epoch, last_epoch)
    if len(entries) != epoch - 1:
        raise ValueError("Save requires exactly the preceding completed epoch checkpoints")
    path = artifact(directory / "checkpoints" / f"epoch_{epoch:04d}.pt")
    if path.parent != directory / "checkpoints":
        raise ValueError("Checkpoint directory must be owned directly by this run")
    payload = {
        "schema_version": 1, "kind": KIND, "pretrained_weights": False,
        "research_only": True, "training_run": _relative(run),
        "training_run_id": record["run_id"], "trained_epochs": epoch,
        "optimizer_steps": optimizer_steps, **metadata,
        "state_dict": _state(state, torch, copy=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint")
    temporary = artifact(path.parent / f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    relative = path.relative_to(directory).as_posix()
    entries.append({"epoch": epoch, "optimizer_steps": optimizer_steps,
                    "artifact": relative, "sha256": digest})
    record["checkpoint"] = relative
    record["artifacts"][relative] = digest
    return path


def load_checkpoint(path, torch):
    """Admit completed-run weights with weights_only=True; never construct a model.

    Return (payload, owning training record, unique checkpoint inventory entry).
    No optimizer/gradient restoration or training continuation is supported.
    """
    path = project_path(ROOT, path)
    if path.parent.name != "checkpoints" or path.suffix != ".pt":
        raise ValueError("Require a direct .pt child of an owning checkpoints directory")
    directory = path.parent.parent
    record = read_run(ROOT, directory / "run.json", kind="learning_experiments.train")
    run = _owner(directory, record)
    metadata, per_epoch, last_epoch = _metadata(record, completed=True)
    entries = _inventory(record, directory, per_epoch, last_epoch)
    if not entries or entries[-1]["epoch"] != last_epoch:
        raise ValueError("Completed training must inventory its final epoch checkpoint")
    matches = [entry for entry in entries if directory / entry["artifact"] == path]
    if len(matches) != 1:
        raise ValueError("Checkpoint is not uniquely owned by the completed training run")
    entry = matches[0]
    # Hash and deserialize the same open file, then hash again to detect mutation.
    with path.open("rb") as stream:
        if not 1 <= os.fstat(stream.fileno()).st_size <= MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint file exceeds the bounded loading budget")
        def file_digest():
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
        if file_digest() != entry["sha256"]:
            raise ValueError("Checkpoint bytes differ from both owning inventories")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        if file_digest() != entry["sha256"]:
            raise ValueError("Checkpoint changed during deserialization")
    expected = {
        "schema_version": 1, "kind": KIND, "pretrained_weights": False,
        "research_only": True, "training_run": _relative(run),
        "training_run_id": record["run_id"], "trained_epochs": entry["epoch"],
        "optimizer_steps": entry["optimizer_steps"], **metadata,
    }
    if type(payload) is not dict or set(payload) != set(expected) | {"state_dict"}:
        raise ValueError("Unknown or incomplete neural checkpoint payload")
    for key, value in expected.items():
        # Exact serialized metadata comparison also excludes bool-as-int counters.
        if not _agrees(payload[key], value):
            raise ValueError(f"Checkpoint {key} differs from its admitted training record")
    _state(payload["state_dict"], torch, copy=False)
    return payload, record, entry
