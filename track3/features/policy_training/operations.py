"""Compose a complete budget, train unchanged upstream, select an inspected checkpoint."""

from decimal import Decimal
import math

from track3.gpu import GpuLease
from track3.provenance import input_file, sha256_file, write_json
from ._common import (FEATURE, ROOT, artifact, authorize, checked_entry, command, linux_path,
                      local_path, operation_run, producer, read_json, upstream)
from .checkpoints import admit_checkpoint, require_same_composition
from .recipes import recipe_overrides

MOVABLE_KEYS = ("output_dir", "hydra.run.dir")
FRACTIONS = ("0.0786", "0.1376", "0.1966", "0.2556", "0.3146", "0.3736", "0.4325", "0.4915", "0.5505", "0.8000", "1.0")


def _budget(request, record):
    from omegaconf import OmegaConf
    value = read_json(FEATURE / "budget.json")[request.budget]
    required = {"world_count", "total_environment_steps", "curriculum_thresholds", "updates_per_collection", "minimum_size",
                "capacity", "checkpoint_interval", "observation_storage_dtype", "save_replay", "note"}
    if set(value) != required:
        raise ValueError("Budget entries must be complete and contain only named fields")
    for key in ("world_count", "total_environment_steps", "minimum_size", "capacity", "checkpoint_interval"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"Invalid budget {key}")
    worlds, total = value["world_count"], value["total_environment_steps"]
    # The recipe's replay batch is 2,048 and upstream requires batch_size <= capacity
    # (training/flash_sac/config.py:204-205, replay.py:69-70); the evaluator re-checks it.
    if (total % worlds or value["capacity"] < max(worlds, 2048) or not 0 < value["minimum_size"] <= min(total, value["capacity"]) or
            value["checkpoint_interval"] % worlds or value["observation_storage_dtype"] != "float16" or
            type(value["save_replay"]) is not bool or not math.isfinite(value["updates_per_collection"]) or value["updates_per_collection"] <= 0):
        raise ValueError("Budget violates the world, replay, checkpoint or storage contract")
    if request.budget == "zero_update":
        original = OmegaConf.load(upstream("src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml", record))
        if (request.policy_kind == "trained" or total != worlds or value["capacity"] != max(worlds, 2048) or
                value["minimum_size"] > worlds or value["curriculum_thresholds"] != list(original.thresholds) or
                int(value["updates_per_collection"]) < 1):
            raise ValueError("Resting/replay policies require the one-collection zero_update budget")
    else:
        if request.policy_kind == "replay":
            raise ValueError("Replay policies must use zero_update")
        thresholds = [int(Decimal(total) * Decimal(fraction) / worlds) * worlds for fraction in FRACTIONS]
        if (value["curriculum_thresholds"] != thresholds or thresholds[0] <= 0 or
                any(a >= b for a, b in zip(thresholds, thresholds[1:])) or
                value["minimum_size"] != 100 * worlds or value["updates_per_collection"] != 8 * worlds / 4096 or
                value["checkpoint_interval"] != max(worlds, total // (25 * worlds) * worlds)):
            raise ValueError("Scaled budget must preserve the recipe's fractions and collection ratios")
    return value


def _reference(parent, request, record):
    report = read_json(artifact(parent, "inspect.json", record))
    if report["blocking"] or any(row["blocking"] for row in report["findings"]):
        raise ValueError("Retargeting inspection contains blocking findings")
    if report["fps"] != 20.0:
        raise ValueError("The training reference must be 20 Hz")
    bundle = producer(report["bundle_run"]["path"], "retargeting.support", request)
    checked_entry(report["bundle_run"], record)  # the bundle is unchanged since inspection
    for name in bundle[1]["artifacts"]:
        artifact(bundle, name, record)
    reference = checked_entry(report["processed_parquet"], record)
    checked_entry(report["support_file"], record)
    checked_entry(report["scene_run"], record)
    if (request.policy_kind == "resting") != (report["scene_mode"] == "resting"):
        raise ValueError("Resting policies require resting scenes; replay/trained policies require tracked scenes")
    record.update(retargeting_run=input_file(ROOT, parent[0]), scene_run=report["scene_run"],
                  reference=report["processed_parquet"], frame_count=report["frame_count"],
                  object_names=report["object_names"], scene_support=report["scene.support"])
    return reference, report


def _overrides(request, budget, reference, report, directory, record):
    values = {
        "experiment": ("sharpa_flash_sac", "Released floating Sharpa FlashSAC recipe"),
        "task.parquet": (linux_path(ROOT, reference), "Inspected immutable episode reference"),
        "task.motion_speed": (1.0, "One control step per video frame"),
        "logging.mode": ("disabled", "No remote logging or uploads"),
        "output_dir": (linux_path(ROOT, directory / "checkpoints"), "Project-owned checkpoints"),
        "hydra.run.dir": (linux_path(ROOT, directory / "hydra"), "Project-owned Hydra output"),
        "scene.world_count": (budget["world_count"], "Complete named budget"),
        "scene.support": (report["scene.support"], "Required support setting from retargeting inspection"),
        "training.total_environment_steps": (budget["total_environment_steps"], "Whole collections only"),
        "training.updates_per_collection": (budget["updates_per_collection"], "Preserve training ratio; positive schedule horizon for zero-update"),
        "training.replay.minimum_size": (budget["minimum_size"], "Budget-scaled warmup"),
        "training.replay.capacity": (budget["capacity"], "Bound GPU replay allocation"),
        "training.replay.observation_storage_dtype": ("float16", "Bound observation memory"),
        "training.checkpoint.save_interval_environment_steps": (budget["checkpoint_interval"], "Budget-scaled checkpoint interval"),
        "training.checkpoint.save_replay": (budget["save_replay"], "Explicit replay persistence choice"),
        "curriculum.thresholds": (budget["curriculum_thresholds"], "Scaled assistance schedule, or unchanged one-collection schedule"),
        "reset.seed": (request.seed, "Reproducible recorded seed"),
    }
    from omegaconf import OmegaConf
    curriculum = OmegaConf.load(upstream("src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml", record))
    # Keep noncontact interpolations unresolved until composed with the full job.
    stages = OmegaConf.to_container(curriculum.stages, resolve=False)
    return ([{"key": key, "value": value, "reason": reason} for key, (value, reason) in values.items()]
            + recipe_overrides(request.recipe_variant, stages))


def _resolved(path, overrides, frame_count):
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    for item in overrides:
        expected = item["value"]
        if item["key"] == "curriculum.stages":
            # Resolve preserved pinned interpolations in the complete job context.
            from copy import deepcopy
            expected_config = deepcopy(config)
            expected_config.curriculum.stages = expected
            expected = OmegaConf.to_container(expected_config.curriculum.stages, resolve=True)
        actual = OmegaConf.select(config, item["key"])
        if OmegaConf.is_config(actual):
            actual = OmegaConf.to_container(actual, resolve=True)
        if item["key"] not in ("experiment", "hydra.run.dir") and actual != expected:
            raise ValueError(f"Resolved training configuration ignored {item['key']}")
    if config.sim.fps != 20.0 or config.task.source_frame_playback or config.task.motion_start_frame != 0 or config.task.motion_end_frame != -1:
        raise ValueError("Training must preserve the entire 20 Hz episode")
    import pyarrow.parquet as pq
    file = pq.ParquetFile(local_path(config.task.parquet))
    if file.metadata.num_rows != 1:
        raise ValueError("Expected a single reference row")
    row = file.read(columns=["fps", "object_body_position"]).to_pylist()[0]
    if row["fps"] != 20.0 or len(row["object_body_position"]) != frame_count:
        raise ValueError("Reference clock differs from inspection")
    return OmegaConf.to_container(config, resolve=True, enum_to_str=True)


def _fixed(overrides):
    return {row["key"]: row["value"] for row in overrides if row["key"] not in MOVABLE_KEYS}


def compose(request):
    authorize("compose", request)
    parent = producer(request.retargeting_run, "retargeting.inspect", request)
    with operation_run("compose", request, [parent]) as (directory, record):
        reference, report = _reference(parent, request, record)
        budget = _budget(request, record)
        overrides = _overrides(request, budget, reference, report, directory, record)
        write_json(directory / "overrides.json", overrides)
        write_json(directory / "budget.json", budget)
        path = command({item["key"]: item["value"] for item in overrides}, directory, record, compose=True)
        resolved = _resolved(path, overrides, report["frame_count"])
        write_json(directory / "resolved-config.json", resolved)
        bodies = len(report["object_names"])
        bytes_per_transition = 4 * (286 + 218 * bodies) + 4 * (56 + 11 + 2 * (4 + 6 * bodies)) + 2
        record["memory_model"] = {"replay_bytes": budget["capacity"] * bytes_per_transition,
                                  "bytes_per_transition": bytes_per_transition,
                                  "collections_in_replay": budget["capacity"] / budget["world_count"],
                                  "simulation_bytes": None, "measured": False}
    return directory / "run.json"


def train(request):
    authorize("train", request)
    parent = producer(request.compose_run, "policy_training.compose", request)
    if any(parent[1]["request"].get(key, "contact_aware" if key == "recipe_variant" else None) != getattr(request, key)
           for key in ("budget", "policy_kind", "seed", "recipe_variant")):
        raise ValueError("Training request differs from the composed budget, kind or seed")
    inspection = producer(parent[1]["retargeting_run"]["path"], "retargeting.inspect", request)
    if input_file(ROOT, inspection[0]) != parent[1]["retargeting_run"]:
        raise ValueError("Retargeting inspection changed")
    with operation_run("train", request, [parent, inspection]) as (directory, record):
        reference, report = _reference(inspection, request, record)
        budget = read_json(artifact(parent, "budget.json", record))
        if budget != _budget(request, record):
            raise ValueError("Named budget changed since composition")
        artifact(parent, "resolved-config.yaml", record)
        composed = read_json(artifact(parent, "resolved-config.json", record))
        previous = read_json(artifact(parent, "overrides.json", record))
        overrides = _overrides(request, budget, reference, report, directory, record)
        if _fixed(previous) != _fixed(overrides):
            raise ValueError("Only output locations may change after composition")
        write_json(directory / "overrides.json", overrides)
        write_json(directory / "budget.json", budget)
        values = {item["key"]: item["value"] for item in overrides}
        resolved = _resolved(command(values, directory, record, compose=True), overrides, report["frame_count"])
        require_same_composition(composed, resolved, ROOT)
        write_json(directory / "resolved-config.json", resolved)
        for name in ("training/checkpoint.py", "training/checkpoint_metadata.py", "training/flash_sac/checkpoint.py"):
            upstream("src/flash_chord/" + name, record)
        lease = GpuLease(ROOT, request.headroom_mib, request.admission_allowance_mib, stage="policy_training.train")
        try:
            with lease:
                command(values, directory, record)
                lease.require_headroom()
        finally:
            record["gpu"] = lease.report()
        checkpoints = record["checkpoints"] = []
        for path in sorted((directory / "checkpoints").glob("policy_*.safetensors")):
            admitted = admit_checkpoint(path, root=ROOT, expected_config=resolved, reference=reference,
                                        frame_count=report["frame_count"], budget=budget,
                                        budget_name=request.budget, policy_kind=request.policy_kind)
            checkpoints.append({"file": path.relative_to(directory).as_posix(), "sha256": sha256_file(path), **admitted})
        if not checkpoints or max(item["environment_steps"] for item in checkpoints) != budget["total_environment_steps"]:
            raise ValueError("The final-budget checkpoint is missing")
        record["checkpoints"] = sorted(checkpoints, key=lambda item: item["environment_steps"])
        if any(a["global_update_step"] > b["global_update_step"] for a, b in zip(record["checkpoints"], record["checkpoints"][1:])):
            raise ValueError("Checkpoint optimizer counters go backwards")
    return directory / "run.json"


def select(request):
    authorize("select", request)
    training = producer(request.training_run, "policy_training.train", request)
    inspections = [producer(value, "policy_rollout.inspect", request) for value in request.rollout_inspections]
    if request.policy_kind != training[1]["policy_kind"]:
        raise ValueError("Selection policy kind differs from training")
    with operation_run("select", request, [training, *inspections]) as (directory, record):
        expected = read_json(artifact(training, "resolved-config.json", record))
        budget = read_json(artifact(training, "budget.json", record))
        reference = checked_entry(training[1]["reference"], record)
        training_pin = input_file(ROOT, training[0])
        candidates = []
        for inspection in inspections:
            report = read_json(artifact(inspection, "inspect.json", record))
            if (report.get("blocking") is not False or inspection[1].get("blocking") is not False or
                    report["training_run"] != training_pin):
                continue
            if (report.get("policy_kind") != request.policy_kind or
                    inspection[1].get("policy_kind") != request.policy_kind or
                    inspection[1].get("training_run") != report["training_run"] or
                    inspection[1].get("checkpoint") != report["checkpoint"]):
                raise ValueError("Inspection metadata differs from its report or selected policy kind")
            matches = [item for item in training[1]["checkpoints"] if item["sha256"] == report["checkpoint"]["sha256"]]
            if len(matches) != 1:
                raise ValueError("Rollout checkpoint is not uniquely owned by this training run")
            item = matches[0]
            checkpoint = artifact(training, item["file"], record)
            if input_file(ROOT, checkpoint) != report["checkpoint"]:
                raise ValueError("Inspected checkpoint path/hash differs")
            admitted = admit_checkpoint(checkpoint, root=ROOT, expected_config=expected, reference=reference,
                                        frame_count=training[1]["frame_count"], budget=budget,
                                        budget_name=training[1]["request"]["budget"], policy_kind=request.policy_kind)
            if any(item.get(key) != value for key, value in admitted.items()):
                raise ValueError("Selected checkpoint differs from its admitted training inventory")
            candidates.append((item["environment_steps"], checkpoint, inspection, report))
        if not candidates:
            raise ValueError("No inspected valid rollout exists for a checkpoint of this training run")
        steps, checkpoint, inspection, report = max(candidates, key=lambda item: item[0])
        if request.final_checkpoint_only and steps != budget["total_environment_steps"]:
            raise ValueError("The frozen comparison forbids fallback from the final scheduled checkpoint")
        write_json(directory / "selection.json", {"checkpoint": input_file(ROOT, checkpoint),
            "training_run": training_pin, "inspection_run": input_file(ROOT, inspection[0]),
            "policy_kind": request.policy_kind,
            "reason": "Final scheduled checkpoint only" if request.final_checkpoint_only else "Latest saved checkpoint with an inspected valid rollout",
            "reference": training[1]["reference"], "tracking_metrics": report.get("tracking_metrics"),
            "metrics_scope": "Tracking against the reconstructed reference, not leaderboard scores"})
    return directory / "run.json"
