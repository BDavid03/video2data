"""Accounting-only adapter for an existing whole-arm scratch research receipt."""

from dataclasses import asdict, dataclass

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from .operations import ROOT, FEATURE, _load, _missing_phases, _number, _run_id, _sha256, _span_ok

TRAIN = (0, 11, 12, 18, 21, 23, 26, 31)
TUNE = (39, 42)


@dataclass(frozen=True)
class GroupedAccountingRequest:
    receipt: str
    receipt_sha256: str
    accounting_only: bool = True

    @classmethod
    def from_file(cls, path):
        return cls(**_load(project_path(ROOT, path)))

    def __post_init__(self):
        if self.accounting_only is not True:
            raise ValueError("Whole-arm research receipts support accounting only")
        if not _sha256(self.receipt_sha256):
            raise ValueError("Predeclare the whole-arm receipt SHA256")


def _admit_trace(request, path, runtime):
    """Return the trace, its owning producer and the executed training episodes."""
    if sha256_file(path) != request.receipt_sha256:
        raise ValueError("Whole-arm receipt differs from its predeclared identity")
    trace = _load(path)
    producer_path = path.parent / "run.json"
    producer = _load(producer_path)
    if (producer.get("kind") != "scratch_depth.research_train"
            or producer.get("status") not in ("completed", "failed", "interrupted")
            or producer.get("provenance") != "video_only"
            or producer.get("artifacts", {}).get(path.name) != request.receipt_sha256
            or project_path(ROOT, trace.get("producer_run", "")) != producer_path
            or trace.get("schema_version") != 1 or trace.get("scope") != "whole_training_arm"
            or trace.get("source_split") != "public"):
        raise ValueError("Whole-arm trace lacks its terminal owning producer binding/public scope")
    # The fit may use a subset of training episodes; declared groups retain the frozen protocol.
    if trace.get("episode_groups") != {"train": list(TRAIN), "tuning": list(TUNE)}:
        raise PermissionError("Whole-arm receipt must declare the frozen public train/tuning groups")
    actual = trace.get("episode_indices")
    if (not isinstance(actual, list) or not actual or actual != sorted(set(actual))
            or any(type(episode) is not int or episode not in TRAIN for episode in actual)
            or not set(actual).issubset(runtime.get("authorized_episode_indices", []))):
        raise PermissionError("Whole-arm fit admits explicit training episodes only, excluding 41/43/44")
    return trace, producer_path, producer, actual


def _admit_dataset(producer, actual):
    """Return the preparation dataset path, its run path, its entry and its samples."""
    dataset_entry = producer.get("source_dataset", {})
    dataset_path = project_path(ROOT, dataset_entry.get("path", ""))
    dataset_parts = dataset_path.relative_to(ROOT).parts
    if (len(dataset_parts) != 5 or dataset_parts[:3] != ("outputs", "scratch_depth", "research_prepare")
            or dataset_path.name != "dataset.json" or input_file(ROOT, dataset_path) != dataset_entry
            or dataset_entry not in producer.get("inputs", [])):
        raise ValueError("Training dataset must be an exact owned preparation input")
    prepared_path = dataset_path.parent / "run.json"
    prepared = _load(prepared_path)
    if (prepared.get("kind") != "scratch_depth.research_prepare" or prepared.get("status") != "completed"
            or prepared.get("provenance") != "video_only"
            or prepared.get("artifacts", {}).get(dataset_path.name) != dataset_entry["sha256"]):
        raise ValueError("Whole-arm training requires completed video-only preparation")
    dataset = _load(dataset_path)
    samples = dataset.get("samples")
    if (dataset.get("source_split") != "public" or dataset.get("real_captured_only") is not True
            or not isinstance(samples, list) or not 1 <= len(samples) <= 256):
        raise ValueError("Admit bounded real public preparation samples")
    used = set()
    for sample in samples:
        episode = sample.get("episode_index")
        if type(episode) is not int or episode not in (*TRAIN, *TUNE):
            raise PermissionError("Excluded episode in grouped dataset manifest")
        if sample.get("partition") != ("train" if episode in TRAIN else "tuning"):
            raise ValueError("Whole episodes must remain in their frozen partition")
        if episode in TRAIN:
            used.add(episode)
    if sorted(used) != actual or producer.get("no_tuning_or_holdout_loader") is not True:
        raise ValueError("Trace training scope differs from prepared rows or loader admission")
    return dataset_path, prepared_path, dataset_entry, samples


def aggregate_grouped(request):
    """Report actual grouped training scope; never infer a per-episode benchmark."""
    runtime = _load(FEATURE / "runtime.json")
    if runtime.get("aggregate_grouped_authorized") is not True:
        raise PermissionError("Whole-arm runtime accounting remains disabled")
    output = output_path(ROOT, "outputs/runtime_measurement/grouped_accounting")
    path = project_path(ROOT, request.receipt)
    parts = path.relative_to(ROOT).parts
    if (len(parts) != 5 or parts[:3] != ("outputs", "scratch_depth", "research_train")
            or parts[-1] != "runtime_trace.json"):
        raise PermissionError("Only owned scratch research_train receipts are admitted before reading")
    trace, producer_path, producer, actual = _admit_trace(request, path, runtime)
    dataset_path, prepared_path, dataset_entry, samples = _admit_dataset(producer, actual)
    workload = trace.get("workload", {})
    if (workload.get("request") != producer.get("request")
            or workload.get("dataset_sha256") != dataset_entry["sha256"]
            or workload.get("optimizer_steps") != producer.get("optimizer_steps")
            or workload.get("training_clip_centres") != sum(s["partition"] == "train" for s in samples)
            or workload.get("architecture") != producer.get("architecture")):
        raise ValueError("Whole-arm workload does not agree with its producer and prepared dataset")
    if trace.get("timing_source") not in producer.get("sources", []):
        raise ValueError("Timing helper source identity must be bound by the producer")
    spans = trace.get("spans")
    if not isinstance(spans, list) or not 1 <= len(spans) <= 4096:
        raise ValueError("Bounded actual timing spans are required")
    processed = workload.get("processed_examples")
    if type(processed) is not int or not 0 <= processed <= 1280:
        raise ValueError("Whole-arm example ceiling exceeded")
    for span in spans:
        if (not _span_ok(span) or not 0 <= span["iteration"] <= 1280
                or (span["status"] == "failed") != bool(span.get("failure"))):
            raise ValueError("Malformed whole-arm phase span")
    accounting = [{**span, "seconds": (span["stop_ns"] - span["start_ns"]) / 1e9} for span in spans]
    peak = producer.get("gpu_torch", {})
    device_peak = producer.get("gpu", {}).get("peak_sampled_total_device_used_mib")
    memory = {"allocated_peak_bytes": peak.get("peak_allocated_bytes"),
              "reserved_peak_bytes": peak.get("peak_reserved_bytes"),
              "sampled_device_peak_bytes": device_peak * 1024**2 if _number(device_peak) else None}
    if any(value is not None and not _number(value) for value in memory.values()):
        raise ValueError("Whole-arm memory fields must be nonnegative bytes or null")
    wall = producer.get("elapsed_seconds")
    if not _number(wall):
        raise ValueError("Terminal producer must declare finite whole-operation wall time")
    inputs = [path, producer_path, dataset_path, prepared_path, FEATURE / "runtime.json", ROOT / "track3/timing.py"]
    record = run_fields(ROOT, "runtime_measurement", "aggregate_grouped", asdict(request), inputs=inputs,
                        allowed=("video_only",))
    record.update(status="completed", accounting_only=True, comparison_eligible=False,
                  scope="whole_training_arm", episode_groups=trace["episode_groups"], executed_episode_indices=actual,
                  producer_status=producer["status"], producer_failure=producer.get("error"),
                  producer_sources=producer.get("sources", []), workload=workload, memory=memory,
                  producer_wall_seconds=wall, exceeds_declared_arm_seconds=wall > 1800, spans=accounting,
                  missing_phases=_missing_phases(spans),
                  producer_environment=producer.get("environment"), producer_budget=producer.get("budget"),
                  semantics=trace.get("semantics"),
                  limitations=["Whole training-arm accounting; not the 900-second matched benchmark protocol.",
                               "First32 sampled examples are not repeated identical warm workloads.",
                               "End-to-end includes initialization/checkpoint; phase spans overlap and are incomplete.",
                               "Decode is prepared NPZ read; no upstream RGB video decode is measured.",
                               "No per-episode allocation, backend speedup, correctness or bottleneck inference."])
    directory = output / _run_id()
    write_json(directory / "run.json", record)
    return directory / "run.json"
