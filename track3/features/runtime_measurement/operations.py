"""Account for bounded producer receipts without running the measured workload."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.timing import PHASES, REGIMES

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
EPISODES = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)
MEMORY = ("allocated_peak_bytes", "reserved_peak_bytes", "sampled_device_peak_bytes")
WORKLOAD = {"input_sha256", "frames", "max_side", "batch_size", "numerical_mode", "parameters_sha256",
            "output_policy", "threads", "environment_identity", "backend", "seed"}
WORKLOAD_CEILINGS = {"frames": 64, "max_side": 128, "batch_size": 4, "threads": 64}


def _load(path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Receipt/configuration exceeds 4 MiB ceiling")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]


def _span_ok(span):
    """Checks shared by both receipt forms; callers add their own iteration and wall bounds."""
    return (span.get("phase") in PHASES and span.get("regime") in REGIMES
            and type(span.get("iteration")) is int
            and span.get("device") in ("cpu", "gpu") and span.get("clock") == "perf_counter_ns"
            and all(type(span.get(key)) is int for key in ("start_ns", "stop_ns"))
            and 0 <= span["start_ns"] <= span["stop_ns"]
            and span.get("status") in ("completed", "failed")
            and (span["device"] != "gpu" or span.get("synchronized") is True))


def _missing_phases(spans):
    return [phase for phase in PHASES if not any(s["phase"] == phase for s in spans)]


@dataclass(frozen=True)
class Request:
    episode_index: int
    receipts: list[dict]
    max_wall_seconds: int = 900
    max_cold_iterations: int = 3
    max_warmup_iterations: int = 5
    max_warm_iterations: int = 20

    @classmethod
    def from_file(cls, path):
        return cls(**_load(project_path(ROOT, path)))

    def __post_init__(self):
        if type(self.episode_index) is not int or self.episode_index not in EPISODES:
            raise PermissionError("Only public training/tuning episodes; 41/43/44 excluded")
        if not 1 <= len(self.receipts) <= 2:
            raise ValueError("Admit one baseline and at most one challenger receipt")
        for name, ceiling in (("max_wall_seconds", 900), ("max_cold_iterations", 3),
                              ("max_warmup_iterations", 5), ("max_warm_iterations", 20)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}")
        for entry in self.receipts:
            if set(entry) != {"path", "sha256"} or not isinstance(entry["path"], str):
                raise ValueError("Receipts require exact path and predeclared SHA256")
            if not _sha256(entry["sha256"]):
                raise ValueError("Declare a lowercase SHA256 receipt identity")


def _check_binding(trace, producer, entry, request, name):
    if (producer.get("status") not in ("completed", "failed") or producer.get("provenance") != "video_only"
            or producer.get("episode_index") != request.episode_index or producer.get("split") != "public"
            or producer.get("artifacts", {}).get(name) != entry["sha256"]):
        raise ValueError("Producer must explicitly bind the trace, episode, provenance and terminal status")
    if (trace.get("schema_version") != 1 or trace.get("producer_status") != producer["status"]
            or trace.get("episode_index") != request.episode_index or trace.get("split") != "public"
            or trace.get("sources") != producer.get("sources") or not trace.get("sources")):
        raise ValueError("Trace source identity and terminal status must agree with producer")
    if any(not isinstance(source, dict) or set(source) != {"path", "sha256"}
           or not isinstance(source["path"], str) or not _sha256(source["sha256"]) for source in trace["sources"]):
        raise ValueError("Producer source identities must be explicit")


def _check_workload(workload):
    if not isinstance(workload, dict) or set(workload) != WORKLOAD or any(workload[k] is None for k in WORKLOAD):
        raise ValueError("Declare exact workload, environment and backend before comparison")
    if not all(_sha256(workload[key]) for key in ("input_sha256", "parameters_sha256")):
        raise ValueError("Workload input and parameter identities require SHA256")
    if any(not isinstance(workload[key], str) or not workload[key].strip()
           for key in ("numerical_mode", "output_policy", "environment_identity", "backend")):
        raise ValueError("Workload mode/policy/environment/backend require nonempty identities")
    if type(workload["seed"]) is not int or workload["seed"] < 0:
        raise ValueError("Declare a nonnegative integer workload seed")
    if any(type(workload[key]) is not int or not 1 <= workload[key] <= ceiling
           for key, ceiling in WORKLOAD_CEILINGS.items()):
        raise ValueError("Workload exceeds declared source-only protocol ceiling")


def _phase_accounting(spans, wall_seconds, request):
    if not isinstance(spans, list) or not 1 <= len(spans) <= 4096:
        raise ValueError("One to 4096 real phase spans required")
    ceilings = dict(zip(REGIMES, (request.max_cold_iterations, request.max_warmup_iterations,
                                  request.max_warm_iterations)))
    groups = {}
    for span in spans:
        if (not _span_ok(span) or not 0 <= span["iteration"] < ceilings[span["regime"]]
                or (span["stop_ns"] - span["start_ns"]) / 1e9 > wall_seconds):
            raise ValueError("Invalid/unbounded span or unsynchronized GPU host-wall measurement")
        if (span["status"] == "failed") != bool(span.get("failure")):
            raise ValueError("Failed spans require explicit failure type")
        groups.setdefault((span["phase"], span["regime"]), []).append(span)
    if (max(s["stop_ns"] for s in spans) - min(s["start_ns"] for s in spans)) / 1e9 > wall_seconds:
        raise ValueError("Span clock extent exceeds declared whole-operation wall time")
    accounting = []
    for (phase, regime), values in sorted(groups.items()):
        durations = [(s["stop_ns"] - s["start_ns"]) / 1e9 for s in values if s["status"] == "completed"]
        accounting.append({"phase": phase, "regime": regime, "completed_spans": len(durations),
                           "failed_spans": len(values) - len(durations), "seconds": durations})
    return accounting


def _admit(entry, request):
    path = project_path(ROOT, entry["path"])
    relative = path.relative_to(ROOT).parts
    episode = f"episode_{request.episode_index:06d}"
    if (len(relative) < 6 or relative[0] != "outputs" or relative[2:4] != ("public", episode)
            or path.name != "runtime_trace.json"):
        raise PermissionError("Trace must be an admitted public episode producer artifact before reading")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("Declared trace hash differs")
    trace = _load(path)
    producer_path = path.parent / "run.json"
    producer = _load(producer_path)
    _check_binding(trace, producer, entry, request, path.name)
    # Hash identities are historical, not a requirement that current source equals old source.
    workload = trace.get("workload")
    _check_workload(workload)
    if not _number(trace.get("wall_seconds")) or trace["wall_seconds"] > request.max_wall_seconds:
        raise ValueError("Missing wall time or exceeded producer wall budget")
    memory = trace.get("memory")
    if not isinstance(memory, dict) or set(memory) != set(MEMORY):
        raise ValueError("Allocated/reserved/device memory must be distinct nullable fields")
    if any(value is not None and (type(value) is not int or value < 0) for value in memory.values()):
        raise ValueError("Memory fields are bytes or null, never fabricated zeros")
    spans = trace.get("spans")
    accounting = _phase_accounting(spans, trace["wall_seconds"], request)
    return {"receipt": entry, "producer": input_file(ROOT, producer_path), "producer_status": producer["status"],
            "failure": producer.get("error"), "sources": trace["sources"], "workload": workload,
            "wall_seconds": trace["wall_seconds"], "memory": memory, "phase_accounting": accounting,
            "missing_phases": _missing_phases(spans)}


def aggregate(request):
    """Persist receipt accounting only; never invokes a benchmark, profiler or compiler."""
    runtime = _load(FEATURE / "runtime.json")
    if runtime.get("aggregate_authorized") is not True:
        raise PermissionError("Runtime receipt aggregation remains disabled")
    if request.episode_index not in runtime.get("authorized_episode_indices", []):
        raise PermissionError("Runtime episode requires explicit admission")
    output = output_path(ROOT, Path("outputs/runtime_measurement/public") /
                         f"episode_{request.episode_index:06d}")
    rows = [_admit(entry, request) for entry in request.receipts]
    if sum(row["wall_seconds"] for row in rows) > request.max_wall_seconds:
        raise ValueError("Combined producer wall budget exceeded")
    workload_equal = len(rows) == 2 and all(rows[0]["workload"][key] == rows[1]["workload"][key]
                                              for key in rows[0]["workload"] if key != "backend")
    inputs = [project_path(ROOT, row["receipt"]["path"]) for row in rows]
    inputs += [project_path(ROOT, row["producer"]["path"]) for row in rows]
    record = run_fields(ROOT, "runtime_measurement", "aggregate", asdict(request),
                        inputs=[*inputs, FEATURE / "runtime.json", ROOT / "track3/timing.py"], allowed=("video_only",))
    record.update(status="completed", episode_index=request.episode_index, split="public",
                  rows=rows, matched_declared_workload=workload_equal,
                  limitations=["Receipts are producer declarations, not independent profiler evidence.",
                               "Span durations may overlap; never sum phase spans as end-to-end time.",
                               "Synchronization changes overlap; host-wall times are not CUDA-event times.",
                               "Matched declarations alone do not establish numerical correctness or speedup."])
    directory = output / _run_id()
    write_json(directory / "run.json", record)
    return directory / "run.json"
