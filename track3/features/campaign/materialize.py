"""Materialize reviewable producer requests from an owned queue; execute nothing."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from uuid import uuid4

from track3.paths import permitted_root, output_path, project_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json

from .request_drafts import unresolved_fields

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent


@dataclass(frozen=True)
class MaterializeRequest:
    queue_run: str | None = None
    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig")))

    def __post_init__(self):
        if self.queue_run is not None and (type(self.queue_run) is not str or not self.queue_run):
            raise ValueError("queue_run must name the completed campaign.queue run.json")
        if (type(self.overrides) is not dict or any(type(key) is not str or type(value) is not dict
                                                  for key, value in self.overrides.items())):
            raise ValueError("overrides must map exact task IDs to request-field objects")
        json.dumps(asdict(self), allow_nan=False)


def _task_id(row):
    split, episode, stage = row.get("split"), row.get("episode_index"), row.get("stage")
    if (split not in ("public", "evaluation") or type(episode) is not int or episode < 0 or episode == 41
            or type(stage) is not str or re.fullmatch(r"[a-zA-Z0-9_.:-]+", stage) is None):
        raise ValueError("Queue row has an invalid split, episode or stage")
    return f"{split}:{episode}:{stage}"


def _scene_choices(draft, names):
    """Materialization admits explicit scene choices without inventing physics."""
    missing = []
    for name in ("mass_kg", "supports"):
        value = draft.get(name)
        if type(value) is not dict or set(value) != set(names):
            missing.append(name)
    if "mass_kg" not in missing:
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in draft["mass_kg"].values()):
            raise ValueError("Each explicit scene mass must be finite and positive")
    if "supports" not in missing:
        supports = draft["supports"]
        if any(type(target) is not str or target not in ["table", *names] or name == target
               for name, target in supports.items()):
            raise ValueError("Support must name the table or a different declared object")
        remaining = set(names)
        while remaining:
            ready = {name for name in remaining if supports[name] not in remaining}
            if not ready:
                raise ValueError("Explicit support choices contain a cycle")
            remaining -= ready
    return missing


def _admit_queue(request):
    """The completed queue producer and its artifact, if nothing they planned from has changed since."""
    queue_path = project_path(ROOT, request.queue_run)
    if queue_path.name != "run.json":
        raise ValueError("queue_run must be its owning run.json")
    producer = read_run(ROOT, queue_path, kind="campaign.queue", allowed=("video_only",))
    run_id = producer.get("run_id")
    if (type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None
            or run_id in (".", "..")
            or queue_path != ROOT / "outputs/campaign/queue" / run_id / "run.json"):
        raise ValueError("Queue must remain in its exact owning campaign output directory")
    producer_hash = sha256_file(queue_path)
    snapshots = []
    for entry in producer.get("inputs", []):
        path = project_path(ROOT, entry["path"])
        if (path.is_relative_to(ROOT / "outputs") or path.is_relative_to(ROOT / "runs")
                or entry["path"] == "reports/data-audit/trajectories.json"):
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("Queue records, pin history or roster changed; regenerate the queue")
            snapshots.append(entry)
    pins_path = ROOT / "outputs/campaign/pins.jsonl"
    if pins_path.exists() and not any(project_path(ROOT, entry["path"]) == pins_path for entry in snapshots):
        raise ValueError("Pins were added after queue creation; regenerate the queue")
    queue_file = project_path(ROOT, queue_path.parent / "queue.json")
    queue_hash = sha256_file(queue_file)
    if producer.get("artifacts", {}).get("queue.json") != queue_hash:
        raise ValueError("Queue artifact differs from its completed producer")
    document = json.loads(queue_file.read_text(encoding="utf-8-sig"))
    if type(document.get("queue")) is not list:
        raise ValueError("Queue artifact must contain an ordered task list")
    return (queue_path, producer, {"path": queue_path.relative_to(ROOT).as_posix(), "sha256": producer_hash},
            queue_file, {"path": queue_file.relative_to(ROOT).as_posix(), "sha256": queue_hash}, document)


def _prepare(index, task_id, row, overrides, producer, identity):
    """One reviewable request: the queue draft plus admitted overrides, with its missing fields."""
    draft = row.get("request_draft")
    fields = row.get("request_fields")
    template_name = row.get("request_template")
    command = row.get("command_argv")
    if (type(draft) is not dict or type(fields) is not list or not fields
            or any(type(name) is not str for name in fields)
            or set(draft) - set(fields) or type(template_name) is not str
            or type(command) is not list or any(type(arg) is not str for arg in command)
            or command.count("{request_file}") != 1):
        raise ValueError("Queue needs current request schemas, templates and argv metadata; regenerate older queues")
    if identity(template_name) not in producer["inputs"]:
        raise ValueError("Queue producer did not bind its request template bytes")
    request_source = row.get("request_source")
    if type(request_source) is not str:
        raise ValueError("Queue row requires its producer request source identity")
    source_entry = identity(request_source)
    if source_entry not in producer["inputs"] or source_entry["sha256"] != row.get("request_source_sha256"):
        raise ValueError("Producer request dataclass changed; regenerate the queue")
    draft = json.loads(json.dumps(draft, allow_nan=False))
    if any(type(key) is not str for key in overrides) or set(overrides) - set(fields):
        raise ValueError(f"Unknown producer request field for {task_id}")
    protected = {"source_split", "episode_index", *row.get("request_bindings", {}),
                 *row.get("resolved_bindings", {})}
    planned = row.get("planned_parameters")
    if type(planned) is not dict:
        raise ValueError("Queue must declare its fixed planner parameters; regenerate older queues")
    protected |= {key for key, value in planned.items() if key in fields
                  and not (key == "parked_hand_run" and value is None)}
    if row["operation"] == "campaign.pin":
        protected |= {"run", "artifact_name", "tier"}
    for key, value in overrides.items():
        if key in protected and json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(
                draft.get(key), sort_keys=True, allow_nan=False):
            raise ValueError("Overrides cannot change task identity or bypass explicitly pinned dependency bindings")
        draft[key] = value
    if draft.get("source_split") != row["split"] or draft.get("episode_index") != row["episode_index"]:
        raise ValueError("Materialized request identity differs from the queue")
    missing = unresolved_fields(row["operation"], draft)
    if row["operation"] == "scene_assembly.assemble":
        names = row.get("object_names")
        if type(names) is not list or not names or any(type(name) is not str for name in names):
            raise ValueError("Scene draft requires the queue's authoritative object roster")
        missing = sorted(set(missing) | set(_scene_choices(draft, names)))
    dependencies = row.get("depends_on")
    if type(dependencies) is not list or any(type(stage) is not str for stage in dependencies):
        raise ValueError("Queue row requires unresolved dependency stage identities")
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    filename = f"requests/{index:04d}_{row['split']}_episode_{row['episode_index']:06d}_{digest}.json"
    return {"task_id": task_id, "stage": row["stage"], "split": row["split"],
            "episode_index": row["episode_index"], "operation": row["operation"],
            "request_file": filename, "request": draft, "depends_on": dependencies,
            "missing_fields": missing, "request_complete": not dependencies and not missing,
            "execution_authorized": False, "command_argv": command}


def materialize_requests(request: MaterializeRequest):
    runtime = json.loads((FEATURE / "runtime.json").read_text(encoding="utf-8-sig"))
    if runtime["execution"].get("materialize_authorized") is not True:
        raise PermissionError("Campaign request materialization remains disabled")
    permitted_root()
    if request.queue_run is None:
        raise ValueError("Supply an exact completed campaign.queue producer")
    queue_path, producer, queue_entry, queue_file, queue_artifact, document = _admit_queue(request)
    rows = document["queue"]
    ids = [_task_id(row) for row in rows]
    if len(set(ids)) != len(ids) or set(request.overrides) - set(ids):
        raise ValueError("Duplicate queue task or override for an unknown exact task ID")
    for path in (FEATURE / "request_drafts.py", FEATURE / "_planning.py"):
        if input_file(ROOT, path) not in producer.get("sources", []):
            raise ValueError("Queue request schema implementation changed; regenerate the queue")
    template_inputs = {}

    def identity(name):
        # Many rows share one template or dataclass source; hash each file once.
        if name not in template_inputs:
            template_inputs[name] = input_file(ROOT, project_path(ROOT, name))
        return template_inputs[name]

    prepared = [_prepare(index, task_id, row, request.overrides.get(task_id, {}), producer, identity)
                for index, (task_id, row) in enumerate(zip(ids, rows))]
    inputs = [queue_file, FEATURE / "runtime.json", *[ROOT / value for value in template_inputs]]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/campaign/materialize/{run_id}")
    record = run_fields(ROOT, "campaign", "materialize", asdict(request), inputs=inputs, parents=[queue_path],
                        allowed=("video_only",))
    record.update(schema_version=1, run_id=run_id, artifacts={},
                  model_execution=False, producers_launched=False, tests_run=False, validation_run=False,
                  official_evaluation=False, request_count=len(prepared))
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(directory / "run.json", record)
    try:
        manifest = {"schema_version": 1, "queue_run": queue_entry,
                    "queue_artifact": queue_artifact, "requests": [],
                    "unplanned_episodes": document.get("blocked", []),
                    "execution_authorized": False, "producers_launched": False}
        for row in prepared:
            destination = output_path(ROOT, directory / row["request_file"])
            write_json(destination, row.pop("request"))
            row["sha256"] = sha256_file(destination)
            relative = destination.relative_to(ROOT).as_posix()
            row["command_argv"] = [relative if arg == "{request_file}" else arg for arg in row["command_argv"]]
            manifest["requests"].append(row)
        manifest["complete_request_count"] = sum(row["request_complete"] for row in prepared)
        manifest["incomplete_request_count"] = len(prepared) - manifest["complete_request_count"]
        manifest["completion_scope"] = "Required fields and dependency bindings only; not runtime correctness or permission"
        write_json(directory / "requests.json", manifest)
        record.update(status="completed", manifest="requests.json",
                      complete_request_count=manifest["complete_request_count"],
                      incomplete_request_count=manifest["incomplete_request_count"])
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        record["artifacts"] = {path.relative_to(directory).as_posix(): sha256_file(path)
                               for path in directory.rglob("*") if path.is_file() and path.name != "run.json"}
        write_json(directory / "run.json", record)
    return directory / "requests.json"
