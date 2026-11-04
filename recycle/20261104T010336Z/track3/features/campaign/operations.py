"""Standard-library campaign inventory and explicit append-only run selection."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from uuid import uuid4

from track3.paths import permitted_root, output_path, project_path
from track3.provenance import (PROVENANCE_TAGS, input_file, read_run, run_fields, sha256_file,
                               write_json)

from ._planning import episode_graph, plan_episode
from .selections import entry_pins, inspection_tier, read_pin_history, role_pins, run_role

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
ROSTER = "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
PUBLIC_ROSTER = "reports/data-audit/trajectories.json"
PUBLIC_SPLIT = "configs/splits/public-v1.json"
STAGES = ("depth.generate", "stereo_depth.rectify", "stereo_depth.generate", "stereo_depth.anchor",
          "camera_motion.estimate", "object_masks.opening_clusters", "object_masks.generate",
          "hand_motion.estimate", "hand_motion.kinematics", "hand_motion.lift",
          "object_pose.initialise", "object_pose.track", "object_pose.compose",
          "object_mesh.resolve", "object_mesh.reconstruct", "scene_assembly.assemble",
          "retargeting.export", "retargeting.assets", "retargeting.retarget", "retargeting.support", "retargeting.inspect",
          "policy_training.compose", "policy_training.train", "policy_training.select",
          "policy_rollout.rollout", "policy_rollout.inspect", "policy_rollout.self_score")


def _path(value):
    return project_path(ROOT, value)


def _json(value):
    return json.loads(_path(value).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class CampaignRequest:
    source_split: str | None = None
    episode_index: int | None = None
    run: str | None = None
    artifact_name: str | None = None
    reason: str | None = None
    tier: str | None = None
    target_tier: str = "tracked"
    parked_hand_run: str | None = None
    submission_mesh_method: str = "opening_hull"

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if self.source_split not in (None, "public", "evaluation"):
            raise ValueError("Campaign split must be public, evaluation or null for both")
        if self.episode_index is not None and (type(self.episode_index) is not int or self.episode_index < 0 or self.episode_index == 41):
            raise ValueError("Episode must be nonnegative and episode 41 stays excluded")
        if self.target_tier not in ("resting", "tracked"):
            raise ValueError("Queue target must be resting or tracked")
        if self.submission_mesh_method not in ("opening_hull", "opening_tsdf"):
            raise ValueError("Select opening_hull or opening_tsdf for submission reconstruction")
        if self.parked_hand_run is not None and (not isinstance(self.parked_hand_run, str) or not self.parked_hand_run):
            raise ValueError("Name a completed parked-hand template run or leave it null")
        if self.tier not in (None, "resting", "tracked"):
            raise ValueError("An inspected rollout may be explicitly selected as resting or tracked")


def _authorize(operation):
    if _json(FEATURE / "runtime.json")["execution"].get(operation + "_authorized") is not True:
        raise PermissionError(f"Campaign {operation} remains disabled")
    permitted_root()


def _conformance(record):
    missing = [name for name in ("kind", "status", "request", "inputs", "sources", "provenance") if name not in record]
    if missing:
        return "missing fields: " + ", ".join(missing)
    if (not isinstance(record["kind"], str) or not record["kind"] or
            record["status"] not in ("running", "completed", "failed", "interrupted") or
            record["provenance"] not in PROVENANCE_TAGS or not isinstance(record["request"], dict)):
        return "invalid kind, status, request or provenance"
    for field in ("inputs", "sources"):
        if not isinstance(record[field], list):
            return f"{field} is not a path/hash list"
        for entry in record[field]:
            if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or
                    not isinstance(entry.get("sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is None):
                return f"invalid {field} path/hash entry"
            try:
                _path(entry["path"])
            except ValueError as error:
                return str(error)
    if not any(entry["path"] == "configs/sources.lock.json" for entry in record["sources"]):
        return "source-lock hash is missing"
    return None


def _roster():
    split = _json("configs/splits/public-v1.json")
    evaluation = _json(ROSTER)["episodes"]
    return {"evaluation": sorted(int(key) for key in evaluation if int(key) != 41),
            "public": sorted(index for index in split["development_episode_indices"] + split["validation_episode_indices"] if index != 41)}


def _public_inventory():
    """Read only identity metadata from the pinned public audit, never labels."""
    inventory = _json(PUBLIC_ROSTER)
    split = _json(PUBLIC_SPLIT)
    dataset = _json("configs/sources.lock.json")["dataset"]
    if not isinstance(inventory, dict) or not isinstance(split, dict) or not isinstance(dataset, dict):
        raise ValueError("Public roster, frozen split and locked dataset must be identity documents")
    if (dataset.get("immutable") is not True or dataset.get("track") != "track_3" or
            inventory.get("status") != "audited" or inventory.get("revision") != dataset["revision"] or
            split.get("dataset_revision") != dataset["revision"] or
            split.get("dataset_repo") != dataset["repo_id"] or split.get("source_split") != "public" or
            split.get("status") != "frozen_provisional_engineering_split_not_official"):
        raise ValueError("Public roster must match the locked dataset and frozen public split")
    development, validation = split["development_episode_indices"], split["validation_episode_indices"]
    for indices in (development, validation):
        if (not isinstance(indices, list) or any(type(index) is not int or index < 0 for index in indices) or
                len(indices) != len(set(indices))):
            raise ValueError("Frozen public split must contain unique nonnegative episode IDs")
    if set(development) & set(validation):
        raise ValueError("Frozen public development and validation episodes overlap")
    if not isinstance(inventory.get("episodes"), list) or any(not isinstance(row, dict) for row in inventory["episodes"]):
        raise ValueError("Public roster has no episode identity list")
    source = {**input_file(ROOT, PUBLIC_ROSTER), "dataset_revision": dataset["revision"],
              "dataset_repo": dataset["repo_id"], "source_split": "public",
              "frozen_split": input_file(ROOT, PUBLIC_SPLIT),
              "source_lock": input_file(ROOT, "configs/sources.lock.json"),
              "use": "Ordered object identities only; no poses, timestamps, labels, masses or support defaults"}
    return inventory["episodes"], set(development + validation) - {41}, source


def _safe_object_names(names):
    return (isinstance(names, list) and bool(names) and
            all(isinstance(name, str) and re.fullmatch(r"[a-z0-9_]+", name) for name in names) and
            len(names) == len(set(names)))


def _scan():
    records, problems = {}, []
    for folder in (ROOT / "outputs", ROOT / "runs"):
        for candidate in sorted(folder.rglob("run.json")):
            try:
                path = _path(candidate)
                record = _json(path)
                error = _conformance(record)
                relative = path.relative_to(ROOT).as_posix()
                if error:
                    problems.append({"path": relative, "error": error, "status": "non_conforming"})
                else:
                    records[relative] = record
            except (OSError, ValueError, TypeError) as error:
                problems.append({"path": str(candidate), "error": str(error), "status": "non_conforming"})
    return records, problems


def _key(value):
    return value["split"], value["episode_index"], value.get("artifact_name", value.get("kind"))


def _lineage_checker(records, selections):
    """Walk actual run paths, including unpinned ancestors; roles are not nodes."""
    memo, hashes = {}, {}

    def digest(path):
        if path not in hashes:
            try:
                hashes[path] = sha256_file(_path(path))
            except OSError:
                hashes[path] = None
        return hashes[path]

    def visit(path, trail=frozenset()):
        if path in trail:
            return [{"reason": "dependency_cycle", "run": path}]
        if path in memo:
            return memo[path]
        source = records.get(path)
        if source is None or source.get("status") != "completed" or digest(path) is None:
            return [{"reason": "run_missing_nonconforming_or_incomplete", "run": path}]
        reasons = []
        for entry in source["inputs"]:
            if not entry["path"].endswith("/run.json"):
                continue
            upstream = records.get(entry["path"])
            if digest(entry["path"]) != entry["sha256"]:
                reasons.append({"reason": "upstream_record_bytes_changed_or_missing", "input": entry["path"]})
            if upstream:
                key = (upstream.get("split"), upstream.get("episode_index"), upstream["kind"], run_role(upstream, records))
                current = selections.get(key)
                if current and (current["run"] != entry["path"] or current["sha256"] != entry["sha256"]):
                    reasons.append({"reason": "upstream_pin_changed", "input": entry["path"], "current": current["run"],
                                    "role": key[-1]})
            inherited = visit(entry["path"], trail | {path})
            if inherited:
                reasons.append({"reason": "upstream_stale", "input": entry["path"], "chain": inherited})
        memo[path] = reasons
        return reasons

    return visit, digest


def _stale(records, history):
    selections = role_pins(history, records)
    inspect, digest = _lineage_checker(records, selections)
    result = []
    for key, pin in sorted(selections.items()):
        reasons = list(inspect(pin["run"]))
        if digest(pin["run"]) != pin["sha256"]:
            reasons.append({"reason": "pinned_run_bytes_changed_or_missing", "run": pin["run"]})
        if reasons:
            result.append({"split": key[0], "episode_index": key[1], "artifact_name": key[2], "role": key[3],
                           "run": pin["run"], "reasons": reasons})
    return result


def _stage_grid(split, episode, candidates, records, pins, selections):
    grid = []
    for stage in sorted(set(STAGES) | {row["kind"] for row in candidates.values()}):
        completed = [(row.get("run_id", ""), path, row) for path, row in candidates.items()
                     if row["kind"] == stage and row["status"] == "completed"]
        newest = max(completed, default=None, key=lambda item: (item[0], item[1]))
        pinned = pins.get((split, episode, stage))
        source = records.get(pinned["run"]) if pinned else None
        grid.append({"stage": stage, "newest_completed": newest[1] if newest else None,
                     "pinned": pinned["run"] if pinned else None,
                     "role_pins": [{"role": key[3], "run": pin["run"]}
                                   for key, pin in selections.items() if key[:3] == (split, episode, stage)],
                     "status": source["status"] if source else ("missing" if pinned else "unpinned"),
                     "provenance": source["provenance"] if source else None,
                     "newest_provenance": newest[2]["provenance"] if newest else None})
    return grid


def _board(roster, records, pins, problems, stale, request, history):
    result = {"counts": {"none": 0, "resting": 0, "tracked": 0}, "episodes": [], "non_conforming": problems,
              "stale": stale, "selection_note": "Tier comes only from an explicit pin of a completed inspection run; newest never selects a run."}
    promotions = entry_pins(history)
    selections = role_pins(history, records)
    inspect, digest = _lineage_checker(records, selections)
    for split, episodes in roster.items():
        if request.source_split is not None and request.source_split != split:
            continue
        for episode in episodes:
            if request.episode_index is not None and request.episode_index != episode:
                continue
            candidates = {path: row for path, row in records.items() if row.get("split") == split and row.get("episode_index") == episode}
            grid = _stage_grid(split, episode, candidates, records, pins, selections)
            selected = promotions.get((split, episode, "policy_rollout.inspect"))
            state = "none"
            if selected and selected.get("tier") in ("resting", "tracked"):
                inspection = records.get(selected["run"])
                if (inspection and inspection["status"] == "completed" and
                        (split != "evaluation" or inspection["provenance"] == "video_only") and
                        inspection_tier(inspection) == selected["tier"] and not inspect(selected["run"]) and
                        digest(selected["run"]) == selected["sha256"]):
                    state = selected["tier"]
            result["episodes"].append({"split": split, "episode_index": episode, "state": state,
                                       "selected_entry": selected, "stages": grid})
            if split == "evaluation":
                result["counts"][state] += 1
    return result


def _parked_template(request, records):
    """The optional neutral MANO template run shared by every scene, as a project-relative path."""
    if request.parked_hand_run is None:
        return None
    path = _path(request.parked_hand_run)
    template = read_run(ROOT, path, kind="hand_motion.parked_template", allowed=("video_only",))
    if (template.get("scope") != "model_asset_template" or template.get("observed") is not False or
            template.get("split") is not None or template.get("episode_index") is not None or
            template.get("units") != "metres" or template.get("joint_order") != "MANO21" or
            template.get("joint_frame_convention") != "anatomical_unmirrored"):
        raise ValueError("Queue template must be a neutral MANO model asset, not an episode observation")
    parked_hand_run = path.relative_to(ROOT).as_posix()
    if parked_hand_run not in records:
        raise ValueError("Queue template must have a conforming project output run record")
    return parked_hand_run


def _priorities(roster):
    priorities = _json(FEATURE / "priorities.json")
    order = priorities["training_episode_indices"]
    if len(order) != len(set(order)) or any(type(value) is not int or value == 41 or value not in roster["evaluation"] for value in order):
        raise ValueError("Priorities must name unique allowed evaluation episodes")
    freeze = datetime.fromisoformat(priorities["freeze_utc"])
    if freeze.tzinfo is None:
        raise ValueError("Freeze time must include its UTC offset")
    budget = priorities["training_budget"]
    if budget not in _json(ROOT / "track3/features/policy_training/budget.json"):
        raise ValueError("Queue training budget is absent")
    return order, freeze, budget


def _timing_samples(records):
    """Completed wall times keyed by (kind, role, budget)."""
    samples = {}
    for row in records.values():
        seconds = row.get("timing", {}).get("wall_seconds")
        if row["status"] != "completed" or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            continue
        key = (row["kind"], run_role(row, records), row.get("request", {}).get("budget"))
        samples.setdefault(key, []).append(seconds)
    return samples


def _public_identities(roster_inputs):
    """Public identity rows, allowed episodes, source record and error; failures still record source bytes."""
    try:
        public_rows, public_episodes, public_source = _public_inventory()
        roster_inputs += [{"path": public_source["path"], "sha256": public_source["sha256"]},
                          public_source["frozen_split"], public_source["source_lock"]]
        return public_rows, public_episodes, public_source, None
    except (OSError, ValueError, TypeError, KeyError) as error:
        for source_path in (PUBLIC_ROSTER, PUBLIC_SPLIT, "configs/sources.lock.json"):
            try:
                roster_inputs.append(input_file(ROOT, source_path))
            except OSError:
                pass
        return [], set(), None, str(error)


def _episode_names(split, index, evaluation_objects, public, records):
    """Ordered object names for one episode, or a blocker describing why they are unavailable."""
    if split == "evaluation":
        names = evaluation_objects[str(index)]
    else:
        public_rows, public_episodes, _, public_error = public
        identities = [row for row in public_rows if type(row.get("episode_index")) is int and row["episode_index"] == index]
        if public_error or index not in public_episodes or len(identities) != 1:
            return None, {"split": split, "episode_index": index, "roster_path": PUBLIC_ROSTER,
                          "reason": public_error or "Public episode needs exactly one identity in the frozen roster"}
        identity = identities[0]
        names = identity.get("objects")
        if identity.get("path") != f"track_3/public/data/chunk-000/episode_{index:06d}.parquet":
            return None, {"split": split, "episode_index": index,
                          "roster_path": PUBLIC_ROSTER, "reason": "Public roster identity has a different source namespace or episode path"}
        conflicting = [path for path, row in records.items() if row.get("split") == split and
                       row.get("episode_index") == index and "object_names" in row and row["object_names"] != names]
        if conflicting:
            return None, {"split": split, "episode_index": index, "roster_path": PUBLIC_ROSTER,
                          "producer_runs": sorted(conflicting),
                          "reason": "Producer object_names differ from the public roster order or membership"}
    if not _safe_object_names(names):
        return None, {"split": split, "episode_index": index,
                      "roster_path": PUBLIC_ROSTER if split == "public" else ROSTER,
                      "reason": "Object roster must contain unique safe nonempty object IDs"}
    return names, None


def _annotate(item, samples, order, split, index, state, cpu_only):
    """Attach expected hours from completed runs and the scheduling sort key."""
    timings = [] if item["operation"] == "campaign.pin" else samples.get(
        (item["producer_kind"], item["phase"], item["budget"]), [])
    expected = statistics.median(timings) / 3600 if timings else None
    item.update(expected_wall_hours=expected, expected_gpu_hours=expected if item["gpu"] else None,
                timing_samples=len(timings), timing_basis="Median completed wall time for this role and budget")
    training = item["operation"] == "policy_training.train"
    priority = order.index(index) if split == "evaluation" and index in order else len(order) + index
    tracked = "tracked" in item["phase"] or item["producer_kind"] in (
        "object_masks.generate", "hand_motion.estimate", "hand_motion.kinematics", "hand_motion.lift", "object_pose.track")
    item["sort_key"] = [1 if tracked else 0, 0 if state == "none" else 1, 0 if cpu_only else 1,
                        2 if training else 1 if item["gpu"] else 0,
                        priority if training else expected if item["gpu"] and expected is not None else float("inf")]


def _schedule(rows):
    """Dependency-respecting order: lowest sort key first, preferring to stay in the current environment."""
    scheduled, done, environment = [], set(), None
    while rows:
        available = [item for item in rows if all((item["split"], item["episode_index"], dependency) in done
                                                 for dependency in item["depends_on"])]
        if not available:
            raise ValueError("Campaign task graph contains an unresolved dependency or cycle")
        selected = min(available, key=lambda item: (*item["sort_key"], item["environment"] != environment,
                                                   item["split"], item["episode_index"], item["stage"]))
        rows.remove(selected)
        done.add((selected["split"], selected["episode_index"], selected["stage"]))
        environment = selected["environment"]
        selected.pop("sort_key")
        scheduled.append(selected)
    return scheduled


def _queue(roster, records, pins, problems, stale, request, history):
    parked_hand_run = _parked_template(request, records)
    order, freeze, budget = _priorities(roster)
    samples = _timing_samples(records)
    selections = role_pins(history, records)
    inspect, digest = _lineage_checker(records, selections)
    board = _board(roster, records, pins, problems, stale, request, history)
    promotions = entry_pins(history)
    evaluation_objects = _json(ROSTER)["episodes"] if any(row["split"] == "evaluation" for row in board["episodes"]) else {}
    roster_inputs = [input_file(ROOT, ROSTER)] if evaluation_objects else []
    public = [], set(), None, None
    if any(row["split"] == "public" for row in board["episodes"]):
        public = _public_identities(roster_inputs)
    template_inputs = {}
    rows, blockers = [], []
    for episode in board["episodes"]:
        split, index, state = episode["split"], episode["episode_index"], episode["state"]
        names, blocker = _episode_names(split, index, evaluation_objects, public, records)
        if blocker:
            blockers.append(blocker)
            continue
        graph = episode_graph("wooden_spoon" in names, request.target_tier, budget, parked_hand_run,
                              request.submission_mesh_method)
        pending = plan_episode(graph, split, index, state, records, selections, promotions,
                               lambda path: not inspect(path), digest)
        cpu_only = not any(item["gpu"] for item in pending)
        for item in pending:
            item["object_names"] = list(names)
            for field in ("request_template", "request_source"):
                try:
                    entry = input_file(ROOT, _path(item[field]))
                    template_inputs[entry["path"]] = entry
                except (OSError, ValueError) as error:
                    blockers.append({"split": split, "episode_index": index, "stage": item["stage"],
                                     "reason": "Request template identity is unavailable: " + str(error)})
            _annotate(item, samples, order, split, index, state, cpu_only)
            rows.append(item)
    scheduled = _schedule(rows)
    return {"target_tier": request.target_tier, "submission_mesh_method": request.submission_mesh_method,
            "queue": scheduled, "blocked": blockers, "stale": stale,
            "roster_inputs": roster_inputs, "public_roster_source": public[2],
            "request_template_inputs": [template_inputs[path] for path in sorted(template_inputs)],
            "shared_inputs": {"parked_hand_template": {"run": parked_hand_run,
                "producer_kind": "hand_motion.parked_template", "environment": ".runtime/mano",
                "request_file": "track3/features/hand_motion/parked-template.json",
                "execution_authorized": False}},
            "non_conforming": problems,
            "hours_until_freeze": max(0, (freeze - datetime.now(timezone.utc)).total_seconds() / 3600),
            "execution": "Plan only. Request drafts are incomplete; inputs_selected describes pins, not permission. "
                         "Every row remains execution-disabled. Model/runtime provisioning and slice authorization "
                         "must be handled separately. Unknown timings remain null."}


def _validate_pin(request, roster):
    """Admit the explicitly named producer run before any campaign write."""
    if (request.source_split is None or request.episode_index not in roster[request.source_split] or
            request.run is None or not request.reason or not request.reason.strip()):
        raise ValueError("Pin requires split, episode, existing run and a concrete reason")
    path = _path(request.run)
    if not any(path.is_relative_to(ROOT / folder) for folder in ("outputs", "runs")) or path.name != "run.json":
        raise ValueError("Pins must name a project output run.json")
    source = read_run(ROOT, path, allowed=("video_only",) if request.source_split == "evaluation" else PROVENANCE_TAGS)
    error = _conformance(source)
    if error:
        raise ValueError("Cannot pin non-conforming run: " + error)
    if source.get("split") != request.source_split or source.get("episode_index") != request.episode_index or not source.get("run_id"):
        raise ValueError("Pin identity differs from the producer run")
    if request.artifact_name not in (None, source["kind"]):
        raise ValueError("artifact_name must equal the producer's slice.operation kind")
    if request.tier is not None and source["kind"] != "policy_rollout.inspect":
        raise ValueError("Only completed rollout inspections can be pinned with an entry tier")
    if request.tier is not None and inspection_tier(source) != request.tier:
        raise ValueError("Entry tier must match a nonblocking resting or trained rollout inspection")
    return path, source


def _append_pin(request, path, source, history, pins_path, directory, record):
    records, _ = _scan()
    relative = path.relative_to(ROOT).as_posix()
    if request.tier is not None:
        inspect, _ = _lineage_checker(records, role_pins(history, records))
        if inspect(relative):
            raise ValueError("Cannot promote an inspection with stale or incomplete lineage")
    entry = {"split": request.source_split, "episode_index": request.episode_index,
             "artifact_name": source["kind"], "run_id": source["run_id"], "run": relative,
             "sha256": sha256_file(path), "reason": request.reason.strip(), "tier": request.tier,
             "role": run_role(source, records), "time_utc": datetime.now(timezone.utc).isoformat()}
    with pins_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(directory / "pin.json", entry)
    record["artifacts"]["pin.json"] = sha256_file(directory / "pin.json")


def _write_report(directory, operation, report):
    """Write <operation>.json and a Markdown view; return their hashes."""
    write_json(directory / (operation + ".json"), report)
    lines = ["# Campaign " + operation, ""]
    if operation == "board":
        lines += ["Evaluation: " + ", ".join(f"{name}={count}" for name, count in report["counts"].items()), "",
                  "| Split | Episode | State | Stage | Newest completed | Pinned | Status | Provenance |",
                  "|---|---|---|---|---|---|---|---|"]
        for episode in report["episodes"]:
            for stage in episode["stages"]:
                lines.append(f"| {episode['split']} | {episode['episode_index']} | {episode['state']} | {stage['stage']} | {stage['newest_completed'] or ''} | {stage['pinned'] or ''} | {stage['status']} | {stage['provenance'] or ''} |")
    lines += ["", "```json", json.dumps(report, indent=2), "```", ""]
    (directory / (operation + ".md")).write_text("\n".join(lines), encoding="utf-8")
    return {name: sha256_file(directory / name) for name in (operation + ".json", operation + ".md")}


def campaign(operation, request: CampaignRequest):
    if operation not in ("board", "pin", "stale", "queue"):
        raise ValueError("Unknown campaign operation")
    _authorize(operation)
    roster = _roster()
    selected = _validate_pin(request, roster) if operation == "pin" else None
    inputs = [ROSTER, "configs/splits/public-v1.json", FEATURE / "runtime.json"]
    if operation == "queue":
        inputs += [FEATURE / "priorities.json", ROOT / "track3/features/policy_training/budget.json"]
    pins_path = output_path(ROOT, "outputs/campaign/pins.jsonl")
    record = run_fields(ROOT, "campaign", operation, asdict(request), inputs=inputs,
                        parents=[selected[0]] if selected else [])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/campaign/{operation}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    record.update(run_id=run_id, gpu=None, artifacts={})
    started = time.perf_counter()
    write_json(directory / "run.json", record)
    lock_path = output_path(ROOT, "outputs/campaign/pins.lock")
    owned = False
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            owned = True
            json.dump({"pid": os.getpid(), "operation": operation, "run_id": run_id}, stream)
        history, _ = read_pin_history(ROOT, pins_path)
        pins = {_key(entry): entry for entry in history}
        if pins_path.exists():
            record["inputs"].append(input_file(ROOT, pins_path))
        if operation == "pin":
            _append_pin(request, *selected, history, pins_path, directory, record)
        else:
            records, problems = _scan()
            # Run-record metadata only; each listed row carries its own run's provenance tag.
            own = (directory / "run.json").relative_to(ROOT).as_posix()
            for path in records:
                if path != own:
                    record["inputs"].append(input_file(ROOT, path))
            stale = _stale(records, history)
            if operation == "queue":
                report = _queue(roster, records, pins, problems, stale, request, history)
                for entry in report["roster_inputs"] + report["request_template_inputs"]:
                    if entry not in record["inputs"]:
                        record["inputs"].append(entry)
            elif operation == "board":
                report = _board(roster, records, pins, problems, stale, request, history)
            else:
                report = {"stale": stale, "non_conforming": problems}
            record["artifacts"] = _write_report(directory, operation, report)
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if owned:
            lock_path.unlink()
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        write_json(directory / "run.json", record)
    return directory / "run.json"
