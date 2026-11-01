"""Offline achieved-rollout comparison of pinned-kit self-scores against a frozen plan; never simulates."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math

from track3.paths import output_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from ._common import FEATURE, KIT, ROOT, local_path, new_run_id, read_json, runtime_flag

METRICS = ("mppe_cm", "add_auc", "spider_sr", "maniptrans_sr", "rpe_cm")
RECIPE_KEYS = ("seed", "budget", "policy_kind", "recipe_variant")


@dataclass(frozen=True)
class AchievedComparisonRequest:
    protocol_run: str
    outcomes: tuple

    @classmethod
    def from_file(cls, path):
        value = read_json(path)
        return cls(protocol_run=value["protocol_run"], outcomes=tuple(value["outcomes"]))


def _entry(entry):
    path = local_path(entry["path"])
    if input_file(ROOT, path) != entry:
        raise ValueError("A pinned comparison input changed")
    return path


def _run(entry, kind):
    path = _entry(entry)
    record = read_run(ROOT, path, kind=kind, allowed=("video_only",))
    if record.get("episode_index") in (41, 43, 44):
        raise PermissionError("Excluded and consumed holdout episodes cannot enter this comparison")
    return path, record


def _report(parent, name):
    path = parent[0].parent / name
    if sha256_file(path) != parent[1]["artifacts"].get(name):
        raise ValueError("Comparison producer artifact changed")
    return read_json(path)


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Comparison timestamps require a time zone")
    return parsed


def _started(record):
    return datetime.strptime(record["run_id"].split("-")[0], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)


def _mean(values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _recipe_mismatch(settings, arm):
    """First recipe key whose training setting differs from the frozen arm, else None."""
    return next((key for key in RECIPE_KEYS
                 if settings.get(key, "contact_aware" if key == "recipe_variant" else None) != arm[key]), None)


def _events(definitions, scores, episode):
    """Attribute frozen intervals without inventing contact/loss measurements."""
    result = []
    count = scores["coverage"]["frame_count"]
    for event in definitions:
        producer = read_run(ROOT, _entry(event["producer"]), allowed=("video_only",))
        if producer.get("episode_index") != episode or producer.get("split") != "public":
            raise ValueError("Event source belongs to another episode or provenance channel")
        start, stop = event["start_frame"], event["stop_frame_exclusive"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= count:
            raise ValueError("Event interval is outside the full reference clock")
        names = event.get("object_names", scores["coverage"]["object_names"])
        if not names or not set(names) <= set(scores["per_object"]):
            raise ValueError("Event names must belong to the complete object roster")
        values = [scores["frame_mppe_cm"][name][frame]
                  for name in names for frame in range(start, stop)]
        result.append({**event, "mppe_cm": _mean(values), "expected_object_frames": len(values),
                       "scope": "Attributed interval; alignment performed once at full-attempt start"})
    return result


def _score(expected, outcome, frozen_at, scene_pin, source_pins):
    score = _run(outcome["self_score_run"], "policy_rollout.self_score")
    report = _report(score, "self_score.json")
    manifest = read_json(KIT / "KIT_MANIFEST.json")
    for name in ("v2dlb/object_pose.py", "v2dlb/vertices.py"):
        pin = input_file(ROOT, KIT / name)
        if pin["sha256"] != manifest["files"][name] or pin not in score[1]["sources"]:
            raise ValueError("Score did not use the pinned official metric/vertex implementation")
    if report["align_initial_pose"] is not True or set(report["metrics"]) != set(METRICS):
        raise ValueError("Score alignment or metric channel differs from the protocol")
    inspection = _run(report["inspection_run"], "policy_rollout.inspect")
    inspected = _report(inspection, "inspect.json")
    rollout = _run(report["rollout_run"], "policy_rollout.rollout")
    training = _run(inspected["training_run"], "policy_training.train")
    episode = expected["episode_index"]
    for _, record in (score, inspection, rollout, training):
        if record.get("split") != "public" or record.get("episode_index") != episode:
            raise ValueError("Achieved comparison inputs differ in split or episode")
    if _started(rollout[1]) <= frozen_at or _started(training[1]) <= frozen_at:
        raise ValueError("Freeze the protocol before producing training or achieved rollout outcomes")
    training_inputs = training[1]["sources"] + training[1]["inputs"]
    for pin in source_pins:
        if (pin["path"].endswith(".py") or pin["path"].endswith("/budget.json")) and pin not in training_inputs:
            raise ValueError("Training source or budget differs from its frozen identity")
    if (inspected["blocking"] or inspected["rollout_run"] != report["rollout_run"] or
            rollout[1]["training_run"] != inspected["training_run"] or
            rollout[1]["checkpoint"] != inspected["checkpoint"] or report["scene_run"] != scene_pin):
        raise ValueError("Achieved comparison lineage or common reference differs")
    _run(scene_pin, "scene_assembly.assemble")
    _entry(inspected["pose_file"])
    _entry(inspected["checkpoint"])
    key = _recipe_mismatch(training[1]["request"], expected)
    if key is not None:
        raise ValueError("Achieved training recipe differs from the frozen arm: " + key)
    if training[1]["retargeting_run"] != input_file(ROOT, local_path(expected["retargeting_run"])):
        raise ValueError("Achieved arm used a different frozen reconstructed reference")
    admission = rollout[1]["checkpoint_admission"]
    if admission["environment_steps"] != expected["checkpoint_environment_steps"]:
        raise ValueError("Only the predeclared final scheduled checkpoint is eligible")
    budget = _report(training, "budget.json")
    if budget["total_environment_steps"] != expected["total_environment_steps"]:
        raise ValueError("Achieved arm exceeds or differs from its matched frozen budget")
    if expected["arm"] in ("resting", "zero_update_replay") and admission["global_update_step"] != 0:
        raise ValueError("The zero-update baseline contains optimizer updates")
    coverage = report["coverage"]
    if (coverage["frame_count"] != inspected["frame_count"] or
            coverage["object_names"] != inspected["object_names"] or
            coverage["expected_poses"] != inspected["frame_count"] * len(inspected["object_names"]) or
            coverage["achieved_poses"] != coverage["expected_poses"] or
            set(report["per_object"]) != set(inspected["object_names"])):
        raise ValueError("A score is missing full-roster achieved coverage")
    for name in coverage["object_names"]:
        values = report["frame_mppe_cm"][name]
        if len(values) != coverage["frame_count"] or any(not math.isfinite(value) for value in values):
            raise ValueError("Incomplete frame attribution; rerun eligible self-score explicitly")
    return {"self_score_run": outcome["self_score_run"], "achieved_rollout": report["rollout_run"],
            "pose_file": inspected["pose_file"], "checkpoint": inspected["checkpoint"],
            "metrics": report["metrics"], "per_object": report["per_object"], "coverage": coverage,
            "left_working_area": report["left_working_area"],
            "filled_reference_poses": report["filled_reference_poses"],
            "resource_evidence": {"environment_steps": admission["environment_steps"],
                "training_wall_seconds": training[1]["timing"]["wall_seconds"],
                "rollout_wall_seconds": rollout[1]["timing"]["wall_seconds"],
                "training_gpu": training[1].get("gpu"), "rollout_gpu": rollout[1].get("gpu")},
            "events": _events(expected.get("events", []), report, episode),
            "event_limits": "Contact loss/drop/recovery require frozen, producer-attributed intervals; none inferred from MPPE"}


def _frozen_plan(request, runtime):
    """Admit the comparison protocol run and its unchanged freeze receipt."""
    protocol_path = local_path(request.protocol_run)
    protocol = (protocol_path, read_run(ROOT, protocol_path, kind="policy_training.comparison", allowed=("video_only",)))
    plan = _report(protocol, "protocol.json")
    receipt = {key: value for key, value in plan.items() if key not in ("freeze_sha256", "created_at", "frozen_at")}
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != plan["freeze_sha256"] or digest != protocol[1]["freeze_sha256"]:
        raise ValueError("Frozen comparison receipt changed")
    for pin in plan["source_pins"]:
        _entry(pin)
    frozen_at = _time(plan["created_at"])
    episodes = {row["episode_index"] for row in plan["attempts"]}
    if not episodes or not episodes <= {39, 42}:
        raise PermissionError("Only the frozen two tuning episodes are eligible")
    if not episodes <= set(runtime["authorized_episode_indices"]["public"]):
        raise PermissionError("Explicit comparison episode authorization is required")
    return protocol_path, plan, frozen_at, episodes


def _rosters(plan, episodes):
    rosters = {}
    for episode in episodes:
        scene = _run(plan["tracked_scene_runs"][str(episode)], "scene_assembly.assemble")
        document = _report(scene, "scene.json")
        names = [item["name"] for item in document["objects"]]
        if document["mode"] != "full" or not names or document["frame_count"] <= 0:
            raise ValueError("Frozen comparison requires a full tracked scene roster")
        rosters[episode] = {"frame_count": document["frame_count"], "object_names": names,
                            "expected_poses": document["frame_count"] * len(names), "achieved_poses": 0}
    return rosters


def _admit_failure(failure, arm, frozen_at):
    """Bind an explicit failed/interrupted run to the frozen arm's recipe and reference."""
    failed = read_json(_entry(failure))
    if (failed.get("status") not in ("failed", "interrupted") or failed.get("provenance") != "video_only" or
            failed.get("episode_index") != arm["episode_index"] or
            failed.get("kind") not in ("policy_rollout.rollout", "policy_training.train") or
            _started(failed) <= frozen_at):
        raise ValueError("Failure evidence does not belong to an eligible frozen attempt")
    training = failed
    if failed["kind"] == "policy_rollout.rollout":
        training_pin = failed.get("training_run") or input_file(ROOT, local_path(failed["request"]["training_run"]))
        training = _run(training_pin, "policy_training.train")[1]
    settings = training["request"]
    if _recipe_mismatch(settings, arm) is not None:
        raise ValueError("Failure belongs to another recipe or seed")
    reference_pin = training.get("retargeting_run")
    if reference_pin is None:
        composed = _run(input_file(ROOT, local_path(settings["compose_run"])), "policy_training.compose")
        reference_pin = composed[1]["retargeting_run"]
    if reference_pin != input_file(ROOT, local_path(arm["retargeting_run"])):
        raise ValueError("Failure belongs to another retargeting reference")


def _summary(rows):
    """Per-arm counts and completion-conditional means; the denominator is always the full plan."""
    summary = {}
    for arm in dict.fromkeys(row["arm"] for row in rows):
        selected = [row for row in rows if row["arm"] == arm]
        valid = [row for row in selected if row["status"] == "completed"]
        summary[arm] = {"planned_attempts": len(selected), "completed_attempts": len(valid),
                        "failed_attempts": sum(row["status"] == "failed" for row in selected),
                        "unassessed_attempts": sum(row["status"] == "unassessed" for row in selected),
                        "expected_object_frames": sum(row["coverage"]["expected_poses"] for row in selected),
                        "achieved_object_frames": sum(row["coverage"]["achieved_poses"] for row in selected),
                        "conditional_metrics": {key: _mean([row["metrics"][key] for row in valid]) for key in METRICS},
                        "metric_attempt_counts": {key: sum(row["metrics"][key] is not None for row in valid) for key in METRICS}}
    return summary


def _matched_pairs(rows, episodes):
    pairs = []
    for episode in sorted(episodes):
        for seed in (31, 47, 73):
            by_arm = {row["arm"]: row for row in rows
                      if row["episode_index"] == episode and row["seed"] == seed and row["status"] == "completed"}
            if {"motion_only", "contact_aware"} <= by_arm.keys():
                pairs.append({"episode_index": episode, "seed": seed,
                              "motion_only_mppe_cm": by_arm["motion_only"]["metrics"]["mppe_cm"],
                              "contact_aware_mppe_cm": by_arm["contact_aware"]["metrics"]["mppe_cm"]})
    return pairs


def compare_achieved(request):
    """Aggregate completed attempts and explicit failures against a frozen plan."""
    runtime = runtime_flag(FEATURE, "compare_achieved_authorized", "Achieved comparison")
    protocol_path, plan, frozen_at, episodes = _frozen_plan(request, runtime)
    expected = plan["attempts"]
    rosters = _rosters(plan, episodes)
    admitted = {row["attempt_id"]: row for row in expected}
    outcomes = {row["attempt_id"]: row for row in request.outcomes}
    if len(admitted) != len(expected) or len(outcomes) != len(request.outcomes) or admitted.keys() != outcomes.keys():
        raise ValueError("Supply exactly one outcome/disposition for every frozen attempt")
    parents = [protocol_path]
    failure_inputs = []
    rows = []
    for identity, arm in admitted.items():
        outcome = outcomes[identity]
        row = {"attempt_id": identity, "arm": arm["arm"], "episode_index": arm["episode_index"],
               "seed": arm["seed"], "stage": arm["stage"], "coverage": rosters[arm["episode_index"]]}
        if outcome.get("self_score_run") is not None:
            if outcome.get("failure_run") is not None or outcome.get("failure_reason"):
                raise ValueError("A completed score cannot simultaneously be a failure")
            row.update(status="completed", **_score(arm, outcome, frozen_at,
                       plan["tracked_scene_runs"][str(arm["episode_index"])], plan["source_pins"]))
            if any(row["coverage"][key] != rosters[arm["episode_index"]][key]
                   for key in ("frame_count", "object_names", "expected_poses")):
                raise ValueError("Achieved score and frozen reference rosters differ")
            parents.append(_entry(outcome["self_score_run"]))
        else:
            if not outcome.get("failure_reason"):
                raise ValueError("Every unscored attempt requires an explicit blocker/failure reason")
            failure = outcome.get("failure_run")
            if failure is not None:
                _admit_failure(failure, arm, frozen_at)
                failure_inputs.append(failure)
            row.update(status="failed" if failure else "unassessed", failure_reason=outcome["failure_reason"], failure_run=failure)
        rows.append(row)
    early = [row for row in rows if row["stage"] in ("baseline", "feasibility")]
    if any(row["stage"] == "extension" and row["status"] != "unassessed" for row in rows):
        if any(row["status"] != "completed" for row in early):
            raise ValueError("Extension attempts require every initial-stage attempt to complete")
    summary = _summary(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    directory = output_path(ROOT, "runs/rollout/comparisons/" + new_run_id())
    record = run_fields(ROOT, "policy_rollout", "compare_achieved", asdict(request), parents=parents,
                        inputs=[FEATURE / "runtime.json"], allowed=("video_only",))
    record["inputs"].extend(failure_inputs)
    report = {"protocol_run": input_file(ROOT, protocol_path), "scope": "Reconstruction self-score; not public-label or leaderboard evidence",
              "primary_metric": "mppe_cm from pinned Submission Kit; mean over objects", "attempts": rows,
              "summary": summary, "matched_objective_pairs": _matched_pairs(rows, episodes),
              "worst_completed_attempts": [row["attempt_id"] for row in sorted(
                  [row for row in completed if row["metrics"]["mppe_cm"] is not None],
                  key=lambda row: row["metrics"]["mppe_cm"], reverse=True)],
              "public_diagnostic_export": [
                  {key: row[key] for key in ("attempt_id", "episode_index", "seed", "pose_file", "checkpoint")}
                  for row in completed],
              "resource_limits": {"maximum_environment_steps": plan["maximum_environment_steps"],
                  "maximum_gpu_wall_seconds": plan["maximum_gpu_wall_seconds"],
                  "completed_training_steps": sum(row["resource_evidence"]["environment_steps"] for row in completed),
                  "completed_training_rollout_wall_seconds": sum(
                      row["resource_evidence"]["training_wall_seconds"] + row["resource_evidence"]["rollout_wall_seconds"]
                      for row in completed),
                  "scope": "Completed-attempt costs only; failed/partial work and composition need their original records. This report is not a simulation budget controller."},
              "limitations": ["Only two tuning episodes; no untouched final assessment", "Frame zero is after the first control step",
                              "Conditional means do not impute failed attempts", "Event coverage depends on frozen attributed intervals"]}
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "comparison.json", report)
    record.update(status="completed", artifacts={"comparison.json": sha256_file(directory / "comparison.json")})
    write_json(directory / "run.json", record)
    return directory / "run.json"
