"""Freeze a matched research roster without launching composition or simulation."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from track3.paths import output_path
from track3.provenance import input_file, write_json, run_fields
from track3.runs import now_utc, runtime_flag
from ._common import FEATURE, ROOT, producer, read_json, PolicyRequest
from .checkpoints import _canonical_sha256 as _digest


@dataclass(frozen=True)
class ComparisonRequest:
    # Ordered (episode, resting inspection, tracked inspection) pairs, with no outcomes.
    references: tuple[tuple[int, str, str], ...]

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if tuple(row[0] for row in self.references) != (39, 42):
            raise ValueError("This fixed development protocol requires episodes 39/42 in order")
        if any(len(row) != 3 or any(not isinstance(path, str) or not path for path in row[1:])
               for row in self.references):
            raise ValueError("Name immutable resting and tracked inspection runs")


def build_comparison(request):
    """Pure source plan; no producer reads, writes, outcomes or simulator imports."""
    attempts = []
    for episode, resting, tracked in request.references:
        for arm in ("resting", "zero_update_replay"):
            attempts.append(_attempt(episode, 31, arm, "baseline", resting if arm == "resting" else tracked))
        for seed in (31, 47, 73):
            for arm in ("motion_only", "contact_aware"):
                attempts.append(_attempt(episode, seed, arm, "feasibility" if seed == 31 else "extension", tracked))
    payload = {
        "schema": "policy_comparison_v1", "source_split": "public", "attempts": attempts,
        "episode_indices": [39, 42], "excluded_episode_indices": [41, 43, 44],
        "maximum_environment_steps": 1600000, "planned_environment_steps": 1536256,
        "maximum_gpu_wall_seconds": 7200, "world_count": 64, "evaluation_world_count": 1,
        "control_fps": 20.0, "motion_speed": 1.0, "reset_frame": 0,
        "checkpoint_rule": "final_scheduled_only_no_fallback",
        "extension_condition": "All seed31 attempts complete without admission or resource failure; independent of scores",
        "stopping_rule": "Stop at either aggregate ceiling, admission failure or memory failure; no retries or installation",
        "primary_metric": "Pinned Submission Kit MPPE; full fixed roster denominator",
        "secondary_metrics": ["AUC", "SP-SR", "MP-SR", "RPE"],
        "holdout_eligibility": "43/44 consumed; no untouched final assessment in this protocol",
        "assistance_rule": "No achieved learning claim unless final scheduled checkpoint reaches zero-assistance stages",
        "existing_policy": "Unavailable; no immutable existing checkpoint supplied",
    }
    payload["protocol_sha256"] = _digest(payload)
    return payload


def _attempt(episode, seed, arm, stage, reference):
    trained = arm in ("motion_only", "contact_aware")
    return {
        "attempt_id": f"{arm}-ep{episode}-seed{seed}", "stage": stage, "arm": arm,
        "source_split": "public", "episode_index": episode, "seed": seed,
        "retargeting_run": reference, "budget": "contact_feasibility" if trained else "zero_update",
        "policy_kind": "trained" if trained else ("resting" if arm == "resting" else "replay"),
        "recipe_variant": arm if trained else "contact_aware",
        "total_environment_steps": 128000 if trained else 64,
        "checkpoint_environment_steps": 128000 if trained else 64,
        "final_checkpoint_only": True,
    }


def persist_comparison(request):
    """Disabled planning persistence; never composes a job or launches a run."""
    runtime_flag(FEATURE, "comparison_authorized", "Policy comparison planning persistence")
    directory = output_path(ROOT, "runs/policy-comparisons/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    protocol = build_comparison(request)
    pins, scene_pins, tracked_scene_runs, parents = [], [], {}, []
    for episode, resting, tracked in request.references:
        scope = PolicyRequest(episode_index=episode)
        scene_pair = {"episode_index": episode}
        for path, kind in ((resting, "resting"), (tracked, "tracked")):
            parent = producer(path, "retargeting.inspect", scope)
            report_path = parent[0].parent / "inspect.json"
            report = read_json(report_path)
            if (parent[1].get("status") != "completed" or report.get("blocking") is not False
                    or report.get("scene_mode") != ("full" if kind == "tracked" else "resting")
                    or report.get("fps") != 20.0):
                raise ValueError("Comparison needs completed admitted 20 Hz references")
            report_pin = input_file(ROOT, report_path)
            if report_pin["sha256"] != parent[1]["artifacts"]["inspect.json"]:
                raise ValueError("Changed retargeting inspection")
            run_pin = input_file(ROOT, parent[0])
            pins.append({"episode_index": episode, "kind": kind, "run": run_pin, "report": report_pin})
            parents.append(parent[0])
            scene_pair[kind + "_inspection"] = run_pin
            scene = producer(report["scene_run"]["path"], "scene_assembly.assemble", scope)
            if input_file(ROOT, scene[0]) != report["scene_run"]:
                raise ValueError("Changed frozen scene")
            parents.append(scene[0])
            if kind == "tracked":
                tracked_scene_runs[str(episode)] = report["scene_run"]
        scene_pins.append(scene_pair)
    protocol.update(reference_pins=pins, scene_pins=scene_pins, tracked_scene_runs=tracked_scene_runs,
                    source_pins=[input_file(ROOT, path) for path in
                                 [*sorted(FEATURE.rglob("*.py")), FEATURE / "budget.json", FEATURE / "research-protocol.md"]])
    # Receipt hash covers admitted inputs, source and budget identities.
    protocol["freeze_sha256"] = _digest(protocol)
    protocol["frozen_at"] = protocol["created_at"] = now_utc()
    record = run_fields(ROOT, "policy_training", "comparison", asdict(request), parents=parents,
                        inputs=[FEATURE / "budget.json", FEATURE / "runtime.json"])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol)
    record.update({"status": "completed", "protocol_sha256": protocol["protocol_sha256"],
        "freeze_sha256": protocol["freeze_sha256"], "frozen_at": protocol["frozen_at"],
        "created_at": protocol["created_at"],
        "artifacts": {"protocol.json": input_file(ROOT, directory / "protocol.json")["sha256"]},
        "execution_scope": "Plan persisted only; no composition, training or simulation"})
    write_json(directory / "run.json", record)
    return directory / "run.json"
