"""Feature-local gates, bounds, episode policy and artifact persistence."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path

from track3.paths import output_path
from track3.provenance import run_fields, sha256_file, write_json
from track3.runs import ROOT, SPLIT, new_run_id, read_json, recorded, runtime_flag, source_hashes  # noqa: F401
from track3 import runs

FEATURE = Path(__file__).resolve().parent
REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
DEVELOPMENT = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42, 43, 44)
TECHNIQUES = ("ann", "cnn", "transformer", "gradient_boosting")
OPERATIONS = {"plan", "model", "train", "predict", "holdout_predict", "boost_train", "boost_predict", "score", "tune",
              "ensemble", "execute", "investigate", "report", "geometry_selection", "geometry_freeze"}


def authorize(operation):
    if operation not in OPERATIONS:
        raise ValueError("Unknown research operation")
    return runtime_flag(FEATURE, f"{operation}_authorized", f"learning_experiments {operation}")


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def number(value, name, minimum, maximum):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return float(value)


def episode_partition(train, tuning, holdout):
    split = read_json(SPLIT)
    if (split["dataset_revision"] != REVISION
            or split["split_id"] != "track3-public-dev13-val7-v1"
            or tuple(e for e in split["development_episode_indices"] if e != 41) != DEVELOPMENT):
        raise ValueError("Frozen development policy changed; explicit design revision required")
    groups = (("train", train), ("tuning", tuning), ("holdout", holdout))
    for name, group in groups:
        if (not isinstance(group, (tuple, list)) or not 1 <= len(group) <= len(DEVELOPMENT)
                or any(type(e) is not int or e not in DEVELOPMENT for e in group)
                or len(set(group)) != len(group)):
            raise ValueError(f"{name} must contain unique frozen development episodes, excluding 41")
    if any(set(left) & set(right) for left, right in
           ((train, tuning), (train, holdout), (tuning, holdout))):
        raise ValueError("Training, tuning and final holdout episodes must be disjoint")
    return {"split_id": split["split_id"], "dataset_revision": REVISION,
            "train_episodes": sorted(train), "tuning_episodes": sorted(tuning),
            "holdout_episodes": sorted(holdout),
            "selection_objective": "tuning_only_log_depth_mae",
            "holdout_used_for_selection": False,
            "excluded_episodes": [41], "group_unit": "episode_all_frames_objects_streams_derivatives",
            "official_validation_used": False}


def artifact(value):
    return output_path(ROOT, value)


def heavy_execution(operation):
    """Gate before torch imports; caches and temp files stay inside the checkout."""
    authorize("model")
    return runs.heavy_execution(authorize(operation))


def _record(operation, request, inputs=(), parents=()):
    record = run_fields(ROOT, "learning_experiments", operation, json.loads(json.dumps(asdict(request), allow_nan=False)),
                        inputs=[*inputs, FEATURE / "runtime.json", SPLIT], parents=parents)
    record.update(schema_version=1, run_id=new_run_id(), artifacts={}, tests_run=False, validation_run=False,
                  official_evaluation=False)
    return record


@contextmanager
def operation_run(operation, request, *, inputs=(), parents=()):
    """Own a research operation's source/input identity and partial artifacts."""
    authorize(operation)
    record = _record(operation, request, inputs, parents)
    record["source_hashes"] = source_hashes(FEATURE, suffixes=(".py", ".json", ".md", ".cpp", ".cu"))
    record["sources"] = [{"path": path, "sha256": digest} for path, digest in record["source_hashes"].items()]
    record.update(research_only=True, submission_candidate=False, scratch_checkpoint_admitted=False)
    with recorded(artifact(f"outputs/learning_experiments/{operation}/{record['run_id']}"), record) as result:
        yield result


def persist_plan(request, manifest):
    authorize("plan")
    record = _record("plan", request, inputs=[FEATURE / "search.json"])
    record["sources"] += [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)}
                          for p in sorted(FEATURE.glob("*")) if p.suffix in {".json", ".md"}]
    record.update(model_execution=False, training_executed=False, prediction_executed=False, best_model=None, scores=None)
    directory = artifact(f"outputs/learning_experiments/plan/{record['run_id']}")
    with recorded(directory, record):
        write_json(directory / "plan.json", manifest)
    return directory / "plan.json"
