"""Gates, run ownership and helpers for the depth_student slice (same run contract as learning_experiments)."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

from track3.paths import output_path
from track3.provenance import run_fields
from track3.runs import ROOT, SPLIT, new_run_id, read_json, recorded, runtime_flag, source_hashes  # noqa: F401
from track3 import runs

FEATURE = Path(__file__).resolve().parent
TRAIN_EPISODES = (0, 11, 12, 18, 21, 23, 26, 31)
TUNING_EPISODES = (39, 42)
HOLDOUT_EPISODES = (43, 44)


def authorize(operation):
    return runtime_flag(FEATURE, f"{operation}_authorized", f"depth_student {operation}")


def heavy_execution(operation):
    return runs.heavy_execution(authorize(operation))


def partition():
    from track3.features.learning_experiments._common import episode_partition
    return episode_partition(TRAIN_EPISODES, TUNING_EPISODES, HOLDOUT_EPISODES)


@contextmanager
def operation_run(operation, request, *, inputs=(), parents=()):
    authorize(operation)
    record = run_fields(ROOT, "depth_student", operation, json.loads(json.dumps(asdict(request), allow_nan=False)),
                        inputs=[*inputs, FEATURE / "runtime.json", SPLIT], parents=parents)
    record["source_hashes"] = source_hashes(FEATURE, extra=["track3/features/learning_experiments/data.py"])
    record["sources"] = [{"path": path, "sha256": digest} for path, digest in record["source_hashes"].items()]
    record.update(schema_version=1, run_id=new_run_id(), artifacts={}, tests_run=False, validation_run=False,
                  official_evaluation=False, research_only=True, submission_candidate=False, pretrained_weights=False)
    with recorded(output_path(ROOT, f"outputs/depth_student/{operation}/{record['run_id']}"), record) as result:
        yield result
