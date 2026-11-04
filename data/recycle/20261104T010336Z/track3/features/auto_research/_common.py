"""Feature-local gates, state locations and small helpers for the research loop."""

import hashlib
import json
from pathlib import Path

from track3.paths import output_path
from track3.provenance import write_json
from track3.runs import ROOT, new_run_id, now_utc as now, read_json, runtime_flag  # noqa: F401
from track3 import runs

FEATURE = Path(__file__).resolve().parent
STATE = "outputs/auto_research"
TECHNIQUES = ("ann", "cnn", "transformer", "gradient_boosting", "scratch", "student")
TRAIN_EPISODES = (0, 11, 12, 18, 21, 23, 26, 31)
TUNING_EPISODES = (39, 42)
HOLDOUT_EPISODES = (43, 44)


def authorize(operation):
    return runtime_flag(FEATURE, f"{operation}_authorized", f"auto_research {operation}")


def heavy_execution(operation):
    return runs.heavy_execution(authorize(operation))


def artifact(value):
    return output_path(ROOT, value)


def state_dir():
    path = artifact(STATE)
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    write_json(path, value)
    return path
