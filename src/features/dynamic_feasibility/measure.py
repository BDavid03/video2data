"""Measure admitted task arrays and official exports using the existing CPU runtime."""

import argparse
import importlib.util
import json
from pathlib import Path

from src.provenance import sha256_file, write_json
from src.runs import heavy_execution, runtime_flag
from . import handler


def score_export(directory, task, kit):
    """Admit the official full rollout, preserve its clock and measure raw errors."""
    import numpy as np
    import pyarrow.parquet as pq
    from ._metrics import pose_metrics

    table = pq.read_table(directory / "trajectory.parquet")
    frame = table.to_pandas().sort_values(["reference_frame", "environment_id", "object_body_id"])
    metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    expected = len(task["reference"])
    if (len(frame) != expected or frame.duplicated(["reference_frame", "environment_id", "object_body_id"]).any()
            or not np.array_equal(frame.reference_frame, np.arange(expected))
            or not (frame.environment_id == 0).all() or not (frame.object_body_id == 0).all()
            or not (frame.object_body_name == "tissue_box_simple").all()):
        raise ValueError("Incomplete or incorrectly indexed example rollout")
    suffixes = [f"position_{axis}" for axis in "xyz"] + [f"quaternion_{axis}" for axis in "xyzw"]
    achieved, reference = (frame[[f"{prefix}_{suffix}" for suffix in suffixes]].to_numpy()[:, None, None]
                           for prefix in ("achieved", "reference"))
    if not np.allclose(reference, task["reference"], atol=1e-6, rtol=0):
        raise ValueError("Evaluator resampled or changed the task reference")
    official = json.loads((directory / "official_metrics.json").read_text())
    completed = (bool(frame.environment_valid.all()) and not official["non_finite_worlds"]
                 and official["reference_tracking"]["reference_end_fraction"] == 1.
                 and official["step_count"] == expected and official["sim_control_fps"] == float(task["fps"]))
    if metadata.get("flash_chord.eval_script_sha256") != sha256_file(handler.EVALUATOR):
        raise ValueError("Official evaluator script digest mismatch")
    metrics = pose_metrics(achieved, reference, task["vertices"], kit)
    return {**metrics, "completed": completed, "frames": len(frame), "evaluator_metadata": metadata,
            "official_position_cm": official["metrics"]["object_position_error_m"] * 100}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--campaign", choices=("D0", "D1"), default="D0")
    args = parser.parse_args()
    runtime = (handler.scaling_runtime("measure_authorized") if args.campaign == "D1" else
               runtime_flag(handler.FEATURE, "measure_authorized", "D0 CPU measurement"))
    heavy_execution({**runtime, "python": runtime["measurement_python"]})
    if not all(path.resolve().is_relative_to(handler.OUTPUT) for path in (args.task, args.out)):
        raise ValueError("Measurement paths must be inside the D0 output directory")
    trial_root = handler.SCALING_OUTPUT if args.campaign == "D1" else handler.OUTPUT
    if args.directory and not args.directory.resolve().is_relative_to(trial_root / "runs"):
        raise ValueError("Only the selected campaign trial exports may be measured")
    import numpy as np
    from ._metrics import pose_metrics
    spec = importlib.util.spec_from_file_location("dynamic_kit_auc", handler.KIT_METRIC)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    with np.load(args.task, allow_pickle=False) as data:
        task = dict(data)
    if args.directory:
        metrics = score_export(args.directory, task, kit)
    else:
        static = task["reference"][:1].repeat(len(task["reference"]), axis=0)
        metrics = pose_metrics(static, task["reference"], task["vertices"], kit)
    write_json(args.out, metrics)


if __name__ == "__main__":
    main()
