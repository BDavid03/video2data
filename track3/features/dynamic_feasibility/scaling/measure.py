"""Reuse the unchanged D0 metric boundary on D1 official exports."""

import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    from track3.features.dynamic_feasibility import handler as d0
    from track3.features.dynamic_feasibility.measure import score_export
    from track3.features.dynamic_feasibility.scaling import handler
    from track3.runs import heavy_execution, runtime_flag
    from track3.provenance import write_json
    runtime = runtime_flag(handler.FEATURE, "measure_authorized", "D1 measurement")
    heavy_execution({**runtime, "python": runtime["measurement_python"]})
    if not args.directory.resolve().is_relative_to(handler.OUTPUT / "runs"):
        raise ValueError("Only D1 exports can enter D1 measurement")
    import numpy as np
    spec = importlib.util.spec_from_file_location("d1_kit_auc", d0.KIT_METRIC)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    with np.load(d0.OUTPUT / "task.npz", allow_pickle=False) as arrays:
        task = dict(arrays)
    write_json(args.directory / "metrics.json", score_export(args.directory, task, kit))


if __name__ == "__main__":
    main()
