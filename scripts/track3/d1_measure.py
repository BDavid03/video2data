"""Run mesh_extents.measure for every episode in track3/features/mesh_extents/assignments.json.

    PYTHONPATH=. .runtime/linux-cuda/bin/python scripts/track3/d1_measure.py [--only split:episode ...] [--out runs.json]

Each entry names its depth run, frames, object -> frame-0 cluster ids and a note. Entries with
"objects": {} still run, so their overlay can be inspected before an assignment is recorded.
Writes the list of measure run.json paths (assigned entries only) to --out for the build step.
"""

import argparse
import json
from pathlib import Path

from track3.features.mesh_extents import MeasureRequest, measure

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "track3/features/mesh_extents/assignments.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-assign", action="store_true", help="measure clusters only, to re-check overlays")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())["episodes"]
    runs = []
    for key, entry in plan.items():
        if args.only and key not in args.only:
            continue
        split, episode = key.split(":")
        request = MeasureRequest(source_split=split, episode_index=int(episode), depth_run=entry["depth_run"],
                                 frames=entry.get("frames"), objects={} if args.no_assign else entry.get("objects", {}),
                                 assignment_note=entry.get("note"), **entry.get("overrides", {}))
        path = measure(request)
        record = json.loads(path.read_text())
        print(key, path.relative_to(ROOT), json.dumps({k: [round(x, 4) for x in v["median_m"]]
                                                        for k, v in record["objects"].items()}))
        if entry.get("objects"):
            runs.append(path.relative_to(ROOT).as_posix())
    if args.out:
        args.out.write_text(json.dumps(runs, indent=1) + "\n")


if __name__ == "__main__":
    main()
