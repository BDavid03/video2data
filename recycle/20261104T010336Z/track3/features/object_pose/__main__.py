"""Geometric object pose commands; authorization belongs to runtime.json."""

import argparse
from pathlib import Path

from . import ObjectPoseRequest, PoseRefinementRequest, compose, initialise, track, refine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("initialise", "track", "compose", "refine"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    request_type = PoseRefinementRequest if args.operation == "refine" else ObjectPoseRequest
    print({"initialise": initialise, "track": track, "compose": compose, "refine": refine}[args.operation](request_type.from_file(path)))


if __name__ == "__main__":
    main()
