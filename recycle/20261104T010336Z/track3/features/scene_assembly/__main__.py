"""Scene assembly; authorization belongs to runtime.json."""

import argparse
from pathlib import Path

from . import (AssembleSceneRequest, RefineSceneRequest, PrepareRefinementRequest,
               assemble, refine, prepare_refinement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("assemble", "refine", "prepare_refinement"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    request_path = args.request or Path(__file__).with_name(args.operation + ".json")
    request_type, handler = {"assemble": (AssembleSceneRequest, assemble),
                             "refine": (RefineSceneRequest, refine),
                             "prepare_refinement": (PrepareRefinementRequest, prepare_refinement)}[args.operation]
    print(handler(request_type.from_file(request_path)))


if __name__ == "__main__":
    main()
