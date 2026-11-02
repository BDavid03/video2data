"""Mesh extents commands; authorization belongs to runtime.json."""

import argparse
from pathlib import Path

from .adopt import AdoptRequest, adopt
from .build import BuildRequest, build
from .measure import MeasureRequest, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("measure", "build", "adopt"))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    handler, request_type = {"measure": (measure, MeasureRequest), "build": (build, BuildRequest),
                             "adopt": (adopt, AdoptRequest)}[args.operation]
    print(handler(request_type.from_file(args.request)))


if __name__ == "__main__":
    main()
