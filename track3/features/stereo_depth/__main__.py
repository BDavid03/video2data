"""Stereo depth commands; authorization belongs to runtime.json."""

import argparse
from pathlib import Path

from . import StereoDepthRequest, anchor, generate, rectify
from .correspondences import correspondences
from .report import GeometryReportRequest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("rectify", "generate", "anchor", "correspondences", "report"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    request_type = GeometryReportRequest if args.operation == "report" else StereoDepthRequest
    print({"rectify": rectify, "generate": generate, "anchor": anchor,
           "correspondences": correspondences, "report": report}[args.operation](request_type.from_file(path)))


if __name__ == "__main__":
    main()
