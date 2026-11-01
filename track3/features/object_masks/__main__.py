"""Full-video and opening-cluster masks; authorization belongs to runtime.json."""

import argparse
from pathlib import Path

from . import GenerateMasksRequest, OpeningClustersRequest, generate, opening_clusters
from . import CorrespondencesRequest, correspondences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "opening-clusters", "correspondences"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    request_type, handler = {"generate": (GenerateMasksRequest, generate),
                             "opening-clusters": (OpeningClustersRequest, opening_clusters),
                             "correspondences": (CorrespondencesRequest, correspondences)}[args.operation]
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    print(handler(request_type.from_file(path)))


if __name__ == "__main__":
    main()
