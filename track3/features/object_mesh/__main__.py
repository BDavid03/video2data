"""Object meshes; no registration, learned model or authorization bypass."""

import argparse
from pathlib import Path

from . import ObjectMeshRequest, MovingFusionRequest, reconstruct, resolve, fuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("resolve", "reconstruct", "fuse"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    request_type = MovingFusionRequest if args.operation == "fuse" else ObjectMeshRequest
    print({"resolve": resolve, "reconstruct": reconstruct, "fuse": fuse}[args.operation](request_type.from_file(path)))


if __name__ == "__main__":
    main()
