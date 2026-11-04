"""Retarget one scene through separately gated operations."""

import argparse
from pathlib import Path

from . import RetargetRequest, export, assets, retarget, support, inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export", "assets", "retarget", "support", "inspect"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    print({"export": export, "assets": assets, "retarget": retarget, "support": support, "inspect": inspect}[
          args.operation](RetargetRequest.from_file(path)))


if __name__ == "__main__":
    main()
