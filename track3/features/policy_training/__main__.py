"""Compose, train or select an explicitly authorized episode policy."""

import argparse
from pathlib import Path

from . import PolicyRequest, compose, train, select
from .comparison import ComparisonRequest, persist_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("compose", "train", "select", "comparison"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    if args.operation == "comparison":
        print(persist_comparison(ComparisonRequest.from_file(path)))
    else:
        print({"compose": compose, "train": train, "select": select}[args.operation](PolicyRequest.from_file(path)))


if __name__ == "__main__":
    main()
