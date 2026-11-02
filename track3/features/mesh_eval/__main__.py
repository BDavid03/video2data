"""Run the recorded CPU pair evaluation."""

import argparse
from pathlib import Path

from .pairs import PairRequest, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pairs",))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(PairRequest.from_file(args.request), args.request))


if __name__ == "__main__":
    main()
