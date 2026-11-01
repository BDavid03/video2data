"""Camera-motion estimation; no authorization bypass."""

import argparse
from pathlib import Path

from . import EstimateCameraRequest, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("estimate",))
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("estimate.json"))
    args = parser.parse_args()
    print(estimate(EstimateCameraRequest.from_file(args.request)))


if __name__ == "__main__":
    main()
