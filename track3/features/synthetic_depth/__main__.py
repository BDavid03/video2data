"""Generate an explicitly authorized bounded procedural training dataset."""

import argparse
from pathlib import Path
from . import SyntheticDepthRequest, generate_synthetic_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default=str(Path(__file__).with_name("generate.json")))
    args = parser.parse_args()
    print(generate_synthetic_depth(SyntheticDepthRequest.from_file(args.request)))


if __name__ == "__main__":
    main()
