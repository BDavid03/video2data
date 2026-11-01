"""Command entry point for the depth feature, disabled pending run authorization."""

import argparse
from pathlib import Path

from .generate import GenerateDepthRequest, generate_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path,
                        default=Path(__file__).with_name("generate.json"))
    args = parser.parse_args()
    result = generate_depth(GenerateDepthRequest.from_file(args.request))
    print(result)


if __name__ == "__main__":
    main()
