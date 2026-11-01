"""Public diagnostics; no evaluation-label or authorization bypass."""

import argparse
from pathlib import Path

from . import DiagnosticRequest, diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("floor", "score", "frame-convention", "oracle", "agreement", "mesh"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    print(diagnose(args.operation, DiagnosticRequest.from_file(path)))


if __name__ == "__main__":
    main()
