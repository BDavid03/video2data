"""Execute the explicitly authorized bounded opening experiment request."""

import argparse
from pathlib import Path
from . import OpeningExperimentRequest, run_opening_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default=str(Path(__file__).with_name("run.json")))
    args = parser.parse_args()
    print(run_opening_experiments(OpeningExperimentRequest.from_file(args.request)))


if __name__ == "__main__":
    main()
