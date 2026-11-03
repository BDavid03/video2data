"""Run the resumable D0 queue."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path(__file__).with_name("queue.json"))
    args = parser.parse_args()
    from .loop import run
    run(args.queue)


if __name__ == "__main__":
    main()
