"""Run the shared D0/D1 queue or render saved historical ledgers."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path(__file__).with_name("queue.json"))
    parser.add_argument("--render-only", type=Path, metavar="OUTPUT_DIRECTORY")
    args = parser.parse_args()
    from .loop import run, rerender
    if args.render_only:
        from track3.paths import output_path
        from track3.runs import ROOT
        rerender(args.queue, output_path(ROOT, args.render_only))
    else:
        run(args.queue)


if __name__ == "__main__":
    main()
