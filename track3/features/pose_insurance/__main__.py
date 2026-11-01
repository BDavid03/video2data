"""Command line entry point for one task or one policy run."""
import argparse
from .request import Request


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=["build", "run"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    from .scene import build
    from .handler import run
    print({"build":build,"run":run}[args.operation](Request.from_file(args.request)))


if __name__ == "__main__":
    main()
