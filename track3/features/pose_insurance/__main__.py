"""Command line entry point for one task or one policy run."""
import argparse
from .request import Request


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=["build", "run", "placement_diagnostic", "length_diagnostic"])
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    if args.operation == "length_diagnostic":
        from .length_diagnostic import LengthRequest, diagnose
        print(diagnose(LengthRequest.from_file(args.request), args.request))
        return
    if args.operation == "placement_diagnostic":
        from .placement_diagnostic import PlacementRequest, diagnose
        print(diagnose(PlacementRequest.from_file(args.request)))
        return
    from .scene import build
    from .handler import run
    print({"build":build,"run":run}[args.operation](Request.from_file(args.request)))


if __name__ == "__main__":
    main()
