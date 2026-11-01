import argparse
from pathlib import Path

from . import SubmissionRequest, plan, check, pack, upload


def main():
    parser = argparse.ArgumentParser(description="Disabled submission operations")
    parser.add_argument("operation", choices=("plan", "check", "pack-poses", "pack-meshes", "upload"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    request = SubmissionRequest.from_file(args.request or Path(__file__).with_name(args.operation + ".json"))
    if args.operation.startswith("pack-"):
        result = pack(args.operation, request)
    else:
        result = {"plan": plan, "check": check, "upload": upload}[args.operation](request)
    print(result)


if __name__ == "__main__":
    main()
