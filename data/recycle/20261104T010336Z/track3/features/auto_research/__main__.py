"""Thin dispatch: loop, finalize, report. Requests default to the feature's JSON drafts."""

import argparse

from ._common import FEATURE
from .loop import LoopRequest, run_loop
from .finalize import FinalizeRequest, finalize
from .report import ReportRequest, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("loop", "finalize", "report"))
    parser.add_argument("--request")
    args = parser.parse_args()
    request_type, handler = {"loop": (LoopRequest, run_loop), "finalize": (FinalizeRequest, finalize),
                             "report": (ReportRequest, write_report)}[args.operation]
    print(handler(request_type.from_file(args.request or str(FEATURE / f"{args.operation}.json"))))


if __name__ == "__main__":
    main()
