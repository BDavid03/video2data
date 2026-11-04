"""Mesh extents commands; authorization belongs to runtime.json."""

import argparse
from pathlib import Path

from .build import BuildRequest, build
from .measure import MeasureRequest, measure
from .proxy import ProxyRequest, proxy
from .proxy_insights import InsightsRequest, proxy_insights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("measure", "build", "adopt", "proxy", "proxy-insights"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="proxy: target masks and placement only")
    args = parser.parse_args()
    if args.dry_run and args.operation != "proxy":
        parser.error("--dry-run is available only for proxy")
    if args.operation == "proxy":
        print(proxy(ProxyRequest.from_file(args.request), dry_run=args.dry_run))
        return
    if args.operation == "adopt":
        from .adopt import AdoptRequest, adopt

        handler, request_type = adopt, AdoptRequest
    else:
        handler, request_type = {"measure": (measure, MeasureRequest), "build": (build, BuildRequest),
                                 "proxy-insights": (proxy_insights, InsightsRequest)}[args.operation]
    print(handler(request_type.from_file(args.request)))


if __name__ == "__main__":
    main()
