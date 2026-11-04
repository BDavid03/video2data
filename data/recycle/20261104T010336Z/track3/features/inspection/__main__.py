"""Explicit export and separately gated serving; neither invokes models."""

import argparse
from .operations import InspectionRequest, export, serve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("operation", choices=("export", "serve", "encode"))
parser.add_argument("path", help="Request JSON for export; inspection run.json for serve")
args = parser.parse_args()
if args.operation == "export":
    print(export(InspectionRequest.from_file(args.path)))
elif args.operation == "encode":
    from .reels import encode
    print(encode(args.path))
else:
    server = serve(args.path)
    try:
        input("Inspection available on loopback. Press Enter to stop.\n")
    finally:
        server.stop()
