"""Aggregate previously recorded, explicitly admitted producer traces."""

import argparse
from .operations import Request, aggregate
from .grouped import GroupedAccountingRequest, aggregate_grouped

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("request")
parser.add_argument("--grouped-accounting", action="store_true")
args = parser.parse_args()
request_type, handler = (GroupedAccountingRequest, aggregate_grouped) if args.grouped_accounting else (Request, aggregate)
print(handler(request_type.from_file(args.request)))
