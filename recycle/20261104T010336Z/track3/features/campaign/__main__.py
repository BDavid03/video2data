"""Campaign board, explicit pins and recursive staleness."""

import argparse
from pathlib import Path

from . import CampaignRequest, campaign
from .materialize import MaterializeRequest, materialize_requests
from .evidence import EvidenceRequest, campaign_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("board", "pin", "stale", "queue", "materialize", "evidence", "freeze"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    if args.operation in ("evidence", "freeze"):
        print(campaign_evidence(args.operation, EvidenceRequest.from_file(path)))
    elif args.operation == "materialize":
        print(materialize_requests(MaterializeRequest.from_file(path)))
    else:
        print(campaign(args.operation, CampaignRequest.from_file(path)))


if __name__ == "__main__":
    main()
