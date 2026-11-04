"""Hand estimation, kinematics and lifting; all require explicit runtime authorization."""

import argparse
from pathlib import Path

from . import (EstimateHandsRequest, KinematicsRequest, LiftHandsRequest, ParkedTemplateRequest, PreviewHandsRequest,
              RefineHandsRequest, ContactHandsRequest, estimate, kinematics, lift, parked_template, preview, refine, contacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("estimate", "kinematics", "lift", "parked-template", "preview", "refine", "contacts"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    request_type, handler = {"estimate": (EstimateHandsRequest, estimate), "kinematics": (KinematicsRequest, kinematics),
                             "lift": (LiftHandsRequest, lift),
                             "preview": (PreviewHandsRequest, preview),
                             "refine": (RefineHandsRequest, refine), "contacts": (ContactHandsRequest, contacts),
                             "parked-template": (ParkedTemplateRequest, parked_template)}[args.operation]
    path = args.request or Path(__file__).with_name(args.operation + ".json")
    print(handler(request_type.from_file(path)))


if __name__ == "__main__":
    main()
