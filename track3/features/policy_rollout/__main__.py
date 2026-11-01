import argparse
from pathlib import Path

from . import RolloutRequest, rollout, inspect, self_score
from .achieved_comparison import AchievedComparisonRequest, compare_achieved


def main():
    parser = argparse.ArgumentParser(description="Disabled policy rollout operations")
    parser.add_argument("operation", choices=("rollout", "inspect", "self-score", "compare-achieved"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.operation == "compare-achieved":
        request = AchievedComparisonRequest.from_file(args.request or Path(__file__).with_name("compare-achieved.json"))
        print(compare_achieved(request))
        return
    request = RolloutRequest.from_file(args.request or Path(__file__).with_name(args.operation + ".json"))
    print({"rollout": rollout, "inspect": inspect, "self-score": self_score}[args.operation](request))


if __name__ == "__main__":
    main()
