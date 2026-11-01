"""Reference entry point; all operations are disabled in the shipped runtime.json."""

import argparse
from pathlib import Path

from .prepare import PrepareDepthDataRequest, prepare_depth_data
from .train import TrainDepthRequest, train_depth
from .predict import PredictDepthRequest, predict_depth
from .research import (PrepareLearningRequest, LearningRequest, PredictLearningRequest,
                       prepare_learning, train_learning, predict_learning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "train", "predict", "research-prepare", "research-train", "research-predict"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    request_path = args.request or Path(__file__).with_name(f"{args.operation}.json")
    request_type, handler = {
        "prepare": (PrepareDepthDataRequest, prepare_depth_data),
        "train": (TrainDepthRequest, train_depth),
        "predict": (PredictDepthRequest, predict_depth),
        "research-prepare": (PrepareLearningRequest, prepare_learning),
        "research-train": (LearningRequest, train_learning),
        "research-predict": (PredictLearningRequest, predict_learning),
    }[args.operation]
    print(handler(request_type.from_file(request_path)))


if __name__ == "__main__":
    main()
