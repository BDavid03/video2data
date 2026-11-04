"""Thin dispatch: train, predict, export with --request JSON."""

import argparse

from ._common import FEATURE
from .train import TrainStudentRequest, train_student
from .predict import PredictStudentRequest, predict_student
from .export import ExportStudentRequest, export_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "predict", "export"))
    parser.add_argument("--request")
    args = parser.parse_args()
    request_type, handler = {"train": (TrainStudentRequest, train_student), "predict": (PredictStudentRequest, predict_student),
                             "export": (ExportStudentRequest, export_student)}[args.operation]
    print(handler(request_type.from_file(args.request or str(FEATURE / f"{args.operation}.json"))))


if __name__ == "__main__":
    main()
