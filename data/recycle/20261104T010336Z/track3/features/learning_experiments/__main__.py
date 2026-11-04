"""Feature-owned operations guarded by explicit runtime authorization."""

import argparse

from ._common import FEATURE, authorize
from .planning import PlanRequest, write_plan
from .train import TrainRequest, train_model
from .predict import PredictRequest, predict_model
from .score import ScoreRequest, score_predictions
from .tune import TuneRequest, tune_models
from .boosting_operations import BoostTrainRequest, BoostPredictRequest, train_boost, predict_boost
from .ensemble_operations import EnsembleRequest, ensemble_predictions
from .execute import ExecuteRequest, execute_tuning
from .investigate import InvestigateRequest, investigate_models
from .report import ReportRequest, write_report
from .selection import (GeometrySelectionRequest, GeometryFreezeRequest, compare_geometry, freeze_geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "train", "predict", "boost-train", "boost-predict",
                                               "ensemble", "score", "tune", "execute", "investigate", "report",
                                               "geometry-selection", "geometry-freeze"))
    parser.add_argument("--request")
    args = parser.parse_args()
    authorize(args.operation.replace("-", "_"))
    request_type, handler = {"geometry-selection": (GeometrySelectionRequest, compare_geometry),
                             "geometry-freeze": (GeometryFreezeRequest, freeze_geometry),
                             "report": (ReportRequest, write_report),
                             "investigate": (InvestigateRequest, investigate_models),
                             "execute": (ExecuteRequest, execute_tuning),
                             "plan": (PlanRequest, write_plan),
                             "train": (TrainRequest, train_model),
                             "predict": (PredictRequest, predict_model),
                             "boost-train": (BoostTrainRequest, train_boost),
                             "boost-predict": (BoostPredictRequest, predict_boost),
                             "ensemble": (EnsembleRequest, ensemble_predictions),
                             "score": (ScoreRequest, score_predictions),
                             "tune": (TuneRequest, tune_models)}[args.operation]
    path = args.request or str(FEATURE / f"{args.operation}.json")
    print(handler(request_type.from_file(path)))


if __name__ == "__main__":
    main()
