"""Coursework-based scratch-depth vertical slice; importing it does not load torch."""

from .prepare import PrepareDepthDataRequest, prepare_depth_data
from .train import TrainDepthRequest, train_depth
from .predict import PredictDepthRequest, predict_depth

__all__ = ["PrepareDepthDataRequest", "prepare_depth_data", "TrainDepthRequest", "train_depth",
           "PredictDepthRequest", "predict_depth"]
