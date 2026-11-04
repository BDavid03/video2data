"""Bounded standard-library residual regression in log optical-axis depth."""

from dataclasses import dataclass
import math

from ._common import authorize, episode_partition
from ._config import parameters

MAX_ROWS = 2048
FEATURE_SCHEMA = ("red_0_1", "green_0_1", "blue_0_1")
TARGET_TRANSFORM = "natural_log_of_positive_optical_axis_depth_metres"


@dataclass(frozen=True)
class BoostRequest:
    estimators: int = 16
    learning_rate: float = 0.1
    max_thresholds: int = 16
    train_episodes: tuple[int, ...] = (0, 11, 12, 18, 21, 23, 26, 31)
    tuning_episodes: tuple[int, ...] = (39, 42)
    holdout_episodes: tuple[int, ...] = (43, 44)


@dataclass(frozen=True)
class DecisionStump:
    feature: int | None
    threshold: float
    left: float
    right: float


@dataclass(frozen=True)
class BoostModel:
    intercept: float
    learning_rate: float
    stumps: tuple[DecisionStump, ...]
    train_episodes: tuple[int, ...]
    tuning_episodes: tuple[int, ...]
    holdout_episodes: tuple[int, ...]
    training_rows: int
    feature_schema: tuple[str, ...] = FEATURE_SCHEMA
    target_transform: str = TARGET_TRANSFORM
    scratch_checkpoint_admitted: bool = False
    submission_candidate: bool = False

    def predict(self, features, *, episodes):
        return predict(self, features, episodes=episodes)


def _rows(features):
    if not isinstance(features, (list, tuple)) or not 1 <= len(features) <= MAX_ROWS:
        raise ValueError(f"Require 1–{MAX_ROWS} explicitly selected valid RGB rows")
    result = []
    for row in features:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError("Each row is exactly [red, green, blue]")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in row):
            raise ValueError("RGB features must be finite in [0,1]")
        result.append(tuple(float(v) for v in row))
    return result


def _episodes(episodes, rows, allowed):
    if (not isinstance(episodes, (tuple, list)) or len(episodes) != rows
            or any(type(e) is not int or e not in allowed for e in episodes)):
        raise ValueError("One declared, permitted episode is required per row")


def _stump_value(stump, row):
    return stump.left if stump.feature is None or row[stump.feature] <= stump.threshold else stump.right


def _orders(rows, max_thresholds):
    """Bounded deterministic cut positions, computed once from training features."""
    result = []
    for feature in range(3):
        order = sorted(range(len(rows)), key=lambda i: (rows[i][feature], i))
        cuts = [i for i in range(1, len(order))
                if rows[order[i - 1]][feature] < rows[order[i]][feature]]
        if len(cuts) > max_thresholds:
            positions = ([len(cuts) // 2] if max_thresholds == 1 else
                         [i * (len(cuts) - 1) // (max_thresholds - 1) for i in range(max_thresholds)])
            cuts = [cuts[i] for i in positions]
        thresholds = []
        for cut in cuts:
            low, high = rows[order[cut - 1]][feature], rows[order[cut]][feature]
            midpoint = low + (high - low) / 2
            # Adjacent floating values may round the midpoint to the right value.
            thresholds.append((cut, midpoint if midpoint < high else low))
        result.append((feature, order, thresholds))
    return result


def _fit_stump(residuals, orders):
    total = math.fsum(residuals)
    squares = math.fsum(v * v for v in residuals)
    count = len(residuals)
    mean = total / count
    best = DecisionStump(None, 0.0, mean, mean)
    best_error = max(0.0, squares - total * total / count)
    # Stable feature/cut order and strict improvement resolve ties deterministically.
    for feature, order, cuts in orders:
        prefix = [0.0]
        for index in order:
            prefix.append(prefix[-1] + residuals[index])
        for cut, threshold in cuts:
            left_sum, right_sum = prefix[cut], total - prefix[cut]
            error = max(0.0, squares - left_sum * left_sum / cut
                        - right_sum * right_sum / (count - cut))
            if error < best_error:
                best_error = error
                best = DecisionStump(feature, threshold, left_sum / cut, right_sum / (count - cut))
    return best


def fit(request: BoostRequest, features, target_log_depth, *, episodes):
    """Fit additive decision stumps on training episodes only; no automatic split.

    Targets are already ln(depth_metres), restricted to depth in [0.001,1000].
    The caller must exclude invalid target/RGB pixels before supplying rows.
    No holdout or tuning targets are accepted, loaded, scored or selected here.
    """
    authorize("train")
    config = parameters("gradient_boosting", {
        "estimators": request.estimators, "learning_rate": request.learning_rate,
        "max_thresholds": request.max_thresholds})
    partition = episode_partition(request.train_episodes, request.tuning_episodes, request.holdout_episodes)
    rows = _rows(features)
    _episodes(episodes, len(rows), partition["train_episodes"])
    if not isinstance(target_log_depth, (list, tuple)) or len(target_log_depth) != len(rows):
        raise ValueError("Require one log-depth target per RGB row")
    low, high = math.log(0.001), math.log(1000.0)
    if any(type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high
           for v in target_log_depth):
        raise ValueError("Targets must be finite natural logs of depths in [0.001,1000] metres")
    targets = [float(v) for v in target_log_depth]
    intercept = math.fsum(targets) / len(targets)
    fitted = [intercept] * len(rows)
    orders = _orders(rows, config["max_thresholds"])
    stumps = []
    for _ in range(config["estimators"]):
        residuals = [target - estimate for target, estimate in zip(targets, fitted)]
        stump = _fit_stump(residuals, orders)
        stumps.append(stump)
        fitted = [estimate + config["learning_rate"] * _stump_value(stump, row)
                  for estimate, row in zip(fitted, rows)]
    return BoostModel(intercept, config["learning_rate"], tuple(stumps),
                      tuple(partition["train_episodes"]), tuple(partition["tuning_episodes"]),
                      tuple(partition["holdout_episodes"]), len(rows))


def predict(model: BoostModel, features, *, episodes):
    """Return positive depths in metres, one per row; authorization checked first."""
    authorize("predict")
    if not isinstance(model, BoostModel):
        raise ValueError("Require this slice's standalone BoostModel")
    partition = episode_partition(model.train_episodes, model.tuning_episodes, model.holdout_episodes)
    parameters("gradient_boosting", {"estimators": len(model.stumps),
                                     "learning_rate": model.learning_rate})
    if (model.feature_schema != FEATURE_SCHEMA or model.target_transform != TARGET_TRANSFORM
            or not math.isfinite(model.intercept)):
        raise ValueError("Model feature/target contract is invalid")
    for stump in model.stumps:
        if (not isinstance(stump, DecisionStump)
                or (stump.feature is not None and (type(stump.feature) is not int or stump.feature not in (0, 1, 2)))
                or any(not math.isfinite(v) for v in (stump.threshold, stump.left, stump.right))):
            raise ValueError("Invalid decision stump")
    rows = _rows(features)
    allowed = partition["train_episodes"] + partition["tuning_episodes"] + partition["holdout_episodes"]
    _episodes(episodes, len(rows), allowed)
    result = []
    for row in rows:
        log_depth = model.intercept + model.learning_rate * math.fsum(
            _stump_value(stump, row) for stump in model.stumps)
        try:
            depth = math.exp(log_depth)
        except OverflowError as error:
            raise ValueError("Prediction overflowed depth") from error
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("Prediction is not finite positive depth")
        result.append(depth)
    return result
