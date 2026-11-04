"""Finite parameter bounds shared by this feature's planner and factories."""

from ._common import integer, number

BOUNDS = {
    "ann": {"width": (4, 128), "layers": (1, 4)},
    "cnn": {"width": (4, 64), "layers": (1, 6)},
    "transformer": {"width": (8, 128), "layers": (1, 4), "heads": (1, 8),
                    "patch_size": (4, 32), "max_tokens": (1, 256)},
    "gradient_boosting": {"estimators": (1, 128), "max_thresholds": (1, 32)},
}
DEFAULTS = {
    "ann": {"width": 32, "layers": 2},
    "cnn": {"width": 16, "layers": 3},
    "transformer": {"width": 32, "layers": 2, "heads": 4, "patch_size": 16, "max_tokens": 256},
    "gradient_boosting": {"estimators": 16, "max_thresholds": 16, "learning_rate": 0.1},
}


def parameters(technique, values, *, planning=False):
    if technique not in BOUNDS or not isinstance(values, dict):
        raise ValueError("Unknown technique or invalid parameter object")
    bounds = dict(BOUNDS[technique])
    if planning and technique != "gradient_boosting":
        bounds.update(epochs=(1, 100), batch_size=(1, 8))
    allowed = set(bounds)
    if planning or technique == "gradient_boosting":
        allowed.add("learning_rate")
    if planning and technique != "gradient_boosting":
        allowed.add("weight_decay")
    if set(values) - allowed:
        raise ValueError("Unknown hyperparameter")
    result = {**DEFAULTS[technique], **values}
    for name, (low, high) in bounds.items():
        if name in result:
            integer(result[name], name, low, high)
    if "learning_rate" in result:
        number(result["learning_rate"], "learning_rate", 0.000001,
               1.0 if technique == "gradient_boosting" else 0.1)
    if "weight_decay" in result:
        number(result["weight_decay"], "weight_decay", 0, 1)
    if technique == "transformer" and result["width"] % result["heads"]:
        raise ValueError("Transformer width must be divisible by heads")
    return result
