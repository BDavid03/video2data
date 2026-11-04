"""Masked weighted geometric mean for already aligned depth tensors."""

from ._common import authorize, heavy_execution, number


def ensemble_depth(predictions, valid_masks, weights):
    """Return (N1HW depth, boolean validity); unsupported pixels contain NaN.

    Members must share pixels, camera geometry, optical-axis units and absolute
    scale. This function does no registration, calibration or model prediction.
    Weights must be chosen independently of the final holdout, then frozen.
    """
    authorize("ensemble")
    authorize("predict")
    authorize("model")
    if (not isinstance(predictions, (tuple, list)) or not 1 <= len(predictions) <= 8
            or not isinstance(valid_masks, (tuple, list)) or not isinstance(weights, (tuple, list))
            or len(predictions) != len(valid_masks) or len(predictions) != len(weights)):
        raise ValueError("Provide 1–8 members, masks and weights of matching length")
    weights = [number(w, "ensemble weight", 0, 1000) for w in weights]
    if not any(weights):
        raise ValueError("At least one ensemble weight must be positive")
    heavy_execution("predict")
    import torch

    first = predictions[0]
    if (not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[1] != 1
            or not 1 <= first.shape[0] <= 8 or min(first.shape[-2:]) < 1
            or first.shape[-2] * first.shape[-1] > 65536 or not first.is_floating_point()):
        raise ValueError("Expected floating N1HW depths with bounded batch and spatial size")
    # Float64 accumulation prevents float16 log/weight underflow and overflow.
    numerator = torch.zeros_like(first, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for depth, mask, weight in zip(predictions, valid_masks, weights):
        if (not isinstance(depth, torch.Tensor) or not isinstance(mask, torch.Tensor)
                or depth.shape != first.shape or mask.shape != first.shape
                or depth.device != first.device or mask.device != first.device
                or not depth.is_floating_point() or mask.dtype != torch.bool):
            raise ValueError("Every depth and boolean mask must share N1HW shape and device")
        valid = mask & torch.isfinite(depth) & (depth > 0)
        safe = torch.where(valid, depth.to(torch.float64), torch.ones_like(numerator))
        contribution = valid.to(torch.float64) * weight
        numerator = numerator + safe.log() * contribution
        denominator = denominator + contribution
    valid = denominator > 0
    depth = (numerator / torch.where(valid, denominator, torch.ones_like(denominator))).exp()
    depth = depth.to(first.dtype)
    valid = valid & torch.isfinite(depth) & (depth > 0)
    return torch.where(valid, depth, torch.full_like(depth, float("nan"))), valid
