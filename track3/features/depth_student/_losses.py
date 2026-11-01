"""Masked depth losses on the teacher's absolute scale. Imported inside GPU operations only.

  log_l1    |log d - log t|                      metric accuracy (the selection objective)
  silog     sqrt(var(log d - log t) + lambda*mean^2)  scale-invariant structure (lambda=0.85 keeps scale)
  gradient  multi-scale |grad log d - grad log t|     edge fidelity: penalises blur, rewards sharp structure
The combined loss is sum_k w_k * loss_k with request weights; w=0 disables a term.
"""

import torch
from torch.nn import functional as F


def _masked_log_diff(prediction, target, valid):
    mask = valid.bool() & torch.isfinite(target) & (target > 0)
    if not bool(mask.any()):
        raise ValueError("Training batch contains no usable depth labels")
    safe_target = torch.where(mask, target, torch.ones_like(target))
    return prediction.log() - safe_target.log(), mask


def log_l1(diff, mask):
    return diff.abs().masked_select(mask).mean()


def silog(diff, mask, lam=0.85):
    values = diff.masked_select(mask)
    return torch.sqrt((values ** 2).mean() - lam * values.mean() ** 2 + 1e-8)


def gradient_matching(diff, mask, scales=4):
    """Mean |d/dx, d/dy| of the masked log ratio over a pyramid; zero where either pixel is invalid."""
    total, count = diff.new_zeros(()), 0
    d, m = diff, mask.float()
    for _ in range(scales):
        d = torch.where(m > 0, d, torch.zeros_like(d))
        gx, gy = d[..., :, 1:] - d[..., :, :-1], d[..., 1:, :] - d[..., :-1, :]
        mx, my = m[..., :, 1:] * m[..., :, :-1], m[..., 1:, :] * m[..., :-1, :]
        n = mx.sum() + my.sum()
        if n > 0:
            total = total + ((gx.abs() * mx).sum() + (gy.abs() * my).sum()) / n
            count += 1
        if min(d.shape[-2:]) < 8:
            break
        d = F.avg_pool2d(d * m, 2) / F.avg_pool2d(m, 2).clamp_min(1e-6)
        m = (F.avg_pool2d(m, 2) > 0.5).float()
    return total / max(count, 1)


def combined(prediction, target, valid, weights):
    prediction, target = prediction.float(), target.float()
    diff, mask = _masked_log_diff(prediction, target, valid)
    parts = {}
    if weights.get("log_l1", 0) > 0:
        parts["log_l1"] = log_l1(diff, mask)
    if weights.get("silog", 0) > 0:
        parts["silog"] = silog(diff, mask)
    if weights.get("gradient", 0) > 0:
        parts["gradient"] = gradient_matching(diff, mask)
    if not parts:
        raise ValueError("At least one loss weight must be positive")
    total = sum(weights[name] * value for name, value in parts.items())
    return total, {name: float(value.detach()) for name, value in parts.items()}
