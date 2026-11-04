"""Optional ablations, imported only after research_train authorization.

The historical network definition remains unchanged for checkpoint admission.
No pretrained constructors or external models are used.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ._network import CourseworkDepthCNN


class AdmittedDepthCNN(CourseworkDepthCNN):
    """Matched direct/residual models on RGB/log-z/validity/producer confidence.

    Context is producer-warped to the centre optical grid, with missing samples
    explicitly zeroed. Output support stays limited to admitted centre depth.
    """

    def __init__(self, *, mode, context=False, correction_bound=0.5):
        super().__init__()
        self.encoder1[0] = nn.Conv2d(18 if context else 6, 24, 3, padding=1)
        self.mode, self.context, self.correction_bound = mode, context, correction_bound

    def forward(self, inputs, base_depth):
        if inputs.ndim != 4 or inputs.shape[1] != (18 if self.context else 6):
            raise ValueError("Research input channels differ from the frozen recipe")
        first = self.encoder1(inputs)
        second = self.encoder2(self.pool(first))
        third = self.encoder3(self.pool(second))
        features = self.bottleneck(self.pool(third))
        for skip, decoder in ((third, self.decoder3), (second, self.decoder2), (first, self.decoder1)):
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat((features, skip), dim=1))
        raw = self.depth_head(features)
        if self.mode == "residual":
            return base_depth * torch.exp(self.correction_bound * torch.tanh(raw))
        return F.softplus(raw) + 1e-3


def _mean(values, support, name):
    if not bool(support.any()):
        raise ValueError(f"No admitted support for enabled {name} objective")
    selected = values.masked_select(support)
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError(f"Nonfinite {name} residual")
    return selected.mean(), int(support.sum().item())


def objectives(prediction, batch, request):
    """Metric log-z objectives with native support counters; no scale alignment.

    Normal masks must exclude instance boundaries; temporal depth is already
    transformed into the centre camera's optical z by an admitted producer.
    """
    support = batch["target_valid"] & batch["base_valid"]
    target = torch.where(support, batch["target_depth_m"], torch.ones_like(prediction))
    log_prediction, log_target = prediction.log(), target.log()
    residual = log_prediction - log_target
    if request.robust_log:
        errors = F.huber_loss(log_prediction, log_target, reduction="none", delta=0.1)
    else:
        errors = residual.abs()
    loss, count = _mean(errors, support, "log-depth")
    counters = {"log_depth_pixels": count}
    terms = {"log_depth": float(loss.detach().item())}
    if request.boundary:
        total, pairs = loss * 0, 0
        for axis in (-1, -2):
            left, right = [slice(None)] * 4, [slice(None)] * 4
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            mask = support[left] & support[right]
            if bool(mask.any()):
                value, n = _mean((residual[left] - residual[right]).abs(), mask, "boundary")
                total, pairs = total + value * n, pairs + n
        if not pairs:
            raise ValueError("No adjacent target support for boundary objective")
        value = total / pairs
        loss = loss + 0.1 * value
        counters["boundary_pairs"], terms["boundary"] = pairs, float(value.detach().item())
    if request.normals:
        height, width = prediction.shape[-2:]
        y, x = torch.meshgrid(torch.arange(height, device=prediction.device),
                              torch.arange(width, device=prediction.device), indexing="ij")
        pixels = torch.stack((x, y, torch.ones_like(x))).float().reshape(3, -1)
        rays = (torch.linalg.inv(batch["intrinsics"]) @ pixels).reshape(-1, 3, height, width)
        def normals(depth):
            points = rays * depth
            origin = points[..., :-1, :-1]
            dx = points[..., :-1, 1:] - origin
            dy = points[..., 1:, :-1] - origin
            cross = torch.linalg.cross(dx, dy, dim=1)
            return F.normalize(cross, dim=1, eps=1e-8), cross.square().sum(dim=1, keepdim=True) > 1e-16
        predicted, good_prediction = normals(prediction)
        expected, good_target = normals(target)
        mask = (support[..., :-1, :-1] & support[..., :-1, 1:] & support[..., 1:, :-1]
                & batch["normal_support"][..., :-1, :-1] & good_prediction & good_target)
        value, n = _mean((1 - (predicted * expected).sum(dim=1, keepdim=True)).clamp(0, 2), mask, "normal")
        loss = loss + 0.1 * value
        counters["normal_neighbourhoods"], terms["normal"] = n, float(value.detach().item())
    if request.temporal:
        mask = batch["temporal_support"] & batch["base_valid"]
        aligned = torch.where(mask, batch["temporal_depth_m"], torch.ones_like(prediction))
        value, n = _mean(F.huber_loss(log_prediction.expand_as(aligned), aligned.log(),
                                     reduction="none", delta=0.1), mask, "motion-compensated temporal")
        loss = loss + 0.1 * value
        counters["temporal_pixels"], terms["temporal"] = n, float(value.detach().item())
    return loss, terms, counters
