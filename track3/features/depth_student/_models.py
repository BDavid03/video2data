"""Fully convolutional depth students. Imported only inside authorized GPU operations.

All networks are randomly initialised, resolution-agnostic (input sides padded to a
multiple of 2**stages internally) and emit positive depth in metres via softplus.

Architectures
  unet      residual encoder-decoder with skips; GroupNorm + SiLU; optional dilated
            bottleneck (parallel dilations 1/2/4/8) and optional self-attention at
            the bottleneck for global context (metric scale needs global cues).
  dilated   single-resolution residual stack with growing dilation (no pooling),
            a cheap high-detail baseline for edge fidelity.
  coursework the original coursework CNN wrapped for the same interface.
"""

import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import nn
from torch.nn import functional as F


def _norm(channels, kind):
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class ResBlock(nn.Module):
    def __init__(self, inputs, outputs, norm="group", dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=dilation, dilation=dilation)
        self.norm1 = _norm(outputs, norm)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=dilation, dilation=dilation)
        self.norm2 = _norm(outputs, norm)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(self.drop(y)))
        return F.silu(y + self.skip(x))


class DilatedBottleneck(nn.Module):
    """Parallel dilated branches (1, 2, 4, 8) fused by a 1x1: wide receptive field at low cost."""

    def __init__(self, channels, norm):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels // 4, 3, padding=d, dilation=d),
                                                     _norm(channels // 4, norm), nn.SiLU()) for d in (1, 2, 4, 8)])
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return F.silu(self.fuse(torch.cat([b(x) for b in self.branches], dim=1)) + x)


class AttentionBottleneck(nn.Module):
    """Single multi-head self-attention block over bottleneck tokens with learned 2-D positional bias."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        self.norm2 = nn.LayerNorm(channels)
        for layer in (self.attention.out_proj, self.mlp[-1]):  # the block starts as an identity residual
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Float32 regardless of autocast, and the math SDPA kernel: the fused fp32 kernels on this GPU return
        # gradients ~30x too large through the bottleneck (verified 2026-10-29), which diverges training.
        with torch.autocast(device_type=x.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            return self._forward(x.float()).to(x.dtype)

    def _forward(self, x):
        n, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        # Positional features from normalised coordinates, built with tensor ops so exported graphs stay dynamic in H/W.
        ys = (torch.arange(h, device=x.device, dtype=x.dtype) / torch.clamp(torch.tensor(h - 1, device=x.device, dtype=x.dtype), min=1)) * 2 - 1
        xs = (torch.arange(w, device=x.device, dtype=x.dtype) / torch.clamp(torch.tensor(w - 1, device=x.device, dtype=x.dtype), min=1)) * 2 - 1
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(1, -1, 2)
        position = torch.cat([torch.sin(grid * math.pi * k) for k in (1, 2, 4)] +
                             [torch.cos(grid * math.pi * k) for k in (1, 2, 4)], dim=-1)  # 12 features
        tokens = tokens + F.pad(position, (0, c - 12))
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        update = attended + self.mlp(self.norm2(tokens + attended))
        return x + update.transpose(1, 2).reshape(n, c, h, w)


class UNetStudent(nn.Module):
    def __init__(self, width=32, stages=4, norm="group", bottleneck="dilated", attention=False,
                 dropout=0.0, blocks_per_stage=1):
        super().__init__()
        self.stages = stages
        widths = [width * 2 ** i for i in range(stages + 1)]
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), _norm(width, norm), nn.SiLU())
        self.encoders = nn.ModuleList()
        for i in range(stages):
            layers = [ResBlock(widths[i], widths[i + 1], norm, dropout=dropout)]
            layers += [ResBlock(widths[i + 1], widths[i + 1], norm, dropout=dropout) for _ in range(blocks_per_stage - 1)]
            self.encoders.append(nn.Sequential(*layers))
        deep = widths[stages]
        middle = [ResBlock(deep, deep, norm, dropout=dropout)]
        if bottleneck == "dilated":
            middle.append(DilatedBottleneck(deep, norm))
        if attention:
            middle.append(AttentionBottleneck(deep, heads=4 if deep % 4 == 0 else 1))
        self.middle = nn.Sequential(*middle)
        self.decoders = nn.ModuleList()
        for i in reversed(range(stages)):
            self.decoders.append(ResBlock(widths[i + 1] + widths[i + 1], widths[i], norm, dropout=dropout))
        self.head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 1, 1))

    def forward(self, image):
        h, w = image.shape[-2], image.shape[-1]
        multiple = 2 ** self.stages
        x = F.pad(image, (0, (-w) % multiple, 0, (-h) % multiple), mode="replicate")
        x = self.stem(x)
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.middle(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        depth = F.softplus(self.head(x)) + 1e-3
        return depth[..., :h, :w]


class DilatedStudent(nn.Module):
    def __init__(self, width=32, depth=6, norm="group", dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), _norm(width, norm), nn.SiLU())
        self.body = nn.Sequential(*[ResBlock(width, width, norm, dilation=min(2 ** (i % 5), 16), dropout=dropout)
                                    for i in range(depth)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, image):
        return F.softplus(self.head(self.body(self.stem(image)))) + 1e-3


class CourseworkStudent(nn.Module):
    def __init__(self, width=24, dropout=0.1):
        super().__init__()
        from track3.features.scratch_depth._network import CourseworkDepthCNN
        self.net = CourseworkDepthCNN(base_channels=width, dropout=dropout)

    def forward(self, image):
        h, w = image.shape[-2], image.shape[-1]
        x = F.pad(image, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        return self.net(x)[..., :h, :w]


from ._config import ARCHITECTURES, BOUNDS, DEFAULTS, architecture_config  # noqa: F401


def build(arch, config):
    if arch == "unet":
        return UNetStudent(**config)
    if arch == "dilated":
        return DilatedStudent(**config)
    return CourseworkStudent(**config)
