"""Train a depth student from random weights on prepared MoGe2 pairs.

Recipe surface: architecture (see _models), loss weights (see _losses), AdamW with
cosine schedule and linear warmup, optional EMA of weights, BF16 autocast,
channels-last, and augmentation (horizontal flip, colour jitter, random crop).
Only training episodes are loaded; there is no tuning loader or best-epoch pick.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
import math

from track3.gpu import GpuLease
from track3.paths import project_path
from track3.provenance import write_json
from ._common import (ROOT, authorize, heavy_execution,
                      operation_run, partition, read_json)
from ._config import architecture_config

LOSS_NAMES = ("log_l1", "silog", "gradient")


@dataclass(frozen=True)
class TrainStudentRequest:
    dataset: str | None = None
    arch: str = "unet"
    architecture: dict = field(default_factory=dict)
    loss_weights: dict = field(default_factory=lambda: {"log_l1": 1.0, "silog": 0.0, "gradient": 0.0})
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    warmup_epochs: int = 1
    schedule: str = "cosine"
    ema_decay: float = 0.0
    bf16_autocast: bool = True
    channels_last: bool = False
    augment_flip: bool = True
    augment_color: float = 0.1
    augment_crop: float = 0.0
    grad_clip: float = 1.0
    seed: int = 31

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        architecture_config(self.arch, self.architecture)
        if set(self.loss_weights) - set(LOSS_NAMES) or any(
                type(v) not in (int, float) or not 0 <= v <= 10 for v in self.loss_weights.values()):
            raise ValueError(f"loss_weights keys must be in {LOSS_NAMES} with values in [0, 10]")
        if not any(self.loss_weights.get(k, 0) > 0 for k in LOSS_NAMES):
            raise ValueError("At least one loss weight must be positive")
        for name, low, high in (("epochs", 1, 300), ("batch_size", 1, 16), ("warmup_epochs", 0, 20), ("seed", 0, 2**31 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, low, high in (("learning_rate", 1e-6, 0.1), ("weight_decay", 0.0, 1.0), ("ema_decay", 0.0, 0.9999),
                                ("augment_color", 0.0, 0.5), ("augment_crop", 0.0, 0.5), ("grad_clip", 0.0, 100.0)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.schedule not in ("cosine", "constant"):
            raise ValueError("schedule must be cosine or constant")
        for name in ("bf16_autocast", "channels_last", "augment_flip"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


class _Augmented:
    """Map-style wrapper: uint8 HWC rgb -> float CHW in [0,1]; log-space-safe augmentation."""

    def __init__(self, pairs, request, torch, generator):
        self.pairs, self.request, self.torch, self.generator = pairs, request, torch, generator

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        torch, r = self.torch, self.request
        sample = self.pairs[index]
        image = torch.from_numpy(sample["rgb"]).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(sample["depth_m"]).float()
        valid = torch.from_numpy(sample["valid"])
        depth = torch.where(valid, depth, torch.ones_like(depth))
        if r.augment_flip and torch.rand((), generator=self.generator) < 0.5:
            image, depth, valid = image.flip(-1), depth.flip(-1), valid.flip(-1)
        if r.augment_color > 0:
            brightness = 1 + (torch.rand((), generator=self.generator) * 2 - 1) * r.augment_color
            contrast = 1 + (torch.rand((), generator=self.generator) * 2 - 1) * r.augment_color
            mean = image.mean()
            image = ((image - mean) * contrast + mean) * brightness
            image = image.clamp(0, 1)
        if r.augment_crop > 0:
            h, w = depth.shape
            fraction = 1 - float(torch.rand((), generator=self.generator)) * r.augment_crop
            ch, cw = max(16, int(h * fraction)), max(16, int(w * fraction))
            top = int(torch.randint(0, h - ch + 1, (), generator=self.generator))
            left = int(torch.randint(0, w - cw + 1, (), generator=self.generator))
            image, depth, valid = image[:, top:top + ch, left:left + cw], depth[top:top + ch, left:left + cw], valid[top:top + ch, left:left + cw]
        return image, depth, valid


def _collate(batch, torch):
    """Pad a batch to its largest crop; padded pixels are invalid."""
    h = max(item[0].shape[-2] for item in batch)
    w = max(item[0].shape[-1] for item in batch)
    images, depths, valids = [], [], []
    for image, depth, valid in batch:
        ph, pw = h - image.shape[-2], w - image.shape[-1]
        images.append(torch.nn.functional.pad(image, (0, pw, 0, ph)))
        depths.append(torch.nn.functional.pad(depth, (0, pw, 0, ph), value=1.0))
        valids.append(torch.nn.functional.pad(valid, (0, pw, 0, ph), value=False))
    return torch.stack(images), torch.stack(depths), torch.stack(valids)


def _lr_at(step, total, warmup, request):
    if warmup and step < warmup:
        return request.learning_rate * (step + 1) / warmup
    if request.schedule == "constant":
        return request.learning_rate
    progress = (step - warmup) / max(1, total - warmup)
    return request.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_student(request: TrainStudentRequest):
    runtime = authorize("train")
    if request.dataset is None:
        raise ValueError("Supply the dataset.json of a completed scratch_depth.prepare run")
    heavy_execution("train")
    import numpy as np
    import torch
    from track3.features.learning_experiments.data import PreparedPairs
    from ._models import build
    from ._losses import combined

    config = architecture_config(request.arch, request.architecture)
    groups = partition()
    pairs = PreparedPairs(project_path(ROOT, request.dataset), groups, "train", np)
    with operation_run("train", request, inputs=pairs.input_paths, parents=[pairs.producer_path]) as (directory, record):
        record.update(arch=request.arch, architecture=config, episode_partition=groups, dataset_sha256=pairs.dataset_sha256,
                      data_contract=pairs.data_contract, training_samples=len(pairs), initialization="random_no_pretrained_weights",
                      optimizer="AdamW", epochs=[], optimizer_steps=0, checkpoint=None,
                      selection_note="Fixed epochs; no tuning loader, early stopping or best-epoch selection")
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"], stage="depth_student_train")
        with lease:
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(request.seed)
            generator = torch.Generator().manual_seed(request.seed)
            model = build(request.arch, config).cuda().train()
            if request.channels_last:
                model = model.to(memory_format=torch.channels_last)
            ema = None
            if request.ema_decay > 0:
                ema = build(request.arch, config).cuda().eval()
                ema.load_state_dict(model.state_dict())
                for parameter in ema.parameters():
                    parameter.requires_grad_(False)
            record["parameter_count"] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate, weight_decay=request.weight_decay)
            loader = torch.utils.data.DataLoader(_Augmented(pairs, request, torch, generator), batch_size=request.batch_size,
                                                 shuffle=True, num_workers=0, drop_last=False,
                                                 generator=torch.Generator().manual_seed(request.seed),
                                                 collate_fn=lambda batch: _collate(batch, torch))
            steps_per_epoch = len(loader)
            total_steps, warmup_steps = steps_per_epoch * request.epochs, steps_per_epoch * request.warmup_epochs
            step = 0
            try:
                for epoch in range(1, request.epochs + 1):
                    sums, pixels = {}, 0
                    for image, depth, valid in loader:
                        lease.require_headroom()
                        lr = _lr_at(step, total_steps, warmup_steps, request)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        image, depth, valid = (v.cuda(non_blocking=True) for v in (image, depth, valid))
                        if request.channels_last:
                            image = image.contiguous(memory_format=torch.channels_last)
                        optimizer.zero_grad(set_to_none=True)
                        context = torch.autocast("cuda", dtype=torch.bfloat16) if request.bf16_autocast else nullcontext()
                        with context:
                            prediction = model(image)
                        with torch.autocast("cuda", enabled=False):
                            loss, parts = combined(prediction[:, 0], depth, valid, request.loss_weights)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite training loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), request.grad_clip if request.grad_clip > 0 else float("inf"),
                                                              error_if_nonfinite=False)
                        if not torch.isfinite(norm):
                            # A non-finite gradient is skipped and counted; more than 5% skipped steps fails the run honestly.
                            optimizer.zero_grad(set_to_none=True)
                            record["skipped_nonfinite_steps"] = record.get("skipped_nonfinite_steps", 0) + 1
                            if record["skipped_nonfinite_steps"] > max(10, total_steps // 20):
                                raise FloatingPointError("Too many non-finite gradient steps")
                            step += 1
                            continue
                        optimizer.step()
                        step += 1
                        if ema is not None:
                            with torch.no_grad():
                                for e, p in zip(ema.parameters(), model.parameters()):
                                    e.mul_(request.ema_decay).add_(p.detach(), alpha=1 - request.ema_decay)
                                for e, b in zip(ema.buffers(), model.buffers()):
                                    e.copy_(b)
                        n = int(valid.sum())
                        pixels += n
                        for name, value in parts.items():
                            sums[name] = sums.get(name, 0.0) + value * n
                        sums["total"] = sums.get("total", 0.0) + float(loss.detach()) * n
                    record["optimizer_steps"] = step
                    record["epochs"].append({"epoch": epoch, "lr": lr, "supervised_pixels": pixels,
                                             "training_log_depth_mae": sums.get("log_l1", sums["total"]) / max(pixels, 1),
                                             **{f"loss_{k}": v / max(pixels, 1) for k, v in sums.items()}})
                    print(f"depth_student epoch {epoch}/{request.epochs} loss {sums['total'] / max(pixels, 1):.5f} lr {lr:.2e}", flush=True)
                    write_json(directory / "run.json", record)
                torch.cuda.synchronize(0)
                weights = (ema if ema is not None else model).state_dict()
                payload = {"schema_version": 1, "kind": "depth_student.checkpoint", "arch": request.arch, "architecture": config,
                           "state_dict": {k: v.detach().cpu() for k, v in weights.items()},
                           "weights_are_ema": ema is not None, "episode_partition": groups,
                           "data_contract": pairs.data_contract, "dataset_sha256": pairs.dataset_sha256,
                           "request": json.loads(json.dumps(request.__dict__)), "epochs": request.epochs,
                           "optimizer_steps": step, "pretrained_weights": False}
                (directory / "checkpoints").mkdir()
                torch.save(payload, directory / "checkpoints" / "final.pt")
                record["checkpoint"] = "checkpoints/final.pt"
            finally:
                record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                model = ema = optimizer = None
                torch.cuda.empty_cache()
        record["gpu"] = lease.report()
    return directory / "checkpoints" / "final.pt"


import json  # noqa: E402  (used for request serialisation inside the checkpoint)
