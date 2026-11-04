"""Owned neural training from random weights; disabled before any heavy import."""

from contextlib import nullcontext
from dataclasses import dataclass, field

from track3.gpu import GpuLease
from track3.paths import project_path
from track3.provenance import write_json
from ._common import (ROOT, authorize, episode_partition, heavy_execution,
                      integer, number, operation_run, read_json)
from ._config import parameters


@dataclass(frozen=True)
class TrainRequest:
    dataset: str | None = None
    technique: str = "cnn"
    model_parameters: dict = field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    seed: int = 31
    bf16_autocast: bool = False
    train_episodes: tuple[int, ...] = (0, 11, 12, 18, 21, 23, 26, 31)
    tuning_episodes: tuple[int, ...] = (39, 42)
    holdout_episodes: tuple[int, ...] = (43, 44)

    @classmethod
    def from_file(cls, path):
        value = read_json(path)
        return cls(**value)

    def __post_init__(self):
        if self.dataset is not None and (not isinstance(self.dataset, str) or not self.dataset):
            raise ValueError("dataset must name an explicit prepared dataset.json")
        if self.technique not in ("ann", "cnn", "transformer"):
            raise ValueError("Neural training requires ann, cnn or transformer")
        parameters(self.technique, self.model_parameters)
        integer(self.epochs, "epochs", 1, 100)
        integer(self.batch_size, "batch_size", 1, 8)
        integer(self.seed, "seed", 0, 2**32 - 1)
        number(self.learning_rate, "learning_rate", 0.000001, 0.1)
        number(self.weight_decay, "weight_decay", 0, 1)
        if type(self.bf16_autocast) is not bool:
            raise ValueError("bf16_autocast must be a boolean")


class _TensorPairs:
    def __init__(self, pairs, torch):
        self.pairs, self.torch = pairs, torch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sample = self.pairs[index]
        torch = self.torch
        return (torch.from_numpy(sample["rgb"]).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(sample["depth_m"]).unsqueeze(0),
                torch.from_numpy(sample["valid"]).unsqueeze(0),
                sample["row"]["episode_index"])


def masked_loss(prediction, target, valid, torch):
    """Float32 masked log-depth MAE, finite logs and zero excluded gradients."""
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid must have identical shapes")
    prediction, target = prediction.float(), target.float()
    mask = valid.bool() & torch.isfinite(target) & (target > 0)
    pixels = int(mask.sum().item())
    if pixels == 0:
        raise ValueError("No positive finite supervised depth pixels")
    selected = prediction.masked_select(mask)
    if not bool((torch.isfinite(selected) & (selected > 0)).all()):
        raise FloatingPointError("Supervised prediction is not finite positive depth")
    safe_prediction = torch.where(mask, prediction, torch.ones_like(prediction))
    safe_target = torch.where(mask, target, torch.ones_like(target))
    return (safe_prediction.log() - safe_target.log()).abs().masked_select(mask).mean(), pixels


def require_model_geometry(config, technique, geometry):
    width, height = geometry["output_size_width_height"]
    if width * height > 65536 or max(width, height) > 256:
        raise ValueError("Prepare research pairs with maximum side 256; no implicit geometry resampling")
    if technique == "transformer":
        patch = config["patch_size"]
        tokens = ((width + patch - 1) // patch) * ((height + patch - 1) // patch)
        if tokens > config["max_tokens"]:
            raise ValueError("Prepared geometry exceeds the transformer's padded token budget")


def train_model(request: TrainRequest):
    runtime = authorize("train")
    authorize("model")
    if request.dataset is None:
        raise ValueError("Supply the dataset.json from a completed authorized preparation")
    partition = episode_partition(request.train_episodes, request.tuning_episodes, request.holdout_episodes)
    config = parameters(request.technique, request.model_parameters)
    heavy_execution("train")
    import numpy as np
    import torch
    from .data import PreparedPairs
    from .models import ModelRequest, create_model
    from .checkpoints import save_checkpoint

    pairs = PreparedPairs(project_path(ROOT, request.dataset), partition, "train", np)
    require_model_geometry(config, request.technique, pairs.document["geometry"])
    with operation_run("train", request, inputs=pairs.input_paths,
                       parents=[pairs.producer_path]) as (directory, record):
        record.update(technique=request.technique, model_config=config, episode_partition=partition,
                      dataset_sha256=pairs.dataset_sha256, data_contract=pairs.data_contract,
                      training_samples=len(pairs), initialization="random_no_pretrained_weights",
                      optimizer="AdamW", optimization={"bf16_autocast": request.bf16_autocast},
                      epochs=[], optimizer_steps=0, checkpoints=[],
                      training_executed=True, prediction_executed=False,
                      selection_note="Fixed epochs; no tuning/final-holdout loader or automatic model selection")
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                         stage="learning_experiments_train")
        model = optimizer = image = target = valid = prediction = loss = state = None
        try:
            with lease:
                try:
                    torch.cuda.set_device(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    if request.bf16_autocast and not torch.cuda.is_bf16_supported(including_emulation=False):
                        raise RuntimeError("BF16 requires native GPU support")
                    torch.manual_seed(request.seed)
                    model = create_model(ModelRequest(
                        technique=request.technique, seed=request.seed, parameters=config,
                        train_episodes=tuple(request.train_episodes),
                        tuning_episodes=tuple(request.tuning_episodes),
                        holdout_episodes=tuple(request.holdout_episodes)), operation="train").to("cuda:0").train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate,
                                                  weight_decay=request.weight_decay)
                    loader = torch.utils.data.DataLoader(
                        _TensorPairs(pairs, torch), batch_size=request.batch_size, shuffle=True,
                        num_workers=0, drop_last=False,
                        generator=torch.Generator().manual_seed(request.seed))
                    record["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
                    for epoch in range(1, request.epochs + 1):
                        total_error, total_pixels = 0.0, 0
                        for image, target, valid, episodes in loader:
                            lease.require_headroom()
                            image, target, valid = (value.to("cuda:0") for value in (image, target, valid))
                            optimizer.zero_grad(set_to_none=True)
                            context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                                       if request.bf16_autocast else nullcontext())
                            with context:
                                prediction = model(image, episodes=episodes.tolist(), operation="train")
                            with torch.autocast(device_type="cuda", enabled=False):
                                loss, pixels = masked_loss(prediction, target, valid, torch)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Nonfinite log-depth training loss")
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                            optimizer.step()
                            record["optimizer_steps"] += 1
                            total_error += float(loss.detach().item()) * pixels
                            total_pixels += pixels
                            lease.require_headroom()
                        record["epochs"].append({"epoch": epoch, "optimizer_steps": record["optimizer_steps"],
                                                 "training_log_depth_mae": total_error / total_pixels,
                                                 "supervised_pixels": total_pixels})
                        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                        save_checkpoint(directory, record, state, epoch=epoch,
                                        optimizer_steps=record["optimizer_steps"], torch=torch)
                        state = None
                        write_json(directory / "run.json", record)
                    torch.cuda.synchronize(0)
                finally:
                    if torch.cuda.is_initialized():
                        record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                        model = optimizer = image = target = valid = prediction = loss = state = None
                        torch.cuda.empty_cache()
        finally:
            record["gpu"] = lease.report()
    return directory / record["checkpoint"]
