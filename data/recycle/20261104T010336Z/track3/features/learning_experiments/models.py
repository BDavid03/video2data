"""Lazy original PyTorch factories. Importing this module never imports torch."""

from dataclasses import dataclass, field

from ._common import authorize, episode_partition, heavy_execution, integer
from ._config import parameters


@dataclass(frozen=True)
class ModelRequest:
    technique: str = "cnn"
    seed: int = 31
    parameters: dict = field(default_factory=dict)
    train_episodes: tuple[int, ...] = (0, 11, 12, 18, 21, 23, 26, 31)
    tuning_episodes: tuple[int, ...] = (39, 42)
    holdout_episodes: tuple[int, ...] = (43, 44)


def create_model(request: ModelRequest, *, operation="predict"):
    """Random initialization only; both model and operation flags must be true."""
    authorize("model")
    if operation not in ("train", "predict"):
        raise ValueError("Factory purpose must be train or predict")
    authorize(operation)
    if request.technique not in ("ann", "cnn", "transformer"):
        raise ValueError("Use the self-contained boosting API for gradient_boosting")
    integer(request.seed, "seed", 0, 2**32 - 1)
    config = parameters(request.technique, request.parameters)
    partition = episode_partition(request.train_episodes, request.tuning_episodes, request.holdout_episodes)
    heavy_execution(operation)
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class ResearchDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.research_metadata = {
                "technique": request.technique, "parameters": config.copy(), "seed": request.seed,
                "episode_partition": partition, "pretrained_weights": False,
                "scratch_checkpoint_admitted": False, "submission_candidate": False,
            }
            width = config["width"]
            if request.technique in ("ann", "cnn"):
                kernel = 1 if request.technique == "ann" else 3
                blocks = []
                for index in range(config["layers"]):
                    blocks.extend((nn.Conv2d(3 if index == 0 else width, width,
                                             kernel, padding=kernel // 2), nn.ReLU()))
                self.body = nn.Sequential(*blocks, nn.Conv2d(width, 1, 1))
            else:
                patch = config["patch_size"]
                self.embedding = nn.Conv2d(3, width, patch, stride=patch)
                self.position = nn.Parameter(torch.zeros(1, config["max_tokens"], width))
                nn.init.normal_(self.position, mean=0.0, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=width, nhead=config["heads"], dim_feedforward=width * 2,
                    dropout=0.0, activation="gelu", batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
                # Independently initialize cloned blocks, without pretrained weights.
                for block in self.encoder.layers:
                    nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                    nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                    nn.init.xavier_uniform_(block.linear1.weight)
                    nn.init.xavier_uniform_(block.linear2.weight)
                self.decoder = nn.ConvTranspose2d(width, 1, patch, stride=patch)

        def forward(self, image, *, episodes, operation="predict"):
            authorize("model")
            if operation not in ("train", "predict"):
                raise ValueError("Unknown forward operation")
            authorize(operation)
            if operation == "predict" and self.training:
                raise ValueError("Call eval() before prediction")
            if operation == "train" and not self.training:
                raise ValueError("Call train() before a training forward pass")
            if (image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point()
                    or not 1 <= image.shape[0] <= 8 or min(image.shape[-2:]) < 1
                    or image.shape[-2] * image.shape[-1] > 65536):
                raise ValueError("Expected floating RGB N3HW, batch <=8 and <=65536 pixels per image")
            allowed = partition["train_episodes"] if operation == "train" else (
                partition["train_episodes"] + partition["tuning_episodes"] + partition["holdout_episodes"])
            if (not isinstance(episodes, (list, tuple)) or len(episodes) != image.shape[0]
                    or any(type(e) is not int or e not in allowed for e in episodes)):
                raise ValueError("Batch episodes violate the declared episode partition")
            if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
                raise ValueError("RGB must be finite and normalized to [0,1]")
            if request.technique == "transformer":
                height, width = image.shape[-2:]
                patch = config["patch_size"]
                rows, columns = (height + patch - 1) // patch, (width + patch - 1) // patch
                tokens = rows * columns
                if tokens > config["max_tokens"]:
                    raise ValueError("Padded patch grid exceeds the token budget")
                padded = functional.pad(image, (0, columns * patch - width, 0, rows * patch - height))
                encoded = self.embedding(padded).flatten(2).transpose(1, 2)
                encoded = self.encoder(encoded + self.position[:, :tokens])
                grid = encoded.transpose(1, 2).reshape(image.shape[0], config["width"], rows, columns)
                logits = self.decoder(grid)[..., :height, :width]
            else:
                logits = self.body(image)
            depth = functional.softplus(logits) + 0.001
            if not bool(torch.isfinite(depth).all()):
                raise ValueError("Model produced nonfinite depth")
            return depth

    # Preserve the caller's CPU random stream; all construction is on CPU.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(request.seed)
        model = ResearchDepth()
    if operation == "predict":
        model.eval()
    return model
