"""Train the original CNN from random weights on prepared development pairs."""

from dataclasses import asdict, dataclass
import math
import os

from track3.gpu import GpuLease
from track3.provenance import inherited_provenance, input_file, read_run, sha256_file, write_json
from ._common import (ARCHITECTURE, FEATURE, ROOT, authorize, local_path, operation_run,
                      positive_integer, read_json, relative, within)
from ._checkpoint import DATA_FIELDS, MODEL_CONFIG, require_finite_state
from ._optimization import (OPTION_FIELDS, configure_training, forward_context,
                            optimization_options, require_compile_authorization, training_loss)


@dataclass(frozen=True)
class TrainDepthRequest:
    dataset: str | None = None
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    seed: int = 31
    brightness_jitter: float = 0.1
    bf16_autocast: bool = False
    channels_last: bool = False
    torch_compile: bool = False
    fused_adam: bool = False
    loss_backend: str = "torch"

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        for name in OPTION_FIELDS:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean")
        if self.loss_backend not in ("torch", "cuda"):
            raise ValueError("loss_backend must be 'torch' or 'cuda'")
        positive_integer(self.epochs, "epochs", 1, 1000)
        positive_integer(self.batch_size, "batch_size", 1, 8)
        positive_integer(self.seed, "seed", 0, 2**31 - 1)
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be finite and in (0, 0.1]")
        if not math.isfinite(self.weight_decay) or not 0 <= self.weight_decay <= 1:
            raise ValueError("weight_decay must be finite and in [0, 1]")
        if not math.isfinite(self.brightness_jitter) or not 0 <= self.brightness_jitter <= 0.5:
            raise ValueError("brightness_jitter must be finite and in [0, 0.5]")


class _TrainingPairs:
    """Map-style dataset; zero workers, explicit files, no downloads or split mixing."""

    def __init__(self, path, np, torch):
        self.np, self.torch = np, torch
        self.directory = path.parent
        self.document = read_json(path)
        producer = read_run(ROOT, path.parent / "run.json", kind="scratch_depth.prepare")
        inherited_provenance([self.document])
        digest = sha256_file(path)
        if (producer.get("dataset_sha256") != digest
                or producer.get("artifacts", {}).get(path.name) != digest
                or within(path.parent, producer["dataset"]) != path
                or local_path(self.document["producer_run"]) != path.parent / "run.json"
                or self.document["provenance"] != producer["provenance"]):
            raise ValueError("Dataset differs from its tagged preparation run")
        split = read_json("configs/splits/public-v1.json")
        revision = read_json("configs/sources.lock.json")["dataset"]["revision"]
        doc = self.document
        if (doc.get("kind") != "scratch_depth_training_pairs" or doc.get("status") != "completed"
                or doc.get("source_split") != "public" or doc.get("subset") != "development"
                or doc.get("target_kind") != "moge2_pseudo_depth" or doc.get("camera") != "ego_cam_a"
                or doc["dataset_revision"] != revision or split["dataset_revision"] != revision
                or doc["split_id"] != split["split_id"]):
            raise ValueError("Training requires prepared, pinned public development pseudo-depth pairs")
        self.rows = doc["samples"]
        if not self.rows:
            raise ValueError("Training dataset is empty")
        seen = set()
        for row in self.rows:
            key = (row["episode_index"], row["frame_index"])
            if key in seen or key[0] == 41 or key[0] not in split["development_episode_indices"]:
                raise ValueError("Duplicate sample or held-out episode in training data")
            seen.add(key)
            if sha256_file(within(self.directory, row["artifact"])) != row["sha256"]:
                raise ValueError("Training sample bytes no longer match the dataset manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        np, torch = self.np, self.torch
        path = within(self.directory, self.rows[index]["artifact"])
        if sha256_file(path) != self.rows[index]["sha256"]:
            raise ValueError("Training sample changed after dataset admission")
        with np.load(path, allow_pickle=False) as arrays:
            rgb = arrays["rgb"].copy()
            depth = arrays["depth_m"].copy()
            valid = arrays["valid"].copy()
            k = arrays["input_intrinsics_pixels"].copy()
        width, height = self.document["geometry"]["output_size_width_height"]
        if (rgb.shape != (height, width, 3) or rgb.dtype != np.uint8
                or depth.shape != (height, width) or depth.dtype.kind != "f"
                or valid.shape != depth.shape or valid.dtype != np.dtype("bool")
                or k.shape != (3, 3)
                or not np.allclose(k, self.document["geometry"]["input_intrinsics_pixels"], rtol=1e-5, atol=1e-5)):
            raise ValueError("Training sample geometry or RGB encoding is inconsistent")
        # Preserve scalar-bound precision and recheck after the training cast.
        # Excluded labels are replaced before conversion, avoiding overflow.
        minimum, maximum = self.document["minimum_depth_m"], self.document["maximum_depth_m"]
        depth64 = depth.astype(np.float64)
        valid &= (np.isfinite(depth64) & (depth64 > 0)
                  & (depth64 >= minimum) & (depth64 <= maximum))
        depth = np.where(valid, depth, np.nan).astype(np.float32)
        depth64 = depth.astype(np.float64)
        valid &= (np.isfinite(depth64) & (depth64 > 0)
                  & (depth64 >= minimum) & (depth64 <= maximum))
        if int(valid.sum()) < 64:
            raise ValueError("Training sample has fewer than 64 valid depth pixels")
        return (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(depth).unsqueeze(0),
                torch.from_numpy(valid).unsqueeze(0))


def train_depth(request: TrainDepthRequest):
    # authorize() creates cache directories: refuse compile opt-ins before it.
    require_compile_authorization(request, read_json(FEATURE / "runtime.json"))
    runtime = authorize("train")
    require_compile_authorization(request, runtime)
    if request.dataset is None:
        raise ValueError("Supply dataset.json from an explicitly authorized preparation operation")
    import numpy as np
    import torch
    from ._network import CourseworkDepthCNN, masked_log_depth_loss

    dataset_path = local_path(request.dataset)
    pairs = _TrainingPairs(dataset_path, np, torch)
    with operation_run("train", request, inputs=[dataset_path],
                       parents=[dataset_path.parent / "run.json"]) as (directory, record):
        record["inputs"] += [input_file(ROOT, within(dataset_path.parent, row["artifact"]))
                             for row in pairs.rows]
        (directory / "checkpoints").mkdir()
        record.update(dataset=relative(dataset_path), dataset_sha256=sha256_file(dataset_path),
                      dataset_revision=pairs.document["dataset_revision"], split_id=pairs.document["split_id"],
                      target_kind=pairs.document["target_kind"], initialization="random, no pretrained weights",
                      training_samples=len(pairs), epochs=[], checkpoints=[], optimizer_steps=0,
                      model_config=dict(MODEL_CONFIG), model_source=input_file(ROOT, FEATURE / "_network.py"),
                      optimization=optimization_options(request),
                      optimization_sources=[input_file(ROOT, FEATURE / "_optimization.py"),
                                            input_file(ROOT, FEATURE / "train.py")],
                      data_contract={key: pairs.document[key] for key in DATA_FIELDS},
                      validation_note="No validation loader, held-out pass, early stopping or best-model selection")
        if request.loss_backend == "cuda":
            # Bind the adapter and kernel sources without loading
            # or compiling the backend outside the admitted lease.
            cuda_sources = [FEATURE / "_cuda_loss.py", *sorted(
                path for path in FEATURE.rglob("*")
                if path.is_file() and path.suffix in {".cpp", ".cu"})]
            record["optimization_sources"] += [input_file(ROOT, path) for path in cuda_sources]
        for source in record["optimization_sources"]:
            if record["source_hashes"].get(source["path"]) != source["sha256"]:
                raise ValueError("Optimization source is missing or changed in the run source inventory")
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                         stage="scratch_depth_train")
        model = forward = optimizer = image = target = valid = prediction = loss = None
        try:
            with lease:
                try:
                    require_compile_authorization(request, read_json(FEATURE / "runtime.json"))
                    torch.cuda.set_device(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    torch.manual_seed(request.seed)
                    generator = torch.Generator().manual_seed(request.seed)
                    loader = torch.utils.data.DataLoader(pairs, batch_size=request.batch_size, shuffle=True,
                                                         num_workers=0, generator=generator, drop_last=False)
                    model_config = dict(MODEL_CONFIG)
                    model = CourseworkDepthCNN(**model_config).to("cuda:0").train()
                    forward, optimizer = configure_training(model, request, torch)
                    record["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
                    lease.require_headroom()
                    for epoch in range(1, request.epochs + 1):
                        total_loss, total_pixels = 0.0, 0
                        for image, target, valid in loader:
                            lease.require_headroom()
                            image, target, valid = (value.to("cuda:0") for value in (image, target, valid))
                            if request.brightness_jitter:
                                gain = 1 + (torch.rand((image.shape[0], 1, 1, 1), device=image.device) * 2 - 1) * request.brightness_jitter
                                image = (image * gain).clamp(0, 1)
                                del gain
                            if request.channels_last:
                                image = image.contiguous(memory_format=torch.channels_last)
                            optimizer.zero_grad(set_to_none=True)
                            with forward_context(request, torch):
                                prediction = forward(image)
                            loss = training_loss(prediction, target, valid, request, torch,
                                                 masked_log_depth_loss)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Nonfinite training loss")
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                            optimizer.step()
                            pixels = int(valid.sum().item())
                            total_loss += float(loss.item()) * pixels
                            total_pixels += pixels
                            record["optimizer_steps"] += 1
                            lease.require_headroom()
                        record["epochs"].append({"epoch": epoch, "training_log_depth_mae": total_loss / total_pixels,
                                                 "supervised_pixels": total_pixels})
                        checkpoint = directory / "checkpoints" / f"epoch_{epoch:04d}.pt"
                        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
                        require_finite_state(state, torch)
                        payload = {
                            "schema_version": 2, "architecture": ARCHITECTURE, "model_config": model_config,
                            "provenance": record["provenance"],
                            "state_dict": state, "model_source": record["model_source"],
                            "optimization": record["optimization"],
                            "optimization_sources": record["optimization_sources"],
                            "training_run_id": record["run_id"], "training_run": relative(directory / "run.json"),
                            "coursework_hashes": record["coursework_hashes"],
                            "trained_epochs": epoch, "optimizer_steps": record["optimizer_steps"],
                            "training_request": asdict(request), "dataset_sha256": record["dataset_sha256"],
                            "dataset_revision": pairs.document["dataset_revision"], "split_id": pairs.document["split_id"],
                            "geometry": pairs.document["geometry"], "max_image_side": pairs.document["max_image_side"],
                            "calibration_sha256": pairs.document["calibration_sha256"],
                            "minimum_depth_m": pairs.document["minimum_depth_m"],
                            "maximum_depth_m": pairs.document["maximum_depth_m"],
                            "target_kind": pairs.document["target_kind"], "pretrained_weights": False,
                        }
                        temporary = checkpoint.with_suffix(".pt.tmp")
                        try:
                            with temporary.open("xb") as stream:
                                torch.save(payload, stream)
                            os.replace(temporary, checkpoint)
                        finally:
                            temporary.unlink(missing_ok=True)
                        record["checkpoints"].append({"epoch": epoch, "artifact": checkpoint.relative_to(directory).as_posix(),
                                                      "sha256": sha256_file(checkpoint), "optimizer_steps": record["optimizer_steps"]})
                        record["checkpoint"] = checkpoint.relative_to(directory).as_posix()
                        write_json(directory / "run.json", record)
                        print(f"Scratch depth epoch {epoch}/{request.epochs}: training loss {total_loss / total_pixels:.6f}", flush=True)
                    torch.cuda.synchronize(0)
                finally:
                    if torch.cuda.is_initialized():
                        record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                        model = forward = optimizer = image = target = valid = prediction = loss = None
                        torch.cuda.empty_cache()
        finally:
            record["gpu"] = lease.report()
    return directory / record["checkpoint"]
