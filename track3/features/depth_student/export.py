"""Export a student checkpoint to ONNX and verify it against PyTorch with onnxruntime.

The graph takes float32 RGB NCHW in [0,1] with dynamic batch/height/width and returns
depth in metres N1HW. Parity is measured on real prepared tuning samples.
"""

from dataclasses import dataclass

from track3.paths import project_path
from track3.provenance import sha256_file
from ._common import ROOT, authorize, heavy_execution, operation_run, read_json


@dataclass(frozen=True)
class ExportStudentRequest:
    checkpoint: str | None = None
    dataset: str | None = None
    opset: int = 18
    samples: int = 4

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if type(self.opset) is not int or not 11 <= self.opset <= 21 or type(self.samples) is not int or not 1 <= self.samples <= 64:
            raise ValueError("opset must be in [11, 21] and samples in [1, 64]")


def load_any_checkpoint(path, torch):
    """Student checkpoints natively; the coursework scratch-CNN checkpoint is wrapped as arch=coursework."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("kind") == "depth_student.checkpoint":
        from .predict import load_checkpoint
        return load_checkpoint(path, torch)
    if raw.get("architecture") == "coursework-depth-cnn-v1" and "state_dict" in raw and "model_config" in raw:
        config = {"width": raw["model_config"]["base_channels"], "dropout": raw["model_config"]["dropout"]}
        training = raw.get("training_request", {})
        partition = {"train_episodes": training.get("train_episodes"), "tuning_episodes": [39, 42], "holdout_episodes": [43, 44]}
        geometry = raw["geometry"]
        return {"arch": "coursework", "architecture": config, "state_dict": {"net." + k: v for k, v in raw["state_dict"].items()},
                "episode_partition": partition, "data_contract": {"geometry": geometry}, "pretrained_weights": False,
                "source": "scratch_depth coursework checkpoint"}
    raise ValueError("Unsupported checkpoint kind for export")


def export_student(request: ExportStudentRequest):
    authorize("export")
    if request.checkpoint is None:
        raise ValueError("Supply a depth_student checkpoint")
    heavy_execution("export")
    import numpy as np
    import torch
    import onnx
    import onnxruntime
    from ._models import build
    from .predict import load_checkpoint

    checkpoint_path = project_path(ROOT, request.checkpoint)
    payload = load_any_checkpoint(checkpoint_path, torch)
    model = build(payload["arch"], payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    inputs = [checkpoint_path]
    pairs = None
    if request.dataset:
        from track3.features.learning_experiments.data import PreparedPairs
        from ._common import partition as standard_partition
        pairs = PreparedPairs(project_path(ROOT, request.dataset), standard_partition(), "tuning", np)
        inputs += pairs.input_paths[:1]
    with operation_run("export", request, inputs=inputs, parents=[checkpoint_path.parent.parent / "run.json"]) as (directory, record):
        width, height = payload["data_contract"]["geometry"]["output_size_width_height"]
        example = torch.rand(1, 3, height, width)
        destination = directory / "depth_student.onnx"
        batch, h_dim, w_dim = torch.export.Dim("batch"), torch.export.Dim("height", min=16), torch.export.Dim("width", min=16)
        exported = torch.onnx.export(model, (example,), input_names=["rgb"], output_names=["depth_m"],
                                     opset_version=request.opset, dynamo=True, dynamic_shapes={"image": {0: batch, 2: h_dim, 3: w_dim}},
                                     optimize=True, verbose=False)
        exported.save(str(destination))
        onnx.checker.check_model(str(destination))
        graph = onnx.load(str(destination))
        session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        record.update(arch=payload["arch"], architecture=payload["architecture"], opset=request.opset,
                      onnx_ir_version=graph.ir_version, onnx_sha256=sha256_file(destination),
                      onnx_bytes=destination.stat().st_size, input="rgb float32 NCHW in [0,1], dynamic N/H/W",
                      output="depth_m float32 N1HW metres", trained_geometry_width_height=[width, height],
                      parameter_count=sum(p.numel() for p in model.parameters()), parity=[])
        tests = []
        if pairs is not None:
            for index in range(min(request.samples, len(pairs))):
                sample = pairs[index]
                tests.append((f"tuning sample {index}", torch.from_numpy(sample["rgb"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0))
        tests.append(("random same size", example))
        tests.append(("random 1.5x size", torch.rand(1, 3, int(height * 1.5), int(width * 1.5))))
        worst = 0.0
        for name, image in tests:
            with torch.inference_mode():
                reference = model(image).numpy()
            produced = session.run(None, {"rgb": image.numpy()})[0]
            abs_diff = float(np.abs(produced - reference).max())
            rel_diff = float((np.abs(produced - reference) / np.maximum(reference, 1e-3)).max())
            worst = max(worst, rel_diff)
            record["parity"].append({"case": name, "shape": list(image.shape), "max_abs_diff_m": abs_diff, "max_rel_diff": rel_diff})
        record["parity_max_rel_diff"] = worst
        record["parity_ok"] = worst < 1e-3
        if not record["parity_ok"]:
            raise ValueError(f"ONNX output differs from PyTorch beyond tolerance (max rel {worst:.2e})")
    return destination
