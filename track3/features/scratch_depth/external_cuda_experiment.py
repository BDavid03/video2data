"""One real-batch CUDA loss experiment in an explicitly approved external Python."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

from track3.gpu import GpuLease
from track3.paths import output_path, project_path
from track3.provenance import run_fields, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parent
CONFIG = FEATURE / "external-compiler.json"


@dataclass(frozen=True)
class ExternalCudaRequest:
    dataset: str
    seed: int = 31

    def __post_init__(self):
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("Supply the explicit prepared dataset.json")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")


def _authorize():
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    if config.get("execution", {}).get("external_cuda_authorized") is not True:
        raise PermissionError("External CUDA experiment is disabled")
    permitted = Path("Z:/hugging-face/video2data" if os.name == "nt"
                     else "/mnt/z/hugging-face/video2data").resolve()
    if ROOT != permitted:
        raise PermissionError("Run from the canonical Z: checkout")
    approved = Path("Z:/hugging-face/locals/.venv/Scripts/python.exe")
    if config.get("python") != approved.as_posix() or Path(sys.executable).resolve() != approved.resolve():
        raise PermissionError("Use the explicitly approved read-only external interpreter")
    for name, folder in {
        "TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "extensions", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "HF_HOME": "huggingface",
        "XDG_CACHE_HOME": "xdg", "PYTHONPYCACHEPREFIX": "pycache",
    }.items():
        path = output_path(ROOT, f".cache/external-cuda/{folder}")
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    sys.dont_write_bytecode = True
    os.environ.update(MAX_JOBS="2", TORCH_CUDA_ARCH_LIST="12.0",
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")


def run_external_cuda_experiment(request: ExternalCudaRequest):
    """Persist success or failure; never change the external environment."""
    _authorize()
    dataset = project_path(ROOT, request.dataset)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/scratch_depth/external_cuda_experiment/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "kind": "scratch_depth.external_cuda_experiment",
              "run_id": run_id, "status": "running", "request": asdict(request),
              "research_only": True, "submission_candidate": False, "training_executed": False,
              "tests_run": False, "validation_run": False, "optimizer_steps": 0, "artifacts": {},
              "kernel_executed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "interpretation": "One real training batch; no general speedup or accuracy claim"}
    write_json(directory / "run.json", record)
    lease = None
    model = prediction = loss = image = target = valid = reference = candidate = None
    started = time.perf_counter()
    try:
        record.update(run_fields(ROOT, "scratch_depth", "external_cuda_experiment", asdict(request),
                                 inputs=[dataset, CONFIG], parents=[dataset.parent / "run.json"]))
        sources = [Path(__file__), CONFIG, FEATURE / "_network.py", FEATURE / "_cuda_loss.py",
                   FEATURE / "cuda/masked_log_depth.cpp", FEATURE / "cuda/masked_log_depth.cu",
                   ROOT / "track3/features/learning_experiments/data.py", ROOT / "track3/gpu.py"]
        record["source_hashes"] = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sources}
        import numpy as np
        import torch
        from torch.utils.cpp_extension import load
        from track3.features.learning_experiments.data import PreparedPairs
        from ._cuda_loss import _autograd_function, _require_inputs
        from ._network import CourseworkDepthCNN, masked_log_depth_loss

        split = json.loads((ROOT / "configs/splits/public-v1.json").read_text(encoding="utf-8-sig"))
        partition = {"dataset_revision": split["dataset_revision"], "split_id": split["split_id"],
                     "train_episodes": [0, 11, 12, 18, 21, 23, 26, 31],
                     "tuning_episodes": [39, 42], "holdout_episodes": [43, 44]}
        pairs = PreparedPairs(dataset, partition, "train", np)
        sample = pairs[0]
        sample_path = project_path(ROOT, dataset.parent / sample["row"]["artifact"])
        record["inputs"].append({"path": sample_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(sample_path)})
        record.update(sample=sample["row"], data_contract=pairs.data_contract, episode_partition=partition,
                      torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                      python=sys.executable, tools={name: shutil.which(name) for name in ("nvcc", "cl", "ninja")},
                      cuda_home=os.environ.get("CUDA_HOME"), model_config={"base_channels": 8, "dropout": 0.0})
        lease = GpuLease(ROOT, 2048, 4096, stage="scratch_depth_external_cuda")
        with lease:
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(request.seed)
            model = CourseworkDepthCNN(**record["model_config"]).to("cuda:0").train()
            image = torch.from_numpy(sample["rgb"]).permute(2, 0, 1).unsqueeze(0).to("cuda:0").float() / 255.0
            target = torch.from_numpy(sample["depth_m"]).unsqueeze(0).unsqueeze(0).to("cuda:0")
            valid = torch.from_numpy(sample["valid"]).unsqueeze(0).unsqueeze(0).to("cuda:0")

            def measure(loss_function):
                model.zero_grad(set_to_none=True)
                lease.require_headroom()
                torch.cuda.synchronize(0)
                tick = time.perf_counter()
                output = model(image).contiguous()
                output.retain_grad()
                value = loss_function(output, target, valid)
                value.backward()
                torch.cuda.synchronize(0)
                elapsed = time.perf_counter() - tick
                tensors = [output, output.grad, *[p.grad for p in model.parameters()]]
                if not bool(torch.isfinite(value)) or any(t is None or not bool(torch.isfinite(t).all()) for t in tensors):
                    raise FloatingPointError("Nonfinite output, loss or gradient")
                lease.require_headroom()
                return float(value.detach().item()), [t.detach().cpu().clone() for t in tensors], elapsed

            reference = measure(masked_log_depth_loss)
            record.update(eager_loss=reference[0], first_eager_seconds=reference[2],
                          difference_order=["prediction", "prediction_gradient", *dict(model.named_parameters())])
            write_json(directory / "run.json", record)
            _authorize()
            identity = json.dumps([torch.__version__, record["source_hashes"]], sort_keys=True).encode()
            name = "scratch_external_cuda_" + hashlib.sha256(identity).hexdigest()[:16]
            build = output_path(ROOT, Path(".cache/external-cuda/extensions") / name)
            build.mkdir(parents=True, exist_ok=True)
            record.update(extension_name=name, build_directory=build.relative_to(ROOT).as_posix())
            tick = time.perf_counter()
            extension = load(name=name, sources=[str(FEATURE / "cuda" / f"masked_log_depth.{suffix}")
                                                for suffix in ("cpp", "cu")], build_directory=str(build),
                             with_cuda=True, extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
                             extra_cuda_cflags=["--ftz=false", *(["-Xcompiler=/Zc:preprocessor"]
                                                               if os.name == "nt" else [])], verbose=False)
            record["compile_seconds"] = time.perf_counter() - tick
            function = _autograd_function(torch)

            def cuda_loss(output, labels, mask):
                _require_inputs(torch, output, labels, mask)
                return function.apply(output, labels, mask, extension)

            candidate = measure(cuda_loss)
            record.update(kernel_executed=True, loss_absolute_difference=abs(reference[0] - candidate[0]),
                          maximum_absolute_differences=[float((left - right).abs().max().item())
                                                        for left, right in zip(reference[1], candidate[1])],
                          tolerance={"rtol": 1e-4, "atol": 1e-6})
            if (not torch.isclose(torch.tensor(reference[0]), torch.tensor(candidate[0]), rtol=1e-4, atol=1e-6)
                    or any(not torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(reference[1], candidate[1]))):
                raise ArithmeticError("Real-batch loss/gradient agreement exceeded recorded tolerance")
            record.update(eager_loss=reference[0], cuda_loss=candidate[0], first_cuda_seconds=candidate[2],
                          warm_eager_seconds=measure(masked_log_depth_loss)[2], warm_cuda_seconds=measure(cuda_loss)[2])
            record["status"] = "completed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if lease is not None:
            record["gpu"] = lease.report()
        try:
            if "torch" in locals() and torch.cuda.is_initialized():
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                model = prediction = loss = image = target = valid = reference = candidate = None
                torch.cuda.empty_cache()
        except Exception as error:
            record.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        record.update(elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "run.json", record)
    return {"status": record["status"], "run": (directory / "run.json").relative_to(ROOT).as_posix()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    document = json.loads(project_path(ROOT, parser.parse_args().request).read_text(encoding="utf-8-sig"))
    result = run_external_cuda_experiment(ExternalCudaRequest(document.get("dataset"), document.get("seed", 31)))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 1)
