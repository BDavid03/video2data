"""Explicit compiler experiments on one real training pair; no installation."""

from dataclasses import dataclass, replace
import os
import shutil
import time

from track3.gpu import GpuLease
from track3.provenance import sha256_file
from ._common import (FEATURE, ROOT, authorize, local_path, operation_run,
                      positive_integer, read_json, relative)


@dataclass(frozen=True)
class CompilerExperimentRequest:
    dataset: str
    seed: int = 31
    backend: str = "torch_inductor"

    def __post_init__(self):
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("Supply an explicit prepared dataset.json")
        positive_integer(self.seed, "seed", 0, 2**31 - 1)
        if self.backend not in ("torch_inductor", "cuda_loss"):
            raise ValueError("Unknown compiler experiment backend")


def _authorize(request):
    flag = ("pytorch_compile_authorized" if request.backend == "torch_inductor"
            else "cuda_compile_authorized")
    if read_json(FEATURE / "runtime.json")["execution"].get(flag) is not True:
        raise PermissionError(f"Compiler experiment requires {flag}=true")
    return authorize("compile_experiment")


def _attempt(request):
    runtime = _authorize(request)
    import numpy as np
    import torch
    from track3.features.learning_experiments.data import PreparedPairs
    from ._network import CourseworkDepthCNN, masked_log_depth_loss

    split = read_json("configs/splits/public-v1.json")
    partition = {"dataset_revision": split["dataset_revision"], "split_id": split["split_id"],
                 "train_episodes": [0, 11, 12, 18, 21, 23, 26, 31],
                 "tuning_episodes": [39, 42], "holdout_episodes": [43, 44]}
    path = local_path(request.dataset)
    pairs = PreparedPairs(path, partition, "train", np)
    sample = pairs[0]
    sample_path = local_path(path.parent / sample["row"]["artifact"])
    directory = None
    try:
        with operation_run("compile_experiment", request, inputs=[path, sample_path],
                           parents=[pairs.producer_path]) as (directory, record):
            record.update(research_only=True, submission_candidate=False,
                          scratch_checkpoint_admitted=False, training_executed=False,
                          optimizer_steps=0, model_config={"base_channels": 8, "dropout": 0.0},
                          dataset_sha256=sha256_file(path), data_contract=pairs.data_contract,
                          episode_partition=partition, sample=sample["row"], batch_size=1,
                          experiment_status="running", backend_executed=False,
                          compiler_tools={name: shutil.which(name) for name in ("cl", "nvcc", "ninja")},
                          cuda_home=os.environ.get("CUDA_HOME"),
                          comparison_note="One real training batch; no accuracy score or checkpoint")
            lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                             stage=f"scratch_depth_{request.backend}")
            model = forward = image = target = valid = prediction = loss = None
            reference = candidate = None
            try:
                with lease:
                    try:
                        _authorize(request)
                        torch.cuda.set_device(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        torch.manual_seed(request.seed)
                        model = CourseworkDepthCNN(**record["model_config"]).to("cuda:0").train()
                        image = (torch.from_numpy(sample["rgb"]).permute(2, 0, 1)
                                 .unsqueeze(0).to("cuda:0").float() / 255.0)
                        target = torch.from_numpy(sample["depth_m"]).unsqueeze(0).unsqueeze(0).to("cuda:0")
                        valid = torch.from_numpy(sample["valid"]).unsqueeze(0).unsqueeze(0).to("cuda:0")

                        def measure(call, loss_function):
                            model.zero_grad(set_to_none=True)
                            lease.require_headroom()
                            torch.cuda.synchronize(0)
                            started = time.perf_counter()
                            output = call(image)
                            value = loss_function(output, target, valid)
                            value.backward()
                            torch.cuda.synchronize(0)
                            elapsed = time.perf_counter() - started
                            gradients = [parameter.grad.detach().cpu().clone()
                                         for parameter in model.parameters()]
                            if (not bool(torch.isfinite(output).all())
                                    or not bool(torch.isfinite(value))
                                    or any(not bool(torch.isfinite(g).all()) for g in gradients)):
                                raise FloatingPointError("Compiler experiment produced nonfinite values")
                            lease.require_headroom()
                            return {"loss": float(value.detach().item()), "seconds": elapsed,
                                    "output": output.detach().cpu(), "gradients": gradients}

                        reference = measure(model, masked_log_depth_loss)
                        record["eager"] = {key: reference[key] for key in ("loss", "seconds")}
                        _authorize(request)
                        started = time.perf_counter()
                        if request.backend == "torch_inductor":
                            forward = torch.compile(model, backend="inductor", mode="default",
                                                    fullgraph=False, dynamic=False)
                            candidate = measure(forward, masked_log_depth_loss)
                        else:
                            from ._cuda_loss import masked_log_depth_loss as cuda_loss
                            candidate = measure(model, cuda_loss)
                        record.update(backend_executed=True, experiment_status="completed",
                                      candidate={key: candidate[key] for key in ("loss", "seconds")},
                                      candidate_total_seconds=time.perf_counter() - started,
                                      loss_absolute_difference=abs(candidate["loss"] - reference["loss"]),
                                      output_max_absolute_difference=float((candidate["output"] - reference["output"]).abs().max().item()),
                                      parameter_gradient_max_absolute_difference=max(
                                          float((left - right).abs().max().item())
                                          for left, right in zip(candidate["gradients"], reference["gradients"])))
                    finally:
                        record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                        model = forward = image = target = valid = prediction = loss = None
                        reference = candidate = None
                        torch.cuda.empty_cache()
            except Exception:
                record["experiment_status"] = "failed"
                raise
            finally:
                record["gpu"] = lease.report()
    except Exception as error:
        return {"backend": request.backend, "status": "failed",
                "run": relative(directory / "run.json") if directory is not None else None,
                "error": f"{type(error).__name__}: {error}"}
    return {"backend": request.backend, "status": "completed", "run": relative(directory / "run.json")}


def run_compiler_experiments(request: CompilerExperimentRequest):
    """Attempt both backends independently, preserving each failed owning run."""
    requests = [replace(request, backend=name) for name in ("torch_inductor", "cuda_loss")]
    for attempt in requests:
        _authorize(attempt)
    return [_attempt(attempt) for attempt in requests]
