"""Small local torch probe. This is not a simulator or a learned robot policy.

No packages/models downloaded. CUDA only runs when explicitly requested.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


def device_sample():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}


def _peak_used_mib(samples):
    """Largest sampled memory.used of GPU 0, or None when nvidia-smi was unavailable."""
    return max((int(fields[-1].strip()) for sample in samples if isinstance(sample, list)
                for fields in (row.split(",") for row in sample)
                if len(fields) == 4 and fields[0].strip() == "0"), default=None)


def _probe(torch, device, checkpoint, report):
    """Inference, one Adam update and an exact checkpoint round trip on a tiny MLP."""
    def mlp():
        return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)).to(device)

    torch.set_num_threads(2)
    torch.manual_seed(31)
    report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected interpreter")
        report.update(device_name=torch.cuda.get_device_name(0),
                      device_capability=list(torch.cuda.get_device_capability(0)),
                      torch_arch_list=torch.cuda.get_arch_list())
        free, total = torch.cuda.mem_get_info()
        report.update(cuda_free_before_bytes=free, cuda_total_bytes=total)
        if free < 3 * 1024**3:
            raise RuntimeError("Need at least 3 GiB free for 1 GiB probe + 2 GiB headroom")
        torch.cuda.reset_peak_memory_stats()
        report["total_device_memory_samples"].append(device_sample())
    model = mlp()
    inputs = torch.randn(32, 16, device=device)
    targets = torch.randn(32, 4, device=device)
    with torch.inference_mode():
        prediction = model(inputs)
        assert tuple(prediction.shape) == (32, 4)
        assert torch.isfinite(prediction).all().item()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    assert torch.isfinite(loss).item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "seed": 31, "synthetic": True}, checkpoint)
    restored = mlp()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(state["model"])
    torch.optim.Adam(restored.parameters(), lr=0.001).load_state_dict(state["optimizer"])
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    if device == "cuda":
        torch.cuda.synchronize()
        report.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        report["total_device_memory_samples"].append(device_sample())
    report.update(status="passed", inference_shape=[32, 4], training_updates=1,
                  checkpoint_reload_identical=True, loss=float(loss.detach().cpu()),
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    report = {"kind": "synthetic_torch_runtime_smoke", "simulator_tested": False,
              "device_requested": args.device, "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(),
              "status": "failed", "total_device_memory_samples": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop_monitor = threading.Event()
    monitor = None
    if args.device == "cuda":
        def sample_loop():
            while not stop_monitor.is_set():
                report["total_device_memory_samples"].append(device_sample())
                stop_monitor.wait(0.1)
        monitor = threading.Thread(target=sample_loop, daemon=True)
        monitor.start()
    try:
        import torch
        _probe(torch, args.device, args.checkpoint, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_monitor.set()
        if monitor is not None:
            monitor.join(timeout=11)
    report["peak_sampled_total_device_used_mib"] = _peak_used_mib(report["total_device_memory_samples"])
    report["elapsed_seconds"] = time.monotonic() - start
    report["memory_measurement_limit"] = "nvidia-smi samples every 100ms plus command latency; short peaks may be missed. Includes other applications and display."
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
