"""D1 GPU operations, with unchanged D0 training/evaluation helpers."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

from track3.features.dynamic_feasibility import handler as d0
from track3.gpu import GpuLease
from track3.provenance import run_fields, write_json
from track3.runs import ROOT, recorded, runtime_flag

FEATURE = Path(__file__).parent
QUEUE = FEATURE / "queue.json"
OUTPUT = d0.OUTPUT / "d1/sizing-fixed"


@dataclass(frozen=True)
class Request:
    seed: int
    budget: dict
    maximum_seconds: float
    arm: str = "D1"


def environment(flag):
    """Require this operation's authorization before entering the existing runtime."""
    runtime_flag(FEATURE, flag, "Dynamic scaling")
    return d0.environment()


@contextmanager
def gpu(record, maximum_seconds):
    """Lease the GPU and record all operation time, including compilation and failure."""
    main_root = next(parent for parent in ROOT.parents if (parent / ".git").is_dir())
    if main_root != ROOT and (main_root / ".runtime/gpu-stage.lock").exists():
        raise RuntimeError("Main-checkout GPU lease already exists")
    started = time.monotonic()
    with GpuLease(ROOT, 1024, 4096, stage="dynamic_scaling") as lease:
        record["gpu_started_epoch"] = time.time()
        try:
            yield time.time() + maximum_seconds
            lease.require_headroom()
        finally:
            record.update(gpu=lease.report(), gpu_seconds=time.monotonic() - started)


def probe(worlds, maximum_seconds, identity):
    """Run one throughput-only sizing attempt and retain failures without a retry."""
    env = environment("probe_authorized")
    directory = OUTPUT / "probes" / f"worlds-{worlds}"
    record = run_fields(ROOT, "dynamic_feasibility", "sizing_probe",
                        {"world_count": worlds, "maximum_seconds": maximum_seconds},
                        inputs=[QUEUE, FEATURE / "runtime.json", ROOT / "docs/chain/dynamic-scaling-protocol.md"],
                        provenance="public_asset_diagnostic")
    record.update(campaign_identity=identity, pid=os.getpid(), tracking_error_values_read=False)
    with recorded(directory, record):
        with gpu(record, maximum_seconds) as deadline:
            command = [str(d0.PYTHON), "-B", "-m", "track3.features.dynamic_feasibility._worker",
                       str(FEATURE / "probe.py"), "--worlds", str(worlds),
                       "--out", str(directory / "throughput.json")]
            d0.execute(command, directory / "probe.log", env, deadline, record, "probe")
            record["throughput"] = json.loads((directory / "throughput.json").read_text())
    return directory / "run.json"


@contextmanager
def temporary_checkpoints(scratch, final_steps):
    """Bound disk use by removing obsolete generated checkpoints only inside owned scratch."""
    stop = threading.Event()
    retention = {"scratch": str(scratch.relative_to(ROOT)), "removed_files": 0, "removed_bytes": 0,
                 "keep_recent_steps": 2, "errors": []}

    def reap():
        while not stop.wait(1):
            try:
                paths = list((scratch / "training").glob("*.safetensors"))
                steps = sorted({int(path.stem.rsplit("_", 1)[1]) for path in paths})
                keep = set(steps[-2:]) | {final_steps}
                for path in paths:
                    if int(path.stem.rsplit("_", 1)[1]) not in keep:
                        size = path.stat().st_size
                        path.unlink()
                        retention["removed_files"] += 1
                        retention["removed_bytes"] += size
            except Exception as error:
                retention["errors"].append(f"{type(error).__name__}: {error}")
                return

    thread = threading.Thread(target=reap, daemon=True)
    thread.start()
    try:
        yield retention
    finally:
        stop.set()
        thread.join()


def run(request, task, identity):
    """Train one D1 seed to its final checkpoint, then use the official D0 evaluator."""
    env = environment("run_authorized")
    directory = OUTPUT / "runs" / f"seed{request.seed}"
    record = run_fields(ROOT, "dynamic_feasibility", "scaling_run", asdict(request),
                        inputs=[QUEUE, FEATURE / "runtime.json", OUTPUT / "selection.json",
                                ROOT / "docs/chain/dynamic-scaling-protocol.md"],
                        provenance="public_asset_diagnostic")
    record.update(campaign_identity=identity, pid=os.getpid(), task_inputs=task["inputs"],
                  diagnostic_only=True, challenge_episode_access=False, fps=task["fps"])
    with recorded(directory, record):
        scratch = Path(tempfile.mkdtemp(prefix=f"dynamic-scaling-seed{request.seed}-", dir=ROOT / ".runtime/tmp"))
        steps = request.budget["total_environment_steps"]
        with gpu(record, request.maximum_seconds) as deadline:
            command = d0.training_command(request, task, directory)
            command = [f"output_dir={json.dumps(str(scratch / 'training'))}" if arg.startswith("output_dir=")
                       else arg for arg in command]
            with temporary_checkpoints(scratch, steps) as retention:
                record["checkpoint_retention"] = retention
                d0.execute(command, directory / "training.log", env, deadline - 90, record, "training")
            if retention["errors"]:
                raise RuntimeError(retention["errors"][-1])
            final = directory / "training"
            final.mkdir()
            for name in (f"policy_{steps}.safetensors", f"state_{steps}.safetensors"):
                shutil.copyfile(scratch / "training" / name, final / name)
            checkpoint = final / f"policy_{steps}.safetensors"
            record.update(d0.checkpoint_counts(checkpoint, final / f"state_{steps}.safetensors", "D1"))
            command = [str(d0.PYTHON), "-B", "-m", "track3.features.dynamic_feasibility._worker", str(d0.EVALUATOR),
                       f"evaluation.checkpoint={checkpoint}", "evaluation.world_count=1",
                       "evaluation.reset_mode=explicit", "evaluation.start_frame=0",
                       f"evaluation.object_trajectories_output={directory / 'trajectory.parquet'}",
                       f"evaluation.metrics_output={directory / 'official_metrics.json'}",
                       f"hydra.run.dir={directory / 'evaluate_hydra'}", "logging.mode=disabled"]
            d0.execute(command, directory / "evaluation.log", env, deadline, record, "evaluation")
        runtime = runtime_flag(FEATURE, "measure_authorized", "D1 measurement")
        command = [str(ROOT / runtime["measurement_python"]), "-B", "-m",
                   "track3.features.dynamic_feasibility.scaling.measure", str(directory)]
        record.setdefault("commands", []).append(command)
        with (directory / "measurement.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": ""},
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        record["metrics"] = json.loads((directory / "metrics.json").read_text())
    return directory / "run.json"
