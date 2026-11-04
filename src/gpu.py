"""Cooperative GPU ownership and telemetry shared by local feature operations."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading


def clear_stale_lock(root):
    """Remove the cooperative lock only when its recorded owner pid is no longer alive.

    Returns the stale record that was removed, or None. A live owner is never touched.
    """
    path = Path(root) / "data/.runtime/gpu-stage.lock"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        pid = int(record.get("pid"))
    except (ValueError, TypeError, OSError):
        record, pid = {"unreadable": True}, None
    if pid is not None:
        try:
            os.kill(pid, 0)
            return None  # owner alive
        except ProcessLookupError:
            pass
        except PermissionError:
            return None
    path.unlink(missing_ok=True)
    return record


class GpuLease:
    """Cooperative project GPU lock; no stale lock is removed automatically."""

    def __init__(self, root, headroom_mib, admission_allowance_mib, stage="depth"):
        self.path = Path(root) / "data/.runtime/gpu-stage.lock"
        self.stage = stage
        self.headroom_mib = headroom_mib
        self.admission_allowance_mib = admission_allowance_mib
        self.interval_seconds = 2.0  # one nvidia-smi query costs ~0.25 s on WSL
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = None
        self.owned = False

    def _sample(self):
        result = subprocess.run(
            ["nvidia-smi", "--id=0",
             "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = [line for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError("This operation requires telemetry for physical GPU 0")
        index, uuid, total, used, free = [part.strip() for part in rows[0].split(",")]
        sample = {"time_utc": datetime.now(timezone.utc).isoformat(),
                  "index": int(index), "uuid": uuid, "total_mib": int(total),
                  "used_mib": int(used), "free_mib": int(free)}
        self.samples.append(sample)
        return sample

    def _poll(self):
        while not self.stop.wait(self.interval_seconds):
            try:
                self._sample()
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                return

    def __enter__(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "0"):
            raise RuntimeError("This bounded feature supports physical GPU 0 only")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                self.owned = True
                json.dump({"stage": self.stage, "pid": os.getpid(),
                           "created_utc": datetime.now(timezone.utc).isoformat()}, stream)
            first = self._sample()
            if first["free_mib"] < self.headroom_mib + self.admission_allowance_mib:
                raise RuntimeError("Insufficient device-wide free VRAM for operation admission")
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def require_headroom(self):
        """Check the sampler thread's telemetry; never spawns nvidia-smi on the training step."""
        if self.errors:
            raise RuntimeError(f"GPU telemetry unavailable: {self.errors[-1]}")
        if any(sample["free_mib"] < self.headroom_mib for sample in self.samples):
            raise RuntimeError("Device-wide VRAM crossed the configured headroom boundary")

    def report(self):
        return {
            "samples": list(self.samples), "errors": list(self.errors),
            "peak_sampled_total_device_used_mib": max(
                (sample["used_mib"] for sample in self.samples), default=None),
            "minimum_sampled_free_mib": min(
                (sample["free_mib"] for sample in self.samples), default=None),
            "sampling_interval_seconds": self.interval_seconds,
            "measurement_limit": "Includes display and other applications; sampling can miss short peaks.",
            "admission_allowance_mib": self.admission_allowance_mib,
            "required_headroom_mib": self.headroom_mib,
            "headroom_limit": "Cooperative admission; a running CUDA operation cannot be interrupted at the memory boundary.",
            "peak_host_ram_bytes": None,
            "host_ram_measurement": "Not instrumented by this feature.",
        }

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=11)
        if self.owned:
            self.path.unlink(missing_ok=True)
            self.owned = False
