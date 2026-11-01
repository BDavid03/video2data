"""Opt-in, in-memory producer phase spans; never invokes a model or writes files."""

from contextlib import contextmanager
from time import perf_counter_ns

PHASES = ("decode", "preprocess", "transfer_to_device", "model", "loss_backward_update",
          "transfer_to_host", "compile", "artifact_io", "end_to_end")
REGIMES = ("cold", "warmup", "warm")


class PhaseRecorder:
    """Bounded host-wall timing. A GPU span requires a caller-owned synchronizer.

    Synchronization changes overlap and is not a CUDA-event measurement. Concurrent
    or nested spans may overlap; consumers must never sum them as total runtime.
    The owning producer must gate execution and persist its own source identities.
    """

    def __init__(self, *, enabled=False, max_spans=4096):
        if type(enabled) is not bool or type(max_spans) is not int or not 1 <= max_spans <= 4096:
            raise ValueError("Declare a boolean opt-in and 1–4096 span ceiling")
        self.enabled = enabled
        self.max_spans = max_spans
        self.spans = []
        self._pending = 0

    @contextmanager
    def span(self, phase, regime, iteration, *, device="cpu", synchronize=None):
        if not self.enabled:
            yield
            return
        if (phase not in PHASES or regime not in REGIMES or type(iteration) is not int
                or iteration < 0 or device not in ("cpu", "gpu")):
            raise ValueError("Unknown timing phase/regime/device or negative iteration")
        if device == "gpu" and not callable(synchronize):
            raise ValueError("GPU host-wall spans require explicit device synchronization")
        if len(self.spans) + self._pending >= self.max_spans:
            raise RuntimeError("Declared phase-span budget exhausted")
        sync = synchronize if device == "gpu" else lambda: None
        sync()
        start = perf_counter_ns()
        self._pending += 1
        failure = None
        try:
            yield
        except BaseException as exc:
            failure = type(exc).__name__
            raise
        finally:
            try:
                sync()
            except BaseException as exc:
                # Preserve the original producer exception if synchronization also fails.
                if failure is None:
                    failure = type(exc).__name__
                    raise
            finally:
                self._pending -= 1
                self.spans.append({"phase": phase, "regime": regime, "iteration": iteration,
                                   "device": device, "start_ns": start, "stop_ns": perf_counter_ns(),
                                   "clock": "perf_counter_ns", "synchronized": device == "gpu",
                                   "status": "failed" if failure else "completed", "failure": failure})

    def snapshot(self):
        """A copy for the producer's receipt; an empty recorder is not a measurement."""
        return [dict(span) for span in self.spans]
