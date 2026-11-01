"""Training requests, input lineage and Linux subprocess boundaries."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys

from track3.paths import output_path, project_path, linux_path  # noqa: F401
from track3.provenance import input_file, run_fields, sha256_file
from track3.runs import (LOCK, ROOT, artifact, authorize_episode, checked_entry, dataset_revision,  # noqa: F401
                         new_run_id, read_json, record_input, recorded, runtime_flag, same_revision)
from track3.runs import producer as _producer
from .recipes import RECIPE_VARIANTS

FEATURE = Path(__file__).parent
CANONICAL_LINUX_ROOT = "/mnt/z/hugging-face/video2data"


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class PolicyRequest:
    source_split: str = "public"
    episode_index: int = 39
    retargeting_run: str | None = None
    compose_run: str | None = None
    training_run: str | None = None
    rollout_inspections: list[str] = field(default_factory=list)
    budget: str = "zero_update"
    policy_kind: str = "resting"
    recipe_variant: str = "contact_aware"
    final_checkpoint_only: bool = False
    seed: int = 31
    headroom_mib: int = 1024
    admission_allowance_mib: int = 1024

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.recipe_variant not in RECIPE_VARIANTS:
            raise ValueError("Unknown immutable policy recipe")
        if type(self.final_checkpoint_only) is not bool:
            raise ValueError("final_checkpoint_only must be boolean")
        if self.source_split not in ("public", "evaluation") or self.policy_kind not in ("resting", "replay", "trained"):
            raise ValueError("Invalid training split or policy kind")
        for name in ("episode_index", "seed", "headroom_mib", "admission_allowance_mib"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if not isinstance(self.rollout_inspections, list) or any(not isinstance(item, str) for item in self.rollout_inspections):
            raise ValueError("rollout_inspections must name explicit runs")


def authorize(operation, request):
    authorize_episode(runtime_flag(FEATURE, operation + "_authorized", f"Policy {operation}"), request, "policies")
    config = read_json("configs/runtime/policy-env.json")
    if config["linux_root"] != CANONICAL_LINUX_ROOT or ROOT != Path(CANONICAL_LINUX_ROOT).resolve():
        raise PermissionError("Policy artifacts require the canonical Z: project checkout")
    if (sys.platform != "linux" or os.environ.get("WSL_DISTRO_NAME") != config["distro"] or
            Path(sys.prefix).resolve() != Path(config["policy_environment"]).resolve()):
        raise PermissionError("Use the dedicated Linux policy Python environment")
    installation = read_json(config["installation_record"])
    if installation.get("status") != "completed" or installation.get("source_commit") != read_json(LOCK)["policy_runtime"]["commit"]:
        raise ValueError("A completed pinned installation record is required")
    return config


def _allowed(request):
    return ("video_only",) if request.source_split == "evaluation" else ("video_only", "public_label_diagnostic")


def producer(value, kind, request):
    return _producer(value, kind, request, allowed=_allowed(request))


def upstream(relative, record):
    config = read_json("configs/runtime/policy-env.json")
    path = Path(config["checkout"]) / "robotic_grounding/flash_chord" / relative
    reference = local_path(read_json(LOCK)["policy_runtime"]["path"]) / "robotic_grounding/flash_chord" / relative
    if path.read_bytes() != reference.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError(f"Installed LF source differs: {relative}")
    record_input(input_file(ROOT, reference), record, "sources")
    record.setdefault("runtime_sources", []).append({"path": str(path), "sha256": sha256_file(path)})
    return path


def command(overrides, directory, record, compose=False):
    script = upstream("scripts/train_flash_sac.py", record)
    arguments = [sys.executable, str(script)]
    if compose:
        arguments += ["--cfg", "job", "--resolve"]
    arguments += [key + "=" + json.dumps(value, separators=(",", ":")) for key, value in overrides.items()]
    caches = {"FLASH_CHORD_CACHE_DIR": "flash-chord", "WARP_CACHE_PATH": "warp", "XDG_CACHE_HOME": "xdg",
              "TMPDIR": "policy-tmp", "TEMP": "policy-tmp", "TMP": "policy-tmp",
              "TORCH_HOME": "torch-policy", "TORCH_EXTENSIONS_DIR": "torch-extensions-policy",
              "TORCHINDUCTOR_CACHE_DIR": "torch-inductor-policy", "TRITON_CACHE_DIR": "triton-policy",
              "CUDA_CACHE_PATH": "cuda-policy"}
    environment = {**os.environ, "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   **{name: str(ROOT / ".cache" / relative) for name, relative in caches.items()}}
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    path = directory / ("resolved-config.yaml" if compose else "training.log")
    record["compose_command" if compose else "training_command"] = arguments
    with path.open("x", encoding="utf-8") as stream:
        subprocess.run(arguments, cwd=directory, env=environment, stdout=stream,
                       stderr=subprocess.PIPE if compose else subprocess.STDOUT, text=True, check=True)
    return path


@contextmanager
def operation_run(operation, request, parents):
    authorize(operation, request)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "policy_training", operation, asdict(request), parents=[path for path, _ in parents],
                        allowed=_allowed(request), inputs=[FEATURE / "runtime.json", FEATURE / "budget.json",
                        "configs/splits/public-v1.json", "configs/runtime/policy-env.json", "reports/runtime/policy-env.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, policy_kind=request.policy_kind, artifacts={}, frames=[], gpu=None,
                  source_commit=read_json(LOCK)["policy_runtime"]["commit"])
    with recorded(output_path(ROOT, f"runs/policy/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
