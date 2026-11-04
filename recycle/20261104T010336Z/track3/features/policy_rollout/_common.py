"""Rollout requests and run records."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys

from track3.paths import output_path, project_path, linux_path  # noqa: F401
from track3.provenance import input_file, run_fields, sha256_file
from track3.runs import (LOCK, ROOT, artifact, authorize_episode, checked_entry, dataset_revision,  # noqa: F401
                         new_run_id, read_json, record_input, recorded, runtime_flag, same_revision)
from track3.runs import producer as _producer

FEATURE = Path(__file__).parent
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class RolloutRequest:
    source_split: str = "public"
    episode_index: int = 39
    training_run: str | None = None
    checkpoint: str | None = None
    rollout_run: str | None = None
    inspection_run: str | None = None
    scene_run: str | None = None
    resting_inspection: str | None = None
    source_root: str | None = None
    headroom_mib: int = 1024
    admission_allowance_mib: int = 1024

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation"):
            raise ValueError("Invalid rollout split")
        for key in ("episode_index", "headroom_mib", "admission_allowance_mib"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")


def authorize(operation, request):
    flag = operation.replace("-", "_") + "_authorized"
    authorize_episode(runtime_flag(FEATURE, flag, f"Policy rollout {operation}"), request, "rollouts")
    config = read_json("configs/runtime/policy-env.json")
    policy = (sys.platform == "linux" and os.environ.get("WSL_DISTRO_NAME") == config["distro"] and
              Path(sys.prefix).resolve() == Path(config["policy_environment"]).resolve())
    submission = sys.platform == "win32" and Path(sys.prefix).resolve() == ROOT / ".runtime/submission-python313"
    if not policy and (operation == "rollout" or not submission):
        raise PermissionError("Use the dedicated policy or submission environment")
    installed = read_json(config["installation_record"])
    if installed.get("status") != "completed" or installed.get("source_commit") != read_json(LOCK)["policy_runtime"]["commit"]:
        raise ValueError("A completed pinned policy installation is required")
    return config


def _allowed(request):
    return ("video_only",) if request.source_split == "evaluation" else ("video_only", "public_label_diagnostic")


def producer(value, kind, request):
    return _producer(value, kind, request, allowed=_allowed(request))


def artifact_pin(parent, name, record):
    return input_file(ROOT, artifact(parent, name, record))


def kit_file(name, record):
    path = KIT / name
    if sha256_file(path) != read_json(KIT / "KIT_MANIFEST.json")["files"][name]:
        raise ValueError("Submission Kit differs from its manifest")
    record_input(input_file(ROOT, path), record, "sources")
    return path


@contextmanager
def operation_run(operation, request, parents):
    authorize(operation, request)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "policy_rollout", operation.replace("-", "_"), asdict(request),
                        parents=[path for path, _ in parents], allowed=_allowed(request),
                        inputs=[FEATURE / "runtime.json", "configs/runtime/policy-env.json",
                                "reports/runtime/policy-env.json", "configs/splits/public-v1.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, artifacts={}, frames=[], gpu=None)
    with recorded(output_path(ROOT, f"runs/rollout/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
