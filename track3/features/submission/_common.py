"""Submission-only authorization and immutable operation records."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
import sys

from track3.paths import output_path, project_path
from track3.provenance import input_file, read_run, run_fields
from track3.runs import (LOCK, ROOT, artifact, checked_entry, dataset_revision, new_run_id, read_json,  # noqa: F401
                         record_input, recorded, runtime_flag)

FEATURE = Path(__file__).parent
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class SubmissionRequest:
    label: str = "resting"
    rollouts: dict[str, list[str]] = field(default_factory=dict)
    self_scores: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)
    meshes: dict[str, dict[str, str]] = field(default_factory=dict)
    selection_margin: float = 0.0
    margin_reason: str = "Development setting not yet chosen"
    baseline_submission: str | None = None
    single_change: str = "Initial resting baseline"
    submission_dir: str | None = None
    check_run: str | None = None
    pack_run: str | None = None
    upload_kind: str = "poses"
    message: str = ""
    only: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", self.label):
            raise ValueError("Use a short lowercase submission label")
        if not math.isfinite(self.selection_margin) or self.selection_margin < 0 or not self.margin_reason.strip():
            raise ValueError("Record a nonnegative selection margin and its reason")
        if not self.single_change.strip() or "\n" in self.single_change or "\r" in self.single_change:
            raise ValueError("State the single change in one sentence")
        if self.upload_kind not in ("poses", "meshes") or not isinstance(self.only, list) or any(not isinstance(value, str) for value in self.only):
            raise ValueError("Invalid upload kind or explicit retry legs")
        if any(int(value) == 41 for mapping in (self.rollouts, self.meshes, self.overrides) for value in mapping):
            raise PermissionError("Episode 41 remains excluded")


def authorize(operation):
    runtime_flag(FEATURE, operation.replace("-", "_") + "_authorized", f"Submission {operation}")
    if sys.platform != "win32" or Path(sys.prefix).resolve() != ROOT / ".runtime/submission-python313":
        raise PermissionError("Use the dedicated Windows submission environment")


def producer(value, kind=None, episode=None):
    if value is None:
        raise ValueError("Name an explicit completed input run")
    path = local_path(value)
    record = read_run(ROOT, path, kind=kind, allowed=("video_only",))
    if episode is not None and (record.get("split") != "evaluation" or record.get("episode_index") != episode or episode == 41):
        raise ValueError("Submission input is not from the named evaluation episode")
    return path, record


def artifact_entry(parent, name, record):
    return input_file(ROOT, artifact(parent, name, record))


def kit_sources(record):
    manifest = read_json(KIT / "KIT_MANIFEST.json")
    for name, digest in manifest["files"].items():
        entry = input_file(ROOT, KIT / name)
        if entry["sha256"] != digest:
            raise ValueError("Kit differs from its manifest: " + name)
        record_input(entry, record, "sources")
    return manifest


def submission(request):
    if request.submission_dir is None:
        raise ValueError("Name the exact submission directory")
    directory = local_path(request.submission_dir)
    if directory.parent != ROOT / "submissions":
        raise ValueError("Expected an exact top-level submission directory")
    return directory, producer(directory / "run.json", "submission.plan")


@contextmanager
def operation_run(operation, request, parents=(), submission_directory=None):
    authorize(operation)
    record = run_fields(ROOT, "submission", operation.replace("-", "_"), asdict(request),
                        parents=[path for path, _ in parents], allowed=("video_only",),
                        inputs=[FEATURE / "runtime.json", KIT / "KIT_MANIFEST.json", "reports/runtime/policy-env.json"])
    run_id = new_run_id(request.label)
    directory = output_path(ROOT, f"submissions/{run_id}") if operation == "plan" else output_path(
        ROOT, submission_directory / "operations" / (operation + "-" + run_id))
    record.update(run_id=run_id, split="evaluation", gpu=None, artifacts={}, dataset_revision=dataset_revision())
    if submission_directory:
        record["submission_dir"] = submission_directory.relative_to(ROOT).as_posix()
    with recorded(directory, record) as result:
        yield result
