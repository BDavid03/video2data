"""Retargeting requests, gated Linux runtime and immutable run artifacts."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import subprocess
import sys

from track3.paths import output_path, project_path, linux_path  # noqa: F401
from track3.provenance import input_file, run_fields, sha256_file
from track3.runs import (LOCK, ROOT, artifact, authorize_episode, dataset_revision,  # noqa: F401
                         new_run_id, read_json, record_input, recorded, runtime_flag, same_revision)
from track3.runs import producer as _producer

FEATURE = Path(__file__).parent
ROSTER = "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class RetargetRequest:
    source_split: str = "public"
    episode_index: int = 39
    scene_run: str | None = None
    input_run: str | None = None
    contact_run: str | None = None
    uncertainty_run: str | None = None
    minimum_contact_confidence: float = 0.75
    maximum_randomization_translation_m: float = 0.03
    maximum_randomization_rotation_rad: float = 0.15
    mano_to_robot_scale: float = 1.2
    smooth_reference: bool = False
    smooth_reference_window: int = 9
    smooth_reference_polyorder: int = 2
    surface_project: bool = False
    surface_margin: float = 0.005
    surface_granularity: str = "finger"
    adaptive_offset: bool = False
    adaptive_offset_alpha: float = 1.2
    adaptive_offset_cap: float = 0.03
    adaptive_offset_smooth_window: int = 9
    hand_object_offset: float = 0.0
    ik_position_error_m: float = 0.02

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation") or type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("Retargeting requires a named split and episode")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if self.contact_run or self.uncertainty_run:
            if self.source_split != "public" or self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42):
                raise PermissionError("Research contact/uncertainty inputs require public training or tuning episodes")
        for key in ("minimum_contact_confidence", "maximum_randomization_translation_m", "maximum_randomization_rotation_rad"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        if self.minimum_contact_confidence > 1:
            raise ValueError("Contact confidence threshold must be at most one")
        for key in ("smooth_reference", "surface_project", "adaptive_offset"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("mano_to_robot_scale", "surface_margin", "adaptive_offset_alpha", "adaptive_offset_cap", "ik_position_error_m"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        if not math.isfinite(self.hand_object_offset) or self.hand_object_offset < 0 or (self.adaptive_offset and self.hand_object_offset):
            raise ValueError("Invalid or mutually exclusive hand offsets")
        for value in (self.smooth_reference_window, self.adaptive_offset_smooth_window):
            if type(value) is not int or value < 5 or value % 2 != 1:
                raise ValueError("Smoothing windows must be odd and at least five")
        if type(self.smooth_reference_polyorder) is not int or not 0 <= self.smooth_reference_polyorder < self.smooth_reference_window:
            raise ValueError("Invalid smoothing polynomial order")
        if self.surface_granularity not in ("finger", "hand", "joint"):
            raise ValueError("Invalid surface granularity")


def authorize(operation, request):
    runtime = runtime_flag(FEATURE, operation + "_authorized", f"Retargeting {operation}")
    for value, flag in ((request.contact_run, "contact_transfer_authorized"),
                        (request.uncertainty_run, "uncertainty_proposal_authorized")):
        if value and runtime["execution"].get(flag) is not True:
            raise PermissionError(f"Retargeting {flag} remains disabled")
    authorize_episode(runtime, request, "retargeting")
    config = read_json("configs/runtime/policy-env.json")
    if sys.platform != "linux" or os.environ.get("WSL_DISTRO_NAME") != config["distro"]:
        raise PermissionError("Use only the dedicated policy distribution")
    if Path(sys.prefix).resolve() != Path(config["retarget_environment"]).resolve():
        raise PermissionError("Use the isolated retargeting Python environment")
    installed = read_json(config["installation_record"])
    if installed.get("status") != "completed" or installed.get("source_commit") != read_json(LOCK)["policy_runtime"]["commit"]:
        raise ValueError("A completed pinned installation record is required")
    return config


def _allowed(request):
    return ("video_only",) if request.source_split == "evaluation" else ("video_only", "public_label_diagnostic")


def producer(value, kind, request):
    return _producer(value, kind, request, allowed=_allowed(request))


def runtime_source(relative, record):
    config = read_json("configs/runtime/policy-env.json")
    path = Path(config["checkout"]) / "robotic_grounding" / relative
    reference = local_path(read_json(LOCK)["policy_runtime"]["path"]) / "robotic_grounding" / relative
    if path.read_bytes() != reference.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError(f"Installed upstream source differs from the pinned LF reference: {relative}")
    record_input(input_file(ROOT, reference), record, "sources")
    record.setdefault("runtime_sources", []).append({"path": str(path), "sha256": sha256_file(path)})
    return path


def run_upstream(relative, arguments, directory, record):
    script = runtime_source(relative, record)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
                   "FLASH_CHORD_CACHE_DIR": str(ROOT / ".cache/flash-chord"),
                   "ROBOTIC_GROUNDING_INTERMEDIATE_DIR": str(directory / "intermediate"),
                   "TMPDIR": str(ROOT / ".cache/policy-tmp")}
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script), *map(str, arguments)]
    record["command"] = command
    with (directory / "upstream.log").open("x", encoding="utf-8") as stream:
        subprocess.run(command, cwd=directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def bundle_root(directory):
    return directory / "assets/human_motion_data/ego_recon"


def partition(directory, stage, request):
    return bundle_root(directory) / stage / f"sequence_id=episode_{request.episode_index:06d}" / "robot_name=sharpa_wave"


@contextmanager
def operation_run(operation, request, parents):
    authorize(operation, request)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "retargeting", operation, asdict(request), parents=[path for path, _ in parents],
                        allowed=_allowed(request), inputs=[FEATURE / "runtime.json", "configs/runtime/policy-env.json",
                        "reports/runtime/policy-env.json", "configs/splits/public-v1.json", ROSTER,
                        "reports/data-audit/videos.json", "reports/data-audit/trajectories.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, artifacts={}, frames=[], gpu=None)
    with recorded(output_path(ROOT, f"outputs/retargeting/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
