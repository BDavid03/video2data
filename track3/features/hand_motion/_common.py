"""Feature-local hand requests, authorization, input lineage and persistence."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from track3 import runs
from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields
from track3.runs import (ROOT, artifact, save_npz, authorize_episode, dataset_revision, new_run_id, producer,  # noqa: F401
                         read_json, recorded, require_input, runtime_flag, same_revision)

FEATURE = Path(__file__).parent


def require_episode(request, operation):
    if (request.source_split not in ("public", "evaluation") or type(request.episode_index) is not int
            or request.episode_index < 0):
        raise ValueError(f"Hand {operation} requires a named public/evaluation episode")
    if request.episode_index == 41:
        raise PermissionError("Episode 41 remains excluded")


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class LiftHandsRequest:
    source_split: str = "public"
    episode_index: int = 39
    kinematics_run: str | None = None
    masks_run: str | None = None
    depth_run: str | None = None
    stereo_run: str | None = None
    anchor_run: str | None = None
    camera_run: str | None = None
    minimum_score: float = 0.5
    minimum_in_frame_fraction: float = 0.5
    minimum_depth_support: int = 20
    fit_iterations: int = 5
    maximum_depth_shift_m: float = 0.5
    maximum_depth_residual_m: float = 0.025
    outlier_floor_m: float = 0.01

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        require_episode(self, "lifting")
        for name in ("minimum_depth_support", "fit_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid hand integer field: {name}")
        for name in ("minimum_score", "minimum_in_frame_fraction", "maximum_depth_shift_m",
                     "maximum_depth_residual_m", "outlier_floor_m"):
            if not is_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid hand threshold: {name}")
        if max(self.minimum_score, self.minimum_in_frame_fraction) > 1:
            raise ValueError("Hand score and in-frame thresholds must be at most one")


@dataclass(frozen=True)
class EstimateHandsRequest:
    source_split: str = "public"
    episode_index: int = 39
    masks_run: str | None = None
    depth_run: str | None = None
    minimum_score: float = 0.5
    minimum_mask_iou: float = 0.1
    maximum_detections: int = 8
    maximum_track_gap: int = 5
    maximum_center_step: float = 0.2
    handedness_penalty: float = 0.2
    assignment_margin: float = 0.05
    crop_expansion: float = 2.5
    gpu_headroom_mib: int = 2048
    gpu_admission_mib: int = 4096

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        require_episode(self, "estimation")
        for name in ("maximum_detections", "maximum_track_gap", "gpu_headroom_mib", "gpu_admission_mib"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid hand estimation integer: {name}")
        if self.maximum_detections > 32:
            raise ValueError("Hand detections are bounded to at most 32 per frame")
        for name in ("minimum_score", "minimum_mask_iou", "maximum_center_step", "handedness_penalty", "assignment_margin"):
            if not is_number(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"Invalid hand estimation fraction: {name}")
        if not is_number(self.crop_expansion) or self.crop_expansion < 1:
            raise ValueError("Hand crop expansion must be finite and at least one")


@dataclass(frozen=True)
class KinematicsRequest:
    source_split: str = "public"
    episode_index: int = 39
    estimate_run: str | None = None
    masks_run: str | None = None
    depth_run: str | None = None
    batch_size: int = 16

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        require_episode(self, "kinematics")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 256:
            raise ValueError("Kinematics batch size must be between 1 and 256")


def authorize(request, operation="lift"):
    if operation not in ("estimate", "kinematics", "lift", "preview", "refine", "contacts"):
        raise ValueError("Unknown hand operation")
    authorize_episode(runtime_flag(FEATURE, operation + "_authorized", f"Hand {operation}"), request, "hand motion")


def require_rotations(value, np):
    if (not np.isfinite(value).all() or
            not np.allclose(np.swapaxes(value, -1, -2) @ value, np.eye(3), atol=1e-5, rtol=0) or
            not np.allclose(np.linalg.det(value), 1, atol=1e-5, rtol=0)):
        raise ValueError("Rotations must be finite proper rotation matrices")


def record_input(path, record):
    return runs.record_input(input_file(ROOT, path), record)


def rgb_clock(request, record):
    record_input("reports/data-audit/videos.json", record)
    inventory = read_json("reports/data-audit/videos.json")
    rows = [row for row in inventory["videos"] if row["split"] == request.source_split and
            row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if (inventory["status"] != "completed" or inventory["revision"] != record["dataset_revision"] or
            len(rows) != 1 or rows[0]["status"] != "ok"):
        raise ValueError("Hand operation requires the completed pinned RGB clock inventory")
    clock = rows[0]
    if len(clock["pts_integer"]) != clock["decoded_frame_count"] or clock["decoded_frame_count"] <= 0:
        raise ValueError("Incomplete RGB clock inventory")
    return clock


def rgb_video(depth, clock, record):
    """The pinned episode video, which must be the one the depth producer and the clock inventory used."""
    source = read_json("configs/sources.lock.json")["dataset"]
    video = project_path(ROOT, depth[1]["source_video"])
    expected = project_path(ROOT, f"{source['cache_path']}/snapshots/{source['revision']}/{clock['path']}")
    entry = input_file(ROOT, video)
    if video != expected or entry["sha256"] != depth[1]["input_hashes"]["video_sha256"]:
        raise ValueError("Hand RGB video differs from the selected depth and clock inventory")
    record["inputs"].append(entry)
    return video


def load_npz(path, np):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def arrays(parent, name, record, np, digest=None):
    return load_npz(artifact(parent, name, record, digest), np)


def frame_arrays(parent, index, pts, record, np):
    rows = [row for row in parent[1]["frames"] if row["frame_index"] == index]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["video_pts"] != pts:
        raise ValueError("Missing, incomplete or mismatched hand input frame")
    return arrays(parent, rows[0]["artifact"], record, np, rows[0]["sha256"])


@contextmanager
def operation_run(request, parents, operation="lift"):
    authorize(request, operation)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "hand_motion", operation, asdict(request),
                        parents=[path for path, _ in parents], allowed=("video_only",),
                        inputs=[FEATURE / "runtime.json", "configs/splits/public-v1.json",
                                "configs/perception/moge/pylock.python313.toml"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, frames=[], artifacts={}, gpu=None)
    with recorded(output_path(ROOT, f"outputs/hand_motion/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
