"""Private contracts and persistence for geometric object-pose operations."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from track3.paths import output_path, project_path
from track3.provenance import run_fields
from track3.runs import (ROOT, artifact, save_npz, authorize_episode, dataset_revision, new_run_id, producer,  # noqa: F401
                         read_json, recorded, require_input, runtime_flag, same_revision)

FEATURE = Path(__file__).parent


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class ObjectPoseRequest:
    source_split: str = "public"
    episode_index: int = 39
    method: str = "geometric"
    mesh_run: str | None = None
    masks_run: str | None = None
    depth_run: str | None = None
    stereo_run: str | None = None
    anchor_run: str | None = None
    camera_run: str | None = None
    hand_run: str | None = None
    pose_run: str | None = None
    point_budget: int = 2000
    mesh_sample_count: int = 4000
    icp_iterations: int = 30
    yaw_steps: int = 12
    minimum_points: int = 50
    residual_limit_m: float = 0.025
    trim_fraction: float = 0.8
    color_weight_m: float = 0.02
    ambiguity_distance_m: float = 0.002
    ambiguity_color_margin: float = 0.03
    allow_degraded_opening: bool = False
    seed: int = 31
    minimum_confidence: float = 0.2
    minimum_surface_support: float = 0.1
    recovery_surface_support: float = 0.3
    maximum_translation_step_m: float = 0.15
    maximum_rotation_step_rad: float = 1.0
    hand_mask_dilation: int = 5

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if type(self.allow_degraded_opening) is not bool:
            raise ValueError("allow_degraded_opening must be a boolean")
        if self.source_split not in ("public", "evaluation") or self.method != "geometric":
            raise ValueError("Only geometric poses on a named public/evaluation episode are supported")
        for name in ("episode_index", "point_budget", "mesh_sample_count", "icp_iterations", "yaw_steps", "minimum_points", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name in ("episode_index", "seed") else 1):
                raise ValueError(f"Invalid object-pose request field: {name}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if self.minimum_points < 6 or min(self.point_budget, self.mesh_sample_count) < self.minimum_points:
            raise ValueError("Point budgets must cover the minimum registration support")
        values = (self.residual_limit_m, self.trim_fraction, self.color_weight_m,
                  self.ambiguity_distance_m, self.ambiguity_color_margin)
        if (not all(math.isfinite(value) for value in values) or self.residual_limit_m <= 0 or
                not 0 < self.trim_fraction <= 1 or min(values[2:]) < 0):
            raise ValueError("Invalid registration limits")
        for name in ("minimum_confidence", "minimum_surface_support", "recovery_surface_support"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid tracking fraction: {name}")
        if self.recovery_surface_support < self.minimum_surface_support:
            raise ValueError("Recovery requires at least the tracking surface support")
        for name in ("maximum_translation_step_m", "maximum_rotation_step_rad"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid tracking motion limit: {name}")
        if type(self.hand_mask_dilation) is not int or self.hand_mask_dilation < 1 or self.hand_mask_dilation % 2 != 1:
            raise ValueError("Hand-mask dilation must be a positive odd integer")


def authorize(operation, request):
    authorize_episode(runtime_flag(FEATURE, operation + "_authorized", f"Object-pose {operation}"), request, "object poses")


def _load(path, np):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def arrays(parent, name, record, np):
    return _load(artifact(parent[0], name, record, parent[1]["artifacts"][name]), np)


def frame_arrays(parent, index, pts, record, np):
    rows = [row for row in parent[1]["frames"] if row["frame_index"] == index]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["video_pts"] != pts:
        raise ValueError("Missing, incomplete or mismatched object-pose input frame")
    return _load(artifact(parent[0], rows[0]["artifact"], record, rows[0]["sha256"]), np)


def require_rigid(matrices, np):
    if (matrices.shape[-2:] != (4, 4) or not np.isfinite(matrices).all() or
            not np.allclose(matrices[..., 3, :], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(np.swapaxes(matrices[..., :3, :3], -1, -2) @ matrices[..., :3, :3], np.eye(3), atol=1e-5) or
            not np.allclose(np.linalg.det(matrices[..., :3, :3]), 1, atol=1e-5)):
        raise ValueError("Observed poses must be finite proper rigid transforms")


def average(poses, np):
    from scipy.spatial.transform import Rotation

    require_rigid(poses, np)
    center = np.median(poses[:, :3, 3], axis=0)
    rotation = Rotation.from_matrix(poses[:, :3, :3]).mean()
    distances = np.linalg.norm(poses[:, :3, 3] - center, axis=1)
    angles = (rotation.inv() * Rotation.from_matrix(poses[:, :3, :3])).magnitude()
    keep = ((distances <= np.median(distances) + 3 * np.median(abs(distances - np.median(distances))) + 1e-8)
            & (angles <= np.median(angles) + 3 * np.median(abs(angles - np.median(angles))) + 1e-8))
    if not keep.any():
        keep[:] = True
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.median(poses[keep, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(poses[keep, :3, :3]).mean().as_matrix()
    return result, float(np.median(distances)), float(np.degrees(np.median(angles)))


@contextmanager
def operation_run(operation, request, parents):
    authorize(operation, request)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "object_pose", operation, asdict(request), parents=[path for path, _ in parents],
                        allowed=("video_only",), inputs=["configs/sources.lock.json", "configs/splits/public-v1.json",
                        "configs/perception/moge/pylock.python313.toml", FEATURE / "runtime.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, frames=[], artifacts={}, gpu=None, method="geometric")
    with recorded(output_path(ROOT, f"outputs/object_pose/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
