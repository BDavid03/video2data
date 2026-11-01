"""Private mesh request, lineage and run persistence."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re

from track3.paths import output_path, project_path
from track3.provenance import run_fields
from track3.runs import (ROOT, artifact, authorize_episode, dataset_revision, new_run_id, producer,  # noqa: F401
                         read_json, recorded, require_input, runtime_flag, same_revision)

FEATURE = Path(__file__).parent
ROSTER = "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class ObjectMeshRequest:
    source_split: str = "public"
    episode_index: int = 39
    method: str = "opening_hull"
    reconstruction_run: str | None = None
    scans_only: bool = False
    masks_run: str | None = None
    depth_run: str | None = None
    anchor_run: str | None = None
    camera_run: str | None = None
    pose_run: str | None = None
    objects: list[str] = field(default_factory=list)
    cluster_assignments: dict[str, int] = field(default_factory=dict)
    minimum_points: int = 50
    voxel_size_m: float = 0.002
    truncation_voxels: float = 4.0
    minimum_fusion_views: int = 2
    maximum_grid_points: int = 2000000
    maximum_surface_faces: int = 200000
    maximum_point_voxels: int = 500000

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if type(self.scans_only) is not bool or (self.scans_only and self.reconstruction_run):
            raise ValueError("scans_only must be boolean and cannot select a reconstruction")
        if self.source_split not in ("public", "evaluation") or self.method not in ("opening_hull", "opening_tsdf", "moving_tsdf"):
            raise ValueError("Select a named opening_hull, opening_tsdf or moving_tsdf method")
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("Invalid episode index")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if type(self.minimum_points) is not int or self.minimum_points < 4:
            raise ValueError("Hull requires at least four measured points")
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0:
            raise ValueError("Voxel size must be finite and positive")
        if not math.isfinite(self.truncation_voxels) or not 2 <= self.truncation_voxels <= 16:
            raise ValueError("TSDF truncation must be between 2 and 16 voxels")
        for name, value, low, high in (("minimum_fusion_views", self.minimum_fusion_views, 1, 10000),
                                      ("maximum_point_voxels", self.maximum_point_voxels, 4, 2000000),
                                      ("maximum_grid_points", self.maximum_grid_points, 8, 8000000),
                                      ("maximum_surface_faces", self.maximum_surface_faces, 1, 1000000)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        if not isinstance(self.objects, list) or len(self.objects) != len(set(self.objects)):
            raise ValueError("objects must be an explicit unique list, or empty for the whole episode")
        if not isinstance(self.cluster_assignments, dict):
            raise ValueError("cluster_assignments must map object names to opening cluster indices")
        for name in [*self.objects, *self.cluster_assignments]:
            if not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None:
                raise ValueError("Unsafe or invalid object name")
        if any(type(value) is not int or value < 0 for value in self.cluster_assignments.values()):
            raise ValueError("Cluster indices must be nonnegative integers")


def authorize(operation, request):
    authorize_episode(runtime_flag(FEATURE, operation + "_authorized", f"Mesh {operation}"), request, "meshes")


def names_for(request):
    source = ROSTER if request.source_split == "evaluation" else "reports/data-audit/trajectories.json"
    inventory = read_json(source)
    if request.source_split == "evaluation":
        names = inventory["episodes"][str(request.episode_index)]
    else:
        if inventory["revision"] != dataset_revision():
            raise ValueError("Object-name inventory revision differs")
        rows = [row for row in inventory["episodes"] if row["episode_index"] == request.episode_index]
        if len(rows) != 1:
            raise ValueError("Exactly one object-name inventory is required")
        names = rows[0]["objects"]
    if not names or len(set(names)) != len(names) or any(re.fullmatch(r"[a-z0-9_]+", name) is None for name in names):
        raise ValueError("Object-name roster is invalid")
    return names, source


def _load(path, np):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def arrays(parent, name, record, np):
    return _load(artifact(parent[0], name, record, parent[1]["artifacts"][name]), np)


def frame_arrays(parent, index, pts, record, np):
    rows = [row for row in parent[1]["frames"] if row["frame_index"] == index]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["video_pts"] != pts:
        raise ValueError("Missing, incomplete or mismatched mesh input frame")
    return _load(artifact(parent[0], rows[0]["artifact"], record, rows[0]["sha256"]), np)


def mesh_metadata(mesh, np):
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Empty or nonfinite mesh")
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces), "closed": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent), "extent_m": mesh.extents.tolist()}


@contextmanager
def operation_run(operation, request, parents=()):
    authorize(operation, request)
    names, roster = names_for(request)
    revision = same_revision(parents, dataset_revision())
    record = run_fields(ROOT, "object_mesh", operation, asdict(request), parents=[path for path, _ in parents],
                        allowed=("video_only",), inputs=[roster, "configs/sources.lock.json", "configs/splits/public-v1.json",
                        "configs/perception/moge/pylock.python313.toml", FEATURE / "runtime.json", FEATURE / "sources.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, frames=[], artifacts={}, gpu=None, object_names=names)
    with recorded(output_path(ROOT, f"outputs/object_mesh/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
