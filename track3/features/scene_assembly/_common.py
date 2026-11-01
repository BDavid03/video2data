"""Private scene request, input contracts and persistence."""

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import math
from pathlib import Path

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields
from track3.runs import (ROOT, artifact, save_npz, authorize_episode, dataset_revision, new_run_id, producer,  # noqa: F401
                         read_json, recorded, require_input, runtime_flag, same_revision)

FEATURE = Path(__file__).parent


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class AssembleSceneRequest:
    source_split: str = "public"
    episode_index: int = 39
    mode: str = "resting"
    pose_run: str | None = None
    camera_run: str | None = None
    mesh_run: str | None = None
    hand_run: str | None = None
    parked_hand_run: str | None = None
    maximum_gap_frames: int = 5
    smoothing_radius: int = 2
    minimum_rest_frames: int = 5
    parking_transition_frames: int = 10
    rest_translation_m: float = 0.005
    rest_rotation_rad: float = 0.05
    hand_near_m: float = 0.08
    seating_limit_m: float = 0.05
    contact_distance_m: float = 0.015
    parking_clearance_m: float = 0.25
    supports: dict[str, str] = field(default_factory=dict)
    mass_kg: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation") or self.mode not in ("resting", "full"):
            raise ValueError("Scene requires a named public/evaluation episode and resting/full mode")
        for name in ("episode_index", "maximum_gap_frames", "smoothing_radius", "minimum_rest_frames", "parking_transition_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name in ("episode_index", "smoothing_radius") else 1):
                raise ValueError(f"Invalid scene integer field: {name}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        for name in ("rest_translation_m", "rest_rotation_rad", "hand_near_m", "seating_limit_m", "contact_distance_m", "parking_clearance_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid scene distance/angle: {name}")
        if not isinstance(self.supports, dict) or not isinstance(self.mass_kg, dict):
            raise ValueError("supports and mass_kg must be object-name mappings")
        if any(not math.isfinite(value) or value <= 0 for value in self.mass_kg.values()):
            raise ValueError("Mass estimates must be finite positive kilograms")


def authorize(request):
    authorize_episode(runtime_flag(FEATURE, "assemble_authorized", "Scene assembly"), request, "scene assembly")


def clock_row(request, revision, error):
    """Return the one completed camA clock row for the request episode, or raise ``error``."""
    inventory = read_json("reports/data-audit/videos.json")
    rows = [row for row in inventory["videos"] if row["split"] == request.source_split
            and row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if inventory["status"] != "completed" or inventory["revision"] != revision or len(rows) != 1 or rows[0]["status"] != "ok":
        raise ValueError(error)
    return rows[0]


def template_producer(value):
    path, source = producer(value, "hand_motion.parked_template")
    if (source.get("scope") != "model_asset_template" or source.get("observed") is not False or
            source.get("units") != "metres" or source.get("joint_order") != "MANO21" or
            source.get("joint_frame_convention") != "anatomical_unmirrored" or
            source.get("split") is not None or source.get("episode_index") is not None):
        raise ValueError("Parked hands require a neutral model-asset template, not an episode observation")
    return path, source


def arrays(parent, name, record, np):
    path = artifact(parent[0], name, record, parent[1]["artifacts"][name])
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@contextmanager
def operation_run(request, parents):
    authorize(request)
    lock = read_json("configs/sources.lock.json")
    revision = same_revision(parents, lock["dataset"]["revision"], skip=("hand_motion.parked_template",))
    record = run_fields(ROOT, "scene_assembly", "assemble", asdict(request), parents=[path for path, _ in parents],
                        allowed=("video_only",), inputs=["configs/sources.lock.json", "configs/splits/public-v1.json",
                        "reports/data-audit/videos.json", "configs/perception/moge/pylock.python313.toml",
                        FEATURE / "runtime.json"])
    reference = local_path(lock["policy_runtime"]["path"]) / (
        "robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/params.py")
    record["sources"].append(input_file(ROOT, reference))
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, frames=[], artifacts={}, gpu=None, mode=request.mode)
    with recorded(output_path(ROOT, f"outputs/scene_assembly/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
