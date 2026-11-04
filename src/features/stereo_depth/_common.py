"""Private request, input and persistence conventions for baseline stereo."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from src.paths import output_path, project_path
from src.provenance import input_file, run_fields
from src.runs import (ROOT, artifact, save_npz, authorize_episode, new_run_id, producer,  # noqa: F401
                         read_json, recorded, same_revision)

FEATURE = Path(__file__).parent


def local_path(value):
    return project_path(ROOT, value)


@dataclass(frozen=True)
class StereoDepthRequest:
    source_split: str = "public"
    episode_index: int = 39
    start_frame: int = 0
    stop_frame: int | None = 20
    max_image_side: int = 768
    rectification_run: str | None = None
    depth_run: str | None = None
    stereo_run: str | None = None
    masks_run: str | None = None
    minimum_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    left_right_tolerance_px: float = 1.0
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 20.0
    minimum_support_pixels: int = 64
    minimum_frame_share: float = 0.5
    holdback_modulus: int = 0
    episode_aggregation: str = "median"
    vertical_tolerance_px: float = 3.0
    reprojection_tolerance_px: float = 2.0
    frame_indices: tuple[int, ...] | None = None

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation"):
            raise ValueError("source_split must be public or evaluation")
        for name in ("episode_index", "start_frame", "minimum_disparity", "num_disparities",
                     "block_size", "max_image_side", "minimum_support_pixels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"Invalid integer request field: {name}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if self.frame_indices is not None:
            if (not self.frame_indices or any(type(index) is not int or index < self.start_frame for index in self.frame_indices)
                    or list(self.frame_indices) != sorted(set(self.frame_indices))
                    or (self.stop_frame is not None and self.frame_indices[-1] >= self.stop_frame)):
                raise ValueError("Explicit frame identities must be unique, increasing and inside the selected interval")
        if type(self.holdback_modulus) is not int or self.holdback_modulus not in (0, 5):
            raise ValueError("holdback_modulus must be zero (legacy) or five")
        if self.episode_aggregation not in ("median", "log_median", "huber_log"):
            raise ValueError("Unknown episode aggregation")
        if self.stop_frame is not None and (type(self.stop_frame) is not int or self.stop_frame <= self.start_frame):
            raise ValueError("stop_frame is exclusive and must follow start_frame")
        if (self.num_disparities < 16 or self.num_disparities % 16 or
                not 3 <= self.block_size <= 11 or self.block_size % 2 != 1 or
                not 64 <= self.max_image_side <= 2048 or self.minimum_support_pixels < 1):
            raise ValueError("Invalid stereo resolution, block size, disparity range or support count")
        if not all(math.isfinite(value) for value in (self.minimum_depth_m, self.maximum_depth_m,
                    self.left_right_tolerance_px, self.minimum_frame_share,
                    self.vertical_tolerance_px, self.reprojection_tolerance_px)):
            raise ValueError("Stereo thresholds must be finite")
        if not (0 < self.minimum_depth_m < self.maximum_depth_m and
                0 < self.left_right_tolerance_px and 0 < self.minimum_frame_share <= 1):
            raise ValueError("Invalid depth, consistency or frame-share limits")
        if not 0 < self.vertical_tolerance_px <= 5 or not 0 < self.reprojection_tolerance_px <= 5:
            raise ValueError("Invalid sparse correspondence admission thresholds")


def authorize(operation, request):
    runtime = read_json(FEATURE / "runtime.json")
    if request.frame_indices is not None and operation not in ("anchor", "report", "correspondences"):
        raise ValueError("Explicit sparse frames are supported only by research anchor/correspondences/report")
    if operation in ("report", "correspondences") or (operation == "anchor" and
            (request.holdback_modulus or request.episode_aggregation != "median" or request.frame_indices is not None)):
        if request.source_split != "public" or request.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42):
            raise PermissionError("New geometry research is training/tuning only; 41/43/44 are excluded")
        if operation == "anchor" and runtime["execution"].get("anchor_research_authorized") is not True:
            raise PermissionError("Research anchor variants remain disabled")
        if operation == "correspondences" and (request.stop_frame is None or
                len(selected_indices(request, request.stop_frame)) > 32 or request.max_image_side > 384):
            raise ValueError("Sparse correspondence research requires <=32 explicit frames, maximum side384")
    if runtime["execution"].get(operation + "_authorized") is not True:
        raise PermissionError(f"Stereo {operation} remains disabled")
    authorize_episode(runtime, request, "stereo")


def select_episode(request):
    source = read_json("data/configs/sources.lock.json")["dataset"]
    episodes = read_json("data/manifests/dataset/episodes.json")
    videos = read_json("data/reports/data-audit/videos.json")
    policy = read_json("data/configs/calibration/ego.json")
    split = read_json("data/configs/splits/public-v1.json")
    if any(revision != source["revision"] for revision in
           (episodes["revision"], videos["revision"], policy["dataset_revision"], split["dataset_revision"])):
        raise ValueError("Stereo input revisions differ")
    if (policy["orientation"]["additional_pixel_rotation_deg"] != 0 or
            policy["stereo"]["left"] != "ego_cam_c" or policy["stereo"]["right"] != "ego_cam_b"):
        raise ValueError("Expected the recorded upright C-left/B-right calibration policy")
    matches = [row for row in episodes["episodes"] if row["split"] == request.source_split
               and row["episode_index"] == request.episode_index]
    if len(matches) != 1 or matches[0]["issues"]:
        raise ValueError("Exactly one usable inventoried episode is required")
    snapshot = local_path(Path(source["cache_path"]) / "snapshots" / source["revision"])
    clocks, paths = {}, {}
    for camera in ("ego_cam_a", "ego_cam_b", "ego_cam_c"):
        relative = matches[0]["videos"][camera]
        rows = [row for row in videos["videos"] if row["path"] == relative]
        if videos["status"] != "completed" or len(rows) != 1 or rows[0]["status"] != "ok":
            raise ValueError("Completed timestamp inventories are required for all three streams")
        clock = rows[0]
        if (clock.get("rotation_metadata_lines") or
                any(len(clock[key]) != clock["decoded_frame_count"] for key in ("pts_integer", "pts_seconds")) or
                clock["time_base_and_frame_rate"] != [["1/10240", "20/1"]]):
            raise ValueError("Incomplete timestamps, unexpected clock or unsupported rotation metadata")
        if clocks and clock["pts_integer"] != clocks["ego_cam_a"]["pts_integer"]:
            raise ValueError("The streams have different frame-index/PTS identities")
        clocks[camera], paths[camera] = clock, local_path(snapshot / relative)
    count = clocks["ego_cam_a"]["decoded_frame_count"]
    stop = count if request.stop_frame is None else request.stop_frame
    if not request.start_frame < stop <= count:
        raise ValueError("Requested interval is outside the inventoried clip")
    if request.frame_indices is not None and request.frame_indices[-1] >= stop:
        raise ValueError("Explicit frame identities exceed the inventoried clip")
    return {"revision": source["revision"], "paths": paths, "clocks": clocks, "stop": stop,
            "calibration": local_path(snapshot / "track_3/public/meta/camera_calibration.json")}


def selected_indices(request, stop):
    return list(request.frame_indices) if request.frame_indices is not None else list(range(request.start_frame, stop))


def frame_file(producer_path, source, index, pts, record):
    rows = [row for row in source["frames"] if row["frame_index"] == index]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["video_pts"] != pts:
        raise ValueError("Missing, incomplete or mismatched input frame")
    return artifact(producer_path, rows[0]["artifact"], record, rows[0]["sha256"])


def is_rigid(matrix, np):
    """True for a finite proper 4x4 rigid transform (to 1e-5)."""
    return (matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1])
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5))


@contextmanager
def stereo_captures(selection, record, cv2):
    """Open the C-left/B-right streams without auto-rotation, recording each as an input."""
    captures = []
    try:
        for camera in ("ego_cam_c", "ego_cam_b"):
            record["inputs"].append(input_file(ROOT, selection["paths"][camera]))
            capture = cv2.VideoCapture(str(selection["paths"][camera]), cv2.CAP_FFMPEG)
            captures.append(capture)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {camera}")
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        yield captures
    finally:
        for capture in captures:
            capture.release()


@contextmanager
def operation_run(operation, request, selection, parents=()):
    """Create the run directory and record; callers authorize before reading producers."""
    record = run_fields(ROOT, "stereo_depth", operation, asdict(request), parents=parents,
                        allowed=("video_only",), inputs=[selection["calibration"], FEATURE / "runtime.json",
                        "data/configs/sources.lock.json", "data/configs/calibration/ego.json", "data/configs/splits/public-v1.json",
                        "data/configs/perception/moge/pylock.python313.toml", "data/manifests/dataset/episodes.json",
                        "data/reports/data-audit/videos.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=selection["revision"], frames=[], artifacts={}, gpu=None)
    with recorded(output_path(ROOT, f"data/outputs/stereo_depth/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as result:
        yield result
