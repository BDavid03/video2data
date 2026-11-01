"""Private input, artifact and execution conventions for the scratch-depth slice."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from track3.paths import output_path, project_path
from track3.provenance import environment, run_fields, sha256_file, write_json
from track3.runs import ROOT, read_json, runtime_flag, new_run_id, recorded, save_npz, source_hashes  # noqa: F401
from track3 import runs

FEATURE = Path(__file__).parent
ARCHITECTURE = "coursework-depth-cnn-v1"


def local_path(value):
    return project_path(ROOT, value)


def relative(path):
    return local_path(path).relative_to(ROOT).as_posix()


def within(directory, value):
    path = local_path(directory / value)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact reference escapes its run directory")
    return path


def positive_integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: {value!r}")


def authorize(operation):
    """Refuse before heavy imports or artifact creation; redirects caches into the checkout."""
    return runs.heavy_execution(runtime_flag(FEATURE, f"{operation}_authorized", f"Scratch-depth {operation}"))


@contextmanager
def operation_run(operation, request, *, inputs=(), parents=()):
    authorize(operation)
    contract = run_fields(ROOT, "scratch_depth", operation, asdict(request), inputs=[
        *inputs, FEATURE / "runtime.json", FEATURE / "coursework.json",
        "configs/sources.lock.json", "configs/splits/public-v1.json",
        "configs/calibration/ego.json", "configs/perception/moge/pylock.python313.toml",
        "manifests/dataset/episodes.json", "reports/data-audit/videos.json"], parents=parents)
    record = {"schema_version": 1, "feature": "scratch_depth", "operation": operation, "status": "running",
              "run_id": new_run_id(), "request": asdict(request), "architecture": ARCHITECTURE,
              "environment": environment(), "artifacts": {}, "tests_run": False, "validation_run": False,
              "official_evaluation": False, "confidence": None, **contract}
    record["source_hashes"] = source_hashes(FEATURE, extra=["configs/perception/moge/pylock.python313.toml"],
                                            suffixes=(".py", ".json", ".md", ".cpp", ".cu", ".h", ".cuh"))
    record["coursework_hashes"] = {source["path"]: sha256_file(local_path(source["path"]))
                                   for source in read_json(FEATURE / "coursework.json")["sources"]}
    record["sources"] = [{"path": path, "sha256": digest}
                         for path, digest in {**record["source_hashes"], **record["coursework_hashes"]}.items()]
    directory = output_path(ROOT, f"outputs/scratch_depth/{operation}/{record['run_id']}")
    try:
        with recorded(directory, record) as result:
            write_json(directory / "source-files.json", {"files": record["sources"]})
            yield result
    finally:
        record["elapsed_seconds"] = record["timing"]["wall_seconds"]
        write_json(directory / "run.json", record)


def select_episode(episode_index):
    if episode_index == 41:
        raise PermissionError("Episode 41 remains excluded from scratch-depth execution")
    sources = read_json("configs/sources.lock.json")["dataset"]
    split = read_json("configs/splits/public-v1.json")
    episodes = read_json("manifests/dataset/episodes.json")
    videos = read_json("reports/data-audit/videos.json")
    policy = read_json("configs/calibration/ego.json")
    revision = sources["revision"]
    if any(value != revision for value in (split["dataset_revision"], episodes["revision"],
                                           videos["revision"], policy["dataset_revision"])):
        raise ValueError("Scratch-depth inputs refer to different dataset revisions")
    if episode_index not in split["development_episode_indices"]:
        raise ValueError("Scratch depth is restricted to the frozen public development episodes")
    if policy["orientation"]["additional_pixel_rotation_deg"] != 0:
        raise ValueError("Scratch depth expects the released upright RGB pixels")
    matches = [item for item in episodes["episodes"]
               if item["split"] == "public" and item["episode_index"] == episode_index]
    if len(matches) != 1 or matches[0]["issues"]:
        raise ValueError("Exactly one usable inventoried public episode is required")
    episode = matches[0]
    clocks = [item for item in videos["videos"] if item["path"] == episode["videos"]["ego_cam_a"]]
    if videos["status"] != "completed" or len(clocks) != 1 or clocks[0]["status"] != "ok":
        raise ValueError("A completed RGB timestamp inventory is required")
    clock = clocks[0]
    if clock.get("rotation_metadata_lines"):
        raise ValueError("Rotation-tagged video requires a separate orientation decision")
    if any(len(clock[key]) != clock["decoded_frame_count"] for key in ("pts_integer", "pts_seconds")):
        raise ValueError("RGB timestamp inventory is incomplete")
    snapshot = local_path(Path(sources["cache_path"]) / "snapshots" / revision)
    video = local_path(snapshot / episode["videos"]["ego_cam_a"])
    calibration = local_path(snapshot / "track_3/public/meta/camera_calibration.json")
    return {
        "episode": episode, "clock": clock, "video": video, "revision": revision,
        "split_id": split["split_id"], "calibration": calibration,
        "camera": read_json(calibration)["cameras"]["ego_cam_a"],
    }


def pinhole(camera, max_side, cv2, np):
    """Canonical scratch-model pixel grid; no orientation or world transform."""
    width, height = camera["resolution"]
    k = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    d = np.asarray(camera["distortion_coefficients"], dtype=np.float64)
    if (camera["distortion_model"] != "Perspective" or d.shape != (14,) or k.shape != (3, 3)
            or not np.isfinite(k).all() or not np.isfinite(d).all()
            or min(width, height, k[0, 0], k[1, 1]) <= 0):
        raise ValueError("Expected the factory 14-coefficient Perspective camera")
    scale = min(1.0, max_side / max(width, height))
    size = (round(width * scale), round(height * scale))
    focal = k[0, 0] * size[0] / width
    output_k = np.array([[focal, 0, (size[0] - 1) / 2],
                         [0, focal, (size[1] - 1) / 2], [0, 0, 1]], dtype=np.float64)
    x, y = cv2.initUndistortRectifyMap(k, d, np.eye(3), output_k, size, cv2.CV_32FC1)
    valid = (np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1)
             & (y >= 0) & (y <= height - 1))
    geometry = {"camera": "ego_cam_a", "source_size_width_height": [width, height],
                "output_size_width_height": list(size), "input_intrinsics_pixels": output_k.tolist(),
                "factory_intrinsics_pixels": k.tolist(), "factory_distortion_coefficients": d.tolist(),
                "output_distortion": "zero", "additional_rotation_degrees": 0,
                "camera_frame": "OpenCV optical x-right y-down z-forward"}
    return x, y, valid, output_k, geometry


def open_video(inputs, cv2):
    capture = cv2.VideoCapture(str(inputs["video"]), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open the inventoried RGB video")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    return capture


def rgb_frame(capture, inputs, maps, cv2):
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("RGB decode stopped before the selected frame interval ended")
    width, height = inputs["camera"]["resolution"]
    if frame.shape[:2] != (height, width):
        raise ValueError("Decoded RGB dimensions differ from factory calibration")
    image = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
