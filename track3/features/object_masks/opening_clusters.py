"""Opening-only instance masks from table-relative metric connected components."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from track3.paths import output_path
from track3.provenance import run_fields, sha256_file, write_json
from track3.runs import (ROOT, artifact, save_npz as _save, authorize_episode, dataset_revision, new_run_id, producer, read_json,
                         recorded, runtime_flag, same_revision)

FEATURE = Path(__file__).parent
ROSTER = "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
PUBLIC_NAMES = "reports/data-audit/trajectories.json"


@dataclass(frozen=True)
class OpeningClustersRequest:
    source_split: str = "public"
    episode_index: int = 39
    depth_run: str | None = None
    anchor_run: str | None = None
    camera_run: str | None = None
    stop_frame: int = 20
    minimum_height_m: float = 0.015
    maximum_height_m: float = 0.5
    neighbour_distance_m: float = 0.025
    minimum_cluster_pixels: int = 50
    border_margin_pixels: int = 2

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation"):
            raise ValueError("source_split must be public or evaluation")
        for name in ("episode_index", "stop_frame", "minimum_cluster_pixels", "border_margin_pixels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (1 if name in ("stop_frame", "minimum_cluster_pixels") else 0):
                raise ValueError(f"Invalid opening-cluster request field: {name}")
        if self.episode_index == 41:
            raise PermissionError("Episode 41 remains excluded")
        if (not all(math.isfinite(value) for value in (self.minimum_height_m, self.maximum_height_m,
                                                     self.neighbour_distance_m)) or
                not 0 < self.minimum_height_m < self.maximum_height_m or self.neighbour_distance_m <= 0):
            raise ValueError("Opening-cluster geometry thresholds must be finite and positive")


def _authorize(request, flag, what):
    authorize_episode(runtime_flag(FEATURE, flag, what.capitalize()), request, what)


def _names_source(request):
    return ROSTER if request.source_split == "evaluation" else PUBLIC_NAMES


def _object_names(request, revision):
    """Ordered episode roster from the evaluation kit or the pinned public inventory."""
    inventory = read_json(_names_source(request))
    if request.source_split == "evaluation":
        names = inventory["episodes"][str(request.episode_index)]
    else:
        rows = [row for row in inventory["episodes"] if row["episode_index"] == request.episode_index]
        if inventory["revision"] != revision or len(rows) != 1:
            raise ValueError("A pinned public object-name inventory is required")
        names = rows[0]["objects"]
    if not names or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("Mask object names must be unique and in roster order")
    return names


def _clusters(points, keep, distance, minimum, np):
    """Four-connected components of kept pixels joined only across metric edges <= distance, largest first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ids = np.arange(keep.size).reshape(keep.shape)
    rows, cols = [], []
    with np.errstate(invalid="ignore"):
        for a, b in ((np.s_[:-1], np.s_[1:]), (np.s_[:, :-1], np.s_[:, 1:])):
            delta = points[b] - points[a]
            link = keep[a] & keep[b] & ((delta * delta).sum(axis=-1) <= distance * distance)
            rows.append(ids[a][link])
            cols.append(ids[b][link])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(keep.size, keep.size))
    labels = connected_components(graph, directed=False)[1].reshape(keep.shape)
    found, first, sizes = np.unique(labels[keep], return_index=True, return_counts=True)
    chosen = sizes >= minimum
    # Equal sizes keep raster order of each component's first pixel.
    order = np.lexsort((first[chosen], -sizes[chosen]))
    return labels[None] == found[chosen][order][:, None, None]


def _table_world(camera, grid, record, np):
    """Support plane (z-up normal, offset) and table extent from the camera-motion world."""
    world = read_json(artifact(camera, "world.json", record))
    if world["up_axis"] != "+z" or world["grid"] != grid:
        raise ValueError("Expected the colour-grid z-up table world")
    plane = world["support_plane"]
    normal, offset = np.asarray(plane["normal"], dtype=np.float64), float(plane["offset_m"])
    extent = np.asarray(plane["extent_xy_m"], dtype=np.float64)
    if (normal.shape != (3,) or not np.allclose(normal, [0, 0, 1]) or not math.isfinite(offset) or
            extent.shape != (2, 2) or not np.isfinite(extent).all() or (extent[0] >= extent[1]).any()):
        raise ValueError("Invalid support plane or table extent")
    return normal, offset, extent


def _above_table(index, pts, pose, scale_row, depth_row, depth, k, input_valid, table, request, record, np):
    """World points of one opening frame and the mask of measured pixels inside the above-table slab."""
    if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Invalid observed camera pose")
    if (scale_row is None or scale_row["scale_status"] != "observed" or scale_row["video_pts"] != pts or
            depth_row is None or depth_row["status"] != "completed" or depth_row["video_pts"] != pts):
        raise ValueError("Opening frame lacks a matching observed scale or depth")
    scale = scale_row["scale"]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid observed metric scale")
    path = artifact(depth[0], depth_row["artifact"], record, depth_row["sha256"])
    with np.load(path, allow_pickle=False) as arrays:
        z, valid = arrays["depth_m"].astype(np.float64) * scale, arrays["valid"].astype(bool)
        if (z.shape != input_valid.shape or valid.shape != z.shape or
                not np.allclose(k, arrays["input_intrinsics_pixels"], atol=1e-5)):
            raise ValueError("Opening depth differs from its declared grid")
    normal, offset, extent = table
    v, u = np.indices(z.shape)
    points = np.stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1)
    points = points @ pose[:3, :3].T + pose[:3, 3]
    height = points @ normal + offset
    keep = (valid & input_valid & np.isfinite(points).all(axis=-1) & (z > 0)
            & (height >= request.minimum_height_m) & (height <= request.maximum_height_m)
            & (points[..., 0] >= extent[0, 0]) & (points[..., 0] <= extent[1, 0])
            & (points[..., 1] >= extent[0, 1]) & (points[..., 1] <= extent[1, 1]))
    return points, keep


def opening_clusters(request: OpeningClustersRequest):
    _authorize(request, "opening_clusters_authorized", "opening clusters")
    depth = producer(request.depth_run, "depth.generate", request)
    anchor = producer(request.anchor_run, "stereo_depth.anchor", request)
    camera = producer(request.camera_run, "camera_motion.estimate", request)
    grid = depth[1]["preprocessing_artifact_sha256"]
    if anchor[1].get("grid") != grid or camera[1].get("grid") != grid:
        raise ValueError("Opening inputs use different colour pixel grids")
    revision = same_revision((depth, anchor, camera), dataset_revision())
    names = _object_names(request, revision)
    record = run_fields(ROOT, "object_masks", "opening_clusters", asdict(request),
                        parents=[path for path, _ in (depth, anchor, camera)], allowed=("video_only",),
                        inputs=[_names_source(request), "configs/sources.lock.json", "configs/splits/public-v1.json",
                                "configs/perception/moge/pylock.python313.toml", FEATURE / "runtime.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, grid=grid, frames=[], artifacts={}, gpu=None,
                  method="opening_clusters", instances=[], object_names=names,
                  opening_start_frame=0, opening_stop_frame=0)
    with recorded(output_path(ROOT, f"outputs/object_masks/{request.source_split}/episode_{request.episode_index:06d}/{run_id}"),
                  record) as (directory, record):
        import numpy as np
        from scipy.ndimage import binary_dilation

        with np.load(artifact(depth[0], "preprocessing.npz", record, grid), allow_pickle=False) as arrays:
            k, input_valid = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].astype(bool)
        table = _table_world(camera, grid, record, np)
        trajectory_path = artifact(camera, "trajectory.npz", record)
        with np.load(trajectory_path, allow_pickle=False) as arrays:
            trajectory = {name: arrays[name].copy() for name in arrays.files}
        count = len(trajectory["frame_index"])
        if (request.stop_frame > count or not np.array_equal(trajectory["frame_index"], np.arange(count)) or
                trajectory["world_T_camA"].shape != (count, 4, 4) or trajectory["valid"].shape != (count,)):
            raise ValueError("Camera trajectory has an invalid frame grid")
        anchor_path = artifact(anchor, "anchor.json", record)
        scales = {row["frame_index"]: row for row in read_json(anchor_path)["frames"]}
        depth_frames = {row["frame_index"]: row for row in depth[1]["frames"]}
        if len(depth_frames) != len(depth[1]["frames"]):
            raise ValueError("Duplicate depth frame identities")
        (directory / "frames").mkdir()
        # A cluster entering through the remap's invalid corners is a border entry too.
        margin = request.border_margin_pixels
        near_invalid = binary_dilation(~input_valid, iterations=margin) if margin else ~input_valid
        v, u = np.indices(input_valid.shape)
        border = ((v <= margin) | (u <= margin) | (v >= input_valid.shape[0] - 1 - margin)
                  | (u >= input_valid.shape[1] - 1 - margin) | near_invalid)
        first_count = previous_centroids = None
        for index in range(request.stop_frame):
            pts = int(trajectory["pts"][index])
            row = {"frame_index": index, "video_pts": pts, "status": "running"}
            record["frames"].append(row)
            if not trajectory["valid"][index]:
                raise ValueError(f"Missing camera observation at opening frame {index}")
            points, keep = _above_table(index, pts, trajectory["world_T_camA"][index], scales.get(index),
                                        depth_frames.get(index), depth, k, input_valid, table, request, record, np)
            layers = _clusters(points, keep, request.neighbour_distance_m, request.minimum_cluster_pixels, np)
            touches_border = bool((layers & border).any())
            if index == 0 and touches_border:
                raise ValueError("opening_not_quiet: an above-table cluster touches the image border")
            if first_count is None:
                first_count = len(layers)
                record["instances"] = [f"cluster_{i}" for i in range(first_count)]
                record["fewer_clusters_than_objects"] = first_count < len(names)
            elif touches_border or len(layers) != first_count:
                # The boundary frame has no artifact, so it must not read as a completed frame.
                row.update(status="not_processed", opening_boundary=True,
                           reason="border_entry" if touches_border else "cluster_count_changed")
                record["opening_end_reason"] = row["reason"]
                break
            centroids = np.stack([points[layer].mean(axis=0) for layer in layers]) if len(layers) else np.zeros((0, 3))
            if previous_centroids is not None and len(layers):
                # Frame 0 fixes layer order; later frames follow the previous frame's nearest world centroid.
                distance = np.linalg.norm(previous_centroids[:, None] - centroids[None], axis=-1)
                order = distance.argmin(axis=1)
                if (len(set(order.tolist())) != first_count or
                        (distance.argmin(axis=0)[order] != np.arange(first_count)).any()):
                    row.update(status="not_processed", opening_boundary=True, reason="cluster_identity_ambiguous")
                    record["opening_end_reason"] = row["reason"]
                    break
                layers, centroids = layers[order], centroids[order]
            previous_centroids = centroids
            destination = directory / "frames" / f"frame_{index:06d}.npz"
            _save(destination, np, masks=layers, present=np.ones(first_count, dtype=bool),
                  score=np.full(first_count, np.nan, dtype=np.float32), score_valid=np.zeros(first_count, dtype=bool),
                  source=np.full(first_count, 3, dtype=np.uint8),
                  frame_index=np.int64(index), pts=np.int64(pts))
            row.update(status="completed", artifact=destination.relative_to(directory).as_posix(),
                       sha256=sha256_file(destination), cluster_pixels=[int(layer.sum()) for layer in layers],
                       cluster_centroids_world=centroids.tolist())
            record["opening_stop_frame"] = index + 1
            write_json(directory / "run.json", record)
        write_json(directory / "keyframes.json", {"method": "opening_clusters", "keyframes": [],
                   "detector": None, "prompt": None, "scores_available": False})
        record["artifacts"]["keyframes.json"] = sha256_file(directory / "keyframes.json")
    return directory / "run.json"
