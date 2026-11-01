"""measure: per-frame table-relative clusters of one episode's depth, and the extents of assigned objects."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

from track3.paths import output_path
from track3.provenance import run_fields, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, dataset_revision, new_run_id, producer,
                         read_json, recorded, runtime_flag, same_revision)

FEATURE = Path(__file__).parent


@dataclass(frozen=True)
class MeasureRequest:
    source_split: str = "public"
    episode_index: int = 39
    depth_run: str | None = None
    frames: list[int] | None = None  # None: every completed frame of the depth run
    objects: dict[str, list[int]] = field(default_factory=dict)  # object -> frame-0 cluster ids to unite
    assignment_note: str | None = None
    reference_pixels: dict[str, list[int]] = field(default_factory=dict)  # reviewed frame-0 size of each id
    reference_tolerance: float = 0.15
    minimum_height_m: float = 0.008
    maximum_height_m: float = 0.5
    neighbour_distance_m: float = 0.02
    minimum_cluster_pixels: int = 300
    overhang_m: float = 0.15
    table_band_m: float = 0.015  # table region (radius, refit ring); MoGe2 table tops curve by about 1 cm
    ring_pixels: int = 40
    track_dilation_pixels: int = 15
    pixel_ratio_range: tuple[float, float] = (0.8, 1.25)  # frames kept against the first frame's object mask size
    seed: int = 31

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation") or self.episode_index in (41, 43, 44):
            raise PermissionError("Only public development or evaluation episodes; 41, 43 and 44 are excluded")
        if not all(math.isfinite(v) and v > 0 for v in (self.minimum_height_m, self.maximum_height_m,
                                                       self.neighbour_distance_m, self.overhang_m)):
            raise ValueError("Geometry thresholds must be finite and positive")
        if self.minimum_height_m >= self.maximum_height_m:
            raise ValueError("minimum_height_m must be below maximum_height_m")
        if any(not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
               for ids in self.objects.values()):
            raise ValueError("Each assigned object needs a non-empty list of frame-0 cluster ids")
        claimed = [i for ids in self.objects.values() for i in ids]
        if len(claimed) != len(set(claimed)):
            raise ValueError("A cluster may belong to one object only")


def _overlay(rgb, label, rows, np, cv2):
    colours = np.array([(0, 220, 0), (0, 160, 255), (255, 80, 200), (255, 220, 0), (0, 255, 255), (160, 80, 255)])
    vis = np.zeros_like(rgb)
    for row in rows:
        vis[label == row["cluster"] + 1] = (0, 0, 255) if row["touches_border"] else colours[row["cluster"] % len(colours)]
    image = cv2.addWeighted(rgb, 0.45, vis, 0.55, 0)
    for row in rows:
        ys, xs = np.nonzero(label == row["cluster"] + 1)
        cv2.putText(image, str(row["cluster"]), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
    return np.concatenate([rgb, image], axis=1)


def measure(request: MeasureRequest):
    runtime = runtime_flag(FEATURE, "measure_authorized", "Mesh extents measurement")
    authorize_episode(runtime, request, "mesh extents measurement")
    depth = producer(request.depth_run, "depth.generate", request)
    revision = same_revision([depth], dataset_revision())
    record = run_fields(ROOT, "mesh_extents", "measure", asdict(request), parents=[depth[0]], allowed=("video_only",),
                        inputs=["configs/sources.lock.json", "configs/splits/public-v1.json", FEATURE / "runtime.json",
                                "track3/features/object_masks/opening_clusters.py"])  # _clusters fixes cluster ids
    run_id = new_run_id()
    record.update(run_id=run_id, split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, frames=[], artifacts={},
                  scale="MoGe2 monocular depth, not anchored: extents share one unknown scale per frame; "
                        "proportions are what this run measures")
    directory = output_path(ROOT, f"outputs/mesh_extents/{request.source_split}/episode_{request.episode_index:06d}/{run_id}")
    with recorded(directory, record) as (directory, record):
        import cv2
        import numpy as np
        from ._geometry import frame_clusters, object_extents, object_mask

        rng = np.random.default_rng(request.seed)
        completed = {row["frame_index"]: row for row in depth[1]["frames"] if row["status"] == "completed"}
        frames = sorted(completed) if request.frames is None else list(request.frames)
        if not frames or any(index not in completed for index in frames):
            raise ValueError("Every requested frame must be a completed frame of the depth run")
        with np.load(artifact(depth[0], "preprocessing.npz", record, depth[1]["preprocessing_artifact_sha256"]),
                     allow_pickle=False) as arrays:
            k, input_valid = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].astype(bool)
            source_x, source_y = arrays["source_x"].copy(), arrays["source_y"].copy()
        reference = {}
        for position, index in enumerate(frames):
            row = {"frame_index": index, "status": "running"}
            record["frames"].append(row)
            path = artifact(depth[0], completed[index]["artifact"], record, completed[index]["sha256"])
            with np.load(path, allow_pickle=False) as arrays:
                z, valid = arrays["depth_m"].astype(np.float64), arrays["valid"] & arrays["input_valid"]
            label, clusters, plane, scene = frame_clusters(z, valid & input_valid, k, request, rng)
            masks = {}
            for name, ids in request.objects.items():
                if position == 0:
                    if max(ids) >= len(clusters):
                        raise ValueError(f"{name}: cluster id beyond the {len(clusters)} clusters of frame {index}")
                    # Ids come from a size ordering; a rerun that reorders them must not rename objects silently.
                    for i, expected in zip(ids, request.reference_pixels.get(name, [])):
                        if abs(clusters[i]["pixels"] - expected) > request.reference_tolerance * expected:
                            raise ValueError(f"{name}: cluster {i} has {clusters[i]['pixels']} px, the reviewed "
                                             f"assignment had {expected}; re-check the overlay")
                    reference[name] = np.isin(label, [i + 1 for i in ids])
                    masks[name] = reference[name]
                else:
                    masks[name] = object_mask(label, reference[name], request.track_dilation_pixels)
            # Exclusive ownership: a later-frame pixel claimed by two objects belongs to neither.
            shared = sum(mask.astype(np.int8) for mask in masks.values()) > 1 if masks else None
            objects = {name: object_extents(scene, mask & ~shared, request) if (mask & ~shared).any() else None
                       for name, mask in masks.items()}
            row["shared_pixels"] = int(shared.sum()) if masks else 0
            np.savez_compressed(directory / f"labels_{index:06d}.npz", label=label)
            if position == 0:
                video = next((entry for entry in depth[1]["inputs"] if entry["path"] == depth[1]["source_video"]), None)
                if video is None:
                    raise ValueError("The depth run does not record its source video")
                checked_entry(video, record)
                capture = cv2.VideoCapture(str(ROOT / depth[1]["source_video"]))
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                capture.release()
                if not ok:
                    raise ValueError("Could not decode the first measured frame for the overlay")
                rgb = cv2.remap(image, source_x, source_y, cv2.INTER_LINEAR)
                cv2.imwrite(str(directory / f"overlay_{index:06d}.png"), _overlay(rgb, label, clusters, np, cv2))
            row.update(status="completed", plane=plane, clusters=clusters, objects=objects)
        summary = {}
        for name in request.objects:
            # A hand reaching the object merges with it; frames whose mask grew or shrank against the first
            # frame's are not the resting object and are left out.
            first = record["frames"][0]["objects"][name]["pixels"]
            used = [row["frame_index"] for row in record["frames"] if row["objects"][name]
                    and request.pixel_ratio_range[0] <= row["objects"][name]["pixels"] / first <= request.pixel_ratio_range[1]]
            values = [row["objects"][name] for row in record["frames"] if row["frame_index"] in used]
            dims = np.array([sorted(v["in_plane_m"], reverse=True) + [v["height_m"]] for v in values])
            # Each frame has its own unknown depth scale: pool shapes (dims over their geometric mean), then
            # restore the median size, so the frame scales cannot change the proportions.
            size = np.exp(np.log(dims).mean(axis=1))
            pooled = np.median(dims / size[:, None], axis=0) * np.median(size)
            summary[name] = {"frames": len(values), "frames_used": used, "median_m": pooled.tolist(),
                             "spread_m": (np.percentile(dims, 90, axis=0) - np.percentile(dims, 10, axis=0)).tolist(),
                             "axes": ["in_plane_long", "in_plane_short", "height"]}
        record["objects"] = summary
        write_json(directory / "extents.json", {"objects": summary, "assignment": request.objects,
                                                "assignment_note": request.assignment_note, "scale": record["scale"],
                                                "provenance": record["provenance"]})
    return directory / "run.json"
