"""Adapt admitted stereo/identity pose evidence into the joint solver contract."""

from dataclasses import asdict, dataclass
from collections import Counter
from datetime import datetime, timezone
import json
from uuid import uuid4

from track3.paths import output_path
from track3.provenance import input_file, run_fields, write_json
from ._common import (ROOT, FEATURE, arrays, artifact, clock_row, dataset_revision, local_path, read_json, recorded,
                      require_input, runtime_flag, same_revision)
from .refine import _parent


@dataclass(frozen=True)
class PrepareRefinementRequest:
    pose_run: str
    camera_run: str
    correspondence_run: str
    depth_run: str
    observations_file: str
    observations_sha256: str
    source_split: str = "public"
    episode_index: int = 39
    maximum_frames: int = 512
    maximum_points_per_frame: int = 128
    minimum_points: int = 6

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split != "public" or type(self.episode_index) is not int or self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42):
            raise PermissionError("Preparation admits frozen training/tuning only; 41/43/44 excluded")
        if type(self.maximum_frames) is not int or not 1 <= self.maximum_frames <= 512:
            raise ValueError("Preparation admits at most 512 source frames")
        if (type(self.maximum_points_per_frame) is not int or not 6 <= self.maximum_points_per_frame <= 128 or
                type(self.minimum_points) is not int or not 6 <= self.minimum_points <= self.maximum_points_per_frame):
            raise ValueError("Preparation admits 6–128 points/frame before splitting residual families")


def _states(pose, camera, request, revision, record, np, require_rigid):
    """Load dense pose/camera arrays on the pinned clock; composed poses must equal camera @ camA pose."""
    objects, motion = arrays(pose, "poses.npz", record, np), arrays(camera, "trajectory.npz", record, np)
    if (not np.array_equal(objects["pts"], motion["pts"]) or
            not np.array_equal(objects["frame_index"], np.arange(len(objects["pts"]))) or
            not np.array_equal(motion["frame_index"], objects["frame_index"])):
        raise ValueError("Preparation needs bounded dense source-aligned pose/camera clocks")
    count, bodies = len(objects["pts"]), len(objects["object_names"])
    if (objects["world_T_object"].shape != (count, bodies, 4, 4) or
            objects["camA_T_object"].shape != (count, bodies, 4, 4) or
            objects["world_valid"].shape != (count, bodies) or objects["valid"].shape != (count, bodies) or
            motion["world_T_camA"].shape != (count, 4, 4) or motion["valid"].shape != (count,) or
            motion["observed"].shape != (count,) or
            any(objects[key].dtype != np.bool_ for key in ("valid", "world_valid")) or
            motion["valid"].dtype != np.bool_ or motion["observed"].dtype != np.bool_ or
            not np.array_equal(objects["world_valid"], objects["valid"] & motion["valid"][:, None])):
        raise ValueError("Preparation requires explicit observed-state dimensions and boolean masks")
    require_rigid(objects["world_T_object"][objects["world_valid"]], np)
    require_rigid(objects["camA_T_object"][objects["valid"]], np)
    require_rigid(motion["world_T_camA"][motion["valid"]], np)
    frames, slots = np.nonzero(objects["world_valid"])
    if not np.allclose(objects["world_T_object"][frames, slots],
                       motion["world_T_camA"][frames] @ objects["camA_T_object"][frames, slots], atol=1e-6, rtol=0):
        raise ValueError("Preparation composed poses differ from the exact camera producer")
    error = "Preparation RGB clock differs from the pinned full source inventory"
    clock = clock_row(request, revision, error)
    if (clock["time_base_and_frame_rate"] != [["1/10240", "20/1"]] or
            not np.array_equal(objects["pts"], np.asarray(clock["pts_integer"], dtype=np.int64))):
        raise ValueError(error)
    return objects, motion


def _rows(admitted, names, ambiguous, objects, motion, np):
    """Split each admitted point into reprojection and (finite) depth records; skip ambiguous objects."""
    rows, skipped, frame_counts = [], [], Counter()
    for (frame, slot), (points, pixels, depths, k, sigma_px, sigma_z) in sorted(admitted.items()):
        if names[slot] in ambiguous or not motion["observed"][frame]:
            skipped.append({"frame_index": frame, "object_name": names[slot],
                            "reason": "symmetry_or_camera_observation_unsupported"})
            continue
        for point, pixel, depth in zip(points, pixels, depths):
            common = {"frame_index": frame, "video_pts": int(objects["pts"][frame]),
                      "role": "optimization", "motion_class": "dynamic", "entity": f"object:{slot}",
                      "local_point_m": point.tolist(), "intrinsics_pixels": k.tolist(),
                      "observed": True, "visible": True, "accepted": True, "occluded": False,
                      "body_point_origin": "identity_origin_direct_stereo_backprojection",
                      "identity_supported": True}
            rows.append(dict(common, family="reprojection", target=pixel.tolist(), sigma=sigma_px))
            if np.isfinite(depth):
                rows.append(dict(common, family="depth", target=float(depth), sigma=sigma_z, depth_valid=True))
        frame_counts[frame] += len(points) + int(np.isfinite(depths).sum())
    if not rows or any(count > 256 for count in frame_counts.values()):
        raise ValueError("Preparation needs admitted records, at most 256/frame across objects/families")
    return rows, skipped


def prepare_refinement(request):
    runtime = runtime_flag(FEATURE, "prepare_refinement_authorized", "Scene observation preparation")
    if request.episode_index not in runtime.get("refine_episode_indices", []):
        raise PermissionError("Named training/tuning episode is not authorized for preparation")
    directory = output_path(ROOT, f"outputs/scene_assembly/{request.source_split}/episode_{request.episode_index:06d}/prepare-refinement-" +
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    import numpy as np
    # The object-pose producer owns admission of measured body origins, direct stereo support and identity clocks.
    from track3.features.object_pose.refine import admit_refinement_observations
    from track3.features.object_pose._common import require_rigid

    pose = _parent(request.pose_run, request, ("object_pose.compose",))
    camera = _parent(request.camera_run, request, ("camera_motion.estimate",))
    tracks = _parent(request.correspondence_run, request, ("object_masks.correspondences",))
    stereo = _parent(request.depth_run, request, ("stereo_depth.generate",))
    mono = _parent(tracks[1]["request"]["depth_run"], request, ("depth.generate",))
    parents = [pose, camera, tracks, stereo]
    require_input(pose, camera)
    require_input(tracks, mono)
    require_input(stereo, mono)
    revision = same_revision([*parents, mono], dataset_revision())
    if any(parent[1].get("grid") != camera[1]["grid"] for parent in parents):
        raise ValueError("Observation preparation pixel grids differ")
    record = run_fields(ROOT, "scene_assembly", "prepare_refinement", asdict(request),
                        parents=[parent[0] for parent in [*parents, mono]],
                        inputs=[FEATURE / "runtime.json", "reports/data-audit/videos.json"], allowed=("video_only",))
    record["sources"].extend(input_file(ROOT, ROOT / "track3/features/object_pose" / name)
                             for name in ("refine.py", "_common.py"))
    record.update(split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=revision, grid=camera[1]["grid"], artifacts={}, frames=[])
    with recorded(directory, record):
        objects, motion = _states(pose, camera, request, revision, record, np, require_rigid)
        receipt_path = local_path(request.observations_file)
        if receipt_path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Source receipt exceeds 64 MiB admission ceiling")
        if len({row["frame_index"] for row in read_json(receipt_path).get("observations", [])}) > request.maximum_frames:
            raise ValueError("Source receipt exceeds the declared observed-frame budget")
        receipt, admitted = admit_refinement_observations(request, parents, objects, motion, record, np)
        if receipt.get("time_base") != [1, 10240]:
            raise ValueError("Joint preparation requires the pinned 1/10240 RGB clock")
        opening = read_json(artifact(pose[0], "opening.json", record, pose[1]["artifacts"]["opening.json"]))
        alternatives = receipt["symmetry_alternatives"]
        names = objects["object_names"].tolist()
        if not isinstance(alternatives, dict) or set(alternatives) != set(names):
            raise ValueError("Declare every object's symmetry alternatives, including empty lists")
        ambiguous = {item["name"] for item in opening["objects"] if item.get("rotation_ambiguous")}
        ambiguous.update(name for name, hypotheses in alternatives.items() if hypotheses)
        rows, skipped = _rows(admitted, names, ambiguous, objects, motion, np)
        bundle = {"schema": "scene_refinement_observations_v1", "units": "metres", "clock": "camA_rgb_pts",
                  "time_base": [1, 10240], "grid_sha256": camera[1]["grid"],
                  "pose_run_sha256": input_file(ROOT, pose[0])["sha256"],
                  "camera_run_sha256": input_file(ROOT, camera[0])["sha256"],
                  "parents": [input_file(ROOT, parent[0]) for parent in parents],
                  "source_receipt": input_file(ROOT, request.observations_file),
                  "observations": rows, "skipped": skipped,
                  "unsupported": ["static_background", "hands", "contact", "silhouette"],
                  "independent_evidence": "none; these records are fitting evidence"}
        # The joint consumer admits <=64 MiB JSON; enforce that bound before persistence.
        if len(json.dumps(bundle, allow_nan=False, indent=2).encode("utf-8")) + 1 > 64 * 1024 * 1024:
            raise ValueError("Prepared JSON exceeds 64 MiB; reduce explicit source budget")
        write_json(directory / "scene-refinement-observations.json", bundle)
        record.update(observation_records=len(rows), unsupported=bundle["unsupported"])
    return directory / "run.json"
