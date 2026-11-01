"""Recorded CPU kinematic placement diagnostic under protocol Amendment 1."""
from dataclasses import dataclass, asdict
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import numpy as np
from track3.paths import output_path
from track3.provenance import run_fields, write_json
from track3.runs import (ROOT, read_json, producer, artifact, runtime_flag, authorize_episode,
                         heavy_execution, recorded, new_run_id)
from .scene import D1, KIT, select_measure
from ._scene import camera_points
from ._placement import (place_boxes, convention_estimate, apply_conventions, transform_errors,
                         oracle_replacements, flip_anchor)

FEATURE = Path(__file__).parent
PROTOCOL = ROOT / "docs/chain/pose-placement-protocol.md"
OUTPUT = "outputs/pose_insurance/placement"
ROSTER = {0: ["white_pot", "white_pot_lid"], 11: ["blue_cup", "beige_cup"],
          12: ["blue_cup", "plastic_dish_rack"], 18: ["mini_sweeper", "mini_dust_pan"],
          21: ["blue_cup", "water_pitcher"], 23: ["wooden_piece_1", "wooden_piece_2"],
          26: ["plastic_dish_rack", "white_pot"], 31: ["mini_sweeper", "plastic_dish_rack"]}
CONFIRM_ROSTER = {2: ["white_pot", "white_pot_lid"], 27: ["plastic_dish_rack", "white_pot"]}


def episode_names(episode):
    """Resolve explicit analogue slots without expanding the convention-fitting train set."""
    return (CONFIRM_ROSTER if episode in CONFIRM_ROSTER else ROSTER)[episode]


def placement_file(episode):
    """Keep confirmation observations separate from the fixed train placements."""
    return FEATURE / ("placements_confirm.json" if episode in CONFIRM_ROSTER else "placements_train.json")


@dataclass(frozen=True)
class PlacementRequest:
    episode_index: int
    candidate: str = "v1"
    k1: str | None = None
    source_split: str = "public"

    def __post_init__(self):
        if self.source_split != "public" or self.episode_index not in (ROSTER | CONFIRM_ROSTER):
            raise PermissionError("Placement diagnostics require a protocol train or confirmation episode")
        if self.episode_index in CONFIRM_ROSTER and self.candidate not in ("v1", "K1a"):
            raise PermissionError("Amendment 2 confirmation compares only v1 and K1a")
        if self.candidate not in ("v1", "K1a", "K1b", "K2", "K1+K2"):
            raise ValueError("Unknown placement candidate")
        if (self.candidate == "K1+K2" and self.k1 not in ("K1a", "K1b")) or (
                self.candidate != "K1+K2" and self.k1 is not None):
            raise ValueError("Only the cumulative candidate takes an explicit winning K1")

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


@lru_cache(maxsize=1)
def baseline_module():
    """Load the existing baseline's kit integration without executing its campaign."""
    spec = importlib.util.spec_from_file_location("placement_static_metric", ROOT / "scripts/track3/static_pose_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=10)
def reference(episode):
    """Load nearest-visible-filled mocap and reorder explicitly into analogue slots."""
    if episode not in (ROSTER | CONFIRM_ROSTER):
        raise PermissionError("Reference episode outside placement protocol")
    names, poses = baseline_module().load_reference(episode)
    return poses[:, :, [names.index(name) for name in episode_names(episode)]]


def label_paths(episode, all_train=False):
    dataset = next(ROOT.glob("datasets--*/snapshots/*/track_3/public"))
    return [dataset / f"data/chunk-000/episode_{ep:06d}.parquet"
            for ep in (ROSTER if all_train else [episode])]


def conventions(episode):
    """Fit per-object up conventions from other train episodes and report fold membership."""
    result = {}
    for name in ROSTER[episode]:
        fitting = [ep for ep, names in ROSTER.items() if ep != episode and name in names]
        rotations = [reference(ep)[0, 0, ROSTER[ep].index(name), 3:] for ep in fitting]
        result[name] = dict(up=convention_estimate(rotations).tolist(), episodes=fitting,
                            singleton_fallback=not fitting)
    return result


def video_geometry(request, record):
    """Admit video producers and pass their arrays to the shared scene computation."""
    placement_path = placement_file(request.episode_index)
    entries = read_json(placement_path)["episodes"][str(request.episode_index)]
    if request.episode_index in CONFIRM_ROSTER and entries["measurable"] is not True:
        raise ValueError("Confirmation episode cannot measure merged-cluster placement")
    d1 = producer(D1, "mesh_extents.build")
    measure = select_measure(request, d1, [entries])
    depth = producer(measure[1]["request"]["depth_run"], "depth.generate", request)
    objects = read_json(artifact(d1, "objects.json", record))["objects"]
    first = measure[1]["frames"][0]
    if first["frame_index"] != 0:
        raise ValueError("Placement protocol requires frame-zero measure labels")
    with np.load(artifact(measure, "labels_000000.npz", record)) as arrays:
        labels = arrays["label"]
    frame = next(row for row in depth[1]["frames"] if row["frame_index"] == 0 and row["status"] == "completed")
    with np.load(artifact(depth, frame["artifact"], record, frame["sha256"])) as arrays:
        z = arrays["depth_m"].astype(float)
    with np.load(artifact(depth, "preprocessing.npz", record, depth[1]["preprocessing_artifact_sha256"])) as arrays:
        points = camera_points(z, arrays["input_intrinsics_pixels"])
    artifact(measure, "overlay_000000.png", record)
    assignment = read_json(FEATURE.parent / "mesh_extents/assignments.json")["episodes"].get(
        f"public:{request.episode_index}", {"objects": {}})
    proportions, measurements, clusters, movable = [], [], [], []
    for name in episode_names(request.episode_index):
        proportions.append(objects[name]["extents_m"])
        measurements.append(measure[1].get("objects", {}).get(name, {}).get("median_m", objects[name]["extents_m"]))
        entry = entries["slots"].get(name)
        ids = entry["cluster_ids"] if entry else assignment["objects"][name]
        clusters.append(points[np.isin(labels, [i+1 for i in ids])] if ids is not None else None)
        movable.append(entry is not None)
    ids = entries["merged_cluster_ids"]
    merged = points[np.isin(labels, [i+1 for i in ids])] if ids is not None else None
    k1 = request.k1 if request.candidate == "K1+K2" else request.candidate if request.candidate in ("K1a", "K1b") else None
    geometry = place_boxes(proportions, measurements, clusters, np.asarray(first["plane"]["normal"], float),
                           movable, merged, k1, entries.get("stacked_slot", 1))
    return geometry


def diagnose(request):
    """Score one static candidate and its explicitly diagnostic oracle replacements."""
    runtime = runtime_flag(FEATURE, "placement_diagnostic_authorized", "Kinematic placement diagnostic")
    authorize_episode(runtime, request, "kinematic placement diagnostic")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    heavy_execution(dict(runtime, python=runtime["diagnostic_python"]))
    sampling = read_json(FEATURE / "diagnostic_sampling.json")
    if sampling["surface_points_per_object"] != 1000 or sampling["seed"] != 0:
        raise ValueError("The fixed protocol requires 1,000 scan samples per object with seed zero")
    entries = read_json(placement_file(request.episode_index))["episodes"][str(request.episode_index)]
    measure = producer(entries["measure_run"], "mesh_extents.measure", request)
    depth = producer(measure[1]["request"]["depth_run"], "depth.generate", request)
    baseline = baseline_module()
    names = episode_names(request.episode_index)
    dataset = next(ROOT.glob("datasets--*/snapshots/*/track_3/public"))
    meshes = []
    for name in names:
        mesh = dataset / f"mesh/{name}/{name}.glb"
        meshes.append(mesh if mesh.exists() else mesh.with_name(f"{name}_visual.glb"))
    record = run_fields(ROOT, "pose_insurance", "placement_diagnostic", asdict(request),
                        parents=[D1, measure[0], depth[0]], provenance="public_label_diagnostic",
                        inputs=[PROTOCOL, placement_file(request.episode_index), FEATURE / "runtime.json",
                                FEATURE / "diagnostic_sampling.json", FEATURE.parent / "mesh_extents/assignments.json",
                                ROOT / "scripts/track3/static_pose_baseline.py", KIT / "metric_code/track_3/AUC.py",
                                *label_paths(request.episode_index, "K2" in request.candidate), *meshes])
    record.update(split="public", episode_index=request.episode_index, object_names=names,
                  dataset_revision=depth[1]["dataset_revision"])
    directory = output_path(ROOT, f"{OUTPUT}/runs/{new_run_id()}")
    with recorded(directory, record):
        geometry = video_geometry(request, record)
        fold = conventions(request.episode_index) if "K2" in request.candidate else None
        reexpression = None
        if fold:
            reexpression = apply_conventions(geometry["quaternions"], geometry["dimensions"], [fold[n]["up"] for n in names])
            geometry["quaternions"] = reexpression.pop("quaternions")
        start = np.concatenate((geometry["positions"], geometry["quaternions"][:, [1, 2, 3, 0]]), axis=1)
        ref = reference(request.episode_index)
        kit = baseline.load_kit()
        vertices = np.stack([baseline.vertices(name) for name in names])
        def score(initial):
            return baseline.score(kit, np.repeat(initial[None, None], len(ref), axis=0), ref, vertices)
        metrics, oracle = score(start), score(ref[0, 0])
        replacements = {name: score(poses) for name, poses in oracle_replacements(start, ref[0, 0]).items()}
        flip = score(flip_anchor(start))
        result = dict(candidate=request.candidate, k1=request.k1, episode=request.episode_index,
                      names=names, frames=len(ref), metrics=metrics, oracle_static=oracle,
                      frame0=transform_errors(start, ref[0, 0]), oracle_replacements=replacements,
                      heading_flip=dict(as_built=metrics, flipped=flip, best=flip if flip["AUC"] > metrics["AUC"] else metrics),
                      scene={key: value.tolist() for key, value in geometry.items()}, calibration=fold,
                      geometry_reexpression={key: value.tolist() if isinstance(value, np.ndarray) else value
                                             for key, value in reexpression.items()} if reexpression else None,
                      sampling=sampling)
        write_json(directory / "metrics.json", result)
        record["metrics"] = metrics
    return directory / "run.json"
