"""build: pool each object's measured proportions over episodes and write one closed primitive per object.

CD-O registers every candidate with its own scale and pose, so only proportions count. Each measure run
gives an object's median (in-plane long, in-plane short, height) in that episode's unanchored MoGe2
scale; build divides by the geometric mean, takes the per-axis median over episodes, and restores the
median geometric mean. The same object mesh is written for every scored (episode, object) pair.
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

from track3.paths import output_path, project_path
from track3.provenance import read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, dataset_revision, new_run_id, read_json, recorded, runtime_flag, same_revision

FEATURE = Path(__file__).parent
ROSTER = "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
SHAPES = ("ellipsoid", "box", "vessel_rule")


@dataclass(frozen=True)
class BuildRequest:
    measure_runs: list[str] = field(default_factory=list)
    shape: str = "ellipsoid"
    shape_overrides: dict[str, str] = field(default_factory=dict)
    ellipsoid_subdivisions: int = 3
    require_every_pair: bool = True
    aspect_threshold: float = 1.3
    handle_height_fraction: float = 0.6
    rule: str | None = None
    rule_parameter: float | None = None
    attributes_file: str | None = None
    rule_queue: str = "track3/features/mesh_extents/rules-round2.json"

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.rule is None and (self.rule_parameter is not None or self.attributes_file is not None):
            raise ValueError("A rule parameter or attributes file requires a named rule")
        if self.shape not in SHAPES or any(value not in SHAPES for value in self.shape_overrides.values()):
            raise ValueError(f"shape must be one of {SHAPES}")
        if not self.measure_runs:
            raise ValueError("Name at least one completed mesh_extents.measure run")
        if "vessel_rule" in (self.shape, *self.shape_overrides.values()):
            if not math.isfinite(self.aspect_threshold) or self.aspect_threshold < 1:
                raise ValueError("aspect_threshold must be finite and at least 1")
            if not math.isfinite(self.handle_height_fraction) or not 0 < self.handle_height_fraction <= 1:
                raise ValueError("handle_height_fraction must be in (0, 1]")


def _primitive(shape, dims, subdivisions, trimesh):
    if shape == "box":
        return trimesh.creation.box(extents=dims)
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.5)
    mesh.apply_scale(dims)
    return mesh


def _rule_inputs(request: BuildRequest):
    """Read the named frozen rule and its video-only attributes for a production build."""
    if request.rule is None:
        return None, {}, [], {}
    from .rules import read_queue

    queue_path = project_path(ROOT, request.rule_queue)
    queue = read_queue(queue_path)
    rule = next((row for row in queue["rules"] if row["name"] == request.rule), None)
    if rule is None or request.rule_parameter not in rule["grid"]:
        raise ValueError("Name a frozen rule and one of its recorded parameters")
    if request.attributes_file != queue["attributes"]:
        raise ValueError("Use the queue's frozen attributes file")
    attributes = read_json(request.attributes_file)
    if attributes["provenance"] != "video_only":
        raise ValueError("Production attributes require video-only provenance")
    return rule, attributes["objects"], [queue_path, request.attributes_file], queue.get("candidate_mesh_sha256", {})


def build(request: BuildRequest):
    runtime_flag(FEATURE, "build_authorized", "Mesh extents build")
    rule, attributes, rule_inputs, frozen_meshes = _rule_inputs(request)
    parents, episodes = [], set()
    for value in request.measure_runs:
        path = ROOT / value
        record = read_run(ROOT, path, kind="mesh_extents.measure", allowed=("video_only",))
        key = (record.get("split"), record.get("episode_index"))
        if record["status"] != "completed" or key[1] in (41, 43, 44) or key in episodes:
            raise ValueError(f"Not an admissible completed measure run, or a second run of one episode: {value}")
        episodes.add(key)
        parents.append((path, record))
    revision = same_revision(parents, dataset_revision())
    snapshot = asdict(request)
    if request.rule_queue == BuildRequest.rule_queue:
        snapshot.pop("rule_queue")
    if request.rule is None:
        for key in ("rule", "rule_parameter", "attributes_file"):
            snapshot.pop(key)
    record = run_fields(ROOT, "mesh_extents", "build", snapshot, parents=[path for path, _ in parents],
                        inputs=[ROSTER, FEATURE / "runtime.json", *rule_inputs], allowed=("video_only",))
    run_id = new_run_id()
    record.update(run_id=run_id, dataset_revision=revision, artifacts={})
    with recorded(output_path(ROOT, f"outputs/mesh_extents/build/{run_id}"), record) as (directory, record):
        import numpy as np
        import trimesh

        from . import _shapes
        from ._rule_scores import CATEGORIES, pool_extents, shape_for
        from ._union import mesh_for

        contributions = {}
        for path, parent in parents:
            extents = read_json(artifact((path, parent), "extents.json", record))
            for name, row in extents["objects"].items():
                dims = np.asarray(row["median_m"], dtype=np.float64)
                if row["frames"] < 1 or not np.isfinite(dims).all() or (dims <= 0).any():
                    raise ValueError(f"{name} in {path}: no usable measurement")
                contributions.setdefault(name, []).append(
                    {"split": parent["split"], "episode_index": parent["episode_index"], "frames": row["frames"],
                     "median_m": dims.tolist(), "run": path.relative_to(ROOT).as_posix()})
        objects = {}
        (directory / "objects").mkdir()
        for name, rows in sorted(contributions.items()):
            pooled = pool_extents([row["median_m"] for row in rows])
            shape = request.shape_overrides.get(name, request.shape)
            if rule is not None:
                from .rules import check_mesh

                shape = shape_for(rule, name, pooled, request.rule_parameter, attributes)
                mesh = mesh_for(shape, rule, name, pooled, request.rule_parameter, attributes)
                check_mesh(mesh, pooled)
            elif shape == "vessel_rule" and name in CATEGORIES["round_vessel"]:
                if pooled[0] / pooled[1] > request.aspect_threshold:
                    shape = "vessel_with_handle"
                    mesh = _shapes.vessel_with_handle(pooled, request.handle_height_fraction)
                else:
                    shape = "cylinder"
                    mesh = _shapes.cylinder(pooled)
            else:
                shape = "box" if shape == "vessel_rule" else shape
                mesh = _primitive(shape, pooled, request.ellipsoid_subdivisions, trimesh)
            if not mesh.is_watertight:
                raise ValueError(f"{name}: primitive is not closed")
            target = directory / "objects" / f"{name}.ply"
            mesh.export(target)
            mesh_hash = sha256_file(target)
            if frozen_meshes and mesh_hash != frozen_meshes[name]:
                raise ValueError(f"Production mesh differs from frozen geometry: {name}")
            objects[name] = {"shape": shape, "extents_m": pooled.tolist(), "proportions": (pooled / pooled.max()).tolist(),
                             "episodes": rows, "file": target.relative_to(directory).as_posix(),
                             "sha256": mesh_hash, "faces": int(len(mesh.faces))}
        roster = read_json(ROSTER)["episodes"]
        missing, pairs = [], []
        for episode, names in sorted(roster.items(), key=lambda item: int(item[0])):
            folder = directory / "meshes" / f"episode_{int(episode):06d}"
            for name in names:
                if name not in objects:
                    missing.append(f"{episode}:{name}")
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{name}.ply").write_bytes((directory / objects[name]["file"]).read_bytes())
                pairs.append(f"{episode}:{name}")
        if missing and request.require_every_pair:
            raise ValueError(f"No measured object for scored pairs: {missing}")
        record.update(objects=sorted(objects), pairs_written=len(pairs), pairs_missing=missing)
        write_json(directory / "objects.json", {"objects": objects, "pairs_written": pairs, "pairs_missing": missing,
                                                "scale": "unanchored MoGe2 units; CD-O ignores scale",
                                                "provenance": record["provenance"]})
    return directory / "run.json"
