"""build: pool each object's measured proportions over episodes and write one closed primitive per object.

CD-O registers every candidate with its own scale and pose, so only proportions count. Each measure run
gives an object's median (in-plane long, in-plane short, height) in that episode's unanchored MoGe2
scale; build divides by the geometric mean, takes the per-axis median over episodes, and restores the
median geometric mean. The same object mesh is written for every scored (episode, object) pair.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from track3.paths import output_path
from track3.provenance import read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, dataset_revision, new_run_id, read_json, recorded, runtime_flag, same_revision

FEATURE = Path(__file__).parent
ROSTER = "upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json"
SHAPES = ("ellipsoid", "box")


@dataclass(frozen=True)
class BuildRequest:
    measure_runs: list[str] = field(default_factory=list)
    shape: str = "ellipsoid"
    shape_overrides: dict[str, str] = field(default_factory=dict)
    ellipsoid_subdivisions: int = 3
    require_every_pair: bool = True

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.shape not in SHAPES or any(value not in SHAPES for value in self.shape_overrides.values()):
            raise ValueError(f"shape must be one of {SHAPES}")
        if not self.measure_runs:
            raise ValueError("Name at least one completed mesh_extents.measure run")


def _primitive(shape, dims, subdivisions, trimesh):
    if shape == "box":
        return trimesh.creation.box(extents=dims)
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.5)
    mesh.apply_scale(dims)
    return mesh


def build(request: BuildRequest):
    runtime_flag(FEATURE, "build_authorized", "Mesh extents build")
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
    record = run_fields(ROOT, "mesh_extents", "build", asdict(request), parents=[path for path, _ in parents],
                        inputs=[ROSTER, FEATURE / "runtime.json"], allowed=("video_only",))
    run_id = new_run_id()
    record.update(run_id=run_id, dataset_revision=revision, artifacts={})
    with recorded(output_path(ROOT, f"outputs/mesh_extents/build/{run_id}"), record) as (directory, record):
        import numpy as np
        import trimesh

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
            dims = np.array([row["median_m"] for row in rows])
            size = np.exp(np.log(dims).mean(axis=1))
            pooled = np.median(dims / size[:, None], axis=0) * np.median(size)
            shape = request.shape_overrides.get(name, request.shape)
            mesh = _primitive(shape, pooled, request.ellipsoid_subdivisions, trimesh)
            if not mesh.is_watertight:
                raise ValueError(f"{name}: primitive is not closed")
            target = directory / "objects" / f"{name}.ply"
            mesh.export(target)
            objects[name] = {"shape": shape, "extents_m": pooled.tolist(), "proportions": (pooled / pooled.max()).tolist(),
                             "episodes": rows, "file": target.relative_to(directory).as_posix(),
                             "sha256": sha256_file(target), "faces": int(len(mesh.faces))}
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
