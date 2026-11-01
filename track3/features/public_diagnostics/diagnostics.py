"""Disabled public-only CPU reports using the unchanged kit metric modules."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from track3.paths import output_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit/v2dlb"
METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm")


def _path(value):
    path = (ROOT / Path(value)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Diagnostic inputs must stay inside the repository")
    return path


def _json(value):
    return json.loads(_path(value).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class DiagnosticRequest:
    episodes: list[int] = field(default_factory=lambda: [39])
    input_runs: dict[str, str] = field(default_factory=dict)
    mesh_baseline_runs: dict[str, str] = field(default_factory=dict)
    frozen_settings: str | None = None
    include_resting_axes: bool = True
    candidate_runs: dict[str, list[str]] = field(default_factory=dict)
    self_score_runs: dict[str, str] = field(default_factory=dict)
    time_window_frames: int = 3
    rest_position_m: float = 0.002
    rest_rotation_degrees: float = 2.0

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if (not isinstance(self.episodes, list) or not self.episodes or len(self.episodes) != len(set(self.episodes)) or
                any(type(value) is not int or value < 0 or value == 41 for value in self.episodes)):
            raise ValueError("Name unique public episodes explicitly, excluding episode 41")
        if not isinstance(self.input_runs, dict) or type(self.include_resting_axes) is not bool:
            raise ValueError("Invalid diagnostic input mapping or axis-report option")
        if (not isinstance(self.mesh_baseline_runs, dict) or
                any(not isinstance(key, str) or not isinstance(value, str) or not value.strip()
                    for key, value in self.mesh_baseline_runs.items())):
            raise ValueError("Mesh baselines must map episode strings to explicit reconstruction runs")
        if type(self.time_window_frames) is not int or not 0 <= self.time_window_frames <= 20:
            raise ValueError("Use a small nonnegative oracle time window")
        if not 0 < self.rest_position_m <= 0.1 or not 0 < self.rest_rotation_degrees <= 30:
            raise ValueError("Invalid label-rest thresholds")


def _authorize(operation, request):
    if request.mesh_baseline_runs and operation != "mesh":
        raise ValueError("mesh_baseline_runs applies only to mesh diagnostics")
    runtime = _json(FEATURE / "runtime.json")
    if runtime["execution"].get(operation.replace("-", "_") + "_authorized") is not True:
        raise PermissionError(f"Public diagnostic {operation} remains disabled")
    split = _json("configs/splits/public-v1.json")
    validation = []
    for episode in request.episodes:
        if episode in split["validation_episode_indices"]:
            if operation == "frame-convention":
                raise ValueError("Resting-axis convention selection is development-only")
            if runtime["execution"].get("validation_authorized") is not True or request.frozen_settings is None:
                raise PermissionError("Validation requires its separate gate and an explicit frozen-settings record")
            if _json(request.frozen_settings).get("status") != "frozen":
                raise ValueError("Validation settings record is not frozen")
            validation.append(episode)
        elif episode not in split["development_episode_indices"]:
            raise ValueError("Diagnostics accept only frozen public development/validation episode identities")
    return split, validation


def _kit_module(name):
    path = KIT / (name + ".py")
    spec = importlib.util.spec_from_file_location("_track3_kit_" + name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _record_input(entry, record):
    if entry not in record["inputs"]:
        record["inputs"].append(entry)


def _consume(path, record):
    _record_input(input_file(ROOT, path), record)
    return _path(path)


def _artifact_entry(parent, name, record):
    """Hash one producer-owned artifact once, check it against its producer and record it."""
    path = _path(parent[0].parent / name)
    if not path.is_relative_to(parent[0].parent):
        raise ValueError("Diagnostic artifact escapes its producer")
    expected = parent[1]["artifacts"].get(name)
    if expected is None:
        raise ValueError("Diagnostic trajectory differs from its producer record")
    entry = input_file(ROOT, path)
    if entry["sha256"] != expected:
        raise ValueError("Diagnostic trajectory differs from its producer record")
    _record_input(entry, record)
    return entry


def _artifact(parent, name, record):
    return _path(_artifact_entry(parent, name, record)["path"])


def _reference(snapshot, episode, inventory, record, np, pq, opening_only=False):
    rows = [row for row in inventory["episodes"] if row["split"] == "public" and row["episode_index"] == episode]
    if len(rows) != 1 or rows[0]["issues"]:
        raise ValueError("Exactly one usable public episode inventory is required")
    path = _consume(snapshot / rows[0]["trajectory"], record)
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    if b"objects" not in metadata:
        raise ValueError("Public labels do not record object slot order")
    if metadata.get(b"pose_convention") != b"world_T_object":
        raise ValueError("Public labels do not declare the world_T_object convention")
    names = json.loads(metadata[b"objects"].decode())
    if opening_only:
        batch = next(parquet.iter_batches(batch_size=1, columns=["observation.objects"]), None)
        raw_rows = [] if batch is None else batch.to_pylist()
    else:
        raw_rows = parquet.read(columns=["observation.objects"]).to_pylist()
    poses = np.empty((len(raw_rows), 1, len(names), 7), dtype=np.float64)
    for index, row in enumerate(raw_rows):
        by_name = {item["name"]: item for item in row["observation.objects"]}
        if len(by_name) != len(row["observation.objects"]) or set(by_name) != set(names):
            raise ValueError("Public label object membership changed within the episode")
        # Slot order is the metadata order; the row's own order is not relied on.
        for slot, item in enumerate(by_name[name] for name in names):
            values = np.asarray(item["pose"], dtype=np.float64)
            if not item["visible"] or values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("Invisible or malformed public label; labels are never filled")
            q = values[[4, 5, 6, 3]]
            norm = np.linalg.norm(q)
            if norm < 1e-8:
                raise ValueError("Public label has a degenerate quaternion")
            poses[index, 0, slot] = np.r_[values[:3], q / norm]
    if not len(poses) or not names:
        raise ValueError("Public reference is empty")
    return poses, names, rows[0]["task"]


def _matrix_poses(matrices, valid, np):
    """Valid rigid 4x4 poses to [x, y, z, qx, qy, qz, qw] via the symmetric quaternion eigenproblem (no SciPy)."""
    result = np.zeros((*valid.shape, 7), dtype=np.float64)
    result[..., 6] = 1
    m = matrices[valid]
    r = m[:, :3, :3]
    if (m.shape[1:] != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[:, 3], [0, 0, 0, 1])
            or not np.allclose(np.swapaxes(r, 1, 2) @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5).all()):
        raise ValueError("Observed trajectory contains a malformed rigid pose")
    k = np.stack([
        np.stack([r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], r[:, 1, 0] + r[:, 0, 1], r[:, 2, 0] + r[:, 0, 2], r[:, 2, 1] - r[:, 1, 2]], -1),
        np.stack([r[:, 1, 0] + r[:, 0, 1], r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], r[:, 2, 1] + r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0]], -1),
        np.stack([r[:, 2, 0] + r[:, 0, 2], r[:, 2, 1] + r[:, 1, 2], r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], r[:, 1, 0] - r[:, 0, 1]], -1),
        np.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1], np.trace(r, axis1=1, axis2=2)], -1),
    ], 1) / 3
    if len(k):
        result[valid] = np.concatenate([m[:, :3, 3], np.linalg.eigh(k)[1][:, :, -1]], axis=1)
    return result[:, None]


def _prediction(parent, episode, names, count, pts, record, np, pq):
    kind = parent[1]["kind"]
    # object_pose.initialise writes no world poses, so it has nothing to score.
    if kind in ("object_pose.compose", "scene_assembly.assemble"):
        name = "trajectories.npz" if kind.startswith("scene_assembly") else "poses.npz"
        with np.load(_artifact(parent, name, record), allow_pickle=False) as data:
            if (data["object_names"].tolist() != names or not np.array_equal(data["frame_index"], np.arange(count)) or
                    not np.array_equal(data["pts"], pts)):
                raise ValueError("Diagnostic trajectory has different object/frame/PTS identities")
            valid = np.ones((count, len(names)), dtype=bool) if kind.startswith("scene_assembly") else data["world_valid"].astype(bool)
            # A scene bundle is complete by contract; report how much of it was filled, not observed.
            filled = data["object_filled"].astype(bool) if kind.startswith("scene_assembly") else None
            matrices = data["world_T_object"].copy()
        if valid.shape != (count, len(names)) or matrices.shape != (count, len(names), 4, 4):
            raise ValueError("Diagnostic trajectory dimensions differ from the public reference")
        if filled is not None and filled.shape != valid.shape:
            raise ValueError("Scene bundle fill flags differ from the trajectory grid")
        return _matrix_poses(matrices, valid, np), valid, filled
    if kind != "policy_rollout.rollout":
        raise ValueError("Unsupported public trajectory producer")
    path = _artifact(parent, f"episode_{episode:06d}.parquet", record)
    rows = pq.read_table(path).to_pylist()
    achieved = np.zeros((count, 1, len(names), 7), dtype=np.float64)
    achieved[..., 6] = 1
    valid, seen = np.zeros((count, len(names)), dtype=bool), set()
    fields = ["achieved_position_" + axis for axis in "xyz"] + ["achieved_quaternion_" + axis for axis in "xyzw"]
    for row in rows:
        index, slot = row["reference_frame"], row["object_body_id"]
        if (row["environment_id"] != 0 or not 0 <= index < count or not 0 <= slot < len(names) or
                (index, slot) in seen or row["object_body_name"] != names[slot]):
            raise ValueError("Rollout public frame, world or object grid is invalid")
        seen.add((index, slot))
        value = np.asarray([row[key] for key in fields], dtype=np.float64)
        if row["environment_valid"] and np.isfinite(value).all() and np.linalg.norm(value[3:]) > 1e-8:
            value[3:] /= np.linalg.norm(value[3:])
            achieved[index, 0, slot], valid[index, slot] = value, True
    return achieved, valid, None


def _reference_vertices(snapshot, names, record, sampler, np):
    samples, missing = [], []
    for name in names:
        if name == "wooden_spoon":
            missing.append(name)
            continue
        path = _consume(sampler.mesh_path_for(snapshot / "track_3/public/mesh", name), record)
        samples.append(sampler.sample_vertices(path, count=500, seed=0))
    coverage = {"reference_scan_coverage": len(samples) / len(names),
                "reference_scan_object_count": len(samples), "requested_object_count": len(names),
                "missing_reference_objects": missing,
                "add_auc_unavailable_reason": "No wooden_spoon reference scan in the pinned public release" if missing else None}
    # ADD averages all objects. A subset would change the metric's support.
    return (None if missing else np.asarray(samples)), coverage


def _metric_values(achieved, reference, vertices, metric, np):
    scores = dict.fromkeys(METRICS)
    for name in METRICS:
        if name == "add_auc" and vertices is None:
            continue
        value = float(metric.episode_metric(name, achieved, reference, np.arange(achieved.shape[2]),
                                           vertices, align_initial_pose=False))
        scores[name] = value if np.isfinite(value) else None
    return scores


def _score_aligned(achieved, reference, valid, vertices, metric, np):
    complete = valid.all(axis=1)
    scores = _metric_values(achieved[complete], reference[complete], vertices, metric, np) if complete.any() else None
    penalty = dict(scores) if scores is not None else dict.fromkeys(METRICS)
    if not valid.all():
        auc = None
        if vertices is not None:
            add = metric._object_add(achieved, reference, vertices)
            add[:, 0][~valid] = np.inf
            auc = float(metric._add_auc(add))
        penalty.update(add_auc=auc, spider_sr=0.0, maniptrans_sr=0.0, rpe_cm=None, mppe_cm=None)
    return {"coverage": float(valid.mean()), "coverage_per_object": valid.mean(axis=0).tolist(),
            "status": "scored" if valid.all() else "scored_with_gaps", "complete_frame_count": int(complete.sum()),
            "valid_frames_only": scores, "missing_as_miss": penalty,
            "gap_policy": "AUC missing object-frames miss every threshold; incomplete episodes fail success gates. Distance metrics have no defined miss penalty and are null. Valid-only metrics use frames where all objects are observed, with the original alignment."}


def _score(achieved, reference, valid, vertices, metric, np):
    coverage = float(valid.mean())
    report = {"coverage": coverage, "coverage_per_object": valid.mean(axis=0).tolist()}
    if not valid[0, 0]:
        return {**report, "status": "unscorable", "reason": "slot-0 pose at frame 0 is missing"}
    warped = metric._warp_to_reference_start(achieved, reference)
    return _score_aligned(warped, reference, valid, vertices, metric, np)


def _score_means(rows, np):
    result = {"episode_count": sum(row.get("valid_frames_only") is not None for row in rows),
              "requested_episode_count": len(rows),
              "unscorable_episode_count": sum(row.get("status") == "unscorable" for row in rows),
              "label": "Equal episode weight on available support; metric denominators are explicit; nulls are never zero"}
    for mode in ("valid_frames_only", "missing_as_miss"):
        means, counts = {}, {}
        for name in METRICS:
            values = [row[mode][name] for row in rows if row.get(mode) is not None
                      and row[mode].get(name) is not None and np.isfinite(row[mode][name])]
            means[name] = float(np.mean(values)) if values else None
            counts[name] = len(values)
        result[mode], result[mode + "_episode_counts"] = means, counts
    # Preserve the existing top-level valid-only metric keys.
    result.update(result["valid_frames_only"])
    result["mean_pose_coverage"] = float(np.mean([row["coverage"] for row in rows])) if rows else None
    return result


def _report_files(directory, report):
    write_json(directory / "report.json", report)
    lines = ["# Public diagnostic report", "", "Reports only; never a pipeline input.", "",
             "| Episode | Subset | Task | Status | Coverage |", "|---|---|---|---|---|"]
    for row in report["episodes"]:
        lines.append(f"| {row['episode_index']} | {row['subset']} | {row.get('task', '')} | {row.get('status', 'reported')} | {row.get('coverage', '')} |")
    lines += ["", "```json", json.dumps(report, indent=2, allow_nan=False), "```", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]


@contextmanager
def _diagnostic_run(operation, record, report):
    """Own the run directory; always persist the report and run record, including on failure."""
    run_id = _run_id()
    directory = output_path(ROOT, f"outputs/public_diagnostics/{operation}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    record.update(run_id=run_id, split="public", gpu=None, artifacts={})
    started = time.perf_counter()
    write_json(directory / "run.json", record)
    try:
        yield directory
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(status=record["status"], inputs=record["inputs"])
        _report_files(directory, report)
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        record["artifacts"] = {name: sha256_file(directory / name) for name in ("report.json", "report.md")}
        write_json(directory / "run.json", record)


def _claim_validation(operation, validation, record, request):
    """Spend the score-once claim; call only after imports and revision checks have passed."""
    for episode in validation:
        claim = output_path(ROOT, f"outputs/public_diagnostics/validation/{operation}-episode_{episode:06d}.json")
        claim.parent.mkdir(parents=True, exist_ok=True)
        with claim.open("x", encoding="utf-8") as stream:
            json.dump({"run_id": record["run_id"], "settings": input_file(ROOT, request.frozen_settings)}, stream)


def _clock_row(clocks, episode, frame_count, message):
    rows = [item for item in clocks["videos"] if item["split"] == "public" and item["episode_index"] == episode
            and item["camera"] == "ego_cam_a"]
    if clocks["status"] != "completed" or len(rows) != 1 or rows[0]["status"] != "ok" or rows[0]["decoded_frame_count"] != frame_count:
        raise ValueError(message)
    return rows[0]


def _scan_origins(snapshot, revision, record, sampler, np, trimesh, report):
    """Distances from each public scan origin to its centroid, bounding-box centre and base centre."""
    public_names = _json("reports/data-audit/trajectories.json")
    if public_names["revision"] != revision:
        raise ValueError("Public scan-name inventory revision differs")
    for name in sorted({name for row in public_names["episodes"] for name in row["objects"]}):
        if name == "wooden_spoon":
            report.setdefault("unavailable_scans", []).append({"name": name,
                "reason": "No wooden_spoon reference scan in the pinned public release"})
            continue
        path = _consume(sampler.mesh_path_for(snapshot / "track_3/public/mesh", name), record)
        vertices = np.asarray(trimesh.load(str(path), force="mesh", process=False).vertices, dtype=np.float64)
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError("Invalid scan vertices")
        centroid = vertices.mean(axis=0)
        bounds_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        base_center = bounds_center.copy()
        base_center[2] = vertices[:, 2].min()
        candidates = {"vertex_centroid": float(np.linalg.norm(centroid)),
                      "bounding_box_center": float(np.linalg.norm(bounds_center)),
                      "base_center": float(np.linalg.norm(base_center))}
        report["scans"].append({"name": name, "origin_minus_vertex_centroid_m": (-centroid).tolist(),
            "origin_minus_bbox_center_m": (-bounds_center).tolist(),
            "origin_minus_lowest_point_m": (-vertices[np.argmin(vertices[:, 2])]).tolist(),
            "minimum_z_m": float(vertices[:, 2].min()), "origin_candidate_distances_m": candidates,
            "closest_origin_candidate": min(candidates, key=candidates.get)})


def _resting_axes(reference, names, metric, np):
    rotation = metric._rotation_matrix(reference[0, 0, :, 3:])
    objects = []
    for slot, name in enumerate(names):
        alignment = rotation[slot, 2]
        axis = int(np.argmax(abs(alignment)))
        objects.append({"name": name, "body_axis_closest_to_up": ("+" if alignment[axis] >= 0 else "-") + "xyz"[axis],
                        "axis_alignment_cosine": float(abs(alignment[axis])), "origin_height_m": float(reference[0, 0, slot, 2])})
    return objects


def _convention_summary(report):
    observed = [item for row in report["episodes"] for item in row.get("objects", [])]
    axes = {item["body_axis_closest_to_up"] for item in observed}
    covered = {item["name"] for item in observed}
    report["shared_resting_axis"] = next(iter(axes)) if len(axes) == 1 and covered == {item["name"] for item in report["scans"]} else None
    origin_candidates = {item["closest_origin_candidate"] for item in report["scans"]}
    report["origin_candidate_agreement"] = len(origin_candidates) == 1
    report["shared_closest_origin_candidate"] = next(iter(origin_candidates)) if len(origin_candidates) == 1 else None
    report["convention_note"] = "Closest-origin agreement is an inference from the reported distances, not an official convention. A common resting axis is reported only when every scan has a selected development observation. Review the offsets and alignments before choosing a spoon-frame setting."


def _score_parents(request):
    if set(request.input_runs) != {str(episode) for episode in request.episodes}:
        raise ValueError("Score requires exactly one explicitly named run per public episode")
    parents = {}
    for episode in request.episodes:
        path = _path(request.input_runs[str(episode)])
        source = read_run(ROOT, path)
        if source.get("split") != "public" or source.get("episode_index") != episode:
            raise ValueError("Diagnostics refuse evaluation trajectories and episode mismatches")
        parents[episode] = (path, source)
    return parents


def diagnose(operation, request: DiagnosticRequest):
    if operation in ("oracle", "agreement", "mesh"):
        _authorize(operation, request)
        from .advanced import diagnose_advanced
        return diagnose_advanced(operation, request)
    if operation not in ("floor", "score", "frame-convention"):
        raise ValueError("Only floor, score and frame-convention are implemented")
    split, validation = _authorize(operation, request)
    parents = _score_parents(request) if operation == "score" else {}
    convention = operation == "frame-convention"
    provenance = "public_asset_diagnostic" if convention and not request.include_resting_axes else "public_label_diagnostic"
    inputs = [FEATURE / "runtime.json", "configs/sources.lock.json", "configs/splits/public-v1.json",
              "manifests/dataset/episodes.json", "reports/data-audit/videos.json", "reports/data-audit/trajectories.json",
              KIT / "object_pose.py", KIT / "vertices.py"]
    if request.frozen_settings:
        inputs.append(request.frozen_settings)
    record = run_fields(ROOT, "public_diagnostics", operation.replace("-", "_"), asdict(request),
                        parents=[path for path, _ in parents.values()], inputs=inputs, provenance=provenance)
    record["sources"] += [input_file(ROOT, KIT / name) for name in ("object_pose.py", "vertices.py")]
    report = {"operation": operation, "provenance": record["provenance"], "episodes": [], "scans": [], "means": {}}
    with _diagnostic_run(operation, record, report) as directory:
        import numpy as np
        import pyarrow.parquet as pq
        import trimesh

        metric, sampler = _kit_module("object_pose"), _kit_module("vertices")
        source = _json("configs/sources.lock.json")["dataset"]
        snapshot = _path(Path(source["cache_path"]) / "snapshots" / source["revision"])
        inventory, clocks = _json("manifests/dataset/episodes.json"), _json("reports/data-audit/videos.json")
        if any(revision != source["revision"] for revision in (inventory["revision"], clocks["revision"], split["dataset_revision"])):
            raise ValueError("Diagnostic dataset, split and clock revisions differ")
        record["dataset_revision"] = source["revision"]
        if any(parent[1]["dataset_revision"] != source["revision"] for parent in parents.values()):
            raise ValueError("Trajectory producer dataset revision differs")
        _claim_validation(operation, validation, record, request)
        if convention:
            _scan_origins(snapshot, source["revision"], record, sampler, np, trimesh, report)
        for episode in request.episodes:
            subset = "validation" if episode in validation else "development"
            if convention and not request.include_resting_axes:
                continue
            reference, names, task = _reference(snapshot, episode, inventory, record, np, pq, opening_only=convention)
            row = {"episode_index": episode, "subset": subset, "task": task}
            if convention:
                row["objects"] = _resting_axes(reference, names, metric, np)
            else:
                clock = _clock_row(clocks, episode, len(reference), "Reference rows differ from the public video frame inventory")
                vertices, scan_coverage = _reference_vertices(snapshot, names, record, sampler, np)
                row.update(scan_coverage)
                if operation == "floor":
                    achieved = np.repeat(reference[:1], len(reference), axis=0)
                    valid = np.ones((len(reference), len(names)), dtype=bool)
                else:
                    achieved, valid, filled = _prediction(parents[episode], episode, names, len(reference), clock["pts_integer"], record, np, pq)
                    if filled is not None:
                        row["filled_share_per_object"] = filled.mean(axis=0).tolist()
                row.update(_score(achieved, reference, valid, vertices, metric, np))
            report["episodes"].append(row)
            _report_files(directory, report)
        for subset in ("development", "validation"):
            rows = [row for row in report["episodes"] if row["subset"] == subset]
            if rows and not convention:
                report["means"][subset] = _score_means(rows, np)
                report.setdefault("per_task", {})[subset] = {
                    task: _score_means([row for row in rows if row["task"] == task], np)
                    for task in sorted({row["task"] for row in rows})}
        if convention:
            _convention_summary(report)
    return directory / "run.json"
