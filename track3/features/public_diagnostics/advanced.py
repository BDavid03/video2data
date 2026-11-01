"""Label-fitted attribution and ranking reports; never pipeline artifacts."""

from dataclasses import asdict
import importlib.util
import sys
import types

from track3.provenance import input_file, read_run, run_fields
from track3.paths import project_path
from .diagnostics import (ROOT, FEATURE, KIT, METRICS, _artifact, _authorize, _claim_validation, _clock_row,
                          _diagnostic_run, _json, _kit_module, _matrix_poses, _prediction, _reference,
                          _reference_vertices, _report_files, _score, _score_aligned, _score_means)
from ._mesh import compare_meshes, mesh_means

SUBSTITUTIONS = ("scale", "anchor", "body_frame", "time", "rest")


def _parent(value, episode, kind=None):
    path = project_path(ROOT, value)
    record = read_run(ROOT, path, kind=kind)
    if record.get("split") != "public" or record.get("episode_index") != episode:
        raise ValueError("Advanced diagnostics accept only the named public episode")
    return path, record


def _mesh_module():
    """Load the pinned kit's object_mesh (which imports v2dlb.mesh_common), then restore sys.modules."""
    keys = ("v2dlb", "v2dlb.mesh_common", "v2dlb.object_mesh")
    saved = {key: sys.modules.get(key) for key in keys}
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        package = types.ModuleType("v2dlb")
        package.__path__ = [str(KIT)]
        sys.modules["v2dlb"] = package
        for name in ("mesh_common", "object_mesh"):
            spec = importlib.util.spec_from_file_location("v2dlb." + name, KIT / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules["v2dlb." + name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _matrices(poses, metric, np):
    matrices = np.broadcast_to(np.eye(4), poses.shape[:-1] + (4, 4)).copy()
    matrices[..., :3, :3] = metric._rotation_matrix(poses[..., 3:])
    matrices[..., :3, 3] = poses[..., :3]
    return matrices[:, 0]


def _rigid(source, target, np):
    """Kabsch: the proper rigid transform taking source points onto target points."""
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _scale(aligned, reference, valid, np):
    origin = reference[0, 0, 0, :3]
    source = (aligned[:, 0, :, :3] - origin)[valid]
    target = (reference[:, 0, :, :3] - origin)[valid]
    denominator = float((source * source).sum())
    scale = float((source * target).sum() / denominator) if denominator > 1e-12 else 1.0
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    scaled = aligned.copy()
    scaled[..., :3] = origin + scale * (scaled[..., :3] - origin)
    return scaled, {"factor": scale, "origin": origin.tolist(), "fit": "Positive least-squares translation scale about slot-0 start"}


def _anchor(achieved, labels, valid, metric, np):
    predicted = _matrices(achieved, metric, np)
    keypoints = np.concatenate((np.eye(3), -np.eye(3))) * 0.05
    source_points = (np.einsum("tbij,kj->tbki", predicted[..., :3, :3], keypoints) + predicted[..., None, :3, 3])[valid].reshape(-1, 3)
    target_points = (np.einsum("tbij,kj->tbki", labels[..., :3, :3], keypoints) + labels[..., None, :3, 3])[valid].reshape(-1, 3)
    transform = _rigid(source_points, target_points, np)
    return _matrix_poses(transform @ predicted, valid, np), {
        "world_transform": transform.tolist(), "alignment": "All-frame rigid fit replaces frame-0 alignment; no second warp"}


def _body_frame(aligned, labels, valid, metric, np):
    """Per-object constant right-multiplied correction: chordal-mean rotation and mean translation."""
    matrices = _matrices(aligned, metric, np)
    corrected = matrices.copy()
    offsets = []
    for slot in range(valid.shape[1]):
        observed = valid[:, slot]
        delta = np.eye(4)
        if observed.any():
            relative = np.linalg.inv(matrices[observed, slot]) @ labels[observed, slot]
            u, _, vt = np.linalg.svd(relative[:, :3, :3].mean(axis=0))
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            delta[:3, :3] = u @ correction @ vt
            delta[:3, 3] = relative[:, :3, 3].mean(axis=0)
        corrected[:, slot] = matrices[:, slot] @ delta
        offsets.append(delta.tolist())
    return _matrix_poses(corrected, valid, np), {"object_side_transforms": offsets}


def _time_shift(aligned, reference, valid, vertices, metric, request, np):
    """The frame shift within the window with best AUC; ties prefer the smallest shift."""
    candidates = []
    target_indices = np.arange(len(aligned))
    for shift in range(-request.time_window_frames, request.time_window_frames + 1):
        candidate = aligned.copy()
        mask = np.zeros_like(valid)
        keep = (target_indices + shift >= 0) & (target_indices + shift < len(aligned))
        candidate[keep] = aligned[target_indices[keep] + shift]
        mask[keep] = valid[target_indices[keep] + shift]
        score = _score_aligned(candidate, reference, mask, vertices, metric, np)
        candidates.append((score["missing_as_miss"]["add_auc"], -abs(shift), shift, candidate, mask))
    _, _, shift, candidate, mask = max(candidates, key=lambda row: row[:2])
    return candidate, mask, {"shift_frames": shift, "window_frames": request.time_window_frames,
                             "boundary": "Out-of-range frames remain missing and count as misses"}


def _rest(aligned, reference, valid, metric, request, np):
    translation = np.linalg.norm(reference[:, 0, :, :3] - reference[0, 0, :, :3], axis=-1)
    angles = metric._quaternion_geodesic_angle(reference[:, 0, :, 3:], reference[0, 0, :, 3:])
    resting = (translation <= request.rest_position_m) & (angles <= np.deg2rad(request.rest_rotation_degrees)) & valid
    candidate = aligned.copy()
    candidate[:, 0][resting] = np.broadcast_to(reference[0, 0], candidate[:, 0].shape)[resting]
    return candidate, {"replaced_object_frames": int(resting.sum()), "position_m": request.rest_position_m,
                       "rotation_degrees": request.rest_rotation_degrees}


def _oracles(achieved, reference, valid, vertices, metric, request, np):
    """Score each label-fitted substitution independently against the warped baseline."""
    if not valid[0, 0]:
        return {"status": "unscorable", "reason": "Missing slot-0 start anchor", "coverage": float(valid.mean())}
    aligned = metric._warp_to_reference_start(achieved, reference)
    baseline = _score_aligned(aligned, reference, valid, vertices, metric, np)
    results = []

    def add(name, candidate, mask, parameters):
        score = _score_aligned(candidate, reference, mask, vertices, metric, np)
        gains = {}
        for key in METRICS:
            before, after = baseline["missing_as_miss"][key], score["missing_as_miss"][key]
            gains[key] = None if before is None or after is None else (before - after if key.endswith("_cm") else after - before)
        results.append({"substitution": name, "parameters": parameters, **score, "gain": gains})

    labels = _matrices(reference, metric, np)
    candidate, parameters = _scale(aligned, reference, valid, np)
    add("scale", candidate, valid, parameters)
    candidate, parameters = _anchor(achieved, labels, valid, metric, np)
    add("anchor", candidate, valid, parameters)
    candidate, parameters = _body_frame(aligned, labels, valid, metric, np)
    add("body_frame", candidate, valid, parameters)
    if vertices is not None:
        candidate, mask, parameters = _time_shift(aligned, reference, valid, vertices, metric, request, np)
        add("time", candidate, mask, parameters)
    else:
        results.append({"substitution": "time", "status": "unscorable",
                        "reason": "Time-shift selection requires ADD/AUC; no wooden_spoon reference scan",
                        "gain": dict.fromkeys(METRICS)})
    candidate, parameters = _rest(aligned, reference, valid, metric, request, np)
    add("rest", candidate, valid, parameters)
    if vertices is not None:
        results.sort(key=lambda row: row["gain"]["add_auc"], reverse=True)
    return {**baseline, "substitutions": results,
            "substitution_order": "Descending AUC gain" if vertices is not None else "Declared order; AUC unavailable",
            "interpretation": "Independent label-fitted diagnostics; never composed or exported"}


def _parents(operation, request):
    """Admit every named producer for the requested public episodes before any write."""
    parents, candidates, self_scores, mesh_baselines = {}, {}, {}, {}
    episodes = {str(episode) for episode in request.episodes}
    if request.mesh_baseline_runs:
        if operation != "mesh" or set(request.mesh_baseline_runs) != episodes:
            raise ValueError("Mesh comparison needs one baseline reconstruction for every requested episode")
        mesh_baselines = {episode: _parent(request.mesh_baseline_runs[str(episode)], episode,
                                          "object_mesh.reconstruct") for episode in request.episodes}
    if operation in ("oracle", "mesh"):
        if set(request.input_runs) != episodes:
            raise ValueError("Name one input run per episode")
        parents = {episode: _parent(request.input_runs[str(episode)], episode,
                                   "object_mesh.reconstruct" if operation == "mesh" else None) for episode in request.episodes}
    else:
        if set(request.candidate_runs) != episodes:
            raise ValueError("Agreement requires candidate runs for every named episode")
        for episode in request.episodes:
            values = request.candidate_runs[str(episode)]
            if len(values) < 2 or len(values) != len(set(values)):
                raise ValueError("Agreement needs distinct resting and tracked rollouts")
            candidates[episode] = [_parent(value, episode, "policy_rollout.rollout") for value in values]
            for value, parent in zip(values, candidates[episode]):
                self_scores[parent[0]] = _parent(request.self_score_runs[value], episode, "policy_rollout.self_score")
    return parents, candidates, self_scores, mesh_baselines


def _agreement(row, candidates, self_scores, reference, names, valid_clock, vertices, metric, record, np, pq):
    """Rank rollouts by self-score and by labels; report pairwise sign agreement."""
    values, scene_entries = [], []
    for parent in candidates:
        achieved, valid, _ = _prediction(parent, row["episode_index"], names, len(reference), valid_clock, record, np, pq)
        if not valid.all():
            raise ValueError("Agreement rankings require complete valid rollouts")
        score = _score(achieved, reference, valid, vertices, metric, np)
        self_report = _json(_artifact(self_scores[parent[0]], "self_score.json", record))
        rollout_entry = input_file(ROOT, parent[0])
        if self_report["rollout_run"] != rollout_entry:
            raise ValueError("Self-score belongs to a different rollout")
        for entry in (self_report["scene_run"], self_report["inspection_run"]):
            if input_file(ROOT, project_path(ROOT, entry["path"])) != entry:
                raise ValueError("Self-score lineage changed")
            record["inputs"].append(entry)
        scene_entries.append(self_report["scene_run"])
        values.append({"rollout_run": rollout_entry, "policy_kind": parent[1]["policy_kind"],
                       "label_metrics": score["valid_frames_only"], "self_metrics": self_report["metrics"],
                       "left_working_area": self_report["left_working_area"]})
    if any(entry != scene_entries[0] for entry in scene_entries):
        raise ValueError("Candidates must share one pinned tracked yardstick")
    resting = [value for value in values if value["policy_kind"] == "resting"]
    if len(resting) != 1 or not any(value["policy_kind"] in ("replay", "trained") for value in values):
        raise ValueError("Need one resting and at least one tracked candidate")
    if vertices is None:
        row.update(status="unscorable", candidates=values, pairs=[], challengers=[],
                   agreement_fraction=None, reason="AUC ranking requires every object's reference scan")
        return
    pairs = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            a, b = values[left], values[right]
            own = a["self_metrics"]["add_auc"] - b["self_metrics"]["add_auc"]
            label = a["label_metrics"]["add_auc"] - b["label_metrics"]["add_auc"]
            pairs.append({"left": left, "right": right, "self_margin": float(own), "label_margin": float(label),
                          "agrees": bool(np.sign(own) == np.sign(label))})
    challengers = [{"self_margin": value["self_metrics"]["add_auc"] - resting[0]["self_metrics"]["add_auc"],
                    "label_improves": value["label_metrics"]["add_auc"] > resting[0]["label_metrics"]["add_auc"],
                    "left_working_area": value["left_working_area"]}
                   for value in values if value["policy_kind"] != "resting"]
    row.update(status="ranked", candidates=values, pairs=pairs, challengers=challengers,
               agreement_fraction=float(np.mean([item["agrees"] for item in pairs])))


def _agreement_means(rows, np):
    ranked = [row for row in rows if row["agreement_fraction"] is not None]
    trials = [item for row in rows for item in row["challengers"] if not item["left_working_area"] and item["self_margin"] > 0]
    failures = [item["self_margin"] for item in trials if not item["label_improves"]]
    margin = max(failures, default=0.0)
    retained = [item for item in trials if item["self_margin"] > margin]
    return {"agreement_fraction": float(np.mean([row["agreement_fraction"] for row in ranked])) if ranked else None,
            "smallest_strict_margin_with_observed_agreement": margin if retained else None,
            "supporting_comparisons": len(retained), "episode_count": len(rows),
            "ranked_episode_count": len(ranked), "unscorable_episode_count": len(rows) - len(ranked),
            "interpretation": "Finite observed sample only; null when no comparisons remain. Freeze development choice before validation."}


def _oracle_means(rows, np):
    scored = [row for row in rows if "substitutions" in row]
    summary = _score_means(rows, np)
    gains = {name: [item["gain"]["add_auc"] for row in scored for item in row["substitutions"]
                    if item["substitution"] == name and item["gain"]["add_auc"] is not None]
             for name in SUBSTITUTIONS}
    summary.update(auc_gain_by_substitution={name: float(np.mean(values)) if values else None for name, values in gains.items()},
                   auc_gain_episode_counts={name: len(values) for name, values in gains.items()})
    return summary


def diagnose_advanced(operation, request):
    split, validation = _authorize(operation, request)
    parents, candidates, self_scores, mesh_baselines = _parents(operation, request)
    all_parents = (list(parents.values()) + list(mesh_baselines.values()) +
                   [parent for group in candidates.values() for parent in group] + list(self_scores.values()))
    provenance = "public_asset_diagnostic" if operation == "mesh" else "public_label_diagnostic"
    inputs = [FEATURE / "runtime.json", "configs/splits/public-v1.json", "manifests/dataset/episodes.json",
              "reports/data-audit/videos.json", "reports/data-audit/trajectories.json"]
    if request.frozen_settings:
        inputs.append(request.frozen_settings)
    record = run_fields(ROOT, "public_diagnostics", operation, asdict(request), parents=[path for path, _ in all_parents],
                        inputs=inputs, provenance=provenance)
    manifest = _json(KIT.parent / "KIT_MANIFEST.json")
    for name in ("object_pose", "vertices", "mesh_common", "object_mesh"):
        entry = input_file(ROOT, KIT / (name + ".py"))
        if entry["sha256"] != manifest["files"]["v2dlb/" + name + ".py"]:
            raise ValueError("Kit metric differs from its manifest")
        record["sources"].append(entry)
    report = {"operation": operation, "provenance": record["provenance"], "episodes": [], "means": {}}
    with _diagnostic_run(operation, record, report) as directory:
        import numpy as np
        import trimesh
        if operation != "mesh":
            import pyarrow.parquet as pq
        metric, sampler = _kit_module("object_pose"), _kit_module("vertices")
        source = _json("configs/sources.lock.json")["dataset"]
        snapshot = project_path(ROOT, source["cache_path"]) / "snapshots" / source["revision"]
        inventory, clocks = _json("manifests/dataset/episodes.json"), _json("reports/data-audit/videos.json")
        if any(value != source["revision"] for value in (inventory["revision"], clocks["revision"], split["dataset_revision"])) or any(
                parent[1].get("dataset_revision") != source["revision"] for parent in all_parents):
            raise ValueError("Diagnostic dataset revisions differ")
        record["dataset_revision"] = source["revision"]
        _claim_validation(operation, validation, record, request)
        mesh_metric = _mesh_module() if operation == "mesh" else None
        for episode in request.episodes:
            row = {"episode_index": episode, "subset": "validation" if episode in validation else "development"}
            if operation == "mesh":
                public = _json("reports/data-audit/trajectories.json")
                inventory_rows = [item for item in public["episodes"] if item["episode_index"] == episode]
                if public["revision"] != source["revision"] or len(inventory_rows) != 1:
                    raise ValueError("Public mesh identity inventory differs")
                names = inventory_rows[0]["objects"]
                row.update(compare_meshes(parents[episode], mesh_baselines.get(episode), episode, names,
                                          snapshot, record, mesh_metric, sampler, np, trimesh))
            else:
                reference, names, task = _reference(snapshot, episode, inventory, record, np, pq)
                row["task"] = task
                clock = _clock_row(clocks, episode, len(reference), "Public label and video clocks differ")
                vertices, scan_coverage = _reference_vertices(snapshot, names, record, sampler, np)
                row.update(scan_coverage)
                if operation == "oracle":
                    achieved, valid, filled = _prediction(parents[episode], episode, names, len(reference), clock["pts_integer"], record, np, pq)
                    row.update(_oracles(achieved, reference, valid, vertices, metric, request, np))
                    if filled is not None:
                        row["filled_share_per_object"] = filled.mean(axis=0).tolist()
                else:
                    _agreement(row, candidates[episode], self_scores, reference, names, clock["pts_integer"],
                               vertices, metric, record, np, pq)
            report["episodes"].append(row)
            _report_files(directory, report)
        for subset in ("development", "validation"):
            rows = [row for row in report["episodes"] if row["subset"] == subset]
            if not rows:
                continue
            if operation == "agreement":
                report["means"][subset] = _agreement_means(rows, np)
            elif operation == "mesh":
                report["means"][subset] = mesh_means(rows, np)
            else:
                report["means"][subset] = _oracle_means(rows, np)
    return directory / "run.json"
