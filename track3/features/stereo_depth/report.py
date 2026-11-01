"""Independent sparse fixed-scale geometry and matched-support candidate reports."""

from dataclasses import dataclass
import math
import time

from track3.provenance import input_file, sha256_file, write_json
from ._common import (ROOT, StereoDepthRequest, artifact, authorize, frame_file, is_rigid,
                      local_path, operation_run, producer, read_json, same_revision, select_episode, selected_indices)
from .support import PARTITION, reserved

STRATA = {0: "unknown", 1: "static", 2: "hands", 3: "objects"}
UNITS = {"log_depth": "log_ratio", "depth": "m", "relative_depth": "ratio", "reprojection": "px"}


def _depth_producer(value, request):
    parent = producer(value, None, request)
    if parent[1]["kind"] not in ("depth.generate", "depth.ingest_challenger", "scratch_depth.research_predict"):
        raise ValueError("Expected a calibrated depth or admitted offline challenger producer")
    if parent[1]["kind"] == "depth.ingest_challenger" and parent[1].get("depth_kind") not in (
            "metric_optical_z_m", "relative_optical_z"):
        raise ValueError("Undeclared challenger depth units")
    if parent[1]["kind"] == "scratch_depth.research_predict" and parent[1].get("depth_kind") != "metric_optical_z_estimate":
        raise ValueError("Research predictor must explicitly declare metric optical-z estimates")
    if parent[1]["kind"] == "scratch_depth.research_predict" and request.episode_index not in (39, 42):
        raise PermissionError("Research predictor comparison retains the producer's tuning-only admission")
    return parent


@dataclass(frozen=True)
class GeometryReportRequest(StereoDepthRequest):
    max_image_side: int = 384
    correspondences_run: str | None = None
    baseline_run: str | None = None
    regions_run: str | None = None
    motion_run: str | None = None
    frozen_protocol: str | None = None
    temporal_relative_tolerance: float = .03
    maximum_wall_seconds: float = 1200.

    def __post_init__(self):
        super().__post_init__()
        if (self.source_split != "public" or self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)
                or self.stop_frame is None or len(selected_indices(self, self.stop_frame)) > 32):
            raise ValueError("Geometry reports require at most 32 explicit train/tuning frames")
        if self.max_image_side > 384:
            raise ValueError("Geometry report resolution ceiling is 384")
        if (not math.isfinite(self.temporal_relative_tolerance) or not 0 < self.temporal_relative_tolerance <= .1
                or not math.isfinite(self.maximum_wall_seconds) or not 0 < self.maximum_wall_seconds <= 1200):
            raise ValueError("Invalid geometry support tolerance or wall budget")


def _summary(values, eligible, np, units):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite admitted residual")
    return {"units": units, "numerator": float(np.sum(abs(values))), "denominator": len(values),
            "eligible": int(eligible), "coverage": len(values) / eligible if eligible else None,
            "mean_absolute": float(np.mean(abs(values))) if len(values) else None,
            "median_absolute": float(np.median(abs(values))) if len(values) else None,
            "p90_absolute": float(np.quantile(abs(values), .9)) if len(values) else None,
            "maximum_absolute": float(np.max(abs(values))) if len(values) else None,
            "reason": None if len(values) else "no_admitted_support"}


def _load_depth(path, source, index, pts, grid, record, np):
    if source["preprocessing_artifact_sha256"] != grid:
        raise ValueError("Candidate grids differ")
    file = frame_file(path, source, index, pts, record)
    with np.load(file, allow_pickle=False) as arrays:
        key = "depth_relative" if source.get("depth_kind") == "relative_optical_z" else "depth_m"
        depth = arrays[key].astype(np.float64)
        valid = arrays["valid"].astype(bool)
        k = arrays["intrinsics_pixels" if source["kind"] == "depth.ingest_challenger" else "input_intrinsics_pixels"].copy()
    if depth.ndim != 2 or valid.shape != depth.shape or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Invalid depth artifact")
    return depth, valid & np.isfinite(depth) & (depth > 0), k


def _project(points, projection, np):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ projection.T
    positive = homogeneous[:, 2] > 0
    uv = homogeneous[:, :2] / np.where(positive[:, None], homogeneous[:, 2:3], np.nan)
    return uv, positive & np.isfinite(uv).all(axis=1)


def _regions(parent, index, pts, shape, grid, record, np):
    if parent is None:
        return np.zeros(shape, dtype=np.uint8)
    if parent[1].get("grid") != grid:
        raise ValueError("Regions must use the same calibrated grid")
    file = frame_file(*parent, index, pts, record)
    with np.load(file, allow_pickle=False) as arrays:
        if parent[1]["kind"] == "object_masks.generate":
            layers = arrays["masks"].astype(bool)
            names = parent[1]["instances"]
            if layers.shape != (len(names), *shape):
                raise ValueError("Mask strata dimensions disagree")
            hands = np.zeros(shape, dtype=bool)
            objects = np.zeros(shape, dtype=bool)
            for name, layer in zip(names, layers):
                if name in ("hand_0", "hand_1"):
                    hands |= layer
                elif name in parent[1]["object_names"]:
                    objects |= layer
            labels = np.zeros(shape, dtype=np.uint8)
            labels[objects & ~hands] = 3
            labels[hands & ~objects] = 2
            # Overlaps and the complement remain unknown, never inferred static.
        elif parent[1].get("region_contract") == "geometry-regions-v1":
            labels = arrays["region_labels"].copy()
        else:
            raise ValueError("Regions require admitted masks or explicit geometry-regions-v1")
    if labels.shape != shape or labels.dtype.kind not in "iu" or not np.isin(labels, tuple(STRATA)).all():
        raise ValueError("Region labels must be integer unknown/static/hands/objects")
    return labels


def _temporal(first, second, transform, tolerance, np):
    """Forward z-buffer plus reverse depth visibility on admitted rigid support."""
    depth, valid, k = first
    target, target_valid, target_k = second
    v, u = np.nonzero(valid)
    rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(k).T
    points = rays * depth[v, u, None]
    moved = points @ transform[:3, :3].T + transform[:3, 3]
    projected = moved @ target_k.T
    uv = projected[:, :2] / np.where(projected[:, 2:3] > 0, projected[:, 2:3], np.nan)
    height, width = target.shape
    finite = np.isfinite(uv).all(axis=1) & (moved[:, 2] > 0)
    pixels = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(np.int64)
    inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    candidates = np.flatnonzero(inside)
    front = np.full(height * width, np.inf)
    bins = pixels[candidates, 1] * width + pixels[candidates, 0]
    np.minimum.at(front, bins, moved[candidates, 2])
    candidates = candidates[moved[candidates, 2] <= front[bins] + 1e-8]
    pu, pv = pixels[candidates].T
    observed = target_valid[pv, pu]
    candidates = candidates[observed]
    pu, pv = pixels[candidates].T
    target_z = target[pv, pu]
    predicted_z = moved[candidates, 2]
    # Front/back depth agreement is an admission threshold, not a claim about excluded errors.
    forward_visible = abs(target_z - predicted_z) <= tolerance * np.maximum(target_z, predicted_z)
    rays_back = np.column_stack((pu, pv, np.ones(len(pu)))) @ np.linalg.inv(target_k).T
    back = rays_back * target_z[:, None]
    inverse = np.linalg.inv(transform)
    back = back @ inverse[:3, :3].T + inverse[:3, 3]
    projected_back = back @ k.T
    back_uv = projected_back[:, :2] / np.where(projected_back[:, 2:3] > 0, projected_back[:, 2:3], np.nan)
    backward_visible = (np.isfinite(back_uv).all(axis=1) & (back[:, 2] > 0)
                        & (np.linalg.norm(back_uv - np.column_stack((u[candidates], v[candidates])), axis=1) <= 1.5)
                        & (abs(back[:, 2] - depth[v[candidates], u[candidates]])
                           <= tolerance * depth[v[candidates], u[candidates]]))
    accepted = forward_visible & backward_visible
    return {"log_z": _summary(np.log(predicted_z[accepted] / target_z[accepted]), len(v), np, "log_ratio"),
            "distance": _summary(predicted_z[accepted] - target_z[accepted], len(v), np, "m"),
            "source_support": len(v), "out_of_field_or_behind": int((~inside).sum()),
            "zbuffer_occluded": int(inside.sum() - len(candidates) - (~observed).sum()),
            "target_missing": int((~observed).sum()),
            "visibility_rejected": int((~accepted).sum()),
            "note": "Conditional on bidirectional depth tolerance; rejected pixels remain visible in counts."}


def _protocol(request, selection, correspond):
    """Read the predeclared protocol and require it to bind the selected frames and query producer."""
    if request.frozen_protocol is None:
        raise ValueError("Supply a hash-bound predeclared protocol; report never selects scale from outcomes")
    protocol = read_json(request.frozen_protocol)
    if (protocol.get("schema") != "independent-geometry-protocol-v1" or not protocol.get("frozen_utc")
            or protocol.get("holdback_partition") != PARTITION or protocol.get("holdback_modulus") != 5):
        raise ValueError("Missing predeclared geometry protocol and fitting-support holdback")
    if (protocol.get("episode_index") != request.episode_index
            or protocol.get("frame_indices") != selected_indices(request, selection["stop"])
            or protocol.get("correspondences_run_sha256") != sha256_file(correspond[0])):
        raise ValueError("Protocol must bind the complete selected frame/query producer universe")
    return protocol


def _fixed_scale(name, parent, protocol, grid, request, record):
    """The protocol's fixed metric scale for one candidate, checking a declared disjoint anchor."""
    declared = protocol.get(name, {})
    if declared.get("run_sha256") != sha256_file(parent[0]):
        raise ValueError("Protocol does not bind this candidate producer")
    scale = declared.get("fixed_metric_scale")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Each candidate needs one fixed positive metric scale")
    if declared.get("scale_origin") not in ("native_metric", "reserved_disjoint_anchor"):
        raise ValueError("Scale provenance must be explicit")
    if declared["scale_origin"] == "native_metric" and scale != 1:
        raise ValueError("Native metric values must retain unit scale; scale corrections need a disjoint anchor")
    if parent[1].get("depth_kind") == "relative_optical_z" and declared["scale_origin"] != "reserved_disjoint_anchor":
        raise ValueError("Relative challenger depth requires a frozen disjoint metric anchor")
    if declared["scale_origin"] == "reserved_disjoint_anchor":
        anchor_parent = producer(declared.get("anchor_run"), "stereo_depth.anchor", request)
        if declared.get("anchor_run_sha256") != sha256_file(anchor_parent[0]):
            raise ValueError("Frozen protocol must also bind the scale-anchor producer hash")
        anchor_document = read_json(artifact(anchor_parent[0], "anchor.json",
                                    record, anchor_parent[1]["artifacts"]["anchor.json"]))
        if (anchor_document.get("grid") != grid or anchor_document.get("holdback_partition") != PARTITION
                or anchor_document.get("holdback_modulus") != 5
                or local_path(anchor_parent[1]["request"]["depth_run"]) != parent[0]
                or not math.isclose(anchor_document["episode_scale"], scale, rel_tol=1e-12)):
            raise ValueError("Scale is not the declared disjoint fixed episode anchor")
        record["inputs"].append(input_file(ROOT, anchor_parent[0]))
        # Preserve the anchor's complete inherited producer lineage as an explicit parent.
        record.setdefault("scale_parents", []).append(input_file(ROOT, anchor_parent[0]))
    return float(scale)


def _queries(correspond, index, pts, record, np):
    with np.load(frame_file(*correspond, index, pts, record), allow_pickle=False) as arrays:
        queries = {key: arrays[key].copy() for key in arrays.files}
    ids, uv = queries["query_ids"], queries["rgb_uv"]
    if (ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or uv.shape != (len(ids), 2)
            or uv.dtype.kind not in "iu" or not np.isfinite(queries["stereo_depth_m"]).all()
            or np.any(queries["stereo_depth_m"] <= 0)):
        raise ValueError("Malformed independent query identities/support")
    return queries


def _residuals(depth, valid, k, uv, held, queries, np):
    """Per-query admission, residuals and predicted stereo endpoints for one scaled candidate."""
    u, v = uv.T
    z = depth[v, u]
    rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(k).T
    errors, visible, endpoints = [], np.ones(len(u), dtype=bool), {}
    for side in ("left", "right"):
        predicted, positive = _project(rays * z[:, None], queries[f"{side}_projection"], np)
        endpoints[side] = predicted
        errors.append(np.linalg.norm(predicted - queries[f"{side}_uv"], axis=1))
        visible &= positive
    with np.errstate(invalid="ignore", divide="ignore"):
        residuals = {"log_depth": np.log(z / queries["stereo_depth_m"]),
                     "depth": z - queries["stereo_depth_m"],
                     "relative_depth": z / queries["stereo_depth_m"] - 1,
                     "reprojection": np.maximum(*errors)}
    return held & valid[v, u] & visible, residuals, endpoints


def _strata(support, residuals, held, sampled_labels, np):
    metrics = {}
    for label, title in STRATA.items():
        eligible = held & (sampled_labels == label)
        accepted = support & eligible
        metrics[title] = {key: _summary(values[accepted], eligible.sum(), np, UNITS[key])
                          for key, values in residuals.items()}
        metrics[title]["observed_log_stereo_over_candidate_scale"] = (
            float(np.median(-residuals["log_depth"][accepted])) if accepted.any() else None)
    return metrics


def _paired(support, residuals, held, np):
    intersection = support["candidate"] & support["baseline"]
    union = support["candidate"] | support["baseline"]
    return {"intersection": int(intersection.sum()), "union": int(union.sum()),
            "both_missing": int((held & ~union).sum()),
            "candidate_only": int((support["candidate"] & ~support["baseline"]).sum()),
            "baseline_only": int((support["baseline"] & ~support["candidate"]).sum()),
            "mean_absolute_error_delta_on_intersection": {
                key: float(np.mean(abs(residuals["candidate"][key][intersection])
                                    - abs(residuals["baseline"][key][intersection]))) if intersection.any() else None
                for key in residuals["candidate"]}}


def _query_rows(request, index, pts, queries, held, sampled_labels, support, residuals, endpoints, np):
    rows = []
    for j in np.flatnonzero(held):
        observed = bool(support[j])
        rows.append({"episode_index": request.episode_index, "frame_index": index,
                     "video_pts": pts, "query_id": str(queries["query_ids"][j]), "object_id": None,
                     "event_id": None, "event_stratum": None, "stratum": STRATA[int(sampled_labels[j])],
                     "observed": observed,
                     "depth_log_residual": float(residuals["log_depth"][j]) if observed else None,
                     "depth_error_m": float(residuals["depth"][j]) if observed else None,
                     "reprojection_error_px": float(residuals["reprojection"][j]) if observed else None,
                     "rgb_uv": queries["rgb_uv"][j].tolist(),
                     "left_observed_uv": queries["left_uv"][j].tolist(),
                     "right_observed_uv": queries["right_uv"][j].tolist(),
                     "left_predicted_uv": endpoints["left"][j].tolist() if observed else None,
                     "right_predicted_uv": endpoints["right"][j].tolist() if observed else None,
                     "temporal_residual_m": None, "rigidity_drift_m": None,
                     "catastrophic_identity_loss": None, "new_invalid_event": None})
    return rows


def _motion_transform(motion, trajectory, previous, index, pts, grid, record, np):
    """The exact observed previous-to-current camera transform; never interpolated."""
    if trajectory is not None:
        previous_index, previous_pts = previous[:2]
        for checked_index, checked_pts in ((index, pts), (previous_index, previous_pts)):
            if (trajectory["frame_index"][checked_index] != checked_index
                    or trajectory["pts"][checked_index] != checked_pts
                    or not trajectory["valid"][checked_index]
                    or not trajectory["observed"][checked_index]):
                raise ValueError("Temporal comparison needs exact observed camera poses, no interpolation")
        transform = np.linalg.inv(trajectory["world_T_camA"][index]) @ trajectory["world_T_camA"][previous_index]
    else:
        if motion[1].get("motion_contract") != "calibrated-camera-motion-v1" or motion[1].get("grid") != grid:
            raise ValueError("Motion producer needs an explicit calibrated-camera-motion-v1 contract")
        with np.load(frame_file(*motion, index, pts, record), allow_pickle=False) as arrays:
            transform = arrays["current_from_previous"].copy()
            previous_index, previous_pts = int(arrays["previous_frame_index"]), int(arrays["previous_pts"])
    if previous_index != previous[0] or previous_pts != previous[1] or not is_rigid(transform, np):
        raise ValueError("Temporal transform is not the admitted consecutive calibrated SE(3)")
    return transform, previous_pts


def _scale_drift(scales, frames, np):
    drift = {}
    for name in scales:
        drift[name] = {}
        for title in STRATA.values():
            observed = [(row["frame_index"], row["metrics"][name][title]["observed_log_stereo_over_candidate_scale"])
                        for row in frames]
            retained = [(index, value) for index, value in observed if value is not None]
            values = np.array([value for _, value in retained])
            drift[name][title] = {"frames_requested": len(observed), "frames_supported": len(retained),
                                  "missing_frame_indices": [index for index, value in observed if value is None],
                                  "fixed_origin_frame": retained[0][0] if retained else None,
                                  "fixed_origin_log_scale": retained[0][1] if retained else None,
                                  "median_log_scale": float(np.median(values)) if retained else None,
                                  "p90_absolute_deviation": float(np.quantile(abs(values - np.median(values)), .9)) if retained else None,
                                  "first_to_last_log_scale_change": float(values[-1] - values[0]) if retained else None,
                                  "note": "Diagnostic observed scale ratios only; candidate depth is never normalized by these values."}
    return drift


def report(request):
    authorize("report", request)
    selection = select_episode(request)
    candidate = _depth_producer(request.depth_run, request)
    correspond = producer(request.correspondences_run, "stereo_depth.correspondences", request)
    baseline = _depth_producer(request.baseline_run, request) if request.baseline_run else None
    regions = producer(request.regions_run, None, request) if request.regions_run else None
    motion = producer(request.motion_run, None, request) if request.motion_run else None
    present = [parent for parent in (candidate, correspond, baseline, regions, motion) if parent]
    protocol = _protocol(request, selection, correspond)
    with operation_run("report", request, selection, [parent[0] for parent in present]) as (directory, record):
        import numpy as np

        started = time.monotonic()
        record["inputs"].append(input_file(ROOT, request.frozen_protocol))
        same_revision(present, selection["revision"])
        if (correspond[1].get("evidence_contract") != "independent-image-correspondences-v1"
                or correspond[1].get("depth_values_used") is not False):
            raise ValueError("Only independently produced image queries are admitted")
        grid = candidate[1]["preprocessing_artifact_sha256"]
        if correspond[1]["grid"] != grid:
            raise ValueError("Correspondence and candidate grids differ")
        scales = {name: _fixed_scale(name, parent, protocol, grid, request, record)
                  for name, parent in (("candidate", candidate), ("baseline", baseline)) if parent is not None}
        trajectory = None
        if motion is not None and motion[1]["kind"] == "camera_motion.estimate":
            if motion[1].get("grid") != grid:
                raise ValueError("Camera motion uses a different calibrated grid")
            path = artifact(motion[0], "trajectory.npz", record, motion[1]["artifacts"]["trajectory.npz"])
            with np.load(path, allow_pickle=False) as arrays:
                trajectory = {key: arrays[key].copy() for key in arrays.files}
        previous = None
        for index in selected_indices(request, selection["stop"]):
            if time.monotonic() - started > request.maximum_wall_seconds:
                raise TimeoutError("Declared geometry report budget exhausted")
            pts = selection["clocks"]["ego_cam_a"]["pts_integer"][index]
            row = {"frame_index": index, "video_pts": pts, "time_base": [1, 10240],
                   "episode_index": request.episode_index, "event_id": None, "status": "running"}
            record["frames"].append(row)
            queries = _queries(correspond, index, pts, record, np)
            ids, uv = queries["query_ids"], queries["rgb_uv"]
            candidate_arrays = _load_depth(*candidate, index, pts, grid, record, np)
            shape = candidate_arrays[0].shape
            if max(shape) > request.max_image_side:
                raise ValueError("Candidate exceeds declared report resolution")
            if not np.allclose(candidate_arrays[2], queries["input_intrinsics_pixels"], atol=1e-5):
                raise ValueError("Correspondence and depth intrinsics differ")
            if (np.any(uv < 0) or np.any(uv[:, 0] >= shape[1]) or np.any(uv[:, 1] >= shape[0])
                    or any(identity != f"a:{v}:{u}" for identity, (u, v) in zip(ids, uv))):
                raise ValueError("Query identity does not match its admitted colour-grid pixel")
            held = np.array([reserved(request.episode_index, index, str(identity)) for identity in ids], dtype=bool)
            labels = _regions(regions, index, pts, shape, grid, record, np)
            sampled_labels = labels[uv[:, 1], uv[:, 0]]
            arrays_by_name = {"candidate": candidate_arrays}
            if baseline:
                arrays_by_name["baseline"] = _load_depth(*baseline, index, pts, grid, record, np)
                if (arrays_by_name["baseline"][0].shape != shape
                        or not np.allclose(arrays_by_name["baseline"][2], candidate_arrays[2], atol=1e-5)):
                    raise ValueError("Paired candidates must use identical dimensions and intrinsics")
            support, residuals, endpoints = {}, {}, {}
            for name, (depth, valid, k) in arrays_by_name.items():
                depth *= scales[name]  # in place: the temporal check below also uses scaled depth
                support[name], residuals[name], endpoints[name] = _residuals(depth, valid, k, uv, held, queries, np)
            metrics = {name: _strata(support[name], residuals[name], held, sampled_labels, np)
                       for name in arrays_by_name}
            paired = _paired(support, residuals, held, np) if baseline else None
            query_rows = _query_rows(request, index, pts, queries, held, sampled_labels, support["candidate"],
                                     residuals["candidate"], endpoints["candidate"], np)
            temporal = {"available": False, "reason": "missing_admitted_motion_or_static_regions_or_previous_frame"}
            if previous is not None and index != previous[0] + 1:
                temporal["reason"] = "sparse_or_nonconsecutive_frames_cannot_establish_temporal_consistency"
            if previous is not None and index == previous[0] + 1 and motion is not None and regions is not None:
                transform, previous_pts = _motion_transform(motion, trajectory, previous, index, pts, grid, record, np)
                current = arrays_by_name["candidate"]
                first = (previous[2][0], previous[2][1] & (previous[3] == 1), previous[2][2])
                second = (current[0], current[1] & (labels == 1), current[2])
                temporal = _temporal(first, second, transform, request.temporal_relative_tolerance, np)
                temporal.update(available=True, delta_pts=pts - previous_pts,
                                delta_seconds=(pts - previous_pts) / 10240,
                                excluded_dynamic_or_unknown=int((labels != 1).sum()),
                                dynamic_objects_reason="accepted_object_motion_transforms_not_supplied")
            previous = (index, pts, arrays_by_name["candidate"], labels)
            row.update(status="completed", reserved_queries=int(held.sum()),
                       fit_queries_excluded=int((~held).sum()), producer_support=next(
                           entry for entry in correspond[1]["frames"] if entry["frame_index"] == index),
                       metrics=metrics, paired=paired, queries=query_rows, temporal=temporal,
                       vertical_residual=_summary((queries["left_uv"][:, 1] - queries["right_uv"][:, 1])[held], held.sum(), np, "px"))
            write_json(directory / "run.json", record)
        document = {"schema": "independent-geometry-report-v1", "channel": "independent_image_geometry",
                    "dataset_revision": selection["revision"], "episode_index": request.episode_index,
                    "split": request.source_split, "grid": grid, "fixed_metric_scales": scales,
                    "holdback_partition": PARTITION, "protocol": input_file(ROOT, request.frozen_protocol),
                    "sources": record["sources"], "inputs": record["inputs"], "frames": record["frames"],
                    "scale_drift": _scale_drift(scales, record["frames"], np),
                    "limitations": ["Sparse reciprocal matching approximates visibility; textureless/specular coverage remains missing.",
                                    "No diagnostic labels, teacher agreement, achieved behaviour, full surfaces or unused views are assessed.",
                                    "No object identities/events are invented; dynamic temporal comparison awaits accepted object motion.",
                                    "No ranking, adoption or new holdout assessment occurs."]}
        write_json(directory / "geometry.json", document)
    return directory / "run.json"
