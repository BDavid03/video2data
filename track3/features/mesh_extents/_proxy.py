"""Array-only placement, silhouette projection and descriptive rank concordance."""

from collections import defaultdict

import cv2
import numpy as np


def placement(points, normal, offset):
    """Recover the measured footprint axes and bounds centre on a support plane."""
    normal = np.asarray(normal, dtype=float)
    first = np.array([1., 0., 0.]) - normal[0] * normal
    first /= np.linalg.norm(first)
    basis = np.stack((first, np.cross(normal, first)))
    uv = points @ basis.T
    axes = np.linalg.svd(uv - uv.mean(0), full_matrices=False)[2]
    bounds = np.percentile(uv @ axes.T, [2, 98], axis=0)
    order = np.argsort(-(bounds[1] - bounds[0]), kind="stable")
    long_axis = axes[order[0]] @ basis
    if long_axis[np.abs(long_axis).argmax()] < 0:
        long_axis = -long_axis
    short_axis = np.cross(normal, long_axis)
    frame = np.stack((long_axis, short_axis, normal))
    low, high = np.percentile(points @ frame[:2].T, [2, 98], axis=0)
    centre = ((low + high) / 2) @ frame[:2] - offset * normal
    return {"axes": frame, "centre": centre, "long_extent": float(high[0] - low[0]),
            "footprint_extents": high - low}


def prepare_view(z, valid, intrinsics, input_valid, labels, cluster_ids, plane, settings):
    """Reconstruct target, table-relative placement and serializable geometry from cached arrays."""
    from ._geometry import _local_plane, camera_points, table_component

    target = np.isin(labels, [i + 1 for i in cluster_ids]) & input_valid
    if not target.any():
        raise ValueError("Empty frozen target")
    points = camera_points(z, intrinsics)
    ok = valid & input_valid & np.isfinite(points).all(-1) & (z > 0)
    normal, offset = np.asarray(plane["normal"]), plane["offset_m"]
    table = table_component(points @ normal + offset, ok, settings["table_band_m"])
    _, normal, offset, local = _local_plane(points, table, target, normal, offset, settings["ring_pixels"])
    pose = placement(points[target], normal, offset)
    values = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in pose.items()}
    values.update(normal=normal.tolist(), offset_m=float(offset), local_plane=local)
    return {"target": target, "input_valid": input_valid, "intrinsics": intrinsics,
            "pose": pose, "placement": values}


def place_vertices(vertices, pose, reverse=False):
    """Preserve shape proportions, match long extent and rest the base on the plane."""
    low, high = vertices.min(0), vertices.max(0)
    local = vertices - np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    local *= pose["long_extent"] / (high[0] - low[0])
    if reverse:
        local[:, :2] *= -1
    return local @ pose["axes"] + pose["centre"]


def silhouette(vertices_camera, faces, intrinsics, input_valid):
    """Rasterize the opaque union of projected triangles on the rectified depth grid."""
    if not np.isfinite(vertices_camera).all() or np.any(vertices_camera[:, 2] <= 0):
        raise ValueError("Projection needs finite vertices strictly in front of the camera")
    xy = vertices_camera[:, :2] / vertices_camera[:, 2, None]
    xy = xy * np.array([intrinsics[0, 0], intrinsics[1, 1]]) + intrinsics[:2, 2]
    pixels = np.rint(xy * 256).astype(np.int32)
    rendered = np.zeros(input_valid.shape, np.uint8)
    # One call per triangle gives a union; fillPoly on all faces together uses parity filling.
    for triangle in pixels[faces]:
        cv2.fillPoly(rendered, [triangle], 1, shift=8)
    return rendered.astype(bool) & input_valid


def compare(vertices, faces, pose, intrinsics, input_valid, target):
    """Return both yaw measurements and the first maximum-IoU silhouette."""
    masks = [silhouette(place_vertices(vertices, pose, reverse), faces, intrinsics, input_valid)
             for reverse in (False, True)]
    rows = []
    for yaw, mask in zip((0, 180), masks):
        union = int((mask | target).sum())
        if not union:
            raise ValueError("Empty silhouette union")
        rows.append({"yaw_degrees": yaw, "iou": float((mask & target).sum() / union),
                     "projected_area_pixels": int(mask.sum())})
    best = int(rows[1]["iou"] > rows[0]["iou"])
    return {"iou": rows[best]["iou"], "selected_yaw_degrees": rows[best]["yaw_degrees"],
            "mask_area_pixels": int(target.sum()), "yaws": rows}, masks[best]


def average_candidates(candidates, results, expected_views):
    """Average complete per-object view sets equally; retain unavailable candidates as null."""
    by_candidate = defaultdict(list)
    for row in results:
        by_candidate[(row["object"], row["sha256"])].append(row)
    objects = defaultdict(list)
    for candidate in candidates:
        rows = by_candidate[(candidate["object"], candidate["sha256"])]
        expected = expected_views.get(candidate["object"], 0)
        complete = expected > 0 and len(rows) == expected
        objects[candidate["object"]].append({**candidate,
            "mean_iou": float(np.mean([r["iou"] for r in rows])) if complete else None,
            "undefined_reason": None if complete else f"Available views {len(rows)}; required {expected}"})
    return objects


def average_spoon(candidates, views, results, expected_views):
    """Compare both spoon meshes per view and average only a complete view set."""
    indexed = {(r["sha256"], r["view"]): r for r in results if r["object"] == "wooden_spoon"}
    spoon = []
    for view in (v for v in views if v["object"] == "wooden_spoon"):
        values = {c["label"]: indexed.get((c["sha256"], view["id"]),
                  {"iou": None, "undefined_reason": "No completed candidate/view trial"}) for c in candidates}
        before, after = values["v3_box"]["iou"], values["v4_bowl"]["iou"]
        complete = before is not None and after is not None
        spoon.append({"view": view["id"], **values,
                      "v4_minus_v3_iou": after - before if complete else None,
                      "undefined_reason": None if complete else "Incomplete spoon comparison"})
    complete = expected_views > 0 and len(spoon) == expected_views and all(r["v4_minus_v3_iou"] is not None for r in spoon)
    mean = float(np.mean([r["v4_minus_v3_iou"] for r in spoon])) if complete else None
    return spoon, mean, None if complete else "No complete frozen spoon view set"


def rank_panel(objects, gates):
    """Summarize within-object concordance without treating pairs as independent samples."""
    from scipy.stats import kendalltau, spearmanr

    rows = []
    tool_transfers = []
    for name, candidates in sorted(objects.items()):
        quality = np.array([-c["cd_o_cm"] for c in candidates])
        proxy = np.array([np.nan if c["mean_iou"] is None else c["mean_iou"] for c in candidates])
        defined = np.isfinite(proxy).all() and np.isfinite(quality).all() and np.ptp(proxy) > 0 and np.ptp(quality) > 0
        rho = float(spearmanr(proxy, quality).statistic) if defined else None
        tau = float(kendalltau(proxy, quality, variant="b").statistic) if defined else None
        reason = None if defined else "Missing candidate means or constant proxy/CD-O ranking"
        rows.append({"object": name, "candidates": len(candidates), "rho": rho, "tau_b": tau,
                     "undefined_reason": reason,
                     "proxy_ties": int(np.isfinite(proxy).sum() - len(np.unique(proxy[np.isfinite(proxy)]))),
                     "cd_o_ties": len(quality) - len(np.unique(quality)),
                     "within_object_pairs": len(proxy) * (len(proxy) - 1) // 2,
                     "ranking": sorted(candidates, key=lambda c: (c["mean_iou"] is None,
                                       -(c["mean_iou"] or 0), c["sha256"]))})
        if name in ("mini_dust_pan", "mini_sweeper"):
            box = next((c for c in candidates if c["shape"] == "box"), None)
            r8 = next((c for c in candidates if any(a["rule"].startswith("R8 ") for a in c["aliases"])), None)
            available = box is not None and r8 is not None and box["mean_iou"] is not None and r8["mean_iou"] is not None
            difference = r8["mean_iou"] - box["mean_iou"] if available else None
            tool_transfers.append({"object": name, "r8_minus_box_iou": difference,
                                   "r8_ranked_above_box": difference > 0 if available else None,
                                   "tie": difference == 0 if available else None,
                                   "undefined_reason": None if available else "Box or R8 view means unavailable"})
    qualifying = [r for r in rows if r["candidates"] >= 4]
    all_tau = bool(rows) and all(r["tau_b"] is not None for r in rows)
    all_rho = bool(qualifying) and all(r["rho"] is not None for r in qualifying)
    mean_tau = float(np.mean([r["tau_b"] for r in rows])) if all_tau else None
    median_rho = float(np.median([r["rho"] for r in qualifying])) if all_rho else None
    passing = sum(r["rho"] is not None and r["rho"] >= gates["rho"] for r in qualifying)
    tools = [r for r in rows if r["object"] in ("mini_dust_pan", "mini_sweeper")]
    checks = {"all_required_correlations_defined": all_tau and all_rho,
              "mean_tau_b": mean_tau is not None and mean_tau >= gates["mean_tau_b"],
              "median_rho": median_rho is not None and median_rho >= gates["rho"],
              "qualifying_count": len(qualifying) == 8 and passing >= gates["passing_objects"],
              "both_tools": len(tools) == 2 and all(r["rho"] is not None and r["rho"] >= gates["rho"] for r in tools)}
    return {"objects": rows, "mean_tau_b": mean_tau, "qualifying_median_rho": median_rho,
            "undefined_reasons": {"mean_tau_b": None if all_tau else "Undefined per-object tau-b",
                                  "qualifying_median_rho": None if all_rho else "Undefined qualifying-object rho"},
            "qualifying_passing": passing, "qualifying_objects": len(qualifying),
            "gates": checks, "validated": len(rows) == 10 and all(checks.values()),
            "tool_r8_vs_box_report_only": tool_transfers}
