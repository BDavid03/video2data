"""Array-only placement, silhouette projection and descriptive rank concordance."""

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


def rank_panel(objects, gates):
    """Summarize within-object concordance without treating pairs as independent samples."""
    from scipy.stats import kendalltau, spearmanr

    rows = []
    tool_transfers = []
    for name, candidates in sorted(objects.items()):
        quality = np.array([-c["cd_o_cm"] for c in candidates])
        proxy = np.array([np.nan if c["mean_iou"] is None else c["mean_iou"] for c in candidates])
        defined = np.isfinite(proxy).all() and np.ptp(proxy) > 0 and np.ptp(quality) > 0
        rho = float(spearmanr(proxy, quality).statistic) if defined else None
        tau = float(kendalltau(proxy, quality, variant="b").statistic) if defined else None
        rows.append({"object": name, "candidates": len(candidates), "rho": rho, "tau_b": tau,
                     "proxy_ties": int(np.isfinite(proxy).sum() - len(np.unique(proxy[np.isfinite(proxy)]))),
                     "cd_o_ties": len(quality) - len(np.unique(quality)),
                     "within_object_pairs": len(proxy) * (len(proxy) - 1) // 2,
                     "ranking": sorted(candidates, key=lambda c: (c["mean_iou"] is None,
                                       -(c["mean_iou"] or 0), c["sha256"]))})
        if name in ("mini_dust_pan", "mini_sweeper"):
            box = next(c for c in candidates if c["shape"] == "box")
            r8 = next(c for c in candidates if any(a["rule"].startswith("R8 ") for a in c["aliases"]))
            difference = r8["mean_iou"] - box["mean_iou"]
            tool_transfers.append({"object": name, "r8_minus_box_iou": difference,
                                   "r8_ranked_above_box": difference > 0, "tie": difference == 0})
    qualifying = [r for r in rows if r["candidates"] >= 4]
    all_tau = all(r["tau_b"] is not None for r in rows)
    all_rho = all(r["rho"] is not None for r in qualifying)
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
            "qualifying_passing": passing, "qualifying_objects": len(qualifying),
            "gates": checks, "validated": len(rows) == 10 and all(checks.values()),
            "tool_r8_vs_box_report_only": tool_transfers}
