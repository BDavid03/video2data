"""Array-only placement, silhouette projection and descriptive rank concordance."""

from collections import defaultdict
from json import JSONDecodeError
import json
import re

import cv2
import numpy as np


def placement(points: np.ndarray, normal: np.ndarray, offset: float) -> dict:
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


def prepare_view(z: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray, input_valid: np.ndarray,
                 labels: np.ndarray, cluster_ids: list[int], plane: dict, settings: dict,
                 depth_valid: np.ndarray) -> dict:
    """Reconstruct target, table-relative placement and serializable geometry from cached arrays."""
    from ._geometry import _local_plane, camera_points, table_component

    z, input_valid = z.astype(np.float64), input_valid.astype(bool)
    valid = valid & depth_valid & input_valid
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
            "pose": pose, "placement": values, "mask_area_pixels": int(target.sum())}


def place_vertices(vertices: np.ndarray, pose: dict, reverse: bool = False) -> np.ndarray:
    """Preserve shape proportions, match long extent and rest the base on the plane."""
    low, high = vertices.min(0), vertices.max(0)
    local = vertices - np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    local *= pose["long_extent"] / (high[0] - low[0])
    if reverse:
        local[:, :2] *= -1
    return local @ pose["axes"] + pose["centre"]


def silhouette(vertices_camera: np.ndarray, faces: np.ndarray, intrinsics: np.ndarray, input_valid: np.ndarray) -> np.ndarray:
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


def compare(vertices: np.ndarray, faces: np.ndarray, pose: dict, intrinsics: np.ndarray,
            input_valid: np.ndarray, target: np.ndarray) -> tuple[dict, np.ndarray]:
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


def average_candidates(candidates: list[dict], results: list[dict], expected_views: dict[str, int]) -> dict:
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


def average_spoon(candidates: list[dict], views: list[dict], results: list[dict],
                  expected_views: int) -> tuple[list[dict], float | None, str | None]:
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


def rank_panel(objects: dict[str, list[dict]], gates: dict[str, float]) -> dict:
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
    qualified_tau = [r["tau_b"] for r in qualifying]
    qualified_mean = float(np.mean(qualified_tau)) if qualifying and all(v is not None for v in qualified_tau) else None
    return {"objects": rows, "mean_tau_b": mean_tau, "qualifying_median_rho": median_rho,
            "post_result_reporting": {"mean_tau_b_ge_four_variants": qualified_mean,
                "objects": len(qualifying), "used_for_gate": False,
                "undefined_reason": None if qualified_mean is not None else "Undefined qualifying-object tau-b",
                "hypothesis": "Silhouettes may miss wall or brush-depth differences; this explanation has not been "
                              "separated from mask and placement error."},
            "undefined_reasons": {"mean_tau_b": None if all_tau else "Undefined per-object tau-b",
                                  "qualifying_median_rho": None if all_rho else "Undefined qualifying-object rho"},
            "qualifying_passing": passing, "qualifying_objects": len(qualifying),
            "gates": checks, "validated": len(rows) == 10 and all(checks.values()),
            "tool_r8_vs_box_report_only": tool_transfers}


def summarize(data: dict, results: list[dict], missing: list[dict]) -> dict:
    """Assemble complete or reasoned unavailable concordance and spoon results."""
    expected = {row["object"]: row["views"] for row in data["panel"]}
    summary = rank_panel(average_candidates(data["candidates"], results, expected), data["gates"])
    spoon, mean, reason = average_spoon(data["spoon"], data["views"], results, expected["wooden_spoon"])
    complete = not (missing or data["missing_views"]) and len(results) == sum(row["trials"] for row in data["panel"])
    summary["gates"]["complete_inputs"] = complete
    if not complete:
        summary["validated"] = False
    summary.update(spoon=spoon, spoon_mean_difference=mean, spoon_undefined_reason=reason,
                   gate_thresholds=data["gates"],
                   spoon_history_reference_only=data["spoon_history"], panel=data["panel"], trials=results,
                   missing_views=data["missing_views"], missing_inputs=list(missing),
                   verdict="PANEL CONCORDANCE PASSED" if summary["validated"] else "PROXY NOT VALIDATED FOR SPOON CD-O INFERENCE",
                   interpretation="No spoon CD-O gain or regression is established; packed D1 v4 is unchanged.",
                   incomplete_panel_reason=None if complete else "Missing frozen inputs, views or completed trials",
                   limitations="Reused development objects and construction views; small n, silhouette ties, "
                               "MoGe depth/mask distortion and placement error; gates are descriptive.")
    return summary


def _format(value: float | None) -> str:
    """Render an optional scalar with three decimal places."""
    return "null" if value is None else f"{value:.3f}"


def summary_markdown(summary: dict) -> str:
    """Format concordance, gate outcomes and limitations as compact Markdown tables."""
    gates, thresholds = summary["gates"], summary["gate_thresholds"]
    tools = [r for r in summary["objects"] if r["object"] in ("mini_dust_pan", "mini_sweeper")]
    passing_tools = sum(r["rho"] is not None and r["rho"] >= thresholds["rho"] for r in tools)
    gate_rows = [
        ("Mean tau-b", _format(summary["mean_tau_b"]), f">={_format(thresholds['mean_tau_b'])}", gates["mean_tau_b"]),
        ("Qualifying median rho", _format(summary["qualifying_median_rho"]), f">={_format(thresholds['rho'])}", gates["median_rho"]),
        ("Qualifying objects passing rho", f"{summary['qualifying_passing']}/{summary['qualifying_objects']}",
         f">={thresholds['passing_objects']}", gates["qualifying_count"]),
        ("Both scanned tools", f"{passing_tools}/{len(tools)}", "all", gates["both_tools"]),
        ("Required correlations defined", "all" if gates["all_required_correlations_defined"] else "incomplete",
         "all", gates["all_required_correlations_defined"]),
        ("Inputs complete", "yes" if gates["complete_inputs"] else "no", "yes", gates["complete_inputs"])]
    gate_rows += [(r["object"] + " rho", _format(r["rho"]), f">={_format(thresholds['rho'])}",
                   r["rho"] is not None and r["rho"] >= thresholds["rho"]) for r in tools]
    lines = ["# Spoon proxy diagnostic", "", summary["verdict"], "", summary["interpretation"], "",
             "## Pre-registered gates", "", "| Gate | Value | Requirement | Result |", "| --- | ---: | ---: | --- |"]
    lines += [f"| {name} | {value} | {requirement} | {'pass' if passed else 'fail'} |"
              for name, value, requirement, passed in gate_rows]
    lines += ["", "## Per-object concordance", "",
              "| Object | Candidates | Rho | Tau-b | Proxy ties | Undefined reason |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in summary["objects"]:
        lines.append(f"| {row['object']} | {row['candidates']} | {_format(row['rho'])} | "
                     f"{_format(row['tau_b'])} | {row['proxy_ties']} | {row['undefined_reason'] or ''} |")
    lines += ["", "## R8 versus box (report only)", "",
              "| Object | R8 minus box IoU | R8 above box | Tie | Undefined reason |", "| --- | ---: | --- | --- | --- |"]
    for row in summary["tool_r8_vs_box_report_only"]:
        above = "null" if row["r8_ranked_above_box"] is None else ("yes" if row["r8_ranked_above_box"] else "no")
        tie = "null" if row["tie"] is None else ("yes" if row["tie"] else "no")
        lines.append(f"| {row['object']} | {_format(row['r8_minus_box_iou'])} | {above} | {tie} | {row['undefined_reason'] or ''} |")
    lines += ["", "## Spoon silhouettes", "", "| Spoon view | v3 IoU | v4 IoU | Difference |", "| --- | ---: | ---: | ---: |"]
    for row in summary["spoon"]:
        lines.append(f"| {row['view']} | {_format(row['v3_box']['iou'])} | {_format(row['v4_bowl']['iou'])} | "
                     f"{_format(row['v4_minus_v3_iou'])} |")
    lines += ["", f"Spoon equal-view mean difference: {_format(summary['spoon_mean_difference'])}", "",
              "Full precision, both yaws, areas, placements, rankings and overlay paths are in summary.json.", "",
              "## Post-result reporting (not pre-registered)", ""]
    post = summary["post_result_reporting"]
    lines += [f"Pre-registered mean tau-b: {_format(summary['mean_tau_b'])}. Mean over the {post['objects']} objects "
              f"with at least four variants: {_format(post['mean_tau_b_ge_four_variants'])}; this adds no gate."]
    two_variant = [r for r in summary["objects"] if r["candidates"] == 2]
    lines += ["Two-variant objects: " + ", ".join(f"{r['object']} tau-b {_format(r['tau_b'])}" for r in two_variant) +
              ". With two untied variants, tau-b can only be +1 or -1."]
    pan = next((r for r in tools if r["object"] == "mini_dust_pan"), None)
    if pan is not None and pan["rho"] == thresholds["rho"]:
        lines += [f"Dust pan passes at exactly rho = {_format(pan['rho'])}, the frozen threshold."]
    lines += ["Hypothesis: " + post["hypothesis"], "", summary["limitations"]]
    reasons = [r["reason"] for r in [*summary["missing_views"], *summary["missing_inputs"]]]
    reasons += [r for r in [*summary["undefined_reasons"].values(), summary["spoon_undefined_reason"]] if r]
    if reasons:
        lines += ["", "## Unavailable inputs or values", "", *("- " + reason for reason in reasons)]
    if not summary["validated"]:
        lines += ["", "Proxy not validated for spoon CD-O inference; proxy says nothing about spoon CD-O; v4 unchanged."]
    return "\n".join(lines) + "\n"


def truncated_tail(text: str, error: JSONDecodeError) -> bool:
    """Recognize incomplete final JSON objects without accepting malformed content."""
    text = text.rstrip()
    suffix = text[error.pos:]
    literal = error.msg == "Expecting value" and any(t.startswith(suffix) for t in ("true", "false", "null"))
    number = error.msg == "Expecting ',' delimiter" and re.search(
        r"(?:^|[:\[,]\s*)-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?)?$", text)
    return text.startswith("{") and bool(error.pos >= len(text) or error.msg.startswith("Unterminated string") or literal or number)


def overlay_image(image: np.ndarray, target: np.ndarray, projected: np.ndarray | None) -> np.ndarray:
    """Tint the target green and outline the selected silhouette in magenta."""
    image = image[:, :target.shape[1]].copy()
    image[target] = (image[target] * 0.5 + (0, 127, 0)).astype("uint8")
    if projected is not None:
        contours, _ = cv2.findContours(projected.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 0, 255), 1)
    return image


def valid_extents(extents: np.ndarray) -> bool:
    """Check finite positive extents under the x-long/y-short construction convention."""
    x, y, z = extents
    return bool(np.isfinite(extents).all() and min(x, y, z) > 0 and x >= y * (1 - 1e-6))


def dry_summary(data: dict, masks: list[dict], frozen_id: str) -> tuple[dict, str]:
    """Describe target-only preparation without candidate projections or scores."""
    summary = {"frozen_identity": frozen_id, "panel": data["panel"], "views": masks,
               "missing_views": data["missing_views"],
               "candidate_projections": 0, "proxy_values_computed": False, "correlations_computed": False}
    text = ["# Spoon proxy target-only dry run", "", "No candidate projected; no IoU or correlation computed.", "",
            "| View | Frame | Target pixels |", "| --- | ---: | ---: |"]
    text += [f"| {row['view']} | {row['frame']} | {row['mask_area_pixels']} |" for row in masks]
    text += ["", "Missing views: " + json.dumps(data["missing_views"])]
    return summary, "\n".join(text) + "\n"
