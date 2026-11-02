"""Pure selected-yaw replay and descriptive within-object projection comparisons."""

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau

from ._proxy import place_vertices, silhouette

if TYPE_CHECKING:
    from matplotlib.figure import Figure

CONDITION = ("All signatures remain conditional on recorded views, normalization and selected yaw; "
             "hidden geometry versus mask/placement error remains unproven.")


def replay(vertices: np.ndarray, faces: np.ndarray, placement: dict, yaw: int,
           intrinsics: np.ndarray, valid: np.ndarray, labels: np.ndarray, cluster_ids: list[int]) -> tuple[dict, np.ndarray]:
    """Render the recorded pose and selected yaw without reconstructing or searching placement."""
    valid = valid.astype(bool)
    pose = {**placement, "axes": np.asarray(placement["axes"]), "centre": np.asarray(placement["centre"])}
    target = np.isin(labels, [i + 1 for i in cluster_ids]) & valid
    projected = silhouette(place_vertices(vertices, pose, yaw == 180), faces, intrinsics, valid)
    union = int((target | projected).sum())
    extents = np.ptp(vertices, axis=0)
    return {"iou": float((target & projected).sum() / union),
            "projected_area_pixels": int(projected.sum()), "mask_area_pixels": int(target.sum()),
            "original_extents_xyz": extents.tolist(), "applied_scale": float(pose["long_extent"] / extents[0])}, projected


def require_replay(expected: dict, actual: dict) -> None:
    """Refuse any difference in parsed IoU or selected projected and target areas."""
    selected = next(y for y in expected["yaws"] if y["yaw_degrees"] == expected["selected_yaw_degrees"])
    reference = {"iou": expected["iou"], "projected_area_pixels": selected["projected_area_pixels"],
                 "mask_area_pixels": expected["mask_area_pixels"]}
    if any(actual[key] != value for key, value in reference.items()):
        raise ValueError(f"Replay differs from the first-run ledger: {expected['identity']}")


def replay_summary(original: list[dict], actual: list[dict], new: int) -> dict:
    """Summarize exact observable equality and the number of new or resumed replays."""
    selected = [next(y for y in r["yaws"] if y["yaw_degrees"] == r["selected_yaw_degrees"]) for r in original]
    return {"trials": len(actual), "max_abs_iou_difference": max(abs(a["iou"] - b["iou"]) for a, b in zip(original, actual)),
            "projected_areas_equal": all(a["projected_area_pixels"] == b["projected_area_pixels"] for a, b in zip(selected, actual)),
            "target_areas_equal": all(a["mask_area_pixels"] == b["mask_area_pixels"] for a, b in zip(original, actual)),
            "new": new, "resumed": len(actual) - new}


def pair_distances(masks: list[np.ndarray], i: int, j: int) -> np.ndarray:
    """Measure one-minus-IoU between selected projections in every recorded view."""
    values = []
    for view in masks:
        union = int((view[i] | view[j]).sum())
        values.append(1 - int((view[i] & view[j]).sum()) / union if union else 0.)
    return np.asarray(values)


def order_counts(pairs: list[dict]) -> dict:
    """Count correct, reversed and tied orders with ties retained in concordance denominators."""
    counts = {key: sum(p["order"] == key for p in pairs) for key in ("correct", "reversed", "tied")}
    return {"eligible_pairs": len(pairs), **counts, "failures": counts["reversed"] + counts["tied"],
            "concordance": counts["correct"] / len(pairs) if pairs else None,
            "undefined_reason": None if pairs else "No eligible pairs"}


def classify(pairs: list[dict], gap: float) -> dict:
    """Apply the fixed eligibility cutoff and descriptive signature rules."""
    eligible = [p for p in pairs if p["cd_o_gap_cm"] >= gap]
    outline = [p for p in eligible if p["outline_changing"]]
    near = [p for p in eligible if not p["outline_changing"]]
    total, changed, similar = order_counts(eligible), order_counts(outline), order_counts(near)
    h = similar["failures"] / total["failures"] if total["failures"] else None
    signature = "not separated"
    if total["failures"] >= 3 and len(outline) >= 3 and h >= 2 / 3 and changed["concordance"] >= .70:
        signature = "projection-indistinguishability signature"
    elif len(outline) >= 3 and changed["concordance"] < .50:
        signature = "outline-score disagreement"
    return {"gap_cm": gap, "all": total, "outline": changed, "near": similar, "H": h,
            "H_undefined_reason": None if h is not None else "No failed eligible pairs",
            "near_projection_share": len(near) / len(eligible) if eligible else None, "signature": signature}


def analyse_object(name: str, candidates: list[dict], views: list[str], q: np.ndarray,
                   masks: list[np.ndarray]) -> tuple[dict, list[dict]]:
    """Compare every unordered candidate pair and summarize its two fixed eligibility panels."""
    means = q.mean(axis=1)
    pairs = []
    for i, j in combinations(range(len(candidates)), 2):
        a, b = candidates[i], candidates[j]
        distances = pair_distances(masks, i, j)
        signed = q[i] - q[j]
        delta = float(means[i] - means[j])
        cdelta = a["cd_o_cm"] - b["cd_o_cm"]
        outline = bool(distances.max() >= .05)
        pairs.append({"object": name, "candidate_i": a["sha256"], "candidate_j": b["sha256"],
            "label_i": a["label"], "label_j": b["label"], "cd_o_gap_cm": abs(cdelta),
            "mean_projection_distance": float(distances.mean()), "max_projection_distance": float(distances.max()),
            "outline_changing": outline, "eligible_003": abs(cdelta) >= .03, "eligible_006": abs(cdelta) >= .06,
            "mean_iou_i": float(means[i]), "mean_iou_j": float(means[j]), "signed_mean_iou_difference": delta,
            "max_abs_view_iou_gap": float(np.abs(signed).max()), "abs_mean_iou_gap": abs(delta),
            "per_view_score_compression": bool(np.abs(signed).max() <= .01), "mean_score_compression": abs(delta) <= .01,
            "order": "tied" if abs(delta) <= 1e-12 else ("correct" if delta * cdelta < 0 else "reversed"),
            "views": [{"view": view, "projection_distance": float(distance), "signed_iou_difference": float(difference)}
                      for view, distance, difference in zip(views, distances, signed)]})
    base, sensitivity = classify(pairs, .03), classify(pairs, .06)
    signature = base["signature"] if base["signature"] == sensitivity["signature"] else "not separated (gap-sensitive)"
    taus = [{"view_i": views[i], "view_j": views[j],
             "tau_b": float(kendalltau(q[:, i], q[:, j], variant="b").statistic)}
            for i, j in combinations(range(len(views)), 2)]
    defined = bool(taus) and all(np.isfinite(row["tau_b"]) for row in taus)
    t = float(np.mean([r["tau_b"] for r in taus])) if defined else None
    for row in taus:
        if not np.isfinite(row["tau_b"]):
            row["tau_b"] = None
    return {"object": name, "candidates": len(candidates), "views": len(views), "at_003": base,
            "at_006": sensitivity, "signature": signature, "B": float(q.max(axis=0).mean()), "T": t,
            "T_undefined_reason": None if defined else "Fewer than two views or an undefined view-pair tau-b",
            "view_pair_tau_b": taus}, pairs


def group_summary(objects: list[dict], names: list[str]) -> dict:
    """Average defined object fractions equally and retain their contributing-object counts."""
    rows = [r for r in objects if r["object"] in names]
    values = {"B": [r["B"] for r in rows], "T": [r["T"] for r in rows]}
    for cutoff in ("at_003", "at_006"):
        values.update({cutoff + "." + key: [r[cutoff][key] for r in rows] for key in ("H", "near_projection_share")})
        values.update({cutoff + "." + key + ".concordance": [r[cutoff][key]["concordance"] for r in rows]
                       for key in ("all", "outline", "near")})
    return {"objects": len(rows), "means": {key: {"mean": float(np.mean([v for v in items if v is not None]))
            if any(v is not None for v in items) else None, "contributing_objects": sum(v is not None for v in items)}
            for key, items in values.items()}}


def assemble(objects: list[dict], pairs: list[dict], groups: dict[str, list[str]]) -> dict:
    """Assemble conditional group signatures, complete-group warning flags and compressed-score pairs."""
    grouped = {name: group_summary(objects, names) for name, names in groups.items()}
    failed = [r["signature"] for r in objects if r["object"] in groups["failed"]]
    common = failed[0] if len(failed) == 3 and len(set(failed)) == 1 else "not separated"
    if common not in ("projection-indistinguishability signature", "outline-score disagreement"):
        common = "not separated"
    flags = {}
    for metric, threshold in (("B", -.10), ("T", -.30)):
        a, b = (grouped[g]["means"][metric] for g in ("failed", "contrast"))
        complete = all(grouped[g]["means"][metric]["contributing_objects"] == len(groups[g]) for g in groups)
        delta = a["mean"] - b["mean"] if complete else None
        flags[metric] = {"failed_minus_contrast": delta, "threshold": threshold,
                         "flag": delta <= threshold if complete else None,
                         "undefined_reason": None if complete else "Every object's value is required for this group difference"}
    compression = [p for p in pairs if p["eligible_003"] and p["outline_changing"]
                   and (p["per_view_score_compression"] or p["mean_score_compression"])]
    return {"condition": CONDITION, "amendment_3_verdict": "NOT VALIDATED", "packed_meshes": "unchanged",
            "objects": objects, "groups": grouped, "failed_group_signature": common, "B_T_flags": flags,
            "score_compression_pairs": compression}


def summary_markdown(summary: dict) -> str:
    """Format conditional object signatures, equal-object summaries and warning indicators."""
    fmt = lambda value: "null" if value is None else f"{value:.3f}"
    lines = ["# Descriptive proxy failure analysis", "", CONDITION, "",
             "Amendment 3 remains NOT VALIDATED; packed meshes remain unchanged.", "",
             "| Replay trials | Max absolute IoU difference | Projected / target areas equal | New / resumed |",
             "| ---: | ---: | --- | ---: |",
             f"| {summary['replay']['trials']} | {fmt(summary['replay']['max_abs_iou_difference'])} | yes / yes | "
             f"{summary['replay']['new']} / {summary['replay']['resumed']} |", "",
             "## Recorded-view object signatures", "",
             "| Object | Views | Candidates | Gap cm | Eligible | Outline | Near | All correct / reversed / tied | Outline correct / reversed / tied | Near correct / reversed / tied | Concordance | Outline concordance | Near concordance | H | Signature | B | T |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in summary["objects"]:
        for cutoff in ("at_003", "at_006"):
            m, a = row[cutoff], row[cutoff]["all"]
            lines.append(f"| {row['object']} | {row['views']} | {row['candidates']} | {fmt(m['gap_cm'])} | {a['eligible_pairs']} | {m['outline']['eligible_pairs']} | "
                         f"{m['near']['eligible_pairs']} | {a['correct']} / {a['reversed']} / {a['tied']} | "
                         f"{m['outline']['correct']} / {m['outline']['reversed']} / {m['outline']['tied']} | "
                         f"{m['near']['correct']} / {m['near']['reversed']} / {m['near']['tied']} | "
                         f"{fmt(a['concordance'])} | {fmt(m['outline']['concordance'])} | {fmt(m['near']['concordance'])} | {fmt(m['H'])} | "
                         f"{m['signature']} | {fmt(row['B'])} | {fmt(row['T'])} |")
    lines += ["", "Sensitivity-retained signatures: " + "; ".join(f"{r['object']}: {r['signature']}" for r in summary['objects']),
              "", "Failed-group signature: **" + summary["failed_group_signature"] + "**.", "",
              "## Equal-object summaries", "", "| Group | Fraction | Mean | Contributing / group objects |", "| --- | --- | ---: | ---: |"]
    for group, rows in summary["groups"].items():
        lines += [f"| {group} | {key} | {fmt(value['mean'])} | {value['contributing_objects']} / {rows['objects']} |"
                  for key, value in rows["means"].items()]
    lines += ["", "## Descriptive B/T flags", "", "| Indicator | Failed minus contrast | Threshold | Flag | Undefined reason |",
              "| --- | ---: | ---: | --- | --- |"]
    lines += [f"| {key} | {fmt(row['failed_minus_contrast'])} | <= {fmt(row['threshold'])} | {row['flag']} | "
              f"{row['undefined_reason'] or ''} |" for key, row in summary['B_T_flags'].items()]
    lines += ["", "## Eligible outline-changing score-compression pairs", "",
              "| Object | Candidate i | Candidate j | CD-O gap cm | Max view IoU gap | Mean IoU gap | Per-view <=0.01 | Mean <=0.01 | Eligible at 0.06 cm |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    lines += [f"| {p['object']} | {p['label_i']} | {p['label_j']} | {fmt(p['cd_o_gap_cm'])} | {fmt(p['max_abs_view_iou_gap'])} | "
              f"{fmt(p['abs_mean_iou_gap'])} | {p['per_view_score_compression']} | {p['mean_score_compression']} | {p['eligible_006']} |"
              for p in summary['score_compression_pairs']]
    lines += ["", "Zero denominators are null; reasons and view-pair tau values are in summary.json.",
              "B/T flags do not change signatures or identify mask error. Pairs and views are not independent replicates.",
              "All pair distances, signed view gaps, extents and scales are retained in pairs.json and replay artifacts.", "", CONDITION]
    lines += ["", "Figure PNG: `" + summary["figures"]["png"] + "`", "Figure PDF: `" + summary["figures"]["pdf"] + "`"]
    return "\n".join(lines) + "\n"


def plot_pairs(pairs: list[dict], cd_o: dict[tuple[str, str], float], failed: list[str]) -> "Figure":
    """Plot eligible-pair gaps against score differences oriented to make correct orders positive."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = sorted({p["object"] for p in pairs})
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=True)
    for ax, name in zip(axes.flat, names):
        rows = [p for p in pairs if p["object"] == name and p["eligible_003"]]
        color = "#b64224" if name in failed else "#176b83"
        for outline, marker in ((True, "o"), (False, "s")):
            subset = [p for p in rows if p["outline_changing"] == outline]
            x = [p["cd_o_gap_cm"] for p in subset]
            y = [-np.sign(cd_o[(name, p["candidate_i"])] - cd_o[(name, p["candidate_j"])]) *
                 p["signed_mean_iou_difference"] for p in subset]
            ax.scatter(x, y, marker=marker, s=36, edgecolors=color,
                       facecolors=color if outline else "none", alpha=.8)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_title(name.replace("_", " ") + (" [failed]" if name in failed else " [contrast]"), color=color, fontsize=10)
        ax.set_xlim(left=0)
        ax.grid(alpha=.2)
        if name in failed:
            ax.set_facecolor("#fff2ec")
    fig.suptitle("Recorded-view proxy orders: eligible CD-O gap >= 0.03 cm", y=.98)
    fig.supxlabel("Absolute recorded CD-O gap (cm)", y=.08)
    fig.supylabel("Oriented equal-view mean IoU difference (positive = correct order)")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="#555555", label="Outline-changing"),
                        Line2D([], [], marker="s", linestyle="none", markerfacecolor="none", color="#555555", label="Near-projection")],
               loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.text(.5, .025, "Conditional on recorded views, normalization and selected yaw;\nhidden geometry versus mask/placement error remains unproven.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(.035, .105, 1, .89))
    return fig


def pair_markdown(pairs: list[dict]) -> str:
    """Format all unordered within-object pairs with continuous projection and score gaps."""
    lines = ["# Recorded-view candidate pairs", "", CONDITION, "",
             "| Object | Candidate i | Candidate j | CD-O gap cm | Mean distance | Max distance | Outline | Eligible .03 / .06 | Order | Max score gap | Mean score gap |",
             "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |"]
    lines += [f"| {p['object']} | {p['label_i']} | {p['label_j']} | {p['cd_o_gap_cm']:.3f} | "
              f"{p['mean_projection_distance']:.3f} | {p['max_projection_distance']:.3f} | {p['outline_changing']} | "
              f"{p['eligible_003']} / {p['eligible_006']} | {p['order']} | {p['max_abs_view_iou_gap']:.3f} | {p['abs_mean_iou_gap']:.3f} |"
              for p in pairs]
    return "\n".join(lines) + "\n"
