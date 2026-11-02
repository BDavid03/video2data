"""Pure frame construction and pair/object aggregation."""

import numpy as np
import pandas as pd


def solution_frame(episode: int, slot: int, name: str, reference: tuple, row_id, arrays) -> pd.DataFrame:
    """Represent one budgeted reference in the kit's solution schema."""
    return pd.concat([
        pd.DataFrame({
            "row_id": [row_id(3, episode, name, array, i) for i in range(len(values))],
            "episode_index": episode, "object_slot": slot, "object_name": name,
            "array_name": arrays[array], "point_index": np.arange(len(values)),
            "ref_x": values[:, 0], "ref_y": values[:, 1], "ref_z": values[:, 2],
        }) for array, values in enumerate(reference)
    ], ignore_index=True)


def aggregate(rows: list[dict], pairs: list[dict]) -> dict:
    """Average actual pair scores, then average the per-object means equally."""
    by_key = {(r["file"], r["episode"], r["slot"]): r["cd_o_cm"] for r in rows}
    covered = [p for p in pairs if p["status"] == "covered"]
    labels = sorted({r["file"] for r in rows})
    pair_rows = [{**p, **{label: by_key[label, p["episode"], p["slot"]] for label in labels}}
                 for p in covered]
    for row in pair_rows:
        row["v2_minus_v1_cm"] = row["v2"] - row["v1"]
    objects = []
    for name in sorted({p["object"] for p in covered}):
        group = [p for p in pair_rows if p["object"] == name]
        objects.append({"object": name, "pairs": len(group),
                        **{label: float(np.mean([p[label] for p in group])) for label in labels}})
    means = {label: {"pair_weighted_cm": float(np.mean([p[label] for p in pair_rows])),
                     "per_object_cm": float(np.mean([o[label] for o in objects]))} for label in labels}
    previous = {"v1_pair_reused_cm": 1.25662, "v2_pair_cross_fitted_reused_cm": 1.08742,
                "v2_per_object_fitted_cm": 0.9251, "v1_per_object_reused_cm": 1.00764,
                "v2_per_object_cross_fitted_cm": 0.92572}
    differences = {
        "v1_pair_vs_reused_cm": means["v1"]["pair_weighted_cm"] - previous["v1_pair_reused_cm"],
        "v2_pair_vs_cross_fitted_reused_cm": means["v2"]["pair_weighted_cm"] - previous["v2_pair_cross_fitted_reused_cm"],
        "v2_object_vs_fitted_cm": means["v2"]["per_object_cm"] - previous["v2_per_object_fitted_cm"],
        "v1_object_vs_reused_cm": means["v1"]["per_object_cm"] - previous["v1_per_object_reused_cm"],
        "v2_object_vs_cross_fitted_cm": means["v2"]["per_object_cm"] - previous["v2_per_object_cross_fitted_cm"],
    }
    return {"covered_pairs": len(covered), "total_pairs": len(pairs), "objects": objects,
            "unscored": [p for p in pairs if p["status"] != "covered"], "pairs": pair_rows,
            "means": means, "previous": previous, "differences": differences,
            "largest_absolute_pair_difference": max(pair_rows, key=lambda p: abs(p["v2_minus_v1_cm"])),
            "largest_regression": max(pair_rows, key=lambda p: p["v2_minus_v1_cm"]),
            "improved_pairs": sum(p["v2_minus_v1_cm"] < 0 for p in pair_rows),
            "unchanged_pairs": sum(p["v2_minus_v1_cm"] == 0 for p in pair_rows),
            "worsened_pairs": sum(p["v2_minus_v1_cm"] > 0 for p in pair_rows)}
