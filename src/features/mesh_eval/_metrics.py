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


def aggregate(rows: list[dict], pairs: list[dict], comparison: tuple[str, str]) -> dict:
    """Average actual pair scores, then average the per-object means equally."""
    by_key = {(r["file"], r["episode"], r["slot"]): r["cd_o_cm"] for r in rows}
    covered = [p for p in pairs if p["status"] == "covered"]
    labels = sorted({r["file"] for r in rows})
    pair_rows = [{**p, **{label: by_key[label, p["episode"], p["slot"]] for label in labels}}
                 for p in covered]
    before, after = comparison
    for row in pair_rows:
        row["difference_cm"] = row[after] - row[before]
    objects = []
    for name in sorted({p["object"] for p in covered}):
        group = [p for p in pair_rows if p["object"] == name]
        objects.append({"object": name, "pairs": len(group),
                        **{label: float(np.mean([p[label] for p in group])) for label in labels}})
    means = {label: {"pair_weighted_cm": float(np.mean([p[label] for p in pair_rows])),
                     "per_object_cm": float(np.mean([o[label] for o in objects]))} for label in labels}
    return {"covered_pairs": len(covered), "total_pairs": len(pairs), "objects": objects,
            "unscored": [p for p in pairs if p["status"] != "covered"], "pairs": pair_rows,
            "means": means, "comparison": {"before": before, "after": after},
            "largest_absolute_pair_difference": max(pair_rows, key=lambda p: abs(p["difference_cm"])),
            "largest_regression": max(pair_rows, key=lambda p: p["difference_cm"]),
            "improved_pairs": sum(p["difference_cm"] < 0 for p in pair_rows),
            "unchanged_pairs": sum(p["difference_cm"] == 0 for p in pair_rows),
            "worsened_pairs": sum(p["difference_cm"] > 0 for p in pair_rows)}
