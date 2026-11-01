"""Fixed name categories, folds and cross-fitted mesh rule selection."""

import numpy as np

FOLDS = {
    "A": ("beige_cup", "white_pot_lid", "wooden_piece_1", "plastic_dish_rack", "mini_dust_pan"),
    "B": ("blue_cup", "white_pot", "water_pitcher", "wooden_piece_2", "mini_sweeper"),
}
CATEGORIES = {
    "round_vessel": ("beige_cup", "blue_cup", "white_pot", "white_pot_lid", "water_pitcher"),
    "flat_or_blocky": ("wooden_piece_1", "wooden_piece_2", "plastic_dish_rack"),
    "thin_tool": ("mini_sweeper", "mini_dust_pan", "wooden_spoon"),
}
NAMES = tuple(sorted(name for fold in FOLDS.values() for name in fold))


def shape_for(rule: dict, name: str, extents, parameter) -> str:
    """Apply a category rule before any reference scan is read."""
    category = next(key for key, names in CATEGORIES.items() if name in names)
    shape = rule["shapes"][category]
    if shape == "vessel_with_handle" and extents[0] / extents[1] <= parameter:
        return "cylinder"
    return shape


def select(rule: dict, trials: list[dict], baseline: dict, pair_counts: dict) -> dict:
    """Select on each training fold and combine the opposite folds' object scores."""
    scores = [{row["object"]: row["cd_o_cm"] for row in trial["objects"]} for trial in trials]
    # Queue order breaks ties; it is fixed before scoring and never uses the held-out fold.
    chosen = {fold: min(range(len(scores)), key=lambda i: np.mean([scores[i][n] for n in names]))
              for fold, names in FOLDS.items()}
    held_out = {name: scores[chosen[train]][name]
                for train, test in (("A", "B"), ("B", "A")) for name in FOLDS[test]}
    best = min(range(len(scores)), key=lambda i: np.mean(list(scores[i].values())))
    mean = float(np.mean(list(held_out.values())))
    pair_mean = float(np.average([held_out[n] for n in NAMES], weights=[pair_counts[n] for n in NAMES]))
    box_mean = float(np.mean(list(baseline.values())))
    box_pair_mean = float(np.average([baseline[n] for n in NAMES], weights=[pair_counts[n] for n in NAMES]))
    regression = max(held_out[n] - baseline[n] for n in NAMES)
    gates = {"gain_gt_0.03_cm": box_mean - mean > 0.03,
             "worst_regression_le_0.15_cm": regression <= 0.15,
             "pair_mean_no_worse": pair_mean <= box_pair_mean}
    return {"rule": rule["name"], "parameter": rule["parameter"],
            "chosen_on_fold": {fold: trials[i]["parameter"] for fold, i in chosen.items()},
            "cross_fitted_mean_cm": mean, "in_sample_mean_cm": float(np.mean(list(scores[best].values()))),
            "in_sample_parameter": trials[best]["parameter"], "pair_weighted_mean_cm": pair_mean,
            "worst_held_out_regression_cm": regression, "gates": gates, "adopt": all(gates.values()),
            "per_object_cm": held_out, "trials": trials}
