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


def shape_for(rule: dict, name: str, extents, parameter, attributes: dict | None = None) -> str:
    """Apply a category rule before any reference scan is read."""
    category = next(key for key, names in CATEGORIES.items() if name in names)
    shape = rule["shapes"][category]
    if rule["name"].startswith(("R4 ", "R5 ", "R6 ")):
        attrs = (attributes or {}).get(name, {})
        evidence = attrs.get("evidence", [])
        confirmed = {key for key in ("handle", "open_top", "pegs")
                     if any(e["attribute"] == key for e in evidence)
                     and all(e["confidence"] == "high" for e in evidence if e["attribute"] == key)}
        threshold = rule["aspect_threshold"]
        handled = extents[0] / extents[1] > threshold
        if category == "round_vessel":
            incumbent = "vessel_with_handle" if handled else "cylinder"
            if handled and ("handle" not in confirmed or attrs.get("handle") not in ("rim_flat", "loop")):
                return incumbent
            if not rule["name"].startswith("R4 "):
                if "open_top" not in confirmed or not isinstance(attrs.get("open_top"), bool):
                    return incumbent
                if attrs["open_top"]:
                    return "vessel_shell"
            return "rim_handle" if handled and attrs.get("handle") == "rim_flat" else incumbent
        if category == "flat_or_blocky" and rule["name"].startswith("R6 ") and "pegs" in confirmed:
            pegs = attrs.get("pegs")
            if (isinstance(pegs, dict) and all(isinstance(pegs.get(k), int) and pegs[k] > 0
                                             for k in ("rows", "count_per_row"))):
                return "peg_rack"
        return "box"
    if shape == "vessel_with_handle" and extents[0] / extents[1] <= parameter:
        return "cylinder"
    return shape


def mesh_for(shape: str, rule: dict, name: str, extents, parameter, attributes: dict):
    """Construct a candidate from plain video measurements and frozen attributes."""
    import trimesh

    from . import _shapes

    if shape == "box":
        return trimesh.creation.box(extents=extents)
    if shape == "cylinder":
        return _shapes.cylinder(extents)
    if shape == "vessel_with_handle":
        return _shapes.vessel_with_handle(extents, rule["handle_height_fraction"])
    if shape == "rim_handle":
        return _shapes.rim_handle(extents)
    if shape == "vessel_shell":
        handle = attributes[name].get("handle", "none") if extents[0] / extents[1] > rule["aspect_threshold"] else "none"
        return _shapes.vessel_shell(extents, parameter, handle)
    if shape == "peg_rack":
        return _shapes.peg_rack(extents, **attributes[name]["pegs"])
    return _shapes.head_and_handle(extents, parameter)


def select(rule: dict, trials: list[dict], baseline: dict, pair_counts: dict, thresholds: dict | None = None) -> dict:
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
    thresholds = thresholds or {"minimum_gain_cm": 0.03, "maximum_regression_cm": 0.15}
    baseline_mean = float(np.mean(list(baseline.values())))
    baseline_pair_mean = float(np.average([baseline[n] for n in NAMES], weights=[pair_counts[n] for n in NAMES]))
    regression = max(held_out[n] - baseline[n] for n in NAMES)
    gates = {f"gain_gt_{thresholds['minimum_gain_cm']:g}_cm": baseline_mean - mean > thresholds["minimum_gain_cm"],
             f"worst_regression_le_{thresholds['maximum_regression_cm']:g}_cm": regression <= thresholds["maximum_regression_cm"],
             "pair_mean_no_worse": pair_mean <= baseline_pair_mean}
    return {"rule": rule["name"], "parameter": rule["parameter"],
            "chosen_on_fold": {fold: trials[i]["parameter"] for fold, i in chosen.items()},
            "cross_fitted_mean_cm": mean, "in_sample_mean_cm": float(np.mean(list(scores[best].values()))),
            "in_sample_parameter": trials[best]["parameter"], "pair_weighted_mean_cm": pair_mean,
            "worst_held_out_regression_cm": regression, "gates": gates, "adopt": all(gates.values()),
            "per_object_cm": held_out, "trials": trials}
