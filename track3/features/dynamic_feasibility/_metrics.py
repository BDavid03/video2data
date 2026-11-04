"""Pure pose statistics and fixed-budget arithmetic."""

import numpy as np


def transport(achieved, reference):
    """Return through-origin displacement gain on world-frame positions in metres."""
    achieved, reference = (np.asarray(x, dtype=np.float64) for x in (achieved, reference))
    da, dr = achieved - achieved[0], reference - reference[0]
    return float(np.sum(da * dr) / np.sum(dr * dr))


def motion_score(achieved, reference):
    """Compare 15-frame motion increments with the still-object baseline."""
    achieved, reference = (np.asarray(x, dtype=np.float64) for x in (achieved, reference))
    da, dr = achieved[15:] - achieved[:-15], reference[15:] - reference[:-15]
    return float(1 - np.sum((da - dr) ** 2) / np.sum(dr ** 2))


def perpendicular_error(achieved, reference):
    """Return mean perpendicular displacement in cm where reference travel is >=1 cm."""
    achieved, reference = (np.asarray(x, dtype=np.float64) for x in (achieved, reference))
    da, dr = achieved - achieved[0], reference - reference[0]
    distance = np.linalg.norm(dr, axis=-1)
    selected = distance >= .01
    unit = dr[selected] / distance[selected, None]
    residual = da[selected] - np.sum(da[selected] * unit, axis=-1, keepdims=True) * unit
    return float(np.linalg.norm(residual, axis=-1).mean() * 100)


def jump_hold_controls(reference):
    """Return one jump-at-frame-1 and hold trajectory for every reference target frame."""
    reference = np.asarray(reference, dtype=np.float64)
    controls = np.repeat(reference[:, None, :], len(reference), axis=1)
    controls[:, 0] = reference[0]
    return controls


def m0_thresholds(rows, static_error):
    """Return the unchanged prospective M0 metric thresholds."""
    baseline = max(rows[name]["T"] for name in ("Z", "L1-seed0", "L1-seed1"))
    return {"GO": {"T": .50, "Q": .50, "E": .5 * static_error},
            "SIGNAL": {"T": max(.20, baseline + .10), "Q": .10, "E": static_error}}


def meets_m0(row, thresholds):
    """Apply raw M0 metric limits without rounding or tolerance."""
    return (row["T"] >= thresholds["T"] and row["Q"] >= thresholds["Q"]
            and row["mean_position_cm"] <= thresholds["E"])


def m0_verdict(rows, checks, controls, thresholds):
    """Block invalid measurements and non-discriminating controls before claiming transport."""
    if not all(check["passed"] for check in checks.values()):
        return "INVALID"
    seeds = [rows[f"D1p-seed{seed}"] for seed in (0, 1)]
    for name, passed in (("GO", all(meets_m0(row, thresholds["GO"]) for row in seeds)),
                         ("SIGNAL", any(meets_m0(row, thresholds["SIGNAL"]) for row in seeds))):
        if passed:
            return f"METRIC-NONDISCRIMINATING ({name})" if controls[name] else name
    return "NULL"


def pose_metrics(achieved, reference, vertices, kit):
    """Measure raw pose errors and kit ADD-AUC on matching xyzw pose arrays."""
    achieved, reference = (np.asarray(value, dtype=np.float64).copy()
                           for value in (achieved, reference))
    if achieved.shape != reference.shape or achieved.ndim != 4 or achieved.shape[-1] != 7:
        raise ValueError("Expected matching T x W x B x 7 poses")
    if not np.isfinite(achieved).all() or not np.isfinite(reference).all():
        raise ValueError("Pose arrays contain nonfinite values")
    for value in (achieved, reference):
        norms = np.linalg.norm(value[..., 3:], axis=-1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("Degenerate quaternion")
        value[..., 3:] /= norms
    position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1) * 100
    orientation = np.degrees(kit._quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:]))
    aligned = kit._warp_to_reference_start(achieved, reference)
    return {"mean_position_cm": float(position.mean()),
            "p90_position_cm": float(np.quantile(position, .9)),
            "mean_orientation_deg": float(orientation.mean()),
            "p90_orientation_deg": float(np.quantile(orientation, .9)),
            "add_auc": kit._add_auc(kit._object_add(aligned, reference, vertices)),
            "raw_add_auc": kit._add_auc(kit._object_add(achieved, reference, vertices))}


def scaled_thresholds(thresholds, original_steps, steps, worlds):
    """Preserve a curriculum's relative boundaries on the collection grid."""
    return [int(round(value * steps / original_steps / worlds)) * worlds for value in thresholds]


def extension_steps(training_seconds_per_step, pair_seconds, reserve_seconds, safety_factor, worlds):
    """Choose the largest collection-grid budget under the fixed conservative time model."""
    available = pair_seconds / 2 - reserve_seconds
    return max(0, int(available / (training_seconds_per_step * safety_factor) // worlds) * worlds)


def verdict(rows, static):
    """Apply the committed two-seed position/completion rule, without score selection."""
    limit = min(3., .5 * static["mean_position_cm"])
    passing = []
    for arm in ("L1", "L2"):
        outcomes = [rows.get(f"{arm}-seed{seed}") for seed in (0, 1)]
        if all(row and row["status"] == "completed" and row["metrics"]["completed"]
               and row["metrics"]["mean_position_cm"] <= limit for row in outcomes):
            passing.append(arm)
    return {"verdict": "GO" if passing else "NO-GO", "passing_arms": passing,
            "mean_position_limit_cm": limit,
            "rule": "Some learned arm must complete on BOTH seeds with mean position <=3 cm and <=0.5*S"}
