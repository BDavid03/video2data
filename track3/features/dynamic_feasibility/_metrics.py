"""Pure pose statistics and fixed-budget arithmetic."""

import numpy as np


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
