"""Meaningful comparison metrics beyond the selection objective.

Selection stays on equal-episode log-depth MAE against MoGe2 pseudo-targets.
The extra metrics (AbsRel, RMSE-log, delta thresholds, SILog, support,
generalisation gap, convergence, cost) exist to explain *why* a recipe wins and
to steer the next proposal. They never choose the model.
"""

import math

from track3.paths import project_path
from ._common import ROOT, read_json


def training_summary(record):
    epochs = record.get("epochs") or []
    curve = [e.get("training_log_depth_mae") for e in epochs if isinstance(e.get("training_log_depth_mae"), (int, float))]
    summary = {"kind": record.get("kind"), "final_train_log_mae": curve[-1] if curve else None,
               "best_train_log_mae": min(curve) if curve else None, "epochs_run": len(curve),
               "loss_curve": curve, "parameter_count": record.get("parameter_count"),
               "optimizer_steps": record.get("optimizer_steps"),
               "gpu_peak_allocated_mib": (record.get("gpu_torch") or {}).get("peak_allocated_bytes", 0) / 2**20 or None,
               "producer_wall_seconds": (record.get("timing") or {}).get("wall_seconds")}
    if len(curve) >= 3:
        # epochs needed to get within 5% of the final loss: a convergence-speed signal
        target = curve[-1] * 1.05
        summary["epochs_to_95pct"] = next(i + 1 for i, v in enumerate(curve) if v <= target)
        summary["last_epoch_improvement"] = (curve[-2] - curve[-1]) / max(abs(curve[-2]), 1e-12)
    if record.get("kind") == "learning_experiments.boost_train":
        summary["parameter_count"] = record.get("tree_count") or record.get("estimators")
    return summary


def extended_metrics(dataset, prediction_run, group, score_record):
    """Per-episode and equal-episode depth metrics from stored prediction NPZ files."""
    import numpy as np
    from track3.features.learning_experiments.data import PreparedPairs

    partition = score_record["episode_partition"]
    pairs = PreparedPairs(project_path(ROOT, dataset), partition, group, np)
    producer = read_json(project_path(ROOT, prediction_run))
    frames = {(f["episode_index"], f["frame_index"]): f for f in producer["frames"]}
    base = project_path(ROOT, prediction_run).parent
    per_episode = {}
    for index in range(len(pairs)):
        sample = pairs[index]
        row = sample["row"]
        frame = frames[(row["episode_index"], row["frame_index"])]
        with np.load(base / frame["artifact"], allow_pickle=False) as arrays:
            depth, valid = arrays["depth_m"].astype(np.float64), arrays["valid"]
        target = sample["depth_m"].astype(np.float64)
        mask = valid & sample["valid"] & np.isfinite(depth) & np.isfinite(target) & (depth > 0) & (target > 0)
        if not mask.any():
            continue
        d, t = depth[mask], target[mask]
        log_diff = np.log(d) - np.log(t)
        # Edge fidelity: correlation of log-depth gradients where both are defined.
        # A blurred or constant prediction scores ~0 here even if its MAE is modest.
        lp, lt = np.log(np.where(mask, depth, np.nan)), np.log(np.where(mask, target, np.nan))
        gp = np.concatenate([np.diff(lp, axis=0).ravel(), np.diff(lp, axis=1).ravel()])
        gt = np.concatenate([np.diff(lt, axis=0).ravel(), np.diff(lt, axis=1).ravel()])
        both = np.isfinite(gp) & np.isfinite(gt)
        if both.sum() > 16 and gp[both].std() > 0 and gt[both].std() > 0:
            edge = float(np.corrcoef(gp[both], gt[both])[0, 1])
        else:
            edge = 0.0
        ratio = np.maximum(d / t, t / d)
        acc = per_episode.setdefault(row["episode_index"], {"n": 0, "abs_log": 0.0, "sq_log": 0.0, "log": 0.0,
                                                            "abs_rel": 0.0, "d1": 0, "d2": 0, "d3": 0,
                                                            "target_pixels": 0, "frames": 0, "edge_sum": 0.0})
        acc["n"] += int(mask.sum())
        acc["abs_log"] += float(np.abs(log_diff).sum())
        acc["sq_log"] += float((log_diff ** 2).sum())
        acc["log"] += float(log_diff.sum())
        acc["abs_rel"] += float((np.abs(d - t) / t).sum())
        acc["d1"] += int((ratio < 1.25).sum())
        acc["d2"] += int((ratio < 1.25 ** 2).sum())
        acc["d3"] += int((ratio < 1.25 ** 3).sum())
        acc["target_pixels"] += int(sample["valid"].sum())
        acc["frames"] += 1
        acc["edge_sum"] += edge
    episodes = {}
    for episode, acc in sorted(per_episode.items()):
        n = acc["n"]
        mean_log = acc["log"] / n
        episodes[str(episode)] = {
            "log_depth_mae": acc["abs_log"] / n, "abs_rel": acc["abs_rel"] / n,
            "rmse_log": math.sqrt(acc["sq_log"] / n),
            "silog": math.sqrt(max(acc["sq_log"] / n - mean_log ** 2, 0.0)),
            "mean_log_bias": mean_log, "delta1": acc["d1"] / n, "delta2": acc["d2"] / n, "delta3": acc["d3"] / n,
            "support_fraction": n / max(acc["target_pixels"], 1), "frames": acc["frames"], "pixels": n,
            "edge_correlation": acc["edge_sum"] / acc["frames"]}
    keys = ("log_depth_mae", "abs_rel", "rmse_log", "silog", "mean_log_bias", "delta1", "delta2", "delta3",
            "support_fraction", "edge_correlation")
    overall = {key: sum(e[key] for e in episodes.values()) / len(episodes) for key in keys} if episodes else {}
    overall["episode_spread_log_mae"] = (max(e["log_depth_mae"] for e in episodes.values())
                                         - min(e["log_depth_mae"] for e in episodes.values())) if len(episodes) > 1 else 0.0
    overall["frames"] = sum(e["frames"] for e in episodes.values())
    return {"aggregation": "equal_weight_episode_means", "overall": overall, "episodes": episodes}


def generalisation_gap(outcome):
    train_mae = (outcome.get("training") or {}).get("final_train_log_mae")
    score = outcome.get("score")
    if isinstance(train_mae, (int, float)) and isinstance(score, (int, float)):
        return score - train_mae
    return None


# Usability: a model with low MAE that is blurred, biased in scale, patchy or inconsistent
# across episodes is not a usable depth estimator. Gates are hard requirements; the
# composite ranks usable candidates on all facets, not MAE alone. Lower is better.
GATES = {"support_fraction": (">=", 0.98), "delta1": (">=", 0.70), "abs_mean_log_bias": ("<=", 0.10),
         "edge_correlation": (">=", 0.40), "episode_spread_log_mae": ("<=", 0.05)}
COMPOSITE_WEIGHTS = {"log_depth_mae": 1.0, "silog": 1.0, "one_minus_delta1": 1.0, "abs_mean_log_bias": 1.0,
                     "one_minus_edge_correlation": 0.5}


def usability(entry):
    """Return {usable, failed_gates, composite} from a ledger entry or outcome overall metrics."""
    values = {"support_fraction": entry.get("support_fraction"), "delta1": entry.get("delta1"),
              "abs_mean_log_bias": abs(entry["mean_log_bias"]) if isinstance(entry.get("mean_log_bias"), (int, float)) else None,
              "edge_correlation": entry.get("edge_correlation"),
              "episode_spread_log_mae": entry.get("episode_spread_log_mae"),
              "log_depth_mae": entry.get("tuning_log_mae", entry.get("log_depth_mae")), "silog": entry.get("silog")}
    failed = []
    for name, (op, bound) in GATES.items():
        value = values.get(name)
        if not isinstance(value, (int, float)):
            failed.append(f"{name}: missing")
        elif (op == ">=" and value < bound) or (op == "<=" and value > bound):
            failed.append(f"{name} {value:.3f} {'<' if op == '>=' else '>'} {bound}")
    terms = {"log_depth_mae": values["log_depth_mae"], "silog": values["silog"],
             "one_minus_delta1": None if values["delta1"] is None else 1 - values["delta1"],
             "abs_mean_log_bias": values["abs_mean_log_bias"],
             "one_minus_edge_correlation": None if values["edge_correlation"] is None else 1 - values["edge_correlation"]}
    if any(not isinstance(v, (int, float)) for v in terms.values()):
        composite = None
    else:
        composite = sum(COMPOSITE_WEIGHTS[k] * v for k, v in terms.items())
    return {"usable": not failed, "failed_gates": failed, "composite": composite}
