"""Append-only iteration ledger and the compact summaries the advisor and dashboard consume."""

import json
import math

from ._common import now, state_dir
from .metrics import generalisation_gap, usability
from .recipes import describe


def ledger_path():
    return state_dir() / "ledger.jsonl"


def load():
    path = ledger_path()
    if not path.exists():
        return []
    return [with_usability(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(entry):
    with ledger_path().open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, allow_nan=False) + "\n")
    return entry


def entry_from_outcome(iteration, source, outcome, directory):
    metrics = (outcome.get("metrics") or {}).get("overall", {})
    training = outcome.get("training") or {}
    return with_usability({"iteration": iteration, "run_id": directory.name, "source": source, "recorded_utc": now(),
            "identity": outcome["identity"], "technique": outcome["recipe"]["technique"],
            "recipe": outcome["recipe"], "description": describe(outcome["recipe"]),
            "status": outcome["status"], "error": outcome.get("error"),
            "tuning_log_mae": outcome.get("score"), "tuning_log_mae_std": outcome.get("score_std"),
            "replicates": outcome["recipe"].get("replicates", 1), "completed_replicates": outcome.get("completed_replicates"),
            "abs_rel": metrics.get("abs_rel"), "rmse_log": metrics.get("rmse_log"), "silog": metrics.get("silog"),
            "delta1": metrics.get("delta1"), "mean_log_bias": metrics.get("mean_log_bias"),
            "support_fraction": metrics.get("support_fraction"),
            "episode_spread_log_mae": metrics.get("episode_spread_log_mae"),
            "edge_correlation": metrics.get("edge_correlation"),
            "episode_log_mae": {k: v["log_depth_mae"] for k, v in (outcome.get("metrics") or {}).get("episodes", {}).items()},
            "final_train_log_mae": training.get("final_train_log_mae"),
            "generalisation_gap": generalisation_gap(outcome),
            "epochs_to_95pct": training.get("epochs_to_95pct"),
            "parameter_count": training.get("parameter_count"),
            "gpu_peak_allocated_mib": training.get("gpu_peak_allocated_mib"),
            "train_wall_seconds": training.get("wall_seconds"), "wall_seconds": outcome.get("wall_seconds"),
            "checkpoint": (outcome.get("checkpoint") or {}).get("path"),
            "prediction_run": outcome.get("prediction_run"), "scoring_run": outcome.get("scoring_run"),
            "outcome_path": str(directory / "outcome.json")})


def with_usability(entry):
    """Attach usable/failed_gates/composite (idempotent; safe on older entries)."""
    if entry.get("status") == "completed":
        entry.update(usability(entry))
    else:
        entry.update(usable=False, failed_gates=["not completed"], composite=None)
    return entry


def completed(entries):
    return [e for e in entries if e["status"] == "completed" and isinstance(e.get("tuning_log_mae"), (int, float))]


def best_mae(entries):
    done = completed(entries)
    return min(done, key=lambda e: (e["tuning_log_mae"], e["iteration"])) if done else None


def usable(entries):
    return [e for e in completed(entries) if e.get("usable") and isinstance(e.get("composite"), (int, float))]


def best(entries):
    """Selection: lowest composite usability score among gate-passing entries; MAE only as a fallback."""
    passing = usable(entries)
    if passing:
        return min(passing, key=lambda e: (e["composite"], e["tuning_log_mae"], e["iteration"]))
    return best_mae(entries)


def _objective(e):
    """Per-entry value the loop tries to minimise: composite when usable, else +inf."""
    return e["composite"] if e.get("usable") and isinstance(e.get("composite"), (int, float)) else math.inf


def best_so_far(entries, key="tuning_log_mae"):
    out, current = [], None
    for e in entries:
        value = _objective(e) if key == "composite" else (e["tuning_log_mae"] if e["status"] == "completed" else None)
        if value is not None and math.isfinite(value) and (current is None or value < current):
            current = value
        out.append(current)
    return out


def iterations_since_improvement(entries):
    best_value, since = None, 0
    for e in entries:
        value = _objective(e)
        if math.isfinite(value) and (best_value is None or value < best_value - 1e-9):
            best_value, since = value, 0
        else:
            since += 1
    return since


def summary(entries, recent=12):
    """Compact JSON for the advisor prompt: leaderboard, per-technique bests and the recent tail."""
    done = completed(entries)
    leaderboard = sorted(done, key=lambda e: (_objective(e), e["tuning_log_mae"]))[:8]
    per_technique = {}
    for e in done:
        current = per_technique.get(e["technique"])
        if current is None or (_objective(e), e["tuning_log_mae"]) < (_objective(current), current["tuning_log_mae"]):
            per_technique[e["technique"]] = e
    keep = ("iteration", "technique", "description", "status", "error", "tuning_log_mae", "tuning_log_mae_std",
            "replicates", "abs_rel", "rmse_log",
            "silog", "delta1", "mean_log_bias", "support_fraction", "episode_spread_log_mae", "edge_correlation",
            "usable", "failed_gates", "composite", "episode_log_mae",
            "final_train_log_mae", "generalisation_gap", "epochs_to_95pct", "parameter_count",
            "gpu_peak_allocated_mib", "train_wall_seconds", "source", "recipe")

    def compact(e):
        return {k: e.get(k) for k in keep}

    from .metrics import COMPOSITE_WEIGHTS, GATES
    return {"iterations": len(entries), "completed": len(done), "failed": len(entries) - len(done),
            "selection_rule": {"objective": "minimise composite among entries passing every gate",
                               "gates": {k: f"{op} {v}" for k, (op, v) in GATES.items()},
                               "composite_weights": COMPOSITE_WEIGHTS,
                               "note": "tuning_log_mae alone does not select; a blurred, biased, patchy or inconsistent model is unusable"},
            "best": compact(best(entries)) if done else None,
            "best_by_mae_only": compact(best_mae(entries)) if done else None,
            "usable_count": len(usable(entries)),
            "iterations_since_improvement": iterations_since_improvement(entries),
            "leaderboard": [compact(e) for e in leaderboard],
            "best_per_technique": {k: compact(v) for k, v in per_technique.items()},
            "recent": [compact(e) for e in entries[-recent:]],
            "tried_identities": sorted({e["identity"] for e in entries})}
