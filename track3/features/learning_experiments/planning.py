"""Bounded, deterministic metadata planning; never constructs or scores models."""

from dataclasses import dataclass
import hashlib
import itertools
import json
import random

from ._common import (FEATURE, TECHNIQUES, authorize, episode_partition, integer,
                      persist_plan, read_json)
from ._config import parameters


@dataclass(frozen=True)
class PlanRequest:
    method: str = "random"
    seed: int = 31
    max_trials: int = 12
    techniques: tuple[str, ...] = TECHNIQUES
    train_episodes: tuple[int, ...] = (0, 11, 12, 18, 21, 23, 26, 31)
    tuning_episodes: tuple[int, ...] = (39, 42)
    holdout_episodes: tuple[int, ...] = (43, 44)

    @classmethod
    def from_file(cls, path):
        authorize("plan")
        value = read_json(path)
        for key in ("techniques", "train_episodes", "tuning_episodes", "holdout_episodes"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _candidates(technique, space):
    if not isinstance(space, dict) or not 1 <= len(space) <= 10:
        raise ValueError("Search spaces need 1–10 dimensions")
    names = sorted(space)
    total = 1
    for name in names:
        choices = space[name]
        if (not isinstance(choices, list) or not 1 <= len(choices) <= 16
                or len({_canonical(v) for v in choices}) != len(choices)):
            raise ValueError("Each dimension needs 1–16 unique JSON values")
        total *= len(choices)
        if total > 4096:
            raise ValueError("At most 4096 combinations per technique")
    result = []
    seen = set()
    for values in itertools.product(*(space[name] for name in names)):
        config = parameters(technique, dict(zip(names, values)), planning=True)
        identity = _canonical(config)
        if identity not in seen:
            seen.add(identity)
            result.append(config)
    return result


def plan_trials(request: PlanRequest):
    authorize("plan")
    if request.method not in ("grid", "random"):
        raise ValueError("Only bounded grid and random planning are supported")
    integer(request.seed, "seed", 0, 2**32 - 1)
    integer(request.max_trials, "max_trials", 1, 128)
    if (not isinstance(request.techniques, (tuple, list))
            or not 1 <= len(request.techniques) <= len(TECHNIQUES)
            or any(t not in TECHNIQUES for t in request.techniques)
            or len(set(request.techniques)) != len(request.techniques)):
        raise ValueError("Select unique supported techniques")
    if request.max_trials < len(request.techniques):
        raise ValueError("Budget must cover every selected technique at least once")
    partition = episode_partition(request.train_episodes, request.tuning_episodes, request.holdout_episodes)
    search = read_json(FEATURE / "search.json")
    if search["schema_version"] != 1 or search["metric"] != "log_depth_mae":
        raise ValueError("Unsupported search contract")
    selected = [t for t in TECHNIQUES if t in request.techniques]
    pools = {t: _candidates(t, search["spaces"][t]) for t in selected}
    if request.method == "random":
        # Independent per-technique streams make caller ordering irrelevant.
        for technique, pool in pools.items():
            digest = hashlib.sha256(f"{request.seed}:{technique}".encode()).digest()
            random.Random(int.from_bytes(digest, "big")).shuffle(pool)
    trials = []
    for index in range(max(len(pool) for pool in pools.values())):
        for technique in selected:
            if index >= len(pools[technique]):
                continue
            config = pools[technique][index]
            identity = {"schema_version": 1, "method": request.method,
                        "seed": request.seed, "technique": technique,
                        "parameters": config, "episode_partition": partition}
            digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
            trials.append({"trial_id": f"{technique}-{digest[:24]}",
                           "trial_seed": int(digest[:8], 16),
                           "technique": technique, "parameters": config,
                           "status": "planned_only", "tuning_score": None,
                           "final_holdout_score": None, "checkpoint": None})
            if len(trials) == request.max_trials:
                break
        if len(trials) == request.max_trials:
            break
    return {"schema_version": 1, "feature": "learning_experiments", "operation": "plan",
            "status": "planned_only", "method": request.method, "seed": request.seed,
            "max_trials": request.max_trials, "trial_count": len(trials),
            "candidate_count": sum(len(pool) for pool in pools.values()),
            "coverage": {t: sum(trial["technique"] == t for trial in trials) for t in selected},
            "selection_order": "round_robin_across_techniques_without_replacement",
            "episode_partition": partition,
            "objective": {"metric": "log_depth_mae", "direction": "minimize",
                          "aggregation": "equal_weight_episode_means_of_valid_pixel_log_depth_mae",
                          "selection_episodes": partition["tuning_episodes"],
                          "status": "future_only", "holdout_selection_allowed": False},
            "model_execution": False, "best_model": None, "scores": None,
            "submission_candidate": False, "scratch_checkpoint_admitted": False,
            "trials": trials}


def write_plan(request: PlanRequest):
    """Authorized operation plan: compute metadata and persist a new owning run."""
    return persist_plan(request, plan_trials(request))
