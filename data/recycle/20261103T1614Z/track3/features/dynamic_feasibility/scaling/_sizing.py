"""Pure throughput-only selection for the committed D1 rule."""

from track3.features.dynamic_feasibility._metrics import scaled_thresholds


def choose_budget(probes, base, maximum_peak_mib, duration):
    """Select the fastest memory-admissible probe and scale only its transition curriculum."""
    candidates = [row for row in probes if row["status"] == "completed"
                  and row["peak_gpu_mib"] <= maximum_peak_mib and row["steps_per_second"] > 0]
    if not candidates:
        raise ValueError("No sizing probe fits the committed memory ceiling")
    best = max(candidates, key=lambda row: row["steps_per_second"])
    worlds = best["world_count"]
    steps = int(best["steps_per_second"] * duration // worlds) * worlds
    budget = dict(base, world_count=worlds, total_environment_steps=steps)
    budget["curriculum_thresholds"] = scaled_thresholds(base["curriculum_thresholds"],
                                                       base["total_environment_steps"], steps, worlds)
    if steps < base["minimum_size"]:
        raise ValueError("Sized budget cannot fill the unchanged L1 replay minimum")
    return budget, best
