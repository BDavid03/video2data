"""Pure collection-grid sizing from measured end-to-end training throughput."""

from ._metrics import extension_steps, scaled_thresholds


def choose_budget(measurements, base, maximum_peak_mib, duration):
    """Size from completed training, charging 1.1 times measured seconds per transition."""
    candidates = [row for row in measurements if row["status"] == "completed"
                  and row.get("training_seconds", 0) > 0
                  and row["gpu"]["peak_sampled_total_device_used_mib"] <= maximum_peak_mib
                  and row["request"]["arm"] != "Z"]
    if not candidates:
        raise ValueError("No measured training run fits the memory ceiling")
    # Use the slower replicate for each measured world count, then the fastest admissible count.
    rates = {}
    for row in candidates:
        worlds = row["request"]["budget"]["world_count"]
        rate = row["training_seconds"] / row["request"]["budget"]["total_environment_steps"]
        if worlds not in rates or rate > rates[worlds][0]:
            rates[worlds] = rate, row
    worlds = min(rates, key=lambda count: rates[count][0])
    rate, best = rates[worlds]
    steps = extension_steps(rate, 2 * duration, 0, 1.1, worlds)
    budget = dict(base, world_count=worlds, total_environment_steps=steps)
    budget["curriculum_thresholds"] = scaled_thresholds(base["curriculum_thresholds"],
                                                       base["total_environment_steps"], steps, worlds)
    if steps < base["minimum_size"]:
        raise ValueError("Sized budget cannot fill the unchanged L1 replay minimum")
    return budget, best
