"""Pure D0 and D1 reports from ledger rows and saved campaign decisions."""

import json

from ._metrics import verdict


def d0_summary(rows, static, campaign, l2):
    """Publish compact per-trial measurements and the exact go/no-go decision."""
    decision = verdict(rows, static)
    used = sum(row["gpu_seconds"] for row in rows.values())
    report = {**decision, "campaign_identity": campaign["identity"], "comparator_S": static,
              "trials": list(rows.values()), "l2_decision": l2, "gpu_seconds": used,
              "note": "Raw world-frame position/orientation. ADD-AUC uses kit initial slot-0 alignment. "
                      "1,000 seeded surface samples of the shipped task mesh; not challenge scores."}
    lines = ["# Dynamic feasibility D0", "", f"**{decision['verdict']}**; both-seed position limit "
             f"{decision['mean_position_limit_cm']:.4f} cm. GPU lease time including compilation: {used / 60:.2f} min.",
             "", report["note"], "", "| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | P90 ori deg | ADD-AUC | Complete | Wall s | Peak MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    def metrics_row(name, seed, metrics, complete, wall, memory):
        values = [f"{metrics[key]:.6f}" for key in ("mean_position_cm", "p90_position_cm", "mean_orientation_deg", "p90_orientation_deg", "add_auc")]
        return "| " + " | ".join([name, str(seed), *values, str(complete), str(wall), str(memory)]) + " |"
    lines.append(metrics_row("S (computed)", "—", static, "reference timeline", 0, 0))
    for row in rows.values():
        if row["metrics"]:
            lines.append(metrics_row(row["arm"], row["seed"], row["metrics"], row["metrics"]["completed"],
                                     round(row["wall_seconds"], 2), row["peak_gpu_mib"]))
        else:
            lines.append(f"| {row['arm']} | {row['seed']} | — | — | — | — | — | {row['status']} | {row['wall_seconds']} | {row['peak_gpu_mib']} |")
    ranking = sorted((row for row in rows.values() if row["metrics"]),
                     key=lambda row: row["metrics"]["mean_position_cm"])
    lines += ["", "Measured ranking by mean position: " + ", ".join(row["trial"] for row in ranking),
              "", "L2 decision: " + json.dumps(l2, sort_keys=True), "", decision["rule"],
              "", "A NO-GO means this bounded policy stage did not justify a dynamic pack by the deadline; it does not establish an impossibility of manipulation."]
    for row in rows.values():
        if row["error"]:
            lines.append(f"\n{row['trial']}: {row['error']}")
    return report, "\n".join(lines) + "\n"


def d1_summary(rows, baseline, prior_probes, prior_seconds, selection=None, error=None):
    """Report sizing and both-seed outcomes beside the unchanged S and Z comparators."""
    static = baseline["comparator_S"]
    zero = next(row for row in baseline["trials"] if row["arm"] == "Z")
    trials = [dict(row) for row in rows.values() if row["event"] == "trial"]
    limit = min(3., .5 * static["mean_position_cm"])
    go = all(any(row["seed"] == seed and row["status"] == "completed" and row["metrics"]["completed"]
                 and row["metrics"]["mean_position_cm"] <= limit for row in trials) for seed in (0, 1))
    report = {"verdict": "GO" if go else "NO-GO", "position_limit_cm": limit,
              "probes": [*prior_probes, *[row for row in rows.values() if row["event"] == "probe"]], "selection": selection,
              "trials": trials, "comparator_S": static, "comparator_Z": zero,
              "gpu_seconds": prior_seconds + sum(row["gpu_seconds"] for row in rows.values()), "error": error,
              "setup_correction": "Flattened simulator observation replaced by the existing WarpRLEnv.reset view before any D1 training; all failed sizing time retained and charged"}
    for row in trials:
        if row["metrics"]:
            row["improvement_over_Z"] = 1 - row["metrics"]["mean_position_cm"] / zero["metrics"]["mean_position_cm"]
    lines = ["# Dynamic scaling D1", "", f"**{report['verdict']}**; total GPU time {report['gpu_seconds']/60:.2f} minutes.",
             "", "| Probe worlds | Steps/s | Peak device MiB | Status |", "|---:|---:|---:|---|"]
    for row in report["probes"]:
        lines.append(f"| {row['world_count']} | {row['steps_per_second']:.3f} | {row['peak_gpu_mib']} | {row['status']} |")
    if selection:
        chosen = selection["chosen"]
        lines += ["", f"Budget frozen at **{chosen['written_utc']}**: "
                  f"{chosen['budget']['world_count']} worlds, {chosen['budget']['total_environment_steps']:,} transitions."]
    lines += ["", "Position/orientation are raw world-frame errors. ADD-AUC uses D0's seeded task surface and kit initial alignment.",
              "", "| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | P90 ori deg | ADD-AUC | Complete | Wall s | Peak MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    measurements = [("S, computed", "—", static, "reference", "—", "—"),
                    ("Z", 0, zero["metrics"], True, zero["wall_seconds"], zero["peak_gpu_mib"])]
    for row in trials:
        if row["metrics"]:
            measurements.append(("D1", row["seed"], row["metrics"], row["metrics"]["completed"], row["wall_seconds"], row["peak_gpu_mib"]))
        else:
            lines.append(f"| D1 | {row['seed']} | — | — | — | — | — | {row['status']} | {row['wall_seconds']} | {row['peak_gpu_mib']} |")
    for arm, seed, metrics, complete, wall, memory in measurements:
        values = [f"{metrics[key]:.6f}" for key in ("mean_position_cm", "p90_position_cm", "mean_orientation_deg", "p90_orientation_deg", "add_auc")]
        lines.append("| " + " | ".join([arm, str(seed), *values, str(complete), str(wall), str(memory)]) + " |")
    for row in trials:
        if row["metrics"]:
            lines.append(f"\nSeed {row['seed']} improvement over Z: {100*row['improvement_over_Z']:.4f}%.")
        if row["error"]:
            lines.append(f"\nSeed {row['seed']}: {row['error']}")
    lines += ["", f"GO requires BOTH seeds to complete with mean position <= {limit:.4f} cm."]
    if error:
        lines.append(f"\n{error}")
    return report, "\n".join(lines) + "\n"

