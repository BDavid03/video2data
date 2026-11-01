"""Tables, Amendment 1 adoption gates and the placement comparison figure."""
import os
import numpy as np
from track3.provenance import write_json

METRICS = ("AUC", "SP_SR", "MP_SR", "RPE_cm", "MPPE_cm")


def means(rows, key="metrics"):
    """Reduce one candidate with equal weight per episode."""
    return {metric: float(np.mean([row[key][metric] for row in rows])) for metric in METRICS}


def gate(candidate, predecessor):
    """Apply all four Amendment 1 gates against the specified predecessor."""
    current, prior = means(candidate), means(predecessor)
    delta = current["AUC"] - prior["AUC"]
    loss = max(old["metrics"]["AUC"] - new["metrics"]["AUC"] for new, old in zip(candidate, predecessor))
    counts = {metric: [sum(row["metrics"][metric] > 0 for row in rows) for rows in (predecessor, candidate)]
              for metric in ("SP_SR", "MP_SR")}
    checks = dict(auc_gain=delta >= .02 - 1e-12,
                  success_counts=all(new >= old for old, new in counts.values()),
                  mppe=current["MPPE_cm"] <= prior["MPPE_cm"] + .25 + 1e-12,
                  episode_loss=loss <= .10 + 1e-12)
    return dict(passed=all(checks.values()), checks=checks, auc_gain=delta,
                success_counts=counts, mppe_change_cm=current["MPPE_cm"] - prior["MPPE_cm"],
                worst_auc_loss=loss)


def metric_cells(value):
    return " | ".join(f"{value[key]:.6f}" if key == "AUC" else f"{value[key]:.3f}" for key in METRICS)


def summarize(results, directory, best_k1, executed, skipped):
    """Write the full comparison, diagnostic decomposition and sequential recommendation."""
    averages = {name: means(rows) for name, rows in results.items()}
    oracle = means(results["v1"], "oracle_static")
    expected = dict(zip(METRICS, (.531919, .625, 0., 15.1144, 7.4033)))
    if not all(abs(oracle[key] - expected[key]) < 5e-5 for key in METRICS):
        raise ValueError(f"Analogue-order oracle does not reproduce the protocol comparator: {oracle}")
    predecessors = {"K1a": "v1", "K1b": "v1", "K2": "v1", "K1+K2": best_k1}
    gates = {name: dict(predecessor=prior, **gate(results[name], results[prior])) for name, prior in predecessors.items()}
    recommended = best_k1 if gates[best_k1]["passed"] else "none"
    if recommended != "none" and gates["K1+K2"]["passed"]:
        recommended = "K1+K2"
    report = dict(means=averages, oracle_static=oracle, best_k1=best_k1, gates=gates,
                  recommended=recommended, physical_adoption="pending evaluator; no repack in this CPU job")
    write_json(directory / "summary.json", report)
    lines = ["# Kinematic static-placement diagnostic", "",
             "Protocol: `docs/chain/pose-placement-protocol.md`, Amendment 1. Eight public train episodes; "
             "explicit evaluation-analogue slots; one world; nearest-visible fill. K2 is supervised calibration "
             "from frame-zero public mocap, fitted leave-one-episode-out. Singleton objects keep z up. "
             "Scan samples enter scoring only (1,000 per object, seed 0).", "",
             f"**Recommendation: {recommended}.** Better K1 by mean AUC: **{best_k1}** "
             "(ties choose K1a). K2 alone is informational; the adoption chain is v1 → better K1 → K1+K2. "
             "Passing CPU gates would authorize a GPU follow-up recommendation, not physical adoption.", "",
             f"This invocation executed {executed} trials and resumed {skipped}. "
             "`ledger.jsonl` is append-only; `finished.json` pins the selected identities and run records. "
             "The ranked table below is the complete current queue; prior completed trials are reused, not retuned.", "",
             "## Equal-episode means (ranked by AUC)", "",
             "| Candidate | AUC | SP-SR | MP-SR | RPE cm | MPPE cm |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in sorted(averages, key=lambda name: averages[name]["AUC"], reverse=True):
        lines.append(f"| {name} | {metric_cells(averages[name])} |")
    lines.extend([f"| Oracle static reference | {metric_cells(oracle)} |", "",
                  "The oracle comparator reproduces AUC 0.531919, SP-SR 0.625, MP-SR 0, RPE 15.1144 cm "
                  "and MPPE 7.4033 cm. It is a reference benchmark, not an optimum over static poses.", "",
                  "## Amendment 1 gates", "",
                  "| Candidate | Predecessor | ΔAUC ≥ .02 | SP pass count | MP pass count | ΔMPPE ≤ .25 cm | Worst AUC loss ≤ .10 | Pass |",
                  "|---|---|---:|---:|---:|---:|---:|---|"])
    for name, value in gates.items():
        sp, mp = (value["success_counts"][key] for key in ("SP_SR", "MP_SR"))
        failures = ", ".join(key for key, passed in value["checks"].items() if not passed)
        lines.append(f"| {name} | {value['predecessor']} | {value['auc_gain']:+.6f} | {sp[0]} → {sp[1]} | "
                     f"{mp[0]} → {mp[1]} | {value['mppe_change_cm']:+.3f} | {value['worst_auc_loss']:.6f} | "
                     f"{'yes' if value['passed'] else 'no: ' + failures} |")
    lines.extend(["", "All eight episodes have equal weight; success rates change in steps of 1/8. "
                  "Inclusive gate comparisons allow 1e-12 for numerical roundoff only. "
                  "Both the K1 stage and cumulative stage must pass to recommend K1+K2.", "",
                  "![Per-episode AUC by candidate](per_episode_auc.png)", "", "## Per-episode metrics", "",
                  "| Episode | Candidate | AUC | SP-SR | MP-SR | RPE cm | MPPE cm |",
                  "|---:|---|---:|---:|---:|---:|---:|"])
    for index, base in enumerate(results["v1"]):
        for name, rows in results.items():
            lines.append(f"| {base['episode']} | {name} | {metric_cells(rows[index]['metrics'])} |")
        lines.append(f"| {base['episode']} | Oracle static | {metric_cells(base['oracle_static'])} |")
    lines.extend(["", "## Frame-zero relative-transform decomposition and oracle replacements", "",
                  "Errors are slot 1 relative to slot 0's body frame. Direction includes anchor-frame error; "
                  "the length ratio also includes centroid-versus-scan-origin error. Each oracle changes one "
                  "factor at frame zero and holds that pose static; joint changes all three. Full metric "
                  "vectors and per-object gravity-axis errors are in each run's `metrics.json`.", "",
                  "| Episode | Candidate | Offset error cm | Length ratio | Direction ° | Relative rotation ° | Direction oracle AUC | Length oracle AUC | Rotation oracle AUC | Joint oracle AUC | Heading ceiling AUC |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, rows in results.items():
        for row in rows:
            error, replacements = row["frame0"], row["oracle_replacements"]
            lines.append(f"| {row['episode']} | {name} | {error['offset_error_cm']:.3f} | {error['distance_ratio']:.3f} | "
                         f"{error['direction_error_deg']:.2f} | {error['relative_rotation_error_deg']:.2f} | "
                         + " | ".join(f"{replacements[key]['AUC']:.6f}" for key in ("direction", "length", "rotation", "joint"))
                         + f" | {row['heading_flip']['best']['AUC']:.6f} |")
    lines.extend(["", "## Diagnostic ceilings (eight-episode means)", "",
                  "| Candidate | Direction oracle AUC | Length oracle AUC | Rotation oracle AUC | Joint oracle AUC | Best heading AUC |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, rows in results.items():
        values = [np.mean([row["oracle_replacements"][key]["AUC"] for row in rows]) for key in ("direction", "length", "rotation", "joint")]
        values.append(np.mean([row["heading_flip"]["best"]["AUC"] for row in rows]))
        lines.append(f"| {name} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines.extend(["", "## Cross-fitted K2 body-up calibration", "",
                  "Signed unit vectors are in each object's body frame; the held-out episode never fits its own convention. "
                  "No other train occurrence means z-up fallback. The errors compare body gravity directions "
                  "without assuming a shared world heading.", "",
                  "| Held-out episode | Object | Fitting episodes | Fitted body-up (x, y, z) | Up error ° |",
                  "|---:|---|---|---|---:|"])
    for row in results["K2"]:
        for index, name in enumerate(row["names"]):
            fold = row["calibration"][name]
            vector = ", ".join(f"{value:+.4f}" for value in fold["up"])
            fitting = ", ".join(map(str, fold["episodes"])) or "singleton fallback"
            lines.append(f"| {row['episode']} | {name} | {fitting} | ({vector}) | "
                         f"{row['frame0']['up_axis_error_deg'][index]:.2f} |")
    lines.extend(["", "The heading ceiling selects the better AUC of as-built slot-0 heading and its π-flip "
                  "per episode; this uses reference outcomes and is never a candidate. Joint replacement "
                  "reconstructs the oracle static relative transform and should match the oracle reference.", "",
                  "## Limits and follow-up", "",
                  "Kinematic only: no FlashCHORD evaluation, reset, settling, collision dynamics or packing was run. "
                  "K2 records the inverse body-frame rotation for mesh vertices and collision-box orientation, "
                  "and the full rotated inertia tensor; geometry and inertia remain physically unchanged. "
                  "These transforms are diagnostic artifacts, not exported simulator assets. "
                  "No scan dimensions or scan-origin corrections enter placement.", "",
                  "K3 is omitted because no camera-to-mocap transform is released. Gravity allows signed "
                  "body-up comparisons, but direct heading errors are unobservable. PCA heading remains "
                  "ambiguous; pot support signs disagree between its two train episodes. Board identities "
                  "remain arbitrary. K1 uses image-recorded upper slots: lid over pot (0), pot over rack (26), "
                  "sweeper over rack (31); these are box approximations, not containment models.", "",
                  "Before any repack, evaluate every affected scored episode: first export within 1 cm / 2° "
                  "of intended placement and drift within the slice limit. Fit any submitted K2 convention "
                  "on all eight train episodes; wooden_spoon keeps z up. No excluded episodes were used.", ""])
    (directory / "summary.md").write_text("\n".join(lines))
    plot(results, directory)
    return report


def plot(results, directory):
    """Draw the predeclared standalone and cumulative per-episode AUC comparison."""
    os.environ["MPLCONFIGDIR"] = str(directory / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(13, 5), layout="constrained")
    episodes = [row["episode"] for row in results["v1"]]
    x = np.arange(len(episodes))
    width = .13
    for i, (name, rows) in enumerate(results.items()):
        axis.bar(x + (i - 2) * width, [row["metrics"]["AUC"] for row in rows], width, label=name)
    axis.plot(x, [row["oracle_static"]["AUC"] for row in results["v1"]], "kD--", label="Oracle static reference")
    axis.set(xticks=x, xticklabels=episodes, xlabel="Public train episode (analogue slot order)",
             ylabel="AUC (1,000 scan samples/object; seed 0)", ylim=(0, 1),
             title="Kinematic static placement: standalone and cumulative candidates")
    axis.legend(ncol=3, fontsize=9)
    axis.grid(axis="y", alpha=.2)
    figure.savefig(directory / "per_episode_auc.png", dpi=170)
    plt.close(figure)
