"""Report the pre-registered Amendment 2 comparison without changing its gates."""
from track3.provenance import write_json


def summarize(results, observations, directory, executed, skipped):
    """Require all five metric conditions on every measurable confirmation episode."""
    by_candidate = {name: {row["episode"]: row for row in rows} for name, rows in results.items()}
    verdicts = {}
    lines = ["# K1a confirmation: Amendment 2", "",
             "The original eight-episode gates failed: all candidates tied at AUC 0.268844 "
             "because slot 1 stayed beyond the AUC thresholds. K1a reduced RPE/MPPE, which motivated "
             "the post-result Amendment 2. That amendment pre-registered public 2 and 27 before their "
             "depth, clusters or candidate scores existed. Only v1 and K1a are compared here; no tuning.", "",
             "Placements were recorded from frame-zero overlays before scoring. Analogue slot order; "
             "one world; static poses; official kit functions; 1,000 scan samples/object, seed 0. "
             "Scans enter scoring only. This is a kinematic comparison, not physical evaluation.", "",
             "| Episode | Candidate | AUC | SP-SR | MP-SR | RPE cm | MPPE cm | Frame-0 offset error cm |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for episode in (2, 27):
        observation = observations[str(episode)]
        if observation["measurable"] is not True:
            verdicts[str(episode)] = dict(measurable=False, reason=observation["note"])
            lines.append(f"| {episode} | Cannot be measured | — | — | — | — | — | — |")
            continue
        for candidate in ("v1", "K1a"):
            row = by_candidate[candidate][episode]
            metrics = row["metrics"]
            lines.append(f"| {episode} | {candidate} | {metrics['AUC']:.6f} | {metrics['SP_SR']:.3f} | "
                         f"{metrics['MP_SR']:.3f} | {metrics['RPE_cm']:.6f} | {metrics['MPPE_cm']:.6f} | "
                         f"{row['frame0']['offset_error_cm']:.6f} |")
        old, new = (by_candidate[candidate][episode]["metrics"] for candidate in ("v1", "K1a"))
        checks = {metric: new[metric] < old[metric] for metric in ("RPE_cm", "MPPE_cm")}
        checks.update({metric: new[metric] >= old[metric] - 1e-12 for metric in ("AUC", "SP_SR", "MP_SR")})
        verdicts[str(episode)] = dict(measurable=True, passed=all(checks.values()), checks=checks,
                                      v1=old, K1a=new)
    counted = [row for row in verdicts.values() if row["measurable"]]
    adopt = bool(counted) and all(row["passed"] for row in counted)
    verdict = "adopt" if adopt else "keep v1"
    lines.extend(["", f"**Amendment 2 verdict: {verdict}.** Confirmation rests on {len(counted)} episode(s).", "",
                  "RPE and MPPE must both be strictly lower, and AUC, SP-SR and MP-SR must each "
                  "be no lower on every measurable episode. The latter comparisons allow only "
                  "1e-12 numerical roundoff. No measurable episode means keep v1.", ""])
    for episode, value in verdicts.items():
        status = ("pass" if value["passed"] else "fail") if value["measurable"] else "cannot be measured"
        detail = ", ".join(key for key, passed in value.get("checks", {}).items() if not passed)
        lines.append(f"- Episode {episode}: {status}. {detail or observations[episode]['note']}")
    lines.extend(["", f"This invocation executed {executed} trials and reused {skipped}; "
                  "`confirmation_finished.json` pins the selected run records. "
                  "`placements_confirm.json` records the overlay observations and hashes.", "",
                  "An adopt verdict still requires the three scored episodes' official evaluator exports, "
                  "first-pose agreement within 1 cm / 2 degrees and the existing drift check before packing. "
                  "A keep-v1 verdict stops this job without rebuilding or repacking.", ""])
    report = dict(verdict=verdict, confirmation_episodes=len(counted), episodes=verdicts)
    write_json(directory / "confirmation.json", report)
    (directory / "confirmation.md").write_text("\n".join(lines))
    return report
