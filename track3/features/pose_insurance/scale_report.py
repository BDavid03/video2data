"""Write the fixed scale comparison, integrity checks and adoption decision."""
import numpy as np
from track3.provenance import write_json
from track3.runs import ROOT, read_json, artifact
from ._scale import candidate_gates, adoption

METRICS = ('AUC', 'SP_SR', 'MP_SR', 'RPE_cm', 'MPPE_cm')


def reproduction(results, episodes):
    """Compare newly scored incumbents with the exact recorded placement results."""
    rows = read_json('outputs/pose_insurance/placement/finished.json')['selected']
    text = (ROOT / 'outputs/pose_insurance/placement/summary.md').read_text()
    checks = {}
    for ep in episodes:
        candidate = 'K1a' if ep in (0, 26) else 'v1'
        row = next(r for r in rows if r['candidate'] == candidate and r['episode'] == ep)
        # Historical runtime.json was explicitly authorized to change; admit the owned metric artifact.
        from track3.provenance import read_run
        record = read_run(ROOT, row['run'], kind='pose_insurance.placement_diagnostic')
        old = read_json(artifact((ROOT / row['run'], record), 'metrics.json', {'inputs': []}))['metrics']
        new = results['incumbent'][ep]['metrics']
        differences = {k: new[k] - old[k] for k in METRICS}
        line = f"| {ep} | {candidate} | {old['AUC']:.6f} | {old['SP_SR']:.3f} | {old['MP_SR']:.3f} | {old['RPE_cm']:.3f} | {old['MPPE_cm']:.3f} |"
        checks[str(ep)] = dict(passed=all(v == 0 for v in differences.values()) and line in text,
                               exact_metric_differences=differences, summary_row_matches=line in text,
                               placement_run=row['run'], placement_candidate=candidate)
    return dict(passed=all(r['passed'] for r in checks.values()), episodes=checks)


def invariance(results, controls):
    """Require literal metric equality for the single-object controls, fail closed."""
    checks = {}
    for ep in controls:
        before = results['incumbent'][ep]['metrics']
        for candidate in ('S1', 'S2'):
            after = results[candidate][ep]['metrics']
            delta = {k: after[k] - before[k] for k in METRICS}
            checks[f'{ep}:{candidate}'] = dict(passed=all(v == 0 for v in delta.values()), differences=delta)
    return dict(passed=all(r['passed'] for r in checks.values()), comparisons=checks)


def figure(results, episodes, directory):
    """Plot the two positional metrics by episode with shared candidate styling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(episodes))
    for ax, metric, title in zip(axes, ('RPE_cm', 'MPPE_cm'), ('Relative position error', 'Mean per-point error')):
        for i, candidate in enumerate(('incumbent', 'S1', 'S2')):
            ax.bar(x + (i - 1) * .25, [results[candidate][ep]['metrics'][metric] for ep in episodes],
                   width=.25, label=candidate)
        ax.set(xticks=x, xticklabels=episodes, xlabel='Public train episode', ylabel='Error (cm)', title=title)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle('Stereo-anchored placement scale · static kinematic diagnostic')
    fig.savefig(directory / 'per_episode_errors.png', dpi=180)
    plt.close(fig)


def summarize(results, stereo, queue, directory, executed, skipped):
    """Write ranked means, per-episode diagnostics and each registered gate."""
    episodes, controls = queue['episodes'], queue['controls']
    means = {c: {k: float(np.mean([results[c][ep]['metrics'][k] for ep in episodes])) for k in METRICS}
             for c in queue['candidates']}
    incumbent = [results['incumbent'][ep]['metrics'] for ep in episodes]
    gates = {c: candidate_gates(incumbent, [results[c][ep]['metrics'] for ep in episodes]) for c in ('S1', 'S2')}
    reproduced, invariant = reproduction(results, episodes), invariance(results, controls)
    decision = adoption(gates, means, invariant['passed'], reproduced['passed'])
    report = dict(means=means, gates=gates, decision=decision, reproduction=reproduced,
                  invariance=invariant, stereo=stereo, executed=executed, resumed=skipped,
                  episodes={c: {str(ep): r for ep, r in rows.items()} for c, rows in results.items()})
    write_json(directory / 'summary.json', report)
    lines = ['# Stereo-anchored scale test', '', f'**Decision: {decision}.** No scored scenes, evaluator runs or packing in this job.', '',
             'Fixed protocol: `docs/chain/pose-scale-protocol.md`. Equal weight over eight train episodes; '
             'kit metrics, one world, nearest-visible reference fill, 1,000 scan surface samples/object, seed 0. '
             'Scans enter scoring only. This is a static kinematic diagnostic, not a physical evaluator result.', '',
             f'This invocation executed {executed} trials and resumed {skipped}. Ledger is append-only.', '',
             '## Stereo estimates', '', '| Episode | s | Observed / requested | Median support | Median log-ratio MAD | Valid |',
             '|---:|---:|---:|---:|---:|:---:|']
    for ep in episodes + controls:
        r = stereo[ep]
        fmt = lambda v: 'n/a' if v is None else f'{v:.6f}'
        lines.append(f"| {ep} | {fmt(r['s'])} | {r['observed_frames']} / {r['requested_frames']} | "
                     f"{fmt(r['median_support_share'])} | {fmt(r['median_log_ratio_mad'])} | {r['valid']} |")
    lines += ['', 'Exact depth and anchor run paths and frame identities: `stereo_<episode>.json` and `summary.json`.', '',
              '## Equal-episode means (ranked by MPPE, then RPE, then AUC)', '',
              '| Candidate | AUC | SP-SR | MP-SR | RPE cm | MPPE cm |', '|---|---:|---:|---:|---:|---:|']
    for c in sorted(means, key=lambda c: (means[c]['MPPE_cm'], means[c]['RPE_cm'], -means[c]['AUC'])):
        lines.append('| ' + c + ' | ' + ' | '.join(f'{means[c][k]:.9f}' for k in METRICS) + ' |')
    lines += ['', '## Registered gates', '',
              '| Candidate | RPE gain ≥2 cm and MPPE gain ≥1 cm | AUC/SP-SR/MP-SR nondecreasing | Worst AUC loss ≤.02 and MPPE increase ≤.5 cm | Pass |',
              '|---|---|---|---|---|']
    for c, g in gates.items():
        checks = g['checks']
        lines.append(f"| {c} | {checks['mean_errors']} ({g['rpe_gain_cm']:.6f}, {g['mppe_gain_cm']:.6f} cm) | "
                     f"{checks['guarded_means']} | {checks['episode_safety']} ({g['worst_auc_loss']:.9f}, "
                     f"{g['worst_mppe_increase_cm']:.6f} cm) | {g['passed']} |")
    lines += ['', '## Per-episode before/after diagnostics', '',
              '| Episode | Candidate | s | Slot-1 offset error cm | Distance ratio | AUC | SP-SR | MP-SR | RPE cm | MPPE cm |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ep in episodes:
        for c in queue['candidates']:
            r = results[c][ep]
            lines.append(f"| {ep} | {c} | {stereo[ep]['s']:.6f} | {r['frame0']['offset_error_cm']:.6f} | "
                         f"{r['frame0']['distance_ratio']:.6f} | " + ' | '.join(f'{r["metrics"][k]:.9f}' for k in METRICS) + ' |')
    lines += ['', '![Per-episode RPE and MPPE](per_episode_errors.png)', '', '## Integrity controls', '',
              f"Incumbent reproduction: **{reproduced['passed']}**. Literal equality to all five recorded metrics and "
              'the displayed placement-summary row is required, with K1a on 0/26 and v1 on the other six.', '',
              f"Single-object invariance: **{invariant['passed']}**. Literal equality is required; no tolerance added.", '',
              '| Control | Candidate | Maximum absolute metric difference | Pass |', '|---:|---|---:|---|']
    for key, r in invariant['comparisons'].items():
        ep, c = key.split(':')
        lines.append(f"| {ep} | {c} | {max(abs(v) for v in r['differences'].values()):.16g} | {r['passed']} |")
    lines += ['', '## Interpretation and limits', '',
              'The idea came from mocap distance ratios on train episodes. Selection uses train episodes because tuning '
              '39/42 have one object and cannot measure placement; this departs from PROGRAM.md section 4 as disclosed '
              'in the fixed protocol. The recorded length-only oracle AUC ceiling is 0.276705 versus 0.268844 v1. '
              'Large direction and relative-rotation errors remain.', '',
              'Sparse depth frames are processed with unchanged stereo handlers using singleton intervals; '
              'no research flag or estimator variant is enabled. Completed matching 39 records are reused. '
              'Invalid scales retain the incumbent for every candidate and enter all means. '
              'Support and log-ratio medians use observed frames; only observed frames have log-ratio MAD.', '',
              'S2 scales mesh/collision dimensions equivalently via the diagnostic box dimensions. '
              'The pinned scene builder has fixed mass 0.1 kg, with no density rule; this job preserves that mass '
              'and recomputes box inertia. No physical scene or mesh file is built. The lead must resolve the '
              'density wording before physical follow-up if adoption is recommended.', '',
              'Fail-closed: a failed integrity control blocks adoption even if numerical gates pass. '
              'No thresholds, candidates or settings were changed after scores. Physical checks and any scored-episode '
              'fallbacks remain the lead’s next stage. Episodes 41/43/44 were not accessed.', '']
    (directory / 'summary.md').write_text('\n'.join(lines))
    figure(results, episodes, directory)
    return report
