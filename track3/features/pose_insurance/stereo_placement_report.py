"""Publish T1 train metrics, integrity controls and horizontal direction diagnostics."""
import shutil
from track3.provenance import input_file, run_fields, write_json
from track3.runs import ROOT, new_run_id, read_json, recorded
import numpy as np
from ._shrink import EPISODES, CONTROLS, METRICS, comparison, means
from ._scale import candidate_gates
from ._stereo_placement import offset_comparison

DISCLOSURE = (
    'The idea came from a train diagnostic. Fresh confirmation failed by the letter; the owner '
    'approved this candidate anyway. Train gates were fixed after frame-zero errors were known '
    'and are not independent confirmation. Expected gains are RPE/MPPE; direction remains '
    'unobservable. Episode 12 uses its observed slot-zero mask for the stereo offset, retaining '
    'the S1 default slot-zero position and rotation. Mocap and scans enter scoring and description '
    'only. AUC uses 1,000 scan surface samples per object, seed zero, approximating host sampling. '
    'Only train 0/11/12/18/21/23/26/31 and controls 39/42 are read. No scored stage has run.'
)


def analysis(results: dict, plans: dict, historical: dict, lengths: list) -> dict:
    """Apply fixed-denominator integrity controls and the existing candidate gates."""
    reproduction = {str(ep): comparison(historical['S1'][str(ep)]['metrics'], results['S1'][ep]['metrics'])
                    for ep in (*EPISODES, *CONTROLS)}
    offsets = {str(row['episode']): offset_comparison(plans[row['episode']]['raw_offset_m'], row['vectors_m']['V_stereo'])
               for row in lengths if row['episode'] in EPISODES}
    positions = {str(ep): float(np.max(np.abs(np.asarray(plans[ep]['scenes']['S1']['positions']) -
                          historical['S1'][str(ep)]['scene']['positions']))) for ep in (*EPISODES, *CONTROLS)}
    invariance = {str(ep): comparison(results['S1'][ep]['metrics'], results['T1'][ep]['metrics']) for ep in CONTROLS}
    finite = all(np.isfinite(row['metrics'][key]) for rows in results.values() for row in rows.values() for key in METRICS)
    integrity = dict(s1_reproduction=all(r['passed'] for r in reproduction.values()),
                     offset_reproduction=len(offsets) == len(EPISODES) and all(r['passed'] for r in offsets.values()),
                     s1_position_reproduction=all(value <= 1e-6 for value in positions.values()),
                     single_object_invariance=all(r['passed'] for r in invariance.values()),
                     unchanged_fields=all(p['unchanged_fields'] for p in plans.values()), finite_metrics=finite)
    scores = {name: means(rows, EPISODES) for name, rows in results.items()}
    gates = candidate_gates([results['S1'][ep]['metrics'] for ep in EPISODES],
                            [results['T1'][ep]['metrics'] for ep in EPISODES])
    verdict = ('blocked: integrity control failed' if not all(integrity.values()) else
               'T1 goes to the scored stage' if gates['passed'] else 'D2 v3 stays')
    ranking = sorted(scores, key=lambda name: scores[name]['MPPE_cm'])
    return dict(means=scores, ranking=ranking, gates=gates, verdict=verdict, integrity=integrity,
                reproduction=reproduction, offset_reproduction=offsets, s1_position_difference_m=positions,
                invariance=invariance)


def metric_table(rows: list) -> list:
    """Render all five official metrics with their units."""
    lines = ['| Episode / subset | Candidate | AUC | SP-SR | MP-SR | RPE (cm) | MPPE (cm) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for label, candidate, metrics in rows:
        values = ' | '.join(f'{metrics[key]:.6f}' for key in METRICS)
        lines.append(f'| {label} | {candidate} | {values} |')
    return lines


def markdown(report: dict) -> str:
    """Render the verdict beside its fixed gates, clearance and direction evidence."""
    lines = ['# T1 stereo-placement train result', '', f"**{report['verdict']}**", '', DISCLOSURE, '',
             f"This invocation: {report['executed']} new trials; {report['resumed']} resumed trials.", '',
             '## Ranked equal-episode means', '', 'Eight train episodes, ranked descriptively by MPPE.', '']
    lines += metric_table([('train mean', name, report['means'][name]) for name in report['ranking']])
    lines += ['', '## Per-episode metrics', '']
    lines += metric_table([(str(ep) + (' (control)' if ep in CONTROLS else ''), name,
                            report['episodes'][name][ep]['metrics'])
                           for ep in (*EPISODES, *CONTROLS) for name in ('S1', 'T1')])
    g = report['gates']
    lines += ['', '## Train gates', '', '| Gate | Observed | Required | Passed |', '|---|---|---|---|',
              f"| Mean errors | RPE gain {g['rpe_gain_cm']:.9f} cm; MPPE gain {g['mppe_gain_cm']:.9f} cm | ≥2 cm; ≥1 cm | {g['checks']['mean_errors']} |",
              f"| Guarded means | {g['guarded_changes']} | each ≥−1e-12 | {g['checks']['guarded_means']} |",
              f"| Episode safety | max AUC loss {g['worst_auc_loss']:.9f}; max MPPE increase {g['worst_mppe_increase_cm']:.9f} cm | ≤0.02; ≤0.5 cm | {g['checks']['episode_safety']} |",
              '', '## Integrity controls', '', '| Control | Passed |', '|---|---|']
    lines += [f'| {name} | {passed} |' for name, passed in report['integrity'].items()]
    lines += ['', 'Full-precision metric differences, position differences and offset missingness are in `summary.json`.', '',
              '## Clearance frozen before scoring', '',
              '| Episode | Factor c | Raw gap (cm) | Applied gap (cm) | Result / reason |', '|---:|---:|---:|---:|---|']
    for ep, plan in report['construction'].items():
        row = plan['clearance']
        raw_gap = '—' if 'raw_clearance_m' not in row else f"{100 * row['raw_clearance_m']:.6f}"
        gap = '—' if 'clearance_m' not in row else f"{100 * row['clearance_m']:.6f}"
        factor = '—' if row['factor'] is None else f"{row['factor']:.9f}"
        lines.append(f"| {ep} | {factor} | {raw_gap} | {gap} | {row['status']}: {row['reason']} |")
    lines += ['', '## Horizontal direction reading', '',
              'Angles are predicted minus aligned reference (positive counterclockwise). Reference offsets '
              'are transported into the predicted table frame as in `_length.ray_errors`. Zero offsets have '
              'undefined angles. Threshold flags compare horizontal position error strictly against the kit’s '
              '1–9 cm thresholds; these are not ADD/AUC success flags. Angle tolerance assumes equal lengths. '
              'Raw exact-direction errors are missing when no independent stereo offset exists.', '',
              '| Episode | Candidate | Angle (deg) | Error (cm) | Inside thresholds (cm) | Raw stereo exact-direction error (cm) | Equal-length 9 cm tolerance (deg) |',
              '|---:|---|---:|---:|---|---:|---:|']
    for ep in EPISODES:
        for name in ('S1', 'T1'):
            row = report['episodes'][name][ep]['direction']
            angle = 'undefined' if row['signed_angle_deg'] is None else f"{row['signed_angle_deg']:.3f}"
            exact = '—' if row['raw_exact_direction_error_cm'] is None else f"{row['raw_exact_direction_error_cm']:.3f}"
            inside = ', '.join(key for key, value in row['inside_position_thresholds'].items() if value) or 'none'
            tolerance = row['equal_length_angle_tolerance_deg']
            lines.append(f"| {ep} | {name} | {angle} | {row['horizontal_error_cm']:.3f} | {inside} | {exact} | {tolerance:.3f} |")
    lines += ['', '## Artifacts', '', '![Train RPE and MPPE](metrics.png)', '',
              '`summary.json`: full-precision results and controls. `builds/*/construction.json`: video-only '
              'scenes and clearance brackets. `runs/*/{metrics,run}.json`: recorded trials. `ledger*.jsonl`: '
              'append-only execution history. `reports/`: immutable report snapshots. `finished.json`: latest invocation.', '']
    return '\n'.join(lines)


def figure(report: dict, path) -> None:
    """Plot paired train RPE and MPPE with episode labels and centimetre units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(EPISODES))
    for ax, key, title in zip(axes, ('RPE_cm', 'MPPE_cm'), ('Relative position error', 'Mean per-point position error')):
        for name, shift, color in (('S1', -.18, '#71717a'), ('T1', .18, '#0284c7')):
            values = [report['episodes'][name][ep]['metrics'][key] for ep in EPISODES]
            ax.bar(x + shift, values, .36, label=name, color=color)
        ax.set(xticks=x, xticklabels=EPISODES, xlabel='Train episode', ylabel='Error (cm)', title=title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle('T1 stereo placement versus S1 — static kit diagnostic, one world')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def publish(results, plans, selected, executed, resumed, queue_path, directory) -> None:
    """Record an immutable report and publish its current summary artifacts."""
    paths = ['docs/chain/pose-stereo-placement-protocol.md', queue_path,
             'outputs/pose_insurance/scale/summary.json', 'outputs/pose_insurance/length/summary.json']
    report = analysis(results, plans, read_json(paths[2])['episodes'], read_json(paths[3])['episodes'])
    report.update(episodes=results, construction=plans, executed=executed, resumed=resumed, disclosure=DISCLOSURE)
    record = run_fields(ROOT, 'pose_insurance', 'stereo_placement_report', dict(queue=queue_path),
                        inputs=paths, parents=[row['run'] for row in selected], provenance='public_label_diagnostic')
    snapshot = directory / 'reports' / new_run_id()
    with recorded(snapshot, record):
        write_json(snapshot / 'summary.json', report)
        (snapshot / 'summary.md').write_text(markdown(report))
        figure(report, snapshot / 'metrics.png')
    for name in ('summary.json', 'summary.md', 'metrics.png'):
        shutil.copyfile(snapshot / name, directory / name)
    write_json(directory / 'finished.json', dict(status='completed', verdict=report['verdict'],
               executed=executed, resumed=resumed, selected=selected,
               report=str((snapshot / 'run.json').relative_to(ROOT)), queue=input_file(ROOT, queue_path)))
