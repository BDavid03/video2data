"""Present the registered shrink means, controls, verdicts and conditional H gates."""
import json
import numpy as np
from src.provenance import write_json
from src.runs import ROOT, read_json
from ._scale import candidate_gates
from ._shrink import (EPISODES, CHANGED, CONTROLS, CANDIDATES, METRICS, comparison,
                      means, verdicts, oracle_reading, length_reading)


def controls(results: dict) -> dict:
    """Disqualify only the candidate whose single-object metrics change or are missing."""
    checks = {}
    for c, rows in results.items():
        checks[c] = {}
        for ep in CONTROLS:
            try:
                checks[c][ep] = comparison(results['I'][ep]['metrics'], rows[ep]['metrics'])
            except (KeyError, TypeError):
                checks[c][ep] = dict(passed=False, reason='Missing comparator')
    return {c: dict(passed=all(r['passed'] for r in rows.values()), episodes=rows) for c, rows in checks.items()}


def reproduction(results: dict) -> dict:
    """Compare both registered rows against the full-precision scale summary JSON."""
    old = read_json('data/outputs/pose_insurance/scale/summary.json')['episodes']
    checks = {}
    for c, name in (('I', 'incumbent'), ('S1', 'S1')):
        for ep in (*EPISODES, *CONTROLS):
            try:
                checks[f'{c}:{ep}'] = comparison(old[name][str(ep)]['metrics'], results[c][ep]['metrics'])
            except (KeyError, TypeError):
                checks[f'{c}:{ep}'] = dict(passed=False, reason='Missing reproduction row')
    return dict(passed=all(r['passed'] for r in checks.values()), comparisons=checks,
                scope='I and S1, all 8 train and both controls; five full-precision metrics; absolute tolerance 1e-9')


def analysis(results: dict, plan: dict) -> dict:
    """Keep complete denominators and propagate each integrity failure to its dependants."""
    scores8, scores6, missing = {}, {}, {}
    for c, rows in results.items():
        try:
            scores8[c], scores6[c] = means(rows, EPISODES), means(rows, CHANGED)
        except (KeyError, TypeError, ValueError) as error:
            missing[c] = str(error)
    reproduced = reproduction(results)
    invariant = controls(results)
    eligible = {c: c in scores8 and invariant[c]['passed'] for c in results}
    decisions = verdicts(scores8, eligible, reproduced['passed'], plan['s_bar'])
    return dict(means8=scores8, means6=scores6, reproduction=reproduced, invariance=invariant,
                eligible=eligible, missing=missing, verdicts=decisions)


def descriptive(report: dict, oracles: dict, stereo: dict, plan: dict, directory) -> None:
    """Write the oracle and all 40,320 direction-free assignments without selecting a scene."""
    if not report['reproduction']['passed'] or set(oracles) != set(CHANGED):
        report['verdicts']['3'] = dict(label='blocked: reproduction or missing oracle')
        report['verdicts']['4'] = dict(label='blocked: reproduction or missing length diagnostic')
        return
    report['verdicts']['3'] = oracle_reading(oracles, stereo)
    if not all(report['eligible'].get(c, False) for c in ('I', 'S1', 'G')):
        report['verdicts']['4'] = dict(label='blocked: missing, non-finite or disqualified length comparator')
        return
    offsets = np.array([oracles[e]['incumbent_offset_m'] for e in CHANGED])
    distances = np.array([oracles[e]['mocap_distance_m'] for e in CHANGED])
    report['verdicts']['4'], arrays = length_reading(offsets, distances, plan)
    np.savez_compressed(directory / 'length_assignments.npz', **arrays)


def h_gates(results: dict, report: dict) -> dict:
    """Apply the existing registered train gates against S1 and add H's control gate."""
    if not all(report['eligible'].get(c, False) for c in ('H', 'S1')) or not report['reproduction']['passed']:
        return dict(passed=False, reason='blocked: integrity or missing H/S1 comparator')
    gates = candidate_gates([results['S1'][e]['metrics'] for e in EPISODES],
                            [results['H'][e]['metrics'] for e in EPISODES])
    gates['checks']['single_object_invariance'] = report['invariance']['H']['passed']
    gates['passed'] = all(gates['checks'].values())
    return gates


def summary_changes(current: dict, previous_path) -> dict:
    """Compare scientific results with the backed-up summary, excluding invocation bookkeeping."""
    if previous_path is None:
        return dict(previous_summary=None, changed_sections=[], note='First summary; no previous results.')
    previous = read_json(previous_path)
    ignored = {'executed', 'resumed', 'failures', 'changes_since_previous'}
    sections = sorted((current.keys() | previous.keys()) - ignored)
    changed = [key for key in sections if current.get(key) != previous.get(key)]
    return dict(previous_summary=str(previous_path.relative_to(ROOT)), changed_sections=changed,
                note=('Scientific results changed: ' + ', '.join(changed) + '.' if changed
                      else 'All metrics, verdicts, gates, assignments, scenes and oracle readings are unchanged.'),
                previous_invocation={key: previous.get(key) for key in ('executed', 'resumed', 'failures')},
                current_invocation={key: current.get(key) for key in ('executed', 'resumed', 'failures')})


def summarize(report: dict, results: dict, oracles: dict, plan: dict, directory, previous_path=None) -> None:
    """Write a compact ranked summary and full numerical evidence beside it."""
    summary = json.loads(json.dumps(dict(report, assignments=plan, episodes=results, oracle_runs=oracles)))
    changes = summary_changes(summary, previous_path)
    summary['changes_since_previous'] = changes
    write_json(directory / 'summary.json', summary)
    lines = ['# Pose shrink control', '',
             'Static kinematic diagnostic against public mocap: official kit, one world, nearest-visible fill, '
             '1,000 scan samples/object, seed 0. No physical evaluator, scored H stage or D2 change.', '',
             f"Reproduction: **{report['reproduction']['passed']}** (I/S1, all ten episodes, five metrics, tolerance 1e-9).",
             f"Global factor: **{plan['s_bar']:.15f}** (registered 0.856952, tolerance 1e-6).",
             f"Invocation: {report['executed']} executed, {report['resumed']} resumed. "
             'Assignments and H construction were fsynced before their scoring.', '',
             '## What changed since the last summary', '', changes['note'], '',
             f"Previous summary: `{changes['previous_summary']}`.", '',
             '## Equal-episode means', '',
             '| Candidate | N | AUC | SP-SR | MP-SR | RPE cm | MPPE cm | Eligible |',
             '|---|---:|---:|---:|---:|---:|---:|:---:|']
    for c in sorted(report['means8'], key=lambda c: report['means8'][c]['MPPE_cm']):
        for n, key in ((8, 'means8'), (6, 'means6')):
            values = ' | '.join(f'{report[key][c][k]:.9f}' for k in METRICS)
            lines.append(f"| {c} | {n} | {values} | {report['eligible'][c]} |")
    lines += ['', 'N=8: 0/11/12/18/21/23/26/31. N=6: 11/12/18/21/23/31. '
              'Stacked 0/26 are unchanged. C0 is a physically implausible kinematic bound.', '',
              '## Registered verdicts', '']
    v = report['verdicts']
    lines += [f"1. **{v['1']['label']}**. G minus S1: "
              f"{v['1'].get('global_minus_S1_mppe_cm', 'blocked')} cm MPPE; "
              f"{v['1'].get('global_minus_S1_rpe_cm', 'blocked')} cm RPE. "
              f"S1 no worse than {v['1'].get('S1_no_worse_than_shifts', 'blocked')}/7 shifts. "
              'Margins: 0.25 cm MPPE, 0.50 cm RPE, 6/7 shifts.', '',
              f"2. **{v['2']['label']}**. Best factor {v['2'].get('c_best', 'blocked')} "
              f"({v['2'].get('C_best', 'blocked')}); S1 minus best MPPE "
              f"{v['2'].get('S1_minus_best_mppe_cm', 'blocked')} cm. Requires factor ≤0.50 and gain ≥1 cm.", '',
              '3. **Oracle reading (descriptive)**. '
              f"Median c*: {v['3'].get('median_c_star', 'blocked')}; range: {v['3'].get('range_c_star', 'blocked')}; "
              f"Spearman(s, full-distance ratio): {v['3'].get('spearman_s_length_ratio', 'undefined / blocked')}.", '',
              '| Episode | c* | Stereo s | Mocap / incumbent full distance |', '|---:|---:|---:|---:|']
    for ep, row in v['3'].get('episodes', {}).items():
        lines.append(f"| {ep} | {row['c_star']:.2f} | {row['s']:.9f} | {row['length_ratio']:.9f} |")
    length = v['4']
    lines += ['', f"4. **{length['label']}**. Six-episode length errors (cm): "
              + ', '.join(f'{c}={value:.9f}' for c, value in length.get('means_cm', {}).items()) + '. '
              f"S1 rank {length.get('S1_rank', 'blocked')}/40320, {length.get('exact_ties', 'blocked')} exact ties; "
              'competition rank = 1 + strictly better assignments. Requires S1 at least 0.5 cm below G and best 5%.', '',
              '## Integrity and conditional H', '']
    maximum = max((abs(value) for row in report['reproduction']['comparisons'].values()
                   for value in row.get('differences', {}).values()), default=None)
    lines += [f'Maximum reproduction difference: {maximum}. Single-object 39/42 controls: '
              + ', '.join(f"{c}={row['passed']}" for c, row in report['invariance'].items()) + '.', '',
              f"H: {report['H']['status']}.", '']
    if 'construction' in report['H']:
        h, g = report['H']['construction'], report['H']['gates']
        lines += [f"c_H={h['c_H']}; registered train gates pass: **{g['passed']}**.", '',
                  '| Episode | c_sep | Applied factor | Construction |', '|---:|---:|---:|---|']
        for ep, row in h['episodes'].items():
            lines.append(f"| {ep} | {row['c_sep']} | {row['applied_factor']} | {row['status']} |")
        lines += ['', f"H gains: RPE {g.get('rpe_gain_cm', 'blocked')} cm (≥2), "
                  f"MPPE {g.get('mppe_gain_cm', 'blocked')} cm (≥1). "
                  f"Guarded mean changes: {g.get('guarded_changes', 'blocked')} (≥−1e-12). "
                  f"Worst AUC loss {g.get('worst_auc_loss', 'blocked')} (≤0.02); "
                  f"worst MPPE increase {g.get('worst_mppe_increase_cm', 'blocked')} cm (≤0.5). "
                  f"Checks: {g.get('checks', 'blocked')}.", '']
    lines += ['## Limits and artifacts', '',
              'Seven shifts are descriptive controls, not independent replications. Wrong calibration, '
              'direction error and static-rollout mismatch can favour shrinkage; no causal conclusion follows. '
              'Mocap/scans enter diagnostics and scoring only. No episodes 41/43/44 read.', '',
              'Full metrics, deltas, eligibility, failures and readings: `summary.json`; recorded trials and '
              '151-point oracle grids: `data/runs/*/{metrics,run}.json`; durable ledger: `ledger.jsonl`; '
              'assignments: `assignments.json`; all 8! length errors and permutation indices: '
              '`length_assignments.npz`; conditional H factors and clearance brackets: `H_construction.json`.', '']
    (directory / 'summary.md').write_text('\n'.join(lines))
