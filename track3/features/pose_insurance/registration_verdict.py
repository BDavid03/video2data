"""Record HR2 panel verdicts: python -B -m track3.features.pose_insurance.registration_verdict."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from track3.paths import output_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, new_run_id,
                         producer, read_json, recorded)
from .registration_loop import CODE_FILES, FEATURE, environment

OUTPUT = 'outputs/pose_insurance/registration-hr2-verdict'
QUEUE = 'track3/features/pose_insurance/registration_verdict_queue.json'
SOURCES = (*CODE_FILES, 'track3/features/pose_insurance/registration_verdict.py',
           'track3/features/pose_insurance/_registration_verdict.py',
           'configs/sources.lock.json')


def frozen_context(queue: dict, record: dict) -> tuple:
    """Admit the pinned method, protocol, verdict rules and evaluation roster."""
    if {'pairs', 'mandatory_pairs', 'pair_thresholds', 'up_thresholds'} & queue.keys():
        raise ValueError('Verdict rules belong exclusively in the pinned spec')
    for key in ('method', 'protocol', 'evaluation_queue', 'verdict_spec'):
        checked_entry(queue[key], record)
    method = read_json(queue['method']['path'])
    evaluation = read_json(queue['evaluation_queue']['path'])
    spec = read_json(queue['verdict_spec']['path'])
    if spec['protocol'] != queue['protocol']:
        raise PermissionError('Verdict spec cites a different protocol')
    current = {r['path']: r['sha256'] for r in record['sources']}
    if any(current.get(path) != digest for path, digest in method['code_sha256'].items()):
        raise PermissionError('Frozen HR2 source hashes differ')
    if queue['roster'] != method['evaluation_slots'] or queue['roster'] != evaluation['trials']:
        raise PermissionError('Verdict roster differs from the frozen evaluation roster')
    slots = {(r['episode'], r['slot']) for r in queue['roster']}
    episodes = [r['episode'] for r in spec['pairs']]
    if (len(set(episodes)) != len(episodes) or not set(spec['mandatory_pairs']) <= set(episodes)
            or any(len(p['slots']) != 2 or len(set(p['slots'])) != 2
                   or any((p['episode'], s) not in slots for s in p['slots']) for p in spec['pairs'])):
        raise ValueError('Pair panel must name distinct roster slots and contain mandatory pairs')
    return method, evaluation, spec


def admit_run(entry: dict, kind: str, record: dict, provenance: str) -> tuple:
    """Admit a completed typed producer and its externally recorded content hash."""
    checked_entry(entry, record)
    return producer(entry['path'], kind, allowed=(provenance,))


def saved_panel(queue: dict, evaluation: dict, method: dict, record: dict) -> tuple:
    """Bind the phase-2 errors, report and every prediction to their producer records."""
    report = admit_run(queue['prediction_report'], 'pose_insurance.registration_report',
                       record, 'public_asset_diagnostic')
    errors = admit_run(queue['error_report'], 'pose_insurance.registration_errors',
                       record, 'public_label_diagnostic')
    if (report[1]['request'] != evaluation or errors[1]['request'] != dict(
            prediction_report=queue['prediction_report']['path'])
            or queue['prediction_report'] not in errors[1]['inputs']):
        raise ValueError('Phase-2 producers do not describe the same frozen evaluation')
    if datetime.fromisoformat(report[1]['finished_utc']) >= datetime.fromisoformat(errors[1]['started_utc']):
        raise ValueError('Predictions must be completed before label errors start')
    summary = read_json(artifact(report, 'summary.json', record))
    error_data = read_json(artifact(errors, 'errors.json', record))
    if (summary['phase'] != 'evaluation' or error_data['phase'] != 'evaluation'
            or summary['method_sha256'] != queue['method']['sha256']
            or summary['parameters'] != method['parameters']
            or error_data['parameters'] != method['parameters']):
        raise ValueError('Phase or method differs from the frozen evaluation')
    roster = [{key: r[key] for key in ('episode', 'slot', 'name')} for r in summary['predictions']]
    if roster != queue['roster']:
        raise ValueError('Prediction report must contain the whole ordered roster')
    for row in summary['predictions']:
        entry = next(e for e in report[1]['inputs'] if e['path'] == row['run'])
        trial = admit_run(entry, 'pose_insurance.registration_trial', record, 'public_asset_diagnostic')
        saved = read_json(artifact(trial, 'prediction.json', record))
        if (saved != {k: v for k, v in row.items() if k not in ('run', 'resumed')}
                or trial[1]['request'] != dict(trial={k: row[k] for k in ('episode', 'slot', 'name')},
                                               parameters=method['parameters'])
                or trial[1]['method_sha256'] != queue['method']['sha256']):
            raise ValueError('Report differs from its frozen trial prediction')
    up_roster = [{k: r[k] for k in ('episode', 'slot', 'name')} for r in error_data['up']]
    if up_roster != roster or any(a['accepted'] != b['accepted'] for a, b in zip(
            error_data['up'], summary['predictions'])):
        raise ValueError('Up errors differ from the saved prediction roster or decisions')
    return summary['predictions'], error_data, errors


def load_references(spec: dict, predictions: list, errors: tuple, record: dict) -> dict:
    """Read only the configured pairs' label files after checking their historical hashes."""
    from .placement_diagnostic import baseline_module, label_paths
    by_slot = {(r['episode'], r['slot']): r for r in predictions}
    references = {}
    for pair in spec['pairs']:
        episode = pair['episode']
        for path in label_paths(episode):
            relative = str(path.relative_to(ROOT))
            entry = next(e for e in errors[1]['inputs'] if e['path'] == relative)
            checked_entry(entry, record)
        names, poses = baseline_module().load_reference(episode)
        references[episode] = poses[0, 0, [names.index(by_slot[episode, s]['name']) for s in pair['slots']]]
    return references


def markdown(result: dict, queue: dict) -> str:
    """Format the recorded verdict and descriptive pair diagnostics with explicit units."""
    pair, up = result['pair_panel'], result['up_panel']
    spec, rules = result['verdict_spec'], result['rules_applied']
    pair_limits, up_limits = rules['pair_thresholds'], rules['up_thresholds']
    lines = ['# Recorded HR2 verdict', '',
        f"Overall: {'PASS' if result['overall_passed'] else 'FAIL'}. "
        f"Pair panel: {'PASS' if pair['passed'] else 'FAIL'} "
        f"({pair['accepted_pairs']}/{len(rules['pairs'])} accepted). "
        f"Up panel: {'PASS' if up['passed'] else 'FAIL'} "
        f"({up['accepted_slots']}/{len(queue['roster'])} accepted; "
        f"minimum {up_limits['minimum_accepted_slots']}).", '',
        f"Verdict spec: `{spec['path']}`; SHA-256 `{spec['sha256']}`.", '',
        f"Pair rules: panel {', '.join(str(p['episode']) for p in rules['pairs'])}; "
        f"mandatory pairs {', '.join(map(str, rules['mandatory_pairs']))} must be accepted; "
        f"every accepted pair must have b < {pair_limits['b_error_deg']} deg "
        f"and C < {pair_limits['C_error_deg']} deg.", '',
        f"Up rules: >= {up_limits['minimum_accepted_slots']} accepted slots across "
        f">= {up_limits['minimum_episodes']} episodes and "
        f">= {up_limits['minimum_object_types']} object types; "
        f"every accepted up error < {up_limits['up_error_deg']} deg. Both panels must pass.", '',
        'Retrospective recorded calculation over frozen predictions and disclosed labels; '
        'this is not a new registration or an untouched holdout assessment. '
        'Declined-pair errors describe saved selected rotations only.', '',
        '| Episode | Accepted | b error (deg) | C error (deg) | Selected offset (cm) | Reference offset (cm) |',
        '|---:|:---:|---:|---:|---:|---:|']
    for row in result['pairs']:
        lines.append(f"| {row['episode']} | {row['accepted']} | {row['b_error_deg']:.3f} | "
            f"{row['C_error_deg']:.3f} | {row['predicted_offset_cm']:.3f} | {row['reference_offset_cm']:.3f} |")
    lines += ['', f"Accepted up coverage: {len(up['accepted_episodes'])} episodes, "
        f"{len(up['accepted_object_types'])} object types; "
        f"maximum up error {up['max_accepted_up_error_deg']:.6f} deg.", '']
    return '\n'.join(lines)


def run(queue_path: str = QUEUE) -> Path:
    """Admit frozen evidence and compute a supplemental verdict inside a new run record."""
    runtime = environment()
    queue = read_json(queue_path)
    record = run_fields(ROOT, 'pose_insurance', 'registration_verdict', queue,
        inputs=[queue_path, FEATURE / 'runtime.json'], provenance='public_label_diagnostic')
    record['sources'] = [input_file(ROOT, path) for path in SOURCES]
    directory = output_path(ROOT, f'{OUTPUT}/{new_run_id()}')
    with recorded(directory, record):
        method, evaluation, spec = frozen_context(queue, record)
        for row in queue['roster']:
            authorize_episode(runtime, SimpleNamespace(source_split='public', episode_index=row['episode']),
                              'HR2 recorded verdict')
        predictions, error_data, errors = saved_panel(queue, evaluation, method, record)
        references = load_references(spec, predictions, errors, record)
        from ._registration_verdict import panel_verdict, selected_pairs
        pairs = selected_pairs(predictions, references, spec['pairs'])
        result = dict(method_sha256=queue['method']['sha256'],
            prediction_report=queue['prediction_report']['path'], error_report=queue['error_report']['path'],
            verdict_spec=queue['verdict_spec'], rules_applied=spec,
            up=error_data['up'], pairs=pairs, **panel_verdict(error_data['up'], pairs, spec))
        write_json(directory / 'verdict.json', result)
        (directory / 'summary.md').write_text(markdown(result, queue))
    print(json.dumps(dict(run=str((directory / 'run.json').relative_to(ROOT)),
                          overall_passed=result['overall_passed'])), flush=True)
    return directory / 'run.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default=QUEUE)
    run(parser.parse_args().queue)


if __name__ == '__main__':
    main()
