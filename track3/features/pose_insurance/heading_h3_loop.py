"""Score and report the fixed H3 constructions after scored predictions are saved."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from track3.paths import output_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, new_run_id,
                         producer, read_json, recorded)
from .registration_h3_loop import (CODE_FILES, METHOD_HASH, PROTOCOL, append, contract as registration_contract,
                                   environment, identity, resume)
from .request import Request

FEATURE = Path(__file__).parent
OUTPUT = 'outputs/pose_insurance/heading-h3'
QUEUE = 'track3/features/pose_insurance/heading_h3_queue.json'
CONTRACT_HASH = 'faf77574d78978bc3363cfdb3ea6ba59a202062b21175cd1546cbe472aaf7568'
NOTE = ('H2S-desc trials pinned the HR2 protocol at sha256 27333b23f8909d9d3a525d33b05f3902036cab748a96ed4452511b9cb590fb60; '
        'its restored bytes and the HR2 trials use 687debcf7f37ffffbfc1d423d02d54e0d293025db84ea48f94cdeeccf0bde82f. '
        'No old hash is patched; historical metric admissions verify the run and metric artifacts, '
        'without claiming that all historical source/input hashes match current files.')


def record_for(operation: str, request: dict, paths: list, label_free: bool = False) -> dict:
    """Record H3 sources and the exact consumed inputs."""
    record = run_fields(ROOT, 'pose_insurance', operation, request,
                        inputs=[PROTOCOL, FEATURE / 'runtime.json', *paths],
                        provenance='public_asset_diagnostic' if label_free else 'public_label_diagnostic')
    dependencies = (*CODE_FILES, *(f'track3/features/pose_insurance/{name}.py' for name in
        ('_heading', '_heading_h3', 'heading_h3_loop', '_registration_h3', 'registration_h3_loop',
         'request', '_stereo_placement', 'stereo_placement_loop', '_shrink', 'shrink_loop', 'scale_loop')),
        'configs/sources.lock.json', 'configs/splits/public-v1.json')
    record['sources'] = [input_file(ROOT, path) for path in dict.fromkeys(dependencies)]
    return record


def load_pair(entries: list, request: Request, names: list, record: dict) -> tuple:
    """Admit the frozen search-zero selections and all recorded starts."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    predictions, arrays, registered = [], [], []
    for slot, entry in enumerate(entries):
        path = checked_entry(entry, record)
        parent = resume(str(path), 'pose_insurance.registration_trial')
        expected = dict(episode=request.episode_index, slot=slot, name=names[slot])
        if (parent[1]['request']['trial'] != expected or parent[1]['split'] != request.source_split
                or parent[1]['method_sha256'] != METHOD_HASH):
            raise ValueError('Registration differs from the requested slot or frozen method')
        row = read_json(artifact(parent, 'prediction.json', record))
        with np.load(artifact(parent, 'registrations.npz', record), allow_pickle=False) as data:
            if 'search_0_losses_m' not in data:
                raise ValueError('H3 fails closed when a slot has no completed registration')
            values = {key: data[key].copy() for key in
                      ('search_0_rotations', 'search_0_translations', 'search_0_losses_m')}
        best = int(values['search_0_losses_m'].argmin())
        rotation, translation = values['search_0_rotations'][best], values['search_0_translations'][best]
        if (best != row['best_index'] or not np.array_equal(rotation, row['rotation'])
                or not np.array_equal(translation, row['translation_m'])):
            raise ValueError('Recorded selected rotation differs from search zero')
        predictions.append(row)
        arrays.append(values)
        registered.append(np.r_[translation, Rotation.from_matrix(rotation).as_quat()])
    return predictions, arrays, np.array(registered)


def episode19(queue: dict, queue_path: str, root: Path, reg_queue: dict) -> tuple:
    """Save the label-free TW scene comparison before any train reference is loaded."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from ._heading_h3 import relative_change, table_world
    from ._stereo_placement import start_poses
    reg_queue_path = queue['registration_queue']['path']
    request = Request(source_split=reg_queue['source_split'], episode_index=reg_queue['trials'][0]['episode'])
    names = [trial['name'] for trial in reg_queue['trials']]
    latest_path = ROOT / 'outputs/pose_insurance/registration-h3/latest.json'
    parent = resume(read_json(latest_path)['report'], 'pose_insurance.registration_h3_report')
    summary = read_json(artifact(parent, 'summary.json', {'inputs': []}))
    if parent[1]['request'] != reg_queue or not summary['label_free']:
        raise ValueError('Expected the complete label-free scored registration report')
    record = record_for('heading_h3_scored_pose', dict(episode=request.episode_index),
                        [queue_path, reg_queue_path, parent[0]], label_free=True)
    directory = root / 'scored-pose' / new_run_id()
    with recorded(directory, record):
        entries = [input_file(ROOT, row['run']) for row in summary['predictions']]
        if any('rotation' not in row for row in summary['predictions']):
            for entry in entries:
                trial = resume(str(checked_entry(entry, record)), 'pose_insurance.registration_trial')
                for name in ('prediction.json', 'registrations.npz'):
                    artifact(trial, name, record)
            result = dict(TW=None, predictions=summary['predictions'], label_free=True,
                          construction_failure='A slot has no completed registration; G3 fails closed')
            write_json(directory / f'episode{request.episode_index}.json', result)
            return result, directory / 'run.json'
        predictions, arrays, _ = load_pair(entries, request, names, record)
        plan = read_json(checked_entry(reg_queue['scored_plan'], record))
        measure = producer(plan['observations']['measure_run'], 'mesh_extents.measure', request)
        checked_entry(next(e for e in plan['inputs'] if e['path'] == str(measure[0].relative_to(ROOT))), record)
        scene = dict(positions=np.array([row['position'] for row in plan['scene']]),
                     quaternions=np.array([row['quaternion_wxyz'] for row in plan['scene']]))
        start = start_poses(scene)
        quaternion_check = bool(np.array_equal(start[:, 3:], scene['quaternions'][:, [1, 2, 3, 0]]))
        rotations = np.array([values['search_0_rotations'][row['best_index']]
                              for row, values in zip(predictions, arrays)])
        normal = np.array(measure[1]['frames'][0]['plane']['normal'])
        result, checks = table_world(start, rotations, normal)
        if not quaternion_check or not np.allclose(Rotation.from_quat(start[:, 3:]).as_quat(), start[:, 3:], atol=1e-15):
            raise ValueError('Scored scene quaternion order mismatch')
        checks['scene_wxyz_to_scoring_xyzw'] = quaternion_check
        result = dict(relative_change(start, result), checks=checks, normal=normal.tolist(),
                      start_poses=result.tolist(), predictions=predictions,
                      measurement=str(measure[0].relative_to(ROOT)),
                      registration_report=str(parent[0].relative_to(ROOT)), label_free=True)
        write_json(directory / f'episode{request.episode_index}.json', result)
    return result, directory / 'run.json'


def train_inputs(queue: dict, record: dict) -> tuple:
    """Admit the exact T1 train scene, its plane producer and selected registrations."""
    import numpy as np
    from ._stereo_placement import start_poses
    request = Request(source_split=queue['source_split'], episode_index=queue['episode'])
    build = producer(checked_entry(queue['build'], record), 'pose_insurance.stereo_placement_build', request)
    plan = read_json(artifact(build, 'construction.json', record))
    measure_path = checked_entry(queue['measurement'], record)
    if input_file(ROOT, measure_path) not in build[1]['inputs']:
        raise ValueError('T1 build did not admit this plane producer')
    measure = producer(measure_path, 'mesh_extents.measure', request)
    if plan['episode'] != request.episode_index or not plan['unchanged_fields']:
        raise ValueError('Train construction differs from the frozen T1 pair')
    scene = {key: np.array(value) for key, value in plan['scenes']['T1'].items()}
    start = start_poses(scene)
    if not np.array_equal(start[:, 3:], scene['quaternions'][:, [1, 2, 3, 0]]):
        raise ValueError('Train scene quaternion order mismatch')
    predictions, arrays, registered = load_pair(queue['registrations'], request, plan['names'], record)
    return start, np.array(measure[1]['frames'][0]['plane']['normal']), predictions, arrays, registered, plan['names']


def historical(queue: dict, record: dict) -> tuple:
    """Read only the eight historical T1 metrics and the descriptive A reproduction row."""
    parent = producer(checked_entry(queue['historical_report'], record), 'pose_insurance.heading_report',
                      allowed=('public_label_diagnostic',))
    summary = read_json(artifact(parent, 'summary.json', record))
    if len(summary['selected']) != 72:
        raise ValueError('Historical report must contain all 72 heading results')
    baseline, previous_a = {}, None
    for row in summary['selected']:
        if row['variant'] != 'T1' and not (row['variant'] == 'H2S-desc' and row['episode'] == queue['episode']):
            continue
        trial = producer(row['run'], 'pose_insurance.heading_trial', allowed=('public_label_diagnostic',))
        record['inputs'].append(input_file(ROOT, trial[0]))
        value = read_json(artifact(trial, 'metrics.json', record))
        if trial[1]['identity'] != row['identity'] or value['episode'] != row['episode'] or value['variant'] != row['variant']:
            raise ValueError('Historical metric row differs from its report')
        if row['variant'] == 'T1':
            baseline[str(row['episode'])] = value['metrics']
        else:
            previous_a = value['metrics']
    if set(baseline) != {str(ep) for ep in queue['episodes']} or previous_a is None:
        raise ValueError('Historical T1 or A coverage is incomplete')
    return baseline, previous_a


def score_trial(variant: str, result, start, basis: dict, paths: list, key: str,
                root: Path, request: Request, names: list) -> Path:
    """Score the static scene with the unchanged H2S-desc kit call."""
    runtime = environment('heading_authorized')
    authorize_episode(runtime, request, 'H3 heading score')
    import numpy as np
    from ._heading_h3 import errors
    from .placement_diagnostic import baseline_module
    from .stereo_placement_loop import scoring_data
    record = record_for('heading_h3_trial', dict(episode=request.episode_index, variant=variant), paths)
    record.update(identity=key, split=request.source_split, episode_index=request.episode_index)
    directory = root / 'trials' / new_run_id()
    with recorded(directory, record):
        ref, vertices = scoring_data(request.episode_index, tuple(names))
        baseline = baseline_module()
        metrics = baseline.score(baseline.load_kit(), np.repeat(result[None, None], len(ref), axis=0), ref, vertices)
        checks = errors(start, result, ref[0, 0])
        if checks['positions_max_change_m'] != 0:
            raise ValueError('H3 changed T1 positions')
        row = dict(variant=variant, episode=request.episode_index, metrics=metrics, frame0=checks,
                   start_poses=result.tolist(), basis=basis, frames=len(ref),
                   scene_wxyz_to_scoring_xyzw=True, descriptive=variant in ('A', 'B'))
        record['metrics'] = metrics
        write_json(directory / 'metrics.json', row)
    return directory / 'run.json'


def integrity(queue: dict, record: dict) -> dict:
    """Compare historical file bytes and inventories with the pre-job snapshot."""
    before = read_json(checked_entry(queue['integrity_snapshot'], record))['files']
    changed = [path for path, digest in before.items() if input_file(ROOT, path)['sha256'] != digest]
    folders = ('outputs/pose_insurance/heading', 'outputs/pose_insurance/registration-hr2/trials',
               'outputs/pose_insurance/registration-hr2/reports')
    old = {path for path in before if any(path.startswith(folder + '/') for folder in folders)}
    current = {str(path.relative_to(ROOT)) for folder in folders
               for path in (ROOT / folder).rglob('*') if path.is_file()}
    result = dict(files_hashed=len(before), historical_heading_results=sum(
        '/heading/runs/' in path and path.endswith('/metrics.json') for path in before),
        hr2_trial_report_files=sum('/registration-hr2/' in path for path in before),
        changed=changed, added=sorted(current - old), removed=sorted(old - current),
        passed=not changed and old == current)
    if not result['passed']:
        raise ValueError('Historical files changed during H3')
    return result


def markdown(summary: dict) -> str:
    """Render the complete protocol measurements and gate decisions."""
    lines = ['# H3 fixed table-world heading', '', summary['verdict'], '',
             f"Executed: {summary['executed']}; resumed: {summary['resumed']}. CPU only.", '',
             '| Row | Scope | AUC | RPE cm | MPPE cm | SP-SR | MP-SR |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name in ('T1', *summary['results']):
        metrics = summary['baseline'][str(summary['results']['H3-sel']['episode'])] if name == 'T1' else summary['results'][name]['metrics']
        for scope, values in [('episode 18', metrics), ('eight episodes', summary['means'][name])]:
            lines.append(f"| {name} | {scope} | " + ' | '.join(f'{values[k]:.9f}' for k in
                ('AUC', 'RPE_cm', 'MPPE_cm', 'SP_SR', 'MP_SR')) + ' |')
    lines += ['', '| Row | b direction error deg | b residual cm | C error deg |', '|---|---:|---:|---:|']
    for name, row in [('T1', summary['t1_frame0']), *((name, row['frame0']) for name, row in summary['results'].items())]:
        lines.append(f"| {name} | {row['b_error_deg']:.9f} | {row['offset_error_m'] * 100:.9f} | {row['C_error_deg']:.9f} |")
    for gate in ('G1', 'G2', 'G3'):
        lines += ['', f'## {gate}', '', '```json', json.dumps(summary[gate], indent=2), '```']
    scored = summary['episode19']
    slots = [{key: row.get(key) for key in ('slot', 'name', 'accepted', 'decline_reasons',
             'best_index', 'best_loss_mm', 'rival_index', 'rival_loss_mm', 'rival_relative_margin',
             'rival_separation_deg', 'stability_deg')} for row in scored['predictions']]
    compact_scored = {key: value for key, value in scored.items() if key != 'predictions'}
    compact_scored['slots'] = slots
    lines += ['', '## Episode 19 (label-free)', '', '```json', json.dumps(compact_scored, indent=2), '```',
              '', '## Integrity and reproduction', '', '```json', json.dumps(summary['reproduction'], indent=2), '```',
              '', '```json', json.dumps(summary['integrity'], indent=2), '```',
              '', NOTE, '', 'Fail-closed reading: all qualifying recorded-rival combinations are included; missing '
              'registrations stop construction. G3 requires completed fits and three finite stability measurements; '
              'ambiguity is retained in HR2 predictions and makes G2 required, rather than changing HR2 acceptance.', '']
    return '\n'.join(lines)


def run(queue_path: str = QUEUE) -> Path:
    """Complete scored TW first, then resume the bounded train queue and compute its gates."""
    started = time.monotonic()
    runtime = environment('heading_authorized')
    queue = read_json(queue_path)
    request = Request(source_split=queue['source_split'], episode_index=queue['episode'])
    authorize_episode(runtime, request, 'H3 report')
    import numpy as np
    from ._heading import selected_registration
    from ._heading_h3 import assess, combinations, descriptive_b, errors, table_world
    from .stereo_placement_loop import scoring_inputs
    checked_entry(queue['protocol'], {'inputs': []})
    spec = {key: queue[key] for key in ('source_split', 'episode', 'episodes', 'variants', 'gates')}
    binding = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (queue['protocol']['path'] != PROTOCOL or binding != CONTRACT_HASH
            or not 1 <= queue['max_trials'] <= len(queue['variants']) or not 1 <= queue['max_seconds'] <= 7200):
        raise PermissionError('H3 admits only the frozen train panel and bounded episode-18 trials')
    reg_queue_path = checked_entry(queue['registration_queue'], {'inputs': []})
    reg_queue, method = registration_contract(str(reg_queue_path))
    sampling = read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('H3 requires the exact H2S-desc deterministic sampling')
    root = output_path(ROOT, OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    scored, scored_run = episode19(queue, queue_path, root, reg_queue)
    ledger = root / 'ledger.jsonl'
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    record = record_for('heading_h3_report', queue, [queue_path, scored_run])
    directory = root / 'reports' / new_run_id()
    with recorded(directory, record):
        start, normal, predictions, arrays, registered, names = train_inputs(queue, record)
        baseline, previous_a = historical(queue, record)
        variants, bases = {}, {}
        for combination in combinations(predictions, reg_queue['thresholds']):
            rotations = np.array([data['search_0_rotations'][index] for data, index in zip(arrays, combination['indices'])])
            name = combination['variant']
            variants[name], bases[name] = table_world(start, rotations, normal)
            bases[name]['indices'] = combination['indices']
        variants.update(A=selected_registration(start, registered), B=descriptive_b(start, registered))
        if list(variants) != queue['variants']:
            raise ValueError('Derived rival combinations differ from the committed H3 queue')
        paths = [e['path'] for e in record['inputs'] if e['path'] != str(scored_run.relative_to(ROOT))]
        paths += scoring_inputs(request.episode_index, names)
        rows, results, executed, reused = [], {}, 0, 0
        for name, poses in variants.items():
            if time.monotonic() - started > queue['max_seconds']:
                raise RuntimeError('H3 heading time budget reached; resume queue')
            trial = record_for('heading_h3_trial', dict(episode=request.episode_index, variant=name), paths)
            key = identity(trial)
            old = next((r for r in reversed(prior) if r['identity'] == key and r['status'] == 'completed'), None)
            if old:
                parent = resume(old['run'], 'pose_insurance.heading_h3_trial')
                reused += 1
            else:
                if executed >= queue['max_trials']:
                    raise RuntimeError('H3 heading trial budget reached; resume queue')
                path = score_trial(name, poses, start, bases.get(name, {}), paths, key, root, request, names)
                parent = path, read_json(path)
                append(ledger, dict(identity=key, status='completed', variant=name, run=str(path.relative_to(ROOT))))
                executed += 1
            record['inputs'].append(input_file(ROOT, parent[0]))
            results[name] = read_json(artifact(parent, 'metrics.json', record))
            rows.append(dict(variant=name, identity=key, run=str(parent[0].relative_to(ROOT))))
        from .placement_diagnostic import reference
        t1_frame0 = errors(start, start, reference(request.episode_index)[0, 0])
        reproduction = dict(A_metric_differences={key: results['A']['metrics'][key] - previous_a[key] for key in previous_a},
                            A_exact=results['A']['metrics'] == previous_a,
                            A_expected_auc_rounded='0.616058782',
                            A_auc_matches_9dp=f"{results['A']['metrics']['AUC']:.9f}" == '0.616058782')
        if not reproduction['A_exact'] or not reproduction['A_auc_matches_9dp']:
            raise ValueError('Descriptive A failed exact H2S-desc metric reproduction')
        assessment = dict(episode=request.episode_index, gates=queue['gates'], thresholds=reg_queue['thresholds'],
                          slots=[trial['slot'] for trial in reg_queue['trials']],
                          stability_repeats=len(method['parameters']['subsampling_seeds']))
        summary = dict(assess(results, baseline, scored['predictions'], assessment), results=results,
                       baseline=baseline, t1_frame0=t1_frame0, episode19=scored,
                       selected=rows, executed=executed, resumed=reused, reproduction=reproduction,
                       integrity=integrity(queue, record),
                       provenance_note=NOTE, wall_seconds=time.monotonic() - started,
                       protocol=queue['protocol'], scored_pose_run=str(scored_run.relative_to(ROOT)))
        write_json(directory / 'summary.json', summary)
        (directory / 'summary.md').write_text(markdown(summary))
    write_json(root / 'latest.json', dict(report=str((directory / 'run.json').relative_to(ROOT))))
    write_json(root / 'summary.json', summary)
    (root / 'summary.md').write_text(markdown(summary))
    print(json.dumps(dict(verdict=summary['verdict'], executed=executed, resumed=reused,
                          report=str((directory / 'run.json').relative_to(ROOT)))), flush=True)
    return directory / 'run.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default=QUEUE)
    run(parser.parse_args().queue)
