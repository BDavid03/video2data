"""Run the fixed W1.0 CPU queue using admitted T1 builds and unchanged kit scoring.

Run: PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B -m
track3.features.pose_insurance.position_oracle. Results are diagnostic, never pack inputs.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from track3.paths import output_path, project_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, heavy_execution,
                         new_run_id, recorded, runtime_flag, source_hashes, read_json)

FEATURE = Path(__file__).parent
OUTPUT = 'outputs/pose_insurance/position-oracle'
PROTOCOL = 'docs/chain/position-oracle-protocol.md'
TARGET = 'outputs/pose_insurance/heading/summary.json'
PROTOCOL_HASH = '617de9ac802b15dd0a83045e2898e8d476a382713939093ac785f80af513916d'


@dataclass(frozen=True)
class PositionRequest:
    episode_index: int
    variant: str
    source_split: str = 'public'

    def __post_init__(self):
        from ._position_oracle import ARMS, EPISODES
        if self.source_split != 'public' or self.episode_index not in EPISODES:
            raise PermissionError('Position oracle requires exactly the eight train episodes')
        if self.variant not in ARMS:
            raise ValueError('Unknown position-oracle arm')


def load_queue(path: Path) -> dict:
    """Refuse departures from the committed protocol before opening any episode."""
    from ._position_oracle import ARMS, EPISODES
    queue = read_json(path)
    if (sha256_file(project_path(ROOT, PROTOCOL)) != PROTOCOL_HASH
            or queue['protocol_sha256'] != PROTOCOL_HASH
            or queue['episodes'] != list(EPISODES) or queue['arms'] != list(ARMS)
            or queue['directions_degrees'] != list(range(0, 360, 15))
            or queue['max_trials'] != 56 or not 1 <= queue['max_seconds'] <= 7200):
        raise ValueError('Queue must preserve the committed W1.0 protocol and bounded budget')
    return queue


def resume(row: dict) -> dict:
    """Re-admit the completed trial, its consumed bytes and its result artifact."""
    record = read_run(ROOT, row['run'], kind='pose_insurance.position_oracle_trial')
    if record['identity'] != row['identity']:
        raise ValueError('Ledger identity differs from its recorded trial')
    for entry in [*record['inputs'], *record['sources']]:
        checked_entry(entry, {'inputs': []})
    return read_json(artifact((project_path(ROOT, row['run']), record), 'metrics.json', {'inputs': []}))


def diagnose(request, build: tuple, paths: list, identity: str) -> Path:
    """Score one arm, retaining all 24 stipulated directions for PL-W1."""
    import numpy as np
    from ._position_oracle import construct, residuals, mean_metrics
    from ._stereo_placement import start_poses
    from .stereo_placement_loop import scoring_data
    from .placement_diagnostic import baseline_module
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Position oracle CPU diagnostic')
    authorize_episode(runtime, request, 'Position oracle CPU diagnostic')
    parent, plan, _ = build
    record = run_fields(ROOT, 'pose_insurance', 'position_oracle_trial', asdict(request),
                        inputs=paths, parents=[parent[0]], provenance='public_label_diagnostic')
    record.update(identity=identity, split='public', episode_index=request.episode_index)
    directory = output_path(ROOT, f'{OUTPUT}/runs/{new_run_id()}')
    with recorded(directory, record):
        start = start_poses({key: np.array(value) for key, value in plan['scenes']['T1'].items()})
        ref, vertices = scoring_data(request.episode_index, tuple(plan['names']))
        baseline = baseline_module()
        kit = baseline.load_kit()
        rows = []
        for k in range(24 if request.variant == 'PL-W1' else 1):
            poses = construct(start, ref[0, 0], request.variant, request.episode_index, k)
            static = np.repeat(poses[None, None], len(ref), axis=0)
            warped = kit._warp_to_reference_start(static[:1], ref[:1])[0, 0]
            rows.append(dict(k=k, yaw_degrees=k * 15, start_poses=poses.tolist(),
                             metrics=baseline.score(kit, static, ref, vertices),
                             checks=residuals(start, poses, ref[0, 0], warped, request.variant, request.episode_index)))
        result = dict(episode=request.episode_index, arm=request.variant, names=plan['names'], frames=len(ref),
                      metrics=mean_metrics([row['metrics'] for row in rows]), directions=rows)
        if request.variant == 'PL-W1':
            result['range'] = {key: dict(min=min(row['metrics'][key] for row in rows),
                                        mean=result['metrics'][key], max=max(row['metrics'][key] for row in rows))
                               for key in result['metrics']}
        write_json(directory / 'metrics.json', result)
    return directory / 'run.json'


def validity(results: dict, target: dict, paths: list) -> dict:
    """Apply all five checks and retain per-episode residuals for audit."""
    from ._position_oracle import EPISODES, METRICS, TREATED
    reproduction = {arm: {str(ep): {key: abs(results[arm][str(ep)]['metrics'][key] -
                                                target['results'][arm][str(ep)]['metrics'][key])
                                   for key in METRICS} for ep in EPISODES} for arm in ('T1', 'O2', 'O3')}
    largest = max(v for arm in reproduction.values() for ep in arm.values() for v in ep.values())
    checks = {'1_reproduction': dict(passed=largest <= 1e-9, largest_residual=largest, residuals=reproduction)}
    for number, arm in ((2, 'PO'), (3, 'PL-W1'), (4, 'RC')):
        details = {str(ep): [row['checks'] for row in results[arm][str(ep)]['directions']] for ep in EPISODES}
        maxima = {}
        for rows in details.values():
            for row in rows:
                for key, value in row.items():
                    maxima[key] = (maxima.get(key, True) and value if isinstance(value, bool)
                                   else max(maxima.get(key, 0.), value))
        if arm == 'PL-W1':
            k0 = {str(ep): {key: abs(results[arm][str(ep)]['directions'][0]['metrics'][key] -
                                     results['PO-W1'][str(ep)]['metrics'][key]) for key in METRICS} for ep in EPISODES}
            maxima['k0_metric_difference'] = max(v for row in k0.values() for v in row.values())
            details['k0_metric_differences'] = k0
        passed = all(value if isinstance(value, bool) else value <= 1e-9 for value in maxima.values())
        checks[f'{number}_{arm}'] = dict(passed=passed, maxima=maxima, residuals=details)
        if arm == 'PL-W1':
            checks[f'{number}_{arm}']['geometry_checked_episodes'] = list(TREATED)
    episode_paths = sorted({p for p in paths if Path(p).name.startswith('episode_') and p.endswith('.parquet')})
    opened = sorted(int(Path(p).stem.removeprefix('episode_')) for p in episode_paths)
    checks['5_episode_scope'] = dict(passed=opened == sorted(EPISODES), opened_episode_indices=opened,
                                     episode_inputs=episode_paths, excluded_episodes_read=0,
                                     single_object_controls_run=0, largest_residual=0)
    return checks


def table(headers: list, rows: list) -> str:
    """Render a compact Markdown table of plain values."""
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def render(summary: dict) -> tuple[str, str, str]:
    """Render results, pure marginals and the descriptive scenario calculation."""
    from ._position_oracle import ARMS, EPISODES, METRICS
    fmt = lambda values: [f'{values[key]:.9f}' for key in METRICS]
    means = table(['arm', *METRICS], [[arm, *fmt(summary['means'][arm])] for arm in ARMS])
    per_episode = table(['episode', 'arm', 'AUC', 'RPE cm', 'MPPE cm'],
                        [[ep, arm, *fmt(summary['results'][arm][str(ep)]['metrics'])[:3]] for ep in EPISODES for arm in ARMS])
    ranges = table(['episode', 'metric', 'min', 'mean', 'max'],
                   [[ep, key, *[f'{row[k]:.9f}' for k in ('min', 'mean', 'max')]] for ep in EPISODES
                    for key, row in summary['results']['PL-W1'][str(ep)]['range'].items()])
    marginal = table(['order / difference', *METRICS], [[label, *fmt(row)] for label, row in summary['marginals'].items()])
    scenarios = table(['route', 'g AUC', 'g RPE cm', 'g MPPE cm', 'r', 'h', 'AUC g*r/(20h)',
                       'RPE g*r/(20h)', 'MPPE g*r/(20h)'],
                      [[row['route'], *fmt(row['g'])[:3], row['r'], str(row['h']) + (' (unverified)' if row['hours_unverified'] else ''),
                        *fmt(row['formula'])[:3]] for row in summary['scenarios']])
    notes = ('CPU; one world; 1,000 scan surface samples per object, seed 0 (host-vertex approximation). '
             'Scans and mocap are scoring-only diagnostic inputs; no pack candidate.\n\n'
             'PL-W1 is conditional yaw sensitivity with reference height/radius and stipulated uniform yaw; '
             'it is not a forecast of W1. These full-XYZ repairs are not whole-clip static upper bounds.\n\n'
             'Metric differences are signed: lower RPE/MPPE is better. Scenario g is per treated episode; '
             'rotation rows exclude train 18. Perfect-model rotation figures favour rotation by construction. '
             'This calculation selects nothing.\n\n')
    checks = '\n'.join(f'- {key}: {"PASS" if row["passed"] else "FAIL"}; maxima '
                       f'`{json.dumps(row.get("maxima", {"largest_residual": row.get("largest_residual")}))}`'
                       for key, row in summary['validity'].items())
    text = ('# W1.0 position-only oracle\n\n' + notes + '## Validity\n\n' + checks + '\n\n## Eight-episode means\n\n' + means +
            '\n\n## Per-episode metrics\n\n' + per_episode + '\n\n## PL-W1 direction ranges\n\n' + ranges +
            '\n\n## Pure marginals (both orders)\n\n' + marginal + '\n\n## Fixed scenarios\n\n' + scenarios +
            '\n\nRun: `PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B -m '
            'track3.features.pose_insurance.position_oracle`. Exact identities resume without rescoring. '
            'Per-direction poses, metrics and residuals are retained in summary.json and trial metrics.json.\n')
    return text, marginal + '\n', notes + scenarios + '\n'


def run_queue(queue_value: str) -> None:
    """Resume the 56 fixed trials and publish reports only after validity checks."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Position oracle CPU diagnostic')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    queue_path = project_path(ROOT, queue_value)
    queue = load_queue(queue_path)
    from ._position_oracle import ARMS, EPISODES, calculations
    from .heading_loop import HeadingRequest, admit_build, append, protocol_queue
    from .shrink_loop import resume_ledger
    from .stereo_placement_loop import scoring_inputs
    sampling = read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('W1.0 requires 1,000 samples and seed zero')
    heading = protocol_queue()
    directory = output_path(ROOT, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    # All generated cache locations are beneath this diagnostic's output tree.
    from track3.runs import CACHE_FOLDERS
    import tempfile
    for key in CACHE_FOLDERS:
        path = directory / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = os.environ['TMPDIR']
    rows, ledger = resume_ledger(directory)
    sources = source_hashes(FEATURE)
    common = [queue_path, PROTOCOL, TARGET, FEATURE / 'heading_queue.json']
    builds, inputs = {}, {}
    for ep in EPISODES:
        authorize_episode(runtime, HeadingRequest(ep, 'T1'), 'Position oracle CPU queue')
        builds[ep] = admit_build(ep, heading['builds'][str(ep)])
        paths = [*common, *scoring_inputs(ep, builds[ep][1]['names']), builds[ep][0][0],
                 project_path(ROOT, builds[ep][2]['path'])]
        inputs[ep] = [input_file(ROOT, p) for p in dict.fromkeys(map(str, paths))]
    results, selected, executed, resumed = {arm: {} for arm in ARMS}, [], 0, 0
    started = time.perf_counter()
    for arm in ARMS:
        for ep in EPISODES:
            if time.perf_counter() - started > queue['max_seconds']:
                raise TimeoutError('Queue budget reached; resume with the same command')
            payload = dict(episode=ep, arm=arm, inputs=inputs[ep], sources=sources)
            identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            prior = next((r for r in reversed(rows) if r.get('identity') == identity and r['status'] == 'completed'), None)
            if prior:
                result, row = resume(prior), prior
                resumed += 1
            else:
                row = dict(identity=identity, episode=ep, arm=arm, status='failed')
                tick = time.perf_counter()
                try:
                    request = PositionRequest(ep, arm)
                    path = diagnose(request, builds[ep], [entry['path'] for entry in inputs[ep]], identity)
                    row.update(status='completed', run=path.relative_to(ROOT).as_posix())
                    result = resume(row)
                    executed += 1
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                    raise
                finally:
                    row['seconds'] = time.perf_counter() - tick
                    append(ledger, row)
                rows.append(row)
            results[arm][str(ep)] = result
            selected.append(row)
            print(f'{arm} episode {ep}: {result["metrics"]}', flush=True)
    checks = validity(results, read_json(TARGET), [entry['path'] for rows in inputs.values() for entry in rows])
    passed = all(row['passed'] for row in checks.values())
    calculated = calculations(results)
    summary = dict(results=results, validity=checks, validity_passed=passed, selected=selected,
                   executed=executed, resumed=resumed, protocol=input_file(ROOT, PROTOCOL),
                   ranking=sorted(ARMS, key=lambda arm: calculated['means'][arm]['AUC'], reverse=True) if passed else [],
                   **calculated)
    report = output_path(ROOT, f'{OUTPUT}/reports/{new_run_id()}')
    record = run_fields(ROOT, 'pose_insurance', 'position_oracle_report', dict(queue=queue_value),
                        inputs=common, parents=[row['run'] for row in selected], provenance='public_label_diagnostic')
    with recorded(report, record):
        write_json(report / 'summary.json', summary)
        write_json(report / 'marginals.json', summary['marginals'])
        write_json(report / 'scenarios.json', summary['scenarios'])
        for name, text in zip(('summary.md', 'marginals.md', 'scenarios.md'), render(summary)):
            (report / name).write_text(text, encoding='utf-8')
    for name in ('summary.json', 'summary.md', 'marginals.json', 'marginals.md', 'scenarios.json', 'scenarios.md'):
        (directory / name).write_bytes((report / name).read_bytes())
    write_json(directory / 'finished.json', dict(status='completed' if passed else 'invalid', validity_passed=passed,
               executed=executed, resumed=resumed, report=(report / 'run.json').relative_to(ROOT).as_posix()))
    if not passed:
        raise ValueError('Validity failure blocks ranking; inspect summary.json')
    print(json.dumps(read_json(directory / 'finished.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default='track3/features/pose_insurance/position_oracle_queue.json')
    run_queue(parser.parse_args().queue)
