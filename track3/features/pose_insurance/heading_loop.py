"""Resume the fixed train-only CPU heading queue and publish protocol verdicts."""
import argparse
from dataclasses import asdict, dataclass
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from track3.paths import output_path, project_path
from track3.provenance import input_file, read_run, run_fields, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, heavy_execution,
                         new_run_id, producer, read_json, recorded, runtime_flag, source_hashes)

FEATURE = Path(__file__).parent
OUTPUT = 'outputs/pose_insurance/heading'
PROTOCOL = 'docs/chain/pose-heading-protocol.md'
METRICS = ['AUC', 'RPE_cm', 'MPPE_cm', 'SP_SR', 'MP_SR']


def validate_protocol(queue: dict) -> None:
    """Bind queue decisions to the reviewed protocol bytes and coverage extension."""
    contract = {key: queue[key] for key in
                ('episodes', 'variants', 'descriptive_variants', 'gates', 'prior_consistency')}
    binding = hashlib.sha256(project_path(ROOT, PROTOCOL).read_bytes() +
                             json.dumps(contract, sort_keys=True).encode()).hexdigest()
    # SHA-256(protocol bytes + sorted-key contract JSON); values live only in the queue.
    if binding != '69f4311a980c5574e8eb5d756cc583e44bd66a283a664214a6f683604a660c21':
        raise ValueError('Queue or committed heading protocol differs from the frozen decisions')


@cache
def protocol_queue() -> dict:
    """Load the fixed request domain once, retaining direct-request refusals."""
    queue = read_json(FEATURE / 'heading_queue.json')
    validate_protocol(queue)
    return queue


@dataclass(frozen=True)
class HeadingRequest:
    episode_index: int
    variant: str
    source_split: str = 'public'

    def __post_init__(self):
        queue = protocol_queue()
        if self.source_split != 'public' or self.episode_index not in queue['episodes']:
            raise PermissionError('Heading requires the eight protocol train episodes')
        if self.variant not in queue['variants']:
            raise ValueError('Unknown heading variant')


def append(ledger: Path, row: dict) -> None:
    """Persist one attempt event before returning control to the loop."""
    with ledger.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def admit_build(episode: int, path: str) -> tuple:
    """Admit the recorded T1 construction without re-running its video producers."""
    from .placement_diagnostic import ROSTER
    parent = producer(path, 'pose_insurance.stereo_placement_build', HeadingRequest(episode, 'T1'))
    entry = {'inputs': []}
    plan = read_json(artifact(parent, 'construction.json', entry))
    if plan['episode'] != episode or plan['names'] != ROSTER[episode] or not plan['unchanged_fields']:
        raise ValueError('T1 construction does not match the frozen train roster')
    return parent, plan, entry['inputs'][0]


def trial_inputs(request: HeadingRequest, queue: dict, queue_path: Path, build: tuple) -> list:
    """Name the exact construction, fitting labels, scans and scoring implementation."""
    from .stereo_placement_loop import scoring_inputs
    from .placement_diagnostic import label_paths
    paths = [queue_path, PROTOCOL, *scoring_inputs(request.episode_index, build[1]['names']),
             build[0][0], ROOT / build[2]['path']]
    if request.variant.startswith('U1') or request.variant == 'O2-cups-up':
        fitting = [ep for ep in queue['episodes'] if request.variant != 'U1-LOO' or ep != request.episode_index]
        paths.extend(path for ep in fitting for path in label_paths(ep))
    return list(dict.fromkeys(map(str, paths)))


def resume_result(row: dict, replacements: dict | None = None) -> dict:
    """Re-admit a completed trial and every input and source it consumed."""
    record = read_run(ROOT, row['run'], kind='pose_insurance.heading_trial')
    for entry in [*record['inputs'], *record['sources']]:
        if replacements is not None and entry['path'] in replacements:
            replacements[entry['path']].append(entry['sha256'])
        else:
            checked_entry(entry, {'inputs': []})
    return read_json(artifact((ROOT / row['run'], record), 'metrics.json', {'inputs': []}))


def diagnose(request: HeadingRequest, build: tuple, paths: list, priors: dict, identity: str) -> Path:
    """Score one frame-zero construction with the unchanged kit integration."""
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Heading CPU diagnostic')
    authorize_episode(runtime, request, 'Heading CPU diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    import numpy as np
    from ._heading import oracle, apply_priors, cup_oracle, frame0_checks
    from ._stereo_placement import start_poses
    from .stereo_placement_loop import scoring_data
    from .placement_diagnostic import baseline_module
    parent, plan, _ = build
    record = run_fields(ROOT, 'pose_insurance', 'heading_trial', asdict(request), inputs=paths,
                        parents=[parent[0]], provenance='public_label_diagnostic')
    record.update(split='public', episode_index=request.episode_index, identity=identity,
                  object_names=plan['names'])
    artifact(parent, 'construction.json', record)
    directory = output_path(ROOT, f'{OUTPUT}/runs/{new_run_id()}')
    with recorded(directory, record):
        start = start_poses({key: np.array(value) for key, value in plan['scenes']['T1'].items()})
        ref, vertices = scoring_data(request.episode_index, tuple(plan['names']))
        if request.variant in protocol_queue()['descriptive_variants']:
            result = cup_oracle(start, ref[0, 0], plan['names'], priors if request.variant == 'O2-cups-up' else None)
        else:
            result = (apply_priors(start, plan['names'], priors) if request.variant.startswith('U1')
                      else oracle(start, ref[0, 0], request.variant))
        checks = frame0_checks(start, result, ref[0, 0])
        if request.variant != 'O3' and checks['positions_max_change_m'] != 0:
            raise ValueError('Heading changed T1 positions')
        if request.variant == 'O1' and checks['relative_rotation_change_rad'] > 1e-12:
            raise ValueError('O1 changed relative rotation')
        if request.variant in ('O1', 'O2') and checks['horizontal_direction_defined'] and checks['horizontal_direction_error_rad'] > 1e-12:
            raise ValueError('Direction oracle failed')
        if request.variant == 'O1' and not checks['horizontal_direction_defined'] and not np.array_equal(result, start):
            raise ValueError('Undefined O1 must retain T1')
        if request.variant in ('O2', 'O3') and checks['relative_rotation_error_rad'] > 1e-12:
            raise ValueError('Relative-rotation oracle failed')
        if request.variant == 'O3' and checks['offset_error_m'] > 1e-12:
            raise ValueError('Full frame-zero oracle failed')
        baseline = baseline_module()
        metrics = baseline.score(baseline.load_kit(), np.repeat(result[None, None], len(ref), axis=0), ref, vertices)
        record['metrics'] = metrics
        write_json(directory / 'metrics.json', dict(episode=request.episode_index, variant=request.variant,
                   names=plan['names'], metrics=metrics, frames=len(ref), frame0=checks,
                   start_poses=result.tolist(), priors=priors if request.variant.startswith('U1') or request.variant == 'O2-cups-up' else None,
                   descriptive=request.variant in protocol_queue()['descriptive_variants'],
                   covered_slots=[i for i, name in enumerate(plan['names']) if name in ('blue_cup', 'beige_cup')]))
    return directory / 'run.json'


def fit_folds(episodes: list, consistency: dict) -> tuple:
    """Read only train frame-zero labels and fit the full and episode-excluded priors."""
    from ._heading import body_ups, fit_priors
    from .placement_diagnostic import ROSTER, reference
    occurrences = {name: [] for ep in episodes for name in ROSTER[ep]}
    for ep in episodes:
        for name, up in zip(ROSTER[ep], body_ups(reference(ep)[0, 0])):
            occurrences[name].append(dict(episode=ep, up=up.tolist()))
    return fit_priors(occurrences, consistency), {
        ep: fit_priors({name: [row for row in rows if row['episode'] != ep]
                        for name, rows in occurrences.items()}, consistency) for ep in episodes}


def recorded_trials(queue: dict) -> dict:
    """Admit the pinned historical report supplying the original 48 completed trials."""
    path = checked_entry(queue['recorded_report'], {'inputs': []})
    report = producer(path, 'pose_insurance.heading_report', allowed=('public_label_diagnostic',))
    summary = read_json(artifact(report, 'summary.json', {'inputs': []}))
    trials = {(row['variant'], row['episode']): row for row in summary['selected']}
    if set(trials) != {(v, ep) for v in queue['variants'] if v not in queue['descriptive_variants'] for ep in queue['episodes']}:
        raise ValueError('Historical report must contain exactly the original 48 trials')
    return trials


def resume_recorded(request: HeadingRequest, row: dict, build: tuple, priors: dict) -> tuple:
    """Admit an additive extension only when the historical scored poses are identical."""
    import numpy as np
    from ._heading import oracle, apply_priors
    from ._stereo_placement import start_poses
    from .placement_diagnostic import reference
    replacements = {path: [] for path in (PROTOCOL, 'track3/features/pose_insurance/heading_queue.json',
                    'track3/features/pose_insurance/_heading.py', 'track3/features/pose_insurance/heading_loop.py')}
    result = resume_result(row, replacements)
    record = read_run(ROOT, row['run'], kind='pose_insurance.heading_trial')
    if record['request'] != asdict(request) or record['identity'] != row['identity'] or result['names'] != build[1]['names']:
        raise ValueError('Historical trial does not match its selected request')
    start = start_poses({key: np.array(value) for key, value in build[1]['scenes']['T1'].items()})
    constructed = (apply_priors(start, result['names'], priors) if request.variant.startswith('U1')
                   else oracle(start, reference(request.episode_index)[0, 0], request.variant))
    if not np.array_equal(constructed, result['start_poses']):
        raise ValueError('Historical scored poses differ under the additive extension')
    if request.variant.startswith('U1') and result['priors'] != priors:
        raise ValueError('Historical fitting priors differ')
    admission = dict(run=row['run'], identical_scored_poses=True,
                     replacements={path: dict(recorded_sha256=sorted(set(hashes)),
                                              current=input_file(ROOT, path)) for path, hashes in replacements.items()})
    return result, admission


def analysis(results: dict, queue: dict) -> dict:
    """Admit the historical baseline and aggregate the configured protocol gates."""
    from ._heading import analyze
    return analyze(results, queue, reproduce_t1(results, queue), METRICS)


def operation(request: HeadingRequest, build: tuple, paths: list, priors: dict,
              key: str, rows: list, ledger: Path) -> tuple:
    """Record a new attempt or re-admit an exact completed trial."""
    prior = next((r for r in reversed(rows) if r.get('identity') == key and r['status'] == 'completed'), None)
    if prior:
        return resume_result(prior), prior, False
    row = dict(identity=key, episode=request.episode_index, variant=request.variant, status='started')
    append(ledger, row)
    tick = time.perf_counter()
    try:
        path = diagnose(request, build, paths, priors, key)
        row.update(status='completed', run=path.relative_to(ROOT).as_posix())
        result = resume_result(row)
    except Exception as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        if hasattr(error, 'feature_run_path'):
            row['run'] = error.feature_run_path
        raise
    finally:
        row['seconds'] = time.perf_counter() - tick
        append(ledger, row)
    rows.append(row)
    return result, row, True


def reproduce_t1(results: dict, queue: dict) -> dict:
    """Check full-precision historical T1 metrics before scoring any oracle."""
    differences = {}
    for ep in queue['episodes']:
        old = producer(queue['t1_scores'][str(ep)], 'pose_insurance.stereo_placement_diagnostic',
                       HeadingRequest(ep, 'T1'), allowed=('public_label_diagnostic',))
        previous = read_json(artifact(old, 'metrics.json', {'inputs': []}))
        if previous['candidate'] != 'T1':
            raise ValueError('Reproduction producer is not T1')
        differences[ep] = {key: results['T1'][ep]['metrics'][key] - previous['metrics'][key] for key in METRICS}
    if max(abs(value) for row in differences.values() for value in row.values()) > 1e-9:
        raise ValueError('T1 reproduction failed before oracle evaluation')
    return dict(passed=True, differences=differences, tolerance=1e-9)


def publish(summary: dict, queue: dict, queue_path: Path, directory: Path) -> None:
    """Record an immutable report and update the latest summary pointers."""
    report = output_path(ROOT, f'{OUTPUT}/reports/{new_run_id()}')
    record = run_fields(ROOT, 'pose_insurance', 'heading_report', dict(queue=str(queue_path.relative_to(ROOT))),
                        inputs=[queue_path, PROTOCOL, FEATURE / 'runtime.json', *queue['t1_scores'].values()],
                        parents=[queue['recorded_report']['path'], *[row['run'] for row in summary['selected']]],
                        provenance='public_label_diagnostic')
    with recorded(report, record):
        write_json(report / 'summary.json', summary)
        from ._heading_render import render, figure
        render(summary, report, queue, METRICS)
        figure(summary, report, queue)
    for name in ('summary.json', 'summary.md', 'auc.png'):
        (directory / name).write_bytes((report / name).read_bytes())
    write_json(directory / 'finished.json', dict(status='completed', executed=summary['executed'], resumed=summary['resumed'],
               report=(report / 'run.json').relative_to(ROOT).as_posix(),
               stage_a_passed=summary['stage_a']['passed'], u1_passed=summary['u1_gate']['passed']))
    print(json.dumps(read_json(directory / 'finished.json')), flush=True)


def run_queue(queue_path: str) -> None:
    """Admit the fixed queue, resume exact trials and publish an immutable invocation report."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Heading CPU diagnostic')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    from .shrink_loop import resume_ledger
    queue_path = project_path(ROOT, queue_path)
    queue = read_json(queue_path)
    validate_protocol(queue)
    episodes, variants = queue['episodes'], queue['variants']
    if (not 1 <= queue['max_trials'] <= len(episodes) * len(variants) or not 1 <= queue['max_seconds'] <= 7200
            or set(queue['builds']) != set(map(str, episodes)) or set(queue['t1_scores']) != set(map(str, episodes))):
        raise ValueError('Queue must preserve the fixed heading protocol and bounded budget')
    sampling = read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('Heading requires 1,000 surface samples and seed zero')
    for ep in queue['episodes']:
        authorize_episode(runtime, HeadingRequest(ep, 'T1'), 'Heading queue')
    builds = {ep: admit_build(ep, queue['builds'][str(ep)]) for ep in episodes}
    priors, folds = fit_folds(episodes, queue['prior_consistency'])
    directory = output_path(ROOT, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    rows, ledger = resume_ledger(directory)
    historical = recorded_trials(queue)
    sources = source_hashes(FEATURE)
    results, selected, executed, resumed = {v: {} for v in variants}, [], 0, 0
    admissions = []
    started = time.perf_counter()
    for variant in variants:
        for ep in queue['episodes']:
            request = HeadingRequest(ep, variant)
            fitted = folds[ep] if variant == 'U1-LOO' else priors
            if (variant, ep) in historical:
                row, fresh = historical[variant, ep], False
                result, admission = resume_recorded(request, row, builds[ep], fitted)
                admissions.append(admission)
            else:
                paths = trial_inputs(request, queue, queue_path, builds[ep])
                payload = dict(request=asdict(request), inputs=[input_file(ROOT, p) for p in paths], sources=sources)
                key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                completed = any(r.get('identity') == key and r['status'] == 'completed' for r in rows)
                if not completed and (executed >= queue['max_trials'] or time.perf_counter() - started >= queue['max_seconds']):
                    raise RuntimeError('Heading budget reached; resume the unchanged queue')
                result, row, fresh = operation(request, builds[ep], paths, fitted, key, rows, ledger)
            executed += int(fresh)
            resumed += int(not fresh)
            selected.append(row)
            results[variant][ep] = result
            print(json.dumps(dict(episode=ep, variant=variant, resumed=not fresh, metrics=result['metrics'])), flush=True)
        if variant == 'T1':
            reproduce_t1(results, queue)
    summary = dict(**analysis(results, queue), results=results, priors_full=priors, priors_loo=folds,
                   executed=executed, resumed=resumed, selected=selected, sampling=sampling,
                   recorded_trial_admissions=admissions)
    previous = read_json(directory / 'summary.json') if (directory / 'summary.json').exists() else None
    summary['mean_changes_since_previous'] = ({v: {k: summary['means'][v][k] - previous['means'][v][k]
                                                  for k in METRICS} for v in variants if v in previous['means']} if previous else None)
    publish(summary, queue, queue_path, directory)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default='track3/features/pose_insurance/heading_queue.json')
    run_queue(parser.parse_args().queue)


if __name__ == '__main__':
    main()
