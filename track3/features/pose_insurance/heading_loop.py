"""Resume the fixed train-only CPU heading queue and publish protocol verdicts."""
import argparse
from dataclasses import asdict, dataclass
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
EPISODES = [0, 11, 12, 18, 21, 23, 26, 31]
VARIANTS = ['T1', 'O1', 'O2', 'O3', 'U1-LOO', 'U1-full']
METRICS = ['AUC', 'RPE_cm', 'MPPE_cm', 'SP_SR', 'MP_SR']


@dataclass(frozen=True)
class HeadingRequest:
    episode_index: int
    variant: str
    source_split: str = 'public'

    def __post_init__(self):
        if self.source_split != 'public' or self.episode_index not in EPISODES:
            raise PermissionError('Heading requires the eight protocol train episodes')
        if self.variant not in VARIANTS:
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
    if request.variant.startswith('U1'):
        fitting = [ep for ep in queue['episodes'] if request.variant == 'U1-full' or ep != request.episode_index]
        paths.extend(path for ep in fitting for path in label_paths(ep))
    return list(dict.fromkeys(map(str, paths)))


def resume_result(row: dict) -> dict:
    """Re-admit a completed trial and every input and source it consumed."""
    record = read_run(ROOT, row['run'], kind='pose_insurance.heading_trial')
    for entry in [*record['inputs'], *record['sources']]:
        checked_entry(entry, {'inputs': []})
    return read_json(artifact((ROOT / row['run'], record), 'metrics.json', {'inputs': []}))


def diagnose(request: HeadingRequest, build: tuple, paths: list, priors: dict, identity: str) -> Path:
    """Score one frame-zero construction with the unchanged kit integration."""
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Heading CPU diagnostic')
    authorize_episode(runtime, request, 'Heading CPU diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    import numpy as np
    from ._heading import oracle, apply_priors, frame0_checks
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
                   start_poses=result.tolist(), priors=priors if request.variant.startswith('U1') else None))
    return directory / 'run.json'


def fit_folds(episodes: list) -> tuple:
    """Read only train frame-zero labels and fit the full and episode-excluded priors."""
    from ._heading import body_ups, fit_priors
    from .placement_diagnostic import ROSTER, reference
    occurrences = {name: [] for ep in episodes for name in ROSTER[ep]}
    for ep in episodes:
        for name, up in zip(ROSTER[ep], body_ups(reference(ep)[0, 0])):
            occurrences[name].append(dict(episode=ep, up=up.tolist()))
    return fit_priors(occurrences), {
        ep: fit_priors({name: [row for row in rows if row['episode'] != ep]
                        for name, rows in occurrences.items()}) for ep in episodes}


def analysis(results: dict, queue: dict) -> dict:
    """Apply T1 reproduction, Stage A and Amendment 1's fixed U1 gates."""
    import numpy as np
    means = {v: {key: float(np.mean([results[v][ep]['metrics'][key] for ep in EPISODES]))
                  for key in METRICS} for v in VARIANTS}
    reproduction = reproduce_t1(results, queue)
    t1, loo = means['T1'], means['U1-LOO']
    changes = {key: loo[key] - t1[key] for key in METRICS}
    worst_loss = max(results['T1'][ep]['metrics']['AUC'] - results['U1-LOO'][ep]['metrics']['AUC'] for ep in EPISODES)
    checks = dict(auc_gain=changes['AUC'] >= .010, rpe_not_worse=changes['RPE_cm'] <= 0,
                  mppe_not_worse=changes['MPPE_cm'] <= 0, sp_sr_not_lower=changes['SP_SR'] >= 0,
                  mp_sr_not_lower=changes['MP_SR'] >= 0, worst_episode_auc_loss=worst_loss <= .02)
    stage_a_gain = means['O2']['AUC'] - t1['AUC']
    return dict(means=means, ranking=sorted(VARIANTS, key=lambda v: means[v]['AUC'], reverse=True),
                reproduction=reproduction,
                stage_a=dict(auc_gain=stage_a_gain, passed=stage_a_gain >= .010),
                u1_gate=dict(changes=changes, worst_auc_loss=worst_loss, checks=checks,
                             passed=all(checks.values())))


def render(summary: dict, directory: Path) -> None:
    """Write ranked metrics, all episode scores and prior tables."""
    lines = ['# Recorded heading diagnostic', '',
             'Train labels only; Stage A oracles are ceilings. U1 was chosen after label inspection; '
             'LOO is the gate, full-train is descriptive. Scan ADD uses 1,000 seed-zero surface samples.', '',
             f"Executed {summary['executed']}; resumed {summary['resumed']}. "
             f"T1 reproduction: {summary['reproduction']['passed']}. "
             f"Stage A passes: {summary['stage_a']['passed']}. U1-LOO passes: {summary['u1_gate']['passed']}.", '',
             f"Changes since previous report: `{json.dumps(summary['mean_changes_since_previous'], sort_keys=True)}`.", '',
             '| Variant | AUC | RPE cm | MPPE cm | SP-SR | MP-SR |', '|---|---:|---:|---:|---:|---:|']
    for variant in summary['ranking']:
        lines.append('| ' + variant + ' | ' + ' | '.join(f'{summary["means"][variant][k]:.9f}' for k in METRICS) + ' |')
    lines += ['', '## Per-episode metrics', '', '| Episode | Variant | AUC | RPE cm | MPPE cm | SP-SR | MP-SR |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for ep in EPISODES:
        for variant in VARIANTS:
            metrics = summary['results'][variant][ep]['metrics']
            lines.append(f'| {ep} | {variant} | ' + ' | '.join(f'{metrics[k]:.9f}' for k in METRICS) + ' |')
    lines += ['', '## Full-train priors', '',
              '| Type | Episodes | N | Body-up observations | Max angle deg | Mean norm | Prior up | Prior? |',
              '|---|---|---:|---|---:|---:|---|---|']
    for name, row in summary['priors_full'].items():
        angle = '-' if row['max_pairwise_angle_deg'] is None else f'{row["max_pairwise_angle_deg"]:.6f}'
        vectors = '; '.join(', '.join(f'{x:.6f}' for x in v) for v in row['up_vectors'])
        up = '-' if row['up'] is None else ', '.join(f'{x:.6f}' for x in row['up'])
        lines.append(f'| {name} | {row["episodes"]} | {row["occurrences"]} | {vectors} | {angle} | '
                     f'{row["mean_norm"]:.9f} | {up} | {row["prior"]} |')
    lines += ['', 'All LOO fold memberships, observations and consistency statistics are in summary.json and each U1 trial.', '',
              '## U1-LOO gate', '', f"Mean changes: `{json.dumps(summary['u1_gate']['changes'], sort_keys=True)}`.", '',
              f"Worst episode AUC loss: {summary['u1_gate']['worst_auc_loss']:.9f}.", '',
              *[f'- {key}: {value}' for key, value in summary['u1_gate']['checks'].items()], '',
              'No scored policy, geometry counter-rotation, physical check or pack is part of this diagnostic.', '',
              '![Per-episode AUC](auc.png)', '']
    (directory / 'summary.md').write_text('\n'.join(lines))


def figure(summary: dict, directory: Path) -> None:
    """Plot every protocol variant's train-episode ADD AUC."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for variant in VARIANTS:
        ax.plot(range(len(EPISODES)), [summary['results'][variant][ep]['metrics']['AUC'] for ep in EPISODES],
                marker='o', markersize=3, label=variant)
    ax.set(xticks=range(len(EPISODES)), xticklabels=EPISODES, xlabel='Train episode', ylabel='ADD AUC', ylim=(0, 1))
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'auc.png', dpi=140)
    plt.close(fig)


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
    for ep in EPISODES:
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
                        parents=[row['run'] for row in summary['selected']], provenance='public_label_diagnostic')
    with recorded(report, record):
        write_json(report / 'summary.json', summary)
        render(summary, report)
        figure(summary, report)
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
    if (queue['episodes'] != EPISODES or queue['variants'] != VARIANTS
            or not 1 <= queue['max_trials'] <= 48 or not 1 <= queue['max_seconds'] <= 7200
            or set(queue['builds']) != set(map(str, EPISODES)) or set(queue['t1_scores']) != set(map(str, EPISODES))):
        raise ValueError('Queue must preserve the fixed heading protocol and bounded budget')
    sampling = read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('Heading requires 1,000 surface samples and seed zero')
    for ep in EPISODES:
        authorize_episode(runtime, HeadingRequest(ep, 'T1'), 'Heading queue')
    builds = {ep: admit_build(ep, queue['builds'][str(ep)]) for ep in EPISODES}
    priors, folds = fit_folds(EPISODES)
    directory = output_path(ROOT, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    rows, ledger = resume_ledger(directory)
    sources = source_hashes(FEATURE)
    results, selected, executed, resumed = {v: {} for v in VARIANTS}, [], 0, 0
    started = time.perf_counter()
    for variant in VARIANTS:
        for ep in EPISODES:
            request = HeadingRequest(ep, variant)
            paths = trial_inputs(request, queue, queue_path, builds[ep])
            payload = dict(request=asdict(request), inputs=[input_file(ROOT, p) for p in paths], sources=sources)
            key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            completed = any(r.get('identity') == key and r['status'] == 'completed' for r in rows)
            if not completed and (executed >= queue['max_trials'] or time.perf_counter() - started >= queue['max_seconds']):
                raise RuntimeError('Heading budget reached; resume the unchanged queue')
            result, row, fresh = operation(request, builds[ep], paths, folds[ep] if variant == 'U1-LOO' else priors,
                                           key, rows, ledger)
            executed += int(fresh)
            resumed += int(not fresh)
            selected.append(row)
            results[variant][ep] = result
            print(json.dumps(dict(episode=ep, variant=variant, resumed=not fresh, metrics=result['metrics'])), flush=True)
        if variant == 'T1':
            reproduce_t1(results, queue)
    summary = dict(**analysis(results, queue), results=results, priors_full=priors, priors_loo=folds,
                   executed=executed, resumed=resumed, selected=selected, sampling=sampling)
    previous = read_json(directory / 'summary.json') if (directory / 'summary.json').exists() else None
    summary['mean_changes_since_previous'] = ({v: {k: summary['means'][v][k] - previous['means'][v][k]
                                                  for k in METRICS} for v in VARIANTS} if previous else None)
    publish(summary, queue, queue_path, directory)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default='track3/features/pose_insurance/heading_queue.json')
    run_queue(parser.parse_args().queue)


if __name__ == '__main__':
    main()
