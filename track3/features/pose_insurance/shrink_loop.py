"""Resume the complete fixed CPU shrink queue and its conditional train-only H follow-up."""
import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
import time
from track3.provenance import input_file, read_run, sha256_file, write_json
from track3.runs import ROOT, read_json, source_hashes, artifact, authorize_episode, new_run_id, checked_entry
from .placement_diagnostic import FEATURE
from .scale_loop import append
from ._shrink import (EPISODES, CHANGED, CONTROLS, CANDIDATES, assignments, best_factor, separation_factor)
from .shrink_diagnostic import OUTPUT, ShrinkRequest, cpu_environment, inputs, diagnose, episode_data
from .shrink_report import analysis, descriptive, h_gates, summarize


def identity(request: ShrinkRequest, stereo: dict, queue: dict) -> str:
    """Hash settings, all stereo producers, exact inputs and both slices' sources."""
    _, paths, parents = inputs(request, stereo, queue)
    payload = dict(request=asdict(request), inputs=[input_file(ROOT, p) for p in (*paths, *parents)],
                   sources=source_hashes(FEATURE), stereo_sources=source_hashes(FEATURE.parent / 'stereo_depth'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resume_result(row: dict) -> dict:
    """Admit a completed run and re-hash its consumed inputs and owned result."""
    record = read_run(ROOT, row['run'], kind='pose_insurance.shrink_diagnostic')
    for entry in record['inputs']:
        checked_entry(entry, {'inputs': []})
    return read_json(artifact((ROOT / row['run'], record), 'metrics.json', {'inputs': []}))


def record_plan(path, value: dict, ledger, stage: str) -> None:
    """Persist an identical-or-new construction and fsync its ledger row before scoring."""
    if path.exists() and read_json(path) != json.loads(json.dumps(value)):
        raise ValueError(f'Construction changed; use a fresh reviewed output directory: {path}')
    if not path.exists():
        write_json(path, value)
    append(ledger, dict(stage=stage, status='recorded_before_scoring', plan=value,
                        input=input_file(ROOT, path)))


def h_construction(plan: dict, scores: dict, queue: dict) -> dict:
    """Construct every train/control H factor before scoring any H scene."""
    choices = dict(C25=.25, C50=.5, C75=.75)
    chosen = best_factor(scores, choices)
    c_h = choices[chosen]
    rows, factors = {}, {}
    for ep in (*EPISODES, *CONTROLS):
        geometry = episode_data(ep, queue['tuning_builds'].get(str(ep)))[0]
        if ep in CONTROLS:
            row = dict(status='unchanged: single object', c_sep=None)
        else:
            row = separation_factor(geometry)
        if row['status'].startswith('fallback'):
            factor = plan['factors']['S1'][ep]
        elif row['status'].startswith('unchanged'):
            factor = 1.
        else:
            factor = max(c_h, row['c_sep'])
        rows[ep], factors[ep] = dict(row, applied_factor=factor), factor
    return dict(c_H=c_h, selected_curve_candidate=chosen, episodes=rows, factors=factors,
                composition='Acts once on v2 planar offset, replacing S1; no mocap input')


def run_queue(queue_path: str) -> None:
    """Run all pending candidates, oracle grids and the mechanically triggered H stage."""
    runtime = cpu_environment()
    queue = read_json(queue_path)
    if (queue['episodes'] != list(EPISODES) or queue['controls'] != list(CONTROLS)
            or queue['candidates'] != list(CANDIDATES)):
        raise ValueError('Queue must retain the exact registered episodes and candidates')
    for ep in (*EPISODES, *CONTROLS):
        authorize_episode(runtime, ShrinkRequest(ep, 'I', 1.), 'shrink queue')
    directory = ROOT / OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    previous = [directory / n for n in ('summary.json', 'summary.md', 'finished.json', 'length_assignments.npz')
                if (directory / n).exists()]
    if previous:
        backup = ROOT / '.runtime/tmp/backup' / new_run_id()
        backup.mkdir(parents=True)
        for path in previous:
            shutil.copy2(path, backup / path.name)
    stereo = {ep: read_json(f'outputs/pose_insurance/scale/stereo_{ep:06d}.json')
              for ep in (*EPISODES, *CONTROLS)}
    # Re-admit every recorded anchor before taking its estimate as a fixed input.
    for row in stereo.values():
        for entry in row['anchor_inputs']:
            checked_entry(entry, {'inputs': []})
        for path in row['anchor_runs']:
            read_run(ROOT, path, kind='stereo_depth.anchor')
    plan = assignments(stereo)
    ledger = directory / 'ledger.jsonl'
    record_plan(directory / 'assignments.json', plan, ledger, 'assignments')
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    state = dict(executed=0, resumed=0, selected=[], failures=[])
    started = time.perf_counter()

    def trial(request):
        key = identity(request, stereo, queue)
        prior = next((r for r in reversed(rows) if r.get('identity') == key and r['status'] == 'completed'), None)
        if prior:
            result = resume_result(prior)
            state['resumed'] += 1
            state['selected'].append(prior)
            return result
        if state['executed'] >= queue['max_trials'] or time.perf_counter() - started >= queue['max_seconds']:
            raise RuntimeError('Budget reached; resume the unchanged queue')
        tick = time.perf_counter()
        row = dict(stage='trial', identity=key, request=asdict(request))
        try:
            path = diagnose(request, stereo, queue)
            row.update(status='completed', run=str(path.relative_to(ROOT)))
            result = resume_result(row)
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            if hasattr(error, 'feature_run_path'):
                row['run'] = error.feature_run_path
            state['failures'].append(row.copy())
            result = None
        finally:
            row['seconds'] = time.perf_counter() - tick
            append(ledger, row)
        rows.append(row)
        state['executed'] += 1
        state['selected'].append(row)
        print(json.dumps(row), flush=True)
        return result

    results = {c: {} for c in CANDIDATES}
    for c in CANDIDATES:
        for ep in (*EPISODES, *CONTROLS):
            value = trial(ShrinkRequest(ep, c, plan['factors'][c][ep]))
            if value is not None:
                results[c][ep] = value
    report = analysis(results, plan)
    oracles = {}
    for ep in CHANGED:
        value = trial(ShrinkRequest(ep, 'oracle', 1.))
        if value is not None:
            oracles[ep] = value
    descriptive(report, oracles, stereo, plan, directory)
    if report['verdicts']['2']['label'] == 'Further shrinkage helps':
        if not all(report['eligible'].get(c, False) for c in ('C25', 'C50', 'C75', 'S1')):
            report['H'] = dict(status='blocked: missing or disqualified construction comparator')
        else:
            construction = h_construction(plan, report['means8'], queue)
            record_plan(directory / 'H_construction.json', construction, ledger, 'H_construction')
            results['H'] = {}
            for ep in (*EPISODES, *CONTROLS):
                value = trial(ShrinkRequest(ep, 'H', construction['factors'][ep]))
                if value is not None:
                    results['H'][ep] = value
            readings = report['verdicts']
            report = analysis(results, plan)
            report['verdicts'] = readings
            report['H'] = dict(status='train stage completed; scored stage not run', construction=construction,
                               gates=h_gates(results, report))
    else:
        report['H'] = dict(status='not built: verdict 2 did not trigger')
    report.update(executed=state['executed'], resumed=state['resumed'], failures=state['failures'])
    summarize(report, results, oracles, plan, directory)
    write_json(directory / 'finished.json', dict(status='completed', queue_sha256=sha256_file(ROOT / queue_path),
                                               selected=state['selected'], verdicts=report['verdicts'], H=report['H']))
    print(directory / 'summary.md', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default='track3/features/pose_insurance/shrink_queue.json')
    args = parser.parse_args()
    run_queue(args.queue)


if __name__ == '__main__':
    main()
