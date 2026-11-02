"""Resume the registered CPU stereo-scale queue and write its adoption report."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import shutil
import time
from track3.paths import output_path
from track3.provenance import input_file, read_run, sha256_file, write_json
from track3.runs import (ROOT, read_json, source_hashes, runtime_flag, authorize_episode,
                         heavy_execution, producer, artifact, new_run_id)
from .placement_diagnostic import FEATURE, ROSTER
from .scale_diagnostic import ScaleRequest, OUTPUT, train_parents, trial_inputs, diagnose
from .scale_stereo import episode_stereo
from .scale_report import summarize


def identity(request, stereo, queue):
    """Hash exact trial settings, protocol, anchor producers and computation sources."""
    _, inputs, parents = trial_inputs(request, stereo, queue)
    payload = dict(request=asdict(request), scale=stereo,
                   inputs=[input_file(ROOT, p) for p in (*inputs, *parents)],
                   sources=source_hashes(FEATURE),
                   stereo_sources=source_hashes(FEATURE.parent / 'stereo_depth'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resume_result(row):
    """Admit the completed run, its inputs, and its owned metrics artifact."""
    record = read_run(ROOT, row['run'], kind='pose_insurance.scale_diagnostic')
    for entry in record['inputs']:
        if input_file(ROOT, entry['path']) != entry:
            raise ValueError(f"Scale input changed: {entry['path']}")
    return read_json(artifact((ROOT / row['run'], record), 'metrics.json', {'inputs': []}))


def append(ledger, row):
    """Durably append one completed or failed operation to the ledger."""
    with ledger.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def depth_run(episode, queue):
    """Resolve scene depth from the reviewed train placement or existing tuning build."""
    if episode in ROSTER:
        return str(train_parents(episode)[1][0].relative_to(ROOT))
    build = producer(queue['tuning_builds'][str(episode)], 'pose_insurance.build')
    matches = [entry['path'] for entry in build[1]['inputs']
               if '/depth/public/' in entry['path'] and entry['path'].endswith('/run.json')]
    if len(matches) != 1:
        raise ValueError('Tuning scene must pin exactly one depth run')
    return matches[0]


def run_queue(queue_path):
    """Run all stereo episodes and all missing fixed trials without agent intervention."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    runtime = runtime_flag(FEATURE, 'scale_diagnostic_authorized', 'Scale queue')
    queue = read_json(queue_path)
    if (queue['episodes'] != list(ROSTER) or queue['controls'] != [39, 42]
            or queue['candidates'] != ['incumbent', 'S1', 'S2']):
        raise ValueError('Queue must retain exactly the registered episodes and candidates')
    episodes = queue['episodes'] + queue['controls']
    for ep in episodes:
        authorize_episode(runtime, ScaleRequest(ep, 'incumbent'), 'scale queue')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    directory = output_path(ROOT, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    existing = [p for p in directory.iterdir() if p.is_file() and p.name != 'ledger.jsonl']
    if existing:
        backup = output_path(ROOT, f'.runtime/tmp/ctx05-scale-work/backups/{new_run_id()}')
        backup.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, backup / path.name)
    ledger = directory / 'ledger.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    started = time.perf_counter()
    stereo, selected, results = {}, [], {c: {} for c in queue['candidates']}
    executed = skipped = 0
    for ep in episodes:
        if time.perf_counter() - started >= queue['max_seconds']:
            raise RuntimeError('Time budget reached; resume unchanged queue')
        row = dict(stage='stereo', episode=ep)
        try:
            stereo[ep] = episode_stereo(ep, depth_run(ep, queue))
            row.update(status='completed', anchors=stereo[ep]['anchor_runs'],
                       estimate={k: stereo[ep][k] for k in ('s', 'observed_frames', 'valid')})
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            append(ledger, row)
        print(json.dumps(row), flush=True)
    for candidate in queue['candidates']:
        for ep in episodes:
            request = ScaleRequest(ep, candidate)
            key = identity(request, stereo[ep], queue)
            prior = next((r for r in reversed(rows) if r.get('identity') == key and r['status'] == 'completed'), None)
            if prior:
                row = prior
                result = resume_result(row)
                skipped += 1
            else:
                if executed >= queue['max_trials'] or time.perf_counter() - started >= queue['max_seconds']:
                    raise RuntimeError('Trial/time budget reached; resume unchanged queue')
                tick = time.perf_counter()
                row = dict(stage='trial', identity=key, episode=ep, candidate=candidate)
                try:
                    path = diagnose(request, stereo[ep], queue)
                    row.update(status='completed', run=str(path.relative_to(ROOT)))
                    result = resume_result(row)
                    row['metrics'] = result['metrics']
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
                    if hasattr(error, 'feature_run_path'):
                        row['run'] = error.feature_run_path
                    raise
                finally:
                    row['seconds'] = time.perf_counter() - tick
                    append(ledger, row)
                rows.append(row)
                executed += 1
            results[candidate][ep] = result
            selected.append(row)
            print(json.dumps(dict(candidate=candidate, episode=ep, resumed=bool(prior), metrics=result['metrics'])), flush=True)
    report = summarize(results, stereo, queue, directory, executed, skipped)
    write_json(directory / 'finished.json', dict(status='completed', queue_sha256=sha256_file(ROOT / queue_path),
               selected=selected, executed=executed, resumed=skipped, decision=report['decision']))
    print(directory / 'summary.md', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default='track3/features/pose_insurance/scale_queue.json')
    args = parser.parse_args()
    run_queue(args.queue)


if __name__ == '__main__':
    main()
