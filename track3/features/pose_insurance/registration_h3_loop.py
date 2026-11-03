"""Record the frozen HR2 registration on the two scored episode-19 slots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, checked_entry,
                         heavy_execution, new_run_id, producer, read_json, recorded, runtime_flag)
from .registration_loop import CODE_FILES, METHOD, code_hashes, register_slot, scan_path
from .request import Request

FEATURE = Path(__file__).parent
OUTPUT = 'outputs/pose_insurance/registration-h3'
QUEUE = 'track3/features/pose_insurance/registration_h3_queue.json'
PROTOCOL = 'docs/chain/pose-heading-h3-protocol.md'
METHOD_HASH = 'c1fd86776f0a5e39eaf0f79273cbb8c6098ef585b59783b00a30ef1d65772e25'
SLOTS = [(19, 0, 'mini_sweeper'), (19, 1, 'mini_dust_pan')]


def environment(flag: str) -> dict:
    """Admit the existing CPU gate and keep every writable cache in H3 scratch."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, flag, 'H3 CPU operation')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    scratch = output_path(ROOT, '.runtime/tmp/h3/cache')
    for key in CACHE_FOLDERS:
        path = scratch / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    return runtime


def append(path: Path, row: dict) -> None:
    """Flush each completed trial before the next trial starts."""
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(record: dict) -> str:
    """Bind a trial to its request, consumed bytes and implementation."""
    return hashlib.sha256(json.dumps({key: record[key] for key in
        ('request', 'inputs', 'sources')}, sort_keys=True).encode()).hexdigest()


def resume(path: str, kind: str) -> tuple:
    """Re-admit all consumed bytes and every completed artifact."""
    parent = producer(path, kind, allowed=('public_asset_diagnostic', 'public_label_diagnostic'))
    for entry in (*parent[1]['inputs'], *parent[1]['sources']):
        checked_entry(entry, {'inputs': []})
    for name in parent[1]['artifacts']:
        artifact(parent, name, {'inputs': []})
    return parent


def contract(queue_path: str) -> tuple:
    """Admit exactly the scored pair with the unchanged frozen HR2 implementation."""
    queue = read_json(queue_path)
    roster = [(r['episode'], r['slot'], r['name']) for r in queue['trials']]
    if queue['source_split'] != 'evaluation' or roster != SLOTS or 'parameters' in queue:
        raise PermissionError('H3 registration admits only evaluation 19 sweeper and dust pan')
    if not 1 <= queue['max_trials'] <= 2 or not 1 <= queue['max_seconds'] <= 7200:
        raise ValueError('H3 registration requires a bounded two-slot queue')
    method_path = checked_entry(queue['method'], {'inputs': []})
    method = read_json(method_path)
    if queue['method']['path'] != METHOD or queue['method']['sha256'] != METHOD_HASH:
        raise PermissionError('H3 requires the frozen HR2 method')
    if method['code_sha256'] != code_hashes():
        raise PermissionError('Frozen HR2 implementation changed')
    checked_entry(queue['protocol'], {'inputs': []})
    if queue['protocol']['path'] != PROTOCOL:
        raise PermissionError('H3 protocol required')
    return queue, method


def base_record(operation: str, request: dict, paths: list) -> dict:
    """Record the adapter and its frozen registration dependencies."""
    record = run_fields(ROOT, 'pose_insurance', operation, request,
                        inputs=[PROTOCOL, METHOD, FEATURE / 'runtime.json', *paths],
                        provenance='public_asset_diagnostic')
    record['sources'] = [input_file(ROOT, p) for p in (*CODE_FILES,
        'track3/features/pose_insurance/registration_h3_loop.py',
        'track3/features/pose_insurance/request.py', 'configs/sources.lock.json')]
    return record


def video_inputs(queue: dict, record: dict, request: Request) -> dict:
    """Load the exact frame-zero masks, grid and stereo named in D2 v4's plan."""
    import numpy as np
    from ._length import stereo_clusters
    from ._scene import camera_points
    from .length_diagnostic import frame_arrays
    plan_path = checked_entry(queue['scored_plan'], record)
    plan = read_json(plan_path)
    if plan['episode'] != 19 or [r['name'] for r in plan['scene']] != [s[2] for s in SLOTS]:
        raise ValueError('Scored plan differs from the episode-19 roster')
    for entry in plan['inputs']:
        checked_entry(entry, record)
    observation = plan['observations']
    measure = producer(observation['measure_run'], 'mesh_extents.measure', request)
    stereo = producer(observation['stereo_run'], 'stereo_depth.generate', request)
    depth = producer(measure[1]['request']['depth_run'], 'depth.generate', request)
    for parent in (measure, stereo, depth):
        if input_file(ROOT, parent[0]) not in record['inputs']:
            raise ValueError('Scored plan did not admit the requested video producer')
    original = read_json('outputs/pose_insurance/v3/plans/episode_000019.json')
    old_build = producer(original['base_build'], 'pose_insurance.build', request)
    if input_file(ROOT, measure[0]) not in old_build[1]['inputs']:
        raise ValueError('T1 original scene did not consume this plane measurement')
    artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])
    with np.load(artifact(measure, 'labels_000000.npz', record), allow_pickle=False) as arrays:
        labels = arrays['label'].copy()
    with np.load(project_path(ROOT, depth[0].parent / 'preprocessing.npz'), allow_pickle=False) as grid:
        intrinsics = grid['input_intrinsics_pixels'].copy()
    stereo_arrays = frame_arrays(stereo, record)
    if (int(stereo_arrays['frame_index']) != 0 or int(stereo_arrays['pts']) != 0
            or stereo[1]['grid'] != depth[1]['preprocessing_artifact_sha256']
            or measure[1]['frames'][0]['frame_index'] != 0):
        raise ValueError('Expected exact frame-zero colour-camera grid')
    masks = [np.isin(labels, np.array(ids) + 1) for ids in observation['cluster_ids']]
    clusters, counts = stereo_clusters(stereo_arrays, masks, labels, intrinsics, 3)
    if counts != observation['stereo_valid_pixels']:
        raise ValueError('Stereo counts differ from the scored placement stage')
    valid = (stereo_arrays['rgb_valid'] & np.isfinite(stereo_arrays['rgb_depth_m'])
             & (stereo_arrays['rgb_depth_m'] > 0))
    return dict(names=[s[2] for s in SLOTS], clusters=clusters, counts=counts, labels=labels,
                valid=valid, points=camera_points(stereo_arrays['rgb_depth_m'].astype(float), intrinsics),
                normal_hint=np.array(measure[1]['frames'][0]['plane']['normal']))


def predict(queue_path: str = QUEUE) -> Path:
    """Run or resume both frozen registrations before publishing the prediction report."""
    started = time.monotonic()
    runtime = environment('registration_authorized')
    queue, method = contract(queue_path)
    request = Request(source_split='evaluation', episode_index=19)
    authorize_episode(runtime, request, 'H3 registration')
    root = output_path(ROOT, OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / 'ledger.jsonl'
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    report = base_record('registration_h3_report', queue, [queue_path])
    directory = root / 'reports' / new_run_id()
    with recorded(directory, report):
        data_record = {'inputs': []}
        data = video_inputs(queue, data_record, request)
        rows, executed, reused = [], 0, 0
        for trial in queue['trials']:
            if time.monotonic() - started > queue['max_seconds']:
                raise RuntimeError('H3 registration time budget reached; resume queue')
            record = base_record('registration_trial', dict(trial=trial, parameters=method['parameters']), [queue_path])
            record['inputs'].extend(data_record['inputs'])
            scan = scan_path(trial['name'], method['parameters'])
            if scan is not None:
                record['inputs'].append(input_file(ROOT, scan))
            key = identity(record)
            record.update(identity=key, split='evaluation', episode_index=19, slot=trial['slot'],
                          method_sha256=METHOD_HASH)
            old = next((r for r in reversed(prior) if r['identity'] == key and r['status'] == 'completed'), None)
            if old:
                parent = resume(old['run'], 'pose_insurance.registration_trial')
                row = read_json(parent[0].parent / 'prediction.json')
                reused += 1
            else:
                if executed >= queue['max_trials']:
                    raise RuntimeError('H3 registration trial budget reached; resume queue')
                trial_dir = root / 'trials' / new_run_id()
                with recorded(trial_dir, record):
                    row = register_slot(trial, data, scan, method['parameters'], trial_dir)
                parent = trial_dir / 'run.json', record
                append(ledger, dict(identity=key, status='completed', run=str(parent[0].relative_to(ROOT))))
                executed += 1
            report['inputs'].append(input_file(ROOT, parent[0]))
            rows.append(dict(row, run=str(parent[0].relative_to(ROOT))))
            print(json.dumps(dict(slot=trial['slot'], accepted=row['accepted'],
                                  seconds=row['wall_seconds'], resumed=bool(old))), flush=True)
        summary = dict(predictions=rows, executed=executed, resumed=reused,
                       method_sha256=METHOD_HASH, wall_seconds=time.monotonic() - started,
                       label_free=True, source_split='evaluation', episode=19)
        write_json(directory / 'summary.json', summary)
    write_json(root / 'latest.json', dict(report=str((directory / 'run.json').relative_to(ROOT))))
    print(json.dumps(summary), flush=True)
    return directory / 'run.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default=QUEUE)
    predict(parser.parse_args().queue)
