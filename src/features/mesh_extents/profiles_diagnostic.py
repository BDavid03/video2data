"""Reproduce the frozen CPU R10 kill test and its negative control from cached video geometry."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from src.paths import output_path
from src.provenance import input_file, run_fields, sha256_file, write_json
from src.runs import (ROOT, authorize_episode, heavy_execution, read_json, recorded, runtime_flag, producer, new_run_id)

FEATURE = Path(__file__).parent
REQUEST = 'src/features/mesh_extents/profiles_request.json'


def checked(path, expected, inventory):
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f'Input hash mismatch: {path}')
    inventory[str(path.relative_to(ROOT))] = actual
    return path


def load_frames(run_path, name, recipe, inventory):
    """Rebuild assigned masks and original local geometry from immutable cached depth."""
    import numpy as np
    from . import _profiles as profiles
    from ._geometry import object_mask
    path = ROOT / run_path
    record = read_json(path)
    inventory[run_path] = sha256_file(path)
    if record['status'] != 'completed' or record['episode_index'] in (41, 43, 44):
        raise ValueError('Excluded or incomplete measurement')
    checked(path.parent / 'extents.json', record['artifacts']['extents.json'], inventory)
    request = record['request']
    depth_path = ROOT / request['depth_run']
    depth = read_json(depth_path)
    inventory[request['depth_run']] = sha256_file(depth_path)
    if (depth['status'] != 'completed' or depth['episode_index'] != record['episode_index']
            or depth['split'] != record['split']):
        raise ValueError('Depth parent mismatch')
    prep = checked(depth_path.parent / 'preprocessing.npz', depth['preprocessing_artifact_sha256'], inventory)
    with np.load(prep, allow_pickle=False) as a:
        k, input_valid = a['input_intrinsics_pixels'], a['input_valid'].astype(bool)
    depth_frames = {f['frame_index']: f for f in depth['frames'] if f['status'] == 'completed'}
    reference, admitted, diagnostics = {}, [], []
    used = record['objects'][name]['frames_used']
    for position, row in enumerate(record['frames']):
        index = row['frame_index']
        if position and index not in used:
            diagnostics.append({'frame': index, 'admitted': False, 'reason': 'not in v4 frames_used'})
            continue
        lp = checked(path.parent / f'labels_{index:06d}.npz', record['artifacts'][f'labels_{index:06d}.npz'], inventory)
        with np.load(lp, allow_pickle=False) as a:
            label = a['label']
        if position == 0:
            reference = {n: np.isin(label, np.array(ids) + 1) for n, ids in request['objects'].items()}
            masks = reference
        else:
            masks = {n: object_mask(label, m, request['track_dilation_pixels']) for n, m in reference.items()}
        drow = depth_frames[index]
        dp = checked(depth_path.parent / drow['artifact'], drow['sha256'], inventory)
        with np.load(dp, allow_pickle=False) as arrays:
            z = arrays['depth_m'].astype(np.float64)
            valid = arrays['valid'] & arrays['input_valid'] & input_valid
        frame_id = f"{record['split']}:{record['episode_index']}:{index}"
        prepared, diagnostic = profiles.reconstruct_frame(
            z, valid, k, masks, name, row['plane'], row['objects'][name], request,
            recipe, index in used, frame_id)
        diagnostics.append(diagnostic)
        if diagnostic['admitted']:
            admitted.append({**prepared, 'id': frame_id})
    return admitted, diagnostics


def run_trial(trial, recipe, identity, directory):
    import numpy as np
    from . import _profiles as profiles
    from ._profiles_report import profile_figure
    path = directory / 'trials' / f'{trial["name"]}.json'
    if path.exists():
        result = read_json(path)
        if result['identity'] != identity:
            raise ValueError('Existing result belongs to another recipe/source identity')
        for source, expected in result['input_sha256'].items():
            checked(ROOT/source, expected, {})
        return result
    inventory, frames, diagnostics = {}, [], []
    for run in trial['runs']:
        found, rows = load_frames(run, trial['object'], recipe, inventory)
        frames.extend(found)
        diagnostics.extend(rows)
    frames.sort(key=lambda f: f['id'].split(':')[0] + ':' + ':'.join(f'{int(v):06d}' for v in f['id'].split(':')[1:]))
    result = {'identity': identity, 'trial': trial, 'input_sha256': inventory, 'frames': diagnostics}
    if trial['role'] == 'negative_control':
        result.update(decision='PASS', fit_performed=False, admitted_frames=[f['id'] for f in frames],
                      meaning='mask/table reconstruction only; no vessel or profile inference')
    else:
        gate = profiles.evaluate_gate(frames, recipe)
        result.update(decision=gate['decision'], gate=gate, admitted_frames=[f['id'] for f in frames])
        if 'pooled_fit' in gate:
            result['profile_figure'] = profile_figure(trial['object'], frames, gate, recipe, directory)
            result['pooled_constant_fit'] = profiles.fit_profile(frames, recipe, fixed_radii=np.ones(3))
    write_json(path, result)
    with (directory / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps({'trial': trial['name'], 'identity': identity, 'decision': result['decision'],
                                 'artifact': str(path.relative_to(directory))}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return result


def verify(directory):
    """Recheck the saved gate arithmetic and geometry without fitting again."""
    document = read_json(directory/'report.json')
    admit_queue([row['trial'] for row in document['results']])
    from ._profiles import saved_checks
    recipe = read_json(directory/'recipe.json')
    for path, expected in recipe['sources'].items():
        if sha256_file(ROOT/path) != expected:
            raise ValueError(f'R10 source changed: {path}')
    result = saved_checks(document)
    assert not list(directory.glob('*.ply'))
    write_json(directory/'verification.json', result)
    return result


def admit_queue(queue):
    """Admit every queued producer before importing numerical dependencies."""
    runtime = runtime_flag(FEATURE, 'measure_authorized', 'R10 video profile diagnostic')
    for trial in queue:
        for path in trial['runs']:
            _, record = producer(path, 'mesh_extents.measure')
            if record['split'] != 'public' or record['episode_index'] in (41, 43, 44):
                raise PermissionError('The R10 kill test is public tuning video only')
            authorize_episode(runtime, SimpleNamespace(episode_index=record['episode_index'], source_split=record['split']), 'R10 kill test')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    heavy_execution(runtime)


def run(request_path, destination):
    """Admit the frozen tuning queue, resume completed trials, and preserve the historical report format."""
    from ._profiles_report import markdown
    request = read_json(request_path)
    admit_queue(request['queue'])
    recipe = request['recipe']
    record = run_fields(ROOT, 'mesh_extents', 'profiles_diagnostic', request,
                        inputs=[request_path, *[path for trial in request['queue'] for path in trial['runs']]])
    sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
        Path(__file__), FEATURE/'_profiles.py', FEATURE/'_profiles_report.py', FEATURE/'_geometry.py', FEATURE/'_shapes.py',
        ROOT/'src/features/mesh_extents/requests/build-v4.json')}
    identity = hashlib.sha256(json.dumps(dict(recipe=recipe, queue=request['queue'], sources=sources), sort_keys=True).encode()).hexdigest()
    directory = output_path(ROOT, destination)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'trials').mkdir(exist_ok=True)
    if (directory/'recipe.json').exists() and read_json(directory/'recipe.json')['identity'] != identity:
        raise ValueError('Refusing to overwrite a different run identity')
    write_json(directory/'recipe.json', dict(choices=recipe, identity=identity, sources=sources, state='kill_test_only_not_adopted'))
    # Run records are append-only; the report is the stable historical presentation.
    run_directory = directory/'runs'/new_run_id()
    with recorded(run_directory, record):
        results = [run_trial(trial, recipe, identity, directory) for trial in request['queue']]
        killed = any(row['decision'] != 'PASS' for row in results)
        if not killed:
            raise RuntimeError('Kill test passed; pooled fitting requires a separate request and driver')
        document, text = markdown(results, killed, sha256_file(directory/'recipe.json'))
        write_json(directory/'report.json', document)
        (directory/'report.md').write_text(text)
        for result in results:
            record['inputs'] += [input_file(ROOT, path) for path in result['input_sha256']]
        for name in ('report.json', 'report.md', 'recipe.json', 'white_pot-profile.png'):
            shutil.copy2(directory/name, run_directory/name)
    return directory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--request', default=REQUEST)
    parser.add_argument('--output', required=True, help='New rerun directory, or the same directory to resume')
    parser.add_argument('--verify', action='store_true', help='Recheck saved results without fitting')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(output_path(ROOT, args.output))))
    else:
        print(run(args.request, args.output))


if __name__ == '__main__':
    main()
