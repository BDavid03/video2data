"""Run label-blind HR2 registration; read labels only in a separate errors operation.

Run: CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m
src.features.pose_insurance.registration_loop --queue <queue.json>
Artifacts: immutable per-slot run.json/prediction.json/registrations.npz/landscape.png,
append-only ledger.jsonl, and invocation reports with ranked losses and resume counts.
Development admits only episode 39; evaluation requires the lead's frozen method hash.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from src.paths import output_path, project_path
from src.provenance import input_file, run_fields, sha256_file, write_json
from src.runs import (ROOT, artifact, authorize_episode, checked_entry, heavy_execution,
                         new_run_id, producer, read_json, recorded, runtime_flag, save_npz)

FEATURE = Path(__file__).parent
OUTPUT = 'data/outputs/pose_insurance/registration-hr2'
PROTOCOL = 'docs/chain/pose-heading-hr2-protocol.md'
METHOD = 'docs/chain/pose-heading-method-hr2.json'
DATASET = ('data/datasets--nvidia--video_to_data_challenge/snapshots/'
           '5f68335f3acc802033d1e80728c1633197521de8/track_3/public')
SCAN_NAMES = ('plastic_dish_rack', 'mini_sweeper', 'mini_dust_pan', 'water_pitcher',
              'white_pot', 'white_pot_lid', 'wooden_piece_1', 'wooden_piece_2')
EVALUATION_SLOTS = ((12, 1, 'plastic_dish_rack'), (18, 0, 'mini_sweeper'),
    (18, 1, 'mini_dust_pan'), (21, 1, 'water_pitcher'), (23, 0, 'wooden_piece_1'),
    (23, 1, 'wooden_piece_2'), (24, 0, 'wooden_piece_1'), (24, 1, 'wooden_piece_2'),
    (27, 0, 'plastic_dish_rack'), (27, 1, 'white_pot'), (42, 0, 'white_pot'))
# Hash only dependencies of registration, excluding the independent heading refactor.
CODE_FILES = tuple(f'src/features/pose_insurance/{name}.py' for name in (
    '_registration', 'registration_loop', '_scene', '_length', 'length_diagnostic',
    'placement_diagnostic', 'scale_diagnostic', 'scene', '_placement', '_scale')) + (
    'src/paths.py', 'src/provenance.py', 'src/runs.py', 'src/gpu.py',
    'scripts/track3/static_pose_baseline.py')


def code_hashes() -> dict:
    """Hash the registration implementation and its reused local dependencies."""
    return {name: sha256_file(project_path(ROOT, name)) for name in CODE_FILES}


def environment() -> dict:
    """Admit the CPU runtime before any array or mesh operation."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, 'registration_authorized', 'HR2 registration')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    return runtime


def contract(queue_path: str, method_sha256: str | None) -> tuple:
    """Restrict development to 39 and bind evaluation to externally frozen method bytes."""
    queue = read_json(queue_path)
    roster = [(r['episode'], r['slot'], r['name']) for r in queue['trials']]
    if queue['phase'] == 'development':
        if roster != [(39, 0, 'mini_dust_pan')]:
            raise PermissionError('HR2 development can register only episode 39 slot 0')
        parameters, method = queue['parameters'], None
    elif queue['phase'] == 'evaluation':
        path = project_path(ROOT, METHOD)
        if not method_sha256 or sha256_file(path) != method_sha256:
            raise PermissionError('Evaluation requires the lead-committed method sha256')
        method = read_json(path)
        if method['code_sha256'] != code_hashes() or roster != list(EVALUATION_SLOTS):
            raise PermissionError('Evaluation code or slot panel differs from the frozen method')
        if 'parameters' in queue:
            raise ValueError('Evaluation parameters come exclusively from the frozen method')
        parameters = method['parameters']
    else:
        raise ValueError('Expected development or evaluation phase')
    if not 1 <= queue['max_trials'] <= 11 or not 1 <= queue['max_seconds'] <= 7200:
        raise ValueError('Supply a bounded trial/time budget')
    if (parameters['yaw_step_deg'], parameters['basin_separation_deg'],
        parameters['relative_loss_margin'], parameters['stability_max_deg']) != (5, 30, .05, 10):
        raise ValueError('HR2 outline fixes 5-degree starts and 30/5%/10-degree declines')
    return queue, parameters, method


def base_record(operation: str, request: dict, paths: list, provenance='public_asset_diagnostic') -> dict:
    """Build standard provenance with only this operation's source dependencies."""
    record = run_fields(ROOT, 'pose_insurance', operation, request,
                        inputs=[PROTOCOL, FEATURE / 'runtime.json', *paths], provenance=provenance)
    record['sources'] = [input_file(ROOT, path) for path in (*CODE_FILES, 'data/configs/sources.lock.json')]
    return record


def append(ledger: Path, row: dict) -> None:
    """Durably append one completed or failed trial event."""
    with ledger.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def scan_path(name: str, parameters: dict) -> Path | None:
    """Resolve visual scan files in the method's explicit priority order."""
    for pattern in parameters['scan_resolution_order']:
        path = project_path(ROOT, f'{DATASET}/mesh/{name}/{pattern.format(name=name)}')
        if path.is_file():
            return path
    return None


def inspect_scans(parameters: dict) -> Path:
    """Record geometry-only loadability, vertex counts and bounds for the non-cup scans."""
    import trimesh
    record = base_record('registration_scans', dict(names=SCAN_NAMES), [])
    directory = output_path(ROOT, f'{OUTPUT}/scans/{new_run_id()}')
    rows = []
    with recorded(directory, record):
        for name in SCAN_NAMES:
            path = scan_path(name, parameters)
            if path is None:
                raise FileNotFoundError(f'No visual scan for {name}')
            record['inputs'].append(input_file(ROOT, path))
            mesh = trimesh.load(path, force='mesh', process=False)
            rows.append(dict(name=name, path=path.relative_to(ROOT).as_posix(),
                             vertices=len(mesh.vertices), faces=len(mesh.faces),
                             extents_m=mesh.extents.tolist()))
        write_json(directory / 'scans.json', rows)
    print(json.dumps(dict(scan_report=str(directory.relative_to(ROOT)), scans=rows)), flush=True)
    return directory / 'run.json'


def video_inputs(episode: int, queue: dict, record: dict) -> dict:
    """Admit only the requested frame-zero mask, depth calibration and stereo producers."""
    import numpy as np
    from ._length import cluster_masks, confirmation_masks, stereo_clusters
    from ._scene import camera_points
    from .length_diagnostic import admit, load_stereo
    request = SimpleNamespace(source_split='public', episode_index=episode)
    if episode in (24, 27):
        assignments_path = FEATURE / 'length_confirm_assignments.json'
        state_path = FEATURE / 'length_confirm_preparation.json'
        record['inputs'].extend(input_file(ROOT, p) for p in (assignments_path, state_path))
        entry = read_json(assignments_path)['episodes'][str(episode)]
        measure = producer(entry['measure_run'], 'mesh_extents.measure', request)
        depth = producer(entry['depth_run'], 'depth.generate', request)
        state = read_json(state_path)[str(episode)]
        scale_parent = producer(state['scale'], 'pose_insurance.length_confirmation_scale')
        admit(scale_parent, record)
        scale = read_json(artifact(scale_parent, 'scale.json', record))
        names = entry['names']
    else:
        scale_path = f'data/outputs/pose_insurance/scale/stereo_{episode:06d}.json'
        record['inputs'].append(input_file(ROOT, scale_path))
        scale = read_json(scale_path)
        depth = producer(scale['depth_run'], 'depth.generate', request)
        if episode in (39, 42):
            build = producer(queue['tuning_builds'][str(episode)], 'pose_insurance.build', request)
            admit(build, record)
            measure_path = next(e['path'] for e in build[1]['inputs'] if e['path'].startswith(
                f'data/outputs/mesh_extents/public/episode_{episode:06d}/') and e['path'].endswith('/run.json'))
            names = build[1]['object_names']
        else:
            path = FEATURE / 'placements_train.json'
            record['inputs'].append(input_file(ROOT, path))
            entry = read_json(path)['episodes'][str(episode)]
            measure_path = entry['measure_run']
            from .placement_diagnostic import ROSTER
            names = ROSTER[episode]
        measure = producer(measure_path, 'mesh_extents.measure', request)
    for parent in (measure, depth):
        admit(parent, record)
    if measure[1]['request']['depth_run'] != str(depth[0].relative_to(ROOT)):
        raise ValueError('Mask and stereo calibration depth producers differ')
    with np.load(artifact(measure, 'labels_000000.npz', record), allow_pickle=False) as arrays:
        labels = arrays['label'].copy()
    with np.load(artifact(depth, 'preprocessing.npz', record,
                          depth[1]['preprocessing_artifact_sha256']), allow_pickle=False) as arrays:
        intrinsics = arrays['input_intrinsics_pixels'].copy()
    if episode in (24, 27):
        masks = confirmation_masks(entry, labels)[1]
    else:
        path = FEATURE.parent / 'mesh_extents/assignments.json'
        record['inputs'].append(input_file(ROOT, path))
        assignments = read_json(path)['episodes'][f'public:{episode}']['objects']
        if episode in (39, 42):
            entries = dict(slots={}, merged_cluster_ids=None)
        else:
            entries = entry
        masks = cluster_masks(names, entries, assignments, labels, False)[1]
    stereo = load_stereo(episode, scale, depth, record, request)
    clusters, counts = stereo_clusters(stereo, masks, labels, intrinsics, 3)
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    return dict(names=names, clusters=clusters, counts=counts, labels=labels, valid=valid,
                points=camera_points(stereo['rgb_depth_m'].astype(float), intrinsics),
                normal_hint=np.array(measure[1]['frames'][0]['plane']['normal']))


def landscape(directory: Path, searches: list) -> None:
    """Plot all six start families and the fixed subsampling loss curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for family, ax in enumerate(axes.flat):
        for repeat, row in enumerate(searches):
            count = len(row['losses_m']) // 6
            selected = slice(family * count, (family + 1) * count)
            ax.plot(row['start_yaws_deg'][selected], row['losses_m'][selected] * 1000,
                    label='primary' if repeat == 0 else f'subsample {repeat}', alpha=.75)
        ax.set_title(('+ ' if family % 2 == 0 else '− ') + 'xyz'[family // 2] + ' scan up')
        ax.set_xlabel('Initial table yaw (degrees)')
        ax.set_ylabel('Trimmed distance (mm)')
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / 'landscape.png', dpi=130)
    plt.close(figure)


def register_slot(trial: dict, data: dict, path: Path | None, parameters: dict,
                  directory: Path) -> dict:
    """Write one label-blind prediction and every fitted start for its slot."""
    import numpy as np
    import trimesh
    from ._registration import decision, fit_table, search, surface_samples
    start = time.monotonic()
    slot, name = trial['slot'], trial['name']
    if slot >= len(data['names']) or data['names'][slot] != name:
        raise ValueError('Requested slot name differs from the producer roster')
    observed = data['clusters'][slot]
    row = dict(**trial, accepted=False, decline_reasons=[], stereo_valid_points=data['counts'][slot])
    searches = []
    if path is None:
        row['decline_reasons'] = ['missing_scan']
    elif observed is None:
        row['decline_reasons'] = ['missing_independent_mask_or_stereo']
    else:
        pool = np.concatenate([points for points in data['clusters'] if points is not None])
        try:
            normal, offset, table = fit_table(data['points'], data['valid'], data['labels'],
                                              pool, data['normal_hint'], parameters)
        except ValueError as error:
            row['decline_reasons'] = [f'table_fit: {error}']
        else:
            observed = observed[observed @ normal - offset > parameters['above_table_m']]
            row.update(above_table_points=len(observed), table=table, normal=normal.tolist(), offset_m=offset)
            if len(observed) < parameters['minimum_observed_points']:
                row['decline_reasons'] = ['insufficient_above_table_points']
            else:
                mesh = trimesh.load(path, force='mesh', process=False)
                model = surface_samples(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                        parameters['scan_samples'], parameters['scan_seed'])
                searches = [search(model, observed, normal, offset, parameters, parameters['seed'])]
                for seed in parameters['subsampling_seeds']:
                    searches.append(search(model, observed, normal, offset, parameters, seed,
                                           parameters['subsampling_fraction']))
                row.update(decision(searches[0], searches[1:], parameters))
                best = row['best_index']
                row.update(rotation=searches[0]['rotations'][best].tolist(),
                           translation_m=searches[0]['translations'][best].tolist(),
                           observed_per_search=[int(s['observed_count']) for s in searches])
    arrays = {f'search_{i}_{key}': value for i, search_row in enumerate(searches)
              for key, value in search_row.items()}
    save_npz(directory / 'registrations.npz', np, **arrays)
    if searches:
        landscape(directory, searches)
    row['wall_seconds'] = time.monotonic() - start
    write_json(directory / 'prediction.json', row)
    return row


def resume(row: dict) -> tuple:
    """Verify a completed prediction and all its hashed consumed inputs."""
    parent = producer(row['run'], 'pose_insurance.registration_trial', allowed=('public_asset_diagnostic',))
    for entry in (*parent[1]['inputs'], *parent[1]['sources']):
        checked_entry(entry, {'inputs': []})
    for name in parent[1]['artifacts']:
        artifact(parent, name, {'inputs': []})
    return read_json(parent[0].parent / 'prediction.json'), parent


def predict(queue_path: str, method_sha256: str | None) -> Path:
    """Execute or resume one immutable registration trial per requested episode and slot."""
    runtime = environment()
    queue, parameters, _ = contract(queue_path, method_sha256)
    root = output_path(ROOT, OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / 'ledger.jsonl'
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    started = time.monotonic()
    report_record = base_record('registration_report', queue, [queue_path])
    report_directory = root / 'data/reports' / new_run_id()
    with recorded(report_directory, report_record):
        rows, completed, reused, data_by_episode = [], 0, 0, {}
        for trial in queue['trials']:
            if time.monotonic() - started > queue['max_seconds']:
                raise RuntimeError('HR2 time budget reached; resume the unchanged queue')
            request = SimpleNamespace(source_split='public', episode_index=trial['episode'])
            authorize_episode(runtime, request, 'HR2 registration')
            paths = [queue_path] + ([METHOD] if queue['phase'] == 'evaluation' else [])
            record = base_record('registration_trial', dict(trial=trial, parameters=parameters), paths)
            episode = trial['episode']
            if episode not in data_by_episode:
                data_record = {'inputs': []}
                data_by_episode[episode] = video_inputs(episode, queue, data_record), data_record['inputs']
            data, inputs = data_by_episode[episode]
            record['inputs'].extend(inputs)
            path = scan_path(trial['name'], parameters)
            if path is not None:
                record['inputs'].append(input_file(ROOT, path))
            identity = hashlib.sha256(json.dumps(dict(request=record['request'], inputs=record['inputs'],
                sources=record['sources']), sort_keys=True).encode()).hexdigest()
            record.update(identity=identity, split='public', episode_index=episode,
                          slot=trial['slot'], method_sha256=method_sha256)
            old = next((r for r in reversed(prior) if r['identity'] == identity and r['status'] == 'completed'), None)
            if old:
                result, parent = resume(old)
                reused += 1
            else:
                if completed >= queue['max_trials']:
                    raise RuntimeError('HR2 trial budget reached; resume the unchanged queue')
                directory = root / 'trials' / new_run_id()
                try:
                    with recorded(directory, record):
                        result = register_slot(trial, data, path, parameters, directory)
                except Exception as error:
                    append(ledger, dict(identity=identity, **trial, status='failed',
                        run=str((directory / 'run.json').relative_to(ROOT)), error=str(error)))
                    raise
                parent = (directory / 'run.json', record)
                append(ledger, dict(identity=identity, **trial, status='completed',
                                    run=str(parent[0].relative_to(ROOT))))
                completed += 1
            report_record['inputs'].append(input_file(ROOT, parent[0]))
            rows.append(dict(result, run=str(parent[0].relative_to(ROOT)), resumed=bool(old)))
            print(json.dumps({key: rows[-1].get(key) for key in
                ('episode', 'slot', 'name', 'accepted', 'best_loss_mm', 'decline_reasons', 'resumed', 'wall_seconds')}), flush=True)
        summary = dict(phase=queue['phase'], parameters=parameters, method_sha256=method_sha256,
                       executed=completed, resumed=reused, predictions=rows)
        write_json(report_directory / 'summary.json', summary)
        lines = ['# HR2 label-blind registration', '',
                 f'Executed: {completed}; resumed: {reused}. No reference labels read.', '',
                 '| Episode | Slot | Object | Best loss (mm) | Accepted | Reason | Seconds |',
                 '|---|---|---|---:|---|---|---:|']
        for row in sorted(rows, key=lambda r: r.get('best_loss_mm', float('inf'))):
            lines.append(f"| {row['episode']} | {row['slot']} | {row['name']} | {row.get('best_loss_mm', '—')} | "
                         f"{row['accepted']} | {', '.join(row['decline_reasons'])} | {row['wall_seconds']:.2f} |")
        (report_directory / 'summary.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'latest.json', dict(report=str((report_directory / 'run.json').relative_to(ROOT))))
    print(json.dumps(dict(report=str((report_directory / 'run.json').relative_to(ROOT)),
                         executed=completed, resumed=reused)), flush=True)
    return report_directory / 'run.json'


def errors(report_path: str, method_sha256: str | None) -> Path:
    """Read references only after admitting the completed whole-panel prediction report."""
    environment()
    import numpy as np
    from scipy.spatial.transform import Rotation
    from ._registration import pair_errors, up_error_deg
    from .placement_diagnostic import baseline_module, label_paths
    parent = producer(report_path, 'pose_insurance.registration_report', allowed=('public_asset_diagnostic',))
    request = parent[1]['request']
    report = read_json(artifact(parent, 'summary.json', {'inputs': []}))
    # Apply the same phase restriction before opening any parquet.
    queue_path = next(e['path'] for e in parent[1]['inputs'] if e['path'].endswith('.json')
                      and e['path'] not in (PROTOCOL, str(FEATURE.relative_to(ROOT) / 'runtime.json')))
    queue, parameters, _ = contract(queue_path, method_sha256)
    expected = [(r['episode'], r['slot'], r['name']) for r in queue['trials']]
    actual = [(r['episode'], r['slot'], r['name']) for r in report['predictions']]
    if request != queue or actual != expected or report['method_sha256'] != method_sha256:
        raise ValueError('Prediction report does not match the complete requested phase')
    for row in report['predictions']:
        saved, _ = resume(dict(run=row['run']))
        if saved != {key: value for key, value in row.items() if key not in ('run', 'resumed')}:
            raise ValueError('Report differs from its immutable slot prediction')
    record = base_record('registration_errors', dict(prediction_report=report_path),
                         [parent[0]], provenance='public_label_diagnostic')
    directory = output_path(ROOT, f'{OUTPUT}/errors/{new_run_id()}')
    with recorded(directory, record):
        up, pairs = [], []
        for episode in dict.fromkeys(row['episode'] for row in report['predictions']):
            record['inputs'].extend(input_file(ROOT, path) for path in label_paths(episode))
            names, poses = baseline_module().load_reference(episode)
            rows = [r for r in report['predictions'] if r['episode'] == episode]
            for row in rows:
                error = None
                if 'rotation' in row:
                    ref = Rotation.from_quat(poses[0, 0, names.index(row['name']), 3:]).as_matrix()
                    error = up_error_deg(np.array(row['rotation']), np.array(row['normal']), ref)
                up.append(dict(episode=episode, slot=row['slot'], name=row['name'],
                               accepted=row['accepted'], up_error_deg=error))
            if episode in (18, 23, 24, 27):
                pair = dict(episode=episode, accepted=len(rows) == 2 and all(r['accepted'] for r in rows))
                if pair['accepted']:
                    rows.sort(key=lambda r: r['slot'])
                    pair.update(pair_errors(np.array([r['rotation'] for r in rows]),
                        np.array([r['translation_m'] for r in rows]),
                        poses[0, 0, [names.index(r['name']) for r in rows]]))
                pairs.append(pair)
        write_json(directory / 'errors.json', dict(phase=report['phase'], up=up, pairs=pairs,
                                                  parameters=parameters))
    print(json.dumps(dict(error_report=str((directory / 'run.json').relative_to(ROOT)), up=up, pairs=pairs)), flush=True)
    return directory / 'run.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default='src/features/pose_insurance/registration_queue.json')
    parser.add_argument('--operation', choices=('predict', 'errors', 'scans'), default='predict')
    parser.add_argument('--prediction-report')
    parser.add_argument('--method-sha256')
    args = parser.parse_args()
    if args.operation == 'predict':
        predict(args.queue, args.method_sha256)
    elif args.operation == 'errors':
        if not args.prediction_report:
            parser.error('--prediction-report is required for errors')
        errors(args.prediction_report, args.method_sha256)
    else:
        environment()
        inspect_scans(contract(args.queue, args.method_sha256)[1])


if __name__ == '__main__':
    main()
