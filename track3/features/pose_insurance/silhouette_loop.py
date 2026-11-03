"""Run the frozen CPU S1 prediction panel, then a separately recorded assessment."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, checked_entry,
                         heavy_execution, new_run_id, now_utc, producer, read_json,
                         recorded, runtime_flag, save_npz)

FEATURE = Path(__file__).parent
QUEUE = 'track3/features/pose_insurance/silhouette_queue.json'
OUTPUT = 'outputs/pose_insurance/silhouette'
CONTRACT_HASH = '05c08adfc2709d7c625bd76d571dd4cefd844e938b4b17da31eec0d19a143cce'
SOURCES = tuple(f'track3/features/pose_insurance/{name}.py' for name in (
    '_silhouette', 'silhouette_loop', '_registration', '_scene')) + (
    'track3/paths.py', 'track3/provenance.py', 'track3/runs.py', 'configs/sources.lock.json')


def environment() -> dict:
    """Admit S1's CPU interpreter and direct caches into its scratch directory."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, 'silhouette_authorized', 'S1 silhouette diagnostic')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    for key in CACHE_FOLDERS:
        folder = output_path(ROOT, f'.runtime/tmp/silhouette/cache/{key.lower()}')
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    return runtime


def contract(queue_path: str, runtime: dict) -> dict:
    """Bind the exact input panel and thresholds to the committed S1 protocol."""
    queue = read_json(queue_path)
    spec = {key: queue[key] for key in ('protocol', 'method', 'thresholds', 'trials')}
    if hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() != CONTRACT_HASH:
        raise PermissionError('S1 frozen roster, inputs or thresholds differ')
    if not 1 <= queue['max_trials'] <= 13 or not 1 <= queue['max_seconds'] <= 7200:
        raise ValueError('S1 requires a bounded 13-object queue')
    for trial in queue['trials']:
        if trial['episode'] in (41, 43, 44):
            raise PermissionError('Excluded S1 episode')
        authorize_episode(runtime, SimpleNamespace(source_split=trial['source_split'],
                          episode_index=trial['episode']), 'S1 diagnostic')
    return queue


def base_record(operation: str, request: dict, queue_path: str, queue: dict,
                labels: bool = False) -> dict:
    """Record the queue, frozen protocol, runtime and explicit implementation sources."""
    record = run_fields(ROOT, 'pose_insurance', operation, request,
        inputs=[queue_path, FEATURE / 'runtime.json'],
        provenance='public_label_diagnostic' if labels else 'public_asset_diagnostic')
    for key in ('protocol', 'method'):
        checked_entry(queue[key], record)
    record['sources'] = [input_file(ROOT, p) for p in SOURCES]
    return record


def identity(record: dict) -> str:
    """Bind a completed trial to its request, input bytes and source bytes."""
    return hashlib.sha256(json.dumps({k: record[k] for k in
        ('request', 'inputs', 'sources')}, sort_keys=True).encode()).hexdigest()


def admit_trial(trial: dict, record: dict) -> tuple:
    """Re-admit the registration and the recorded mask, calibration, scan and assignment."""
    checked_entry(trial['registration'], record)
    parent = producer(trial['registration']['path'], 'pose_insurance.registration_trial',
                      allowed=('public_asset_diagnostic',))
    for key, name in (('prediction', 'prediction.json'), ('registrations', 'registrations.npz')):
        path = artifact(parent, name, record)
        if input_file(ROOT, path) != trial[key]:
            raise ValueError('S1 registration artifact differs from its frozen queue')
    for key in ('mask', 'grid', 'scan', 'assignment'):
        if trial[key] not in parent[1]['inputs']:
            raise ValueError(f'Registration did not consume S1 {key}')
        checked_entry(trial[key], record)
    request = SimpleNamespace(source_split=trial['source_split'], episode_index=trial['episode'])
    for key, kind in (('mask', 'mesh_extents.measure'), ('grid', 'depth.generate')):
        path = Path(trial[key]['path'])
        upstream = producer(path.parent / 'run.json', kind, request)
        record['inputs'].append(input_file(ROOT, upstream[0]))
        artifact(upstream, path.name, record,
                 upstream[1]['preprocessing_artifact_sha256'] if key == 'grid' else None)
        if key == 'mask' and upstream[1]['frames'][0]['frame_index'] != 0:
            raise ValueError('S1 requires frame-zero masks')
    assigned = read_json(trial['assignment']['path'])
    if trial['episode'] == 19:
        ids = assigned['observations']['cluster_ids'][trial['slot']]
    elif trial['episode'] in (24, 27):
        ids = assigned['episodes'][str(trial['episode'])]['objects'][trial['name']]
    else:
        ids = assigned['episodes'][f"public:{trial['episode']}"]['objects'][trial['name']]
    if ids != trial['cluster_ids']:
        raise ValueError('S1 mask assignment differs from the registration assignment')
    prediction = read_json(trial['prediction']['path'])
    if any(prediction[k] != trial[k] for k in ('episode', 'slot', 'name')):
        raise ValueError('Registration prediction differs from the S1 slot')
    return parent, prediction


def load_arrays(trial: dict) -> tuple:
    """Load admitted arrays and visual geometry without reading any labels."""
    import numpy as np
    import trimesh
    with np.load(project_path(ROOT, trial['registrations']['path']), allow_pickle=False) as arrays:
        searches = [{k: arrays[f'search_{i}_{k}'].copy() for k in
                     ('rotations', 'translations', 'losses_m')} for i in range(4)]
    with np.load(project_path(ROOT, trial['mask']['path']), allow_pickle=False) as arrays:
        mask = np.isin(arrays['label'], np.array(trial['cluster_ids']) + 1)
    with np.load(project_path(ROOT, trial['grid']['path']), allow_pickle=False) as arrays:
        intrinsics = arrays['input_intrinsics_pixels'].copy()
        shape = arrays['input_valid'].shape
    if (mask.shape != (576, 768) or mask.shape != shape or intrinsics.shape != (3, 3)
            or not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.array_equal(intrinsics[2], [0, 0, 1])):
        raise ValueError('Invalid S1 mask grid or pinhole intrinsics')
    for search in searches:
        if (search['rotations'].shape != (432, 3, 3) or search['translations'].shape != (432, 3)
                or search['losses_m'].shape != (432,)
                or any(not np.isfinite(a).all() for a in search.values())
                or search['losses_m'].min() <= 0):
            raise ValueError('Invalid S1 registration array contract')
    mesh = trimesh.load(project_path(ROOT, trial['scan']['path']), force='mesh', process=False)
    return searches, np.asarray(mesh.vertices), np.asarray(mesh.faces), mask, intrinsics


def append(path: Path, row: dict) -> None:
    """Durably append one completed or failed trial event."""
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def resume(path: str, record: dict) -> tuple:
    """Re-admit a completed S1 trial and all its owned artifacts."""
    parent = producer(path, 'pose_insurance.silhouette_trial', allowed=('public_asset_diagnostic',))
    record['inputs'].append(input_file(ROOT, parent[0]))
    result = read_json(artifact(parent, 'prediction.json', record))
    for name in parent[1]['artifacts']:
        if name != 'prediction.json':
            artifact(parent, name, record)
    return parent, result


def prediction_markdown(rows: list) -> str:
    """Format all candidate scores, margins and abstention reasons without labels."""
    lines = ['# S1 predictions', '', 'IoU variants: original / eroded / dilated. Chamfer units: pixels.', '',
             '| Episode/object | Search | Candidate/index | Loss ratio | IoU O/E/D | Chamfer O/E/D (px) |',
             '|---|---:|---|---:|---|---|']
    for row in rows:
        for search, candidates in enumerate(row['searches']):
            inverse = {local: primary for primary, local in enumerate(row['mappings'][search])}
            for local, c in enumerate(candidates):
                iou = '/'.join('NA' if x is None else f'{x:.6f}' for x in c['iou'])
                chamfer = '/'.join('NA' if x is None else f'{x:.4f}' for x in c['contour_chamfer_px'])
                label = f"c{inverse[local]}" if local in inverse else 'unmatched'
                lines.append(f"| {row['episode']} {row['name']} | {search} | {label}/{c['index']} | "
                             f"{c['loss_ratio']:.6f} | {iou} | {chamfer} |")
    lines += ['', '| Episode/object | Role | Selected identity | Minimum margin | Abstain reasons |',
              '|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['episode']} {row['name']} | {row['role']} | {row['decision']['selected']} | "
                     f"{row['decision']['minimum_margin']} | {'; '.join(row['abstain_reasons'])} |")
    return '\n'.join(lines) + '\n'


def figure(rows: list, directory: Path) -> None:
    """Overlay search-zero candidate outlines for the three requested example objects."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, key in zip(axes, ((18, 'mini_sweeper'), (19, 'mini_sweeper'), (23, 'wooden_piece_1'))):
        row = next(r for r in rows if (r['episode'], r['name']) == key)
        with np.load(project_path(ROOT, row['run']).parent / 'renders.npz', allow_pickle=False) as arrays:
            ax.imshow(arrays['mask'], cmap='gray', vmin=0, vmax=1)
            for local, candidate in enumerate(row['searches'][0]):
                contours, _ = cv2.findContours(arrays[f'search_0_candidate_{local}'].astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                color = plt.get_cmap('tab10')(local % 10)
                for contour in contours:
                    xy = contour[:, 0]
                    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=.8)
                ax.plot([], [], color=color, label=f"c{local}, loss +{100*(candidate['loss_ratio']-1):.2f}%, "
                        f"IoU {candidate['iou'][0]:.4f}" if candidate['iou'][0] is not None else f'c{local}: invalid')
        ax.set_title(f'Episode {key[0]}: {key[1]}')
        ax.set_xlabel('camera A u (px)')
        ax.set_ylabel('camera A v (px)')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=8)
    fig.suptitle('S1 saved registration basins — original mask IoU (unitless)')
    fig.text(.5, .015, 'masks are depth-derived partial regions', ha='center')
    fig.tight_layout(rect=(0, .14, 1, .95))
    fig.savefig(directory / 'candidate-outlines.png', dpi=140)
    plt.close(fig)


def predict(queue_path: str = QUEUE) -> Path:
    """Execute or resume the whole 13-object panel before producing its frozen report."""
    runtime = environment()
    queue = contract(queue_path, runtime)
    import numpy as np
    from ._silhouette import evaluate
    root = output_path(ROOT, OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / 'ledger.jsonl'
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    report = base_record('silhouette_predictions', queue, queue_path, queue)
    directory = root / 'predictions' / new_run_id()
    started, executed, reused, rows = time.monotonic(), 0, 0, []
    with recorded(directory, report):
        for trial in queue['trials']:
            if time.monotonic() - started > queue['max_seconds']:
                raise RuntimeError('S1 time budget reached; resume the unchanged queue')
            record = base_record('silhouette_trial', dict(trial=trial, thresholds=queue['thresholds']), queue_path, queue)
            _, original = admit_trial(trial, record)
            key = identity(record)
            record.update(identity=key, split=trial['source_split'], episode_index=trial['episode'])
            old = next((r for r in reversed(prior) if r['identity'] == key and r['status'] == 'completed'), None)
            if old:
                parent, result = resume(old['run'], report)
                reused += 1
            else:
                if executed >= queue['max_trials']:
                    raise RuntimeError('S1 trial budget reached; resume the unchanged queue')
                trial_dir = root / 'trials' / new_run_id()
                try:
                    with recorded(trial_dir, record):
                        searches, vertices, faces, mask, intrinsics = load_arrays(trial)
                        if int(np.argmin(searches[0]['losses_m'])) != original['best_index']:
                            raise ValueError('S1 depth-best differs from the saved registration')
                        result, renders = evaluate(searches, vertices, faces, mask, intrinsics,
                                                   queue['thresholds'], trial['role'] == 'control')
                        result.update({k: trial[k] for k in ('episode', 'slot', 'name', 'role')})
                        result.update(prediction_saved_utc=now_utc(), label_free=True)
                        write_json(trial_dir / 'prediction.json', result)
                        save_npz(trial_dir / 'renders.npz', np, **renders)
                except Exception as error:
                    append(ledger, dict(identity=key, status='failed', run=str((trial_dir/'run.json').relative_to(ROOT)),
                                        error=str(error)))
                    raise
                parent = (trial_dir / 'run.json', record)
                append(ledger, dict(identity=key, status='completed', run=str(parent[0].relative_to(ROOT))))
                report['inputs'].append(input_file(ROOT, parent[0]))
                executed += 1
            rows.append(dict(result, run=str(parent[0].relative_to(ROOT))))
            print(json.dumps(dict(episode=trial['episode'], name=trial['name'], resumed=bool(old),
                                  decision=result['decision'])), flush=True)
        summary = dict(predictions=rows, executed=executed, resumed=reused,
                       predictions_complete_utc=now_utc(), label_free=True)
        write_json(directory / 'summary.json', summary)
        (directory / 'summary.md').write_text(prediction_markdown(rows))
        figure(rows, directory)
    write_json(root / 'predictions.json', dict(report=str((directory/'run.json').relative_to(ROOT))))
    (root / 'predictions.md').write_text(prediction_markdown(rows))
    print(json.dumps(dict(prediction_report=str((directory/'run.json').relative_to(ROOT)),
                          executed=executed, resumed=reused, finished_utc=report['finished_utc'])), flush=True)
    return directory / 'run.json'


def saved_predictions(report_path: str, record: dict, queue: dict) -> tuple:
    """Refuse assessment unless every frozen trial has a completed hashed prediction."""
    parent = producer(report_path, 'pose_insurance.silhouette_predictions', allowed=('public_asset_diagnostic',))
    if parent[1]['request'] != queue:
        raise ValueError('S1 assessment requires the exact prediction queue')
    record['inputs'].append(input_file(ROOT, parent[0]))
    summary = read_json(artifact(parent, 'summary.json', record))
    rows = summary['predictions']
    if len(rows) != 13 or [(r['episode'], r['slot'], r['name']) for r in rows] != [
            (t['episode'], t['slot'], t['name']) for t in queue['trials']]:
        raise PermissionError('All 13 S1 predictions must exist before label assessment')
    for row in rows:
        entry = next(e for e in parent[1]['inputs'] if e['path'] == row['run'])
        checked_entry(entry, record)
        trial_parent, saved = resume(row['run'], record)
        if saved != {k: v for k, v in row.items() if k != 'run'}:
            raise ValueError('S1 report differs from its saved trial prediction')
        if datetime.fromisoformat(trial_parent[1]['finished_utc']) > datetime.fromisoformat(parent[1]['finished_utc']):
            raise ValueError('S1 prediction report predates a trial')
        for e in (*trial_parent[1]['inputs'], *trial_parent[1]['sources']):
            checked_entry(e, record)
    return rows, parent


def references(queue: dict, record: dict) -> dict:
    """Read only frame-zero mocap objects for the three spent public episodes."""
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation
    dataset = Path(next(t['scan']['path'] for t in queue['trials'])).parents[2]
    result = {}
    for episode in (18, 23, 24):
        path = dataset / f'data/chunk-000/episode_{episode:06d}.parquet'
        record['inputs'].append(input_file(ROOT, path))
        objects = pq.read_table(project_path(ROOT, path), columns=['observation.objects'])['observation.objects'][0].as_py()
        result[episode] = {}
        for obj in objects:
            if not obj['visible']:
                raise ValueError('S1 requires visible frame-zero references')
            q = obj['pose'][3:]
            result[episode][obj['name']] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return result


def assess_case(trial: dict, prediction: dict, reference: dict, queue: dict) -> dict:
    """Apply the precommitted correctness and identity-agreement definitions."""
    import numpy as np
    from ._silhouette import label_errors
    with np.load(project_path(ROOT, trial['registrations']['path']), allow_pickle=False) as arrays:
        rotations = arrays['search_0_rotations'][[c['index'] for c in prediction['searches'][0]]]
    original = read_json(trial['prediction']['path'])
    if trial['episode'] == 18:
        pan_trial = next(t for t in queue['trials'] if t['episode'] == 18 and t['name'] == 'mini_dust_pan')
        pan = np.array(read_json(pan_trial['prediction']['path'])['rotation'])
        errors = label_errors(rotations, np.array(original['normal']), reference[trial['name']],
                              partner=pan, partner_reference=reference['mini_dust_pan'])
        correct = np.array(errors['C_error_deg']) < queue['thresholds']['C_error_deg']
        testable = int(correct.sum()) == 1
        reason = None if testable else 'requires_exactly_one_correct_candidate'
    else:
        canonical = np.array([1. if trial['name'].endswith('_1') else -1., 0, 0])
        errors = label_errors(rotations, np.array(original['normal']), reference[trial['name']], canonical)
        correct = np.array(errors['up_error_deg']) < queue['thresholds']['up_error_deg']
        signed = np.array(errors['canonical_up_error_deg']) < queue['thresholds']['up_error_deg']
        testable = bool(np.array_equal(correct, signed) and correct.any())
        reason = 'identity_uncertain' if not np.array_equal(correct, signed) else (
            'no_correct_candidate' if not correct.any() else None)
    selected = prediction['decision']['selected']
    selected_correct = None if selected is None or not testable else bool(correct[selected])
    return dict(episode=trial['episode'], name=trial['name'], testable=testable, reason=reason,
                correct_candidates=np.flatnonzero(correct).tolist(), selected=selected,
                selected_correct=selected_correct, verdict='not_testable' if not testable else (
                    'abstain' if selected is None else 'correct' if selected_correct else 'WRONG'), **errors)


def assessment_markdown(result: dict) -> str:
    """Present the known-answer cases, control completeness and withheld application."""
    lines = ['# S1 assessment', '', f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**. "
             f"Episode 19: **{result['episode_19']}**.", '',
             f"Predictions completed: {result['predictions_finished_utc']}; "
             f"assessment started: {result['assessment_started_utc']}.", '',
             '| Episode/object | Testable | Correct identities | Selected | Verdict/reason |',
             '|---|---|---|---|---|']
    for case in result['known_cases']:
        lines.append(f"| {case['episode']} {case['name']} | {case['testable']} | {case['correct_candidates']} | "
                     f"{case['selected']} | {case['verdict']}: {case['reason']} |")
    lines += ['', '| Control | Complete | Per-search selected identities | Veto | Reasons |',
              '|---|---|---|---|---|']
    for control in result['controls']:
        lines.append(f"| {control['episode']} {control['name']} | {control['complete']} | "
                     f"{control['search_selections']} | {control['veto']} | {'; '.join(control['reasons'])} |")
    lines += ['', f"Known-answer pass: {result['known_passed']}; controls complete: {result['controls_complete']}; "
              f"control veto: {result['control_veto']}.", '',
              'No wooden decision establishes wooden-flip resolution. Controls test consistency with depth, '
              'not truth. H3 and submissions are unchanged.', '']
    return '\n'.join(lines)


def assess(report_path: str, queue_path: str = QUEUE) -> Path:
    """Assess frozen predictions only after admitting the completed 13-object report."""
    runtime = environment()
    queue = contract(queue_path, runtime)
    record = base_record('silhouette_assessment', dict(prediction_report=str(report_path)), queue_path, queue, labels=True)
    rows, parent = saved_predictions(report_path, record, queue)
    directory = output_path(ROOT, f'{OUTPUT}/assessments/{new_run_id()}')
    with recorded(directory, record):
        if datetime.fromisoformat(parent[1]['finished_utc']) >= datetime.fromisoformat(record['started_utc']):
            raise PermissionError('S1 predictions must finish before assessment')
        reference = references(queue, record)
        cases = [assess_case(t, p, reference[t['episode']], queue)
                 for t, p in zip(queue['trials'], rows) if t['role'] == 'known']
        sweeper = next(c for c in cases if c['episode'] == 18)
        known_passed = sweeper['testable'] and sweeper['selected'] == 0 and all(
            c['selected_correct'] is not False for c in cases if c['testable'])
        controls = []
        for row in rows:
            if row['role'] != 'control':
                continue
            failures = [r for r in row['abstain_reasons'] if r != 'no_common_winner_with_margin']
            selections = [d['selected'] for d in row['search_decisions']]
            controls.append(dict(episode=row['episode'], name=row['name'], complete=not failures,
                                 search_selections=selections, veto=1 in selections, reasons=failures))
        complete = all(c['complete'] for c in controls)
        veto = any(c['veto'] for c in controls)
        passed = bool(known_passed and complete and not veto)
        application = next(r for r in rows if r['role'] == 'application')
        selected = application['decision']['selected']
        outcome = 'application withheld' if not passed else 'abstain' if selected is None else (
            'agree' if selected == 0 else 'disagree')
        result = dict(passed=passed, known_passed=bool(known_passed), known_cases=cases, controls=controls,
                      controls_complete=complete, control_veto=veto, episode_19=outcome,
                      predictions_finished_utc=parent[1]['finished_utc'], assessment_started_utc=record['started_utc'],
                      assessment_computed_utc=now_utc(), prediction_report=str(report_path))
        write_json(directory / 'assessment.json', result)
        (directory / 'summary.md').write_text(assessment_markdown(result))
    root = output_path(ROOT, OUTPUT)
    write_json(root / 'assessment.json', dict(report=str((directory/'run.json').relative_to(ROOT))))
    (root / 'summary.md').write_text(assessment_markdown(result))
    print(json.dumps(dict(assessment_report=str((directory/'run.json').relative_to(ROOT)),
                          passed=passed, episode_19=outcome, finished_utc=record['finished_utc'])), flush=True)
    return directory / 'run.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default=QUEUE)
    parser.add_argument('--operation', choices=('predict', 'assess', 'all'), default='all')
    parser.add_argument('--prediction-report')
    args = parser.parse_args()
    if args.operation in ('predict', 'all'):
        report = predict(args.queue)
    else:
        if not args.prediction_report:
            parser.error('Assessment requires --prediction-report from a complete prediction operation')
        report = args.prediction_report
    if args.operation in ('assess', 'all'):
        assess(report, args.queue)


if __name__ == '__main__':
    main()
