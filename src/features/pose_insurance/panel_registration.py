"""Run Stage B's frozen HR2 slots, publish label-free observations and seal A/B."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

from src.paths import output_path
from src.provenance import input_file, sha256_file, write_json
from src.runs import (ROOT, artifact, authorize_episode, checked_entry, now_utc, producer,
                         read_json, recorded)
from .panel_masks import (FEATURE, OUTPUT, QUEUE, METHOD, METHOD_HASH, ROSTER, base_record,
                          contract, environment, resume as resume_masks)
from .registration_h3_loop import append, identity
from .registration_loop import code_hashes, register_slot, scan_path


def registration_inputs(episode: int, record: dict) -> dict:
    """Extract independent Stage A pixels with the unchanged HR2 stereo loader."""
    import cv2
    import numpy as np
    from ._length import stereo_clusters
    from ._scene import camera_points
    root = output_path(ROOT, OUTPUT)
    stage_a = root/f'stage-a/episode_{episode:06d}/run.json'
    row = resume_masks(stage_a)
    record['inputs'].append(input_file(ROOT, stage_a))
    frame_path = root/f'prepared/episode_{episode:06d}/frame.npz'
    evidence_path = root/f'prepared/episode_{episode:06d}/inputs.json'
    parent = producer(stage_a, 'pose_insurance.panel_masks', allowed=('public_asset_diagnostic',))
    for path in (frame_path, evidence_path):
        entry = next(e for e in parent[1]['inputs'] if e['path'] == str(path.relative_to(ROOT)))
        checked_entry(entry, record)
    evidence = read_json(evidence_path)
    if evidence != row['grid']:
        raise ValueError('Prepared frame no longer matches Stage A grid evidence')
    with np.load(frame_path, allow_pickle=False) as arrays:
        stereo = {k: arrays[k].copy() for k in ('rgb_depth_m','rgb_valid','rgb_intrinsics_pixels')}
        labels = arrays['cluster_labels'].copy()
        normal = arrays['normal_hint'].copy()
    masks = []
    for slot in (0,1):
        path = artifact(parent, f'mask_slot_{slot}.png', record)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        masks.append(mask if row['objects'][slot]['box'] is not None else None)
    clusters, counts = stereo_clusters(stereo, masks, labels, np.array(evidence['intrinsics_pixels']), 3)
    if counts != [r['valid_stereo_pixels'] for r in row['objects']]:
        raise ValueError('Stage B stereo counts differ from Stage A')
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    return dict(names=ROSTER[episode], clusters=clusters, counts=counts, labels=labels,
                valid=valid, points=camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels']),
                normal_hint=normal)


def trial_record(trial: dict, queue: dict, queue_path: str, parameters: dict) -> tuple:
    """Bind a resumable slot to the frozen method, scan and completed Stage A."""
    record = base_record('panel_registration', dict(trial=trial, parameters=parameters), [queue_path])
    data = registration_inputs(trial['episode'], record)
    scan = scan_path(trial['name'], parameters)
    if scan is not None:
        record['inputs'].append(input_file(ROOT, scan))
    key = identity(record)
    record.update(identity=key, label_free=True, split='public', episode_index=trial['episode'],
                  slot=trial['slot'], method_sha256=METHOD_HASH)
    return record, data, scan


def fit(trial: dict, queue: dict, queue_path: str, parameters: dict) -> dict:
    """Run one unmodified HR2 fit and persist the panel's stability-only admission."""
    import numpy as np
    runtime = environment('registration_authorized')
    authorize_episode(runtime, SimpleNamespace(source_split='public', episode_index=trial['episode']), 'Panel registration')
    record, data, scan = trial_record(trial, queue, queue_path, parameters)
    directory = output_path(ROOT, f'{OUTPUT}/stage-b/episode_{trial["episode"]:06d}/slot_{trial["slot"]}')
    with recorded(directory, record):
        row = register_slot(trial, data, scan, parameters, directory)
        row['hr2_accepted'] = row['accepted']
        reasons = [reason for reason in row['decline_reasons'] if reason != 'competing_rotation']
        row.update(admitted='rotation' in row and not reasons, admission_reasons=reasons,
                   admission_rule='mask, loadable scan, HR2 fit and subsampling stability <= 10 degrees',
                   label_free=True, rival_rotation=None, rival_translation_m=None)
        if row.get('rival_index') is not None:
            with np.load(directory/'registrations.npz', allow_pickle=False) as arrays:
                index = row['rival_index']
                row['rival_rotation'] = arrays['search_0_rotations'][index].tolist()
                row['rival_translation_m'] = arrays['search_0_translations'][index].tolist()
        write_json(directory/'panel-prediction.json', row)
    return dict(row, run=str((directory/'run.json').relative_to(ROOT)), identity=record['identity'])


def resume_slot(path: str, expected_identity: str) -> dict:
    """Verify every artifact of an already completed slot."""
    parent = producer(path, 'pose_insurance.panel_registration', allowed=('public_asset_diagnostic',))
    if parent[1]['identity'] != expected_identity:
        raise ValueError('Completed slot differs from requested identity')
    for entry in (*parent[1]['inputs'], *parent[1]['sources']):
        checked_entry(entry, {'inputs': []})
    for name in parent[1]['artifacts']:
        artifact(parent, name, {'inputs': []})
    return dict(read_json(parent[0].parent/'panel-prediction.json'), run=path, identity=expected_identity)


def report(root: Path, stage_a: dict, stage_b: dict) -> None:
    """Present admissions, centroid displacement, clearance and descriptive control IoUs."""
    lines = ['# Frame-zero panel: label-free Stages A and B', '',
             'No mocap labels, reference offsets, scores, Stage C arms or gates were read or computed.',
             'Offsets are visible-surface centroid differences, not body origins. Control IoUs compare depth clusters, not segmentation truth.', '',
             '| Episode | Objects | Stage A | Stereo pixels (0 / 1) | d table XYZ (cm) | XY length (cm) | Clearance (cm) | Placement | Control IoU (0 / 1) |',
             '|---|---|---|---:|---|---:|---:|---|---|']
    for row in stage_a['episodes']:
        vector = 'missing' if row['d_table_m'] is None else ', '.join(f'{x*100:.3f}' for x in row['d_table_m'])
        counts = ' / '.join(str(o['valid_stereo_pixels']) for o in row['objects'])
        ious = ' / '.join('—' if o['control_cluster_iou'] is None else f"{o['control_cluster_iou']:.4f}" for o in row['objects'])
        length = '—' if row['horizontal_length_m'] is None else f"{row['horizontal_length_m']*100:.3f}"
        gap = '—' if row['placement']['clearance_m'] is None else f"{row['placement']['clearance_m']*100:.3f}"
        result = 'admitted' if row['admitted'] else '; '.join(row['reasons'])
        lines.append(f"| {row['episode']} | {', '.join(row['names'])} | {result} | {counts} | {vector} | {length} | {gap} | {row['placement']['classification']} | {ious} |")
    lines += ['', '| Episode | Slot | Object | Panel admission | Best loss (mm) | Rival margin (%) | Rival separation (deg) | Stability (deg) | CPU wall minutes |',
              '|---|---:|---|---|---:|---:|---:|---|---:|']
    for row in stage_b['predictions']:
        margin = '—' if row.get('rival_relative_margin') is None else f"{row['rival_relative_margin']*100:.3f}"
        separation = '—' if row.get('rival_separation_deg') is None else f"{row['rival_separation_deg']:.3f}"
        stability = ', '.join(f'{x:.3f}' for x in row.get('stability_deg', [])) or '—'
        result = 'admitted' if row['admitted'] else ', '.join(row['admission_reasons'])
        lines.append(f"| {row['episode']} | {row['slot']} | {row['name']} | {result} | {row.get('best_loss_mm', '—')} | {margin} | {separation} | {stability} | {row['wall_seconds']/60:.2f} |")
    reports = [read_json(p) for p in root.glob('prepared/episode_*/slot_*/*.json')]
    gpu = [r for r in reports if 'peak_allocated_mib' in r]
    leases = [read_json(p) for p in root.glob('gpu-lease-*.json')]
    telemetry = dict(model_wall_seconds=sum(r['wall_seconds'] for r in gpu),
                     peak_model_allocated_mib=max((r['peak_allocated_mib'] for r in gpu), default=None),
                     peak_model_reserved_mib=max((r['peak_reserved_mib'] for r in gpu), default=None),
                     peak_sampled_device_mib=max((r['peak_sampled_total_device_used_mib'] for r in leases), default=None),
                     cpu_slot_wall_minutes=sum(r['wall_seconds'] for r in stage_b['predictions'])/60,
                     stage_b_elapsed_minutes=stage_b['wall_seconds']/60)
    write_json(root/'telemetry.json', telemetry)
    lines += ['', '## Resources', '', '```json', json.dumps(telemetry, indent=2), '```', '',
              'GPU wall seconds include loading and inference in the isolated workers; this is not CUDA kernel time. Device memory is sampled every two seconds and includes display/other applications.', '',
              '## Reproduce or resume', '', '```bash',
              'PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.panel_masks',
              "CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.panel_registration", '```', '',
              'The fixed queue bounds seven mask pairs and six registrations. Completed identities and artifact hashes are checked on resume; unfinished inference is preserved for review. A single GPU lease covers sequential DINO/SAM workers. Three CPU processes use the original HR2 handler and parameters.', '',
              'Each Stage A directory contains per-object JSON, original largest-component masks, overlap-resolved mask PNGs, the contact sheet, episode offsets and a run record. Stage B saves every search, the original HR2 prediction and a separate panel admission with the selected and rival rotations. The seal inventories immutable outputs and source hashes; caches and the open driver logs are excluded.', '',
              'Stage C, label-dependent errors, scored episodes, physical evaluation and packing remain undone by design.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def seal(root: Path, queue_path: str) -> Path:
    """Hash every immutable A/B artifact before handing the panel to a later job."""
    paths = [p for p in sorted(root.rglob('*')) if p.is_file() and
             'cache' not in p.relative_to(root).parts and p.name not in
             ('label-free-seal.json', 'stage-a-driver.log', 'stage-b-driver.log')]
    payload = dict(schema_version=1, label_free=True, sealed_utc=now_utc(),
                   episodes=list(ROSTER), registration_episodes=[0,26,31],
                   protocol=input_file(ROOT, PROTOCOL_PATH), method=input_file(ROOT, METHOD),
                   queue=input_file(ROOT, queue_path), artifacts=[input_file(ROOT,p) for p in paths],
                   sources=[input_file(ROOT, FEATURE/name) for name in
                            ('_panel_masks.py','panel_masks.py','panel_registration.py','panel_queue.json')],
                   excluded=['cache/', 'stage-a-driver.log', 'stage-b-driver.log', 'label-free-seal.json'],
                   no_label_dependent_computations=True)
    write_json(root/'label-free-seal.json', payload)
    return root/'label-free-seal.json'


PROTOCOL_PATH = 'docs/chain/frame0-panel-protocol.md'


def run(queue_path: str = QUEUE) -> Path:
    """Finish all six frozen fits, report their label-free evidence and seal outputs."""
    started = time.monotonic()
    runtime = environment('registration_authorized')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    queue = contract(queue_path)
    method = read_json(METHOD)
    hashes = code_hashes()
    differences = {path: dict(frozen=digest, current=hashes[path]) for path, digest in
                   method['code_sha256'].items() if hashes[path] != digest}
    unused = 'src/features/pose_insurance/_placement.py'
    if set(differences) - {unused}:
        raise PermissionError('Frozen HR2 registration dependency changed')
    write_json(output_path(ROOT, f'{OUTPUT}/hr2-source-check.json'),
               dict(method_sha256=METHOD_HASH, differences=differences,
                    exception='The adapter calls register_slot, not video_inputs; _placement.py is unused.',
                    registration_kernel_unchanged=hashes['src/features/pose_insurance/_registration.py'] ==
                    method['code_sha256']['src/features/pose_insurance/_registration.py']))
    root = output_path(ROOT, OUTPUT)
    stage_a = read_json(root/'stage-a.json')
    if [r['episode'] for r in stage_a['episodes']] != list(ROSTER):
        raise ValueError('Complete seven-pair Stage A report required')
    ledger = root/'stage-b-ledger.jsonl'
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    rows, pending = [], []
    for episode in queue['registration_episodes']:
        authorize_episode(runtime, SimpleNamespace(source_split='public',episode_index=episode), 'Panel registration')
        for slot, name in enumerate(ROSTER[episode]):
            trial = dict(episode=episode, slot=slot, name=name)
            record, _, _ = trial_record(trial, queue, queue_path, method['parameters'])
            previous = next((r for r in reversed(prior) if r['identity'] == record['identity']), None)
            if previous:
                rows.append(resume_slot(previous['run'], record['identity']))
            else:
                pending.append(trial)
    with ProcessPoolExecutor(max_workers=queue['cpu_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(fit, trial, queue, queue_path, method['parameters']): trial for trial in pending}
        for future in as_completed(futures):
            row = future.result()
            append(ledger, dict(identity=row['identity'], status='completed', episode=row['episode'],
                                slot=row['slot'], run=row['run']))
            rows.append(row)
            print(json.dumps({k:row.get(k) for k in ('episode','slot','admitted','best_loss_mm',
                  'rival_relative_margin','rival_separation_deg','stability_deg','wall_seconds')}), flush=True)
    rows.sort(key=lambda r:(queue['registration_episodes'].index(r['episode']), r['slot']))
    stage_b = dict(label_free=True, method_sha256=METHOD_HASH, predictions=rows,
                   executed=len(pending), resumed=6-len(pending), wall_seconds=time.monotonic()-started)
    write_json(root/'stage-b.json', stage_b)
    report(root, stage_a, stage_b)
    path = seal(root, queue_path)
    print(json.dumps(dict(completed=True, label_free=True, seal=str(path.relative_to(ROOT)))), flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default=QUEUE)
    run(parser.parse_args().queue)
