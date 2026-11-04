"""Run the frozen two-object video queue once, with resumable stage evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from src.gpu import GpuLease
from src.provenance import write_json, sha256_file
from src.runs import now_utc
from src.features.moving_reference import masks, camera, track
from src.features.moving_reference.gates import _amended_check
from ._two_object import select_active
from .two_object_task import ROSTER, PROTOCOL

ROOT = masks.ROOT
STATUS = ROOT / 'data/outputs/moving_policy/two-object-status.md'
QUEUE = Path(__file__).with_name('two_object_queue.json')
EPISODES = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33]


def event(stage, message):
    line = f'- {now_utc()} stage {stage}: {message}'
    with STATUS.open('a') as stream:
        stream.write(line+'\n')
    print(line, flush=True)


def run_masks(episode, name, directory, queue):
    """Reuse the unchanged M1 detector, selection, propagator and GPU lease."""
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'run.json'
    if result_path.exists():
        record = json.loads(result_path.read_text())
        if record['status'] != 'completed':
            raise RuntimeError('Existing incomplete mask attempt cannot be retried')
        return record
    with (directory / 'once.json').open('x') as stream:
        json.dump(dict(episode=episode, object=name, started_utc=now_utc()), stream)
    rule = json.loads((masks.FEATURE / 'runtime.json').read_text())
    rule.update(queue=[dict(split='evaluation', episode=episode)], pan_prompts=[name.replace('_', ' ')],
                stop_utc=queue['stop_utc'])
    job = masks.prepare(rule['queue'][0], rule, directory)
    record = dict(kind='moving_policy.two_object_masks', status='running', episode=episode,
                  object_name=name, stages={}, started_utc=now_utc())
    write_json(result_path, record)
    event(2, f'ep{episode} {name}: one-shot masks started')
    started = time.perf_counter()
    lease = None
    try:
        with GpuLease(ROOT, 1024, 10000, stage='moving_reference.masks') as lease:
            for operation, runtime in [('detect', 'grounding-dino'), ('segment', 'sam2')]:
                lease.require_headroom()
                command = [str(ROOT / f'data/.runtime/{runtime}/venv/bin/python'), '-B', '-m',
                           'src.features.moving_reference._worker', operation, str(directory/'request.json')]
                with (directory/f'{operation}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=masks.environment(directory),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                record['stages'][operation] = json.loads((directory/f'{operation}-report.json').read_text())
                if operation == 'detect' and masks.select_pan_box(directory, rule) is None:
                    record['status'] = 'no_object_detection'
                    break
            else:
                record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started, completed_utc=now_utc(),
                      gpu=lease.report() if lease else None)
        write_json(result_path, record)
        masks.summarize(directory, job, record)
        event(2, f'ep{episode} {name}: masks {record["status"]}; {directory.relative_to(ROOT)}')
    return record


def roles(directories):
    """Measure both full-clip mask-centre paths against independent background flow."""
    jobs = [json.loads((d/'request.json').read_text()) for d in directories]
    if jobs[0]['timestamps'] != jobs[1]['timestamps']:
        raise ValueError('Role masks have different video clocks')
    detections = [json.loads((d/'detections.json').read_text())['pan'] for d in directories]
    if detections[0]['box'] == detections[1]['box']:
        return dict(passed=False, reason='identical detector boxes; no independent object instances')
    centres, motion = [], []
    previous = previous_allowed = None
    for index in range(jobs[0]['frame_count']):
        image = cv2.imread(str(directories[0]/'input_frames'/f'{index:06d}.jpg'), cv2.IMREAD_GRAYSCALE)
        arrays = []
        for directory in directories:
            with np.load(directory/'frames'/f'frame_{index:06d}.npz', allow_pickle=False) as saved:
                arrays.append(saved['masks'].copy())
        combined = np.logical_or.reduce([a.any(axis=0) for a in arrays])
        allowed = ~cv2.dilate(combined.astype(np.uint8), np.ones((21,21),np.uint8)).astype(bool)
        points = []
        for a in arrays:
            yy, xx = np.nonzero(a[0])
            points.append([float(np.median(xx)), float(np.median(yy))] if len(xx) else [np.nan,np.nan])
        centres.append(points)
        if index:
            pixels = camera.corners(previous, previous_allowed)
            flowed, good = camera.flow(previous, image, pixels, previous_allowed, allowed)
            if not good.any():
                return dict(passed=False, reason=f'no valid background optical flow at frame {index}')
            motion.append(np.median(flowed[good]-pixels[good], axis=0).tolist())
        else:
            motion.append([0.,0.])
        previous, previous_allowed = image, allowed
    try:
        result = select_active(np.asarray(centres), np.asarray(motion), image.shape[1])
    except ValueError as error:
        return dict(passed=False, reason=str(error))
    return dict(passed=True, **result, frames=len(centres), centres=centres, background_steps=motion)


def selected_inputs(episode, directory):
    """Bind only the selected evaluation videos and shared public camera calibration."""
    job = json.loads((directory/'request.json').read_text())
    source = json.loads((ROOT/'data/configs/sources.lock.json').read_text())['dataset']
    from src.paths import project_path
    snapshot = project_path(ROOT, Path(source['cache_path'])/'snapshots'/source['revision'])
    paths = [snapshot/f'track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4'
             for key in ('c','b')]
    calibration = snapshot/'track_3/public/meta/camera_calibration.json'
    if sha256_file(calibration) != json.loads((ROOT/'data/configs/calibration/ego.json').read_text())['source_sha256']:
        raise ValueError('Calibration changed from its audited source')
    audit = json.loads((ROOT/'data/reports/data-audit/videos.json').read_text())
    for path in paths:
        row, = [r for r in audit['videos'] if r['path'] == str(path.relative_to(snapshot))]
        if row['pts_seconds'] != job['timestamps'] or row['decoded_frame_count'] != job['frame_count']:
            raise ValueError('Stereo and mask clocks differ')
    return dict(paths=paths, calibration=calibration, masks=directory, job=job,
                mask_files=[directory/'frames'/f'frame_{i:06d}.npz' for i in range(job['frame_count'])])


def mask_gates(selected):
    """Measure the three unchanged spoon/38 operational mask gates."""
    maps, _, q, _ = camera.rectification(json.loads(selected['calibration'].read_text()))
    streams = track.captures(selected['paths'])
    counts, stereo = [], []
    try:
        for index in range(selected['job']['frame_count']):
            _, gray = track.read_pair(streams, maps)
            xyz, _ = camera.stereo(*gray, q)
            _, _, visible, _ = track.masks_at(selected, index, maps, gray[0].shape)
            _, supported = camera.lift(camera.corners(gray[0], visible), xyz)
            counts.append(int(supported.sum()))
            stereo.append(int((visible & np.isfinite(xyz).all(2)).sum()))
            if index % 60 == 0:
                print(f'mask gate frame {index}: {counts[-1]} points, {stereo[-1]} stereo pixels', flush=True)
    finally:
        for stream in streams:
            stream.release()
    usable = np.asarray(counts) >= 30
    missing = np.flatnonzero(~usable)
    chunks = np.split(missing, np.flatnonzero(np.diff(missing)>1)+1)
    gates = dict(mask_usable_fraction=_amended_check(float(usable.mean()),.9,'ge'),
        mask_longest_unusable_interval_frames=_amended_check(max(map(len,chunks)),2),
        object_30_stereo_correspondence_fraction=_amended_check(float(np.mean(np.asarray(stereo)>=30)),.9,'ge'))
    return dict(passed=all(g['passed'] for g in gates.values()), gates=gates,
                texture_counts=counts, stereo_pixels=stereo, labels_read=False, scan_reads=[])


def summary(output, rows):
    lines = ['# Two-object screen', '', '| Episode | Stage | Pass | Result |', '| ---: | --- | --- | --- |']
    for r in rows:
        lines.append(f'| {r["episode"]} | {r["stage"]} | {r["passed"]} | {r.get("reason", "see result.json")} |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'summary.json',dict(episodes=rows, completed_utc=now_utc()))


def run(queue_path=QUEUE):
    """Resume finished episodes; stop at a passing reference or route-wide error."""
    cv2.setNumThreads(2)
    cv2.setRNGSeed(31)
    queue = json.loads(queue_path.read_text())
    if queue['episodes'] != EPISODES or queue['seed'] != 0 or queue['final_steps'] != 66763776:
        raise ValueError('Frozen queue or training recipe differs')
    protocol_hash = subprocess.check_output(['git','hash-object',str(PROTOCOL)],cwd=ROOT,text=True).strip()
    if f'PROTOCOL FROZEN ' not in STATUS.read_text() or protocol_hash not in STATUS.read_text():
        raise PermissionError('Protocol must be frozen before any scored video operation')
    proof = json.loads((ROOT/'data/.runtime/tmp/two-object-build/equivalence.json').read_text())
    loader = json.loads((ROOT/'data/.runtime/tmp/two-object-build/joint-loader.json').read_text())
    if not proof['single_object_parquet_byte_identical'] or len(loader) != 2:
        raise PermissionError('Stage 1 equivalence and joint loader evidence are required')
    output = ROOT/queue['output']
    output.mkdir(parents=True,exist_ok=True)
    identity = dict(queue_sha256=sha256_file(queue_path), protocol_hash_object=protocol_hash,
                    sources={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*two_object*.py')})
    identity_path = output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Screen identity changed; do not resume')
    if not identity_path.exists():
        write_json(identity_path,identity)
    rows = []
    roster = json.loads(ROSTER.read_text())['episodes']
    for episode in queue['episodes']:
        directory = output/f'ep{episode}'
        directory.mkdir(exist_ok=True)
        result_path = directory/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            rows.append(result)
            if result['passed']:
                break
            continue
        if datetime.now(timezone.utc) >= datetime.fromisoformat(queue['last_training_start_utc']):
            event('STOP', 'No new candidate can leave the 4500-second training budget and packing time before 17:00 UTC')
            break
        with (directory/'attempt.json').open('x') as stream:
            json.dump(dict(episode=episode,identity=identity,started_utc=now_utc()),stream)
        try:
            names = roster[str(episode)]
            masks_dirs = [directory/f'masks-{slot}' for slot in (0,1)]
            result = dict(episode=episode, passed=False, stage='detection')
            for name, target in zip(names, masks_dirs):
                produced = run_masks(episode,name,target,queue)
                if produced['status'] != 'completed':
                    result['reason'] = f'no detection for {name}'
                    break
            else:
                role = roles(masks_dirs)
                write_json(directory/'roles.json',role)
                result.update(stage='roles',reason=role.get('reason'))
                if role['passed']:
                    active = role['active_slot']
                    event(2,f'ep{episode}: A={names[active]}, B={names[1-active]}, motion p95={role["displacement_p95_image_width"]}')
                    selected = selected_inputs(episode,masks_dirs[active])
                    gate = mask_gates(selected)
                    write_json(directory/'mask-gates.json',gate)
                    result.update(stage='mask-gates', reason='; '.join(f'{k}={g["value"]} {g["comparison"]} {g["threshold"]}' for k,g in gate['gates'].items()))
                    if gate['passed']:
                        from .two_object_reference import run as reference
                        reference_result = reference(episode, selected, directory/'candidate-c', masks_dirs[1-active])
                        result.update(stage='reference', passed=reference_result['passed'],
                                      reason=reference_result['reason'], active_slot=active)
            result['completed_utc'] = now_utc()
            write_json(result_path,result)
            with (output/'ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            rows.append(result)
            summary(output,rows)
            event(2,f'ep{episode}: {"PASS" if result["passed"] else "STOP episode"} at {result["stage"]}: {result.get("reason")}')
            if result['passed']:
                event(2,f'first passing reference ep{episode}; no later episode attempted')
                break
        except BaseException as error:
            write_json(directory/'route-error.json',dict(error=f'{type(error).__name__}: {error}',time_utc=now_utc()))
            event('STOP',f'route-wide execution error ep{episode}: {type(error).__name__}: {error}; no retry')
            raise
    summary(output,rows)
    event(2,f'screen paused/finished with {len(rows)} terminal episodes and {sum(r["passed"] for r in rows)} passing references')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=QUEUE)
    run(parser.parse_args().queue)
