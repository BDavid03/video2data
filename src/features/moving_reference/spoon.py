"""One-shot, object-name spoon reference operations under the fixed spoon protocol."""

import argparse
from datetime import datetime, timezone
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from src.gpu import GpuLease
from src.provenance import write_json
from src.runs import now_utc
from src.paths import project_path
from . import masks
from . import camera, track
from ._final_gates import FINAL_LIMITS
from .gates import _amended_check

ROOT = masks.ROOT
STATUS = ROOT / 'data/outputs/moving_policy/spoon-status.md'


def inputs(episode):
    """Select only this evaluation clip's videos, masks and shared calibration."""
    if episode not in (35, 36, 37):
        raise PermissionError('Only spoon episodes 35, 36, 37')
    directory = ROOT / f'data/outputs/moving_reference/ep{episode}/masks'
    job = json.loads((directory / 'request.json').read_text())
    source = json.loads((ROOT / 'data/configs/sources.lock.json').read_text())['dataset']
    snapshot = project_path(ROOT, Path(source['cache_path']) / 'snapshots' / source['revision'])
    paths = [snapshot / f'track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4'
             for key in ('c', 'b')]
    calibration = snapshot / 'track_3/public/meta/camera_calibration.json'
    from src.provenance import sha256_file
    if sha256_file(calibration) != json.loads((ROOT / 'data/configs/calibration/ego.json').read_text())['source_sha256']:
        raise ValueError('Calibration differs from audited source')
    clocks = json.loads((ROOT / 'data/reports/data-audit/videos.json').read_text())
    for path in paths:
        row, = [r for r in clocks['videos'] if r['path'] == str(path.relative_to(snapshot))]
        if row['pts_seconds'] != job['timestamps'] or row['decoded_frame_count'] != job['frame_count']:
            raise ValueError('Stereo and mask clocks differ')
    return dict(paths=paths, calibration=calibration, masks=directory, job=job,
                mask_files=[directory / 'frames' / f'frame_{i:06d}.npz' for i in range(job['frame_count'])])


def mask_gates(episode):
    """Measure the existing operational mask gates before spending time on the camera solve."""
    cv2.setNumThreads(2)
    directory = ROOT / f'data/outputs/moving_reference/ep{episode}'
    marker = directory / 'mask-gates-once.json'
    if not marker.exists():
        with marker.open('x') as stream:
            json.dump(dict(started_utc=now_utc()), stream)
    elif not (directory / 'path-resolution-defect.json').exists():
        raise FileExistsError('Mask gate attempt already exists')
    selected = inputs(episode)
    with (directory / 'mask-measurement-started.json').open('x') as stream:
        json.dump(dict(started_utc=now_utc()), stream)
    maps, k, q, _ = camera.rectification(json.loads(selected['calibration'].read_text()))
    streams = track.captures(selected['paths'])
    counts = []
    stereo_pixels = []
    start = time.perf_counter()
    try:
        for index in range(selected['job']['frame_count']):
            _, gray = track.read_pair(streams, maps)
            xyz, _ = camera.stereo(*gray, q)
            _, _, visible, _ = track.masks_at(selected, index, maps, gray[0].shape)
            points = camera.corners(gray[0], visible)
            _, supported = camera.lift(points, xyz)
            counts.append(int(supported.sum()))
            stereo_pixels.append(int((visible & np.isfinite(xyz).all(2)).sum()))
            if index % 40 == 0:
                print(f'ep{episode} mask gate frame {index}: {counts[-1]} supported texture points', flush=True)
    finally:
        for stream in streams:
            stream.release()
    usable = np.asarray(counts) >= 30
    missing = np.flatnonzero(~usable)
    chunks = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    measured = dict(mask_usable_fraction=_amended_check(float(usable.mean()), .9, 'ge'),
                    mask_longest_unusable_interval_frames=_amended_check(max(map(len, chunks)), 2),
                    object_30_stereo_correspondence_fraction=_amended_check(float(np.mean(np.asarray(stereo_pixels) >= 30)), .9, 'ge'))
    passed = all(value['passed'] for value in measured.values())
    result = dict(episode=episode, passed=passed, gates=measured, texture_counts=counts,
                  stereo_pixels=stereo_pixels, wall_seconds=time.perf_counter()-start,
                  definition='Unchanged track.py/camera.py mask usability: >=30 stereo-supported eroded non-hand texture points; amended object correspondence gate counts stereo pixels',
                  labels_read=False, scan_reads=[], completed_utc=now_utc())
    write_json(directory / 'mask-gates.json', result)
    failed = [key for key, value in measured.items() if not value['passed']]
    event(episode, f'mask gates {"PASS" if passed else "STOP; keep v5"}: '
          f'usable={usable.mean():.6f} >=0.9; longest gap={max(map(len, chunks))} <=2; '
          f'stereo support fraction={measured["object_30_stereo_correspondence_fraction"]["value"]:.6f} >=0.9; '
          f'{directory.relative_to(ROOT)}/mask-gates.json')
    if failed:
        gates = {name:dict(value=None, threshold=limit, comparison=comparison, passed=False,
                          status='not_run_after_first_mask_gate_miss')
                 for name, (comparison, limit) in FINAL_LIMITS.items()}
        gates.update(measured)
        write_json(directory / 'gates-final.json', dict(episode=episode, passed=False, status='STOP',
                   gates=gates, failed_measured=failed, stage0_run=False, training_run=False,
                   verdict='keep v5', labels_read=False, scan_reads=[]))
        policy = ROOT / f'data/outputs/moving_policy/ep{episode}-v1'
        policy.mkdir(parents=True, exist_ok=True)
        lines = [f'# Episode {episode}: STOP — keep v5', '',
                 'Reference mask gate failed. No camera solve, tracker, Stage 0, training or scoring follows this miss.', '',
                 '| Gate | Value | Rule | Verdict |', '| --- | ---: | --- | --- |']
        for name, gate in gates.items():
            lines.append(f'| {name} | {gate["value"]} | {gate["comparison"]} {gate["threshold"]} | '
                         f'{"PASS" if gate["passed"] else "FAIL" if name in measured else "NOT RUN"} |')
        lines += ['', 'No labels or scans read. No gate limits changed. No code defect repair or retry.',
                  f'Reference measurements: `data/outputs/moving_reference/ep{episode}/mask-gates.json`.',
                  'Spoon video references have no labelled validation; self-reference checks would not establish accuracy.']
        (policy / 'result.md').write_text('\n'.join(lines) + '\n')
    return passed


def event(episode, message):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open('a') as stream:
        stream.write(f'- {now_utc()} ep{episode}: {message}\n')
    print(f'ep{episode}: {message}', flush=True)


def run_masks(episode, object_name='spoon'):
    """Reuse the unchanged detector and propagator with one object-name prompt."""
    if episode not in (35, 36, 37) or object_name != 'spoon':
        raise PermissionError('Only the three protocol spoon episodes are admitted')
    directory = ROOT / f'data/outputs/moving_reference/ep{episode}/masks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'once.json').open('x') as stream:
        json.dump(dict(started_utc=now_utc(), episode=episode, object_name=object_name), stream)
    rule = json.loads((masks.FEATURE / 'runtime.json').read_text())
    rule.update(queue=[dict(split='evaluation', episode=episode)], pan_prompts=[object_name],
                stop_utc='2026-11-04T15:00:00+00:00')
    job = masks.prepare(rule['queue'][0], rule, directory)
    record = dict(kind='moving_reference.spoon_masks', status='running', stages={},
                  episode=episode, object_name=object_name, started_utc=now_utc())
    start = time.perf_counter()
    write_json(directory / 'run.json', record)
    event(episode, 'masks started; prompt spoon; smallest qualifying box; one shot')
    lease = None
    try:
        with GpuLease(ROOT, 1024, 10000, stage='moving_reference.masks') as lease:
            for operation, runtime in [('detect', 'grounding-dino'), ('segment', 'sam2')]:
                lease.require_headroom()
                command = [str(ROOT / f'data/.runtime/{runtime}/venv/bin/python'), '-B', '-m',
                           'src.features.moving_reference._worker', operation, str(directory / 'request.json')]
                with (directory / f'{operation}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=masks.environment(directory),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                record['stages'][operation] = json.loads((directory / f'{operation}-report.json').read_text())
                if operation == 'detect':
                    selected = masks.select_pan_box(directory, rule)
                    write_json(directory / 'object-detection.json', dict(object_name=object_name, selected=selected))
                    if selected is None:
                        record['status'] = 'no_object_detection'
                        break
                write_json(directory / 'run.json', record)
            else:
                record['status'] = 'completed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-start, completed_utc=now_utc(),
                      gpu=lease.report() if lease else None)
        write_json(directory / 'run.json', record)
        summary = masks.summarize(directory, job, record)
        event(episode, f'masks {record["status"]}; {summary["saved_frames"]}/{job["frame_count"]} frames; '
              f'nonempty object coverage {summary["tracks"]["pan"]}; {directory.relative_to(ROOT)}/summary.json')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['masks', 'mask-gates'])
    parser.add_argument('--episode', type=int, choices=[35, 36, 37], required=True)
    parser.add_argument('--object-name', default='spoon')
    args = parser.parse_args()
    if args.operation == 'masks':
        run_masks(args.episode, args.object_name)
    else:
        mask_gates(args.episode)
