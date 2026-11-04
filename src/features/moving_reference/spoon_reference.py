"""Resume the spoon reference queue after the disclosed mask-usability waiver."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import camera_ba, candidate_c, spoon
from ._final_gates import FINAL_LIMITS
from ._spoon_reference import camera_measurements
from .gates import _amended_check

ROOT = spoon.ROOT
POLICY = ROOT / 'data/outputs/moving_policy'
WAIVED = ('mask_usable_fraction', 'mask_longest_unusable_interval_frames')


def summary(results):
    lines = ['# Spoon references after the mask-usability waiver', '',
             'Existing masks reused. Both measurements of texture usability (fraction and longest gap) are waived; '
             'all other final M1 limits remain unchanged. No labels or scans read. CPU only.', '',
             '| Episode | State | First failed gate | Value | Rule | Evidence |',
             '| --- | --- | --- | ---: | --- | --- |']
    by_episode = {row['episode']: row for row in results}
    for episode in (35, 36, 37):
        row = by_episode.get(episode)
        if row is None:
            lines.append(f'| {episode} | QUEUED | — | — | — | — |')
            continue
        failed = row.get('first_failed_gate')
        gate = row['gates'].get(failed, {})
        rule = f'{gate.get("comparison", "")} {gate.get("threshold", "")}'
        lines.append(f'| {episode} | {row["status"]} | {failed or "—"} | '
                     f'{gate.get("value", "—")} | {rule} | `{row["evidence"]}` |')
    for row in results:
        lines += ['', f'## Episode {row["episode"]}', '', '| Gate | Value | Rule | Verdict |',
                  '| --- | ---: | --- | --- |']
        for name, (comparison, limit) in FINAL_LIMITS.items():
            gate = row['gates'].get(name, {})
            verdict = 'WAIVED' if name in WAIVED else ('PASS' if gate.get('passed') else
                       'FAIL' if gate else 'NOT RUN')
            lines.append(f'| {name} | {gate.get("value", "—")} | {comparison} {limit} | {verdict} |')
    (POLICY / 'spoon-reference-summary.md').write_text('\n'.join(lines) + '\n')


def camera_stage(episode, directory):
    selected = spoon.inputs(episode)
    if json.loads((selected['masks'] / 'run.json').read_text())['status'] != 'completed':
        raise ValueError('Existing masks are incomplete')
    files = [*selected['paths'], selected['calibration'], selected['masks']/'request.json',
             selected['masks']/'summary.json', *selected['mask_files'],
             ROOT/'docs/chain/moving-spoon-protocol.md']
    selected['hashes'] = {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
    request = dict(episode=episode, inputs=selected['hashes'], mask_usability_waived=list(WAIVED),
                   sources={str(path.relative_to(ROOT)): sha256_file(path) for path in
                            [Path(__file__), Path(__file__).with_name('_spoon_reference.py'),
                             Path(camera_ba.__file__), Path(candidate_c.__file__)]},
                   labels_read=False, scan_reads=[], gpu=False)
    request_path = directory/'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Existing camera request identity changed')
    if not request_path.exists():
        write_json(request_path, request)
    if not (directory/'frame0.npz').exists():
        candidate_c.prepare_frame0(selected, directory)
    path = directory/'klt-observations.npz'
    if path.exists():
        with np.load(path) as arrays:
            data = {key: arrays[key] for key in arrays.files}
    else:
        data = camera_ba.observations(selected, directory)
    path = directory/'camera-ba.npz'
    if path.exists():
        with np.load(path) as arrays:
            poses = arrays['world_T_camera']
    else:
        poses, ids, landmarks, report = camera_ba.bundle_adjust(data)
        np.savez_compressed(path, world_T_camera=poses, landmark_ids=ids, landmarks=landmarks)
        write_json(directory/'bundle-adjustment.json', report)
    rows = camera_ba.heldout_rows(data, poses)
    write_json(directory/'camera-diagnostics.json', rows)
    seal = json.loads((ROOT/'data/outputs/moving_reference/ep39/sealed/sealed.json').read_text())
    if seal.get('reference_passed') is not True or seal.get('allowance_passed') is not True:
        raise ValueError('Perpendicular world instrument sealed condition failed')
    values, old_world = camera_measurements(data, poses, rows)
    measured = {name: _amended_check(value, FINAL_LIMITS[name][1], FINAL_LIMITS[name][0])
                for name, value in values.items()}
    measured['camera_heldout_world_residual_median_m'].update(
        old_3d_median_m=old_world, instrument='perpendicular under sealed-pass clause 2')
    mask = json.loads((directory.parent/'mask-gates.json').read_text())['gates']
    measured.update(mask)
    failed = next((name for name in FINAL_LIMITS if name in measured and name not in WAIVED
                   and not measured[name]['passed']), None)
    result = dict(episode=episode, status='STOP' if failed else 'CAMERA PASS; OBJECT PENDING',
                  first_failed_gate=failed, gates=measured, waived=list(WAIVED),
                  evidence=str((directory/'camera-final.json').relative_to(ROOT)),
                  completed_utc=now_utc(), labels_read=False, scan_reads=[], gpu=False)
    write_json(directory/'camera-final.json', result)
    return result


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(31)
    queue = json.loads(Path(__file__).with_name('spoon_queue.json').read_text())
    if queue['episodes'] != [35, 36, 37]:
        raise ValueError('Spoon order must stay 35, 36, 37')
    results = []
    for episode in queue['episodes']:
        directory = ROOT/f'data/outputs/moving_reference/ep{episode}/reference-mask-waiver'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'camera-final.json'
        if path.exists():
            result = json.loads(path.read_text())
        else:
            spoon.event(episode, 'mask-usability waiver: camera solve started on existing masks; CPU')
            start = time.perf_counter()
            with (directory/'run-camera.json').open('x') as stream:
                json.dump(dict(status='running', started_utc=now_utc()), stream)
            result = camera_stage(episode, directory)
            write_json(directory/'camera-completion.json', dict(status='completed',
                       wall_seconds=time.perf_counter()-start, result=result))
            with (POLICY/'spoon-reference-ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            failed = result['first_failed_gate']
            detail = result['gates'].get(failed, {})
            spoon.event(episode, f'{result["status"]}: {failed}; {detail}; {result["evidence"]}')
        results.append(result)
        summary(results)
        if not result['first_failed_gate']:
            return
    spoon.event('all', 'reference queue finished: all stopped before object tracking; no Stage 0 or training')


if __name__ == '__main__':
    main()
