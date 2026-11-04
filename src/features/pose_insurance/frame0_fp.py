"""Screen corrected frame-zero FoundationPose pairs with the fixed registration gates."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from src.gpu import GpuLease
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from src.features.moving_policy import two_object_screen as screen
from src.features.moving_reference import camera, track
from src.features.moving_reference import kit_score
from .combined_frames import ROOT
from . import _frame0_fp_frames as frames

BASE = ROOT / 'data/outputs/pose_insurance/frame0-fp'
OUTPUT = BASE / 'corrected'
PARENT = ROOT / 'submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json'


def event(episode, decision, **details):
    record = dict(utc=now_utc(), episode=episode, decision=decision, **details)
    with (BASE / 'status.md').open('a') as stream:
        stream.write(f'- {record["utc"]} ep{episode}: {decision}; {json.dumps(details)}\n')
    with (OUTPUT / 'ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def mask_pair(episode):
    summary = json.loads((ROOT / 'data/outputs/moving_policy/two-object/summary.json').read_text())
    result = next(r for r in summary['episodes'] if r['episode'] == episode)
    separated = result.get('instance_separation')
    if not separated:
        path = ROOT / f'data/outputs/moving_policy/two-object/instance-separated/ep{episode}/result.json'
        separated = json.loads(path.read_text())
    if not separated['passed']:
        raise ValueError('No saved separated object masks')
    return [ROOT / name for name in separated['directories']]


def prepare(episode, directory):
    directories = mask_pair(episode)
    selected = screen.selected_inputs(episode, directories[0])
    maps, k, q, _ = camera.rectification(json.loads(selected['calibration'].read_text()))
    streams = track.captures(selected['paths'])
    try:
        images, gray = track.read_pair(streams, maps)
    finally:
        for stream in streams:
            stream.release()
    cached = []
    for path in (ROOT / 'data/outputs/moving_policy/two-object').glob(f'campaign*/ep{episode}/candidate-c/frames/frame_000000.npz'):
        cached.append(path)
    if cached:
        with np.load(cached[-1]) as source:
            if not np.array_equal(gray[0], source['gray']):
                raise ValueError('Frame-zero cache and rectified RGB disagree')
            xyz = source['xyz'].copy()
    else:
        xyz, _ = camera.stereo(*gray, q)
    depth = np.where(np.isfinite(xyz[..., 2]), xyz[..., 2], 0).astype(np.float32)
    roster = json.loads(screen.ROSTER.read_text())['episodes'][str(episode)]
    snapshot = selected['calibration'].parents[3]
    requests = []
    for slot, (name, masks) in enumerate(zip(roster, directories)):
        inputs = screen.selected_inputs(episode, masks)
        mask = track.masks_at(inputs, 0, maps, gray[0].shape)[0]
        target = directory / f'{name}-frame0.npz'
        np.savez_compressed(target, rgb=images[0][:, :, ::-1].copy(), depth=depth, mask=mask, K=k)
        stem = 'mini_sweeper_visual' if name == 'mini_sweeper' else name
        glb = snapshot / 'track_3/public/mesh' / name / f'{stem}.glb'
        if not glb.exists():
            glb = snapshot / 'track_3/public/mesh' / name / f'{name}_visual.glb'
        request = dict(episode=episode, slot=slot, name=name, frame=str(target.relative_to(ROOT)),
                       frame_sha256=sha256_file(target), glb=str(glb.relative_to(ROOT)),
                       glb_sha256=sha256_file(glb), output=str((directory / name).relative_to(ROOT)))
        path = directory / f'{name}-request.json'
        write_json(path, request)
        requests.append(path)
    return requests


def conversion(episode, names, directory):
    """Use the exact diagnosis correction with each existing submitted mesh."""
    from . import pack_moving as moving
    from src.features.moving_policy.two_object_task import packed_starts
    trace = json.loads((ROOT / 'data/outputs/pose_insurance/combined-v5/body-frames.json').read_text())['rows']
    starts, binding = packed_starts(episode, names)
    starts = starts[:, [0, 1, 2, 4, 5, 6, 3]]
    anchors, transforms, details, geometries = [], [], [], []
    kit_score.OUTPUT = directory / 'frame-conversion'
    kit_score.OUTPUT.mkdir(exist_ok=True)
    kit = kit_score.scorer('AUC')
    for slot, name in enumerate(names):
        item = next(r for r in trace if r['episode'] == episode and r['slot'] == slot)
        mesh = moving.checked(item['combined_mesh'])
        candidate = kit_score.budget_candidate(mesh)
        geometry = kit_score.reference_geometry(name)
        anchor, rotation = kit._register_object_body(candidate, geometry['mesh'], geometry['anchor'],
                                                      1_000_003 * episode + 101 * slot)
        transform = frames.registered_transform(anchor, rotation)
        probe_input = kit._normalise_quaternions(starts[slot][None], 'candidate', ValueError)
        probe = frames.matrix(kit._convert_object_body(probe_input, anchor, rotation)[0])
        if not np.allclose(probe, frames.matrix(starts[slot]) @ transform, atol=1e-10):
            raise ValueError('Candidate conversion differs from the kit')
        anchors.append(geometry['anchor'])
        transforms.append(transform)
        geometries.append(geometry)
        details.append(dict(name=name, candidate_mesh=item['combined_mesh'],
                            candidate_T_registered=transform.tolist(), scan_anchor=geometry['anchor'].tolist(),
                            scan_geometry=geometry['input']))
    measurements = [json.loads((directory / name / 'result.json').read_text()) for name in names]
    from src.features.moving_policy._two_object_rotation import orientation_degrees
    from scipy.spatial.transform import Rotation
    with np.load(directory / names[1] / 'predictions.npz') as saved:
        hypotheses, scores = saved['hypotheses'].copy(), saved['scores'].copy()
    scan_T_registered = np.eye(4)
    scan_T_registered[:3, 3] = anchors[1]
    registered = hypotheses @ scan_T_registered
    translation = np.linalg.norm(registered[:, :3, 3] - registered[0, :3, 3], axis=1)
    axis = geometries[1]['axis'] if np.any(geometries[1]['axis']) else np.array([0., 0., 1.])
    rotation = orientation_degrees(Rotation.from_matrix(registered[:, :3, :3]).as_quat(),
                                  Rotation.from_matrix(registered[0, :3, :3]).as_quat(), names[1], axis)
    distinct = np.flatnonzero((translation > .01) | (rotation > 5.))
    margins = (scores[0] - scores) / max(abs(float(scores[0])), 1e-12)
    rival = int(distinct[np.argmin(margins[distinct])]) if len(distinct) else None
    measurements[1] = dict(measurements[1], raw_scan_rival_margin=measurements[1]['distinct_rival_relative_margin'],
        distinct_rival_index=rival, distinct_rival_relative_margin=float(margins[rival]) if rival is not None else None,
        ambiguity_passed=rival is None or float(margins[rival]) > .01,
        rival_instrument='Registered body anchors and exact public-kit symmetry axis; all hypotheses retained',
        registered_hypothesis_translation_m=translation.tolist(), registered_hypothesis_rotation_deg=rotation.tolist())
    corrected, relative = frames.correct_pair(starts, np.asarray([r['pose'] for r in measurements]), anchors, transforms)
    if not np.array_equal(starts[0], corrected[0]):
        raise ValueError('FP correction moved A')
    result = dict(episode=episode, names=names, start_xyzw=starts.tolist(), target_xyzw=corrected.tolist(),
                  relative=relative.tolist(), conversions=details, moving_v5_start_binding=binding,
                  helper_sha256=sha256_file(Path(frames.__file__)), labels_read=False,
                  registrations=measurements)
    write_json(directory / 'corrected-placement.json', result)
    return result


def screen_pair(episode):
    directory = OUTPUT / f'ep{episode}'
    directory.mkdir(exist_ok=True)
    final = directory / 'screen.json'
    if final.exists():
        return json.loads(final.read_text())
    event(episode, 'registration started')
    requests = prepare(episode, directory)
    rows = []
    for request_path in requests:
        request = json.loads(request_path.read_text())
        result = ROOT / request['output'] / 'result.json'
        if result.exists():
            rows.append(json.loads(result.read_text()))
            continue
        attempt = directory / f'{request["name"]}-attempt.json'
        with attempt.open('x') as stream:
            json.dump(dict(utc=now_utc(), request_sha256=sha256_file(request_path)), stream)
        with GpuLease(ROOT, 1024, 8000, stage='pose_insurance.frame0_fp') as lease:
            with (directory / f'{request["name"]}.log').open('x') as stream:
                completed = subprocess.run(['bash', str(ROOT / 'data/.runtime/tmp/frame0-fp/run_worker.sh'),
                    str(request_path)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            write_json(directory / f'{request["name"]}-gpu.json', lease.report())
        if completed.returncode:
            raise RuntimeError(f'FP {request["name"]} failed; no retry')
        rows.append(json.loads(result.read_text()))
        event(episode, 'object registration completed', slot=request['slot'], name=request['name'],
              iou=rows[-1]['silhouette_iou'], depth_m=rows[-1]['median_depth_residual_m'],
              rival_margin=rows[-1]['distinct_rival_relative_margin'])
    names = [r['name'] for r in rows]
    placement = conversion(episode, names, directory)
    rows = placement['registrations']
    passed = all(r['silhouette_passed'] and r['depth_passed'] for r in rows) and rows[1]['ambiguity_passed']
    value = dict(episode=episode, passed=passed, registrations=rows, placement=placement,
                 confidence=dict(rival_margin=rows[1]['distinct_rival_relative_margin'],
                                 iou=min(r['silhouette_iou'] for r in rows),
                                 depth=max(r['median_depth_residual_m'] or 1e9 for r in rows)))
    write_json(final, value)
    event(episode, 'registration PASS' if passed else 'KEEP v5: registration gate failed',
          gates=[dict(slot=r['slot'], iou=r['silhouette_iou'], depth_m=r['median_depth_residual_m'],
                      rival_margin=r['distinct_rival_relative_margin']) for r in rows])
    return value


def publish(rows):
    lines = ['# Corrected FoundationPose campaign', '',
             '| Episode | Registration | Object IoUs | Depth residuals m | B rival margin | Policy |',
             '| ---: | --- | --- | --- | --- | --- |']
    for row in rows:
        reg = row.get('registrations', [])
        lines.append(f'| {row["episode"]} | {"PASS" if row.get("passed") else "KEEP v5"} | '
                     f'{[r["silhouette_iou"] for r in reg]} | {[r["median_depth_residual_m"] for r in reg]} | '
                     f'{reg[1]["distinct_rival_relative_margin"] if len(reg)==2 else "not computed"} | '
                     f'{row.get("policy_status", "not run")} |')
    (OUTPUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    write_json(OUTPUT / 'summary.json', dict(utc=now_utc(), episodes=rows))


def main():
    if not json.loads((BASE / 'corrected-step-a.json').read_text())['passed']:
        raise PermissionError('Corrected pilot prerequisite failed')
    (OUTPUT / 'pid').write_text(str(os.getpid()) + '\n')
    queue = json.loads((OUTPUT / 'queue.json').read_text())
    rows = []
    for episode in queue['episodes']:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(queue['cutoff_utc'].replace('Z', '+00:00')):
            event(episode, 'KEEP v5: cutoff reached')
            break
        try:
            row = screen_pair(episode)
        except Exception as error:
            row = dict(episode=episode, passed=False, error=f'{type(error).__name__}: {error}')
            write_json(OUTPUT / f'ep{episode}/error.json', row)
            event(episode, 'KEEP v5: infrastructure/instrument failure', error=row['error'])
        rows.append(row)
        publish(rows)
    write_json(OUTPUT / 'screen-completed.json', dict(utc=now_utc(), passing=[r['episode'] for r in rows if r.get('passed')]))
    write_json(OUTPUT / 'completed.json', dict(utc=now_utc(), adopted=[]))


if __name__ == '__main__':
    main()
