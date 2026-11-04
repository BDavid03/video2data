"""Run M1 model tracking on the existing own spoon asset, without a scan."""

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import camera, camera_ba, full_c, model_track, spoon, spoon_reference, spoon_reference_table
from ._object import initial_pose, iou, pair, render, smooth
from ._final_gates import FINAL_LIMITS
from .gates import evaluate, _amended_check


def object_pass(directory, cameras, k, model, initial, direction, forward=None):
    """Use candidate C's forward/backward fitting with a spoon frame-zero pose."""
    count = len(cameras)
    indices = range(count) if direction == 'forward' else range(count-1, -1, -1)
    poses = np.tile(np.eye(4), (count, 1, 1))
    details = [None]*count
    previous = previous_co = previous_world = None
    for position, index in enumerate(indices):
        frame = full_c.read_frame(directory, index)
        evidence = model_track.image_evidence(frame['object_mask'], frame['hands'], frame['xyz'], frame['visible'])
        if position == 0:
            world = initial.copy() if direction == 'forward' else forward[index].copy()
            co = np.linalg.inv(cameras[index])@world
            detail = dict(iou=iou(model_track.fast_render(model, co, k, frame['gray'].shape),
                                 frame['object_mask'], frame['hands']), cost=0., starts=1,
                          weak_fit=False, opposite_resolved=True, opposite_iou_margin=None)
            corners = camera.corners(frame['gray'], frame['visible'])
            _, supported = camera.lift(corners, frame['xyz'])
            count_pairs = int(supported.sum()) if direction == 'forward' else 0
        else:
            motion, count_pairs = pair(previous['gray'], frame['gray'], previous['xyz'], frame['xyz'],
                                       previous['visible'], frame['visible'])
            starts = [np.linalg.inv(cameras[index])@previous_world]
            if motion is not None:
                starts.append(motion@previous_co)
            co, detail = model_track.select_pose(starts, evidence, model, k)
            world = cameras[index]@co
        poses[index] = world
        detail.update(frame_index=index, correspondences=int(count_pairs),
                      stereo_supported_pixels=evidence['stereo_pixels'],
                      observed=bool(len(evidence['cloud']) >= 6 and detail['opposite_resolved']))
        details[index] = detail
        previous, previous_co, previous_world = frame, co, world
        if position % 20 == 0 or position == count-1:
            print(json.dumps(dict(stage=direction, frame=index, iou=detail['iou'],
                                  klt=count_pairs, stereo=evidence['stereo_pixels'])), flush=True)
    np.savez_compressed(directory/f'object-{direction}.npz', world_T_asset=poses)
    write_json(directory/f'object-{direction}.json', details)
    return poses, details


def object_stage(episode, directory, result):
    """Measure object gates before spending time on the remaining table audits."""
    selected = spoon.inputs(episode)
    asset = directory.parent/'asset/wooden_spoon.ply'
    source = json.loads((directory.parent/'asset/source.json').read_text())
    if source['scan_reads'] or not source['same_bytes']:
        raise ValueError('Own shared spoon asset required')
    expected, = {item['sha256'] for item in source['inputs']}
    if sha256_file(asset) != expected:
        raise ValueError('Spoon asset changed')
    mesh = trimesh.load(asset, force='mesh', process=False)
    with np.load(directory/'camera-table.npz') as arrays:
        cameras = arrays['world_T_camera'].copy()
    with np.load(directory/'klt-observations.npz') as arrays:
        data = {key: arrays[key] for key in arrays.files}
    k = data['k']
    frame = full_c.read_frame(directory, 0)
    # C's hole-based pan heading does not exist on a spoon; use M1's existing
    # generic stereo/PCA four-heading silhouette initializer, with C's fixed base.
    co, initialization = initial_pose(frame['xyz'], frame['visible'], frame['object_mask'], frame['hands'],
                                      cameras[0], mesh.vertices, mesh.faces, k)
    initial = cameras[0]@co
    initial[2, 3] = -mesh.vertices[:, 2].min()
    initialization.update(base_fixed_to_table=True, scan_reads=[], asset_sha256=expected)
    write_json(directory/'initialization.json', initialization)
    prediction = render(mesh.vertices, mesh.faces, np.linalg.inv(cameras[0])@initial, k, frame['gray'].shape)
    model = model_track.tracking_geometry(prediction, k, cameras[0], initial, np.ptp(mesh.vertices[:, 2]),
                                          mesh.vertices, mesh.faces)
    forward, details = object_pass(directory, cameras, k, model, initial, 'forward')
    backward, _ = object_pass(directory, cameras, k, model, initial, 'backward', forward)
    rows = camera_ba.heldout_rows(data, cameras)
    valid = np.asarray([a['observed'] and b['observed'] and initialization['opposite_resolved']
                        for a, b in zip(rows, details)], bool)
    filtered = smooth(forward, valid)
    scores = []
    for index in range(len(cameras)):
        frame = full_c.read_frame(directory, index)
        prediction = render(mesh.vertices, mesh.faces, np.linalg.inv(cameras[index])@filtered[index], k, frame['gray'].shape)
        scores.append(iou(prediction, frame['object_mask'], frame['hands']) if valid[index] else np.nan)
        details[index]['fb_translation_m'] = float(np.linalg.norm(forward[index, :3, 3]-backward[index, :3, 3]))
        details[index]['fb_rotation_deg'] = float(np.rad2deg(Rotation.from_matrix(
            forward[index, :3, :3].T@backward[index, :3, :3]).magnitude()))
    mask = json.loads((directory.parent/'mask-gates.json').read_text())
    times = np.asarray(selected['job']['timestamps'])
    table = json.loads((directory/'table.json').read_text())
    base = float(camera.apply(filtered[0], mesh.vertices)[:, 2].min())
    raw = evaluate(table, rows, details, times, filtered, valid, scores,
                   np.asarray(mask['texture_counts']) >= 30, base,
                   dict(status='not run', maximum_translation_m=None, maximum_rotation_deg=None))
    measured = dict(result['gates'])
    for name, (comparison, limit) in FINAL_LIMITS.items():
        if name in raw['gates'] and name not in measured:
            measured[name] = _amended_check(raw['gates'][name]['value'], limit, comparison)
    measured['object_30_stereo_correspondence_fraction'] = _amended_check(
        float(np.mean([r['stereo_supported_pixels'] >= 30 for r in details])), .9, 'ge')
    failed = next((name for name in FINAL_LIMITS if name in measured and name not in spoon_reference.WAIVED
                   and not measured[name]['passed']), None)
    np.savez_compressed(directory/'tracking.npz', world_T_camera=cameras, world_T_asset=filtered,
                        world_T_asset_forward=forward, world_T_asset_backward=backward, valid=valid, intrinsics=k)
    np.savez_compressed(directory/'reprojection.npz', iou=np.asarray(scores), valid=valid)
    write_json(directory/'object-diagnostics.json', details)
    result = dict(result, status='STOP' if failed else 'OBJECT PASS; TABLE DRIFT PENDING',
                  first_failed_gate=failed, gates=measured, completed_utc=now_utc(),
                  evidence=str((directory/'object-final.json').relative_to(spoon.ROOT)),
                  initialization=initialization, stage0_run=False, training_run=False)
    write_json(directory/'object-final.json', result)
    return result


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(31)
    queue = json.loads(Path(__file__).with_name('spoon_queue.json').read_text())
    if queue['episodes'] != [35, 36, 37]:
        raise ValueError('Spoon order must stay 35, 36, 37')
    results = []
    for episode in queue['episodes']:
        directory = spoon.ROOT/f'data/outputs/moving_reference/ep{episode}/reference-mask-waiver'
        directory.mkdir(parents=True, exist_ok=True)
        result = None
        for phase, function in [('camera', spoon_reference.camera_stage),
                                ('table', spoon_reference_table.table_stage), ('object', object_stage)]:
            path = directory/f'{phase}-final.json'
            if path.exists():
                result = json.loads(path.read_text())
            else:
                spoon.event(episode, f'{phase} started; CPU; existing masks; mask-usability waived only')
                write_json(directory/f'{phase}-request.json', dict(started_utc=now_utc(), episode=episode,
                           source_sha256=sha256_file(Path(__file__)), scan_reads=[], labels_read=False, gpu=False))
                result = function(episode, directory) if phase == 'camera' else function(episode, directory, result)
                spoon.event(episode, f'{result["status"]}: {result["first_failed_gate"]}; '
                            f'{result["gates"].get(result["first_failed_gate"], {})}; {result["evidence"]}')
            spoon_reference.summary([*results, result])
            if result['first_failed_gate']:
                break
        results.append(result)
        with (spoon_reference.POLICY/'spoon-reference-object-ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        if not result['first_failed_gate']:
            return
    spoon.event('all', 'reference queue finished: all stopped; no Stage 0 or training')


if __name__ == '__main__':
    main()
