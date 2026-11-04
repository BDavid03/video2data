"""Run and seal the fixed motion-gate and palm-attachment arms on public CPU caches."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from src.paths import project_path, output_path
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import position_track as previous, depth_track as h, depth_hybrid as hybrid
from . import full_c, camera, _depth_track as fit, kit_score
from . import _contact_probe as probe

ROOT = h.ROOT
OUTPUT = h.OUTPUT / 'contact-arms'


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def sources():
    paths = [Path(module.__file__) for module in (probe, previous, h, hybrid, full_c, camera, fit)]
    paths += [Path(__file__), Path(__file__).with_name('_object.py'), Path(__file__).with_name('gates.py'),
              ROOT / 'src/paths.py', ROOT / 'src/provenance.py', ROOT / 'src/runs.py']
    return {**kit_score.hashes(), **{relative(p): sha256_file(p) for p in paths}}


def prepare():
    """Freeze the same eighteen sealed revision-six inputs and both arm protocols."""
    h.verify_seal(previous.OUTPUT / 'outputs-seal.json')
    path = OUTPUT / 'queue.json'
    if path.exists():
        raise FileExistsError('Resume the frozen queue with run')
    clips = read(previous.OUTPUT / 'queue.json')['clips']
    write_json(path, dict(clips=clips, settings=probe.SETTINGS, sources=sources(), labels_read=False,
        protocols={arm: sha256_file(OUTPUT / f'protocol-{arm}.md') for arm in ('G', 'H')}, utc=now_utc()))
    write_json(OUTPUT / 'queue-seal.json', dict(utc=now_utc(), sha256=sha256_file(path), before_tracking=True))
    print('Frozen objects:', len(clips))


def identity(clip):
    item = clip['item']; cache = project_path(ROOT, item['cache']); masks = project_path(ROOT, item['masks'])
    old = previous.OUTPUT / f"ep{item['episode']}-{item['object']}"
    baseline = project_path(ROOT, clip['C'])
    paths = [cache / 'frames' / f'frame_{i:06d}.npz' for i in range(item['frames'])]
    paths += [masks / 'frames' / f'frame_{i:06d}.npz' for i in range(item['frames'])]
    paths += [old / n for n in ('reference.npz', 'object-diagnostics.json', 'seal.json')]
    paths += [baseline / 'reference.npz', baseline / 'tracking.npz', masks / 'request.json',
              cache / 'camera-diagnostics.json', project_path(ROOT, item['mesh']['path']),
              h.snapshot() / 'track_3/public/meta/camera_calibration.json',
              OUTPUT / 'protocol-G.md', OUTPUT / 'protocol-H.md']
    return dict(clip=clip, settings=probe.SETTINGS, sources=sources(),
        inputs={relative(p): sha256_file(p) for p in paths}, labels_read=False, gpu=False)


def extract(clip, directory, cameras):
    """Recover separate hand masks and their palm centres without recomputing stereo."""
    path = directory / 'evidence.npz'
    if path.exists():
        with np.load(path) as a:
            return a['palms'].copy(), a['point_counts'].copy(), a['contact'].copy()
    item = clip['item']; cache = project_path(ROOT, item['cache']); masks = project_path(ROOT, item['masks'])
    job = read(masks / 'request.json')
    maps, _, _, _ = camera.rectification(read(h.snapshot() / 'track_3/public/meta/camera_calibration.json'))
    palms = []; counts = []; contact = []; remap = None
    for index, world_camera in enumerate(cameras):
        frame = full_c.read_frame(cache, index)
        with np.load(masks / 'frames' / f'frame_{index:06d}.npz') as a:
            if int(a['frame_index']) != index or abs(float(a['timestamp_seconds']) - job['timestamps'][index]) > 1e-6:
                raise ValueError('Raw hand-mask frame/clock differs')
            if remap is None:
                remap = probe.resize_maps(maps[0], a['masks'].shape[1:], job['source_resolution_width_height'])
            slots = probe.rectify_slots(a['masks'], a['track_ids'], remap)
        if not np.array_equal(slots[0], frame['object_mask']) or not np.array_equal(slots[1:].any(0), frame['hands']):
            raise ValueError('Recovered slots do not reproduce cached masks')
        centre, size, overlap = probe.palm_observations(frame, slots[1:], world_camera)
        palms.append(centre); counts.append(size); contact.append(overlap)
        if index % 100 == 0:
            print(json.dumps(dict(stage='hands', episode=item['episode'], object=item['object'], frame=index)), flush=True)
    palms = probe.smooth_palms(np.asarray(palms)); counts = np.asarray(counts); contact = np.asarray(contact)
    np.savez_compressed(path, palms=palms, point_counts=counts, contact=contact, hand_slots=np.array([1, 2]),
                        recovered_mask_equivalence=np.asarray(True))
    return palms, counts, contact


def save_arm(directory, item, cameras, poses, valid, rows, times, k, model, cr, mapping):
    """Persist the existing reference schema while preserving the declared body frame."""
    directory.mkdir(exist_ok=True)
    h.save_track(directory, item, cameras, poses @ mapping, times, valid, rows, k, model, cr)
    quaternion = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    np.savez_compressed(directory / 'reference.npz', position=poses[:, :3, 3], quaternion=quaternion,
        quaternion_xyzw=quaternion, world_T_object=poses, frame_index=np.arange(len(times)),
        time=times, pose=np.c_[poses[:, :3, 3], quaternion], valid=valid, gates_passed=np.asarray(False))
    gates = read(directory / 'gates.json')
    gates.update(method=f'contact arm {directory.name}', campaign_admitted=False)
    gates['gates']['object_reference_valid_fraction']['note'] = 'Original fit residual/camera validity plus current motion-source availability'
    write_json(directory / 'gates.json', gates)


def track_clip(clip):
    """Run both frozen arms once, sharing evidence and exact duplicate residual measurements."""
    item = clip['item']; h.authorize(item['episode'])
    directory = OUTPUT / f"ep{item['episode']}-{item['object']}"
    directory.mkdir(exist_ok=True)
    if (directory / 'seal.json').exists():
        hybrid.check(directory)
        for arm in ('G', 'H'):
            hybrid.check(directory / arm)
        return directory
    request = identity(clip)
    if (directory / 'request.json').exists() and read(directory / 'request.json') != request:
        raise ValueError('Contact input/source identity changed')
    write_json(directory / 'request.json', request)
    started = time.perf_counter()
    old = previous.OUTPUT / directory.name; baseline = project_path(ROOT, clip['C'])
    hybrid.check(old)
    with np.load(old / 'reference.npz') as a:
        position = a['position'].copy(); old_poses = a['world_T_object'].copy(); old_valid = a['valid'].copy(); times = a['time'].copy()
    with np.load(baseline / 'reference.npz') as a:
        initial = a['world_T_object'][0].copy()
    with np.load(baseline / 'tracking.npz') as a:
        cameras = a['world_T_camera'].copy(); k = a['intrinsics'].copy()
    palms, counts, contact = extract(clip, directory, cameras)
    decision = probe.motion_gate(position, contact)
    hand_position, chosen, states, events = probe.hand_attachment(position, initial[:3, 3], palms, contact, decision['carried'])
    write_json(directory / 'decision.json', dict(**decision, events=events, labels_read=False,
        palm_supported_frames_per_slot=np.isfinite(palms).all(2).sum(0).tolist(),
        simultaneous_contact_frames=int(((contact >= .05).sum(1) == 2).sum()),
        attached_frames=int((chosen > 0).sum()), missing_selected_palm_frames=int(sum(
            chosen[i] > 0 and not np.isfinite(palms[i, chosen[i] - 1]).all() for i in range(len(chosen))))))
    poses = {arm: np.repeat(initial[None], len(position), axis=0) for arm in ('G', 'H')}
    if decision['carried']:
        poses['G'] = old_poses.copy()
    poses['H'][:, :3, 3] = hand_position
    np.savez_compressed(directory / 'attachment.npz', hand_slot=chosen, state=states, position=hand_position)
    mesh = trimesh.load(project_path(ROOT, item['mesh']['path']), force='mesh', process=False)
    model = fit.geometry(mesh.vertices, mesh.faces, item['object'])
    cache = project_path(ROOT, item['cache']); cr = read(cache / 'camera-diagnostics.json')
    mapping = np.asarray(clip['body_T_mesh']); old_rows = read(old / 'object-diagnostics.json')
    rows = {arm: [] for arm in poses}; validity = {arm: [] for arm in poses}
    for index, world_camera in enumerate(cameras):
        frame = full_c.read_frame(cache, index); observed = fit.evidence(frame, k)
        measured = {}
        for arm in ('G', 'H'):
            pose = poses[arm][index]; co = np.linalg.inv(world_camera) @ pose @ mapping
            if arm == 'G' and decision['carried']:
                detail = dict(old_rows[index]); good = bool(old_valid[index])
            else:
                if arm == 'H' and np.array_equal(pose, poses['G'][index]):
                    detail, good = measured['G']; detail = dict(detail)
                else:
                    detail = fit.fit_metrics(co, model, observed, k)
                    good = h.frame_valid(detail, cr[index]['observed'], co)
                available = old_rows[index]['centre_observed']
                if arm == 'H' and chosen[index]:
                    available = bool(np.isfinite(palms[index, chosen[index] - 1]).all())
                good = bool(good and available)
            measured[arm] = detail, good
            source = ('revision6' if decision['carried'] else 'static') if arm == 'G' else str(states[index])
            rows[arm].append(dict(detail, frame_index=index, source=source, observed=good))
            validity[arm].append(good)
    for arm in ('G', 'H'):
        target = directory / arm
        save_arm(target, item, cameras, poses[arm], np.asarray(validity[arm], bool), rows[arm], times, k, model, cr, mapping)
        if arm == 'G' and decision['carried']:
            shutil.copyfile(old / 'reference.npz', target / 'reference.npz')
        write_json(target / 'request.json', dict(parent_request_sha256=sha256_file(directory / 'request.json'),
            protocol_sha256=sha256_file(OUTPUT / f'protocol-{arm}.md'), decision_sha256=sha256_file(directory / 'decision.json')))
        hybrid.seal(target)
    write_json(directory / 'arm-seals.json', {relative(directory / arm / 'seal.json'): sha256_file(directory / arm / 'seal.json') for arm in ('G', 'H')})
    if identity(clip) != request:
        raise ValueError('Sources/inputs changed during contact run')
    hybrid.seal(directory)
    write_json(directory / 'run.json', dict(status='completed', wall_seconds=time.perf_counter() - started, labels_read=False, gpu=False))
    return directory


def run():
    """Resume completed objects and seal the entire arm batch before evaluation."""
    cv2.setNumThreads(2)
    queue = read(OUTPUT / 'queue.json')
    if queue['sources'] != sources() or sha256_file(OUTPUT / 'queue.json') != read(OUTPUT / 'queue-seal.json')['sha256']:
        raise ValueError('Frozen source/queue changed')
    completed = []
    for clip in queue['clips']:
        directory = track_clip(clip); completed.append(directory)
        ledger = OUTPUT / 'ledger.jsonl'
        saved = [json.loads(line)['path'] for line in ledger.read_text().splitlines()] if ledger.exists() else []
        if relative(directory) not in saved:
            with ledger.open('a') as stream:
                stream.write(json.dumps(dict(path=relative(directory), utc=now_utc(), status='both arms sealed')) + '\n')
    target = OUTPUT / 'outputs-seal.json'
    if not target.exists():
        write_json(target, dict(utc=now_utc(), labels_read=False, sources=sources(),
            files={relative(d / 'seal.json'): sha256_file(d / 'seal.json') for d in completed}))
    score = read(previous.OUTPUT / 'kit-request.json')
    score['variants'] = ['G', 'H']; score['output'] = relative(OUTPUT / 'kit')
    for case in score['cases']:
        for item in case['objects']:
            directory = OUTPUT / f"ep{case['episode']}-{item['name']}"
            for arm in ('G', 'H'):
                item[arm] = relative(directory / arm / 'reference.npz')
                item['world_T_shared_world'][arm] = item['world_T_shared_world']['C']
                item['world_frame_sources'].append(relative(directory / arm / 'tracking.npz'))
    if not (OUTPUT / 'kit-request.json').exists():
        write_json(OUTPUT / 'kit-request.json', score)
    print('All36 arm trajectories and18 carried decisions sealed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'run'])
    args = parser.parse_args(); h.authorize(); output_path(ROOT, OUTPUT)
    {'prepare': prepare, 'run': run}[args.operation]()


if __name__ == '__main__':
    main()
