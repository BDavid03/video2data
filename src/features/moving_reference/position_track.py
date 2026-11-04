"""Run, seal and score the fixed position-only revision on prepared public clips."""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from src.paths import project_path, output_path
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import depth_track as h, depth_hybrid as hybrid, full_c, kit_score
from . import _depth_track as fit
from ._position_track import centre_observation, surface_offset, position_step

ROOT = h.ROOT
OUTPUT = h.OUTPUT / 'position-v6'
SETTINGS = dict(revision=6, minimum_points=30, depth_sigma_multiplier=2,
                depth_sigma_floor_m=.001, median_window=5, surface_samples=40000, seed=6)


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def sources():
    paths = [Path(__file__), Path(__file__).with_name('_position_track.py')]
    return {**hybrid.sources(), **kit_score.hashes(), **{relative(p): sha256_file(p) for p in paths}}


def prepare():
    """Freeze the prepared population and source identity without reading labels."""
    h.authorize()
    output_path(ROOT, OUTPUT)
    if (OUTPUT / 'queue.json').exists():
        raise FileExistsError('Keep the frozen queue; resume with run')
    clips = []
    for item in hybrid.ready_items():
        if item['cache'] is None:
            cache = hybrid.OUTPUT / 'recovered-inputs' / f"ep{item['episode']}-{item['object']}"
            if not (cache / 'completed.json').exists():
                raise FileNotFoundError(f'CPU cache has not been recovered: {cache}')
            item['cache'] = relative(cache)
        saved = hybrid.OUTPUT / f"ep{item['episode']}-{item['object']}/C-alone"
        if (saved / 'seal.json').exists():
            hybrid.check(saved)
            item = read(saved / 'request.json')['item']
            baseline = saved
        else:
            baseline = OUTPUT / 'baselines' / f"ep{item['episode']}-{item['object']}/C-alone"
        clips.append(dict(item=item, C=relative(baseline), body_T_mesh=np.eye(4).tolist()))
    saved = ROOT / 'data/outputs/moving_reference/ep39/candidate-c'
    item = next(i for i in read(h.OUTPUT / 'queue.json')['clips'] if i['episode'] == 39)
    with np.load(saved / 'reference.npz') as a, np.load(saved / 'tracking.npz') as b:
        mapping = np.linalg.inv(a['world_T_object'][0]) @ b['world_T_asset'][0]
        if not np.allclose(a['world_T_object'] @ mapping, b['world_T_asset'], atol=1e-10):
            raise ValueError('C39 body-to-mesh convention changed')
    item = dict(item, mesh=dict(item['mesh'], path=relative(saved / 'asset/mini_dust_pan.obj')))
    item['mesh']['sha256'] = sha256_file(project_path(ROOT, item['mesh']['path']))
    clips.append(dict(item=item, C=relative(saved), body_T_mesh=mapping.tolist()))
    write_json(OUTPUT / 'queue.json', dict(clips=clips, settings=SETTINGS, sources=sources(),
        protocol_sha256=sha256_file(OUTPUT / 'protocol.md'), utc=now_utc(), labels_read=False))
    print(json.dumps(dict(prepared_objects=len(clips), episodes=sorted({c['item']['episode'] for c in clips}))))


def input_identity(clip):
    item = clip['item']; cache = project_path(ROOT, item['cache'])
    baseline = project_path(ROOT, clip['C'])
    paths = [cache / 'frames' / f'frame_{i:06d}.npz' for i in range(item['frames'])]
    paths += [baseline / 'reference.npz', baseline / 'tracking.npz',
              cache / 'camera-diagnostics.json', project_path(ROOT, item['mesh']['path']),
              project_path(ROOT, item['masks']) / 'request.json', OUTPUT / 'protocol.md']
    return dict(clip=clip, settings=SETTINGS, sources=sources(),
                inputs={relative(p): sha256_file(p) for p in paths}, labels_read=False, gpu=False)


def track_clip(clip):
    """Estimate centres in the saved C camera world and retain frozen residual validity."""
    item = clip['item']; h.authorize(item['episode'])
    directory = OUTPUT / f"ep{item['episode']}-{item['object']}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'seal.json').exists():
        hybrid.check(directory)
        return directory
    request = input_identity(clip)
    write_json(directory / 'request.json', request)
    started = time.perf_counter()
    cache = project_path(ROOT, item['cache']); baseline = project_path(ROOT, clip['C'])
    with np.load(baseline / 'reference.npz') as saved:
        initial_body = saved['world_T_object'][0].copy()
        times = saved['time'].copy()
    with np.load(baseline / 'tracking.npz') as saved:
        cameras = saved['world_T_camera'].copy()
        k = saved['intrinsics'].copy()
    mapping = np.asarray(clip['body_T_mesh']); initial = initial_body @ mapping
    mesh = trimesh.load(project_path(ROOT, item['mesh']['path']), force='mesh', process=False)
    model = fit.geometry(mesh.vertices, mesh.faces, item['object'])
    samples, _ = trimesh.sample.sample_surface(mesh, SETTINGS['surface_samples'], seed=SETTINGS['seed'])
    frame = full_c.read_frame(cache, 0)
    offset, centroid, pixels = surface_offset(samples, np.linalg.inv(cameras[0]) @ initial, k, frame['object_mask'].shape)
    write_json(directory / 'initialization.json', dict(initial_world_T_body=initial_body.tolist(),
        initial_world_T_mesh=initial.tolist(), body_T_mesh=mapping.tolist(),
        visible_surface_camera_centroid_m=centroid.tolist(), signed_view_ray_offset_m=offset,
        visible_mesh_pixels=pixels, own_mesh_only=True))
    camera_rows = read(cache / 'camera-diagnostics.json')
    history = deque(maxlen=SETTINGS['median_window'] - 1)
    previous = None; poses = []; validity = []; rows = []
    for index, world_camera in enumerate(cameras):
        frame = full_c.read_frame(cache, index)
        centre, detail = centre_observation(frame, SETTINGS['minimum_points'])
        raw, position = position_step(centre, world_camera, offset, history, previous, initial[:3, 3])
        history.append(raw); previous = position.copy()
        pose = initial.copy(); pose[:3, 3] = position
        co = np.linalg.inv(world_camera) @ pose
        residual = fit.fit_metrics(co, model, fit.evidence(frame, k), k)
        good = detail['centre_observed'] and h.frame_valid(residual, camera_rows[index]['observed'], co)
        poses.append(pose); validity.append(good)
        rows.append(dict(residual, **detail, frame_index=index, raw_world_position_m=raw.tolist(),
                         observed=bool(good), fitted=False, source='position-only'))
        if index % 80 == 0:
            print(json.dumps(dict(stage='position', episode=item['episode'], object=item['object'], frame=index)), flush=True)
    poses = np.asarray(poses); validity = np.asarray(validity, bool)
    h.save_track(directory, item, cameras, poses, times, validity, rows, k, model, camera_rows)
    # Saved C39 references use a body frame distinct from its actual candidate mesh.
    body = poses @ np.linalg.inv(mapping)
    quaternion = Rotation.from_matrix(body[:, :3, :3]).as_quat()
    np.savez_compressed(directory / 'reference.npz', position=body[:, :3, 3], quaternion=quaternion,
        quaternion_xyzw=quaternion, frame_index=np.arange(len(times)), time=times, valid=validity,
        world_T_object=body, pose=np.c_[body[:, :3, 3], quaternion], gates_passed=np.asarray(False))
    if not np.allclose(body[:, :3, :3], initial_body[:3, :3], atol=1e-12):
        raise ValueError('Position-only orientation changed')
    gates = read(directory / 'gates.json')
    gates.update(method='revision 6 position-only', campaign_admitted=False)
    gates['gates']['object_reference_valid_fraction']['note'] = 'Frozen fit residual validity AND >=30 post-cut stereo points'
    write_json(directory / 'gates.json', gates)
    if input_identity(clip) != request:
        raise ValueError('Position tracker sources/inputs changed during run')
    hybrid.seal(directory)
    write_json(directory / 'run.json', dict(status='completed', wall_seconds=time.perf_counter() - started,
        gpu=False, labels_read=False, reference_sha256=sha256_file(directory / 'reference.npz')))
    return directory


def score_request(queue):
    """Assemble complete static/C/position episodes in a common camera-derived world."""
    groups = {}
    for clip in queue['clips']:
        item = clip['item']; directory = OUTPUT / f"ep{item['episode']}-{item['object']}"
        baseline = project_path(ROOT, clip['C'])
        groups.setdefault(item['episode'], []).append(dict(name=item['object'], mesh=item['mesh']['path'],
            C=relative(baseline / 'reference.npz'), position=relative(directory / 'reference.npz'),
            body_T_mesh=clip['body_T_mesh']))
    for items in groups.values():
        common = project_path(ROOT, items[0]['C']).with_name('tracking.npz')
        with np.load(common) as saved:
            shared = saved['world_T_camera'][0]
        for item in items:
            item['world_T_shared_world'] = {}; item['world_frame_sources'] = [relative(common)]
            for variant in ('C', 'position'):
                source = project_path(ROOT, item[variant]).with_name('tracking.npz')
                with np.load(source) as saved:
                    transform = shared @ np.linalg.inv(saved['world_T_camera'][0])
                item['world_T_shared_world'][variant] = transform.tolist()
                item['world_frame_sources'].append(relative(source))
    path = OUTPUT / 'kit-request.json'
    if not path.exists():
        write_json(path, dict(cases=[dict(episode=ep, objects=items, allow_partial=False) for ep, items in sorted(groups.items())],
            variants=['static', 'C', 'position'], output=relative(OUTPUT / 'kit'), maximum_seconds=3600,
            reference_geometry='public_proxy_evaluation_only'))
    return path


def run():
    """Run the resumable queue, sealing every object before any new labelled evaluation."""
    cv2.setNumThreads(2)
    queue = read(OUTPUT / 'queue.json')
    if queue['sources'] != sources() or queue['settings'] != SETTINGS:
        raise ValueError('Frozen revision source/settings changed')
    ledger = OUTPUT / 'ledger.jsonl'
    directories = []
    for clip in queue['clips']:
        item = clip['item']; baseline = project_path(ROOT, clip['C'])
        if not (baseline / 'reference.npz').exists():
            hybrid.c_track(item, baseline)
        directory = track_clip(clip); directories.append(directory)
        records = ledger.read_text().splitlines() if ledger.exists() else []
        if not any(json.loads(line)['path'] == relative(directory) for line in records):
            with ledger.open('a') as stream:
                stream.write(json.dumps(dict(path=relative(directory), episode=item['episode'], object=item['object'],
                    utc=now_utc(), status='sealed', seal_sha256=sha256_file(directory / 'seal.json'))) + '\n')
    seal = OUTPUT / 'outputs-seal.json'
    if not seal.exists():
        write_json(seal, dict(utc=now_utc(), labels_read=False, sources=sources(),
            files={relative(d / 'seal.json'): sha256_file(d / 'seal.json') for d in directories},
            queue_sha256=sha256_file(OUTPUT / 'queue.json')))
    print('All position outputs sealed:', len(directories), flush=True)
    print('Score request:', relative(score_request(queue)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'run'])
    args = parser.parse_args()
    h.authorize(); output_path(ROOT, OUTPUT)
    {'prepare': prepare, 'run': run}[args.operation]()


if __name__ == '__main__':
    main()
