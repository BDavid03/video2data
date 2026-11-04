"""Register public scans to frame zero of train/tuning pairs and validate the B-in-A pose against labels.

Run: PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_refresh.train_frame0 <refresh dir> [episodes...]
Labels are read only on the frozen train/tuning episodes, only to validate each registration.
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from src.gpu import GpuLease
from src.paths import project_path
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from src.features.moving_reference import camera, track
from src.features.moving_reference._masks import box_iou
from src.features.moving_policy._two_object_rotation import orientation_degrees
from .frame0_fp import ROOT, WORKER

TRAIN = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)
WORKER_SCRIPT = Path(__file__).with_name('_frame0_masks_worker.py')


def snapshot():
    source = json.loads((ROOT / 'data/configs/sources.lock.json').read_text())['dataset']
    return project_path(ROOT, Path(source['cache_path']) / 'snapshots' / source['revision'])


def labels(episode):
    if episode not in TRAIN:
        raise PermissionError('Labels are read only on train/tuning episodes')
    table = pq.read_table(snapshot() / f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet', columns=['observation.objects'])
    row = table.slice(0, 1).to_pylist()[0]['observation.objects']
    names = [o['name'] for o in row]
    poses = {o['name']: np.asarray(o['pose'], float) for o in row}  # x y z qw qx qy qz
    return names, poses


def matrix_wxyz(pose):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    value[:3, 3] = pose[:3]
    return value


def joint_boxes(attempts, threshold=.4):
    candidates = [[dict(box=b, score=s) for b, s in zip(a['boxes'], a['scores']) if s > threshold] for a in attempts]
    best = None
    for combo in itertools.product(*candidates):
        boxes = [np.asarray(c['box'], float) for c in combo]
        overlap = max((box_iou(a, b) for a, b in itertools.combinations(boxes, 2)), default=0.)
        if overlap >= .5:
            continue
        area = sum(float(np.maximum(b[2:] - b[:2], 0).prod()) for b in boxes)
        key = (-sum(c['score'] for c in combo), area)
        if best is None or key < best[0]:
            best = (key, combo, overlap)
    if best is None:
        return None
    return dict(boxes=[c['box'] for c in best[1]], scores=[c['score'] for c in best[1]], max_box_iou=best[2])


def run_worker(python, operation, job, directory, env):
    path = directory / f'{operation}-job.json'
    write_json(path, job)
    with (directory / f'{operation}.log').open('w') as log:
        subprocess.run([str(python), '-B', str(WORKER_SCRIPT), operation, str(path)], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def environment(directory):
    import os
    cache = directory / '.cache'
    cache.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES='0', TMPDIR=str(cache), XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache),
               TORCH_EXTENSIONS_DIR=str(cache / 'ext'), HF_HOME=str(cache / 'hf'), MPLCONFIGDIR=str(cache / 'mpl'))
    return env


def prepare(episode, names, directory):
    """Rectified stereo frame zero, SGBM depth, joint DINO boxes, SAM2 masks remapped to the rectified grid."""
    snap = snapshot()
    paths = [snap / f'track_3/public/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4' for key in ('c', 'b')]
    calibration = json.loads((snap / 'track_3/public/meta/camera_calibration.json').read_text())
    maps, k, q, _ = camera.rectification(calibration)
    streams = track.captures(paths)
    try:
        raw = streams[0].read()[1]
        streams[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
        images, gray = track.read_pair(streams, maps)
    finally:
        for stream in streams:
            stream.release()
    xyz, _ = camera.stereo(*gray, q)
    depth = np.where(np.isfinite(xyz[..., 2]), xyz[..., 2], 0).astype(np.float32)
    height, width = raw.shape[:2]
    scale = min(1., 1024 / max(width, height))
    small = cv2.resize(raw, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else raw
    image = directory / 'frame0-raw.jpg'
    cv2.imwrite(str(image), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    env = environment(directory)
    prompts = [n.replace('_', ' ') for n in names]
    with GpuLease(ROOT, 1024, 10000, stage='pose_refresh.frame0_masks'):
        run_worker(ROOT / 'data/.runtime/grounding-dino/venv/bin/python', 'detect',
                   dict(image=str(image), names=names, prompts=prompts, box_threshold=.4, text_threshold=.3, output=str(directory)), directory, env)
        attempts = json.loads((directory / 'detections.json').read_text())['attempts']
        chosen = joint_boxes(attempts)
        if chosen is None:
            raise ValueError('No joint box assignment with pairwise IoU < 0.5')
        write_json(directory / 'joint-boxes.json', dict(chosen, rule='max total score among pairwise IoU<0.5, ties smaller area'))
        run_worker(ROOT / 'data/.runtime/sam2/venv/bin/python', 'segment',
                   dict(image=str(image), names=names, boxes=chosen['boxes'], output=str(directory)), directory, env)
    with np.load(directory / 'masks.npz') as saved:
        raw_masks = saved['masks'].astype(np.uint8)
    mx = (maps[0][0] + .5) * raw_masks.shape[2] / width - .5
    my = (maps[0][1] + .5) * raw_masks.shape[1] / height - .5
    masks = [cv2.remap(m, mx, my, cv2.INTER_NEAREST).astype(bool) for m in raw_masks]
    overlap = float((masks[0] & masks[1]).sum() / max((masks[0] | masks[1]).sum(), 1))
    requests = []
    for slot, (name, mask) in enumerate(zip(names, masks)):
        target = directory / f'{name}-frame0.npz'
        np.savez_compressed(target, rgb=images[0][:, :, ::-1].copy(), depth=depth, mask=mask, K=k)
        stem = 'mini_sweeper_visual' if name == 'mini_sweeper' else name
        glb = snap / 'track_3/public/mesh' / name / f'{stem}.glb'
        if not glb.exists():
            glb = snap / 'track_3/public/mesh' / name / f'{name}_visual.glb'
        request = dict(episode=episode, slot=slot, name=name, frame=str(target.relative_to(ROOT)), frame_sha256=sha256_file(target),
                       glb=str(glb.relative_to(ROOT)), glb_sha256=sha256_file(glb), output=str((directory / name).relative_to(ROOT)))
        path = directory / f'{name}-request.json'
        write_json(path, request)
        requests.append(path)
    return requests, overlap


def register(requests, directory):
    rows = []
    for request_path in requests:
        request = json.loads(request_path.read_text())
        result = ROOT / request['output'] / 'result.json'
        if not result.exists():
            with GpuLease(ROOT, 1024, 8000, stage='pose_refresh.frame0_fp'):
                with (directory / f'{request["name"]}.log').open('w') as stream:
                    subprocess.run(['bash', str(WORKER), str(request_path)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        rows.append(json.loads(result.read_text()))
    return rows


def validate(episode, names, rows):
    """Compare the FP B-in-A relative pose (scan frames) with the label relative pose at frame zero."""
    _, truth = labels(episode)
    fp = [np.asarray(r['pose']) for r in rows]
    predicted = np.linalg.inv(fp[0]) @ fp[1]
    reference = np.linalg.inv(matrix_wxyz(truth[names[0]])) @ matrix_wxyz(truth[names[1]])
    offset_cm = float(100 * np.linalg.norm(predicted[:3, 3] - reference[:3, 3]))
    axis = np.asarray(rows[1]['symmetry_axis_scan'])
    rotation_deg = float(orientation_degrees(Rotation.from_matrix(predicted[:3, :3]).as_quat()[None],
                                             Rotation.from_matrix(reference[:3, :3]).as_quat()[None], names[1], axis)[0])
    return dict(offset_cm=offset_cm, rotation_deg=rotation_deg, valid=offset_cm <= 2. and rotation_deg <= 10.)


def main():
    base = Path(sys.argv[1]).resolve()
    episodes = [int(e) for e in sys.argv[2:]] or [0, 11, 12, 18, 21, 23, 26, 31]
    results = []
    for episode in episodes:
        directory = base / 'train' / f'ep{episode}'
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / 'result.json'
        if final.exists():
            results.append(json.loads(final.read_text()))
            continue
        names, _ = labels(episode)
        try:
            requests, overlap = prepare(episode, names, directory)
            rows = register(requests, directory)
            check = validate(episode, names, rows)
            value = dict(episode=episode, names=names, frame0_mask_iou=overlap, validation=check, utc=now_utc(),
                         registrations=[dict(name=r['name'], iou=r['silhouette_iou'], depth_m=r['median_depth_residual_m'],
                                             margin=r['distinct_rival_relative_margin']) for r in rows])
        except Exception as error:
            value = dict(episode=episode, names=names, error=f'{type(error).__name__}: {error}', utc=now_utc())
        write_json(final, value)
        print(json.dumps(value), flush=True)
        results.append(value)
    write_json(base / 'train' / 'results.json', dict(utc=now_utc(), episodes=results))


if __name__ == '__main__':
    main()
