"""Run the sealed, scan-free FoundationPose queue (39 then 38).

Run with the project Python: ``-m src.features.moving_reference.candidate_b run``.
The editable queue, frozen recipe, inputs, per-frame depth, ledger, overlays and
results are all saved under each episode's candidate-b directory. No world-frame
trajectory is produced. FoundationPose has a noncommercial licence.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.paths import project_path
from src.provenance import sha256_file, write_json
from . import camera, track
from ._object import iou, render

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'data/outputs/moving_reference'
ASSET = OUTPUT / 'ep39/candidate-c/asset/mini_dust_pan.obj'
DEADLINE = '2026-11-04T09:25:00+00:00'
RECIPE = dict(width=640, seed=0, stereo='OpenCV SGBM via camera.stereo: 128 disparities, block5, uniqueness8, speckle80, LR<=1px, 0.1<z<5m',
              register_iterations=5, track_iterations=2, refinement_chunk=24,
              initialization='Frame0 table normal plus mask ring heading; ep39 uses C frame0 and asset placement; ep38 computes the identical resting construction from its own stereo and ring',
              registration_grid='81: resting, long-axis180, normal-axis180; each with local xyz Euler offsets {-20,0,20}deg. Translation guessed by upstream register from masked stereo',
              selection='Frame0 hypotheses with upward-normal and heading within45deg of resting seed; highest neural score, otherwise closest rotation and mark unresolved',
              mesh_color='Uniform median RGB of frame0 eroded non-hand object mask; geometry unchanged',
              hand_rule='Zero depth on 7x7 dilated hands. Heavy if hand overlap with projected mesh >50% OR visible object<30 pixels: freeze previous camera pose and mark invalid. Same both directions and episodes',
              reregister='Five consecutive non-heavy frames with trailing mean IoU<0.3. Fresh same81 grid about current pose; select highest neural score within60deg of preceding pose; no re-registration on heavy frames',
              backward='Initialize at final forward pose, track every frame in reverse; conditional cycle consistency, not an independent final registration',
              score='Non-hand projected silhouette IoU, not calibrated neural confidence',
              smoothing='None; gate definitions from gates.py applied to all written adjacent frames, including invalid frozen poses',
              valid='Successful non-heavy tracking or registration, finite positive-depth pose, IoU>=0.3, initial facing resolved. FB validity reported separately',
              stop_utc=DEADLINE, operation_start_cutoff_seconds=100, ep39_gpu_budget_seconds=1500,
              memory_limit_mib=16384, licence='NVIDIA FoundationPose noncommercial source code licence',
              prohibited_inputs_read=False, output_camera='rectified left ego_cam_c; R1 and raw-C converted poses included',
              queue=[39, 38])


def stamp():
    return datetime.now(timezone.utc).isoformat()


def select_inputs(episode):
    """Select only the permitted video streams, masks and calibration."""
    if episode not in (39, 38):
        raise PermissionError('Only episodes39/38 are authorized')
    masks = OUTPUT / f'ep{episode}' / ('masks' if episode == 39 else 'masks-smallest')
    job = json.loads((masks / 'request.json').read_text())
    if json.loads((masks / 'summary.json').read_text())['status'] != 'completed':
        raise ValueError('Masks incomplete')
    source = json.loads((ROOT / 'data/configs/sources.lock.json').read_text())['dataset']
    snapshot = project_path(ROOT, f"{source['cache_path']}/snapshots/{source['revision']}")
    split = 'public' if episode == 39 else 'evaluation'
    paths = [snapshot / f'track_3/{split}/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4' for key in ('c', 'b')]
    calibration = snapshot / 'track_3/public/meta/camera_calibration.json'
    audit = ROOT / 'data/configs/calibration/ego.json'
    if sha256_file(calibration) != json.loads(audit.read_text())['source_sha256']:
        raise ValueError('Calibration differs from audit')
    mask_files = [masks / 'frames' / f'frame_{i:06d}.npz' for i in range(job['frame_count'])]
    inputs = [*paths, calibration, audit, masks / 'request.json', masks / 'summary.json', *mask_files, ASSET]
    if episode == 39:
        inputs += [OUTPUT / 'ep39/candidate-c/frame0.npz', OUTPUT / 'ep39/candidate-c/asset/placement.npz']
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}
    return dict(paths=paths, masks=masks, mask_files=mask_files, job=job, calibration=calibration, hashes=hashes)


def resting_seed(mask, xyz, visible, k, mesh):
    """Infer camera-from-object from stereo table and the ring's mask hole."""
    from .model_track import footprint
    hands = np.zeros_like(mask)
    static = ~cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    table, _ = camera.define_world(xyz, static & np.isfinite(xyz).all(2), mask)
    placement, _ = footprint(mask, xyz, visible, k, table, known_thickness=float(np.ptp(mesh.vertices[:, 2])))
    return np.linalg.inv(table) @ placement


def prepare(episode, directory, selected):
    """Decode calibrated stereo and cache metric depth with exact frame clocks."""
    import trimesh
    started = time.perf_counter()
    calibration = json.loads(selected['calibration'].read_text())
    maps, k, q, r1 = camera.rectification(calibration, RECIPE['width'])
    mesh = trimesh.load(ASSET, force='mesh', process=False)
    frames = directory / 'frames'
    frames.mkdir(exist_ok=True)
    streams = track.captures(selected['paths'])
    try:
        for index in range(selected['job']['frame_count']):
            images, gray = track.read_pair(streams, maps)
            destination = frames / f'{index:06d}.npz'
            if destination.exists():
                continue
            xyz, supported = camera.stereo(*gray, q)
            obj, hands, visible, _ = track.masks_at(selected, index, maps, gray[0].shape)
            depth = np.nan_to_num(xyz[..., 2], nan=0., posinf=0., neginf=0.).astype(np.float32)
            np.savez_compressed(destination, rgb=cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB),
                                depth=depth, mask=obj, hands=hands, visible=visible,
                                stereo_valid=supported, frame_index=index,
                                time=selected['job']['timestamps'][index])
            if index == 0:
                if episode == 39:
                    with np.load(OUTPUT / 'ep39/candidate-c/frame0.npz') as data:
                        world_T_camera = data['world_T_camera']
                    with np.load(OUTPUT / 'ep39/candidate-c/asset/placement.npz') as data:
                        placement = data['world_T_asset']
                    seed = np.linalg.inv(world_T_camera) @ placement
                else:
                    seed = resting_seed(obj, xyz, visible, k, mesh)
                np.savez_compressed(directory / 'geometry.npz', K=k, R1=r1, seed=seed,
                                    color=np.median(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)[visible], axis=0))
            if index % 50 == 0:
                print(json.dumps(dict(stage='depth', episode=episode, frame=index)), flush=True)
    finally:
        for stream in streams:
            stream.release()
    return time.perf_counter() - started


def worker_environment(directory):
    """Override install defaults so every generated cache remains in our output."""
    environment = os.environ.copy()
    cache = directory / 'cache'
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'WARP_CACHE_PATH',
                'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'OPEN3D_DATA_ROOT', 'HF_HOME'):
        path = cache / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return environment


def metrics(poses, backward, times, scores, valid, backward_valid):
    """Use gates.py's speed/angular definitions and absolute FB pose distances."""
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / np.diff(times)
    angular = np.rad2deg(Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude()) / np.diff(times)
    fb_m = np.linalg.norm(poses[:, :3, 3] - backward[:, :3, 3], axis=1)
    fb_deg = np.rad2deg(Rotation.from_matrix(poses[:, :3, :3].transpose(0, 2, 1) @ backward[:, :3, :3]).magnitude())
    measured = scores[np.isfinite(scores)]
    both = valid & backward_valid
    both[0] = False  # gates.py excludes frame 0 from forward-backward medians.
    return dict(iou_median=float(np.median(measured)), iou_p10=float(np.percentile(measured, 10)),
                reprojection_coverage_fraction=float(np.mean(np.isfinite(scores))),
                fb_translation_median_cm=float(np.nanmedian(fb_m[1:]) * 100), fb_rotation_median_deg=float(np.nanmedian(fb_deg[1:])),
                fb_both_valid_fraction=float(np.mean(both)),
                fb_both_valid_translation_median_cm=float(np.median(fb_m[both]) * 100) if both.any() else None,
                fb_both_valid_rotation_median_deg=float(np.median(fb_deg[both])) if both.any() else None,
                speed_max_m_s=float(speed.max()), angular_rate_max_deg_s=float(angular.max()),
                valid_fraction=float(valid.mean()),
                bad_iou_frames=np.flatnonzero(scores < .3).tolist()), fb_m, fb_deg


def finish(episode, directory):
    """Write all camera poses, gate statistics, a12-frame strip and hashes."""
    import trimesh
    request = json.loads((directory / 'request.json').read_text())
    worker = json.loads((directory / 'worker-summary.json').read_text())
    with np.load(directory / 'geometry.npz') as arrays:
        k, r1 = arrays['K'], arrays['R1']
    with np.load(directory / 'forward.npz') as arrays:
        poses, scores, valid = (arrays[key] for key in ('poses', 'iou', 'valid'))
        observed = arrays['observed']
    with np.load(directory / 'backward.npz') as arrays:
        backward, backward_valid = arrays['poses'], arrays['valid']
    times = np.asarray(request['timestamps'], float)
    result, fb_m, fb_deg = metrics(poses, backward, times, scores, valid, backward_valid)
    raw = poses.copy()
    raw[:, :3] = np.einsum('ij,tjk->tik', r1.T, poses[:, :3])
    np.savez_compressed(directory / 'T_CO.npz', T_CO=poses, camera_T_asset=poses, T_CrawO=raw,
                        T_CO_backward=backward, frame_index=np.arange(len(times)), time=times,
                        valid=valid, observed=observed, backward_valid=backward_valid,
                        tracking_score=scores, silhouette_iou=scores, fb_translation_m=fb_m,
                        fb_rotation_deg=fb_deg, K=k, R1=r1)
    mesh = trimesh.load(ASSET, force='mesh', process=False)
    tiles = []
    for index in np.linspace(0, len(times) - 1, 12).round().astype(int):
        with np.load(directory / 'frames' / f'{index:06d}.npz') as frame:
            image, mask, hands = frame['rgb'].copy(), frame['mask'], frame['hands']
        projection = render(mesh.vertices, mesh.faces, poses[index], k, mask.shape)
        overlay = image.copy()
        overlay[projection & ~hands] = (0, 255, 255)
        overlay[mask & ~hands] = (255, 200, 0)
        image = cv2.addWeighted(image, .65, overlay, .35, 0)
        for target, color in ((projection, (0, 255, 255)), (mask, (255, 200, 0))):
            contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 1)
        cv2.putText(image, f'{index:03d} t={times[index]:.2f}s IoU={scores[index]:.2f} valid={int(valid[index])}',
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        tiles.append(image)
    strip = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, 12, 3)])
    cv2.imwrite(str(directory / 'overlay-strip.png'), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    summary = dict(episode=episode, status=worker['status'], label_free=True, scan_free_geometry=True,
                   camera_frame_only=True, licence=RECIPE['licence'], recipe=RECIPE,
                   metrics=result, gpu=worker, seconds=dict(depth=request['depth_seconds'], gpu=worker['seconds']),
                   input_hashes=request['input_hashes'], source_hashes=request['source_hashes'],
                   failure_interpretation='See bad_iou_frames, validity and overlay. Unobserved frozen frames never count valid; FB all-frame medians include fallbacks to match gates.py fail-closed semantics.',
                   completed_utc=stamp())
    summary['artifact_hashes'] = {p.name: sha256_file(p) for p in
                                  [directory / name for name in ('T_CO.npz', 'forward.npz', 'backward.npz', 'geometry.npz', 'overlay-strip.png', 'worker-summary.json', 'registrations.jsonl')] if p.exists()}
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'run.json', summary)
    return summary


def run_queue():
    """Execute the ordered fixed queue, skipping completed identities on resume."""
    cv2.setNumThreads(2)
    queue_path = OUTPUT / 'ep39/candidate-b/queue.json'
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    if not queue_path.exists():
        write_json(queue_path, dict(episodes=[39, 38], recipe=RECIPE))
    queue = json.loads(queue_path.read_text())
    if queue['episodes'] != [39, 38] or queue['recipe'] != RECIPE:
        raise ValueError('This sealed job requires the frozen39/38 queue and recipe')
    sources = [Path(__file__), Path(__file__).with_name('candidate_b_worker.py'),
               Path(__file__).with_name('camera.py'), Path(__file__).with_name('_object.py'),
               Path(__file__).with_name('gates.py'), ROOT / 'src/gpu.py']
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
    for episode in queue['episodes']:
        directory = OUTPUT / f'ep{episode}/candidate-b'
        directory.mkdir(exist_ok=True)
        if (directory / 'summary.json').exists():
            previous = json.loads((directory / 'summary.json').read_text())
            if previous['source_hashes'] != source_hashes:
                raise ValueError('Completed sources changed; preserve existing run')
            print(json.dumps(dict(episode=episode, skipped=True)), flush=True)
            continue
        if time.time() > datetime.fromisoformat(DEADLINE).timestamp() - 180:
            print('Deadline: no next episode admitted', flush=True)
            break
        selected = select_inputs(episode)
        request = dict(episode=episode, recipe=RECIPE, timestamps=selected['job']['timestamps'],
                       frame_count=selected['job']['frame_count'], input_hashes=selected['hashes'],
                       source_hashes=source_hashes, started_utc=stamp())
        if (directory / 'request.json').exists():
            old = json.loads((directory / 'request.json').read_text())
            if old['input_hashes'] != request['input_hashes']:
                raise ValueError('Inputs changed on resume')
        write_json(directory / 'request.json', request)
        request['depth_seconds'] = prepare(episode, directory, selected)
        write_json(directory / 'request.json', request)
        if (ROOT / 'data/.runtime/gpu-stage.lock').exists():
            raise RuntimeError('GPU lease already owned; no lock touched')
        # Source the installed environment first; then override its output locations.
        command = ['bash', '-c', 'source data/.runtime/foundationpose/use-env.sh\nexec python -B -m src.features.moving_reference.candidate_b_worker --directory "$1"', 'candidate-b', str(directory)]
        environment = worker_environment(directory)
        # use-env changes caches; worker immediately applies the recorded environment again.
        environment['CANDIDATE_B_CACHE_ROOT'] = str(directory / 'cache')
        with (directory / 'worker.log').open('a') as log:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if not (directory / 'worker-summary.json').exists():
            raise RuntimeError(f'Worker exited{process.returncode}; see{directory}/worker.log')
        summary = finish(episode, directory)
        with (queue_path.parent / 'ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(episode=episode, status=summary['status'], metrics=summary['metrics'],
                                        summary_sha256=sha256_file(directory / 'summary.json'))) + '\n')
        print(json.dumps(dict(episode=episode, status=summary['status'], metrics=summary['metrics'])), flush=True)
        if process.returncode:
            print('Worker incomplete; preserving evidence before next episode', flush=True)
    summaries = []
    for episode in (39, 38):
        path = OUTPUT / f'ep{episode}/candidate-b/summary.json'
        if path.exists():
            item = json.loads(path.read_text())
            m = item['metrics']
            summaries.append(f"|{episode}|{item['status']}|{m['iou_median']:.3f}|{m['iou_p10']:.3f}|{m['fb_translation_median_cm']:.2f}|{m['fb_rotation_median_deg']:.2f}|{m['valid_fraction']:.3f}|")
    (queue_path.parent / 'RESULT.md').write_text('# Candidate B: camera-frame FoundationPose\n\n'
        '|Episode|Status|IoU median|IoU p10|FB cm|FB deg|Valid fraction|\n|---|---|---|---|---|---|---|\n' + '\n'.join(summaries) +
        '\n\nCyan: projected mesh; amber: SAM2 object. Noncommercial FoundationPose. '
        'Backward track starts from the forward endpoint; its agreement is conditional. '
        'No world camera compensation, support admission or mocap scoring in this job.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['run', 'finish'])
    parser.add_argument('--episode', type=int, choices=[39, 38], default=39)
    args = parser.parse_args()
    if args.operation == 'run':
        run_queue()
    else:
        finish(args.episode, OUTPUT / f'ep{args.episode}/candidate-b')
