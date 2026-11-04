"""FoundationPose environment worker; all GPU operations hold the project lease."""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.gpu import GpuLease
from src.provenance import sha256_file, write_json
from ._object import iou, render
from .candidate_b import ASSET, RECIPE, ROOT, stamp


def rotation_grid(seed):
    """Return resting and flipped orientations with a fixed local perturbation grid."""
    offsets = Rotation.from_euler('xyz', np.array(np.meshgrid(*[[-20, 0, 20]] * 3)).reshape(3, -1).T, degrees=True).as_matrix()
    grids = []
    for flip in (np.eye(3), Rotation.from_euler('x', 180, degrees=True).as_matrix(), Rotation.from_euler('z', 180, degrees=True).as_matrix()):
        group = np.repeat(np.eye(4)[None], len(offsets), axis=0)
        group[:, :3, :3] = seed[:3, :3] @ flip @ offsets
        grids.append(group)
    return np.concatenate(grids)


def select_hypothesis(hypotheses, seed, first):
    """Prefer the known table side and ring heading, then neural rank."""
    angles = np.rad2deg(Rotation.from_matrix(seed[:3, :3].T @ hypotheses[:, :3, :3]).magnitude())
    normal = hypotheses[:, :3, 2] @ seed[:3, 2]
    heading = hypotheses[:, :3, 0] @ seed[:3, 0]
    eligible = ((normal > np.cos(np.pi / 4)) & (heading > np.cos(np.pi / 4))) if first else angles <= 60
    candidates = np.flatnonzero(eligible)
    selected = int(candidates[0]) if len(candidates) else int(np.argmin(angles))
    return selected, bool(len(candidates)), angles


def hypothesis_report(hypotheses, scores, selected, seed):
    """Report true distant rivals separately from nearly duplicate hypotheses."""
    winner = hypotheses[selected]
    angles = np.rad2deg(Rotation.from_matrix(winner[:3, :3].T @ hypotheses[:, :3, :3]).magnitude())
    rows = [dict(rank=int(j), score=float(scores[j]), rotation_from_selected_deg=float(angles[j]),
                 seed_rotation_deg=float(np.rad2deg(Rotation.from_matrix(seed[:3, :3].T @ hypotheses[j, :3, :3]).magnitude())),
                 pose=hypotheses[j].tolist()) for j in range(min(12, len(scores)))]
    rivals = {}
    for name, condition in [('distant90', angles >= 90), ('rival180', angles >= 160)]:
        ids = np.flatnonzero(condition)
        j = int(ids[0]) if len(ids) else None
        rivals[name] = None if j is None else dict(rank=j, score=float(scores[j]), angle_deg=float(angles[j]),
                                                  gap_selected_minus_rival=float(scores[selected] - scores[j]),
                                                  pose=hypotheses[j].tolist())
    return dict(selected_rank=selected, selected_score=float(scores[selected]),
                raw_top_score=float(scores[0]), top_runner_gap=float(scores[0] - scores[1]),
                hypothesis_count=len(scores), top=rows, rivals=rivals)


def frame_data(directory, index):
    with np.load(directory / 'frames' / f'{index:06d}.npz') as arrays:
        return {key: arrays[key] for key in arrays.files}


def masked_depth(frame):
    """Clear hand depth without inventing depth on unsupported stereo pixels."""
    hands = cv2.dilate(frame['hands'].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    depth = frame['depth'].copy()
    depth[hands] = 0
    return depth, frame['mask'] & ~hands


def set_pose(estimator, pose):
    """Convert an original asset pose to FoundationPose's centered mesh convention."""
    import torch
    center = estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
    estimator.pose_last = torch.as_tensor(pose @ np.linalg.inv(center), dtype=torch.float32, device='cuda')


def register(estimator, frame, k, seed, first, directory, index, direction):
    """Run native registration, preserving every scored pose and rival margin."""
    import torch
    estimator.rot_grid = torch.as_tensor(rotation_grid(seed), dtype=torch.float32, device='cuda')
    depth, mask = masked_depth(frame)
    # register uses only object depth; track_one retains the surrounding scene.
    depth[~mask] = 0
    estimator.register(K=k, rgb=frame['rgb'], depth=depth, ob_mask=mask,
                       iteration=RECIPE['register_iterations'])
    if not hasattr(estimator, 'poses'):
        raise RuntimeError('Register returned its insufficient-depth fallback')
    hypotheses = (estimator.poses @ estimator.get_tf_to_centered_mesh()).detach().cpu().numpy()
    scores = estimator.scores.detach().cpu().numpy()
    selected, resolved, _ = select_hypothesis(hypotheses, seed, first)
    pose = hypotheses[selected]
    set_pose(estimator, pose)
    report = hypothesis_report(hypotheses, scores, selected, seed)
    report.update(frame_index=index, direction=direction, first=first, facing_resolved=resolved,
                  valid_masked_depth_pixels=int((depth > 0).sum()), time_utc=stamp())
    with (directory / 'registrations.jsonl').open('a') as log:
        log.write(json.dumps(report) + '\n')
    np.savez_compressed(directory / f'register-{direction}-{index:06d}.npz', hypotheses=hypotheses, scores=scores,
                        selected=selected, seed=seed, masked_depth=depth, mask=mask)
    return pose, resolved


def install_chunked_refiner(estimator):
    """Bound independent hypothesis refinement memory without altering its outputs."""
    import torch
    original = estimator.refiner.predict

    def predict(**kwargs):
        poses = kwargs['ob_in_cams']
        if len(poses) <= RECIPE['refinement_chunk']:
            return original(**kwargs)
        groups = []
        for start in range(0, len(poses), RECIPE['refinement_chunk']):
            part = dict(kwargs, ob_in_cams=poses[start:start + RECIPE['refinement_chunk']])
            refined, _ = original(**part)
            groups.append(refined)
        return torch.cat(groups), None

    estimator.refiner.predict = predict


def track_direction(estimator, directory, k, mesh, seed, direction, stop, lease):
    """Run one direction with fixed occlusion/re-registration and incremental output."""
    count = json.loads((directory / 'request.json').read_text())['frame_count']
    poses = np.repeat(seed[None], count, axis=0)
    scores = np.full(count, np.nan)
    valid = np.zeros(count, bool)
    observed = np.zeros(count, bool)
    heavy_flags = np.zeros(count, bool)
    rows, low_scores = [], []
    order = list(range(count)) if direction == 'forward' else list(range(count - 1, -1, -1))
    last = seed.copy()
    resolved = True
    set_pose(estimator, seed)
    for step, index in enumerate(order):
        if time.time() >= stop - RECIPE['operation_start_cutoff_seconds']:
            break
        frame = frame_data(directory, index)
        projection = render(mesh.vertices, mesh.faces, last, k, frame['mask'].shape)
        overlap = (projection & frame['hands']).sum() / max(1, projection.sum())
        depth, mask = masked_depth(frame)
        heavy = overlap > .5 or np.count_nonzero(mask) < 30
        event = 'freeze_heavy_occlusion' if heavy else 'track'
        tick = time.perf_counter()
        pose = last.copy()
        if direction == 'forward' and step == 0:
            pose, resolved = register(estimator, frame, k, seed, True, directory, index, direction)
            event = 'initial_register'
        elif not heavy:
            pose = estimator.track_one(rgb=frame['rgb'], depth=depth, K=k,
                                       iteration=RECIPE['track_iterations'])
        good = bool(np.isfinite(pose).all() and pose[2, 3] > 0)
        if not good:
            pose = last.copy()
            set_pose(estimator, pose)
            event = 'freeze_nonfinite'
        projection = render(mesh.vertices, mesh.faces, pose, k, frame['mask'].shape)
        score = iou(projection, frame['mask'], frame['hands'])
        if heavy:
            low_scores = []
        else:
            low_scores.append(score)
            low_scores = low_scores[-5:]
        if len(low_scores) == 5 and np.mean(low_scores) < .3:
            pose, _ = register(estimator, frame, k, last, False, directory, index, direction)
            projection = render(mesh.vertices, mesh.faces, pose, k, frame['mask'].shape)
            score = iou(projection, frame['mask'], frame['hands'])
            low_scores = []
            event = 'reregister_low_iou'
        poses[index], scores[index] = pose, score
        valid[index] = good and not heavy and resolved and score >= .3
        observed[index], heavy_flags[index] = True, heavy
        rows.append(dict(frame_index=index, event=event, iou=score, valid=bool(valid[index]),
                         hand_overlap_fraction=float(overlap), stereo_nonhand_pixels=int(np.count_nonzero((depth > 0) & mask)),
                         seconds=time.perf_counter() - tick))
        last = pose.copy()
        if step % 20 == 0:
            print(json.dumps(dict(direction=direction, frame=index, event=event, iou=score)), flush=True)
            lease.require_headroom()
        if step % 20 == 0 or step == count - 1:
            np.savez_compressed(directory / f'{direction}.npz', poses=poses, iou=scores,
                                valid=valid, observed=observed, heavy=heavy_flags)
            write_json(directory / f'{direction}-diagnostics.json', rows)
    # Unprocessed slots use the last camera pose and remain invalid/unobserved.
    poses[~observed] = last
    np.savez_compressed(directory / f'{direction}.npz', poses=poses, iou=scores,
                        valid=valid, observed=observed, heavy=heavy_flags)
    write_json(directory / f'{direction}-diagnostics.json', rows)
    return poses, bool(observed.all())


def run(directory):
    """Acquire one lease, initialize native networks and perform both real tracks."""
    cache_root = Path(os.environ['CANDIDATE_B_CACHE_ROOT'])
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'WARP_CACHE_PATH',
                'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'OPEN3D_DATA_ROOT', 'HF_HOME'):
        path = cache_root / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Explicit CUDA_VISIBLE_DEVICES=0 required')
    if not directory.resolve().is_relative_to(ROOT / 'data/outputs/moving_reference') or directory.name != 'candidate-b':
        raise PermissionError('Output must be an episode candidate-b directory')
    cv2.setNumThreads(2)
    logging.getLogger().setLevel(logging.ERROR)
    started = time.time()
    episode = json.loads((directory / 'request.json').read_text())['episode']
    stop = min(datetime.fromisoformat(RECIPE['stop_utc']).timestamp(), started + (1500 if episode == 39 else 1500))
    record = dict(status='running', episode=episode, started_utc=stamp(), licence=RECIPE['licence'], label_access=False,
                  scan_access=False, budget_end_utc=datetime.fromtimestamp(stop).isoformat(),
                  install_note='data/.runtime/foundationpose/INSTALL.md absent after layout move; docs/INSTALL.md read instead')
    lease = GpuLease(ROOT, 512, 7000, stage=f'moving_reference.candidate_b.ep{episode}')
    lease.interval_seconds = .5
    try:
        with lease:
            import torch
            import trimesh
            from estimater import FoundationPose
            torch.manual_seed(0)
            np.random.seed(0)
            mesh = trimesh.load(ASSET, force='mesh', process=False)
            with np.load(directory / 'geometry.npz') as arrays:
                k, seed, color = arrays['K'].astype(np.float64), arrays['seed'], arrays['color']
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.tile(np.r_[color, 255].astype(np.uint8), (len(mesh.vertices), 1)))
            estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                       mesh=mesh, debug=0, debug_dir=str(directory / 'debug'))
            install_chunked_refiner(estimator)
            torch.cuda.reset_peak_memory_stats()
            forward, complete_forward = track_direction(estimator, directory, k, mesh, seed, 'forward', stop, lease)
            backward, complete_backward = track_direction(estimator, directory, k, mesh, forward[-1], 'backward', stop, lease)
            torch.cuda.synchronize()
            record.update(status='completed' if complete_forward and complete_backward else 'budget_partial',
                          forward_completed=complete_forward, backward_completed=complete_backward,
                          peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                          peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2)
            record['runtime_hashes'] = {str(path.relative_to(ROOT)): sha256_file(path) for path in
                                        [ROOT / 'data/.runtime/foundationpose/upstream/estimater.py',
                                         *sorted((ROOT / 'data/.runtime/foundationpose/upstream/weights').glob('*/*'))] if path.is_file()}
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        print(record['error'], flush=True)
    finally:
        record.update(gpu=lease.report(), seconds=time.time() - started, released_utc=stamp())
        write_json(directory / 'worker-summary.json', record)
    return 0 if record['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().directory))
