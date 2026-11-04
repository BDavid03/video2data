"""Run, seal and evaluate a CPU feature-free object-tracking batch."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from src.paths import project_path, output_path
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import camera, full_c
from . import _depth_track as fit
from .gates import evaluate as campaign_instruments, _amended_check

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT/'data/outputs/moving_reference/depth-track'
PROTOCOL = ROOT/'docs/chain/moving-depth-tracker-protocol.md'
ALLOWED = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)
OBJECTS = {0: ['white_pot', 'white_pot_lid'], 11: ['beige_cup', 'blue_cup'],
           12: ['blue_cup', 'plastic_dish_rack'], 18: ['mini_sweeper', 'mini_dust_pan'],
           21: ['water_pitcher', 'blue_cup'], 23: ['wooden_piece_1', 'wooden_piece_2'],
           26: ['white_pot', 'plastic_dish_rack'], 31: ['mini_sweeper', 'plastic_dish_rack'],
           39: ['mini_dust_pan'], 42: ['white_pot']}
QUALITY = ('object_reference_valid_fraction', 'reprojection_iou_median', 'reprojection_iou_p10',
           'reprojection_coverage_fraction', 'object_speed_max_m_s', 'object_angular_rate_max_deg_s')


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def authorize(episode=None):
    """Require the frozen protocol and this public-only task's episode allowlist."""
    output_path(ROOT, OUTPUT)
    if episode is not None and episode not in ALLOWED:
        raise PermissionError('Only the frozen public train/tuning episodes are authorized')
    seal = read(OUTPUT/'protocol-seal.json')
    if seal['sha256'] != sha256_file(PROTOCOL):
        original = OUTPUT/'frozen-v1/docs/chain/moving-depth-tracker-protocol.md'
        if (not original.exists() or sha256_file(original) != seal['sha256'] or
                not PROTOCOL.read_bytes().startswith(original.read_bytes())):
            raise ValueError('Frozen protocol changed instead of receiving dated appendices')
    return seal


def sources():
    names = ['depth_track.py', '_depth_track.py', '_object.py', 'camera.py', 'gates.py', 'full_c.py', 'model_track.py']
    paths = [Path(__file__).with_name(n) for n in names]
    paths += [ROOT/'src/paths.py', ROOT/'src/provenance.py', ROOT/'src/runs.py']
    return {relative(p): sha256_file(p) for p in paths}


def snapshot():
    source = read(ROOT/'data/configs/sources.lock.json')['dataset']
    return project_path(ROOT, Path(source['cache_path'])/'snapshots'/source['revision'])


def mesh_binding(name):
    """Select the first campaign trace for the class and verify its own-mesh bytes."""
    trace = ROOT/'data/outputs/pose_insurance/combined-v5/body-frames.json'
    rows = read(trace)['rows']
    row = next(r for r in rows if r['object'] == name)
    if row['decision'] not in ('identity_frame', 'rotated', 'v4_procedural'):
        raise ValueError('Untraced body convention')
    entry = row['combined_mesh']; path = project_path(ROOT, entry['path'])
    if sha256_file(path) != entry['sha256']:
        raise ValueError('Campaign mesh differs from trace')
    return dict(path=relative(path), sha256=entry['sha256'], trace=relative(trace),
                trace_sha256=sha256_file(trace), episode=row['episode'], decision=row['decision'])


def inventory():
    """Record available caches without opening a pose label or object trajectory."""
    authorize()
    clocks = read(ROOT/'data/reports/data-audit/videos.json')['videos']
    fp = read(ROOT/'data/outputs/foundationpose/inputs/manifest.json')['episodes']
    rows = []; queue = []
    for episode in ALLOWED:
        clock, = [c for c in clocks if c['path'] == f'track_3/public/videos/chunk-000/observation.images.ego_cam_c/episode_{episode:06d}.mp4']
        cached_depth = set()
        for path in (ROOT/f'data/outputs/stereo_depth/public/episode_{episode:06d}').glob('*/run.json'):
            record = read(path)
            if record.get('kind') == 'stereo_depth.generate' and record['status'] == 'completed':
                cached_depth.update(r['frame_index'] for r in record['frames'] if r.get('status') == 'completed')
        for name in OBJECTS[episode]:
            base = OUTPUT/f'inputs/ep{episode}-{name}'
            masks = base/'masks'
            cache = base/'cache'
            if episode == 39:
                masks = ROOT/'data/outputs/moving_reference/ep39/masks'
                cache = ROOT/'data/outputs/moving_reference/ep39/candidate-c'
            count = clock['decoded_frame_count']
            mask_count = len(list((masks/'frames').glob('frame_*.npz')))
            frame_count = len(list((cache/'frames').glob('frame_*.npz')))
            ready = (mask_count == count and frame_count == count and
                     (cache/'camera-ba.npz').exists() and (cache/'camera-diagnostics.json').exists())
            row = dict(episode=episode, object=name, frames=count, sam_frames=mask_count,
                       cached_dense_frames=frame_count, sparse_depth_frames=len(cached_depth),
                       foundationpose_frame0=any(o['name'] == name for o in fp.get(str(episode), {}).get('objects', [])),
                       ready=ready, masks=relative(masks), cache=relative(cache), mesh=mesh_binding(name))
            rows.append(row)
            if ready:
                queue.append(row)
    write_json(OUTPUT/'inventory.json', dict(labels_read=False, episodes=rows,
               missing_class='spoon: no allowed train/tuning clip', created_utc=now_utc()))
    path = OUTPUT/'queue.json'
    if not path.exists():
        write_json(path, dict(revision=1, settings=fit.SETTINGS, clips=queue))
    return rows


def input_identity(item, configuration=None, protocol_path=PROTOCOL):
    """Hash every consumed mask, stereo frame and camera/mesh input before fitting."""
    cache = project_path(ROOT, item['cache']); masks = project_path(ROOT, item['masks'])
    paths = [cache/'camera-ba.npz', cache/'camera-diagnostics.json', masks/'request.json',
             masks/'summary.json', project_path(ROOT, item['mesh']['path']),
             project_path(ROOT, item['mesh']['trace']), protocol_path,
             ROOT/'data/configs/sources.lock.json', snapshot()/'track_3/public/meta/camera_calibration.json']
    paths += [cache/'frames'/f'frame_{i:06d}.npz' for i in range(item['frames'])]
    paths += [masks/'frames'/f'frame_{i:06d}.npz' for i in range(item['frames'])]
    if configuration:
        paths.append(cache/'table.json')
    authorize(item['episode'])
    return dict(item=item, settings=fit.SETTINGS, sources=sources(), configuration=configuration,
                protocol_path=relative(protocol_path), protocol_sha256=sha256_file(protocol_path),
                inputs={relative(p): sha256_file(p) for p in paths})


def frame_valid(detail, camera_observed, pose):
    """Apply the protocol's fit-only object validity, plus camera observation."""
    return bool(camera_observed and detail['visible_pixels'] >= 50 and
                detail['stereo_supported_pixels'] >= 6 and detail['iou'] >= .30 and
                detail['depth_median_m'] is not None and detail['depth_median_m'] <= .015 and
                detail['depth_p90_m'] <= .030 and np.isfinite(pose).all() and pose[2, 3] > .05)


def track_frame(frame, k, model, world_camera, previous, previous_detail=None):
    """Fit a single frame without correspondence or label initialization."""
    observed = fit.evidence(frame, k)
    observed['original_k'] = k
    if observed['stereo_pixels'] < 6 or observed['visible_pixels'] < 50 or not len(observed['pixels']):
        pose = np.linalg.inv(world_camera)@previous if previous is not None else np.eye(4)
        detail = fit.fit_metrics(pose, model, observed, k)
        return pose, dict(detail, fitted=False, evaluations=0)
    configuration = model.get('configuration', {})
    if previous is None:
        starts = (fit.canonical_starts(observed, world_camera, model, model['canonical_rotation'])
                  if configuration else fit.initial_starts(observed, world_camera, model))
    else:
        starts = [np.linalg.inv(world_camera)@previous]
    candidates = []
    for start in starts:
        pose, evaluations = fit.fit_pose(start, model, observed)
        detail = fit.fit_metrics(pose, model, observed, k)
        score = 1-detail['iou'] + detail['depth_median_m']/.01
        candidates.append((score, pose, dict(detail, fitted=True, evaluations=evaluations)))
    if previous is not None and configuration.get('rotation_grid'):
        best = min(candidates, key=lambda row: row[0])
        old_p90 = previous_detail.get('depth_p90_m') if previous_detail else None
        jump = best[2]['depth_p90_m'] > max(.03, 1.5*(old_p90 or .03))
        if jump or best[2]['iou'] < .45:
            for axis in np.eye(3):
                for sign in (-1, 1):
                    start = best[1].copy()
                    start[:3, :3] = start[:3, :3]@Rotation.from_rotvec(sign*np.radians(15)*axis).as_matrix()
                    pose, evaluations = fit.fit_pose(start, model, observed)
                    detail = fit.fit_metrics(pose, model, observed, k)
                    score = 1-detail['iou'] + detail['depth_median_m']/.01
                    candidates.append((score, pose, dict(detail, fitted=True, evaluations=evaluations)))
    _, pose, detail = min(candidates, key=lambda row: row[0])
    return pose, dict(detail, starts=len(candidates))


def instruments(poses, times, valid, rows, camera_rows, model):
    """Compute the campaign instruments without fabricating feature/FB/support evidence."""
    ious = [r['iou'] if valid[i] else np.nan for i, r in enumerate(rows)]
    objects = [dict(r, correspondences=0, fb_translation_m=None, fb_rotation_deg=None) for r in rows]
    support = dict(status='unmeasured for this mesh', maximum_translation_m=None, maximum_rotation_deg=None)
    result = campaign_instruments(dict(residual_rms_m=None, patch_normal_max_deg=None), camera_rows,
        objects, times, poses, valid, ious, [r['stereo_supported_pixels'] >= 30 for r in rows], 0., support)
    # Same symmetry instrument/threshold as two_object_rotation.amend; old scalar retained.
    if model['symmetry']:
        relative_rotation = poses[:-1, :3, :3].transpose(0, 2, 1)@poses[1:, :3, :3]
        rate = fit.symmetry_degrees(relative_rotation, model['axis'], model['symmetry'])/np.diff(times)
        old = result['gates']['object_angular_rate_max_deg_s']
        result['raw_object_angular_rate_max_deg_s'] = old['value']
        result['gates']['object_angular_rate_max_deg_s'] = _amended_check(float(rate.max()), 540)
    # Base/support admission is not measured by this CPU tracker.
    for name in ('frame0_base_gap_m', 'frame0_penetration_m'):
        result['gates'][name] = _amended_check(None, .005)
    result['gates']['object_reference_valid_fraction']['note'] = 'Depth-tracker revision 1 fit validity; see frozen protocol'
    result.update(quality_passed=all(result['gates'][n]['passed'] for n in QUALITY),
                  campaign_admitted=False, label_free=True, object_features_used=False)
    return result


def save_track(directory, item, cameras, poses, times, valid, rows, k, model, camera_rows):
    """Persist the campaign-compatible trajectory and unchanged diagnostic instruments."""
    quaternion = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    for i in range(1, len(quaternion)):
        if quaternion[i]@quaternion[i-1] < 0:
            quaternion[i] *= -1
    np.savez_compressed(directory/'reference.npz', position=poses[:, :3, 3], quaternion=quaternion,
        quaternion_xyzw=quaternion, frame_index=np.arange(len(times)), time=times, valid=valid,
        world_T_object=poses, pose=np.c_[poses[:, :3, 3], quaternion], gates_passed=np.asarray(False))
    np.savez_compressed(directory/'tracking.npz', world_T_camera=cameras, world_T_asset=poses,
        camera_T_asset=np.linalg.inv(cameras)@poses, valid=valid, intrinsics=k)
    np.savez_compressed(directory/'reprojection.npz', iou=[r['iou'] if valid[i] else np.nan for i, r in enumerate(rows)],
                        raw_iou=[r['iou'] for r in rows], valid=valid)
    write_json(directory/'object-diagnostics.json', rows)
    write_json(directory/'gates.json', instruments(poses, times, valid, rows, camera_rows, model))
    write_json(directory/'mesh-identity.json', dict(item['mesh'], symmetry=model['symmetry'], axis_model=model['axis'].tolist()))


def track_clip(item, directory=None, configuration=None, protocol_path=PROTOCOL):
    """Resume an exact-identity CPU track and seal its artifacts before labels."""
    authorize(item['episode'])
    directory = directory or OUTPUT/f"ep{item['episode']}-{item['object']}"
    directory.mkdir(exist_ok=True)
    identity = input_identity(item, configuration, protocol_path)
    request = directory/'request.json'
    if request.exists() and read(request) != identity:
        raise ValueError('Track identity changed; retain original outputs and freeze a new revision')
    write_json(request, identity)
    if (directory/'seal.json').exists():
        verify_seal(directory/'seal.json'); return directory
    write_json(directory/'run.json', dict(status='running', started_utc=now_utc(), labels_read=False, gpu=False))
    started = time.perf_counter()
    cache = project_path(ROOT, item['cache'])
    job = read(project_path(ROOT, item['masks'])/'request.json')
    if job['episode'] != dict(split='public', episode=item['episode']) or job['frame_count'] != item['frames']:
        raise ValueError('Mask producer episode or frame count mismatch')
    times = np.asarray(job['timestamps'], float)
    if len(times) != item['frames'] or np.any(np.diff(times) <= 0):
        raise ValueError('Mask clock incomplete')
    with np.load(cache/'camera-ba.npz', allow_pickle=False) as saved:
        cameras = saved['world_T_camera'].copy()
    if configuration:
        gauge = np.asarray(read(cache/'table.json')['world_new_T_world_initial'])
        cameras = gauge[None]@cameras
    camera_rows = read(cache/'camera-diagnostics.json')
    if len(cameras) != len(times) or len(camera_rows) != len(times):
        raise ValueError('Camera cache incomplete')
    _, k, _, _ = camera.rectification(read(snapshot()/'track_3/public/meta/camera_calibration.json'))
    mesh = trimesh.load(project_path(ROOT, item['mesh']['path']), force='mesh', process=False)
    model = fit.geometry(mesh.vertices, mesh.faces, item['object'])
    if configuration:
        trace = read(project_path(ROOT, item['mesh']['trace']))['rows']
        binding = next(r for r in trace if r['episode'] == item['mesh']['episode'] and r['object'] == item['object'])
        model.update(configuration=configuration, canonical_rotation=np.asarray(binding['coordinate_rotation']))
    poses = []; rows = []; valid = []
    checkpoint = directory/'checkpoint.npz'; diagnostics = directory/'checkpoint.json'
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            poses = list(saved['poses']); valid = list(saved['valid'])
        rows = read(diagnostics)[:len(poses)]
        if len(rows) != len(poses):
            raise ValueError('Incomplete checkpoint diagnostics')
    previous = poses[-1] if poses else None
    for index in range(len(poses), len(times)):
        frame = full_c.read_frame(cache, index)
        co, detail = track_frame(frame, k, model, cameras[index], previous, rows[-1] if rows else None)
        observed = frame_valid(detail, camera_rows[index]['observed'], co) and detail['fitted']
        previous = cameras[index]@co
        poses.append(previous); valid.append(observed)
        rows.append(dict(detail, frame_index=index, observed=observed, keyframe_refined=False))
        if index % 10 == 0 or index == len(times)-1:
            write_json(diagnostics, rows)
            temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, poses=poses, valid=valid)
            temporary.replace(checkpoint)
            print(json.dumps(dict(episode=item['episode'], object=item['object'], frame=index,
                                  total=len(times), iou=detail['iou'], valid=observed)), flush=True)
    poses = np.asarray(poses); valid = np.asarray(valid, bool)
    save_track(directory, item, cameras, poses, times, valid, rows, k, model, camera_rows)
    if input_identity(item, configuration, protocol_path) != identity:
        raise ValueError('Inputs or source changed during the track')
    artifacts = {relative(p): sha256_file(p) for p in directory.iterdir() if p.is_file() and p.name != 'run.json'}
    write_json(directory/'seal.json', dict(sealed_utc=now_utc(), labels_read=False, files=artifacts))
    write_json(directory/'run.json', dict(status='completed', labels_read=False, gpu=False,
               frames=len(times), wall_seconds=time.perf_counter()-started,
               reference_sha256=sha256_file(directory/'reference.npz'), sealed_utc=now_utc()))
    return directory


def verify_seal(path):
    """Refuse a changed output or a changed tracker source/protocol."""
    authorize()
    seal = read(path)
    for name, digest in seal['files'].items():
        if sha256_file(project_path(ROOT, name)) != digest:
            raise ValueError(f'Sealed file changed: {name}')
    return seal


def run_batch(queue_path):
    """Run every ready clip once, then seal the entire batch before evaluation."""
    authorize(); cv2.setNumThreads(2)
    queue = read(queue_path)
    if queue['settings'] != fit.SETTINGS or not queue['clips']:
        raise ValueError('Empty queue or settings differ from frozen recipe')
    if (OUTPUT/'outputs-seal.json').exists():
        verify_seal(OUTPUT/'outputs-seal.json'); return
    started = time.perf_counter(); directories = []
    for item in queue['clips']:
        directory = track_clip(item); directories.append(directory)
        with (OUTPUT/'ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(episode=item['episode'], object=item['object'],
                         status='sealed', path=relative(directory), utc=now_utc()))+'\n')
    files = {relative(d/'seal.json'): sha256_file(d/'seal.json') for d in directories}
    files.update({relative(queue_path): sha256_file(queue_path), relative(PROTOCOL): sha256_file(PROTOCOL)})
    write_json(OUTPUT/'outputs-seal.json', dict(files=files, sources=sources(), settings=fit.SETTINGS,
               directories=[relative(d) for d in directories], sealed_utc=now_utc(), labels_read=False,
               wall_seconds=time.perf_counter()-started))


def evaluate_clip(directory):
    """Open mocap only here, after all output and input identities are sealed."""
    from src.data.episodes import iter_public_frames
    request = read(directory/'request.json'); item = request['item']; episode = item['episode']
    authorize(episode); verify_seal(directory/'seal.json')
    if input_identity(item) != request:
        raise ValueError('Sources or inputs changed before label access')
    with np.load(directory/'reference.npz', allow_pickle=False) as saved:
        poses = saved['world_T_object'].copy(); times = saved['time'].copy()
    label = snapshot()/f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet'
    if not label.is_file():
        raise FileNotFoundError(f'Public labels missing: {label}')
    write_json(directory/'label-access.json', dict(first_access_utc=now_utc(),
               output_seal_sha256=sha256_file(OUTPUT/'outputs-seal.json'), label=relative(label),
               label_sha256=sha256_file(label), purpose='sealed evaluation only'))
    truth = np.tile(np.eye(4), (len(poses), 1, 1)); available = np.zeros(len(poses), bool)
    label_times = np.full(len(poses), np.nan)
    descriptor = dict(split='public', episode_index=episode,
                      trajectory=f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet')
    for row in iter_public_frames(snapshot(), descriptor):
        index = row['frame_index']
        if index >= len(poses):
            raise ValueError('Labels extend past video grid')
        obj, = [o for o in row['objects'] if o['name'] == item['object']]
        label_times[index] = row['timestamp_seconds']
        if obj['valid']:
            p = np.asarray(obj['pose_xyz_wxyz'])
            truth[index, :3, 3] = p[:3]
            truth[index, :3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
            available[index] = True
    if not available[0] or not np.isfinite(label_times).all():
        raise ValueError('Missing frame zero label or incomplete label clock')
    clock_error = float(np.max(abs((label_times-label_times[0])-(times-times[0]))))
    if clock_error > .001:
        raise ValueError(f'Video/mocap clock discrepancy {clock_error:.6f}s exceeds 1ms')
    model_info = read(directory/'mesh-identity.json')
    errors = fit.pose_errors(poses, truth, dict(axis=np.asarray(model_info['axis_model']), symmetry=model_info['symmetry']))
    position, angle, raw_position, raw_angle = [e[available] for e in errors]
    gates = read(directory/'gates.json')
    result = dict(episode=episode, object=item['object'], frames=len(poses), label_frames=int(available.sum()),
        label_coverage=float(available.mean()), position_lt_3cm_fraction=float(np.mean(position < .03)),
        angle_lt_30deg_fraction=float(np.mean(angle < 30)), joint_lt_3cm_30deg_fraction=float(np.mean((position < .03) & (angle < 30))),
        median_position_cm=float(np.median(position)*100), median_angle_deg=float(np.median(angle)),
        raw_unaligned_median_position_cm=float(np.median(raw_position)*100), raw_unaligned_median_angle_deg=float(np.median(raw_angle)),
        label_time_max_discrepancy_s=clock_error, reference_sha256=sha256_file(directory/'reference.npz'),
        initial_alignment='truth[0] @ inverse(prediction[0]); motion diagnostic, not absolute pose accuracy')
    result['passed'] = bool(result['label_coverage'] >= .9 and result['position_lt_3cm_fraction'] >= .9 and
        result['angle_lt_30deg_fraction'] >= .9 and result['joint_lt_3cm_30deg_fraction'] >= .9 and
        result['median_position_cm'] <= 2 and result['median_angle_deg'] <= 15 and gates['quality_passed'])
    with (directory/'evaluation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['frame', 'time_s', 'label_valid', 'position_m', 'angle_deg', 'raw_position_m', 'raw_angle_deg'])
        for i in range(len(poses)):
            writer.writerow([i, times[i], int(available[i]), *[float(e[i]) if available[i] else '' for e in errors]])
    write_json(directory/'evaluation.json', result)
    return result


def evaluate_batch():
    """Evaluate only after the whole batch seal exists and matches frozen code."""
    batch = verify_seal(OUTPUT/'outputs-seal.json')
    if batch['sources'] != sources() or batch['settings'] != fit.SETTINGS:
        raise ValueError('Batch tracker changed since sealing')
    results = []
    for name in batch['directories']:
        directory = project_path(ROOT, name)
        verify_seal(directory/'seal.json')
        path = directory/'evaluation.json'
        result = read(path) if path.exists() else evaluate_clip(directory)
        results.append(result)
    write_json(OUTPUT/'evaluation.json', dict(results=results, all_classes_passed=False,
               reason='Missing full-clip masks/depth for other classes; spoon absent from allowed population', utc=now_utc()))
    summary()


def summary():
    """Write a per-clip table and explicit missing-cache commands."""
    inv = read(OUTPUT/'inventory.json')['episodes']; lines = [
        '# Depth tracker revision 1', '',
        f"Protocol SHA256: `{authorize()['sha256']}`. CPU only; scored episodes untouched.", '',
        'All accuracy values use one frame-zero alignment and include invalid tracker frames. This is a video reference, not a policy rollout.', '',
        '| Clip / object | Frames | Valid | IoU median | Coverage | Max m/s | Max deg/s | <3cm | <30deg | Both | Median cm / deg | Result |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for item in inv:
        directory = OUTPUT/f"ep{item['episode']}-{item['object']}"; gate_path = directory/'gates.json'
        if not gate_path.exists():
            lines.append(f"| {item['episode']} / {item['object']} | {item['frames']} | — | — | — | — | — | — | — | — | — | missing full inputs |")
            continue
        g = read(gate_path)['gates']; ep = directory/'evaluation.json'
        e = read(ep) if ep.exists() else None
        values = [g[n]['value'] for n in QUALITY if n != 'reprojection_iou_p10']
        label = ([e['position_lt_3cm_fraction'], e['angle_lt_30deg_fraction'], e['joint_lt_3cm_30deg_fraction']] if e else [None]*3)
        fmt = lambda v: '—' if v is None else f'{v:.4f}'
        med = f"{e['median_position_cm']:.3f} / {e['median_angle_deg']:.3f}" if e else '—'
        lines.append(f"| {item['episode']} / {item['object']} | {item['frames']} | "+' | '.join(map(fmt, [*values, *label]))+f" | {med} | "+('PASS' if e and e['passed'] else 'FAIL' if e else 'sealed, unevaluated')+' |')
    lines += ['', '## Missing inputs and follow-up', '',
        'No full-clip SAM2 masks exist outside 39. Sparse stereo counts and frame-zero FoundationPose inputs are in inventory.json. Those frame-zero masks are not a SAM2 motion evaluation. No allowed spoon clip exists; spoon generalization remains unmeasured.', '',
        'After the current campaign releases the GPU lease, run the following separately authorized GPU job (one lease/operation at a time). `prepare-inputs` uses the existing DINO/SAM workers under GpuLease, then CPU SGBM and the existing background camera BA. It does not run tracking or read labels. Set a future UTC budget explicitly; old mask-runtime deadlines have expired.', '',
        '```bash', 'export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2',
        '# Set TRACKER_STOP_UTC to the agreed future UTC deadline, e.g. ISO8601 +00:00.']
    for item in inv:
        if not item['ready']:
            lines.append(f"data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.depth_track prepare-inputs --gpu --episode {item['episode']} --object {item['object']} --stop-utc \"$TRACKER_STOP_UTC\"")
    lines += ['```', '',
        'Use a NEW frozen revision for additional class evaluation; do not overwrite this revision\'s sealed queue or outputs. Instance masks for identical/overlapping objects need visual identity review. Failed detection, occlusion or unresolved camera/table checks remain failures; preparation grants no admission.', '',
        'All campaign checks retain their thresholds. Feature-count, FB, table and support admission evidence is absent for this experiment. `gates_passed` remains false. No campaign files were edited. The frozen text calls episode 42 a frying pan; its cached measurement names the object `white_pot`, which is the queue identity used here.']
    (OUTPUT/'summary.md').write_text('\n'.join(lines)+'\n')


def prepare_inputs(episode, name, stop_utc, gpu):
    """Resume missing caches in a separately requested GPU job under the existing lease."""
    from . import masks, track, camera_ba, candidate_c
    from src.gpu import GpuLease
    if not gpu or name not in OBJECTS.get(episode, []):
        raise PermissionError('Preparation needs --gpu and an allowed episode/object')
    authorize(episode)
    base = OUTPUT/f'inputs/ep{episode}-{name}'; mask_dir = base/'masks'; cache = base/'cache'
    output_path(ROOT, base); mask_dir.mkdir(parents=True, exist_ok=True); cache.mkdir(exist_ok=True)
    record_path = base/'run.json'
    identity = dict(sources=sources(), episode=episode, object=name,
        preparation_sources={relative(p): sha256_file(p) for p in
            [Path(masks.__file__), Path(camera_ba.__file__), Path(candidate_c.__file__),
             Path(__file__).with_name('_worker.py'), ROOT/'src/gpu.py']})
    previous = read(record_path) if record_path.exists() else None
    if previous and previous['identity'] != identity:
        raise ValueError('Preparation source changed; preserve this attempt')
    if previous and previous['status'] == 'completed':
        for path, digest in previous['artifacts'].items():
            if sha256_file(project_path(ROOT, path)) != digest:
                raise ValueError('Completed preparation artifact changed')
        print(f'Skip completed: public {episode} {name}', flush=True)
        return
    record = dict(status='running', identity=identity, started_utc=now_utc(), labels_read=False)
    write_json(record_path, record)
    try:
        mask_run = mask_dir/'run.json'
        mask_record = read(mask_run) if mask_run.exists() else dict(status='running',
            kind='moving_reference.depth_track_masks', stages={}, labels_read=False)
        if mask_record['status'] != 'completed':
            with GpuLease(ROOT, 1024, 10000, stage='moving_reference.masks') as lease:
                rule = read(Path(__file__).with_name('runtime.json'))
                rule.update(queue=[dict(split='public', episode=episode)],
                            pan_prompts=[name.replace('_', ' ')], stop_utc=stop_utc)
                job = masks.prepare(rule['queue'][0], rule, mask_dir)
                job['rule']['stop_utc'] = stop_utc
                write_json(mask_dir/'request.json', job)
                write_json(mask_run, mask_record)
                for operation, runtime in [('detect', 'grounding-dino'), ('segment', 'sam2')]:
                    if mask_record['stages'].get(operation, {}).get('status') == 'completed':
                        continue
                    lease.require_headroom()
                    command = [str(ROOT/f'data/.runtime/{runtime}/venv/bin/python'), '-B', '-m',
                               'src.features.moving_reference._worker', operation, str(mask_dir/'request.json')]
                    with (mask_dir/f'{operation}.log').open('a') as stream:
                        subprocess.run(command, cwd=ROOT, env=masks.environment(mask_dir),
                                       stdout=stream, stderr=subprocess.STDOUT, check=True)
                    mask_record['stages'][operation] = read(mask_dir/f'{operation}-report.json')
                    write_json(mask_run, mask_record)
                    if operation == 'detect' and read(mask_dir/'detections.json')['pan'] is None:
                        raise ValueError('Object detection absent; no substitution')
                mask_record.update(status='completed', gpu=lease.report())
                write_json(mask_run, mask_record)
        job = read(mask_dir/'request.json')
        mask_record['wall_seconds'] = sum(stage.get('wall_seconds', 0.)
                                          for stage in mask_record.get('stages', {}).values())
        masks.summarize(mask_dir, job, mask_record)
        videos = [snapshot()/f'track_3/public/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4' for key in ('c', 'b')]
        selected = dict(job=job, paths=videos, masks=mask_dir,
            mask_files=[mask_dir/'frames'/f'frame_{i:06d}.npz' for i in range(job['frame_count'])],
            calibration=snapshot()/'track_3/public/meta/camera_calibration.json')
        if not (cache/'frame0.npz').exists():
            candidate_c.prepare_frame0(selected, cache)
        full_c.cache_frames(selected, cache)
        if not (cache/'camera-ba.npz').exists() or not (cache/'camera-diagnostics.json').exists():
            data = camera_ba.observations(selected, cache)
            cameras, ids, landmarks, report = camera_ba.bundle_adjust(data)
            np.savez_compressed(cache/'camera-ba.npz', world_T_camera=cameras, landmark_ids=ids, landmarks=landmarks)
            write_json(cache/'bundle-adjustment.json', report)
            write_json(cache/'camera-diagnostics.json', camera_ba.heldout_rows(data, cameras))
        if not (cache/'table.json').exists():
            with np.load(cache/'camera-ba.npz') as arrays:
                _, table = full_c.table_fit(cache, arrays['world_T_camera'])
            write_json(cache/'table.json', table)
        files = [p for p in base.rglob('*') if p.is_file() and
                 p.suffix in ('.json', '.npz') and p != record_path and
                 '.cache' not in p.parts and '.tmp' not in p.parts]
        record.update(status='completed', finished_utc=now_utc(),
                      artifacts={relative(p): sha256_file(p) for p in files})
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now_utc())
        raise
    finally:
        write_json(record_path, record)


def evaluate_m1_revision(directory):
    """Score a sealed tuning revision with the independently reproduced M1 evaluator."""
    import pyarrow.parquet as pq
    authorize(39); verify_seal(directory/'seal.json')
    request = read(directory/'request.json'); item = request['item']
    protocol = project_path(ROOT, request['protocol_path'])
    if input_identity(item, request['configuration'], protocol) != request:
        raise ValueError('Revision changed before evaluation')
    path = directory/'evaluation-m1.json'
    if path.exists():
        return read(path)
    reference = directory/'reference.npz'
    write_json(directory/'evaluation-request.json', dict(utc=now_utc(),
        reference_sha256=sha256_file(reference), seal_sha256=sha256_file(directory/'seal.json'),
        evaluator_sha256=sha256_file(ROOT/'data/outputs/moving_reference/ep39/sealed/sealed-m1-27.py'),
        correspondence='M1 equal frame indices, no timestamp interpolation; same initial-pose alignment',
        tracker_pass_bar_changed=False))
    label = snapshot()/'track_3/public/data/chunk-000/episode_000039.parquet'
    rows = pq.read_table(label, columns=['observation.objects']).to_pylist()
    objects = [next(o for o in r['observation.objects'] if o['name'] == item['object']) for r in rows]
    mocap = np.asarray([o['pose'] for o in objects], float)
    available = np.asarray([o['visible'] for o in objects]) & np.isfinite(mocap).all(1)
    available &= abs(np.linalg.norm(mocap[:, 3:], axis=1)-1) <= .001
    with np.load(reference) as saved:
        position = saved['position']; quaternion = saved['quaternion_xyzw']
    if len(position) != len(mocap) or not available[0]:
        raise ValueError('M1 requires equal-length arrays and a valid first label')
    model = read(directory/'mesh-identity.json')
    errors = fit.m1_errors(position, quaternion, mocap[:, :3], mocap[:, [4, 5, 6, 3]],
                          np.asarray(model['axis_model']), model['symmetry'])
    pos, angle = [e[available] for e in errors]
    gates = read(directory/'gates.json')
    result = dict(label_coverage=float(available.mean()), label_frames=int(available.sum()),
        position_lt_3cm_fraction=float(np.mean(pos < .03)), angle_lt_30deg_fraction=float(np.mean(angle < 30)),
        joint_lt_3cm_30deg_fraction=float(np.mean((pos < .03) & (angle < 30))),
        median_position_cm=float(np.median(pos)*100), median_angle_deg=float(np.median(angle)),
        mean_position_cm=float(np.mean(pos)*100), mean_angle_deg=float(np.mean(angle)),
        reference_sha256=sha256_file(reference), label_sha256=sha256_file(label))
    result['passed'] = bool(result['label_coverage'] >= .9 and result['position_lt_3cm_fraction'] >= .9 and
        result['angle_lt_30deg_fraction'] >= .9 and result['joint_lt_3cm_30deg_fraction'] >= .9 and
        result['median_position_cm'] <= 2 and result['median_angle_deg'] <= 15 and gates['quality_passed'])
    baseline = read(OUTPUT/'evaluation-m1/candidate-c.json')
    result['beats_candidate_c'] = bool(result['joint_lt_3cm_30deg_fraction'] > baseline['joint_lt_3cm_30deg_fraction'] and
        result['median_position_cm'] <= baseline['median_position_cm'] and result['median_angle_deg'] <= baseline['median_angle_deg'])
    write_json(path, result)
    np.savez_compressed(directory/'evaluation-m1-errors.npz', position_m=errors[0], angle_deg=errors[1], label_valid=available)
    return result


def comparison_summary():
    """Append the corrected evaluator comparison while retaining the original failed record."""
    baseline = read(OUTPUT/'evaluation-m1/results.json')['results']
    candidates = [('Candidate C', baseline['candidate-c'], ROOT/'data/outputs/moving_reference/ep39/candidate-c/gates-final.json'),
                  ('Revision 1', baseline['revision-1'], OUTPUT/'ep39-mini_dust_pan/gates.json')]
    for revision in (2, 3, 4):
        directory = OUTPUT/f'revision-{revision}'
        if (directory/'evaluation-m1.json').exists():
            candidates.append((f'Revision {revision}', read(directory/'evaluation-m1.json'), directory/'gates.json'))
    lines = ['## Follow-up: corrected M1 evaluator and tuning revisions', '',
        'M1 uses equal frame indices and a single initial rigid alignment, with no timestamp warp. '
        'The unsupported 1ms rejection is corrected; the measured clock discrepancy remains recorded. '
        'Candidate C reproduces its saved errors (position difference 0, angle difference < 1.1e-14 degrees). '
        'The original revision-1 FAIL, bytes, source and protocol are retained under frozen-v1/. '
        'These are tuning-development results on 39, not an independent assessment or campaign admission.', '',
        '| Method | Frames | Valid | IoU median | IoU p10 | Coverage | Max m/s | Max deg/s | <3cm | <30deg | Both | Median cm / deg | Result |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name, e, gate_path in candidates:
        gates = read(gate_path)['gates']
        values = [gates[n]['value'] for n in ('object_reference_valid_fraction', 'reprojection_iou_median',
            'reprojection_iou_p10', 'reprojection_coverage_fraction', 'object_speed_max_m_s', 'object_angular_rate_max_deg_s')]
        values += [e['position_lt_3cm_fraction'], e['angle_lt_30deg_fraction'], e['joint_lt_3cm_30deg_fraction']]
        status = 'baseline (joint <90%)' if name == 'Candidate C' else 'PASS' if e.get('passed') else 'FAIL'
        lines.append(f'| {name} | 319 | '+' | '.join('—' if v is None else f'{v:.4f}' for v in values)+
            f" | {e['median_position_cm']:.3f} / {e['median_angle_deg']:.3f} | {status} |")
    lines += ['', 'Hand/background diagnosis: [8 frames around first failure](diagnosis-r1/first-failure-8-frames.png), '
        '[depth p90 and hand overlap](diagnosis-r1/depth-p90-hand-overlap.png), [quantitative counts](diagnosis-r1/diagnosis.json). '
        'First invalid frame is 0; first position error over 3cm is frame 1. None of 5,341 sampled >30mm outliers '
        'are inside the known hand mask; 16.5% are within 14px of it, 64.5% are outside the mesh projection, '
        'and only 4.0% are >30mm behind the nearest surface. Hand/background leakage is not established as the cause. '
        'The own-mesh canonical up is 77.08 degrees from asset z; initialization is defective. '
        'Known hand pixels were excluded from depth in revision 1 already.', '',
        'Each revision keeps the same mesh and full-cloud validity residuals. Trimming changes only optimization, '
        'not its residual denominator or pass bars. The GPU preparation script is data/.runtime/tmp/depth-track-gpu-prep.sh; '
        'it was not run. No allowed spoon clip exists.']
    text = '\n'.join(lines)+'\n'
    (OUTPUT/'comparison.md').write_text(text)
    summary_path = OUTPUT/'summary.md'
    previous = summary_path.read_text().split('\n## Follow-up: corrected M1 evaluator and tuning revisions')[0]
    summary_path.write_text(previous+'\n'+text)


def run_revisions(queue_path):
    """Execute at most three fixed tuning revisions, recording each protocol before its run."""
    authorize(39); cv2.setNumThreads(2)
    queue = read(queue_path)
    revisions = queue['revisions']
    if len(revisions) > 3 or [r['revision'] for r in revisions] != list(range(2, 2+len(revisions))):
        raise ValueError('Only ordered revisions 2..4 are allowed')
    item, = [i for i in read(OUTPUT/'queue.json')['clips'] if i['episode'] == 39 and i['object'] == 'mini_dust_pan']
    for configuration in revisions:
        revision = configuration['revision']; directory = OUTPUT/f'revision-{revision}'
        directory.mkdir(exist_ok=True)
        protocol = directory/'protocol.md'
        if not protocol.exists():
            entry = (f"\n## Tuning revision {revision} — {now_utc()}\n\n"+
                configuration['reason']+'\n\nSettings fixed before this run: `'+json.dumps(configuration, sort_keys=True)+'`.\n\n'+
                'Same mesh, validity residuals, 90% accuracy bars and all campaign instrument thresholds. '
                'M1 evaluator correction applies unchanged. This is development on tuning 39 only. '
                'Stop early only if this revision passes AND has a higher joint success fraction than candidate C '
                'with both median errors no larger. No scored tracking or admission.\n')
            with PROTOCOL.open('a') as stream:
                stream.write(entry)
            protocol.write_bytes(PROTOCOL.read_bytes())
            write_json(directory/'protocol-seal.json', dict(sha256=sha256_file(protocol), utc=now_utc(), labels_read_for_this_revision=False,
                       protocol=relative(protocol), queue_sha256=sha256_file(queue_path)))
        track_clip(item, directory, configuration, protocol)
        result = evaluate_m1_revision(directory)
        comparison_summary()
        with (OUTPUT/'revision-ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(revision=revision, reference_sha256=result['reference_sha256'],
                passed=result['passed'], beats_candidate_c=result['beats_candidate_c'], utc=now_utc()))+'\n')
        print(json.dumps(dict(revision=revision, **result)), flush=True)
        if result['passed'] and result['beats_candidate_c']:
            print('Stopped early: fixed pass bar and candidate-C comparison both passed', flush=True)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['inventory', 'run', 'evaluate', 'summary', 'prepare-inputs', 'revisions', 'comparison'])
    parser.add_argument('--queue', type=Path, default=OUTPUT/'queue.json')
    parser.add_argument('--episode', type=int, choices=ALLOWED)
    parser.add_argument('--object'); parser.add_argument('--gpu', action='store_true'); parser.add_argument('--stop-utc')
    args = parser.parse_args()
    if args.operation == 'inventory':
        inventory(); summary()
    elif args.operation == 'run':
        run_batch(project_path(ROOT, args.queue)); summary()
    elif args.operation == 'evaluate':
        evaluate_batch()
    elif args.operation == 'summary':
        summary()
    elif args.operation == 'revisions':
        run_revisions(project_path(ROOT, args.queue))
    elif args.operation == 'comparison':
        comparison_summary()
    else:
        if not args.stop_utc:
            parser.error('prepare-inputs requires --stop-utc')
        prepare_inputs(args.episode, args.object, args.stop_utc, args.gpu)


if __name__ == '__main__':
    main()
