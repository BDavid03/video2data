"""Build sealed video-reference tasks and admit their frozen M2v operations."""

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

from . import handler as h
from .geometry import cylinder_support, matrix, orientation_errors

FINAL = 66763776
BOUNDARY = 32814080
# Translation only, from the recorded S2 simulator's initial body position.
M2_POSITION = np.array([-.012269774451851845, -.10382826626300812, .6182625889778137])
ANCHOR_SOURCE = 'data/outputs/moving_policy/ep39-a1/S2/steps.json:measurements.trajectories.steps[0].pose_xyzw[:3]'


def select(episode, output):
    """Constrain each video task to its episode-specific artifact directory."""
    if episode not in (38, 39):
        raise ValueError('Video tasks admit only episodes 38 and 39')
    output = output.resolve()
    output.relative_to(h.ROOT / 'data/outputs/moving_policy')
    if output == h.ROOT / 'data/outputs/moving_policy' or output.name.startswith(('ep39-a1', 'ep39-a0')) or output.name == 'ep39':
        raise ValueError('Historical M2 records are immutable')
    h.OUTPUT = output
    h.TASK = output / f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
    return output


def shared_asset(body_path):
    """Write one profile asset using M1's geometric body constant, without reading a scan."""
    from src.features.mesh_extents._shapes import thin_tool_profile
    from src.features.moving_reference._object import body_axes
    queue = json.loads((h.FEATURE / 'queue.json').read_text())
    measure_path = h.ROOT / queue['measurement']
    attributes = h.ROOT / queue['attributes']
    dimensions = json.loads(measure_path.read_text())['objects']['mini_dust_pan']['median_m']
    profile = json.loads(attributes.read_text())['objects']['mini_dust_pan']['profile']
    body_bytes = body_path.read_bytes()
    body = json.loads(body_bytes)
    if body['trajectory_used'] is not False or body['applies_identically_to_episodes'] != [39, 38]:
        raise ValueError('The shared constant must be geometric and episode independent')
    canonical = thin_tool_profile(dimensions, profile)
    bounds, extents = trimesh.bounds.oriented_bounds(canonical)
    samples, _ = trimesh.sample.sample_surface(canonical, 10000, seed=31)
    profile_axes, details = body_axes(bounds, extents, samples)
    from .video_geometry import profile_body_transform
    body_transform = profile_body_transform(body, profile_axes)
    mesh = canonical.copy()
    mesh.apply_transform(body_transform)
    directory = h.ROOT / 'data/outputs/moving_policy/video-asset'
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(inputs=h.input_inventory([body_path, measure_path, attributes]),
                    body_transform=body_transform.tolist(), dimensions_m=dimensions, profile=profile,
                    profile_T_canonical=profile_axes.tolist(), profile_axes_details=details,
                    method='M1 recorded scan bounding axes composed with inverse profile bounding convention; tracking asset_T_bounding_body removed when present; no new scan read or trajectory registration',
                    scan_reads=[], mass_kg=.3, inertia_diagonal_kg_m2=[.001]*3)
    manifest = directory / 'asset.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity'] != identity:
            raise ValueError('Shared asset identity changed; do not overwrite it')
        for name, digest in previous['hashes'].items():
            if h.digest(directory / name) != digest:
                raise ValueError('Shared asset changed: ' + name)
    else:
        mesh.export(directory / 'mini_dust_pan.obj')
        (directory / 'mini_dust_pan.urdf').write_text('''<robot name="mini_dust_pan"><link name="mini_dust_pan">
<inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
<visual><geometry><mesh filename="mini_dust_pan.obj"/></geometry></visual>
<collision><geometry><mesh filename="mini_dust_pan.obj"/></geometry></collision>
</link></robot>
''')
        (directory / 'body-convention.json').write_bytes(body_bytes)
        h.write(manifest, dict(identity=identity, hashes={name:h.digest(directory / name) for name in ('mini_dust_pan.obj', 'mini_dust_pan.urdf', 'body-convention.json')}))
    return canonical, body_transform, directory, identity


def task_reference(reference, output):
    """Snapshot the hashed input and write its purely translated task-world equivalent."""
    data = reference.read_bytes()
    source_hash = __import__('hashlib').sha256(data).hexdigest()
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        arrays = {key:loaded[key].copy() for key in loaded.files}
    times = np.asarray(arrays['time'], float)
    positions = np.asarray(arrays['position'], float)
    quaternions = np.asarray(arrays['quaternion_xyzw'], float)
    if (len(times) < 2 or positions.shape != (len(times), 3) or quaternions.shape != (len(times), 4)
            or not np.array_equal(arrays['frame_index'], np.arange(len(times)))
            or not np.all(np.diff(times) > 0) or not np.isfinite(np.c_[positions, quaternions]).all()
            or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-6)):
        raise ValueError('Invalid video reference frame grid, clock or poses')
    shift = M2_POSITION - positions[0]
    arrays['position'] = positions + shift
    arrays['pose'] = np.c_[arrays['position'], quaternions]
    arrays['world_T_object'] = arrays['world_T_object'].copy()
    arrays['world_T_object'][:, :3, 3] += shift
    (output / 'reference-source.npz').write_bytes(data)
    np.savez_compressed(output / 'reference-task.npz', **arrays)
    fps = (len(times) - 1) / (times[-1] - times[0])
    record = dict(source_path=str(reference.relative_to(h.ROOT)), source_sha256=source_hash,
                  task_sha256=h.digest(output / 'reference-task.npz'), translation_m=shift.tolist(),
                  frame0_target_m=M2_POSITION.tolist(), anchor_source=ANCHOR_SOURCE,
                  transform='pure translation; rotations unchanged',
                  fps=float(fps), largest_clock_residual_s=float(np.max(np.abs(times-times[0]-np.arange(len(times))/fps))),
                  m1_gates_passed=bool(arrays['gates_passed']), valid_fraction=float(arrays['valid'].mean()))
    h.write(output / 'reference.json', record)
    return np.c_[arrays['position'], quaternions[:, [3, 0, 1, 2]]], times, record


def worker(operation):
    """Use the existing evaluator worker while keeping each candidate's own cache."""
    return h.amended_worker(operation)


def commands(build, episode, output):
    """Prepare final-only evaluator commands and the CPU boundary replay."""
    from src.features.dynamic_feasibility.handler import EVALUATOR
    command = h.training_argv(build)
    score = output / f'score-{FINAL}'
    evaluator = [str(h.PYTHON), '-B', '-m', 'src.features.dynamic_feasibility._worker', str(EVALUATOR),
                 f'evaluation.checkpoint={output / "training" / f"policy_{FINAL}.safetensors"}',
                 'evaluation.world_count=1', 'evaluation.reset_mode=explicit', 'evaluation.start_frame=0',
                 f'evaluation.object_trajectories_output={score / "trajectory.parquet"}',
                 f'evaluation.metrics_output={score / "official_metrics.json"}',
                 f'hydra.run.dir={score / "hydra"}', 'logging.mode=disabled']
    base = [str(h.ROOT / 'data/.runtime/linux-cuda/bin/python'), '-B', '-m', 'src.features.moving_policy.video']
    suffix = ['--episode', str(episode), '--output', str(output)]
    result = dict(train=command, evaluator_final=evaluator,
                  score_self=base + ['score-self', *suffix, '--run'],
                  kill_check=base + ['kill-check', *suffix, '--run'], maximum_seconds=4500,
                  final_steps=FINAL, boundary_steps=BOUNDARY, seed=0)
    if episode == 39:
        result['score_mocap'] = base + ['score-mocap', *suffix, '--run']
    h.write(output / 'commands.json', result)
    for name, argv in result.items():
        if isinstance(argv, list):
            (output / (name.replace('_', '-') + '.argv.txt')).write_text(shlex.join(argv) + '\n')
    return result


def summary(stage, output):
    """Write status first with the measured gates and immutable identities."""
    failed = [name for name, check in stage['checks'].items() if not check['passed']]
    status = 'PASS' if not failed else 'FAIL: ' + ', '.join(failed)
    lines = [f'# Episode {stage["episode"]} Stage 0: {status}', '',
             f'Asset OBJ SHA-256: `{stage["asset_hashes"]["mini_dust_pan.obj"]}`.',
             f'Asset URDF SHA-256: `{stage["asset_hashes"]["mini_dust_pan.urdf"]}`.',
             f'Reference source SHA-256: `{stage["reference"]["source_sha256"]}`; task: `{stage["reference"]["task_sha256"]}`.', '',
             '| Check | Pass | Measurement |', '| --- | --- | --- |']
    for name, check in stage['checks'].items():
        if name == '2_closure': detail = f'{check["frames_passing"]}/{check["post_acquisition_frames"]} frames'
        elif name == '3_penetration': detail = f'{check["maximum_m"]:.7g} m'
        elif name == '4a_finger_limits': detail = f'{check["joint_limit_excess_rad"]:.7g} rad excess'
        elif name == '4b_reachability':
            detail = '; '.join(f'{side}: mean/max {r["mean_position_error_m"]:.7g}/{r["maximum_position_error_m"]:.7g} m, mean {r["mean_orientation_error_deg"]:.7g} deg' for side,r in check['measurements']['diagnostic-feedforward']['hands'].items())
        elif name == '5_support':
            detail = '; '.join(f'{mode}: {r["gated_steps"]} gated steps, {r["gated_maximum_translation_m"]:.7g} m / {r["gated_maximum_rotation_deg"]:.7g} deg' for mode,r in check['measurements'].items())
        elif name == '6_truth': detail = f'{check["maximum_translation_error_m"]:.7g} m; {check["maximum_orientation_error_deg"]:.7g} deg; identity AUC {check["score"]["AUC"]:.7g}'
        else: detail = f'{check["frames"]} frames, {check["fields"]} fields'
        lines.append(f'| {name} | {check["passed"]} | {detail} |')
    b = stage['build']
    lines += ['', f'Grasp: **{stage["selected_grasp"]}**; {b["approach_disclosure"]}.',
              f'Peak wrist speed: {max(stage["checks"]["4a_finger_limits"]["maximum_wrist_speed_m_s"].values()):.7g} m/s. Peak object speed: {b["motion"]["maximum_speed_m_s"]:.7g} m/s; angular rate: {b["maximum_angular_rate_deg_s"]:.7g} deg/s.',
              f'Clock residual: {stage["reference"]["largest_clock_residual_s"]:.7g} s. Translation: `{stage["reference"]["translation_m"]}` m.',
              f'Imported collision pieces: {b["collision_pieces"]}, through installed unedited evaluator CoACD. Support: footprint cylinder, top 1 mm below imported hulls; mass 0.3 kg, diagonal inertia 0.001 kg m².',
              f'M1 source admission at snapshot: **{stage["reference"]["m1_gates_passed"]}**. Passing Stage 0 alone never authorizes training.',
              'No mocap, observation.objects, scan, excluded episode, GPU, training or label score was read/run. The anchor uses only the pre-existing M2 initial simulator translation; no M2 rotation or grasp is reused.',
              'Dry training, final-only self score, and CPU policy_32814080 kill-check: `commands.json` / `*.argv.txt`. All video runtime flags remain false. Any failed check stops this episode; no repair or retry.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def stage0(args):
    """Run the fixed per-episode grasp rule and prescribed CPU Stage 0 once."""
    output = select(args.episode, args.output)
    if (output / 'stage0.json').exists():
        raise FileExistsError('Stage 0 is immutable once recorded')
    output.mkdir(parents=True, exist_ok=True)
    poses, times, reference = task_reference(args.reference.resolve(), output)
    canonical, body_transform, asset, identity = shared_asset(args.body_convention.resolve())
    for name in ('mini_dust_pan.obj', 'mini_dust_pan.urdf'):
        shutil.copyfile(asset / name, output / name)
    h.write(output / 'asset.json', json.loads((asset / 'asset.json').read_text()))
    os.environ.update(h.environment())
    template_path = next(h.TEMPLATE.glob('*.parquet'))
    parquet = pq.ParquetFile(template_path)
    template = parquet.read().to_pylist()[0]
    h.cpu('template', template_path)
    source_paths = [*h.FEATURE.glob('*.py'), h.FEATURE/'queue.json',
                    h.ROOT/'docs/chain/moving-m2v-protocol.md', h.ROOT/'docs/chain/moving-m2-protocol.md',
                    template_path, output/'reference-source.npz', *[h.ROOT/e['path'] for e in identity['inputs'] if e['path'] != str(args.body_convention.resolve().relative_to(h.ROOT))],
                    asset/'body-convention.json', asset/'asset.json',
                    *list((h.FC/'src/flash_chord').rglob('*.py'))]
    sources = h.input_inventory(source_paths)
    candidates = []
    for pair in ('longitudinal', 'transverse'):
        candidate = output / pair
        select(args.episode, candidate)
        candidate.mkdir(exist_ok=True)
        for name in ('mini_dust_pan.obj', 'mini_dust_pan.urdf'):
            shutil.copyfile(asset / name, candidate / name)
        with np.load(output / 'template-fk.npz') as fk:
            table, grasp, acquisition, fps = h.construct(template, parquet.schema_arrow, poses, times,
                    canonical, fk, True, pair, identity['profile']['head_fraction'], body_transform=body_transform)
        row = table.to_pylist()[0]
        row['raw_motion_file'] = f'M2v episode {args.episode}: hashed video reference; object-driven template hands, not recovered human motion'
        # Both tasks point to the very same files, not independently rebuilt assets.
        row['object_mesh_paths'] = [str(asset / 'mini_dust_pan.obj')]
        row['object_urdf_paths'] = [str(asset / 'mini_dust_pan.urdf')]
        row['object_mesh_radius'] = [float(np.linalg.norm(canonical.vertices @ body_transform[:3,:3].T + body_transform[:3,3], axis=1).max())]
        table = pa.Table.from_pylist([row], schema=table.schema)
        h.TASK.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, h.TASK)
        rates = np.degrees((Rotation.from_matrix(matrix(poses[1:, 3:])) * Rotation.from_matrix(matrix(poses[:-1, 3:])).inv()).magnitude()) / np.diff(times)
        build = dict(episode=args.episode, frames=len(times), fps=fps, task=str(h.TASK.relative_to(h.ROOT)),
                     acquisition_end_frame=acquisition, grasp=grasp, bounds=canonical.bounds.tolist(),
                     dimensions_m=identity['dimensions_m'], profile=identity['profile'], body_transform=body_transform.tolist(),
                     motion=h.motion_summary(poses[:, :3], times), maximum_angular_rate_deg_s=float(rates.max()),
                     mass_kg=.3, inertia_kg_m2=(np.eye(3)*.001).tolist(), support_candidate='S2',
                     approach_disclosure='0.5 s approach' if acquisition else 'Hands start at grasp: no-approach branch, reference exceeds 1 mm / 1 degree; no rest prepended')
        h.write(candidate / 'build.json', build)
        h.cpu('inspect', h.TASK)
        info = json.loads((candidate / 'task-fk.json').read_text())
        with np.load(candidate / 'task-fk.npz') as fk:
            pieces = [fk[f'object_piece_{i}'].astype(float) for i in range(info['object_piece_count'])]
        center, radius, top = cylinder_support(pieces, poses[0, :3], matrix(poses[0, 3:]))
        support_path = candidate / f'reconstructed_stage/episode_{args.episode:06d}_sharpa_wave_support.usda'
        h.write_support(support_path, center, radius, top)
        build.update(support_top_m=top, support_radius_m=radius, support_center_xy_m=center,
                     collision_pieces=info['object_piece_count'], initial_hull_gap_m=.001)
        h.write(candidate / 'build.json', build)
        checks = h.geometry_checks(template, table, poses, times, build, body_transform=body_transform)
        with np.load(candidate / 'task-fk.npz') as fk:
            theta = orientation_errors(fk['quaternions'][:, 0, [1, 2, 3, 0]], poses[:, 3:])
        checks['6_truth']['maximum_orientation_error_deg'] = float(theta.max())
        checks['6_truth']['passed'] &= bool(theta.max() <= 1e-4)
        checks['6_truth']['reference_task_sha256'] = reference['task_sha256']
        saved = dict(pair=pair, directory=str(candidate.relative_to(h.ROOT)), build=build, checks=checks,
                     eligible=checks['2_closure']['passed'] and checks['3_penetration']['passed'],
                     peak_wrist_speed_m_s=max(checks['4a_finger_limits']['maximum_wrist_speed_m_s'].values()))
        h.write(candidate / 'candidate.json', saved)
        candidates.append(saved)
    eligible = [c for c in candidates if c['eligible']]
    chosen = min(eligible or candidates, key=lambda c:(c['peak_wrist_speed_m_s'], c['pair']))
    source = h.ROOT / chosen['directory']
    select(args.episode, output)
    for name in ('task-fk.npz', 'task-fk.json', 'build.json'):
        shutil.copyfile(source / name, output / name)
    h.TASK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / h.TASK.relative_to(output), h.TASK)
    support = output / f'reconstructed_stage/episode_{args.episode:06d}_sharpa_wave_support.usda'
    support.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / support.relative_to(output), support)
    build = dict(chosen['build'], task=str(h.TASK.relative_to(h.ROOT)))
    h.write(output / 'build.json', build)
    checks = chosen['checks']
    checks['4b_reachability'] = worker('reachability')
    checks['5_support'] = worker('support')
    prepared = commands(build, args.episode, output)
    hashes = json.loads((asset / 'asset.json').read_text())['hashes']
    artifacts = [h.TASK, support, output/'build.json', output/'commands.json', output/'reference-task.npz',
                 output/'reference-source.npz', output/'reference.json', output/'asset.json',
                 asset/'mini_dust_pan.obj', asset/'mini_dust_pan.urdf', output/'task-fk.npz',
                 output/'steps.json', output/'reachability.json']
    stage = dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'STOP',
                 episode=args.episode, checks=checks, reference=reference, build=build,
                 selected_grasp=chosen['pair'], grasp_candidates=candidates, asset_hashes=hashes,
                 inputs_and_sources=sources, artifacts=h.input_inventory(artifacts), commands=prepared,
                 gpu_used=False, training_run=False, mocap_read=False, scan_reads=[],
                 runtime_flags=json.loads((h.FEATURE/'runtime.json').read_text()))
    with (output/'stage0.json').open('x') as stream:
        json.dump(stage, stream, indent=2, allow_nan=False); stream.write('\n')
    summary(stage, output)
    print(json.dumps(dict(episode=args.episode, status=stage['status'], failed=[k for k,v in checks.items() if not v['passed']], asset_hashes=hashes)), flush=True)


def load_baseline():
    """Load the kit metric adapter without reading any dataset label or scan."""
    spec = importlib.util.spec_from_file_location('video_static_score', h.ROOT/'scripts/track3/static_pose_baseline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(achieved, reference):
    """Compute the established motion metrics and orientation bounds on arrays."""
    from .video_geometry import reference_metrics
    return reference_metrics(achieved, reference)


def score_export(output, mocap=False):
    """Measure one final evaluator export; labels are admitted only by score-mocap on 39."""
    stage = json.loads((output/'stage0.json').read_text())
    directory = output/f'score-{FINAL}'
    name = 'mocap_metrics.json' if mocap else 'self_metrics.json'
    if (directory/name).exists():
        raise FileExistsError('Final metrics are immutable')
    file = directory/'trajectory.parquet'
    parquet = pq.ParquetFile(file)
    frame = parquet.read().to_pandas().sort_values('reference_frame')
    count = stage['build']['frames']
    checkpoint = output/'training'/f'policy_{FINAL}.safetensors'
    metadata = parquet.schema_arrow.metadata or {}
    if (len(frame) != count or not np.array_equal(frame.reference_frame, np.arange(count))
            or not frame.environment_valid.all() or set(frame.environment_id) != {0}
            or set(frame.object_body_id) != {0}
            or metadata.get(b'flash_chord.checkpoint_sha256', b'').decode() != h.digest(checkpoint)
            or not np.allclose(frame.time_s, np.arange(count)/stage['build']['fps'], atol=1e-6, rtol=0)):
        raise ValueError('Invalid final evaluator export grid, clock, or checkpoint')
    achieved = frame[[f'achieved_position_{a}' for a in 'xyz']+[f'achieved_quaternion_{a}' for a in 'xyzw']].to_numpy(float)
    if not np.isfinite(achieved).all() or not np.allclose(np.linalg.norm(achieved[:,3:], axis=1),1,atol=1e-4):
        raise ValueError('Invalid achieved poses')
    official = json.loads((directory/'official_metrics.json').read_text())
    with np.load(output/'reference-task.npz') as task:
        reference = np.c_[task['position'], task['quaternion_xyzw']]
    exported_reference = frame[[f'reference_position_{a}' for a in 'xyz']+[f'reference_quaternion_{a}' for a in 'xyzw']].to_numpy(float)
    if (not np.allclose(exported_reference[:,:3], reference[:,:3], atol=1e-6, rtol=0)
            or np.max(orientation_errors(exported_reference[:,3:], reference[:,[6,3,4,5]])) > 1e-4):
        raise ValueError('Export reference differs from hashed task reference')
    baseline = None
    if mocap:
        if stage['episode'] != 39:
            raise ValueError('Episode 38 has no label scoring operation')
        _, poses, times = h.load_mocap()
        if len(times) != count:
            raise ValueError('Mocap clock differs from final task')
        reference = np.c_[poses[:,:3], poses[:,[4,5,6,3]]]
        baseline = load_baseline()
    result = metrics(achieved, reference)
    result.update(checkpoint_steps=FINAL, checkpoint_sha256=h.digest(checkpoint),
                  task_reference_sha256=h.digest(output/'reference-task.npz'), label_access=mocap,
                  reference_end_fraction=official['reference_tracking']['reference_end_fraction'],
                  completed_without_reset=official['reference_tracking']['reference_end_fraction'] == 1.,
                  source='unedited evaluator achieved poses', kit_geometry_used=False)
    if baseline is not None:
        from .geometry import amended_verdict
        result['kit'] = baseline.score(baseline.load_kit(), achieved[:,None,None].copy(), reference[:,None,None], baseline.vertices('mini_dust_pan')[None])
        result['kit_geometry_used'] = 'Public scan points, scoring only; never the simulated profile'
        verdict = amended_verdict(result['T'],result['Q'],result['E_m'],result['Theta_deg'],result['E_S_m'],result['Theta_S_deg'],result['kit']['AUC'])
        result['verdict'] = 'GO' if verdict == 'GO' else 'NULL'
        result['adoption_complete'] = False
        result['pending'] = 'Self-reference bounds, no-reset end, M1 admission and frozen-camera achieved-pose reprojection gates remain required'
    h.write(directory/name, result)
    return result


def execute(operation, episode, output):
    """Enforce independent runtime flags, immutable Stage 0 and single final label access."""
    select(episode, output)
    flags = json.loads((h.FEATURE/'runtime.json').read_text())
    flag = 'video_' + operation.replace('-', '_') + '_authorized'
    if not flags.get(flag, False):
        raise PermissionError(flag + ' is false; the lead sets it')
    stage = json.loads((output/'stage0.json').read_text())
    if stage['status'] != 'PASS':
        raise PermissionError('Stage 0 failed; this episode stopped')
    h.verify_scientific_inputs(stage)
    if operation == 'train':
        # Source admission can change only by a new build, never by a runtime switch.
        if not stage['reference']['m1_gates_passed']:
            raise PermissionError('M1 reference admission failed at snapshot')
        source = h.ROOT/stage['reference']['source_path']
        if h.digest(source) != stage['reference']['source_sha256']:
            raise ValueError('M1 reference changed after Stage 0; rebuild from the final reference')
    if operation == 'score-mocap':
        if episode != 39:
            raise ValueError('No label score exists for episode 38')
        if not (output/f'score-{FINAL}/self_metrics.json').exists():
            raise PermissionError('Final self score must precede the one sealed label score')
    record_path = output/(operation+'.run.json')
    with record_path.open('x') as stream:
        json.dump(dict(status='running', operation=operation, pid=os.getpid(), started_unix=time.time(),
                       stage0_sha256=h.digest(output/'stage0.json')), stream, indent=2)
    record = json.loads(record_path.read_text())
    prepared = json.loads((output/'commands.json').read_text())
    try:
        if operation == 'score-mocap':
            result = score_export(output, mocap=True)
        elif operation == 'kill-check':
            result = kill_check(output, episode)
        else:
            command = prepared['train' if operation == 'train' else 'evaluator_final']
            from src.gpu import GpuLease
            with GpuLease(h.ROOT, 1024, 6000, stage='moving-video-'+operation) as lease:
                env = {**h.environment(cpu=False), 'DYNAMIC_FEASIBILITY_DEADLINE':str(time.time()+4500)}
                (output/f'score-{FINAL}').mkdir(exist_ok=True)
                with (output/(operation+'.log')).open('w') as log:
                    subprocess.run(command, cwd=h.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                record['gpu'] = lease.report()
            result = score_export(output) if operation == 'score-self' else {'training_completed': True}
        record.update(status='completed', result=result)
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_unix'] = time.time()
        h.write(record_path, record)


def kill_check(output, episode):
    """Replay the fixed boundary policy on CPU and report self-T without labels."""
    diagnostic = h.ROOT/f'data/outputs/visuals/policy-hands/moving-video-ep{episode}-boundary'
    command = [str(h.PYTHON), '-B', str(h.ROOT/'scripts/report/policy_hands.py'),
               '--checkpoint', str(output/'training'/f'policy_{BOUNDARY}.safetensors'),
               '--task', str(h.TASK), '--output', str(diagnostic), '--no-official']
    with (output/'kill-check.log').open('w') as log:
        subprocess.run(command, cwd=h.ROOT, env=h.environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(diagnostic/'replay.npz') as replay, np.load(output/'reference-task.npz') as reference:
        body = int(replay['object_body_ids'][0])
        achieved = replay['body_pose_xyzw'][:,body]
        expected = np.c_[reference['position'], reference['quaternion_xyzw']]
        value = metrics(achieved, expected)
        value.update(kill=value['T'] < .5, checkpoint_steps=BOUNDARY,
                     task_reference_sha256=h.digest(output/'reference-task.npz'),
                     checkpoint_sha256=h.digest(output/'training'/f'policy_{BOUNDARY}.safetensors'),
                     diagnostic=str(diagnostic.relative_to(h.ROOT)), device='cpu', label_access=False,
                     action='Lead stops training and records NULL when kill is true; this command never stops another process')
    h.write(output/'kill-check.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('stage0','train','score-self','score-mocap','kill-check'))
    parser.add_argument('--episode', required=True, type=int, choices=(38,39))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--body-convention', type=Path, default=h.ROOT/'data/outputs/moving_reference/ep39/candidate-c/body-convention.json')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    select(args.episode,args.output)
    if args.operation == 'score-mocap' and args.episode != 39:
        parser.error('Episode 38 has no label score')
    if args.operation == 'stage0':
        if args.reference is None:
            parser.error('stage0 requires --reference')
        stage0(args)
    elif args.run:
        execute(args.operation,args.episode,args.output.resolve())
    else:
        prepared = json.loads((args.output/'commands.json').read_text())
        command = prepared[args.operation.replace('-','_')]
        print(shlex.join(command))


if __name__ == '__main__':
    main()
