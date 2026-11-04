"""Rebuild an FP static placement with unchanged assets and the v5 policy recipe."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from src.gpu import GpuLease
from src.paths import project_path
from src.provenance import sha256_file, write_json
from .combined_frames import ROOT, D2, KIT, task_for
from . import handler, pack_moving as moving
from .request import Request
from ._frame0_fp_frames import matrix, relative_errors

BASE = ROOT / 'data/outputs/pose_insurance/frame0-fp/recalibrated'
REGISTRATIONS = ROOT / 'data/outputs/pose_insurance/frame0-fp/corrected'
PARENT = ROOT / 'submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json'


def checkpoint_data(directory):
    code = """import json,sys
from pathlib import Path
from safetensors import safe_open
from src.features.pose_insurance.scored_scale_handler import optimizer_state
p=Path(sys.argv[1])
with safe_open(str(p/'training/policy_512.safetensors'),framework='numpy') as f:
 config=json.loads(f.metadata()['user.resolved_config_json'])
print(json.dumps(dict(optimizer=optimizer_state(p),config=config)))
"""
    result = subprocess.check_output([str(ROOT/'data/.runtime/flashchord/venv/bin/python'), '-B', '-c', code, str(directory)],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1'), text=True)
    return json.loads(result)


def run(episode, placement):
    directory = BASE / f'ep{episode}/policy-resume'
    directory.mkdir(parents=True, exist_ok=True)
    static = json.loads(D2.read_text())
    export = next(e for e in static['exports'] if Path(e['path']).stem == f'episode_{episode:06d}')
    original_path, original = task_for(export)
    table = pq.ParquetFile(original_path).read()
    task = deepcopy(original)
    targets = np.asarray(placement['target_xyzw'])
    count = len(task['object_body_position'])
    if task['object_body_names'] != placement['names']:
        raise ValueError('Official object slot order differs from FP placement')
    task['object_body_position'] = np.repeat(targets[None, :, :3], count, axis=0).tolist()
    task['object_body_wxyz'] = np.repeat(targets[None, :, [6, 3, 4, 5]], count, axis=0).tolist()
    assets = []
    for field in ('object_mesh_paths', 'object_urdf_paths'):
        task[field] = [str(project_path(ROOT, name)) for name in original[field]]
        assets.extend(moving.entry(name) for name in task[field])
    allowed = {'object_body_position', 'object_body_wxyz', 'object_mesh_paths', 'object_urdf_paths'}
    if any(task[k] != original[k] for k in task if k not in allowed):
        raise ValueError('FP scene changed a field beyond poses and path rebasing')
    task_file = directory / 'task.parquet'
    pq.write_table(pa.Table.from_pylist([task], schema=table.schema), task_file)
    budget = dict(environment_steps=512, worlds=1, replay_minimum=32)
    request = Request(episode_index=episode, budget=budget)
    commands = dict(train=handler.training_command(request, task_file, directory))
    commands['train'].append('reset.seed=0')
    checkpoint = directory / 'training/policy_512.safetensors'
    trajectory = directory / 'trajectory.parquet'
    commands['evaluate'] = [str(ROOT / 'data/.runtime/flashchord/venv/bin/python'), '-B',
        str(KIT / 'flash_chord/scripts/evaluate_policy.py'), f'evaluation.checkpoint={checkpoint}',
        'evaluation.world_count=1', 'evaluation.reset_mode=explicit', 'evaluation.start_frame=0',
        f'evaluation.object_trajectories_output={trajectory}',
        f'evaluation.metrics_output={directory / "official_metrics.json"}',
        f'hydra.run.dir={directory / "evaluate_hydra"}', 'logging.mode=disabled']
    write_json(directory / 'commands.json', commands)
    write_json(directory / 'build.json', dict(episode=episode, task=moving.entry(task_file),
        parent_task=moving.entry(original_path), assets=assets, targets_xyzw=targets.tolist(),
        placement=moving.entry(REGISTRATIONS / f'ep{episode}/corrected-placement.json'),
        budget=budget, seed=0, assets_unchanged=True, labels_read=False))
    from src.features.moving_policy import handler as video_handler
    video_handler.OUTPUT = directory
    env = video_handler.environment(cpu=False)
    temporary = ROOT / 'data/.runtime/tmp/fp'
    temporary.mkdir(exist_ok=True)
    env.update(TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary))
    record = dict(episode=episode, status='running', pid=os.getpid(), commands=commands,
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'run.json', record)
    with GpuLease(ROOT, 2048, 1024, stage='pose_insurance.frame0_fp_static') as lease:
        for operation in ('train', 'evaluate'):
            if operation == 'train' and checkpoint.exists():
                record['optimizer'] = checkpoint_data(directory)['optimizer']
                write_json(directory / 'train-completed.json', dict(utc=datetime.now(timezone.utc).isoformat(), argv=commands['train'], exit_code=0, resumed_from_completed_checkpoint=True))
                continue
            with (directory / f'{operation}.log').open('x') as stream:
                subprocess.run(commands[operation], cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            if operation == 'train':
                record['optimizer'] = checkpoint_data(directory)['optimizer']
                if record['optimizer']['optimizer_updates'] < 1:
                    raise ValueError('Static policy did not learn')
            write_json(directory / f'{operation}-completed.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                argv=commands[operation], exit_code=0))
        record['gpu'] = lease.report()
    roster, objects = moving.official_roster()
    _, code = moving.evaluator_inputs()
    frame, poses = moving.inspect_export(trajectory, sha256_file(checkpoint), episode, roster[episode],
                                        objects[episode].object_names, code)
    parent = json.loads(PARENT.read_text())
    old_export = next(e for e in parent['exports'] if e['episode'] == episode)
    old = pq.read_table(moving.checked(old_export)).to_pandas().sort_values(['reference_frame', 'object_body_id'])
    columns = ['achieved_position_' + a for a in 'xyz'] + ['achieved_quaternion_' + a for a in 'xyzw']
    previous = old[columns].to_numpy(float).reshape(-1, 2, 7)
    errors = np.linalg.norm(poses[0, :, :3] - targets[:, :3], axis=1)
    turns = np.degrees((Rotation.from_quat(targets[:, 3:]).inv() *
                        Rotation.from_quat(poses[0, :, 3:])).magnitude())
    drift = np.linalg.norm(poses[:, :, :3] - poses[:1, :, :3], axis=-1).max(axis=0)
    old_drift = np.linalg.norm(previous[:, :, :3] - previous[:1, :, :3], axis=-1).max(axis=0)
    relative = relative_errors(poses[0], targets)
    from src.features.moving_reference import kit_score
    from src.features.moving_policy._two_object_rotation import orientation_degrees
    axis = kit_score.reference_geometry(placement['names'][1])['axis']
    axis = np.asarray(placement['conversions'][1]['candidate_T_registered'])[:3, :3] @ axis
    if np.any(axis):
        measured = np.linalg.inv(matrix(targets[0])) @ matrix(targets[1])
        actual = np.linalg.inv(matrix(poses[0, 0])) @ matrix(poses[0, 1])
        relative['raw_rotation_deg'] = relative['rotation_deg']
        relative['rotation_deg'] = float(orientation_degrees(
            Rotation.from_matrix(actual[:3, :3]).as_quat(), Rotation.from_matrix(measured[:3, :3]).as_quat(),
            placement['names'][1], axis))
        relative['symmetry_axis_candidate'] = axis.tolist()
    physical = dict(first_translation_m=errors.tolist(), first_rotation_deg=turns.tolist(),
        maximum_displacement_m=drift.tolist(), v5_displacement_m=old_drift.tolist(),
        displacement_limit_m=(old_drift + .01).tolist(), relative_pose_error=relative,
        passed=bool(np.all(errors <= .01) and np.all(turns <= 5.) and np.all(drift <= old_drift + .01)
                    and relative['offset_cm'] <= 2. and relative['rotation_deg'] <= 10.))
    cutoff = datetime.fromisoformat('2026-11-04T21:25:00+00:00')
    completed = datetime.now(timezone.utc)
    physical['before_cutoff'] = completed < cutoff
    physical['passed'] &= physical['before_cutoff']
    record.update(status='completed', finished_utc=completed.isoformat(), physical=physical,
                  checkpoint=moving.entry(checkpoint), trajectory=moving.entry(trajectory))
    write_json(directory / 'run.json', record)
    result = dict(kind='pose_insurance.frame0_fp_admission', episode=episode, passed=physical['passed'],
        policy_directory=str(directory.relative_to(ROOT)), task=moving.entry(task_file),
        checkpoint=moving.entry(checkpoint), trajectory=moving.entry(trajectory),
        metrics=moving.entry(directory / 'official_metrics.json'), run=moving.entry(directory / 'run.json'),
        commands=moving.entry(directory / 'commands.json'), build=moving.entry(directory / 'build.json'),
        placement=moving.entry(REGISTRATIONS / f'ep{episode}/corrected-placement.json'),
        registration=moving.entry(BASE / 'decisions.json'), physical=physical,
        optimizer=record['optimizer'], assets=assets, labels_read=False)
    write_json(directory / 'admission.json', result)
    return result


def inspect_admission(path):
    """Recheck sealed static-policy evidence before consuming a rollout."""
    item = moving.read(path)
    if item.get('kind') != 'pose_insurance.frame0_fp_admission' or item.get('passed') is not True:
        raise ValueError('A completed physical admission is required')
    files = ('task', 'checkpoint', 'trajectory', 'metrics', 'run', 'commands', 'build', 'placement', 'registration')
    checked = {key: moving.checked(item[key]) for key in files}
    for asset in item['assets']:
        moving.checked(asset)
    build = moving.read(checked['build'])
    task = pq.read_table(checked['task']).to_pylist()[0]
    targets = np.asarray(build['targets_xyzw'])
    reference = np.concatenate((np.asarray(task['object_body_position']),
                                np.asarray(task['object_body_wxyz'])[..., [1, 2, 3, 0]]), axis=-1)
    if not np.allclose(reference, targets[None], atol=1e-12, rtol=0):
        raise ValueError('Static task differs from its sealed targets')
    episode = item['episode']
    roster, objects = moving.official_roster()
    _, code = moving.evaluator_inputs()
    frame, poses = moving.inspect_export(checked['trajectory'], item['checkpoint']['sha256'], episode,
                                        roster[episode], objects[episode].object_names, code)
    columns = ['reference_' + key for key in moving.POSE]
    exported_reference = frame[columns].to_numpy(float).reshape(-1, 2, 7)
    if not np.allclose(exported_reference, targets[None], atol=2e-6, rtol=0):
        raise ValueError('Evaluator reference differs from the rebuilt task')
    config = checkpoint_data(checked['run'].parent)['config']
    if (config['reset']['seed'] != 0 or config['scene']['world_count'] != 1
            or config['training']['total_environment_steps'] != 512
            or config['training']['updates_per_collection'] != 1.
            or config['training']['replay']['minimum_size'] != 32
            or config['voc_scale_override'] != 0.):
        raise ValueError('Checkpoint resolved static recipe differs')
    directory = checked['run'].parent
    command_record = moving.read(directory / 'evaluate-completed.json')
    overrides = moving.overrides(command_record['argv'])
    if (command_record['exit_code'] != 0
            or moving.path(overrides['evaluation.checkpoint']) != checked['checkpoint']
            or moving.path(overrides['evaluation.object_trajectories_output']) != checked['trajectory']):
        raise ValueError('Completed evaluator command does not bind this rollout')
    moving.require_completion(moving.read(checked['metrics']), item['checkpoint']['sha256'], len(poses))
    if checkpoint_data(directory)['optimizer']['optimizer_updates'] < 1:
        raise ValueError('Policy has no optimizer updates')
    physical = item['physical']
    if (not physical['passed'] or max(physical['first_translation_m']) > .01
            or max(physical['first_rotation_deg']) > 5
            or np.any(np.asarray(physical['maximum_displacement_m']) > physical['displacement_limit_m'])
            or physical['relative_pose_error']['offset_cm'] > 2
            or physical['relative_pose_error']['rotation_deg'] > 10):
        raise ValueError('Physical guards do not pass')
    return item
