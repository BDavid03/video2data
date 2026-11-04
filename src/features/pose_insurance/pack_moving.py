"""Compose gated moving rollouts with frozen D2 v5 exports, entirely on CPU."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
from src.paths import current_name
from src.provenance import sha256_file
from ._moving import tracking_metrics, translation_errors, pose_matrices, reprojection_summary, aligned_tracking_metrics
from src.features.moving_reference._final_gates import FINAL_LIMITS as M1_LIMITS
from src.features.moving_reference.final_gates import verify_final

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / 'upstream/v2d_submission_kit/v2d_submission_kit'
V5_MANIFEST = 'submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json'
V5_HASH = '0b2606aed015fe0a64e31037ff790b5483680c3da7e7a5ae2d503edeb8261035'
V5_MANIFEST_HASH = '0fcbfd7d48dc137fc5598adbcb8d609c5de3dd717a87ceb886b9b885e7933f7d'
SCRATCH = ROOT / 'data/.runtime/tmp/mpack-dry'
OUTPUT = ROOT / 'data/outputs/pose_insurance/moving-pack'
POSE = [f'{kind}_{axis}' for kind, axes in (('position', 'xyz'), ('quaternion', 'xyzw')) for axis in axes]



def path(value: str | Path) -> Path:
    """Resolve a checkout-owned input without accepting excluded episode paths."""
    name = str(value)
    if name.startswith('/mnt/') or '..' in Path(name).parts:
        raise ValueError('External or parent-relative input path')
    if Path(name).is_absolute():
        name = str(Path(name).relative_to(ROOT))
    result = (ROOT / current_name(name)).resolve()
    result.relative_to(ROOT)
    if any(int(ep) in {2, 13, 22, 24, 27, 41, 43, 44}
           for ep in re.findall(r'(?:episode[_=-]?|ep)(\d+)(?=\D|$)', str(result))):
        raise ValueError('Excluded episode input path')
    return result


def entry(value: str | Path) -> dict:
    """Record a local input's exact current content identity."""
    p = path(value)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha256_file(p))


def read(value: str | Path) -> dict:
    return json.loads(path(value).read_text())


def checked(value: dict) -> Path:
    p = path(value['path'])
    if sha256_file(p) != value['sha256']:
        raise ValueError(f'Input hash mismatch: {p}')
    return p


def write(value: Path, data: dict) -> None:
    value.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Missing or nonfinite measurement: {label}')
    return float(value)


def require_reprojection(data: dict, label: str) -> None:
    for key, minimum in (('median_iou', .60), ('p10_iou', .30), ('usable_fraction', .90)):
        value = number(data.get(key), f'{label}.{key}')
        if not minimum <= value <= 1.:
            raise ValueError(f'{label} reprojection fails {key}')


def require_tracking(data: dict, label: str) -> None:
    t, q, e, es, theta, ts = [number(data.get(k), f'{label}.{k}')
                             for k in ('T', 'Q', 'E_m', 'E_S_m', 'Theta_deg', 'Theta_S_deg')]
    if t < .5 or q < .5 or min(e, es, theta, ts) < 0 or e > min(.5 * es, .03) or theta > min(.5 * ts, 15.):
        raise ValueError(f'{label} fails T/Q/E/Theta adoption bounds')


def require_m1(data: dict, episode: int) -> None:
    label_free = data.get('label_free', data.get('frozen_rule', {}).get('label_free', data.get('finalization', {}).get('label_free')))
    if data.get('episode') != episode or label_free is not True or data.get('passed') is not True:
        raise ValueError('Clause 2 requires this episode\'s passed label-free M1 evidence')
    for key, (direction, limit) in M1_LIMITS.items():
        row = data.get('gates', {}).get(key, {})
        value = number(row.get('value'), f'M1.{key}')
        fraction_invalid = ('fraction' in key or 'iou' in key) and value > 1.
        if value < 0 or fraction_invalid or (value > limit if direction == 'le' else value < limit) or row.get('passed') is not True:
            raise ValueError(f'Clause 2 M1 fails {key}')
    world = data['gates']['camera_heldout_world_residual_median_m']
    if world.get('instrument') == 'perpendicular under sealed-pass clause 2':
        if not data.get('finalization', {}).get('sealed_condition_verified') or world['value'] != world.get('perpendicular_median_m'):
            raise ValueError('Clause 2 conditional world residual has no verified sealed-pass condition')
    elif world.get('instrument') == 'perpendicular':
        control = world.get('control', {})
        if (number(control.get('pairs'), 'noise floor pairs') < 100
                or number(control.get('frames'), 'noise floor frames') < 3
                or number(control.get('norm_median_m'), 'noise floor norm') < number(world.get('old_3d_median_m'), 'old 3D residual')
                or number(control.get('along_median_m'), 'along-ray floor') < 3 * number(control.get('perpendicular_median_m'), 'perpendicular floor')
                or world.get('noise_floor_explains_old') is not True
                or world['value'] != world.get('perpendicular_median_m')):
            raise ValueError('Clause 2 changed world-residual instrument has no explaining noise-floor control')
    elif world.get('old_3d_median_m') is not None and world['value'] != world['old_3d_median_m']:
        raise ValueError('Clause 2 must retain unexplained original 3D world residual')


def inspect_m1(binding: dict, source: dict, episode: int, assets: dict) -> None:
    """Bind amended M1 gates to the independent source reference in the admission."""
    gates = checked(binding['gates'])
    if binding['reference'] != source:
        raise ValueError('Clause 2 M1 evidence does not bind the adopted source reference')
    checked(binding['reference'])
    if gates.name != 'gates-final.json' or gates.parent != path(source['path']).parent:
        raise ValueError('Clause 2 requires candidate-c/gates-final.json beside the source reference')
    require_same_asset(binding.get('asset_hashes'), assets)
    verify_final(gates, episode)
    require_m1(read(gates), episode)


def require_m2v(data: dict) -> None:
    """Require the sealed M2v GO, including achieved-pose video reprojection."""
    if data.get('passed') is not True or data.get('verdict') != 'GO' or data.get('episode') != 39:
        raise ValueError('Clause 1 requires passed M2v GO evidence on 39')
    require_tracking(data['tracking'], 'M2v')
    if not .45 <= number(data.get('AUC'), 'M2v.AUC') <= 1.:
        raise ValueError('Clause 1 M2v AUC outside [.45, 1]')
    require_reprojection(data['reprojection'], 'M2v')


def require_completion(official: dict, checkpoint_hash: str, count: int) -> None:
    """Require a full explicit-reset evaluator attempt with no termination."""
    tracking = official.get('reference_tracking', {})
    failures = tracking.get('failure_cause_fraction', {})
    if (official.get('checkpoint_sha256') != checkpoint_hash or official.get('reset_mode') != 'explicit'
            or official.get('non_finite_worlds') != [] or official.get('start_frame') != 0
            or official.get('world_count') != 1 or official.get('step_count') != count
            or tracking.get('reference_end_fraction') != 1. or not {'object', 'wrist'} <= failures.keys()
            or any(number(v, 'failure fraction') != 0 for v in failures.values())
            or number(official.get('environment_steps'), 'environment_steps') < 1):
        raise ValueError('Clause 3 requires the end reached without a reset or termination')


def require_first(row: dict) -> None:
    if row['first_position_m'] > .01 or row['first_orientation_deg'] > 2.:
        raise ValueError('Clause 4 first pose exceeds 1 cm / 2 degrees')


def official_roster() -> tuple[dict, dict]:
    """Load only the public scored row roster and official object slot order."""
    sys.path.insert(0, str(KIT))
    from tools.pack_submission import public_track3_grid
    config = read(KIT / 'config/tracks.json')['tracks']['track_3']
    episodes = config['scored_episodes']
    if set(episodes) & {2, 13, 22, 24, 27, 39, 41, 42, 43, 44}:
        raise ValueError('Unexpected scored episode roster')
    return public_track3_grid(KIT / 'data/track_3_sample_submission.csv',
                             KIT / 'data/track_3_evaluation_objects.json', episodes)


def frozen_v5() -> dict:
    """Verify the frozen v5 pack and all twenty export byte hashes."""
    if sha256_file(path(V5_MANIFEST)) != V5_MANIFEST_HASH:
        raise ValueError('Frozen D2 v5 manifest identity mismatch')
    manifest = read(V5_MANIFEST)
    if manifest['version'] != 'v5' or manifest['sha256'] != V5_HASH or sha256_file(path(manifest['file'])) != V5_HASH:
        raise ValueError('Frozen D2 v5 CSV identity mismatch')
    roster, _ = official_roster()
    exports = manifest['exports']
    if len(exports) != len(roster) or {int(Path(e['path']).stem.split('_')[1]) for e in exports} != set(roster):
        raise ValueError('Frozen v5 exports do not cover the scored roster exactly')
    for export in exports:
        checked(export)
    return manifest


def evaluator_inputs() -> tuple[dict, str]:
    """Require the kit's unchanged script and pinned upstream evaluation modules."""
    kit_manifest = read(KIT / 'KIT_MANIFEST.json')
    for name in ('flash_chord/scripts/evaluate_policy.py', 'tools/pack_submission.py',
                 'tools/preflight.py', 'submit/v2d_submit.py'):
        if sha256_file(KIT / name) != kit_manifest['files'][name]:
            raise ValueError(f'Unedited kit required: {name}')
    source = ROOT / 'upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/evaluation'
    runtime = ROOT / 'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord/src/flash_chord/evaluation'
    digest = hashlib.sha256()
    inputs = {}
    for name in ('__init__.py', 'harness.py', 'metrics.py', 'rollout.py'):
        p = source / name
        inputs[name] = entry(p)
        # The Linux install normalizes upstream CRLF to LF; no code edits are accepted.
        installed = runtime / name
        if installed.read_bytes() != p.read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError(f'Installed evaluator differs from pinned upstream: {name}')
        inputs['runtime_' + name] = entry(installed)
        digest.update(name.encode() + b'\0' + installed.read_bytes())
    return inputs, digest.hexdigest()


def inspect_export(trajectory: Path, checkpoint_hash: str, episode: int, grid, names: list,
                   eval_code: str) -> tuple[object, np.ndarray]:
    """Admit the evaluator's full schema, provenance, clock and official key grid."""
    table = pq.read_table(trajectory)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if any('fake' in k.lower() or 'plumbing' in k.lower() for k in meta):
        raise ValueError('FAKE/plumbing trajectories are never admissible')
    expected = {'schema': 'flash_chord_object_trajectories_v1', 'position_unit': 'meter',
                'quaternion_order': 'xyzw', 'episode_index': str(episode),
                'row_order': 'step,environment_id,object_body_id', 'pose_width': '7',
                'checkpoint_sha256': checkpoint_hash,
                'eval_script_sha256': sha256_file(KIT / 'flash_chord/scripts/evaluate_policy.py'),
                'eval_code_sha256': eval_code, 'start_frame': '0', 'world_count': '1',
                'object_body_count': str(len(names)), 'step_count': str(len(grid.frame_index.unique()))}
    for key, value in expected.items():
        if meta.get('flash_chord.' + key) != value:
            raise ValueError(f'Evaluator metadata mismatch: {key}')
    required = ['environment_id', 'step', 'reference_frame', 'time_s', 'object_id',
                'object_body_id', 'object_body_name', 'environment_valid',
                *['achieved_' + c for c in POSE], *['reference_' + c for c in POSE]]
    if any(c not in table.column_names or table[c].null_count for c in required):
        raise ValueError('Missing or null evaluator export columns')
    frame = table.to_pandas().sort_values(['reference_frame', 'environment_id', 'object_body_id'])
    keys = list(zip(frame.environment_id, frame.reference_frame, frame.object_body_id))
    wanted = list(grid[['world_index', 'frame_index', 'object_slot']].itertuples(index=False, name=None))
    if len(set(keys)) != len(keys) or sorted(keys) != sorted(wanted):
        raise ValueError('Duplicate, missing or extra official roster keys')
    if not frame.environment_valid.all():
        raise ValueError('Invalid exported environments')
    if not np.array_equal(frame.object_id, frame.object_body_id):
        raise ValueError('Object grouping differs from official rigid-object slots')
    if any(not isinstance(slot, (int, np.integer)) or slot < 0 or slot >= len(names) or name != names[slot]
           for slot, name in zip(frame.object_body_id, frame.object_body_name)):
        raise ValueError('Official object slot order mismatch')
    fps = number(float(meta['flash_chord.control_fps']), 'control_fps')
    if fps <= 0 or not np.array_equal(frame.step, frame.reference_frame) or not np.allclose(frame.time_s, frame.step / fps, atol=1e-7, rtol=0):
        raise ValueError('Export clock/reset sequence mismatch')
    poses = frame[['achieved_' + c for c in POSE]].to_numpy(float).reshape(-1, len(names), 7)
    require_poses(poses)
    return frame, poses


def require_poses(poses: np.ndarray) -> None:
    if not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1., atol=1e-4, rtol=0):
        raise ValueError('Nonfinite poses or nonunit quaternions')


def reference_poses(value: Path, frame, bodies: int) -> np.ndarray:
    """Load the M1 npz pose/clock contract, without reading any labelled dataset."""
    with np.load(value, allow_pickle=False) as data:
        if 'plumbing_only' in data or 'fake' in data:
            raise ValueError('FAKE/plumbing references are never admissible')
        poses = data['pose'].astype(float)
        if poses.ndim == 2:
            poses = poses[:, None, :]
        frames = data['frame_index']
        times = data['time']
    expected_frames = frame.reference_frame.to_numpy().reshape(-1, bodies)[:, 0]
    expected_times = frame.time_s.to_numpy().reshape(-1, bodies)[:, 0]
    if (poses.shape != (len(expected_frames), bodies, 7) or not np.array_equal(frames, expected_frames)
            or times.shape != expected_times.shape or not np.allclose(times - times[0], expected_times, atol=1e-6, rtol=0)):
        raise ValueError('Reference shape/frame/clock differs from export')
    require_poses(poses)
    recorded = frame[['reference_' + c for c in POSE]].to_numpy(float).reshape(poses.shape)
    require_poses(recorded)
    a = poses[..., 3:] / np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
    b = recorded[..., 3:] / np.linalg.norm(recorded[..., 3:], axis=-1, keepdims=True)
    b *= np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -1., 1.)
    if not np.allclose(poses[..., :3], recorded[..., :3], atol=2e-6, rtol=0) or not np.allclose(a, b, atol=2e-6, rtol=0):
        raise ValueError('Independent reference differs from evaluator reference')
    return poses


FINAL_STEPS = 66763776


def overrides(argv: list[str]) -> dict:
    """Read the recorded Hydra overrides, retaining their actual path values."""
    return {key: value.strip('"\'') for arg in argv if '=' in arg for key, value in [arg.split('=', 1)]}


def checkpoint_metadata(checkpoint: Path) -> dict:
    """Read existing safetensors metadata with the installed CPU-capable producer environment."""
    code = '''import json,sys
from safetensors import safe_open
with safe_open(sys.argv[1],framework="numpy") as f:
 m=f.metadata(); c=json.loads(m["user.resolved_config_json"])
 print(json.dumps(dict(steps=int(m["user.environment_steps"]),updates=int(m["user.global_update_step"]),
 algorithm=m["user.algorithm"],kind=m["user.checkpoint_kind"],task=c["task"]["parquet"],
 total=c["training"]["total_environment_steps"])))
'''
    result = subprocess.check_output([str(ROOT / 'data/.runtime/flashchord/venv/bin/python'), '-B', '-c', code, str(checkpoint)],
                                     cwd=ROOT, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    return json.loads(result)


def inspect_learning(directory: Path, checkpoint: Path, task: Path) -> dict:
    """Admit the real completed training record, final log count and checkpoint optimizer metadata."""
    train = read(directory / 'train.run.json')
    stage = entry(directory / 'stage0.json')
    prepared = read(directory / 'commands.json')
    argv = train.get('argv', prepared.get('train', []))
    args = overrides(argv)
    if (train.get('status') != 'completed' or train.get('stage0_sha256') != stage['sha256']
            or ('exit_code' in train and train['exit_code'] != 0)
            or train.get('result', {}).get('training_completed') is False
            or number(train.get('finished_unix'), 'training finish') <= number(train.get('started_unix'), 'training start')
            or not any(str(arg).endswith('/scripts/train_flash_sac.py') for arg in argv)
            or int(args.get('training.total_environment_steps', 0)) != FINAL_STEPS
            or path(args['task.parquet']) != task or path(args['output_dir']) != checkpoint.parent):
        raise ValueError('A completed learned training process with the frozen recipe is required')
    log = path(directory / 'train.log')
    counts = [int(n.replace(',', '')) for n in re.findall(r'completed ([\d,]+) environment steps', log.read_text())]
    if not counts or max(counts) < FINAL_STEPS:
        raise ValueError('Training log does not prove completion of the recipe total')
    meta = checkpoint_metadata(checkpoint)
    if (meta['steps'] < FINAL_STEPS or meta['total'] != FINAL_STEPS or meta['updates'] < 1
            or meta['algorithm'] != 'flash_sac' or meta['kind'] != 'policy' or path(meta['task']) != task
            or checkpoint.name != f'policy_{FINAL_STEPS}.safetensors'):
        raise ValueError('Checkpoint has no actual learned updates at the recipe total or binds another task')
    return dict(train_run=entry(directory / 'train.run.json'), train_log=entry(log),
                checkpoint=entry(checkpoint), commands=entry(directory / 'commands.json'),
                stage0=stage, logged_steps=max(counts), checkpoint_metadata=meta)


def load_reference(value: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(value, allow_pickle=False) as data:
        if 'plumbing_only' in data or 'fake' in data:
            raise ValueError('FAKE/plumbing references are never admissible')
        poses = data['pose'].astype(float)
        if poses.ndim == 2:
            poses = poses[:, None, :]
        frames, times = data['frame_index'].copy(), data['time'].astype(float)
    require_poses(poses)
    return poses, frames, times


def inspect_task(directory: Path, episode: int) -> dict:
    """Bind M1, the moved reference, the actual task and its shared simulator asset."""
    stage = read(directory / 'stage0.json')
    required = {'1_schema', '2_closure', '3_penetration', '4a_finger_limits', '4b_reachability', '5_support', '6_truth'}
    if (stage.get('status') != 'PASS' or stage.get('episode') != episode or not required <= stage.get('checks', {}).keys()
            or not all(c.get('passed') is True for c in stage['checks'].values())):
        raise ValueError('Video Stage 0 did not pass every check for this episode')
    for e in stage['artifacts']:
        checked(e)
    task = path(stage['build']['task'])
    if entry(task) not in stage['artifacts']:
        raise ValueError('Actual task is not hash-bound by Stage 0')
    row = pq.read_table(task).to_pylist()[0]
    moved_entry = entry(directory / 'reference-task.npz')
    source_entry = entry(directory / 'reference-source.npz')
    ref_record = read(directory / 'reference.json')
    original = entry(ref_record['source_path'])
    if (ref_record['source_sha256'] != source_entry['sha256'] or source_entry['sha256'] != original['sha256']
            or ref_record['task_sha256'] != moved_entry['sha256'] or stage['reference'] != ref_record):
        raise ValueError('M1 source/reference-task identity differs from the recorded translation')
    if path(original['path']).parent.name != 'candidate-c':
        raise ValueError('Reference source is not final M1 candidate-c evidence')
    source, sf, st = load_reference(path(source_entry['path']))
    moved, mf, mt = load_reference(path(moved_entry['path']))
    if source.shape != moved.shape or not np.array_equal(sf, mf) or not np.array_equal(st, mt):
        raise ValueError('Source and moved reference clocks/grids differ')
    shift = np.asarray(ref_record['translation_m'], float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('Recorded pure translation is invalid')
    errors = translation_errors(source, moved, shift)
    if errors['maximum_translation_error_m'] > 2e-6 or errors['maximum_orientation_error_deg'] > 1e-4:
        raise ValueError('Task reference does not equal M1 reference after the recorded pure translation')
    task_poses = np.concatenate((np.asarray(row['object_body_position']),
                                np.asarray(row['object_body_wxyz'])[..., [1, 2, 3, 0]]), axis=-1)
    if task_poses.shape != moved.shape:
        raise ValueError('Task object pose shape differs from moved reference')
    task_errors = translation_errors(moved, task_poses, np.zeros(3))
    if task_errors['maximum_translation_error_m'] > 2e-6 or task_errors['maximum_orientation_error_deg'] > 1e-4:
        raise ValueError('Task first pose or trajectory differs from the translated M1 reference')
    clock = np.arange(len(mt)) / float(row['fps'])
    residual = float(np.max(np.abs(mt - mt[0] - clock)))
    if residual > 1e-6 or abs(residual - ref_record['largest_clock_residual_s']) > 1e-9:
        raise ValueError('Reference-task clock differs from the recorded uniform task clock')
    if row['object_body_names'] != ['mini_dust_pan']:
        raise ValueError('Video task body convention/order mismatch')
    meshes, urdfs = row['object_mesh_paths'], row['object_urdf_paths']
    if len(meshes) != 1 or len(urdfs) != 1 or path(meshes[0]).parent != path(urdfs[0]).parent:
        raise ValueError('Task must bind the single shared profile asset')
    shared = path(meshes[0]).parent
    asset = read(directory / 'asset.json')
    if asset != read(shared / 'asset.json') or asset['hashes'] != stage['asset_hashes']:
        raise ValueError('Task and Stage 0 simulator asset manifests differ')
    for name, digest in asset['hashes'].items():
        if sha256_file(shared / name) != digest:
            raise ValueError('Shared simulator asset content hash mismatch')
        if (directory / name).exists() and sha256_file(directory / name) != digest:
            raise ValueError('Episode simulator asset copy differs from shared asset')
    if (sha256_file(path(meshes[0])) != asset['hashes']['mini_dust_pan.obj']
            or sha256_file(path(urdfs[0])) != asset['hashes']['mini_dust_pan.urdf']):
        raise ValueError('Task mesh/URDF does not bind the shared asset hashes')
    body = path(original['path']).parent / 'body-convention.json'
    if sha256_file(body) != asset['hashes']['body-convention.json']:
        raise ValueError('M1 and policy asset use different body conventions')
    return dict(stage0=entry(directory / 'stage0.json'), task=entry(task),
                reference_source=original, reference_task=moved_entry,
                reference_record=entry(directory / 'reference.json'), asset_manifest=entry(directory / 'asset.json'),
                asset_hashes=asset['hashes'], shared_asset_inputs=[entry(shared / n) for n in asset['hashes']],
                translation_errors=errors, task_pose_errors=task_errors, largest_clock_residual_s=residual)


def inspect_policy(directory: Path, episode: int, inputs: dict | None, grid, names: list, eval_code: str) -> dict:
    """Admit a finished video policy using the producer's real output-file contract."""
    if read(directory / 'stage0.json').get('plumbing_only'):
        raise ValueError('FAKE/plumbing Stage 0 cannot admit a policy')
    task = inspect_task(directory, episode)
    checkpoint = directory / 'training' / f'policy_{FINAL_STEPS}.safetensors'
    score = directory / f'score-{FINAL_STEPS}'
    trajectory, official_file = score / 'trajectory.parquet', score / 'official_metrics.json'
    if inputs is not None and any(inputs[k] != entry(p) for k, p in (
            ('trajectory', trajectory), ('checkpoint', checkpoint), ('reference', path(task['reference_task']['path'])))):
        raise ValueError('Adoption inputs differ from the finished policy outputs')
    learning = inspect_learning(directory, checkpoint, path(task['task']['path']))
    prepared = read(directory / 'commands.json')
    record_file = directory / 'score-self.run.json'
    if not record_file.exists():
        record_file = directory / f'score-{FINAL_STEPS}.run.json'
    record = read(record_file)
    if record.get('status') != 'completed' or record.get('stage0_sha256') != task['stage0']['sha256']:
        raise ValueError('Final evaluator process is incomplete or binds a different Stage 0')
    command = record.get('argv', prepared.get('evaluator_final', []))
    if not any(path(arg) == KIT / 'flash_chord/scripts/evaluate_policy.py'
               for arg in command if str(arg).endswith('/evaluate_policy.py')):
        raise ValueError('Policy producer did not record the kit evaluator command')
    args = overrides(command)
    if (path(args['evaluation.checkpoint']) != checkpoint
            or path(args['evaluation.object_trajectories_output']) != trajectory
            or path(args['evaluation.metrics_output']) != official_file
            or args.get('evaluation.reset_mode') != 'explicit'):
        raise ValueError('Evaluator command does not bind checkpoint/export/metrics or permits resets')
    frame, poses = inspect_export(trajectory, sha256_file(checkpoint), episode, grid, names, eval_code)
    reference = reference_poses(path(task['reference_task']['path']), frame, len(names))
    require_completion(read(official_file), sha256_file(checkpoint), len(poses))
    if read(official_file)['environment_steps'] < FINAL_STEPS:
        raise ValueError('Evaluator used a checkpoint before the recipe total')
    for row in tracking_metrics(poses, reference):
        require_tracking(row, f'Episode {episode}')
        require_first(row)
    return dict(**task, learning=learning, evaluator_run=entry(record_file),
                trajectory=entry(trajectory), official_metrics=entry(official_file))


def render_reprojection(trajectory: Path, source: dict, record: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Re-render achieved poses on the frozen M1 camera, mesh and per-frame masks on CPU."""
    import trimesh
    from src.features.moving_reference._object import render, iou
    directory = path(source['path']).parent
    tracking_file, body_file = directory / 'tracking.npz', directory / 'body-convention.json'
    mesh_file = ROOT / 'data/outputs/moving_reference/ep39/candidate-c/asset/mini_dust_pan.obj'
    with np.load(tracking_file, allow_pickle=False) as data:
        cameras, intrinsics, valid = data['world_T_camera'].copy(), data['intrinsics'].copy(), data['valid'].copy()
    mesh = trimesh.load(mesh_file, force='mesh', process=False)
    body = np.asarray(read(body_file)['scan_T_asset'])
    frame = pq.read_table(trajectory).to_pandas().sort_values(['reference_frame', 'object_body_id'])
    achieved = frame[['achieved_' + c for c in POSE]].to_numpy(float)
    poses = pose_matrices(achieved, np.asarray(read(checked(record))['translation_m'])) @ body
    if len(poses) != len(cameras) or valid.shape != (len(poses),):
        raise ValueError('Reprojection frame count differs from achieved poses/camera')
    scores, usable, frames = [], [], []
    for index, pose in enumerate(poses):
        file = directory / f'frames/frame_{index:06d}.npz'
        frames.append(entry(file))
        with np.load(file, allow_pickle=False) as data:
            mask, hands = data['object_mask'], data['hands']
            use = bool(valid[index] and np.count_nonzero(mask & ~hands) > 0)
        prediction = render(mesh.vertices, mesh.faces, np.linalg.inv(cameras[index]) @ pose, intrinsics, mask.shape)
        scores.append(iou(prediction, mask, hands))
        usable.append(use)
    return np.asarray(scores), np.asarray(usable), [entry(tracking_file), entry(body_file), entry(mesh_file), *frames]


def produce_reprojection(directory: Path, policy: dict) -> dict:
    """Write an immutable, input-bound producer record and per-frame IoUs."""
    trajectory = policy['trajectory']
    metadata = pq.read_schema(checked(trajectory)).metadata or {}
    if any(b'fake' in key.lower() or b'plumbing' in key.lower() for key in metadata):
        raise ValueError('FAKE/plumbing trajectories cannot produce admission evidence')
    directory.mkdir(parents=True, exist_ok=False)
    ious, usable, inputs = render_reprojection(checked(trajectory), policy['reference_source'], policy['reference_record'])
    np.savez_compressed(directory / 'achieved-reprojection.npz', iou=ious, usable=usable)
    result = dict(kind='pose_insurance.achieved_reprojection', status='completed',
                  trajectory=trajectory, reference=policy['reference_source'], translation=policy['reference_record'],
                  asset_hashes=policy['asset_hashes'], inputs=inputs,
                  sources=source_entries(), arrays=entry(directory / 'achieved-reprojection.npz'),
                  measurements=reprojection_summary(ious, usable))
    write(directory / 'producer.json', result)
    return entry(directory / 'producer.json')


def inspect_reprojection(producer: dict, trajectory: dict, reference: dict, translation: dict, assets: dict) -> dict:
    """Require hashed producer inputs and recompute its aggregate measurements from frame evidence."""
    if not isinstance(producer, dict) or not {'path', 'sha256'} <= producer.keys():
        raise ValueError('Achieved reprojection requires a hashed producer file')
    data = read(checked(producer))
    if data.get('kind') != 'pose_insurance.achieved_reprojection' or data.get('status') != 'completed':
        raise ValueError('Achieved reprojection requires a completed hashed producer')
    if any(data.get(k) != v for k, v in (('trajectory', trajectory), ('reference', reference), ('translation', translation), ('asset_hashes', assets))):
        raise ValueError('Achieved reprojection is bound to another trajectory/reference/asset')
    if data.get('sources') != source_entries():
        raise ValueError('Achieved reprojection producer source inventory differs')
    for e in [*data['inputs'], *data['sources'], trajectory, reference, translation]:
        checked(e)
    with np.load(checked(data['arrays']), allow_pickle=False) as arrays:
        ious, usable = arrays['iou'], arrays['usable']
    if ious.ndim != 1 or usable.shape != ious.shape or np.any(~np.isfinite(ious)) or np.any((ious < 0) | (ious > 1)):
        raise ValueError('Invalid achieved reprojection frame arrays')
    summary = reprojection_summary(ious, usable)
    if summary != data['measurements']:
        raise ValueError('Achieved reprojection summary differs from recomputed frame measurements')
    recomputed, observed, expected_inputs = render_reprojection(checked(trajectory), reference, translation)
    if (data['inputs'] != expected_inputs or not np.array_equal(usable, observed)
            or not np.array_equal(ious, recomputed)):
        raise ValueError('Achieved reprojection differs from rerendered frozen-camera/mask evidence')
    require_reprojection(summary, 'Achieved')
    return summary


def require_same_asset(first: dict, second: dict) -> None:
    if first != second:
        raise ValueError('M2v and adopted policy do not share one asset hash')


def recompute_aligned_39(trajectory: Path) -> dict:
    """Read only pinned tuning-39 labels and recompute raw/aligned metrics with one initial transform."""
    table = pq.read_table(trajectory)
    metadata = table.schema.metadata or {}
    if metadata.get(b'flash_chord.episode_index') != b'39':
        raise ValueError('Aligned label recomputation is restricted to tuning episode 39')
    frame = table.to_pandas().sort_values(['reference_frame', 'object_body_id'])
    if set(frame.object_body_id) != {0} or set(frame.environment_id) != {0}:
        raise ValueError('Aligned M2v recomputation requires one body/world on 39')
    lock = read(ROOT / 'data/configs/sources.lock.json')
    public = ROOT / 'data/datasets--nvidia--video_to_data_challenge/snapshots' / lock['dataset']['revision'] / 'track_3/public'
    label = public / 'data/chunk-000/episode_000039.parquet'
    reference_table = pq.read_table(label, columns=['frame_index', 'observation.objects'])
    if not np.array_equal(frame.reference_frame, reference_table['frame_index'].to_numpy()):
        raise ValueError('M2v export and tuning-39 mocap frame grids differ')
    rows = reference_table['observation.objects'].to_pylist()
    if any(len(row) != 1 or row[0]['name'] != 'mini_dust_pan' or not row[0]['visible'] for row in rows):
        raise ValueError('Tuning-39 mocap must contain visible dust-pan poses at every frame')
    mocap = np.asarray([row[0]['pose'] for row in rows], float)
    reference = mocap[:, [0, 1, 2, 4, 5, 6, 3]][:, None]
    achieved = frame[['achieved_' + c for c in POSE]].to_numpy(float)[:, None]
    require_poses(achieved); require_poses(reference)
    aligned, metrics = aligned_tracking_metrics(achieved, reference)
    raw = tracking_metrics(achieved, reference)[0]
    scorer = KIT / 'metric_code/track_3/AUC.py'
    spec = importlib.util.spec_from_file_location('moving_admission_auc', scorer)
    kit = importlib.util.module_from_spec(spec); spec.loader.exec_module(kit)
    import trimesh
    mesh_file = public / 'mesh/mini_dust_pan/mini_dust_pan.glb'
    mesh = trimesh.load(mesh_file, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, 1000, seed=0)
    points = np.asarray(points, float)[None]
    def scores(poses):
        warped = kit._warp_to_reference_start(poses[:, None], reference[:, None])
        return dict(AUC=kit._add_auc(kit._object_add(warped, reference[:, None], points)),
                    SP_SR=float(np.mean(kit._spider_success(warped, reference[:, None]))),
                    MP_SR=float(np.mean(kit._maniptrans_success(warped, reference[:, None], np.array([0])))),
                    RPE_cm=kit._relative_position_error_cm(warped, reference[:, None]),
                    MPPE_cm=kit._mean_per_point_error_cm(warped, reference[:, None]))
    metrics['kit'] = scores(achieved)
    metrics['kit_after_aligning_first'] = scores(aligned)
    return dict(recomputed=metrics, raw_recomputed=raw,
                inputs=[entry(trajectory), entry(label), entry(mesh_file), entry(scorer), entry(ROOT / 'data/configs/sources.lock.json')],
                alignment='one rigid first-pose transform from frame 0 and slot 0; no scale, per-frame or time fitting',
                label_episode=39, surface_points=1000, surface_seed=0)


def inspect_aligned_39(evidence: dict, policy: dict) -> dict:
    """Hash the aligned producer, preserve the raw first verdict and independently verify the numbers."""
    aligned_file = checked(evidence['mocap_metrics_aligned'])
    producer = checked(evidence['aligned_producer'])
    raw_entry = evidence['retained_first_verdict']['file']
    raw = read(checked(raw_entry))
    reported = read(aligned_file)
    trajectory = checked(policy['trajectory'])
    directory = trajectory.parent
    if producer != directory / 'm2v-aligned.py' or path(reported.get('producer', '')) != producer:
        raise ValueError('Aligned evidence producer does not belong to this M2v export')
    if (evidence['retained_first_verdict'].get('role') != 'retained first verdict; raw frames'
            or evidence['retained_first_verdict'].get('verdict') != raw.get('verdict')):
        raise ValueError('Raw M2v first verdict must be retained and marked descriptively')
    record = read(checked(evidence['mocap_run']))
    if (record.get('status') != 'completed' or record.get('stage0_sha256') != policy['stage0']['sha256']
            or record.get('result') != raw or raw.get('label_access') is not True
            or raw.get('checkpoint_sha256') != policy['learning']['checkpoint']['sha256']
            or raw.get('task_reference_sha256') != policy['reference_task']['sha256']):
        raise ValueError('Retained raw M2v result is not bound to its completed producer/checkpoint/reference')
    if reported.get('GO_with_alignment') is not True:
        raise ValueError('Aligned M2v result is not GO_with_alignment')
    result = recompute_aligned_39(trajectory)
    metrics = result['recomputed']
    for key in ('T', 'Q', 'E_m', 'E_S_m', 'Theta_deg', 'Theta_S_deg', 'p95_E_m', 'p95_Theta_deg'):
        actual = number(reported.get(key), 'aligned.' + key)
        if not np.isclose(actual, metrics[key], atol=1e-8, rtol=1e-8):
            raise ValueError(f'Aligned M2v numbers do not recompute: {key}')
    for group in ('kit', 'kit_after_aligning_first'):
        for key, expected in metrics[group].items():
            if not np.isclose(number(reported.get(group, {}).get(key), group + '.' + key), expected, atol=1e-8, rtol=1e-8):
                raise ValueError(f'Aligned M2v numbers do not recompute: {group}.{key}')
    for key in ('T', 'Q', 'E_m', 'E_S_m', 'Theta_deg', 'Theta_S_deg'):
        if not np.isclose(number(raw.get(key), 'raw.' + key), result['raw_recomputed'][key], atol=1e-8, rtol=1e-8):
            raise ValueError(f'Retained raw M2v numbers do not recompute: {key}')
    require_tracking(metrics, 'M2v aligned mocap')
    if not .45 <= metrics['kit']['AUC'] <= 1.:
        raise ValueError('Aligned M2v recomputed AUC fails .45')
    return dict(**result, aligned=evidence['mocap_metrics_aligned'], producer=evidence['aligned_producer'],
                retained_first_verdict=evidence['retained_first_verdict'], accepted=True)


def m2v_evidence(directory: Path, policy: dict, projection: dict) -> dict:
    """Bind the actual raw and aligned producer files without rewriting either verdict."""
    score = directory / f'score-{FINAL_STEPS}'
    raw = score / 'mocap_metrics.json'
    return dict(directory=str(directory.relative_to(ROOT)), stage0=policy['stage0'],
                mocap_metrics_aligned=entry(score / 'mocap_metrics_aligned.json'),
                aligned_producer=entry(score / 'm2v-aligned.py'),
                retained_first_verdict=dict(file=entry(raw), verdict=read(raw)['verdict'], role='retained first verdict; raw frames'),
                mocap_run=entry(directory / 'score-mocap.run.json'),
                m1=dict(gates=entry(path(policy['reference_source']['path']).parent / 'gates-final.json'),
                        reference=policy['reference_source'], asset_hashes=policy['asset_hashes']),
                achieved_reprojection=projection)


def inspect_m2v(evidence: dict, eval_code: str) -> dict:
    """Require the real sealed 39 result, self tracking, video reprojection and shared asset."""
    directory = path(evidence['directory'])
    stage = read(checked(evidence['stage0']))
    if entry(directory / 'stage0.json') != evidence['stage0'] or stage['episode'] != 39:
        raise ValueError('M2v evidence must identify the episode-39 video run')
    import pandas as pd
    count = stage['build']['frames']
    grid = pd.DataFrame(dict(world_index=np.zeros(count, int), frame_index=np.arange(count), object_slot=np.zeros(count, int)))
    policy = inspect_policy(directory, 39, None, grid, ['mini_dust_pan'], eval_code)
    score = directory / f'score-{FINAL_STEPS}'
    if (checked(evidence['mocap_metrics_aligned']) != score / 'mocap_metrics_aligned.json'
            or checked(evidence['retained_first_verdict']['file']) != score / 'mocap_metrics.json'):
        raise ValueError('Aligned/raw M2v files do not belong to this policy')
    aligned = inspect_aligned_39(evidence, policy)
    inspect_m1(evidence['m1'], policy['reference_source'], 39, policy['asset_hashes'])
    projection = inspect_reprojection(evidence['achieved_reprojection'], policy['trajectory'], policy['reference_source'],
                                      policy['reference_record'], policy['asset_hashes'])
    require_m2v(dict(episode=39, passed=True, verdict='GO', tracking=aligned['recomputed'],
                     AUC=aligned['recomputed']['kit']['AUC'], reprojection=projection))
    return dict(policy, aligned_mocap=aligned)


def make_admission(directory_name: str, m2v_name: str, output_name: str, disclosure: str) -> Path:
    """Validate M1/M2v first, then build admission only when the scored policy is finished."""
    directory, m2v_dir, output = path(directory_name), path(m2v_name), path(output_name)
    if not output.is_relative_to(SCRATCH) and not output.is_relative_to(OUTPUT):
        raise ValueError('Admission output must be under mpack-dry or moving-pack outputs')
    if output.exists():
        raise FileExistsError('Use a new immutable admission output directory')
    def m1_binding(p):
        source = p['reference_source']
        return dict(gates=entry(path(source['path']).parent / 'gates-final.json'), reference=source,
                    asset_hashes=p['asset_hashes'])
    for episode, folder in ((38, directory), (39, m2v_dir)):
        reference = read(folder / 'reference.json')
        binding = dict(reference_source=entry(reference['source_path']), asset_hashes=read(folder / 'asset.json')['hashes'])
        inspect_m1(m1_binding(binding), binding['reference_source'], episode, binding['asset_hashes'])
    roster, objects = official_roster()
    _, code = evaluator_inputs()
    m2v_stage = read(m2v_dir / 'stage0.json')
    if m2v_stage['episode'] != 39:
        raise ValueError('M2v must be the tuning-39 video run')
    import pandas as pd
    n = m2v_stage['build']['frames']
    grid = pd.DataFrame(dict(world_index=np.zeros(n, int), frame_index=np.arange(n), object_slot=np.zeros(n, int)))
    m2v_policy = inspect_policy(m2v_dir, 39, None, grid, ['mini_dust_pan'], code)
    # Evidence is written in this slice's scratch tree even when the scored GPU
    # job is unfinished. Its output folder remains untouched and reusable.
    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='m2v-admission-', dir=SCRATCH))
    m2v_projection = produce_reprojection(work / 'reprojection', m2v_policy)
    m2v = m2v_evidence(m2v_dir, m2v_policy, m2v_projection)
    verified = inspect_m2v(m2v, code)
    write(work / 'm2v-evidence.json', m2v)
    write(work / 'm2v-validation.json', verified)
    require_same_asset(read(directory / 'asset.json')['hashes'], verified['asset_hashes'])
    missing = []
    for name in ('train.run.json', 'score-self.run.json', f'training/policy_{FINAL_STEPS}.safetensors',
                 f'score-{FINAL_STEPS}/trajectory.parquet', f'score-{FINAL_STEPS}/official_metrics.json'):
        if not (directory / name).is_file():
            missing.append(name)
    if (directory / 'train.run.json').is_file() and read(directory / 'train.run.json').get('status') != 'completed':
        missing.append('training is not completed')
    if missing:
        raise ValueError('Final M1 and aligned M2v passed; missing finished 38 policy evidence: '
                         + '; '.join(missing) + f'; M2v evidence: {work.relative_to(ROOT)}')
    episode = read(directory / 'stage0.json')['episode']
    if episode != 38:
        raise ValueError('Current video recipe admits only scored episode 38')
    policy = inspect_policy(directory, episode, None, roster[episode], objects[episode].object_names, code)
    output.mkdir(parents=True, exist_ok=False)
    projection = produce_reprojection(output / 'ep38-reprojection', policy)
    admission = dict(schema_version=2, episode=episode, policy_directory=str(directory.relative_to(ROOT)),
                     inputs=dict(trajectory=policy['trajectory'], checkpoint=policy['learning']['checkpoint'], reference=policy['reference_task']),
                     m1=m1_binding(policy), m2v=m2v, achieved_reprojection=projection,
                     abstract_disclosure=entry(disclosure))
    write(output / 'admission.json', admission)
    adoption = dict(episode=episode, trajectory=policy['trajectory']['path'], checkpoint=policy['learning']['checkpoint']['path'],
                    reference=policy['reference_task']['path'], admission=str((output / 'admission.json').relative_to(ROOT)))
    admit(adoption, roster[episode], objects[episode].object_names, code)
    write(output / 'adoptions.json', dict(adoptions=[adoption]))
    return output / 'adoptions.json'


def admit(adoption: dict, grid, names: list, eval_code: str) -> dict:
    """Verify all scored-adoption clauses and their content-bound producer evidence."""
    episode = adoption['episode']
    inputs = {key: entry(adoption[key]) for key in ('trajectory', 'checkpoint', 'reference', 'admission')}
    admission = read(adoption['admission'])
    if admission.get('schema_version') != 2 or admission.get('episode') != episode or admission.get('plumbing_only'):
        raise ValueError('Admission identity mismatch or FAKE plumbing admission')
    for key in ('trajectory', 'checkpoint', 'reference'):
        if admission['inputs'][key] != inputs[key]:
            raise ValueError(f'Admission does not bind {key} content')
    frame, poses = inspect_export(path(adoption['trajectory']), inputs['checkpoint']['sha256'], episode, grid, names, eval_code)
    reference = reference_poses(path(adoption['reference']), frame, len(names))
    policy = inspect_policy(path(admission['policy_directory']), episode, inputs, grid, names, eval_code)
    inspect_m1(admission['m1'], policy['reference_source'], episode, policy['asset_hashes'])
    m2v = inspect_m2v(admission['m2v'], eval_code)
    require_same_asset(policy['asset_hashes'], m2v['asset_hashes'])
    projection = inspect_reprojection(admission['achieved_reprojection'], inputs['trajectory'],
                                      policy['reference_source'], policy['reference_record'], policy['asset_hashes'])
    require_reprojection(projection, 'Achieved')
    metrics = tracking_metrics(poses, reference)
    for row in metrics:
        require_tracking(row, f'Episode {episode} slot {row["slot"]}')
        require_first(row)
    disclosure = admission['abstract_disclosure']
    checked(disclosure)
    return dict(episode=episode, source='moving policy', **inputs, policy=policy,
                m2v=admission['m2v'], m1=admission['m1'], abstract_disclosure=disclosure,
                tracking=metrics, achieved_reprojection=admission['achieved_reprojection'],
                m2v_aligned_validation=m2v['aligned_mocap'])



def compose_exports(v5: dict, replacements: dict, directory: Path) -> list[dict]:
    """Copy the twenty chosen exports into a fresh composition directory."""
    if not directory.resolve().is_relative_to(SCRATCH) and not directory.resolve().is_relative_to(OUTPUT):
        raise ValueError('Composition output must be under moving-pack outputs or mpack-dry scratch')
    directory.mkdir(parents=True, exist_ok=False)
    exports = []
    for export in v5['exports']:
        episode = int(Path(export['path']).stem.split('_')[1])
        source = path(replacements.get(episode, export['path']))
        destination = directory / f'episode_{episode:06d}.parquet'
        shutil.copyfile(source, destination)
        exports.append(dict(episode=episode, **entry(destination)))
    return exports


def compare_csv(v5: dict, output: Path, adopted: set) -> dict:
    """Require text equality of every unadopted CSV field apart from the commit URL."""
    with path(v5['file']).open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; before = list(reader)
    with output.open(newline='') as stream:
        reader = csv.DictReader(stream); after = list(reader)
        if reader.fieldnames != fields:
            raise ValueError('Packed CSV header differs from v5')
    if len(after) != len(before) or [r['row_id'] for r in before] != [r['row_id'] for r in after]:
        raise ValueError('Packed CSV must retain all ordered v5 row IDs')
    unchanged, changed = 0, {}
    for old, new in zip(before, after):
        episode = int(old['row_id'].split('/')[1][1:])
        differences = [k for k in fields if k != 'code_commit_url' and old[k] != new[k]]
        if episode not in adopted:
            if differences:
                raise ValueError(f'Unadopted row changed: {old["row_id"]}: {differences}')
            unchanged += 1
        if differences:
            changed[str(episode)] = changed.get(str(episode), 0) + 1
    return dict(total_rows=len(after), unchanged_rows=unchanged, changed_rows_by_episode=changed,
                comparison='all CSV fields as text, preserving signed zeros; only code_commit_url ignored')


def source_entries() -> list[dict]:
    names = [Path(__file__), Path(__file__).with_name('_moving.py'),
             ROOT / 'src/features/dynamic_feasibility/_metrics.py',
             ROOT / 'src/paths.py', ROOT / 'src/provenance.py',
             ROOT / 'scripts/reproduce/d2_poses_moving.sh', ROOT / 'scripts/reproduce/check_release.sh',
             ROOT / 'docs/chain/moving-m1-protocol.md', ROOT / 'docs/chain/moving-m2-protocol.md',
             ROOT / 'docs/chain/moving-m2v-protocol.md',
             ROOT / 'src/features/moving_reference/_object.py', ROOT / 'src/features/moving_reference/camera.py',
             ROOT / 'src/features/moving_reference/_final_gates.py', ROOT / 'src/features/moving_reference/final_gates.py',
             ROOT / 'docs/chain/moving-m1-sealed.md', ROOT / 'docs/chain/moving-m2v-outcome.md']
    return [entry(p) for p in names]


def committed(sources: list[dict], url: str) -> str:
    match = re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/([0-9a-f]{40})', url)
    if match is None or 'PLACEHOLDER' in url:
        raise ValueError('A real full-SHA GitHub commit URL is required')
    for source in sources:
        content = subprocess.check_output(['git', 'show', f'{match[1]}:{source["path"]}'], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != source['sha256']:
            raise ValueError(f'Code commit source differs: {source["path"]}')
    return match[1]


def kit_commands(results: Path, output: Path, url: str) -> list[list[str]]:
    python = str(ROOT / 'data/.runtime/kit-env/bin/python')
    return [[python, '-B', str(KIT / 'tools/pack_submission.py'), '--track', 'track_3',
             '--results', str(results), '--code-commit', url, '--eval-script',
             str(KIT / 'flash_chord/scripts/evaluate_policy.py'), '--skip-crosscheck', '--out', str(output)],
            [python, '-B', str(KIT / 'tools/preflight.py'), '--track', 'track_3', '--results', str(results), '--skip-kaggle']]


def pack(request_file: str, code_url: str, output_name: str, dry_run: bool = False) -> Path:
    """Validate first, then compose, pack, preflight and record an immutable manifest."""
    output = path(output_name)
    allowed = SCRATCH if dry_run else OUTPUT
    if not output.is_relative_to(allowed) or output.suffix != '.csv':
        raise ValueError(f'Output must be a CSV under {allowed}; submission/ is forbidden')
    directory = output.parent
    if directory.exists():
        raise FileExistsError('Use a fresh output subdirectory')
    request = read(request_file)
    roster, objects = official_roster()
    adoptions = request['adoptions']
    episodes = [a['episode'] for a in adoptions]
    if len(episodes) != len(set(episodes)) or any(type(ep) is not int or ep not in roster for ep in episodes):
        raise ValueError('Adoptions must be unique scored evaluation episodes; tuning/validation/excluded episodes forbidden')
    if not dry_run and not episodes:
        raise ValueError('No adopted moving episodes; retain D2 v5')
    v5 = frozen_v5()
    evaluation_sources, eval_code = evaluator_inputs()
    admissions = {a['episode']: admit(a, roster[a['episode']], objects[a['episode']].object_names, eval_code) for a in adoptions}
    sources = source_entries()
    if not dry_run:
        committed(sources, code_url)
    elif not re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}', code_url):
        raise ValueError('Dry pack still requires a syntactically valid full commit URL')
    directory.mkdir(parents=True, exist_ok=False)
    exports = compose_exports(v5, {a['episode']: a['trajectory'] for a in adoptions}, directory / 'results')
    for export in exports:
        ep = export['episode']
        expected = admissions[ep]['trajectory']['sha256'] if ep in admissions else next(
            e['sha256'] for e in v5['exports'] if int(Path(e['path']).stem.split('_')[1]) == ep)
        if export['sha256'] != expected:
            raise ValueError('Export changed between admission and composition')
    commands = kit_commands(directory / 'results', output, code_url)
    for command, name in zip(commands, ('pack.log', 'preflight.log')):
        with (directory / name).open('w') as stream:
            subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    if 'Ready to submit.' not in (directory / 'preflight.log').read_text():
        raise ValueError('Kit preflight not Ready to submit')
    comparison = compare_csv(v5, output, set(episodes))
    provenance = [admissions.get(e['episode'], dict(episode=e['episode'], source='v5 static',
                   trajectory=next(x for x in v5['exports'] if int(Path(x['path']).stem.split('_')[1]) == e['episode']))) for e in exports]
    manifest = dict(kind='pose_insurance.moving_pack', schema_version=1,
                    status='DRY RUN; NOT ADMISSIBLE' if dry_run else 'packed; lead release review required; not uploaded',
                    dry_run=dry_run, code_commit_url=code_url, file=entry(output), request=entry(request_file),
                    v5_manifest=entry(V5_MANIFEST), v5_csv=entry(v5['file']), sources=sources,
                    kit_inputs=[entry(KIT / name) for name in ('tools/pack_submission.py', 'tools/preflight.py',
                                'submit/v2d_submit.py', 'flash_chord/scripts/evaluate_policy.py',
                                'KIT_MANIFEST.json', 'config/tracks.json', 'data/track_3_sample_submission.csv',
                                'data/track_3_evaluation_objects.json')],
                    evaluator_sources=list(evaluation_sources.values()), exports=exports, episodes=provenance,
                    commands=commands, csv_comparison=comparison,
                    preflight_verdict='Ready to submit (Kaggle checks skipped)')
    write(output.with_suffix('.manifest.json'), manifest)
    write(directory / 'release-pose-entry.json', dict(path=manifest['file']['path'],
          sha256=manifest['file']['sha256'], manifest=str(output.with_suffix('.manifest.json').relative_to(ROOT)),
          moving_manifest=str(output.with_suffix('.manifest.json').relative_to(ROOT))))
    return output.with_suffix('.manifest.json')


def verify_manifest(name: str) -> None:
    """Re-admit a moving pack for the optional release entry, refusing all dry packs."""
    manifest = read(name)
    if manifest.get('kind') != 'pose_insurance.moving_pack' or manifest.get('dry_run') is not False:
        raise ValueError('Release requires a real admitted moving pack')
    # Verify explicit hash entries only; the immutable v5 parent may itself pin AGENTS.md.
    def visit(value):
        if isinstance(value, dict):
            if 'sha256' in value:
                if Path(value.get('path', '')).name == 'AGENTS.md':
                    raise ValueError('Moving manifest must not pin AGENTS.md')
                checked(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    if manifest['sources'] != source_entries():
        raise ValueError('Moving release source inventory differs from this implementation')
    committed(manifest['sources'], manifest['code_commit_url'])
    v5 = frozen_v5()
    roster, objects = official_roster()
    _, code = evaluator_inputs()
    request = read(checked(manifest['request']))
    adopted = {}
    for adoption in request['adoptions']:
        ep = adoption['episode']
        if type(ep) is not int or ep not in roster or ep in adopted:
            raise ValueError('Invalid or duplicate adopted episode in release')
        adopted[ep] = admit(adoption, roster[ep], objects[ep].object_names, code)
    if not adopted:
        raise ValueError('A moving release requires at least one admitted moving episode')
    expected_episodes = set(roster)
    if len(manifest['exports']) != len(roster) or {e['episode'] for e in manifest['exports']} != expected_episodes:
        raise ValueError('Moving release export roster mismatch')
    for export in manifest['exports']:
        ep = export['episode']
        expected = adopted[ep]['trajectory']['sha256'] if ep in adopted else next(
            e['sha256'] for e in v5['exports'] if int(Path(e['path']).stem.split('_')[1]) == ep)
        if export['sha256'] != expected:
            raise ValueError('Composed export differs from its admitted episode source')
    if manifest['csv_comparison'] != compare_csv(v5, checked(manifest['file']), set(adopted)):
        raise ValueError('Moving release CSV comparison differs')
    expected_provenance = [adopted.get(e['episode'], dict(episode=e['episode'], source='v5 static',
                          trajectory=next(x for x in v5['exports'] if int(Path(x['path']).stem.split('_')[1]) == e['episode'])))
                          for e in manifest['exports']]
    if manifest['episodes'] != expected_provenance:
        raise ValueError('Per-episode manifest provenance differs from re-admission')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='release-repack-', dir=SCRATCH))
    results = work / 'results'
    results.mkdir()
    for export in manifest['exports']:
        shutil.copyfile(checked(export), results / f'episode_{export["episode"]:06d}.parquet')
    for command, label in zip(kit_commands(results, work / 'poses.csv', manifest['code_commit_url']), ('pack', 'preflight')):
        with (work / (label + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    if sha256_file(work / 'poses.csv') != manifest['file']['sha256']:
        raise ValueError('Moving CSV differs from an official repack of admitted exports')
    print('Moving pack re-admitted; all unadopted v5 rows preserved.')


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--request')
    parser.add_argument('--code-commit')
    parser.add_argument('--out')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-manifest')
    parser.add_argument('--make-admission', metavar='EPISODE_OUTPUT')
    parser.add_argument('--m2v-output', default='data/outputs/moving_policy/ep39-v1')
    parser.add_argument('--abstract-disclosure', default='docs/report/track3/main.tex')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    if args.make_admission:
        if not args.out:
            parser.error('--make-admission requires --out for a new evidence directory')
        print(make_admission(args.make_admission, args.m2v_output, args.out, args.abstract_disclosure))
        return
    if args.verify_manifest:
        verify_manifest(args.verify_manifest)
        return
    if not all((args.request, args.code_commit, args.out)):
        parser.error('--request, --code-commit and --out are required')
    print(pack(args.request, args.code_commit, args.out, args.dry_run))


if __name__ == '__main__':
    main()
