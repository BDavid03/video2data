"""Adapt admitted v3 tasks to T1 and reuse the established policy and physical checks."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from src import historical
from src.gpu import GpuLease
from src.paths import output_path
from src.provenance import input_file, run_fields, write_json
from src.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, heavy_execution,
                         new_run_id, producer, read_json, recorded, runtime_flag)
from .request import Request
from .scored_stereo_config import FEATURE, FC, KIT, BUDGET, POSE_PROVENANCE

PROTOCOL = 'docs/chain/pose-stereo-placement-protocol.md'
OUTPUT = 'data/outputs/pose_insurance/stereo-placement-scored'


def environment(runtime: dict, cpu: bool = False, candidate: str = 'T1') -> dict:
    """Keep the established GPU settings with caches and temporary files in task scratch."""
    scratch = output_path(ROOT, 'data/.runtime/tmp/d2v5' if candidate == 'TW' else 'data/.runtime/tmp/t1-scored')
    original_folders = dict(CACHE_FOLDERS)
    try:
        if candidate == 'TW':
            CACHE_FOLDERS.update({key: str(scratch / key.lower()) for key in original_folders})
        heavy_execution(dict(runtime, python=runtime['diagnostic_python']) if cpu else runtime)
    finally:
        CACHE_FOLDERS.update(original_folders)
    for key in (*CACHE_FOLDERS, 'WARP_CACHE_PATH', 'JAX_COMPILATION_CACHE_DIR'):
        path = scratch / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import sys
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(FC / 'src')]),
                      CUDA_VISIBLE_DEVICES='' if cpu else '0', PYTHONDONTWRITEBYTECODE='1',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_ENABLE_COMPILATION_CACHE='true',
                      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0', WANDB_MODE='disabled',
                      HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1')
    return os.environ.copy()


def admit_tw(plan: dict) -> None:
    """Require the owner's dated approval and the exact H3 gate evidence for episode 19."""
    from src.runs import checked_entry
    from src.provenance import sha256_file
    if plan.get('candidate') != 'TW' or plan['episode'] != 19 or plan['budget'] != BUDGET:
        raise PermissionError('TW is admitted for evaluation episode 19 only')
    owner = checked_entry(plan['approval'], {'inputs': []})
    text = ' '.join(owner.read_text().split())
    if ('2026-11-03: approve D2 v5 heading for episode 19' not in text
            or '"all approved"' not in text or '19:40 UTC' not in text):
        raise PermissionError('The dated owner approval is absent')
    evidence = read_json(checked_entry(plan['h3_summary'], {'inputs': []}))
    if not evidence['gates_passed'] or not all(evidence[key]['passed'] for key in ('G1', 'G2', 'G3')):
        raise PermissionError('TW requires all recorded H3 gates')
    if plan['start_poses'] != evidence['episode19']['start_poses']:
        raise ValueError('TW poses differ from the full-precision recorded selection')
    scene = deepcopy(plan['base_scene'])
    for row, pose in zip(scene, plan['start_poses']):
        if row['position'] != pose[:3]:
            raise ValueError('TW positions differ from v4')
        row['quaternion_wxyz'] = [pose[6], *pose[3:6]]
    if scene != plan['scene']:
        raise ValueError('TW scene differs from the H3-only frame conversion')
    for entry in plan['inputs']:
        checked_entry(entry, {'inputs': []})
    payload = {key: value for key, value in plan.items() if key != 'identity'}
    import hashlib
    import json
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != plan['identity']:
        raise ValueError('TW plan identity differs from its contents')
    if not all(historical.source_matches(path, digest) for path, digest in plan['sources'].items()):
        raise ValueError('TW sources changed after the plan was frozen')


def build(plan: dict, plan_path: str, request: Request) -> Path:
    """Build admitted T1 positions or TW body frames while preserving the parent task."""
    runtime = runtime_flag(FEATURE, 'scored_stereo_authorized', 'Scored stereo build')
    authorize_episode(runtime, request, 'scored stereo build')
    candidate = plan.get('candidate', 'T1')
    if candidate == 'TW':
        admit_tw(plan)
    elif candidate != 'T1':
        raise ValueError('Unknown scored stereo candidate')
    environment(runtime, cpu=Path(os.sys.executable).resolve() == (ROOT / runtime['diagnostic_python']).resolve(), candidate=candidate)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    parent = producer(plan['base_build'], 'pose_insurance.build', request)
    record = run_fields(ROOT, 'pose_insurance', 'build', dict(plan=plan['identity'], candidate=candidate),
                        parents=[parent[0], *([plan['h3_pose_run']] if candidate == 'TW' else [])], inputs=[plan.get('protocol', {}).get('path', PROTOCOL), plan_path,
                        FEATURE / ('scored_heading_v5_queue.json' if candidate == 'TW' else 'scored_stereo_queue.json'),
                        FEATURE / 'runtime.json', *[e['path'] for e in plan.get('inputs', [])]])
    output = 'data/outputs/pose_insurance/heading-v5' if candidate == 'TW' else OUTPUT
    directory = output_path(ROOT, f'{output}/builds/{new_run_id()}/episode_{request.episode_index:06d}')
    with recorded(directory, record):
        original = pq.read_table(artifact(parent, 'task.parquet', record))
        task = original.to_pylist()[0]
        previous = deepcopy(task)
        positions = np.asarray(task['object_body_position'], float)
        if not np.allclose(positions, np.array([r['position'] for r in plan['base_scene']])[None], rtol=0, atol=0):
            raise ValueError('Original task is not the exact v3 initial scene')
        equivalence = None
        if candidate == 'TW':
            from .scored_heading_assets import write_assets
            expected_quaternions = np.array([r['quaternion_wxyz'] for r in plan['base_scene']])
            if not np.array_equal(np.asarray(task['object_body_wxyz']), np.broadcast_to(expected_quaternions, np.asarray(task['object_body_wxyz']).shape)):
                raise ValueError('Parent task rotations are not the exact static v4 scene')
            if plan['base_scene'] != parent[1]['scene'] or any(a['position'] != b['position'] for a, b in zip(plan['scene'], plan['base_scene'])):
                raise ValueError('TW must preserve the admitted v4 positions')
            equivalence = write_assets(task, plan['scene'], directory, record)
            task['object_body_wxyz'] = np.repeat([[r['quaternion_wxyz'] for r in plan['scene']]], len(positions), axis=0).tolist()
            allowed = ('object_body_wxyz', 'object_mesh_paths', 'object_urdf_paths')
        else:
            positions[:, 1, :2] = plan['scene'][1]['position'][:2]
            task['object_body_position'] = positions.tolist()
            allowed = ('object_body_position',)
        if any(task[k] != previous[k] for k in task if k not in allowed):
            raise ValueError('Build changed a field outside the admitted task transform')
        pq.write_table(pa.Table.from_pylist([task], schema=original.schema), directory / 'task.parquet')
        record.update(split='evaluation', episode_index=request.episode_index, scene=plan['scene'],
                      object_names=parent[1]['object_names'], roster_frames=parent[1]['roster_frames'],
                      task_frames=parent[1]['task_frames'], dataset_revision=parent[1]['dataset_revision'],
                      candidate=candidate, base_scene=parent[1]['scene'], plan_identity=plan['identity'],
                      equivalence=equivalence if candidate == 'TW' else 'Only slot-one XY changes in object_body_position; every other field is identical')
        write_json(directory / 'initial_scene.json', {k: record[k] for k in ('scene', 'object_names', 'task_frames')})
    return directory / 'run.json'


def reserved_tw(plan: dict, run_path: str) -> dict:
    """Require the requested policy path and plan to match the sole TW reservation."""
    path = output_path(ROOT, 'data/outputs/pose_insurance/heading-v5/attempt.json')
    if not path.is_file():
        raise PermissionError('TW policy requires a reserved attempt')
    attempt = read_json(path)
    requested = output_path(ROOT, run_path)
    reserved = output_path(ROOT, attempt['run'])
    runs = ROOT / 'data/outputs/pose_insurance/heading-v5/runs'
    if (attempt.get('stage') != 'attempt' or attempt.get('status') != 'started'
            or attempt.get('episode') != 19 or attempt.get('identity') != plan['identity']
            or requested != reserved or not reserved.is_relative_to(runs)
            or reserved.parent.name != 'episode_000019' or reserved.name != 'run.json'):
        raise PermissionError('TW policy differs from the reserved plan or run directory')
    return input_file(ROOT, path)


def claim_tw(plan: dict, directory_name: str) -> list:
    """Claim the reserved TW policy exactly once before initializing caches or writing a run."""
    directory = output_path(ROOT, directory_name)
    reservation = reserved_tw(plan, str(directory / 'run.json'))
    if directory.exists():
        raise FileExistsError('TW reserved run directory already exists; no retry')
    path = output_path(ROOT, 'data/outputs/pose_insurance/heading-v5/attempt-claim.json')
    claim = dict(identity=plan['identity'], run=str((directory / 'run.json').relative_to(ROOT)),
                 reservation=reservation, owner_pid=os.getpid())
    with path.open('x') as stream:
        json.dump(claim, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return [reservation, input_file(ROOT, path)]


def run(plan_path: str, directory_name: str, build_path: str | None = None) -> Path:
    """Run the unchanged 512-step policy and official evaluator under a serial GPU lease."""
    runtime = runtime_flag(FEATURE, 'scored_stereo_authorized', 'Scored stereo evaluator')
    plan = read_json(plan_path)
    request = Request(episode_index=plan['episode'], budget=plan['budget'])
    authorize_episode(runtime, request, 'scored stereo evaluator')
    candidate = plan.get('candidate', 'T1')
    attempt_inputs = []
    if candidate == 'TW':
        admit_tw(plan)
        attempt_inputs = claim_tw(plan, directory_name)
    env = environment(runtime, candidate=candidate)
    from .handler import training_command, execute
    from .scored_scale_handler import optimizer_state, inspect
    task_path = build_path or build(plan, plan_path, request)
    task = producer(task_path, 'pose_insurance.build', request, allowed=POSE_PROVENANCE)
    if candidate == 'TW' and (task[1]['plan_identity'] != plan['identity'] or task[1]['scene'] != plan['scene']):
        raise ValueError('TW policy must use the frozen CPU build')
    record = run_fields(ROOT, 'pose_insurance', 'run', dict(plan=plan['identity'], budget=request.budget, candidate=candidate),
                        parents=[task_path], inputs=[plan.get('protocol', {}).get('path', PROTOCOL), plan_path, FEATURE / 'runtime.json',
                        ROOT / 'data/outputs/pose_insurance/runtime_inventory.json', KIT / 'flash_chord/scripts/evaluate_policy.py',
                        *[entry['path'] for entry in attempt_inputs]])
    record.update(split='evaluation', episode_index=request.episode_index, owner_pid=os.getpid(),
                  dataset_revision=task[1]['dataset_revision'], scene=task[1]['scene'], plan_identity=plan['identity'])
    directory = output_path(ROOT, directory_name)
    with recorded(directory, record):
        task_file = artifact(task, 'task.parquet', record)
        main_lock = ROOT.parents[2] / 'data/.runtime/gpu-stage.lock'
        if main_lock.exists():
            raise RuntimeError(f'Main-checkout GPU lease exists: {main_lock}')
        started = time.perf_counter()
        with GpuLease(ROOT, 2048, 1024, stage='pose_insurance_tw_v5' if candidate == 'TW' else 'pose_insurance_t1_v4') as lease:
            try:
                execute(training_command(request, task_file, directory), directory / 'training.log', env, record)
                record.update(optimizer_state(directory))
                command = [str(ROOT / 'data/.runtime/flashchord/venv/bin/python'), str(KIT / 'flash_chord/scripts/evaluate_policy.py'),
                           f"evaluation.checkpoint={ROOT / record['checkpoint']}", 'evaluation.world_count=1',
                           'evaluation.reset_mode=explicit', 'evaluation.start_frame=0',
                           f'evaluation.object_trajectories_output={directory / "trajectory.parquet"}',
                           f'evaluation.metrics_output={directory / "official_metrics.json"}',
                           f'hydra.run.dir={directory / "evaluate_hydra"}', 'logging.mode=disabled']
                execute(command, directory / 'evaluation.log', env, record)
                lease.require_headroom()
            finally:
                record['gpu'] = lease.report()
        record['policy_seconds'] = time.perf_counter() - started
        record['checks'] = inspect(directory / 'trajectory.parquet', task[1])
        write_json(directory / 'result.json', {k: record[k] for k in ('checks', 'policy_seconds', 'optimizer_updates', 'actor_adam_steps')})
    return directory / 'run.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--build')
    args = parser.parse_args()
    print(run(args.plan, args.directory, args.build), flush=True)
