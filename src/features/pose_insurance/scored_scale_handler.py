"""Build S1 tasks from v2 producers and run the existing FlashSAC/evaluator path."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from src.gpu import GpuLease
from src.paths import output_path
from src.provenance import run_fields, write_json, sha256_file
from src.runs import (ROOT, runtime_flag, authorize_episode, heavy_execution, producer,
                         artifact, recorded, new_run_id, read_json)
from .request import Request
from .handler import training_command, execute, inspect_export
from .scene import FEATURE, FC, KIT
from ._scored_scale import scaled_scene, physical_checks

PROTOCOL = ROOT / 'docs/chain/pose-scale-protocol.md'


def environment(runtime: dict) -> dict:
    """Use the established runtime settings with job-owned temporary storage."""
    heavy_execution(runtime)
    temporary = output_path(ROOT, 'data/.runtime/tmp/ctx05-scored-work/temp')
    temporary.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[key] = str(temporary)
    for key, relative in {'WARP_CACHE_PATH': 'data/.cache/warp', 'JAX_COMPILATION_CACHE_DIR': 'data/.cache/jax'}.items():
        path = output_path(ROOT, relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(FC / 'src')]), CUDA_VISIBLE_DEVICES='0',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_ENABLE_COMPILATION_CACHE='true',
                      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0', WANDB_MODE='disabled',
                      HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1')
    return os.environ.copy()


def build(plan: dict, request: Request) -> Path:
    """Reuse the admitted v2 task, changing only its repeated S1 initial positions."""
    runtime = runtime_flag(FEATURE, 'scored_scale_authorized', 'Scored scale build')
    authorize_episode(runtime, request, 'scored scale build')
    heavy_execution(runtime)
    parent = producer(plan['base_build'], 'pose_insurance.build', request)
    record = run_fields(ROOT, 'pose_insurance', 'build', dict(plan=plan['identity'], candidate='S1'),
                        parents=[parent[0], *[a['path'] for a in plan['stereo']['anchors']]],
                        inputs=[PROTOCOL, FEATURE / 'queue_v3.json', FEATURE / 'runtime.json'])
    directory = output_path(ROOT, f'data/outputs/pose_insurance/build-v3/{new_run_id()}/episode_{request.episode_index:06d}')
    with recorded(directory, record):
        original = pq.read_table(artifact(parent, 'task.parquet', record))
        task = original.to_pylist()[0]
        scene = scaled_scene(parent[1]['scene'], plan['stereo']['s'], plan['stereo']['valid'])
        if scene != plan['scene']:
            raise ValueError('Build differs from the recorded S1 plan')
        previous = deepcopy(task)
        task['object_body_position'] = np.repeat([[r['position'] for r in scene]], len(task['object_body_position']), axis=0).tolist()
        if any(task[k] != previous[k] for k in task if k != 'object_body_position'):
            raise ValueError('S1 must change positions only in the existing task')
        pq.write_table(pa.Table.from_pylist([task], schema=original.schema), directory / 'task.parquet')
        record.update(split='evaluation', episode_index=request.episode_index, scene=scene,
                      object_names=parent[1]['object_names'], roster_frames=parent[1]['roster_frames'],
                      task_frames=parent[1]['task_frames'], dataset_revision=parent[1]['dataset_revision'],
                      scale_s=plan['stereo']['s'], candidate='S1', base_scene=parent[1]['scene'],
                      equivalence='Every task field except object_body_position is identical to the v2 producer')
        write_json(directory / 'initial_scene.json', {k: record[k] for k in ('scene', 'object_names', 'task_frames')})
    return directory / 'run.json'


def optimizer_state(directory: Path) -> dict:
    """Reuse the established checkpoint-layout checks for the fixed 512-step budget."""
    from safetensors import safe_open
    checkpoint = directory / 'training/policy_512.safetensors'
    with safe_open(str(checkpoint), framework='numpy') as policy:
        leaves = len(json.loads(policy.metadata()['manifest']))
    with safe_open(str(directory / 'training/state_512.safetensors'), framework='numpy') as state:
        manifest = json.loads(state.metadata()['manifest'])
        updates = int(state.get_tensor(manifest[-1]['key']).item())
        actor = state.get_tensor(manifest[leaves]['key'])
        if actor.size != 1 or actor.dtype.kind != 'i' or not all(np.isfinite(state.get_tensor(k)).all() for k in state.keys()):
            raise ValueError('Invalid optimizer state')
        actor_steps = int(actor.item())
    if updates < 1 or actor_steps < 1:
        raise ValueError('The learned trajectory requires real optimizer updates')
    return dict(optimizer_updates=updates, actor_adam_steps=actor_steps,
                checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha256_file(checkpoint))


def inspect(path: Path, task: dict) -> dict:
    """Keep numerical physical evidence even when the existing inspector rejects an export."""
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda r: (r['reference_frame'], r['environment_id'], r['object_body_id']))
    columns = [f'achieved_position_{s}' for s in 'xyz'] + [f'achieved_quaternion_{s}' for s in 'xyzw']
    poses = np.array([[r[c] for c in columns] for r in rows]).reshape(-1, 1, len(task['object_names']), 7)
    intended = np.array([r['position'] + r['quaternion_wxyz'][1:] + r['quaternion_wxyz'][:1] for r in task['scene']])
    checks = physical_checks(poses, intended)
    try:
        checks.update(inspect_export(path, task), contract_passed=True)
    except ValueError as error:
        checks.update(contract_passed=False, rejection=str(error))
    checks['accepted'] = checks['physical_passed'] and checks['contract_passed']
    return checks


def run(plan_path: str, directory_name: str) -> Path:
    """Train and evaluate one changed S1 scene once under the cooperative GPU lease."""
    runtime = runtime_flag(FEATURE, 'scored_scale_authorized', 'Scored scale evaluator')
    plan = read_json(plan_path)
    request = Request(episode_index=plan['episode'], budget=plan['budget'])
    authorize_episode(runtime, request, 'scored scale evaluator')
    env = environment(runtime)
    task_path = build(plan, request)
    env = environment(runtime)
    task = producer(task_path, 'pose_insurance.build', request)
    record = run_fields(ROOT, 'pose_insurance', 'run', dict(plan=plan['identity'], budget=request.budget),
                        parents=[task_path], inputs=[PROTOCOL, plan_path, FEATURE / 'runtime.json',
                        ROOT / 'data/outputs/pose_insurance/runtime_inventory.json', KIT / 'flash_chord/scripts/evaluate_policy.py'])
    record.update(split='evaluation', episode_index=request.episode_index, owner_pid=os.getpid(),
                  dataset_revision=task[1]['dataset_revision'], scene=task[1]['scene'], plan_identity=plan['identity'])
    directory = output_path(ROOT, directory_name)
    with recorded(directory, record):
        task_file = artifact(task, 'task.parquet', record)
        main_lock = ROOT.parents[2] / 'data/.runtime/gpu-stage.lock'
        if main_lock.exists():
            raise RuntimeError(f'Main-checkout GPU lease exists: {main_lock}')
        started = time.perf_counter()
        with GpuLease(ROOT, 2048, 1024, stage='pose_insurance_s1_v3') as lease:
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
    args = parser.parse_args()
    print(run(args.plan, args.directory), flush=True)
