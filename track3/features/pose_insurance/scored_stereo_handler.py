"""Adapt admitted v3 tasks to T1 and reuse the established policy and physical checks."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from track3.gpu import GpuLease
from track3.paths import output_path
from track3.provenance import run_fields, write_json
from track3.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, heavy_execution,
                         new_run_id, producer, read_json, recorded, runtime_flag)
from .request import Request
from .scene import FEATURE, FC, KIT
from .handler import training_command, execute
from .scored_scale_handler import optimizer_state, inspect

PROTOCOL = 'docs/chain/pose-stereo-placement-protocol.md'
OUTPUT = 'outputs/pose_insurance/stereo-placement-scored'


def environment(runtime: dict) -> dict:
    """Keep the established GPU settings with caches and temporary files in task scratch."""
    heavy_execution(runtime)
    scratch = output_path(ROOT, '.runtime/tmp/t1-scored')
    for key in (*CACHE_FOLDERS, 'WARP_CACHE_PATH', 'JAX_COMPILATION_CACHE_DIR'):
        path = scratch / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(FC / 'src')]),
                      CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_ENABLE_COMPILATION_CACHE='true',
                      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0', WANDB_MODE='disabled',
                      HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1')
    return os.environ.copy()


def build(plan: dict, plan_path: str, request: Request) -> Path:
    """Change only slot-one XY in the exact v3 task, retaining all other task fields."""
    parent = producer(plan['base_build'], 'pose_insurance.build', request)
    record = run_fields(ROOT, 'pose_insurance', 'build', dict(plan=plan['identity'], candidate='T1'),
                        parents=[parent[0]], inputs=[PROTOCOL, plan_path, FEATURE / 'scored_stereo_queue.json',
                                                   FEATURE / 'runtime.json'])
    directory = output_path(ROOT, f'{OUTPUT}/builds/{new_run_id()}/episode_{request.episode_index:06d}')
    with recorded(directory, record):
        original = pq.read_table(artifact(parent, 'task.parquet', record))
        task = original.to_pylist()[0]
        previous = deepcopy(task)
        positions = np.asarray(task['object_body_position'], float)
        if not np.allclose(positions, np.array([r['position'] for r in plan['base_scene']])[None], rtol=0, atol=0):
            raise ValueError('Original task is not the exact v3 initial scene')
        positions[:, 1, :2] = plan['scene'][1]['position'][:2]
        task['object_body_position'] = positions.tolist()
        restored = deepcopy(task)
        restored['object_body_position'] = previous['object_body_position']
        if restored != previous:
            raise ValueError('T1 changed a non-position task field')
        pq.write_table(pa.Table.from_pylist([task], schema=original.schema), directory / 'task.parquet')
        record.update(split='evaluation', episode_index=request.episode_index, scene=plan['scene'],
                      object_names=parent[1]['object_names'], roster_frames=parent[1]['roster_frames'],
                      task_frames=parent[1]['task_frames'], dataset_revision=parent[1]['dataset_revision'],
                      candidate='T1', base_scene=parent[1]['scene'],
                      equivalence='Only slot-one XY changes in object_body_position; every other field is identical')
        write_json(directory / 'initial_scene.json', {k: record[k] for k in ('scene', 'object_names', 'task_frames')})
    return directory / 'run.json'


def run(plan_path: str, directory_name: str) -> Path:
    """Run the unchanged 512-step policy and official evaluator under a serial GPU lease."""
    runtime = runtime_flag(FEATURE, 'scored_scale_authorized', 'Scored stereo evaluator')
    plan = read_json(plan_path)
    request = Request(episode_index=plan['episode'], budget=plan['budget'])
    authorize_episode(runtime, request, 'scored stereo evaluator')
    env = environment(runtime)
    task_path = build(plan, plan_path, request)
    task = producer(task_path, 'pose_insurance.build', request)
    record = run_fields(ROOT, 'pose_insurance', 'run', dict(plan=plan['identity'], budget=request.budget, candidate='T1'),
                        parents=[task_path], inputs=[PROTOCOL, plan_path, FEATURE / 'runtime.json',
                        ROOT / 'outputs/pose_insurance/runtime_inventory.json', KIT / 'flash_chord/scripts/evaluate_policy.py'])
    record.update(split='evaluation', episode_index=request.episode_index, owner_pid=os.getpid(),
                  dataset_revision=task[1]['dataset_revision'], scene=task[1]['scene'], plan_identity=plan['identity'])
    directory = output_path(ROOT, directory_name)
    with recorded(directory, record):
        task_file = artifact(task, 'task.parquet', record)
        main_lock = ROOT.parents[2] / '.runtime/gpu-stage.lock'
        if main_lock.exists():
            raise RuntimeError(f'Main-checkout GPU lease exists: {main_lock}')
        started = time.perf_counter()
        with GpuLease(ROOT, 2048, 1024, stage='pose_insurance_t1_v4') as lease:
            try:
                execute(training_command(request, task_file, directory), directory / 'training.log', env, record)
                record.update(optimizer_state(directory))
                command = [str(ROOT / '.runtime/flashchord/venv/bin/python'), str(KIT / 'flash_chord/scripts/evaluate_policy.py'),
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
