"""Bind actual joint assets, saved checkpoint configuration and evaluator commands."""

import json
import os
from pathlib import Path
import subprocess

from src.provenance import sha256_file,write_json
from src.runs import now_utc
from src.features.pose_insurance import pack_moving as m
from . import handler as h
from ._two_object_contract import require_stage_checks,require_recipe

EP38=h.ROOT/'data/outputs/moving_policy/ep38-v1/training/policy_66763776.safetensors'


def require_stage0(stage,directory,episode):
    require_stage_checks(stage,episode)
    artifacts=stage.get('artifacts',[])
    for artifact in artifacts:m.checked(artifact)
    task=directory/f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
    support=directory/f'reconstructed_stage/episode_{episode:06d}_sharpa_wave_support.usda'
    if (m.entry(task) not in artifacts or m.entry(support) not in artifacts
            or m.entry(directory/'reference-task.npz') not in artifacts):
        raise ValueError('Actual joint task and support asset must be hash-bound by Stage0')
    return task,support


def resolved_config(checkpoint):
    code='''import json,sys
from safetensors import safe_open
with safe_open(sys.argv[1],framework="numpy") as f:
 print(f.metadata()["user.resolved_config_json"])
'''
    text=subprocess.check_output([str(h.ROOT/'data/.runtime/flashchord/venv/bin/python'),'-B','-c',code,str(checkpoint)],
        cwd=h.ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},text=True)
    return json.loads(text)


def checkpoint_recipe(checkpoint):
    actual=resolved_config(checkpoint);expected=resolved_config(EP38)
    return dict(resolved_recipe=require_recipe(actual,expected),episode38_checkpoint=m.entry(EP38))


def evaluator_record(directory,checkpoint,trajectory,official,stage0):
    record_file=directory/'score-self.run.json';record=m.read(record_file)
    if (record.get('status')!='completed' or record.get('exit_code')!=0
            or record.get('stage0_sha256')!=sha256_file(stage0)):
        raise ValueError('Completed evaluator command must bind the admitted Stage0')
    command=record.get('argv',[])
    if not any(m.path(arg)==m.KIT/'flash_chord/scripts/evaluate_policy.py'
               for arg in command if str(arg).endswith('/evaluate_policy.py')):
        raise ValueError('Recorded evaluator is not the unedited official evaluator')
    args=m.overrides(command)
    wanted={'evaluation.checkpoint':checkpoint,'evaluation.object_trajectories_output':trajectory,'evaluation.metrics_output':official}
    if (any(key not in args or m.path(args[key])!=path for key,path in wanted.items())
            or args.get('evaluation.reset_mode')!='explicit' or args.get('evaluation.world_count')!='1'
            or args.get('evaluation.start_frame')!='0'):
        raise ValueError('Evaluator command does not bind final checkpoint, outputs and full no-reset rollout')
    for key,path in [('checkpoint',checkpoint),('trajectory',trajectory),('metrics',official)]:
        if record.get(key)!=m.entry(path):raise ValueError('Evaluator completed record output/checkpoint hash mismatch')
    return m.entry(record_file)


def wait_preparer(process,directory):
    if process is None:return
    code=process.wait()
    target=directory/'cpu-existing-completed.json'
    if not target.exists():write_json(target,dict(pid=process.pid,exit_code=code,status='completed' if code==0 else 'failed',finished_utc=now_utc()))
    if code!=0:raise RuntimeError(f'CPU preparer PID{process.pid} failed with exit {code}; no further preparation/admission')


def require_resting_start(stage,task):
    import numpy as np
    import pyarrow.parquet as pq
    from .two_object_task import packed_starts
    binding=stage['binding'];resting=1-stage['active_slot']
    names=[o['name'] for o in binding['objects']]
    starts,source=packed_starts(stage['episode'],names)
    row=pq.ParquetFile(task).read().to_pylist()[0]
    positions=np.asarray(row['object_body_position']);quaternions=np.asarray(row['object_body_wxyz'])
    intended=np.asarray(binding['objects'][resting]['pose_wxyz'])
    if (binding.get('resting_start_source')!=source or not np.array_equal(intended,starts[resting])
            or not np.array_equal(positions[:,resting],np.tile(intended[:3],(len(positions),1)))
            or not np.array_equal(quaternions[:,resting],np.tile(intended[3:],(len(positions),1)))):
        raise ValueError('Old or changed resting start: rebuild untrained task from moving-v5 achieved pose')
    return source
