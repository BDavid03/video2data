"""Train each ready joint policy once, export officially, admit and pack cumulatively."""

from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq

from src.gpu import GpuLease
from src.provenance import write_json,sha256_file
from src.runs import now_utc
from src.features.pose_insurance import pack_moving as m
from src.features.pose_insurance._moving import tracking_metrics
from . import handler as h
from ._two_object_rotation import tracking as symmetry_tracking,resting as symmetry_resting
from .two_object_rotation import mesh_axes
from .two_object_train_worker import __file__ as PASSIVE_WORKER
from .two_object_admission import inspect as admit, disclosure
from .two_object_campaign import OUTPUT,CAMPAIGN,CUTOFF,s,read_control
from .two_object_contract import require_stage0,require_resting_start,wait_preparer

FINAL=66763776


def queue_update(episode,**fields):
    path=OUTPUT/'training-queue.json'
    queue=json.loads(path.read_text())
    rows=queue['training']
    match=next((r for r in rows if r['episode']==episode),None)
    if match is None:
        match=dict(episode=episode)
        rows.append(match)
    match.update(fields)
    if fields.get('status')=='training':
        queue['ready']=[r for r in queue['ready'] if r['episode']!=episode]
    queue['updated_utc']=now_utc()
    write_json(path,queue)
    return queue


def measure(directory,active,binding):
    frame=pq.read_table(directory/f'score-{FINAL}/trajectory.parquet').to_pandas().sort_values(['reference_frame','object_body_id'])
    achieved=frame[['achieved_'+c for c in m.POSE]].to_numpy(float).reshape(-1,2,7)
    task=pq.ParquetFile(directory/f'motion/sequence_id=episode_{binding["episode"]:06d}/robot_name=sharpa_wave/task.parquet').read().to_pylist()[0]
    reference=np.concatenate([np.asarray(task['object_body_position']),np.asarray(task['object_body_wxyz'])[...,[1,2,3,0]]],axis=-1)
    names=task['object_body_names'];axes=mesh_axes(task['object_mesh_paths'])
    return dict(tracking_A=symmetry_tracking(achieved,reference,names,axes)[active],
                resting_B=symmetry_resting(achieved[:,1-active],binding['objects'][1-active]['pose_wxyz'],names[1-active],axes[1-active]))


def run(episode,directory,consume_ready=True):
    review=m.read(h.ROOT/'data/outputs/moving_policy/two-object/review-fixes-status.json')
    if review.get('passed') is not True:
        raise PermissionError('Joint review fixes have not passed or loaded; admission/training held')
    directory=m.path(directory)
    result_file=directory/'policy-result.json'
    if result_file.exists():
        q=json.loads((OUTPUT/'training-queue.json').read_text())
        q['ready']=[r for r in q['ready'] if r['episode']!=episode]
        write_json(OUTPUT/'training-queue.json',q)
        return m.read(result_file)
    if datetime.now(timezone.utc)>=CUTOFF:
        result=dict(episode=episode,status='NOT STARTED: after 19:45 UTC',admitted=False)
        queue_update(episode,**{k:v for k,v in result.items() if k!='episode'})
        return result
    if (directory/'train.run.json').exists():
        raise FileExistsError('One training attempt already exists; no seed/restart retry')
    stage=m.read(directory/'stage0.json')
    actual_task,_=require_stage0(stage,directory,episode)
    require_resting_start(stage,actual_task)
    h.OUTPUT=directory
    h.TASK=directory/f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
    build=m.read(directory/'build.json')
    active=stage['active_slot']
    command=h.training_argv(build)
    index=command.index('src.features.dynamic_feasibility._worker')
    command[index:index+1]=['src.features.moving_policy.two_object_train_worker','--active-slot',str(active)]
    from src.features.dynamic_feasibility.handler import EVALUATOR
    score=directory/f'score-{FINAL}'
    evaluator=[str(h.PYTHON),'-B','-m','src.features.dynamic_feasibility._worker',str(EVALUATOR),
        f'evaluation.checkpoint={directory / "training" / f"policy_{FINAL}.safetensors"}',
        'evaluation.world_count=1','evaluation.reset_mode=explicit','evaluation.start_frame=0',
        f'evaluation.object_trajectories_output={score / "trajectory.parquet"}',
        f'evaluation.metrics_output={score / "official_metrics.json"}',f'hydra.run.dir={score / "hydra"}','logging.mode=disabled']
    write_json(directory/'commands.json',dict(train=command,evaluator_final=evaluator,seed=0,final_steps=FINAL,maximum_seconds=4500))
    record=dict(status='running',operation='train',pid=__import__('os').getpid(),argv=command,started_unix=time.time(),
                stage0_sha256=sha256_file(directory/'stage0.json'),passive_body_worker_sha256=sha256_file(PASSIVE_WORKER))
    with (directory/'train.run.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    queue_update(episode,status='training',start_utc=now_utc(),policy=str(directory.relative_to(h.ROOT)))
    s.event(3,f'ep{episode}: training START; seed 0, 1024 worlds, ratio 2, {FINAL} transitions, scheduled final checkpoint')
    cpu=None
    try:
        with GpuLease(h.ROOT,1024,6000,stage='moving-joint-train') as lease:
            cpu_log=(directory/'cpu-existing.log').open('x')
            cpu_env=h.environment(cpu=True)
            cpu=subprocess.Popen([str(h.ROOT/'data/.runtime/linux-cuda/bin/python'),'-B','-m',
                'src.features.moving_policy.two_object_existing'],cwd=h.ROOT,env=cpu_env,stdout=cpu_log,stderr=subprocess.STDOUT)
            cpu_log.close()
            write_json(directory/'cpu-existing-process.json',dict(pid=cpu.pid,log=str((directory/'cpu-existing.log').relative_to(h.ROOT))))
            env={**h.environment(cpu=False),'DYNAMIC_FEASIBILITY_DEADLINE':str(time.time()+4500)}
            with (directory/'train.log').open('x') as log:
                trained=subprocess.run(command,cwd=h.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            lease.require_headroom()
            record.update(status='completed' if trained.returncode==0 else 'failed',exit_code=trained.returncode,
                          finished_unix=time.time(),gpu=lease.report())
        write_json(directory/'train.run.json',record)
        queue_update(episode,status=record['status'],end_utc=now_utc())
        s.event(3,f'ep{episode}: training END status={record["status"]}; code={record["exit_code"]}')
        wait_preparer(cpu,directory)
        cpu=None
        if trained.returncode:
            raise RuntimeError(f'Training failed with exit {trained.returncode}; no retry')
        with GpuLease(h.ROOT,1024,6000,stage='moving-joint-official-evaluation') as lease:
            env={**h.environment(cpu=False),'DYNAMIC_FEASIBILITY_DEADLINE':str(time.time()+900)}
            with (directory/'official-evaluator.log').open('x') as log:
                scored=subprocess.run(evaluator,cwd=h.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(directory/'score-self.run.json',dict(status='completed' if scored.returncode==0 else 'failed',
                argv=evaluator,exit_code=scored.returncode,gpu=lease.report(),stage0_sha256=sha256_file(directory/'stage0.json'),
                checkpoint=m.entry(directory/'training'/f'policy_{FINAL}.safetensors'),
                trajectory=m.entry(score/'trajectory.parquet') if scored.returncode==0 else None,
                metrics=m.entry(score/'official_metrics.json') if scored.returncode==0 else None))
        if scored.returncode:
            raise RuntimeError('Official evaluator failed; no export splice or retry')
        diagnostics=measure(directory,active,stage['binding'])
        write_json(directory/'policy-measurements.json',diagnostics)
        physical=admit(directory,episode,require_disclosure=False)
        write_json(directory/'physical-admission.json',physical)
        s.event(3,f'ep{episode}: physical/reference-policy admission PASS; {diagnostics}')
        try:
            admission=dict(physical,abstract_disclosure=disclosure('docs/report/track3/main.tex',episode,physical['names'],active),admitted=True)
        except ValueError as error:
            if not str(error).startswith('Report disclosure missing:'):
                raise
            result=dict(episode=episode,status='PHYSICAL PASS; REPORT DISCLOSURE PENDING',admitted=False,
                        measurements=diagnostics,reason=str(error),policy=str(directory.relative_to(h.ROOT)))
            write_json(result_file,result)
            queue_update(episode,status=result['status'],admission_numbers=diagnostics)
            s.event(3,f'ep{episode}: report disclosure pending; no candidate adoption yet')
        else:
            admission_file=directory/'admission.json'
            write_json(admission_file,admission)
            result=dict(episode=episode,status='ADMITTED',admitted=True,measurements=diagnostics,admission=m.entry(admission_file))
            write_json(result_file,result)
            q=queue_update(episode,status='ADMITTED',admission=m.entry(admission_file),admission_numbers=diagnostics)
            q['admitted'].append(dict(episode=episode,admission=m.entry(admission_file),admitted_utc=now_utc()))
            write_json(OUTPUT/'training-queue.json',q)
            from src.features.pose_insurance.pack_combined_joint import pack_cumulative
            packed=pack_cumulative([r['admission']['path'] for r in q['admitted']])
            s.event(4,f'candidate PACKED: {packed}')
    except Exception as error:
        if record.get('status')=='running':
            record.update(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
            write_json(directory/'train.run.json',record)
        result=dict(episode=episode,status='NOT ADMITTED',admitted=False,reason=f'{type(error).__name__}: {error}')
        if (directory/'policy-measurements.json').exists():
            result['measurements']=m.read(directory/'policy-measurements.json')
        write_json(result_file,result)
        queue_update(episode,status=result['status'],end_utc=now_utc(),reason=result['reason'])
        s.event('STOP',f'ep{episode}: {result["reason"]}; no training retry')
    finally:
        wait_preparer(cpu,directory)
    if consume_ready and datetime.now(timezone.utc)<CUTOFF:
        ready=[]
        for path in CAMPAIGN.glob('ep*/result.json'):
            row=m.read(path)
            policy=path.parent/'policy'
            if row.get('passed') and not (policy/'train.run.json').exists():
                ready.append((row.get('completed_utc',''),row['episode'],policy))
        for _,other,policy in sorted(ready):
            while read_control()['pause']:
                time.sleep(15)
            if other in read_control()['queue']:
                run(other,policy,consume_ready=False)
    return result
