"""Resume incomplete masks or search later frames for overlapping instances."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from src.gpu import GpuLease
from src.provenance import write_json,sha256_file
from src.runs import now_utc
from src.features.moving_reference import masks
from . import two_object_screen as s
from ._two_object_instances import mask_overlap,choose_pair
from .two_object_instances import OUTPUT,ROOT,run as frame0_run


def command(directory,operation,clip=False):
    executable='grounding-dino' if operation in ('detect','search') else 'sam2'
    module='src.features.moving_policy.two_object_clip_worker' if clip else 'src.features.moving_reference._worker'
    argv=[str(ROOT/f'data/.runtime/{executable}/venv/bin/python'),'-B','-m',module,operation,str(directory/'request.json')]
    with (directory/f'{operation}-{time.time_ns()}.log').open('x') as log:
        subprocess.run(argv,cwd=ROOT,env=masks.environment(directory),stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((directory/f'{"detect" if operation=="search" else operation}-report.json').read_text())
    if report['status']!='completed':
        raise ValueError(f'Incomplete {operation} worker')
    return report


def resume_interrupted(episode,names,queue):
    base=OUTPUT/'instance-separated'/f'ep{episode}'
    result=base/'result.json'
    if result.exists() or not (base/'once.json').exists():
        return frame0_run(episode,names,queue)
    if episode!=20:
        raise ValueError('Only episode 20 was interrupted without a completed instance result')
    directories=[base/f'masks-{slot}' for slot in (0,1)]
    selection=json.loads((base/'joint-boxes.json').read_text())
    if not selection['passed']:
        raise ValueError('A completed failed selection must not be resumed as an interruption')
    evidence=dict(episode=episode,started_utc=now_utc(),reason='lead interruption before both object masks completed',
                  retained=[],continued=[])
    with GpuLease(ROOT,1024,10000,stage='moving_reference.masks') as lease:
        for directory in directories:
            report=directory/'segment-report.json'
            if report.exists():
                saved=json.loads(report.read_text())
                if saved['status']!='completed':
                    raise ValueError('Interrupted partial segmentation requires a fresh explicit method attempt')
                evidence['retained'].append(dict(path=str(report.relative_to(ROOT)),sha256=sha256_file(report)))
            else:
                found=json.loads((directory/'detections.json').read_text())
                slot=directories.index(directory)
                write_json(directory/'detections-independent.json',found)
                found['pan']=dict(frame_index=0,**{k:v for k,v in selection['selected'][slot].items() if k!='area'})
                found['anchors']['0']=[0]
                write_json(directory/'detections.json',found)
                lease.require_headroom()
                command(directory,'segment')
                evidence['continued'].append(str(directory.relative_to(ROOT)))
            job=json.loads((directory/'request.json').read_text())
            record=dict(kind='moving_policy.interruption_resume',status='completed',episode=episode,completed_utc=now_utc(),
                        original_run_sha256=sha256_file(directory/'run.json'),wall_seconds=0.)
            write_json(directory/'run-resume-completed.json',record)
            masks.summarize(directory,job,record)
    with np.load(directories[0]/'frames/frame_000000.npz') as a,np.load(directories[1]/'frames/frame_000000.npz') as b:
        overlap=mask_overlap(a['masks'][0],b['masks'][0])
    final=dict(episode=episode,passed=overlap<.5,reused=False,box_iou=selection['box_iou'],frame0_mask_iou=overlap,
        directories=[str(p.relative_to(ROOT)) for p in directories],reason='distinct instances' if overlap<.5 else 'frame-0 masks have IoU >=0.5',
        interruption_resume=evidence)
    write_json(result,final)
    s.event(2,f'ep{episode}: interrupted masks resumed; {len(evidence["retained"])} completed segmentation retained; frame-0 IoU={overlap}')
    return final


def run(episode,names,queue):
    if episode not in (19,1):
        first=resume_interrupted(episode,names,queue)
        if first['passed'] or first.get('frame0_mask_iou') is not None:
            return first
    base=OUTPUT/'clip-instance-separated'/f'ep{episode}'
    result=base/'result.json'
    if result.exists():
        return json.loads(result.read_text())
    base.mkdir(parents=True,exist_ok=True)
    with (base/'once.json').open('x') as stream:
        json.dump(dict(episode=episode,started_utc=now_utc(),protocol_sha256=sha256_file(ROOT/'docs/chain/moving-two-object-protocol.md'),
                       method='first separated pair on five-frame cadence; bidirectional SAM object seed'),stream,indent=2)
    old=OUTPUT/'instance-separated'/f'ep{episode}'
    directories=[];detected=[]
    for slot in (0,1):
        directory=base/f'masks-{slot}';directory.mkdir();directories.append(directory)
        source=old/f'masks-{slot}'
        job=json.loads((source/'request.json').read_text())
        job.update(output=str(directory.relative_to(ROOT)),joint_names=names)
        # Prepared input JPEGs and their clocks are immutable and reusable.
        (directory/'input_frames').symlink_to((source/'input_frames').resolve(),target_is_directory=True)
        write_json(directory/'request.json',job)
        found=json.loads((source/'detections.json').read_text())
        write_json(directory/'detections.json',found)
        detected.append(found)
    selection=choose_pair(detected)
    selection['seed_frame']=0
    with GpuLease(ROOT,1024,10000,stage='moving_reference.masks') as lease:
        if not selection['passed']:
            command(directories[0],'search',clip=True)
            selection=json.loads((directories[0]/'clip-search.json').read_text())
        write_json(base/'joint-boxes.json',selection)
        if selection['passed']:
            for directory,found,chosen in zip(directories,detected,selection['selected']):
                found['pan']=dict(frame_index=selection['seed_frame'],**{k:v for k,v in chosen.items() if k!='area'})
                found['anchors']['0']=[selection['seed_frame']]
                write_json(directory/'detections.json',found)
                lease.require_headroom()
                segment=command(directory,'segment',clip=True)
                record=dict(kind='moving_policy.clip_instance_masks',status='completed',episode=episode,completed_utc=now_utc(),
                            stages=dict(segment=segment),seed_frame=selection['seed_frame'],wall_seconds=segment['wall_seconds'])
                write_json(directory/'run.json',record)
                masks.summarize(directory,json.loads((directory/'request.json').read_text()),record)
    overlap=None
    if selection['passed']:
        with np.load(directories[0]/'frames/frame_000000.npz') as a,np.load(directories[1]/'frames/frame_000000.npz') as b:
            overlap=mask_overlap(a['masks'][0],b['masks'][0])
    passed=selection['passed'] and overlap<.5
    final=dict(episode=episode,passed=passed,reused=False,box_iou=selection.get('box_iou'),frame0_mask_iou=overlap,
        seed_frame=selection.get('seed_frame'),directories=[str(p.relative_to(ROOT)) for p in directories],
        reason='distinct instances across clip' if passed else selection['reason'] if not selection['passed'] else 'frame-0 masks have IoU >=0.5')
    write_json(result,final)
    s.event(2,f'ep{episode}: clip instance {"PASS" if passed else "STOP"}; seed={final["seed_frame"]}; box IoU={final["box_iou"]}; mask IoU={overlap}')
    return final
