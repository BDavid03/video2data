"""Assign distinct frame-zero instances before unchanged SAM2 propagation."""

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
from ._two_object_instances import choose_pair,mask_overlap

ROOT=s.ROOT
OUTPUT=ROOT/'data/outputs/moving_policy/two-object'


def existing_pair(episode):
    directories=[OUTPUT/f'ep{episode}/masks-{slot}' for slot in (0,1)]
    if not all((p/'run.json').exists() and json.loads((p/'run.json').read_text())['status']=='completed'
               and (p/'frames/frame_000000.npz').exists() for p in directories):
        return None
    with np.load(directories[0]/'frames/frame_000000.npz') as a,np.load(directories[1]/'frames/frame_000000.npz') as b:
        overlap=mask_overlap(a['masks'][0],b['masks'][0])
    return dict(passed=overlap<.5,frame0_mask_iou=overlap,
                directories=[str(p.relative_to(ROOT)) for p in directories],
                reason='existing distinct masks retained' if overlap<.5 else 'existing masks do not separate instances')


def run(episode,names,queue):
    base=OUTPUT/'instance-separated'/f'ep{episode}'
    result=base/'result.json'
    if result.exists():
        return json.loads(result.read_text())
    base.mkdir(parents=True,exist_ok=True)
    previous=existing_pair(episode)
    if previous and previous['passed'] and episode not in (1,8):
        write_json(result,dict(previous,episode=episode,reused=True,
            frame0_masks=[dict(path=str((ROOT/d/'frames/frame_000000.npz').relative_to(ROOT)),
                              sha256=sha256_file(ROOT/d/'frames/frame_000000.npz')) for d in previous['directories']]))
        s.event(2,f'ep{episode}: existing distinct masks retained; frame-0 IoU={previous["frame0_mask_iou"]:.6g}')
        return json.loads(result.read_text())
    resuming=(base/'implementation-resume.json').exists()
    if not resuming:
        with (base/'once.json').open('x') as stream:
            json.dump(dict(episode=episode,started_utc=now_utc(),names=names,
                           protocol_sha256=sha256_file(ROOT/'docs/chain/moving-two-object-protocol.md'),
                           method='joint minimum box-IoU then smallest-box ties',previous_masks=previous),stream,indent=2)
    elif any(base.glob('masks-*/segment-report.json')):
        raise ValueError('The implementation repair may only reuse detection before any segmentation attempt')
    directories=[base/f'masks-{slot}' for slot in (0,1)]
    jobs=[];records=[]
    for name,directory in zip(names,directories):
        directory.mkdir(exist_ok=resuming)
        rule=json.loads((masks.FEATURE/'runtime.json').read_text())
        rule.update(queue=[dict(split='evaluation',episode=episode)],pan_prompts=[name.replace('_',' ')],
                    stop_utc=queue['stop_utc'])
        job=json.loads((directory/'request.json').read_text()) if resuming else masks.prepare(rule['queue'][0],rule,directory)
        job['pan_box_selection']='joint_minimum_iou_then_smallest_box'
        if not resuming:
            write_json(directory/'request.json',job)
        jobs.append(job)
        record=dict(kind='moving_policy.joint_instance_masks',status='running',episode=episode,object_name=name,
                    started_utc=now_utc(),stages={})
        records.append(record)
        write_json(directory/('run-resume.json' if resuming else 'run.json'),record)
    started=time.perf_counter()
    selection=None;lease=None
    try:
        with GpuLease(ROOT,1024,10000,stage='moving_reference.masks') as lease:
            for directory,record in zip(directories,records):
                lease.require_headroom()
                if not resuming:
                    command=[str(ROOT/'data/.runtime/grounding-dino/venv/bin/python'),'-B','-m',
                             'src.features.moving_reference._worker','detect',str(directory/'request.json')]
                    with (directory/'detect.log').open('x') as log:
                        subprocess.run(command,cwd=ROOT,env=masks.environment(directory),stdout=log,stderr=subprocess.STDOUT,check=True)
                record['stages']['detect']=json.loads((directory/'detect-report.json').read_text())
                if record['stages']['detect']['status']!='completed':
                    raise ValueError('Only completed unchanged detector outputs may be reused')
            detected=[json.loads((p/'detections.json').read_text()) for p in directories]
            selection=choose_pair(detected)
            write_json(base/'joint-boxes.json',selection)
            if not selection['passed']:
                for record in records:
                    record['status']='no_instance_separation'
            else:
                for slot,(directory,record,found,chosen) in enumerate(zip(directories,records,detected,selection['selected'])):
                    write_json(directory/'detections-independent.json',found)
                    found['pan']=dict(frame_index=0,**{k:v for k,v in chosen.items() if k!='area'})
                    found['anchors']['0']=[0]
                    found['pan_box_selection']='joint_minimum_iou_then_smallest_box'
                    write_json(directory/'detections.json',found)
                    lease.require_headroom()
                    command=[str(ROOT/'data/.runtime/sam2/venv/bin/python'),'-B','-m',
                             'src.features.moving_reference._worker','segment',str(directory/'request.json')]
                    with (directory/'segment.log').open('x') as log:
                        subprocess.run(command,cwd=ROOT,env=masks.environment(directory),stdout=log,stderr=subprocess.STDOUT,check=True)
                    record['stages']['segment']=json.loads((directory/'segment-report.json').read_text())
                    record['status']='completed'
    except BaseException as error:
        for record in records:
            if record['status']=='running':
                record.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for directory,job,record in zip(directories,jobs,records):
            record.update(wall_seconds=time.perf_counter()-started,completed_utc=now_utc(),gpu=lease.report() if lease else None)
            write_json(directory/('run-resume.json' if resuming else 'run.json'),record)
            masks.summarize(directory,job,record)
    overlap=None
    if selection['passed']:
        with np.load(directories[0]/'frames/frame_000000.npz') as a,np.load(directories[1]/'frames/frame_000000.npz') as b:
            overlap=mask_overlap(a['masks'][0],b['masks'][0])
    passed=selection['passed'] and overlap<.5
    final=dict(episode=episode,passed=passed,reused=False,box_iou=selection.get('box_iou'),frame0_mask_iou=overlap,
               directories=[str(p.relative_to(ROOT)) for p in directories],
               reason='distinct instances' if passed else selection['reason'] if not selection['passed'] else 'frame-0 masks have IoU >=0.5')
    write_json(result,final)
    s.event(2,f'ep{episode}: joint instance selection {"PASS" if passed else "STOP"}; box IoU={final["box_iou"]}; mask IoU={overlap}')
    return final
