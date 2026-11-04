"""Score sealed public trajectories with the unchanged kit's registered metric entrypoints."""
import argparse
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.paths import project_path,output_path
from src.provenance import sha256_file,write_json
from src.runs import now_utc
from . import depth_track as h,depth_hybrid as hybrid
from ._kit_score import pose_in_mesh_frame,pose_in_world_frame

ROOT=Path(__file__).resolve().parents[3]
KIT=ROOT/'data/.runtime/kit-20261102/v2d_submission_kit'
BASE=h.OUTPUT/'kit-score'
OUTPUT=BASE
METRICS={'AUC':'AUC.py','MPPE_cm':'MPPE.py','SP_SR':'SP-SR.py','MP_SR':'MP-SR.py'}


def read(path):
    return json.loads(path.read_text())


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


@lru_cache(maxsize=4)
def scorer(metric):
    result=module('local_official_'+metric,KIT/'metric_code/track_3'/METRICS[metric])
    if not result._REGISTER_OBJECT_MESH or not result._ALIGN_INITIAL_POSE:
        raise ValueError('Unexpected kit flags; do not silently change scorer mode')
    return result


def hashes():
    paths=[Path(__file__),Path(__file__).with_name('_kit_score.py'),
        ROOT/'src/features/pose_insurance/combined_score.py',KIT/'v2dlb/trajectory.py',
        KIT/'v2dlb/mesh_budget.py',KIT/'v2dlb/vertices.py']
    paths += [KIT/'metric_code/track_3'/f for f in METRICS.values()]
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in paths}


def prepare():
    """Discover only completed C/hybrid inputs, freezing a reusable editable score queue."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    groups={}
    for path in sorted(hybrid.OUTPUT.glob('ep*/C-alone/reference.npz')):
        if not (path.parent/'seal.json').exists():continue
        item=read(path.parent/'request.json')['item'];episode=item['episode'];h.authorize(episode)
        hp=path.parents[1]/'hybrid/reference.npz'
        if not hp.exists() or not (hp.parent/'seal.json').exists():continue
        hybrid.check(path.parent);hybrid.check(hp.parent)
        record=dict(name=item['object'],mesh=item['mesh']['path'],body_T_mesh=np.eye(4).tolist(),
                    C=str(path.relative_to(ROOT)),hybrid=str(hp.relative_to(ROOT)))
        groups.setdefault(episode,[]).append(record)
    cdir=ROOT/'data/outputs/moving_reference/ep39/candidate-c'
    with np.load(cdir/'reference.npz') as a:ref=a['world_T_object'].copy()
    with np.load(cdir/'tracking.npz') as a:asset=a['world_T_asset'].copy()
    transform=np.linalg.inv(ref[0])@asset[0]
    if not np.allclose(ref@transform,asset,atol=1e-10):raise ValueError('C39 mesh/body convention is not constant')
    groups[39]=[dict(name='mini_dust_pan',mesh=str((cdir/'asset/mini_dust_pan.obj').relative_to(ROOT)),
        body_T_mesh=transform.tolist(),C=str((cdir/'reference.npz').relative_to(ROOT)),
        hybrid=str((hybrid.OUTPUT/'ep39-mini_dust_pan/hybrid/reference.npz').relative_to(ROOT)))]
    for episode,items in groups.items():
        common_path=(cdir/'tracking.npz') if episode==39 else project_path(ROOT,items[0]['C']).with_name('tracking.npz')
        with np.load(common_path) as saved:shared_camera0=saved['world_T_camera'][0]
        for item in items:
            item['world_T_shared_world']={};item['world_frame_sources']=[str(common_path.relative_to(ROOT))]
            for variant in ('C','hybrid'):
                path=project_path(ROOT,item[variant]).with_name('tracking.npz')
                if episode==39:path=cdir/'tracking.npz'
                with np.load(path) as saved:camera0=saved['world_T_camera'][0]
                item['world_T_shared_world'][variant]=(shared_camera0@np.linalg.inv(camera0)).tolist()
                item['world_frame_sources'].append(str(path.relative_to(ROOT)))
    cases=[dict(episode=ep,objects=objects,allow_partial=True) for ep,objects in sorted(groups.items())]
    queue=dict(cases=cases,variants=['static','C','hybrid'],output=str(OUTPUT.relative_to(ROOT)),
        maximum_seconds=1800,reference_geometry='public_proxy_evaluation_only',static='hold C frame-zero pose; no oracle companion')
    path=OUTPUT/'queue.json'
    if path.exists():raise FileExistsError('Preserve the frozen scoring queue')
    write_json(path,queue)
    freeze(path)
    return path


def freeze(path):
    """Seal an arbitrary local score request without overwriting another measurement."""
    queue=read(path);OUTPUT.mkdir(parents=True,exist_ok=True)
    if not (OUTPUT/'protocol.md').exists():
        protocol=BASE/'shared-world/protocol.md'
        if not protocol.exists():protocol=BASE/'protocol.md'
        (OUTPUT/'protocol.md').write_bytes(protocol.read_bytes())
    inputs={}
    for case in queue['cases']:
        h.authorize(case['episode'])
        for item in case['objects']:
            keys=set(['C','mesh',*[v for v in queue['variants'] if v!='static']])
            for key in keys:
                p=project_path(ROOT,item[key]);inputs[str(p.relative_to(ROOT))]=sha256_file(p)
            for value in item.get('world_frame_sources',[]):
                p=project_path(ROOT,value);inputs[str(p.relative_to(ROOT))]=sha256_file(p)
    target=OUTPUT/'scoring-inputs-seal.json'
    if target.exists():
        previous=read(target)
        if previous['queue_sha256']!=sha256_file(path) or previous['sources']!=hashes() or previous['inputs']!=inputs:
            raise ValueError('Existing measurement changed; use a fresh output directory')
        return
    write_json(OUTPUT/'scoring-inputs-seal.json',dict(utc=now_utc(),queue_sha256=sha256_file(path),
        protocol_sha256=sha256_file(OUTPUT/'protocol.md'),sources=hashes(),inputs=inputs,
        reference_geometry_access='only in scoring; never in tracking',labels_read_for_scoring=False))


def budget_candidate(path):
    """Call the existing kit budgeter in its read-only installed environment."""
    digest=sha256_file(path);directory=OUTPUT/'candidate-cache';directory.mkdir(exist_ok=True)
    target=directory/f'{digest}.npz'
    if not target.exists():
        code=('import sys,numpy as np; from v2dlb.mesh_budget import budget_mesh; '
              'v,f=budget_mesh(sys.argv[1],4096,4096); np.savez_compressed(sys.argv[2],vertices=v,faces=f)')
        env=dict(os.environ,PYTHONPATH=str(KIT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        subprocess.run([str(ROOT/'data/.runtime/kit-env/bin/python'),'-B','-c',code,str(path),str(target)],
                       cwd=ROOT,env=env,check=True)
    with np.load(target) as saved:return saved['vertices'].copy(),saved['faces'].copy()


@lru_cache(maxsize=16)
def reference_geometry(name):
    """Use the existing disclosed public-geometry proxy solely inside this evaluator."""
    sys.path.insert(0,str(KIT)) if str(KIT) not in sys.path else None
    from src.features.pose_insurance import combined_score as public
    public.SCORER=scorer('AUC')
    geometry=public.geometry(name)
    vertices,faces=geometry['mesh']
    # Weld exact duplicates without changing any triangle, material or physical surface.
    unique,inverse=np.unique(vertices,axis=0,return_inverse=True)
    mesh=(unique,inverse[faces])
    return dict(geometry,mesh=mesh,original_vertex_count=len(vertices),welded_vertex_count=len(unique))


def solution_rows(episode,names,reference,frame_index,geometry):
    """Assemble the kit's rectangular per-episode reference rows and geometry payload."""
    count,bodies=reference.shape[:2]
    arrays=dict(names=np.asarray(names))
    for i,g in enumerate(geometry):
        arrays.update({f'v{i}':g['mesh'][0],f'f{i}':g['mesh'][1],f'anchor{i}':g['anchor'],
                       f'points{i}':g['points'],f'axis{i}':g['axis'],f'order{i}':np.asarray(g['order'])})
    blob=scorer('AUC')._encode_archive(**arrays)
    # Verify the host payload size limit before any expensive scoring call.
    scorer('AUC')._decode_archive(blob,16_000_000)
    rows=pd.DataFrame(dict(row_id=[f'local_t3_{episode:06d}_{int(f):06d}_{s}' for f in frame_index for s in range(bodies)],
        episode_index=episode,world_index=0,frame_index=np.repeat(frame_index,bodies),
        object_slot=np.tile(np.arange(bodies),count),object_name=np.tile(names,count)))
    rows[scorer('AUC')._REF_COLUMNS]=reference.reshape(-1,7)
    rows['reference_geometry_b64']='';rows.loc[0,'reference_geometry_b64']=blob
    return rows


def submission_rows(solution,pose,candidates,base_commit):
    """Emit local-only diagnostic rows; these provenance markers are not policy claims."""
    rows=solution[['row_id','episode_index','world_index','frame_index','object_slot','object_name']].copy()
    rows[scorer('AUC')._POSE_COLUMNS]=pose.reshape(-1,7)
    for key in scorer('AUC')._PROVENANCE_COLUMNS:
        rows[key]='local-diagnostic-not-policy'
    rows['code_commit_url']='https://github.com/BDavid03/video2data/commit/'+base_commit
    rows['object_mesh_b64']=''
    for i,(v,f) in enumerate(candidates):
        rows.loc[i,'object_mesh_b64']=scorer('AUC')._encode_archive(vertices=v,faces=f)
    return rows


def score_case(case,variant,base_commit):
    """Call all four unchanged public score() entrypoints for one episode/variant."""
    sys.path.insert(0,str(KIT)) if str(KIT) not in sys.path else None
    from v2dlb.trajectory import read_episode
    episode=case['episode'];h.authorize(episode)
    label=h.snapshot()/f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet'
    truth=read_episode(label,episode)
    lookup={item['name']:item for item in case['objects']}
    missing=[n for n in truth.object_names if n not in lookup]
    if missing and not case.get('allow_partial',False):
        raise ValueError(f'Missing object trajectories: {missing}')
    names=[n for n in truth.object_names if n in lookup]
    if not names:raise ValueError('No tracked object matches this episode')
    reference=truth.pose_xyzw[:,[truth.object_names.index(n) for n in names]]
    achieved=[];candidates=[];geometry=[];input_hashes={str(label.relative_to(ROOT)):sha256_file(label)}
    for name in names:
        item=lookup[name];path=project_path(ROOT,item['C' if variant=='static' else variant])
        with np.load(path,allow_pickle=False) as saved:
            pose=np.c_[saved['position'],saved['quaternion_xyzw']];frames=saved['frame_index']
        if np.any(np.diff(frames)<=0):raise ValueError('Trajectory frame IDs must strictly increase')
        if variant=='static' and frames[0]!=0:raise ValueError('Static floor needs the actual frame-zero pose')
        initial=pose[0].copy()
        selected=np.searchsorted(frames,truth.frame_index)
        if np.any(selected>=len(frames)) or not np.array_equal(frames[selected],truth.frame_index):
            raise ValueError('Trajectory does not cover the reference frame grid')
        pose=pose[selected]
        if variant=='static':pose=np.repeat(initial[None],len(pose),axis=0)
        source_variant='C' if variant=='static' else variant
        pose=pose_in_world_frame(pose,np.asarray(item.get('world_T_shared_world',{}).get(source_variant,np.eye(4))))
        pose=pose_in_mesh_frame(pose,np.asarray(item.get('body_T_mesh',np.eye(4))))
        achieved.append(pose)
        mesh=project_path(ROOT,item['mesh'])
        if mesh.is_relative_to(h.snapshot()):raise PermissionError('Reference scans cannot be candidate meshes')
        candidates.append(budget_candidate(mesh));geometry.append(reference_geometry(name))
        input_hashes.update({str(path.relative_to(ROOT)):sha256_file(path),str(mesh.relative_to(ROOT)):sha256_file(mesh)})
    achieved=np.stack(achieved,axis=1)
    directory=OUTPUT/f'ep{episode}'/variant;directory.mkdir(parents=True,exist_ok=True)
    solution=solution_rows(episode,names,reference,truth.frame_index,geometry)
    submission=submission_rows(solution,achieved,candidates,base_commit)
    solution.to_parquet(directory/'local-solution.parquet',index=False)
    submission.to_parquet(directory/'local-submission.parquet',index=False)
    write_json(directory/'request.json',dict(episode=episode,variant=variant,objects=names,missing_objects=missing,
        sources=hashes(),inputs=input_hashes,geometry=[dict(name=n,input=g['input'],anchor=g['anchor'].tolist(),
        axis=g['axis'].tolist(),order=g['order'],original_vertices=g['original_vertex_count'],
        welded_vertices=g['welded_vertex_count']) for n,g in zip(names,geometry)],
        submission_eligible=False,provenance='literal local-diagnostic-not-policy markers; base commit only; adapter source hash is recorded',utc=now_utc()))
    started=time.perf_counter();metrics={}
    for metric in METRICS:
        metrics[metric]=float(scorer(metric).score(solution.copy(),submission.copy(),'row_id'))
        print(json.dumps(dict(episode=episode,variant=variant,metric=metric,value=metrics[metric])),flush=True)
    row=dict(episode=episode,variant=variant,objects=names,missing_objects=missing,complete_episode=not missing,
        frames=len(reference),label_frames_dropped=truth.dropped_frames,metrics=metrics,
        full_episode_metrics={key:(None if missing and key in ('MP_SR','SP_SR') else value) for key,value in metrics.items()},
        wall_seconds=time.perf_counter()-started,geometry_scope='public proxy; host geometry unavailable',
        request_sha256=sha256_file(directory/'request.json'))
    write_json(directory/'result.json',row)
    return row


def publish(rows):
    lines=['# Unchanged-kit local registered-pose scores','',
        'Public-geometry proxy, not confidential leaderboard scores. Calls unchanged metric score() entrypoints with mesh registration and scene alignment enabled. '
        'Reference anchors are public surface centroids and physical cup/lid axes are public-surface PCA estimates. '
        'No reference geometry is used by any tracker. Static holds each C frame-zero pose. '
        'All predictions, including tracker-invalid frames, are scored. One world; kit SP-SR/MP-SR are episode/world success indicators, not frame fractions.','',
        '| Episode | Scope | Variant | Frames | AUC | MPPE cm | SP-SR | MP-SR | Seconds |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(rows,key=lambda x:(x['episode'],{'static':0,'C':1,'hybrid':2}.get(x['variant'],3),x['variant'])):
        m=row['full_episode_metrics'];scope='all objects' if row['complete_episode'] else 'subset: '+', '.join(row['objects'])
        fmt=lambda value:'N/A' if value is None else f'{value:.6f}'
        lines.append(f"| {row['episode']} | {scope} | {row['variant']} | {row['frames']} | "+
                     ' | '.join(fmt(m[k]) for k in ('AUC','MPPE_cm','SP_SR','MP_SR'))+f" | {row['wall_seconds']:.2f} |")
    lines += ['', 'Subset AUC/MPPE are conditional object-only diagnostics; the whole episode is not ranked when a companion is missing. '
        'The numeric subset SP/MP outcomes remain in result.json solely for debugging. '
        'Episode 11 has duplicate mask streams and is semantically ambiguous. '
        'Local row provenance is explicitly non-policy, non-upload metadata; passing the score parser does not grant submission eligibility.',
        '', 'Frame handling: the kit merges matching row IDs and sorts frame_index/world_index/object_slot; it ignores timestamps. '
        'Its reference reader drops frames only when an object is invisible, preserving original frame indices. '
        'Native dataset object order is retained; it affects registration seeds and scene anchoring.']
    (OUTPUT/'summary.md').write_text('\n'.join(lines)+'\n')
    write_json(OUTPUT/'results.json',dict(rows=rows,utc=now_utc(),submission_eligible=False))


def run(queue_path):
    queue=read(queue_path);output_path(ROOT,OUTPUT)
    seal=read(OUTPUT/'scoring-inputs-seal.json')
    if seal['queue_sha256']!=sha256_file(queue_path) or seal['sources']!=hashes():
        raise ValueError('Scoring request/source changed after freeze')
    for name,digest in seal['inputs'].items():
        if sha256_file(project_path(ROOT,name))!=digest:raise ValueError('Sealed trajectory/mesh changed')
    base_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    ledger=OUTPUT/'ledger.jsonl'
    rows=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    started=time.monotonic()
    for case in queue['cases']:
        for variant in queue['variants']:
            if any(r['episode']==case['episode'] and r['variant']==variant for r in rows):continue
            if time.monotonic()-started>queue['maximum_seconds']:
                publish(rows);raise TimeoutError('Bounded scoring window ended; resume the same queue')
            row=score_case(case,variant,base_commit)
            with ledger.open('a') as stream:stream.write(json.dumps(row)+'\n')
            rows.append(row);publish(rows)
    if hashes()!=seal['sources']:raise ValueError('Kit or adapter source changed during scoring')
    write_json(OUTPUT/'finished.json',dict(utc=now_utc(),cases=len(rows),sources=hashes()))


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','run','score'])
    parser.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:OUTPUT=output_path(ROOT,args.output)
    if args.request:
        args.request=project_path(ROOT,args.request)
        if not args.output:OUTPUT=output_path(ROOT,read(args.request)['output'])
    else:args.request=OUTPUT/'queue.json'
    if not OUTPUT.is_relative_to(ROOT/'data/outputs'):
        raise PermissionError('Score artifacts must stay under data/outputs')
    if args.operation=='prepare':print(prepare())
    else:
        if args.operation=='score':freeze(args.request)
        run(project_path(ROOT,args.request))


if __name__=='__main__':main()
