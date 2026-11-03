"""Frozen G2 queue: video-only admissions, development gates, then a conditional packed veto."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from track3.paths import PROJECT as ROOT, output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.runs import CACHE_FOLDERS, now_utc, runtime_flag

FEATURE=Path(__file__).parent
OUT='outputs/mesh_generative/g2'
QUEUE='track3/features/mesh_generative/g2_queue.json'


def authorize():
    """Admit G2 and direct every cache and temporary file into the checkout."""
    runtime=runtime_flag(FEATURE,'g2_authorized','G2 mesh campaign')
    if Path(sys.executable).resolve()!=project_path(ROOT,runtime['python']):
        raise RuntimeError('Use the main project interpreter for G2 orchestration')
    for key,folder in CACHE_FOLDERS.items():
        folder='.runtime/tmp/g1' if key in ('TMP','TEMP','TMPDIR') else folder
        path=output_path(ROOT,folder); path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
    tempfile.tempdir=os.environ['TMPDIR']; sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']
    os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
                      PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'upstream/v2d_submission_kit/v2d_submission_kit'))
    return runtime


def key(pair):
    return f"{pair['source_split']}-{pair['episode']:06d}-{pair['object']}"


def cpu(operation,job,directory):
    """Call the kit-interpreter worker with a persistent request and log."""
    request=directory/f'{operation}-request.json'; write_json(request,job)
    with (directory/f'{operation}.log').open('w') as log:
        subprocess.run([str(ROOT/'.runtime/kit-env/bin/python'),'-B','-m','track3.features.mesh_generative.g2_cpu',
                        operation,'--request',str(request)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def prepare(pair,directory):
    """Freeze the assigned camera-A crop and its valid frame-zero observed depth cloud."""
    import cv2
    import numpy as np
    from ._geometry import masked_square
    from track3.features.mesh_extents._geometry import camera_points

    parent=ROOT/pair['measurement']; measure=json.loads(parent.read_text())
    if (measure['status']!='completed' or measure['provenance']!='video_only'
        or measure['split']!=pair['source_split'] or measure['episode_index']!=pair['episode']
        or measure['frames'][0]['frame_index']!=0): raise ValueError('Not a matching frame-zero measurement')
    assignment=json.loads((ROOT/'track3/features/mesh_extents/assignments.json').read_text())['episodes'][f"{pair['source_split']}:{pair['episode']}"]
    ids=assignment['objects'][pair['object']]
    if measure['request']['objects'][pair['object']]!=ids: raise ValueError('Assignment changed')
    video=next(e for e in measure['inputs'] if '/observation.images.ego_cam_a/' in e['path'])
    depth_entry=next(e for e in measure['inputs'] if '/depth/' in e['path'] and e['path'].endswith('run.json'))
    if sha256_file(ROOT/depth_entry['path'])!=depth_entry['sha256']: raise ValueError('Depth producer changed')
    depth_path=ROOT/depth_entry['path']; depth=json.loads(depth_path.read_text())
    frame=next(f for f in depth['frames'] if f['frame_index']==0 and f['status']=='completed')
    files=[(parent,None),(parent.parent/'labels_000000.npz',measure['artifacts']['labels_000000.npz']),
           (parent.parent/'overlay_000000.png',measure['artifacts']['overlay_000000.png']),
           (depth_path,None),(depth_path.parent/'preprocessing.npz',depth['preprocessing_artifact_sha256']),
           (depth_path.parent/frame['artifact'],frame['sha256'])]
    inputs=[]
    for path,digest in files:
        entry=input_file(ROOT,path)
        if digest and entry['sha256']!=digest: raise ValueError(f'Producer artifact changed: {path}')
        inputs.append(entry)
    with np.load(files[1][0],allow_pickle=False) as a: mask=np.isin(a['label'],np.array(ids)+1)
    image=cv2.imread(str(files[2][0])); rgb=cv2.cvtColor(image[:,:image.shape[1]//2],cv2.COLOR_BGR2RGB)
    crop,bounds=masked_square(rgb,mask)
    if (directory/'crop.png').exists():
        if not np.array_equal(cv2.cvtColor(cv2.imread(str(directory/'crop.png')),cv2.COLOR_BGR2RGB),crop): raise ValueError('Frozen crop changed')
    else: cv2.imwrite(str(directory/'crop.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
    with np.load(files[4][0],allow_pickle=False) as prep,np.load(files[5][0],allow_pickle=False) as frame:
        z=frame['depth_m'].astype(float)
        valid=mask & prep['input_valid'] & frame['valid'] & frame['input_valid'] & np.isfinite(z) & (z>0)
        observed=camera_points(z,prep['input_intrinsics_pixels'])[valid]
    if not (directory/'observed.npz').exists(): np.savez_compressed(directory/'observed.npz',points=observed)
    inputs += [input_file(ROOT,directory/name) for name in ('crop.png','observed.npz')]
    return dict(inputs=inputs,cluster_ids=ids,mask_pixels=int(mask.sum()),observed_points=len(observed),
                crop_bounds=bounds,source_video=video)


def append(ledger,row):
    """Append one durable completed stage record."""
    with ledger.open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def sealed_admissions(queue,directory):
    """Verify and reuse sealed decisions without rewriting their historical record."""
    path=directory/'admissions.json'
    result=json.loads(path.read_text())
    if (result['status']!='completed' or result['queue_sha256']!=sha256_file(ROOT/QUEUE)
        or json.loads((ROOT/QUEUE).read_text())!=queue):
        raise ValueError('Sealed admission queue changed')
    rows=result['pairs']
    ledger=[r for line in (directory/'ledger.jsonl').read_text().splitlines()
            if (r:=json.loads(line)).get('stage')=='admission']
    if rows!=ledger or len(rows)!=len(queue['pairs']): raise ValueError('Sealed admission decisions changed')
    for pair,row in zip(queue['pairs'],rows):
        if row['key']!=key(pair) or any(row.get(k)!=v for k,v in pair.items()):
            raise ValueError('Sealed admission pair differs from queue')
        if row['status']!='completed' or sha256_file(ROOT/row['v4_mesh'])!=row['v4_mesh_sha256']:
            raise ValueError('Sealed v4 mesh changed')
        expected='v4'
        if pair['measurement'] is not None:
            if sha256_file(ROOT/row['generated_mesh'])!=row['generated_mesh_sha256']:
                raise ValueError('Sealed generated mesh changed')
            if row['generated_residual_m']<=1.5*row['v4_residual_m']: expected='generated'
        selected=row['generated_mesh'] if expected=='generated' else row['v4_mesh']
        if row['decision']!=expected or row['selected_mesh']!=selected:
            raise ValueError('Sealed admission violates the frozen residual decision')
    run_path=directory/'run.json'
    if run_path.exists():
        digest=json.loads(run_path.read_text()).get('admissions_sha256')
        if digest and sha256_file(path)!=digest: raise ValueError('Sealed admission record hash changed')
    return result


def admissions(queue,directory,v4_objects,prepared):
    """Generate each fixed shape once and persist every video-only choice before scan scoring."""
    admission_path=directory/'admissions.json'
    if admission_path.exists(): return sealed_admissions(queue,directory)
    import numpy as np
    from ._admission import fit_residual
    from .pilot import generate

    ledger=directory/'ledger.jsonl'
    done={r['key']:r for line in ledger.read_text().splitlines() if (r:=json.loads(line)).get('stage')=='admission'} if ledger.exists() else {}
    rows=[]
    for pair in queue['pairs']:
        ident=key(pair)
        if ident in done: rows.append(done[ident]); continue
        item=directory/ident; v4=v4_objects/f"{pair['object']}.ply"
        row=dict(**pair,key=ident,stage='admission',provenance='video_only',v4_mesh=str(v4.relative_to(ROOT)),
                 v4_mesh_sha256=sha256_file(v4),generated_residual_m=None,v4_residual_m=None)
        if pair['measurement'] is None:
            row.update(decision='v4',selected_mesh=str(v4.relative_to(ROOT)),reason=pair['fallback'])
        else:
            item.mkdir(parents=True,exist_ok=True)
            if pair.get('reuse_pilot'):
                mesh=ROOT/pair['reuse_pilot']; g1=json.loads((ROOT/'outputs/mesh_generative/g1-pilot/run.json').read_text())
                expected=g1['artifacts'][f"{pair['object']}/raw.ply"]
                if sha256_file(mesh)!=expected or sha256_file(item/'crop.png')!=g1['artifacts'][f"{pair['object']}/crop.png"]:
                    raise ValueError('Pilot reuse differs from its own frozen pair')
                generation=json.loads(mesh.with_suffix('.json').read_text())
            else:
                mesh=item/'raw.ply'; generation=generate(item/'crop.png',mesh)
            arrays=item/'budgeted.npz'
            if not arrays.exists(): cpu('budget',dict(mesh=str(mesh.relative_to(ROOT)),v4=str(v4.relative_to(ROOT)),output=str(arrays.relative_to(ROOT))),item)
            with np.load(arrays,allow_pickle=False) as a,np.load(item/'observed.npz',allow_pickle=False) as points:
                generated=fit_residual(a['vertices'],a['faces'],points['points'],queue['fit'])
                incumbent=fit_residual(a['v4_vertices'],a['v4_faces'],points['points'],queue['fit'])
            accepted=generated['median_residual_m']<=1.5*incumbent['median_residual_m']
            row.update(generated_mesh=str(mesh.relative_to(ROOT)),generated_mesh_sha256=sha256_file(mesh),
                       generated_residual_m=generated['median_residual_m'],v4_residual_m=incumbent['median_residual_m'],
                       generated_fit=generated,v4_fit=incumbent,decision='generated' if accepted else 'v4',
                       selected_mesh=str((mesh if accepted else v4).relative_to(ROOT)),generation=generation,
                       prepared=prepared[ident],reason='generated median <= 1.5 x v4' if accepted else 'generated median > 1.5 x v4')
        row.update(status='completed',completed_utc=now_utc())
        append(ledger,row); rows.append(row)
        print(f"admission {ident}: {row['decision']} gen={row['generated_residual_m']} v4={row['v4_residual_m']}",flush=True)
    result=dict(status='completed',completed_utc=now_utc(),pairs=rows,provenance='video_only',
                scan_scoring_started=False,queue_sha256=sha256_file(ROOT/QUEUE))
    with admission_path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    return result


def run():
    """Run the fixed campaign and enter placeholder packing only after development passes."""
    runtime=authorize(); queue=json.loads((ROOT/QUEUE).read_text())
    for pair in queue['pairs']:
        if pair['episode'] in (41,43,44) or pair['episode'] not in runtime['g2_authorized_episode_indices'][pair['source_split']]:
            raise PermissionError('Pair outside G2 episode allow-list')
    directory=output_path(ROOT,OUT); directory.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/queue['v4_manifest']).read_text()); build=ROOT/manifest['build_run']
    v4_objects=build.parent/'objects'
    prepared={}
    inputs=[QUEUE,FEATURE/'runtime.json',queue['protocol'],queue['roster'],queue['v4_manifest'],queue['development_baseline'],
            'track3/features/mesh_extents/assignments.json',build,build.parent/'objects.json',
            '.runtime/hunyuan-mesh/weights-selected.json','.runtime/hunyuan-mesh/requirements.lock.txt',
            '.runtime/hunyuan-mesh/f8db63096c8282cb27354314d896feba5ba6ff8a.zip',
            '.runtime/hunyuan-mesh/weights-xet/hunyuan3d-dit-v2-mini/config.yaml',
            '.runtime/hunyuan-mesh/weights-xet/hunyuan3d-dit-v2-mini/model.fp16.safetensors']
    for pair in queue['pairs']:
        if pair['measurement']:
            item=directory/key(pair); item.mkdir(parents=True,exist_ok=True)
            prepared[key(pair)]=prepare(pair,item)
            inputs += [ROOT/e['path'] for e in prepared[key(pair)]['inputs']]
    record=run_fields(ROOT,'mesh_generative','g2',dict(queue=QUEUE),inputs=inputs)
    path=directory/'run.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=record[k] for k in ('inputs','sources')): raise ValueError('G2 inputs/sources changed; preserve existing generations')
        if old['status']=='completed': return path
    record.update(started_utc=now_utc(),prepared=prepared,queue=queue)
    write_json(path,record)
    try:
        decisions=admissions(queue,directory,v4_objects,prepared)
        record['admissions_sha256']=sha256_file(directory/'admissions.json')
        write_json(path,record)
        development_path=directory/'development.json'
        if not development_path.exists(): cpu('development',dict(admissions=str((directory/'admissions.json').relative_to(ROOT)),
            queue=QUEUE,v4_objects=str(v4_objects.relative_to(ROOT)),output=str(development_path.relative_to(ROOT))),directory)
        development=json.loads(development_path.read_text())
        packed=None
        if development['verdict']=='PASS':
            from .g2_pack import build_and_check
            packed=build_and_check(queue,decisions)
        from .g2_report import publish
        publish(directory,decisions,development,packed)
        record.update(status='completed',development_verdict=development['verdict'],
                      packed_verdict=packed['verdict'] if packed else 'NOT_RUN',provenance='public_asset_diagnostic',
                      disclosure='G2 selected pot and pitcher after G1 on the same scans; development evidence, not untouched validation.')
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally:
        record.update(finished_utc=now_utc(),artifacts={p.relative_to(directory).as_posix():sha256_file(p)
            for p in directory.rglob('*') if p.is_file() and p.name!='run.json'})
        write_json(path,record)
    return path
