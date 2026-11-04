"""Pack admitted FP static rollouts while preserving the uploaded moving baseline."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from src.provenance import write_json
from .combined_frames import ROOT,OUTPUT,KIT
from . import pack_moving as moving
from . import pack_combined as base
from . import counted
from .frame0_fp_policy import inspect_admission,inspect_export

PARENT=ROOT/'submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json'


def compare(target, admissions, allow_url):
    parent=moving.read(PARENT)
    old=pq.read_table(moving.checked(parent['file']))
    new=pq.read_table(target)
    if old.column_names!=new.column_names or old['row_id'].to_pylist()!=new['row_id'].to_pylist():
        raise ValueError('Packed official row identities/order/schema differ')
    episodes=[int(x.split('/')[1][1:]) for x in old['row_id'].to_pylist()]
    untouched=pa.array([ep not in admissions for ep in episodes])
    for column in old.column_names:
        if allow_url and column=='code_commit_url':continue
        if not pc.filter(old[column],untouched).equals(pc.filter(new[column],untouched)):
            raise ValueError(f'Untouched row values changed: {column}')
    roster,objects=moving.official_roster();_,code=moving.evaluator_inputs()
    for episode,item in admissions.items():
        frame,poses=inspect_export(moving.checked(item['trajectory']),item['checkpoint']['sha256'],
            episode,roster[episode],objects[episode].object_names,code)
        expected={f'track_3/e{episode:06d}/w{int(w):03d}/f{int(f):06d}/o{int(o)}':v
            for (w,f,o),v in zip(frame[['environment_id','reference_frame','object_body_id']].itertuples(index=False,name=None),poses.reshape(-1,7))}
        selected=new.filter(pa.array([ep==episode for ep in episodes])).to_pydict()
        for i,key in enumerate(selected['row_id']):
            if not np.array_equal(np.asarray([selected[c][i] for c in base.POSE]),expected[key]):
                raise ValueError('Packed pose differs from the official rollout')
            if selected['checkpoint_sha256'][i]!=item['checkpoint']['sha256']:
                raise ValueError('Packed checkpoint identity differs')
    return dict(passed=True,adopted_episodes=sorted(admissions),unchanged_rows=sum(untouched.to_pylist()),
        unchanged_fields='all fields exact' if not allow_url else 'all fields exact except owner-approved shared code_commit_url')


def verify_manifest(path,release=False):
    value=moving.read(path)
    for item in [value['file'],value['parent'],value['report'],*value['sources'],*value['admissions'],*value['exports'],*value['meshes']]:
        moving.checked(item)
    admissions={}
    for entry in value['admissions']:
        item=inspect_admission(moving.checked(entry));admissions[item['episode']]=item
    if compare(moving.checked(value['file']),admissions,value['shared_url_update'])!=value['comparison']:
        raise ValueError('Packed comparison changed')
    if release:moving.committed(value['sources'],value['code_commit_url'])
    return value


def pack(paths,url,allow_url=False):
    admissions={}
    for path in paths:
        item=inspect_admission(path)
        if item['episode'] in admissions:raise ValueError('Duplicate admitted episode')
        admissions[item['episode']]=item
    if not admissions:raise ValueError('No admitted rollout')
    parent=moving.read(PARENT);baseline=moving.checked(parent['file'])
    if url!=parent['code_commit_url'] and not allow_url:
        raise ValueError('New shared code URL requires owner approval to change untouched provenance')
    sources=[moving.entry(ROOT/'src/features/pose_insurance'/name) for name in
        ('frame0_fp.py','frame0_fp_worker.py','_frame0_fp_frames.py','frame0_fp_policy.py','frame0_fp_retrain.py','pack_combined_fp.py')]
    moving.committed(sources,url)
    output=OUTPUT/('fp-candidate-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    output.mkdir(parents=True,exist_ok=False);results=output/'results';results.mkdir();meshes=output/'meshes';mesh_entries=[];exports=[]
    for entry in parent['exports']:
        ep=entry['episode'];source=moving.checked(admissions[ep]['trajectory']) if ep in admissions else moving.checked(entry)
        target=results/f'episode_{ep:06d}.parquet';shutil.copyfile(source,target);exports.append(dict(episode=ep,**moving.entry(target)))
    for entry in parent['meshes']:
        source=moving.checked(entry);target=meshes/source.parent.name/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);mesh_entries.append(moving.entry(target))
    target=output/'track3_combined_fp.parquet';python=ROOT/'data/.runtime/kit-env/bin/python'
    commands=[base.command([python,'-B',KIT/'tools/pack_submission.py','--track','track_3','--results',results,'--meshes',meshes,
        '--code-commit',url,'--eval-script',KIT/'flash_chord/scripts/evaluate_policy.py','--skip-crosscheck','-o',target],output/'pack.log')]
    old=pq.read_table(baseline);new=pq.read_table(target);replacement=pa.array([int(x.split('/')[1][1:]) in admissions for x in old['row_id'].to_pylist()])
    arrays=[new[c] if allow_url and c=='code_commit_url' else pc.if_else(replacement,new[c],old[c]) for c in old.column_names]
    pq.write_table(pa.Table.from_arrays(arrays,schema=old.schema),target)
    commands.append(base.command([python,'-B',KIT/'tools/preflight.py','--track','track_3','--results',results,'--skip-kaggle'],output/'preflight.log'))
    import sys
    sys.path.insert(0,str(KIT))
    from src.features.moving_reference import kit_score
    kit_score.scorer('AUC')._check_provenance(pq.read_table(target).to_pandas())
    comparison=compare(target,admissions,allow_url)
    manifest=dict(kind='pose_insurance.combined_fp_candidate',file=moving.entry(target),parent=moving.entry(PARENT),
        code_commit_url=url,shared_url_update=allow_url,comparison=comparison,sources=sources,
        admissions=[moving.entry(path) for path in paths],exports=exports,meshes=mesh_entries,commands=commands,
        report=moving.entry(ROOT/'docs/report/track3/main.tex'),release_eligible=True,status='packed, not uploaded')
    manifest_path=target.with_suffix('.manifest.json');write_json(manifest_path,manifest);verify_manifest(manifest_path,True)
    checkpoints=counted.parent_checkpoints(ROOT,parent['file']['sha256'])
    for ep,item in admissions.items():checkpoints[f'episode_{ep:06d}.safetensors']=item['checkpoint']
    counted.archive_pack(ROOT,target,manifest_path,checkpoints)
    return dict(path=str(target.relative_to(ROOT)),sha256=manifest['file']['sha256'],manifest=str(manifest_path.relative_to(ROOT)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',action='append');parser.add_argument('--code-url');parser.add_argument('--allow-shared-url-update',action='store_true');parser.add_argument('--verify-manifest');parser.add_argument('--release',action='store_true');args=parser.parse_args()
    print(json.dumps(verify_manifest(args.verify_manifest,args.release) if args.verify_manifest else pack(args.admission or [],args.code_url,args.allow_shared_url_update),indent=2))


if __name__=='__main__':main()
