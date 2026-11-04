"""Pack cumulative joint policies beside the immutable uploaded episode-38 policy."""

from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.paths import project_path
from src.provenance import sha256_file,write_json
from . import pack_combined_moving as frozen
from . import pack_combined as base
from . import pack_moving as m
from .combined_frames import ROOT,OUTPUT,KIT,entry
from src.features.moving_policy.two_object_admission import inspect as inspect_joint

PARENT=ROOT/'submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json'


def compare(target,admitted,parent):
    old=pq.read_table(m.checked(parent['file']))
    new=pq.read_table(target)
    if old['row_id'].to_pylist()!=new['row_id'].to_pylist() or old.column_names!=new.column_names:
        raise ValueError('Cumulative pack row IDs/order/columns differ from uploaded parent')
    keys=[int(value.split('/')[1][1:]) for value in old['row_id'].to_pylist()]
    untouched=pa.array([ep not in admitted for ep in keys])
    for column in old.column_names:
        before=pc.filter(old[column],untouched)
        after=pc.filter(new[column],untouched)
        if not before.equals(after):
            raise ValueError(f'Unadopted uploaded-parent values changed: {column}')
        if column in base.POSE and not np.array_equal(before.to_numpy().view(np.uint64),after.to_numpy().view(np.uint64)):
            raise ValueError('Unadopted floating-point bits changed')
    roster,objects=m.official_roster()
    _,code=m.evaluator_inputs()
    for episode,admission in admitted.items():
        frame,poses=m.inspect_export(m.checked(admission['trajectory']),admission['checkpoint']['sha256'],episode,
                                    roster[episode],objects[episode].object_names,code)
        expected={f'track_3/e{episode:06d}/w{int(w):03d}/f{int(f):06d}/o{int(o)}':pose
                  for (w,f,o),pose in zip(frame[['environment_id','reference_frame','object_body_id']].itertuples(index=False,name=None),poses.reshape(-1,7))}
        selected=new.filter(pa.array([ep==episode for ep in keys])).to_pydict()
        if len(selected['row_id'])!=len(expected):
            raise ValueError('Adopted official grid is incomplete')
        for i,row_id in enumerate(selected['row_id']):
            actual=np.asarray([selected[c][i] for c in base.POSE])
            if row_id not in expected or not np.array_equal(actual,expected[row_id]):
                raise ValueError('Adopted pose differs from its single joint official evaluator export')
            if selected['checkpoint_sha256'][i]!=admission['checkpoint']['sha256']:
                raise ValueError('Adopted checkpoint provenance differs')
    return dict(passed=True,rows=len(new),unchanged_rows=int(pc.sum(pc.cast(untouched,pa.int64())).as_py()),
                adopted_episodes=sorted(admitted),unadopted_equality='all fields exact; pose float64 bits exact',
                adopted_equality='single-checkpoint official exports, full grid and exact pose values')


def pack_cumulative(admission_paths):
    if m.read(ROOT/'data/outputs/moving_policy/two-object/review-fixes-status.json').get('passed') is not True:
        raise PermissionError('Joint review corrections are held; no candidate pack')
    parent=m.read(PARENT)
    baseline=PARENT.with_name('track3_combined_v5.parquet')
    if sha256_file(baseline)!=parent['file']['sha256']:
        raise ValueError('Uploaded parent differs from its recorded file hash')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=OUTPUT/f'joint-candidate-{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    admissions={}
    entries=[]
    for path in admission_paths:
        item=m.read(m.path(path))
        if item.get('kind')!='moving_policy.joint_admission' or item.get('admitted') is not True:
            raise ValueError('Only completed joint admissions may enter a cumulative candidate')
        fresh=inspect_joint(item['policy_directory'],item['episode'])
        before={k:v for k,v in item.items() if k!='abstract_disclosure'}
        after={k:v for k,v in fresh.items() if k!='abstract_disclosure'}
        if before!=after:
            raise ValueError('Joint scientific admission changed on independent re-admission')
        if item['episode'] in admissions or item['episode']==38:
            raise ValueError('Duplicate or invalid joint episode adoption')
        admissions[item['episode']]=fresh
        refreshed_file=output/f'ep{item["episode"]}-admission.json'
        write_json(refreshed_file,fresh)
        entries.append(m.entry(refreshed_file))
    if not admissions:
        raise ValueError('No new admitted joint policy; keep uploaded moving v5')
    # Refresh only the report binding in a new episode-38 admission; all original evidence is rechecked.
    original_request=m.read(m.checked(parent['adoptions']))
    adoption,=[r for r in original_request['adoptions'] if r['episode']==38]
    refreshed=m.read(adoption['admission'])
    refreshed['abstract_disclosure']=m.entry(ROOT/'docs/report/track3/main.tex')
    ep38_evidence=output/'ep38-admission.json'
    write_json(ep38_evidence,refreshed)
    adopted38=dict(adoption,admission=str(ep38_evidence.relative_to(ROOT)))
    roster,objects=m.official_roster();_,code=m.evaluator_inputs()
    ep38=m.admit(adopted38,roster[38],objects[38].object_names,code)
    results=output/'results';results.mkdir()
    exports=[]
    for e in parent['exports']:
        episode=e['episode']
        source=m.checked(admissions[episode]['trajectory']) if episode in admissions else m.checked(e)
        destination=results/f'episode_{episode:06d}.parquet'
        shutil.copyfile(source,destination)
        exports.append(dict(episode=episode,**entry(destination)))
    meshes=output/'meshes';mesh_entries=[]
    for e in parent['meshes']:
        source=m.checked(e)
        episode=int(source.parent.name.split('_')[1]);name=source.stem
        if episode in admissions:
            row=pq.read_table(m.checked(admissions[episode]['task'])).to_pylist()[0]
            slot=row['object_body_names'].index(name)
            source=m.path(row['object_mesh_paths'][slot])
        destination=meshes/f'episode_{episode:06d}'/source.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        mesh_entries.append(entry(destination))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    url=parent['code_commit_url'].rsplit('/',1)[0]+'/'+head
    archive=frozen.verify_kit()
    target=output/'track3_combined_joint.parquet'
    python=ROOT/'data/.runtime/kit-env/bin/python'
    commands=[base.command([python,'-B',KIT/'tools/pack_submission.py','--track','track_3',
        '--results',results,'--meshes',meshes,'--code-commit',url,'--eval-script',KIT/'flash_chord/scripts/evaluate_policy.py',
        '--skip-crosscheck','-o',target],output/'pack.log')]
    old=pq.read_table(baseline);new=pq.read_table(target)
    if old['row_id'].to_pylist()!=new['row_id'].to_pylist():
        raise ValueError('Official pack order differs from parent')
    replace=pa.array([int(r.split('/')[1][1:]) in admissions for r in old['row_id'].to_pylist()])
    merged=pa.Table.from_arrays([pc.if_else(replace,new[c],old[c]) for c in old.column_names],schema=old.schema)
    pq.write_table(merged,target)
    commands.append(base.command([python,'-B',KIT/'tools/preflight.py','--track','track_3',
                                 '--results',results,'--skip-kaggle'],output/'preflight.log'))
    if 'Ready to submit.' not in (output/'preflight.log').read_text():
        raise ValueError('Kit preflight did not pass')
    comparison=compare(target,admissions,parent)
    source_files=[Path(__file__),ROOT/'src/features/moving_policy/two_object_admission.py',
        ROOT/'src/features/moving_policy/two_object_run.py',ROOT/'src/features/moving_policy/two_object_stage0.py',
        ROOT/'src/features/moving_policy/two_object_cpu.py',ROOT/'src/features/moving_policy/two_object_train_worker.py',
        ROOT/'src/features/moving_policy/_two_object.py',ROOT/'src/features/moving_policy/_two_object_instances.py',
        ROOT/'src/features/moving_policy/_two_object_symmetry.py',
        *[ROOT/f'src/features/moving_policy/{name}.py' for name in ('_two_object_facing','two_object_facing','_two_object_table','two_object_table')],
        ROOT/'src/features/moving_policy/_two_object_contract.py',ROOT/'src/features/moving_policy/two_object_contract.py',
        ROOT/'src/features/moving_policy/_two_object_camera.py',ROOT/'src/features/moving_policy/_two_object_rotation.py',
        ROOT/'src/features/moving_policy/two_object_rotation.py',ROOT/'src/features/moving_policy/two_object_clip_instances.py',
        ROOT/'src/features/moving_policy/two_object_clip_worker.py',ROOT/'src/features/moving_policy/two_object_campaign.py',
        ROOT/'src/features/moving_policy/two_object_rescore.py',ROOT/'src/features/moving_policy/two_object_existing.py',
        ROOT/'src/features/moving_policy/two_object_instances.py',ROOT/'src/features/moving_policy/two_object_reference.py',
        ROOT/'src/features/moving_policy/two_object_task.py',ROOT/'docs/chain/moving-two-object-protocol.md']
    sources=[entry(p) for p in source_files]
    try:
        m.committed(sources,url)
        committed=True
    except (ValueError,subprocess.CalledProcessError):
        committed=False
    manifest=dict(kind='pose_insurance.combined_joint_candidate',schema_version=1,file=entry(target),
        parent=entry(PARENT),parent_file=parent['file'],new_admissions=entries,episode38=ep38,
        episode38_refreshed_adoption=adopted38,exports=exports,meshes=mesh_entries,sources=sources,
        kit_archive=entry(archive),commands=commands,comparison=comparison,code_commit_url=url,
        source_commit_verified=committed,release_eligible=committed,
        status='candidate; no upload; lead review required' if committed else 'candidate; new source commit required before release',
        report=entry(ROOT/'docs/report/track3/main.tex'))
    manifest_path=target.with_suffix('.manifest.json')
    write_json(manifest_path,manifest)
    verify_manifest(manifest_path)
    write_json(output/'verified.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),file=entry(target),passed=True))
    return dict(path=str(target.relative_to(ROOT)),sha256=sha256_file(target),manifest=str(manifest_path.relative_to(ROOT)),
                adopted_episodes=sorted(admissions),source_commit_verified=committed)


def verify_manifest(path):
    manifest=m.read(path)
    if manifest.get('kind')!='pose_insurance.combined_joint_candidate':
        raise ValueError('Not a joint candidate manifest')
    for e in [manifest['file'],manifest['parent'],manifest['parent_file'],manifest['report'],manifest['kit_archive'],
              *manifest['new_admissions'],*manifest['exports'],*manifest['meshes'],*manifest['sources']]:
        m.checked(e)
    parent=m.read(m.checked(manifest['parent']))
    admitted={}
    for e in manifest['new_admissions']:
        saved=m.read(m.checked(e));fresh=inspect_joint(saved['policy_directory'],saved['episode'])
        if fresh!=saved:
            raise ValueError('Joint candidate admission no longer passes')
        admitted[saved['episode']]=fresh
    if compare(m.checked(manifest['file']),admitted,parent)!=manifest['comparison']:
        raise ValueError('Cumulative pose comparison changed')
    roster,objects=m.official_roster();_,code=m.evaluator_inputs()
    if m.admit(manifest['episode38_refreshed_adoption'],roster[38],objects[38].object_names,code)!=manifest['episode38']:
        raise ValueError('Episode 38 evidence no longer passes')
    for e in manifest['exports']:
        expected=admitted[e['episode']]['trajectory'] if e['episode'] in admitted else next(p for p in parent['exports'] if p['episode']==e['episode'])
        if e['sha256']!=expected['sha256']:
            raise ValueError('Candidate export source mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',action='append')
    parser.add_argument('--verify-manifest')
    args=parser.parse_args()
    print(json.dumps(verify_manifest(args.verify_manifest) if args.verify_manifest else pack_cumulative(args.admission or []),indent=2))


if __name__=='__main__':
    main()
