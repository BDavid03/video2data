"""Byte-preserving G2 production copies, official placeholder packing and gated repacking."""

import json
from pathlib import Path
import shutil
import subprocess

from track3.paths import PROJECT as ROOT, output_path
from track3.provenance import input_file, sha256_file, write_json

OUT='outputs/mesh_generative/g2-packed-check'


def production(queue,decisions,directory):
    """Copy v4 production bytes, replacing only video-admitted evaluation pot/pitcher pairs."""
    manifest=json.loads((ROOT/queue['v4_manifest']).read_text())
    build_path=ROOT/manifest['build_run']; build=json.loads(build_path.read_text())
    roster=json.loads((ROOT/queue['roster']).read_text())['episodes']
    choices={(r['episode'],r['object']):r for r in decisions['pairs'] if r['source_split']=='evaluation'}
    rows=[]
    for ep,names in roster.items():
        for name in names:
            if int(ep) in (41,43,44): raise PermissionError('Excluded episode in roster')
            relative=f'meshes/episode_{int(ep):06d}/{name}.ply'
            original=build_path.parent/relative
            if sha256_file(original)!=build['artifacts'][relative]: raise ValueError('Pinned v4 production mesh changed')
            choice=choices.get((int(ep),name)); selected=ROOT/choice['selected_mesh'] if choice else original
            target=directory/relative; target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copyfile(selected,target)
            digest=sha256_file(target)
            if digest!=sha256_file(selected): raise ValueError('Production copy differs from admitted mesh')
            changed=bool(choice and choice['decision']=='generated')
            if not changed and digest!=sha256_file(original): raise ValueError('V4 fallback bytes changed')
            rows.append(dict(episode=int(ep),object=name,file=str(target.relative_to(ROOT)),sha256=digest,
                             v4_sha256=sha256_file(original),changed=changed,source=input_file(ROOT,selected)))
    write_json(directory/'production.json',dict(status='completed',pairs=rows,
        unchanged_non_target_pairs=sum(r['object'] not in ('white_pot','water_pitcher') for r in rows),
        all_non_target_bytes_identical=all(r['sha256']==r['v4_sha256'] for r in rows if r['object'] not in ('white_pot','water_pitcher'))))
    return manifest


def pack(manifest,directory,url,filename):
    """Use the exact v4 packer command with the new production tree and supplied URL."""
    path=directory/filename
    command=list(manifest['pack_command'])
    for flag,value in (('--meshes',str(directory/'meshes')),('--commit',url),('--out',str(path))):
        command[command.index(flag)+1]=value
    if not path.exists():
        with (directory/'pack.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return path,command


def build_and_check(queue,decisions):
    """Build one placeholder candidate and apply the exact packed veto without repairs."""
    from .g2 import cpu
    directory=output_path(ROOT,OUT); directory.mkdir(parents=True,exist_ok=True)
    verdict=directory/'veto.json'
    if verdict.exists(): return json.loads(verdict.read_text())
    manifest=production(queue,decisions,directory)
    commit=subprocess.run(['git','rev-parse',queue['protocol_commit']],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    url=f'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/{commit}'
    packed,command=pack(manifest,directory,url,'track3_meshes_v5_placeholder.parquet')
    cpu('packed',dict(queue='track3/features/mesh_generative/g2_queue.json',candidate=str(packed.relative_to(ROOT)),
        v4=manifest['file'],directory=str((directory/'pairs').relative_to(ROOT)),
        production=str((directory/'production.json').relative_to(ROOT)),output=str(verdict.relative_to(ROOT))),directory)
    result=json.loads(verdict.read_text())
    preflights=[]
    for track in ('track_3','track_3_mesh'):
        argv=['.runtime/kit-env/bin/python','-B','upstream/v2d_submission_kit/v2d_submission_kit/tools/preflight.py',
              '--track',track,'--skip-kaggle']
        log_path=directory/f'preflight-{track}.log'
        with log_path.open('w') as log:
            subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'Ready to submit' not in log_path.read_text(): raise ValueError('Kit preflight did not pass')
        preflights.append(dict(command=argv,log=str(log_path.relative_to(ROOT)),verdict='Ready to submit'))
    result.update(preflight=preflights,pack_command=command,code_commit_url=url,
                  placeholder_file=input_file(ROOT,packed),production=input_file(ROOT,directory/'production.json'))
    write_json(verdict,result)
    return result


def repack(url,out,execute=False):
    """Repack the frozen passing production tree with the lead's real commit URL."""
    from .g2 import authorize
    authorize()
    directory=ROOT/OUT; veto=json.loads((directory/'veto.json').read_text())
    development=json.loads((ROOT/'outputs/mesh_generative/g2/development.json').read_text())
    if veto['verdict']!='PASS' or development['verdict']!='PASS': raise PermissionError('G2 did not pass both frozen gates')
    if 'PLACEHOLDER' in url or '/commit/' not in url: raise ValueError('Supply the real immutable GitHub commit URL')
    production_path=directory/'production.json'
    if sha256_file(production_path)!=veto['production']['sha256']:
        raise ValueError('Veto-pinned production manifest changed')
    production=json.loads(production_path.read_text())
    for row in production['pairs']:
        if sha256_file(ROOT/row['file'])!=row['sha256']: raise ValueError('Passing production bytes changed')
    placeholder=ROOT/veto['placeholder_file']['path']
    if sha256_file(placeholder)!=veto['placeholder_file']['sha256']:
        raise ValueError('Veto-pinned placeholder changed')
    destination=output_path(ROOT,out)
    if not any(destination.is_relative_to(ROOT/folder) for folder in ('outputs/mesh_generative','.runtime/tmp/g1')):
        raise ValueError('Repack output must remain under outputs/mesh_generative or .runtime/tmp/g1')
    command=list(veto['pack_command'])
    command[command.index('--meshes')+1]=(directory/'meshes').relative_to(ROOT).as_posix()
    command[command.index('--commit')+1]=url; command[command.index('--out')+1]=str(destination)
    print(' '.join(command),flush=True)
    if execute:
        if destination.exists(): raise FileExistsError('Preserve prior packs')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        import pyarrow.parquet as pq
        columns=['row_id','x','y','z']
        if not pq.read_table(placeholder,columns=columns).equals(pq.read_table(destination,columns=columns)):
            raise ValueError('Repacked row IDs or coordinates differ from the verified placeholder')
        write_json(destination.with_suffix('.manifest.json'),dict(status='completed',file=input_file(ROOT,destination),
            code_commit_url=url,pack_command=command,passing_veto=input_file(ROOT,directory/'veto.json'),
            production=input_file(ROOT,production_path),placeholder=input_file(ROOT,placeholder),
            placeholder_geometry_identical=True))
    return destination
