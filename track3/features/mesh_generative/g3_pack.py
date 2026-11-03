"""Byte-preserving v5 production extension and gated D1 v6 packing."""

import json
from pathlib import Path
import shutil
import subprocess

from track3.paths import PROJECT as ROOT, output_path
from track3.provenance import input_file, sha256_file, write_json
from .g2_pack import pack

OUT = 'outputs/mesh_generative/g3-packed-check'


def production(queue, decisions, directory):
    """Copy all frozen v5 PLY bytes, replacing only admitted G3 evaluation pairs."""
    manifest = json.loads((ROOT/queue['v5_manifest']).read_text())
    baseline_path = ROOT/manifest['production']['path']
    if sha256_file(baseline_path) != manifest['production']['sha256']:
        raise ValueError('V5 production manifest changed')
    baseline = json.loads(baseline_path.read_text())
    choices = {(r['episode'],r['object']):r for r in decisions['pairs']
               if r.get('g3_new') and r['source_split'] == 'evaluation' and r['decision'] == 'generated'}
    rows = []
    for old in baseline['pairs']:
        if old['episode'] in (41, 43, 44):
            raise PermissionError('Excluded episode')
        original = ROOT/old['file']
        if sha256_file(original) != old['sha256']:
            raise ValueError('V5 production mesh changed')
        choice = choices.get((old['episode'], old['object']))
        selected = ROOT/choice['selected_mesh'] if choice else original
        target = directory/f"meshes/episode_{old['episode']:06d}/{old['object']}.ply"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(selected, target)
        digest = sha256_file(target)
        if digest != sha256_file(selected):
            raise ValueError('G3 production copy differs from selected mesh')
        if not choice and digest != old['sha256']:
            raise ValueError('Unchanged v5 bytes differ')
        rows.append(dict(episode=old['episode'], object=old['object'], file=str(target.relative_to(ROOT)),
                         sha256=digest, v5_sha256=old['sha256'], changed=digest != old['sha256'],
                         source=input_file(ROOT, selected)))
    result = dict(status='completed', pairs=rows, baseline=input_file(ROOT, baseline_path),
                  unchanged_pairs=sum(not r['changed'] for r in rows),
                  all_unchanged_bytes_identical=all(r['sha256'] == r['v5_sha256'] for r in rows if not r['changed']))
    write_json(directory/'production.json', result)
    return manifest, result


def build_and_check(queue, decisions):
    """Pack once with a placeholder URL and apply the frozen v5 packed veto."""
    from .g3 import cpu
    directory = output_path(ROOT, OUT)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'veto.json'
    if path.exists():
        result = json.loads(path.read_text())
        if 'placeholder_file' not in result:
            raise ValueError('Incomplete packed veto; preserve artifacts')
        return result
    manifest, tree = production(queue, decisions, directory)
    if not any(r['changed'] for r in tree['pairs']):
        result = dict(status='completed', verdict='FAIL', gates=dict(new_pair=False),
                      reason='no evaluation pair changed; stopped before packing')
        write_json(path, result)
        return result
    url = f"https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/{queue['protocol_commit']}"
    try:
        packed, command = pack(manifest, directory, url, 'track3_meshes_v6_placeholder.parquet')
    finally:
        if (directory/'pack.log').exists():
            (directory/'pack.log').replace(directory.parent/'g3-pack.log')
    cpu('packed', dict(queue='track3/features/mesh_generative/g3_queue.json',
        v5=manifest['file']['path'], candidate=str(packed.relative_to(ROOT)),
        production=str((directory/'production.json').relative_to(ROOT)),
        directory=str((directory/'pairs').relative_to(ROOT)), output=str(path.relative_to(ROOT))), directory)
    result = json.loads(path.read_text())
    result.update(pack_command=command, code_commit_url=url, placeholder_file=input_file(ROOT, packed),
                  production=input_file(ROOT, directory/'production.json'), preflight=[])
    if result['verdict'] != 'PASS':
        write_json(path, result)
        return result
    preflights = []
    for track in ('track_3', 'track_3_mesh'):
        argv = ['.runtime/kit-env/bin/python', '-B', 'upstream/v2d_submission_kit/v2d_submission_kit/tools/preflight.py',
                '--track', track, '--skip-kaggle']
        log_path = directory.parent/f'g3-preflight-{track}.log'
        with log_path.open('w') as log:
            subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        ready = 'Ready to submit' in log_path.read_text()
        preflights.append(dict(command=argv, log=str(log_path.relative_to(ROOT)), verdict='Ready to submit' if ready else 'FAIL'))
    result['gates']['kit_preflight'] = all(p['verdict'] == 'Ready to submit' for p in preflights)
    result['gates']['unchanged_rows_identical'] = result['unchanged_geometry_identical']
    result.update(verdict='PASS' if all(result['gates'].values()) else 'FAIL', preflight=preflights,
                  pack_command=command, code_commit_url=url, placeholder_file=input_file(ROOT, packed),
                  production=input_file(ROOT, directory/'production.json'))
    write_json(path, result)
    return result


def repack(url, out, execute=False):
    """Repack only the exact production tree that passed both frozen G3 gates."""
    from .g3 import authorize, OUT as RUN_OUT
    authorize()
    run = json.loads((ROOT/RUN_OUT/'run.json').read_text())
    development = json.loads((ROOT/RUN_OUT/'development.json').read_text())
    directory = ROOT/OUT
    veto = json.loads((directory/'veto.json').read_text())
    if run['status'] != 'completed' or run['control_verdict'] != 'PASS' or development['verdict'] != 'PASS' or veto['verdict'] != 'PASS':
        raise PermissionError('G3 did not pass control, development and packed veto')
    for entry in (veto['production'], veto['placeholder_file']):
        if sha256_file(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Passing G3 packed input changed')
    production_path = ROOT/veto['production']['path']
    for row in json.loads(production_path.read_text())['pairs']:
        if sha256_file(ROOT/row['file']) != row['sha256']:
            raise ValueError('Passing G3 production mesh changed')
    if 'PLACEHOLDER' in url or '/commit/' not in url:
        raise ValueError('Supply a real immutable GitHub commit URL')
    destination = output_path(ROOT, out)
    if not destination.is_relative_to(ROOT/'outputs/mesh_generative'):
        raise ValueError('Repack output must remain under outputs/mesh_generative')
    command = list(veto['pack_command'])
    for flag, value in (('--meshes', str(directory/'meshes')), ('--commit', url), ('--out', str(destination))):
        command[command.index(flag)+1] = value
    print(' '.join(command), flush=True)
    if execute:
        if destination.exists():
            raise FileExistsError('Preserve prior packs')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        import pyarrow.parquet as pq
        columns = ['row_id', 'x', 'y', 'z']
        if not pq.read_table(ROOT/veto['placeholder_file']['path'], columns=columns).equals(pq.read_table(destination, columns=columns)):
            raise ValueError('Repack geometry differs from passing placeholder')
        write_json(destination.with_suffix('.manifest.json'), dict(status='completed', file=input_file(ROOT, destination),
                   code_commit_url=url, pack_command=command, passing_veto=input_file(ROOT, directory/'veto.json'),
                   production=input_file(ROOT, production_path), placeholder=veto['placeholder_file'], placeholder_geometry_identical=True))
    return destination
