"""Byte-preserving v5 production extension and the rejected G3 placeholder check."""

import json
import shutil
import subprocess

from src.paths import PROJECT as ROOT, output_path
from src.provenance import input_file, sha256_file, write_json
from .g2_pack import pack

OUT = 'data/outputs/mesh_generative/g3-packed-check'


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
    cpu('packed', dict(queue='src/features/mesh_generative/g3_queue.json',
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
        argv = ['data/.runtime/kit-env/bin/python', '-B', 'upstream/v2d_submission_kit/v2d_submission_kit/tools/preflight.py',
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
