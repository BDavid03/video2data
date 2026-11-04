"""Frozen G3 loop: lid decisions, control stop, video admissions and conditional score gates."""

import copy
import json
from pathlib import Path
import subprocess
import time

from src.paths import PROJECT as ROOT, output_path
from src.provenance import input_file, run_fields, sha256_file, write_json
from src.runs import now_utc, runtime_flag
from . import g2
from src.features.mesh_extents.proxy import read_ledger

FEATURE = Path(__file__).parent
QUEUE = 'src/features/mesh_generative/g3_queue.json'
OUT = 'data/outputs/mesh_generative/g3'


def authorize():
    """Require the separate G3 flag before reusing G2's interpreter/cache setup."""
    runtime_flag(FEATURE, 'g3_authorized', 'Frozen G3 campaign')
    return g2.authorize()


def cpu(operation, job, directory):
    """Run a G3 CPU operation in the existing kit environment with an external log."""
    request = directory/f'{operation}-request.json'
    write_json(request, job)
    log_path = ROOT/f'data/outputs/mesh_generative/g3-{directory.name}-{operation}.log'
    with log_path.open('w') as log:
        subprocess.run([str(ROOT/'data/.runtime/kit-env/bin/python'), '-B', '-m',
                        'src.features.mesh_generative.g3_cpu', operation, '--request', str(request)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def pin_inputs(queue):
    """Verify the installed models and v5 artifacts and name every immutable runtime input."""
    paths = [QUEUE, FEATURE/'runtime.json', queue['protocol'], queue['roster'], queue['v4_manifest'],
             queue['development_baseline'], queue['g2_admissions'], queue['g2_development'],
             queue['v5_manifest'], queue['v5_veto'], 'data/reports/data-audit/trajectories.json',
             'src/features/mesh_extents/assignments.json']
    for name in ('sam2', 'grounding-dino', 'hunyuan-mesh'):
        base = ROOT/'data/.runtime'/name
        paths += [base/'INSTALL.md', base/'requirements.lock.txt', base/'weights-selected.json']
        selected = json.loads((base/'weights-selected.json').read_text())
        if name == 'hunyuan-mesh':
            paths += [base/'f8db63096c8282cb27354314d896feba5ba6ff8a.zip',
                      base/'weights-xet/hunyuan3d-dit-v2-mini/config.yaml',
                      base/'weights-xet/hunyuan3d-dit-v2-mini/model.fp16.safetensors']
        else:
            expected = queue['detector']['revision'] if name == 'grounding-dino' else queue['segmenter']['source_revision']
            if selected['status'] != 'completed' or selected['revision'] != expected:
                raise ValueError('Mask model installation is incomplete or differs from frozen revision')
            for entry in selected['files']:
                if sha256_file(ROOT/entry['path']) != entry['sha256']:
                    raise ValueError('Installed model/source differs from installation record')
                paths.append(entry['path'])
    manifest = json.loads((ROOT/queue['v5_manifest']).read_text())
    for entry in (manifest['file'], manifest['production'], manifest['passing_veto'], manifest['placeholder']):
        if sha256_file(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Pinned v5 input changed')
        paths.append(entry['path'])
    if manifest['file']['sha256'] != 'bcb93d57d86b4dac06ed21af9af84ce39edffe52b3d9c35e28928391e47e4e58':
        raise ValueError('G3 incumbent is the frozen real-URL v5')
    return paths


def base_decisions():
    """Reuse G2's sealed decisions, preserving all pairs outside the six G3 candidates."""
    original_queue = json.loads((ROOT/g2.QUEUE).read_text())
    return g2.sealed_admissions(original_queue, ROOT/g2.OUT)


def admit(pair, prepared, directory, queue, original):
    """Apply mask rejection, G2 generation/budgeting and the unchanged residual admission."""
    import numpy as np
    from .g3_mask import make_mask
    from ._admission import fit_residual
    from .pilot import generate

    row = copy.deepcopy(original)
    row.update(**pair, g3_new=True, stage='admission', generated_residual_m=None, v4_residual_m=None)
    if prepared['lid']:
        row.update(decision='v4', reason='lid rule: white_pot_lid in episode object list')
    else:
        mask = make_mask(pair, directory, queue, prepared)
        row.update(mask_record=input_file(ROOT, directory/'mask.json'), mask=mask)
        if not mask['accepted']:
            row.update(decision='v4', reason=mask['reason'])
        else:
            mesh = directory/'raw.ply'
            try:
                generation = generate(directory/'crop.png', mesh)
            finally:
                log_path = mesh.with_suffix('.log')
                if log_path.exists():
                    log_path.replace(ROOT/f'data/outputs/mesh_generative/g3-{g2.key(pair)}-generation.log')
            arrays = directory/'budgeted.npz'
            if not arrays.exists():
                cpu('budget', dict(mesh=str(mesh.relative_to(ROOT)), v4=row['v4_mesh'],
                                  output=str(arrays.relative_to(ROOT))), directory)
            with np.load(arrays, allow_pickle=False) as a, np.load(directory/'observed.npz', allow_pickle=False) as cloud:
                generated = fit_residual(a['vertices'], a['faces'], cloud['points'], queue['fit'])
                incumbent = fit_residual(a['v4_vertices'], a['v4_faces'], cloud['points'], queue['fit'])
            accepted = generated['median_residual_m'] <= 1.5*incumbent['median_residual_m']
            row.update(generated_mesh=str(mesh.relative_to(ROOT)), generated_mesh_sha256=sha256_file(mesh),
                       generated_residual_m=generated['median_residual_m'], v4_residual_m=incumbent['median_residual_m'],
                       generated_fit=generated, v4_fit=incumbent, generation=generation,
                       decision='generated' if accepted else 'v4',
                       reason='generated median <= 1.5 x v4' if accepted else 'generated median > 1.5 x v4')
    row.update(selected_mesh=row['generated_mesh'] if row['decision'] == 'generated' else row['v4_mesh'],
               status='completed', completed_utc=now_utc())
    return row


def admissions(queue, prepared, directory, baseline):
    """Seal the six video-only choices before any development or packed scan scoring."""
    path = directory/'admissions.json'
    ledger = directory/'ledger.jsonl'
    done = {r['key']:r for r in read_ledger(ledger) if r.get('stage') == 'admission'}
    if path.exists():
        result = json.loads(path.read_text())
        for row in result['pairs']:
            if row.get('g3_new') and done.get(row['key']) != row:
                raise ValueError('Sealed G3 admission differs from the ledger')
            if sha256_file(ROOT/row['selected_mesh']) != row.get('generated_mesh_sha256' if row['decision'] == 'generated' else 'v4_mesh_sha256'):
                raise ValueError('Admitted mesh changed')
        return result
    rows = copy.deepcopy(baseline['pairs'])
    for pair in queue['pairs']:
        ident = g2.key(pair)
        index = next(i for i, row in enumerate(rows) if row['key'] == ident)
        if ident not in done:
            done[ident] = admit(pair, prepared[ident], directory/ident, queue, rows[index])
            g2.append(ledger, done[ident])
            print(f"admission {ident}: {done[ident]['decision']} ({done[ident]['reason']})", flush=True)
        rows[index] = done[ident]
    result = dict(status='completed', completed_utc=now_utc(), provenance='video_only', pairs=rows,
                  scan_scoring_started=False, queue_sha256=sha256_file(ROOT/QUEUE),
                  g2_admissions=input_file(ROOT, ROOT/queue['g2_admissions']))
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def run():
    """Execute the fixed queue once, stopping immediately at each frozen failed gate."""
    runtime_flag(FEATURE, 'g3_authorized', 'Frozen G3 campaign')
    path = ROOT/OUT/'run.json'
    if path.exists() and json.loads(path.read_text())['status'] == 'completed':
        return path
    from .g3_mask import prepare, make_mask
    from .g3_report import publish
    start = time.perf_counter()
    runtime = authorize()
    queue = json.loads((ROOT/QUEUE).read_text())
    for pair in queue['pairs'] + queue['controls']:
        if pair['episode'] in (41, 43, 44) or pair['episode'] not in runtime['g2_authorized_episode_indices'][pair['source_split']]:
            raise PermissionError('Pair outside the existing episode allow-list')
    directory = output_path(ROOT, OUT)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'run.json'
    inputs = pin_inputs(queue)
    baseline = base_decisions()
    prepared = {}
    for control, pairs in ((False, queue['pairs']), (True, queue['controls'])):
        for pair in pairs:
            ident = g2.key(pair)
            prepared[ident] = prepare(pair, directory/ident, control)
            inputs += [ROOT/e['path'] for e in prepared[ident]['inputs']]
    record = run_fields(ROOT, 'mesh_generative', 'g3', dict(queue=QUEUE), inputs=inputs)
    record['sources'] += [input_file(ROOT, p) for p in (
        'src/features/mesh_extents/_geometry.py', 'src/features/object_masks/opening_clusters.py',
        'src/features/mesh_eval/pairs.py', 'src/features/mesh_eval/_metrics.py',
        'src/features/mesh_eval/local_cdo.py',
        'upstream/v2d_submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py',
        'upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/CD-O.py')]
    if path.exists():
        old = json.loads(path.read_text())
        if any(record[k] != old[k] for k in ('inputs', 'sources')):
            raise ValueError('G3 inputs/sources changed; preserve all existing predictions')
        if old['status'] == 'completed':
            return path
        record = old
    else:
        record.update(started_utc=now_utc(), queue=queue, prepared=prepared,
                      control_verdict='NOT_RUN', development_verdict='NOT_RUN', packed_verdict='NOT_RUN')
    write_json(path, record)
    controls, decisions, development, packed = [], None, None, None
    ledger = directory/'ledger.jsonl'
    if not ledger.exists():
        for pair in queue['pairs']:
            meta = prepared[g2.key(pair)]
            g2.append(ledger, dict(stage='lid_rule', key=g2.key(pair), lid=meta['lid'],
                                  object_names=meta['object_names'], provenance='video_only', completed_utc=now_utc()))
    try:
        for pair in queue['controls']:
            ident = g2.key(pair)
            mask = make_mask(pair, directory/ident, queue, prepared[ident], control=True)
            controls.append(dict(key=ident, iou=mask['control_iou'], passed=mask['control_iou'] >= queue['control_minimum_iou'],
                                 mask=mask))
            if not any(r.get('stage') == 'control' and r.get('key') == ident for r in read_ledger(ledger)):
                g2.append(ledger, dict(stage='control', key=ident, iou=mask['control_iou'],
                                      passed=controls[-1]['passed'], provenance='video_only', completed_utc=now_utc()))
            control_path = directory/'controls.json'
            write_json(control_path, dict(controls=controls, verdict='PASS' if len(controls) == len(queue['controls']) and all(r['passed'] for r in controls) else 'FAIL'))
            print(f"control {ident}: IoU={mask['control_iou']:.8f}", flush=True)
            if not controls[-1]['passed']:
                record.update(control_verdict='FAIL', stop_reason=f'control IoU < 0.70: {ident}')
                break
        else:
            record['control_verdict'] = 'PASS'
            decisions = admissions(queue, prepared, directory, baseline)
            record['admissions_sha256'] = sha256_file(directory/'admissions.json')
            write_json(path, record)
            target = directory/'development.json'
            v4 = json.loads((ROOT/queue['v4_manifest']).read_text())
            objects = (Path(v4['build_run']).parent/'objects').as_posix()
            if not target.exists():
                cpu('development', dict(queue=QUEUE, admissions=str((directory/'admissions.json').relative_to(ROOT)),
                                       v4_objects=objects, output=str(target.relative_to(ROOT))), directory)
            development = json.loads(target.read_text())
            record['development_verdict'] = development['verdict']
            if development['verdict'] == 'PASS':
                from .g3_pack import build_and_check
                packed = build_and_check(queue, decisions)
                record['packed_verdict'] = packed['verdict']
                if packed['verdict'] != 'PASS':
                    record['stop_reason'] = 'packed veto failed'
            else:
                record['stop_reason'] = 'development gate failed'
        record.update(status='completed', wall_seconds=time.perf_counter()-start,
                      provenance='public_asset_diagnostic' if development and development['trials'] else 'video_only')
        publish(directory, queue, prepared, controls, decisions, development, packed, record)
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', wall_seconds=time.perf_counter()-start)
        publish(directory, queue, prepared, controls, decisions, development, packed, record)
        raise
    finally:
        record.update(finished_utc=now_utc(), wall_seconds=time.perf_counter()-start,
                      artifacts={p.relative_to(directory).as_posix():sha256_file(p)
                                 for p in directory.rglob('*') if p.is_file() and p != path})
        packed_directory = ROOT/'data/outputs/mesh_generative/g3-packed-check'
        if packed_directory.exists():
            record['packed_artifacts'] = {p.relative_to(packed_directory).as_posix():sha256_file(p)
                                         for p in packed_directory.rglob('*') if p.is_file()}
        write_json(path, record)
    return path
