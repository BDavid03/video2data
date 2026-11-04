"""Resumable post-hoc admission and generation-seed diagnostics, without packing."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from track3.gpu import GpuLease
from track3.paths import PROJECT as ROOT, output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.runs import CACHE_FOLDERS, now_utc, runtime_flag
from .g2 import append
from track3.features.mesh_extents.proxy import read_ledger
from ._insights_stats import admission_ratio, resources, summarize, visibility_counts
from .insights_report import publish

FEATURE = Path(__file__).parent
QUEUE = 'track3/features/mesh_generative/insights_queue.json'
OUT = 'outputs/mesh_generative/insights'


def authorize() -> dict:
    """Gate execution and redirect numerical-library caches into the checkout."""
    runtime = runtime_flag(FEATURE, 'insights_authorized', 'Mesh pretrained-shape insights')
    if Path(sys.executable).resolve() != project_path(ROOT, runtime['python']):
        raise RuntimeError('Use the main project interpreter')
    for key, folder in CACHE_FOLDERS.items():
        folder = '.runtime/tmp/mesh-insights' if key in ('TMP', 'TEMP', 'TMPDIR') else folder
        path = project_path(ROOT, folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = os.environ['TMPDIR']
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1',
                      PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'upstream/v2d_submission_kit/v2d_submission_kit'))
    return runtime


def generate(job: dict, directory: Path) -> dict:
    """Generate once under the lease and preserve interrupted attempts for inspection."""
    target = ROOT/job['mesh']
    report_path = target.with_suffix('.json')
    if target.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if 'gpu_lease' not in report:
            raise RuntimeError('Incomplete generation telemetry; preserve this attempt')
        return report
    marker = directory/'generation.started.json'
    if marker.exists() or target.exists():
        raise RuntimeError('Incomplete generation; no automatic reroll')
    request = directory/'generation-request.json'
    write_json(request, job)
    with GpuLease(ROOT, 1024, 10000, stage='mesh_generative') as lease:
        write_json(marker, dict(started_utc=now_utc(), request=input_file(ROOT, request)))
        with (directory/'generation.log').open('w') as log:
            subprocess.run([str(ROOT/'.runtime/hunyuan-mesh/venv/bin/python'), '-B', '-m',
                            'track3.features.mesh_generative.insights_worker', '--request', str(request)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        lease.require_headroom()
        telemetry = lease.report()
    report = json.loads(report_path.read_text())
    report['gpu_lease'] = telemetry
    write_json(report_path, report)
    return report


def pair_inputs(row: dict, queue: dict) -> dict:
    """Name the original crop, cloud and exact scoring seeds for one recorded pair."""
    base = ROOT/'outputs/mesh_generative'/('g3' if row.get('g3_new') else 'g2')/row['key']
    seeds = next(p['scoring_seeds'] for p in queue['pairs'] if p['key'] == row['key'])
    return dict(crop=str((base/'crop.png').relative_to(ROOT)),
                observed=str((base/'observed.npz').relative_to(ROOT)), seeds=seeds)


def visibility(row: dict) -> dict:
    """Measure mask contact in original frame coordinates and retain the assignment note."""
    import numpy as np

    assignments = json.loads((ROOT/'track3/features/mesh_extents/assignments.json').read_text())
    assignment = assignments['episodes'][f"{row['source_split']}:{row['episode']}"]
    if row.get('g3_new'):
        base = ROOT/'outputs/mesh_generative/g3'/row['key']
        with np.load(base/'mask.npz', allow_pickle=False) as a:
            mask = a['mask']
        with np.load(base/'frame.npz', allow_pickle=False) as a:
            valid = a['remap_valid']
    else:
        prepared = row['prepared']
        with np.load(ROOT/row['measurement'].replace('run.json', 'labels_000000.npz'), allow_pickle=False) as a:
            mask = np.isin(a['label'], np.array(prepared['cluster_ids'])+1)
        prep = next(e['path'] for e in prepared['inputs'] if e['path'].endswith('preprocessing.npz'))
        with np.load(ROOT/prep, allow_pickle=False) as a:
            valid = a['input_valid']
    return dict(**visibility_counts(mask, valid),
                handle_pixels=None, handle_note=assignment.get('note', 'No handle annotation recorded'),
                interpretation='No isolated handle annotation; zero border contact does not establish a complete handle.')


def score(row: dict, mesh: str, directory: Path, queue: dict) -> dict:
    """Call the kit process and reuse its completed per-trial score artifacts."""
    result = directory/'score.json'
    if not result.exists():
        job = dict(object=row['object'], mesh=mesh, v4=row['v4_mesh'], seeds=pair_inputs(row, queue)['seeds'],
                   output=str(result.relative_to(ROOT)), arrays=str((directory/'budgeted.npz').relative_to(ROOT)))
        request = directory/'score-request.json'
        write_json(request, job)
        with (directory/'score.log').open('w') as log:
            subprocess.run([str(ROOT/'.runtime/kit-env/bin/python'), '-B', '-m',
                            'track3.features.mesh_generative.insights_cpu', '--request', str(request)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(result.read_text())


def protected_files(queue: dict) -> list[Path]:
    """Inventory historical campaigns, production PLYs, protocols and the shipped pack."""
    files = [p for name in ('g1-pilot', 'g2', 'g3', 'g2-packed-check', 'g2-real-pack')
             for p in (ROOT/'outputs/mesh_generative'/name).rglob('*') if p.is_file()]
    files += [ROOT/'docs/chain/mesh-generative-protocol.md', ROOT/'docs/chain/mesh-generative-g3-protocol.md']
    manifest = json.loads((ROOT/queue['v5_manifest']).read_text())
    if sha256_file(ROOT/manifest['file']['path']) != manifest['file']['sha256']:
        raise ValueError('Shipped v5 differs from its manifest')
    return sorted(set(files))


def run() -> Path:
    """Run the bounded queue once, resuming scored trials by their frozen identity."""
    existing = ROOT/OUT/'run.json'
    runtime_flag(FEATURE, 'insights_authorized', 'Mesh insights')
    if existing.exists() and json.loads(existing.read_text())['status'] == 'completed':
        return existing
    runtime = authorize()
    import numpy as np
    from ._admission import fit_residual

    queue = json.loads((ROOT/QUEUE).read_text())
    for pair in queue['pairs']:
        if pair['episode'] in (41, 43, 44) or pair['episode'] not in runtime['g2_authorized_episode_indices'][pair['source_split']]:
            raise PermissionError('Pair outside the existing episode allow-list')
    rows = [r for r in json.loads((ROOT/queue['admissions']).read_text())['pairs'] if r.get('generated_mesh')]
    expected = {(p['source_split'], p['episode'], p['object']) for p in queue['pairs']}
    if {(r['source_split'], r['episode'], r['object']) for r in rows} != expected or len(rows) != len(expected):
        raise PermissionError('Admission rows differ from the diagnostic queue')
    if not set(queue['seed_crops']) <= {p['key'] for p in queue['pairs']}:
        raise PermissionError('Seed crops differ from the diagnostic roster')
    if len(queue['generation_seeds'])*len(queue['seed_crops']) != queue['max_generations']:
        raise PermissionError('Generation count differs from the queue budget')
    directory = output_path(ROOT, OUT)
    directory.mkdir(parents=True, exist_ok=True)
    protected = {str(p.relative_to(ROOT)): sha256_file(p) for p in protected_files(queue)}
    inputs = [ROOT/QUEUE, ROOT/queue['admissions'], ROOT/queue['fit_queue'], ROOT/queue['packed_scores'],
              FEATURE/'runtime.json', ROOT/'track3/features/mesh_extents/assignments.json',
              ROOT/'.runtime/hunyuan-mesh/weights-selected.json', ROOT/'.runtime/hunyuan-mesh/requirements.lock.txt',
              ROOT/'.runtime/hunyuan-mesh/weights-xet/hunyuan3d-dit-v2-mini/config.yaml',
              ROOT/'.runtime/hunyuan-mesh/weights-xet/hunyuan3d-dit-v2-mini/model.fp16.safetensors']
    for row in rows:
        if sha256_file(ROOT/row['generated_mesh']) != row['generated_mesh_sha256']:
            raise ValueError('Recorded generated mesh changed')
        inputs.extend(ROOT/pair_inputs(row, queue)[name] for name in ('crop', 'observed'))
    record = run_fields(ROOT, 'mesh_generative', 'insights', queue, inputs=inputs)
    record['sources'] += [input_file(ROOT, ROOT/p) for p in
                          ('track3/features/mesh_eval/local_cdo.py', 'upstream/v2d_submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py',
                           'upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/CD-O.py')]
    path = directory/'run.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != record[k] for k in ('inputs', 'sources', 'request')) or old['protected_before'] != protected:
            raise ValueError('Frozen diagnostic identity changed')
        record = old
        if record['status'] == 'completed':
            return path
    record.update(status='running', started_utc=record.get('started_utc', now_utc()),
                  protected_before=protected, provenance='public_asset_diagnostic', disclosure=queue['disclosure'])
    write_json(path, record)
    ledger = directory/'ledger.jsonl'
    done = {r['trial']: r for r in read_ledger(ledger)}
    fit = json.loads((ROOT/queue['fit_queue']).read_text())['fit']
    started = time.perf_counter()
    try:
        for row in rows:
            for seed in [12345] + (queue['generation_seeds'] if row['key'] in queue['seed_crops'] else []):
                trial = f"{row['key']}/seed-{seed}"
                if trial in done:
                    for entry in done[trial]['artifacts']:
                        if sha256_file(ROOT/entry['path']) != entry['sha256']:
                            raise ValueError('Completed trial artifact changed')
                    continue
                if time.perf_counter()-started > queue['time_budget_seconds']:
                    raise TimeoutError('Queue time budget reached; completed trials remain resumable')
                item = directory/trial
                item.mkdir(parents=True, exist_ok=True)
                mesh = row['generated_mesh'] if seed == 12345 else str((item/'raw.ply').relative_to(ROOT))
                generation = None if seed == 12345 else generate(dict(mesh=mesh, seed=seed, crop=pair_inputs(row, queue)['crop']), item)
                measured = score(row, mesh, item, queue)
                residual = row['generated_fit']
                if seed != 12345:
                    with np.load(item/'budgeted.npz', allow_pickle=False) as a, np.load(ROOT/pair_inputs(row, queue)['observed'], allow_pickle=False) as cloud:
                        residual = fit_residual(a['vertices'], a['faces'], cloud['points'], fit)
                ratio = admission_ratio(residual, row['v4_residual_m'])
                result = dict(trial=trial, key=row['key'], object=row['object'], source_split=row['source_split'],
                              episode=row['episode'], generation_seed=seed, historical_decision=row['decision'],
                              ratio=ratio, hypothetical_admitted=ratio <= 1.5, fit=residual, score=measured,
                              visibility=visibility(row), generation=generation, mesh=mesh,
                              artifacts=[input_file(ROOT, p) for p in (ROOT/mesh, item/'score.json', item/'budgeted.npz')],
                              status='completed', completed_utc=now_utc())
                append(ledger, result)
                done[trial] = result
                print(f"{trial}: CD-O={measured['generated_cm']:.8f} cm ratio={ratio:.5f}", flush=True)
        record.update(wall_seconds=record.get('wall_seconds', 0)+time.perf_counter()-started,
                      protected_after={p: sha256_file(ROOT/p) for p in protected})
        record['protected_identical'] = record['protected_before'] == record['protected_after']
        if not record['protected_identical']:
            raise ValueError('Historical files changed during diagnostics')
        publish(directory, list(done.values()), queue, summarize(list(done.values()), queue,
                json.loads((ROOT/queue['packed_scores']).read_text()),
                resources(list(done.values()), record['wall_seconds']), record['protected_identical']))
        record.update(status='completed', completed_trials=len(done), finished_utc=now_utc())
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now_utc())
        raise
    finally:
        record['artifacts'] = {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob('*')
                               if p.is_file() and p.name != 'run.json'}
        write_json(path, record)
    return path
