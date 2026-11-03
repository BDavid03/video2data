"""Resumable post-hoc admission and generation-seed diagnostics, without packing."""

import argparse
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

FEATURE = Path(__file__).parent
QUEUE = 'track3/features/mesh_generative/insights_queue.json'
OUT = 'outputs/mesh_generative/insights'


def authorize() -> None:
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


def generation_worker(request: str) -> None:
    """Repeat the frozen worker's pipeline, changing only its explicitly queued seed."""
    runtime_flag(FEATURE, 'insights_authorized', 'Mesh insights generation')
    lock = json.loads((ROOT/'.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'mesh_generative':
        raise PermissionError('Parent handler must own the GPU lease')
    job = json.loads(project_path(ROOT, request).read_text())
    import importlib.metadata
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('Expected the frozen torch/CUDA versions')
    target = ROOT/job['mesh']
    if target.exists():
        raise FileExistsError('Refusing to overwrite generation')
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(ROOT/'.runtime/hunyuan-mesh/weights-xet'), subfolder='hunyuan3d-dit-v2-mini',
        variant='fp16', use_safetensors=True, dtype=torch.float16, device='cuda')
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    mesh = pipe(image=Image.open(ROOT/job['crop']).convert('RGB'), num_inference_steps=30,
                octree_resolution=380, mc_algo='mc',
                generator=torch.Generator(device='cuda').manual_seed(job['seed']))[0]
    torch.cuda.synchronize()
    finished = time.perf_counter()
    mesh.export(target)
    write_json(target.with_suffix('.json'), dict(seed=job['seed'], steps=30, octree_resolution=380,
               mc_algo='mc', load_seconds=loaded-start, generation_seconds=finished-loaded,
               total_seconds=finished-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
               peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
               vertices=len(mesh.vertices), faces=len(mesh.faces), device=torch.cuda.get_device_name(),
               preprocessing='white RGB crop; upstream ImageProcessorV2 defaults (border_ratio=0.15)',
               packages={p: importlib.metadata.version(p) for p in
                         ('torch', 'torchvision', 'diffusers', 'transformers', 'hy3dgen', 'scikit-image')}))


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
                            'track3.features.mesh_generative.insights', '--worker', str(request)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        lease.require_headroom()
        telemetry = lease.report()
    report = json.loads(report_path.read_text())
    report['gpu_lease'] = telemetry
    write_json(report_path, report)
    return report


def pair_inputs(row: dict) -> dict:
    """Name the original crop, cloud and exact scoring seeds for one recorded pair."""
    base = ROOT/'outputs/mesh_generative'/('g3' if row.get('g3_new') else 'g2')/row['key']
    seeds = ([1000003, 3000009, 5000015] if row['source_split'] == 'public'
             else [row['episode']*1000003+row['slot']*101])
    return dict(crop=str((base/'crop.png').relative_to(ROOT)),
                observed=str((base/'observed.npz').relative_to(ROOT)), seeds=seeds)


def visibility(row: dict) -> dict:
    """Measure mask contact in original frame coordinates and retain the assignment note."""
    import numpy as np
    from .g3_math import mask_checks

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
    rule = dict(boundary_margin_pixels=2, minimum_pixels=0, maximum_pixels=mask.size,
                maximum_border_fraction=1, minimum_observed_points=0)
    measured = mask_checks(mask, valid, valid, rule)
    edge = np.zeros_like(mask)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return dict(frame_border_mask_pixels=int((mask & edge).sum()),
                remap_invalid_mask_pixels=int((mask & ~valid).sum()),
                near_invalid_boundary_pixels=measured['near_invalid_boundary_pixels'],
                boundary_pixels=measured['boundary_pixels'], mask_pixels=int(mask.sum()),
                handle_pixels=None, handle_note=assignment.get('note', 'No handle annotation recorded'),
                interpretation='No isolated handle annotation; zero border contact does not establish a complete handle.')


def score(row: dict, mesh: str, directory: Path) -> dict:
    """Call the kit process and reuse its completed per-trial score artifacts."""
    result = directory/'score.json'
    if not result.exists():
        job = dict(object=row['object'], mesh=mesh, v4=row['v4_mesh'], seeds=pair_inputs(row)['seeds'],
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
    authorize()
    import numpy as np
    from ._admission import fit_residual

    queue = json.loads((ROOT/QUEUE).read_text())
    rows = [r for r in json.loads((ROOT/queue['admissions']).read_text())['pairs'] if r.get('generated_mesh')]
    expected = {('public', 21), ('public', 26), ('public', 42), ('evaluation', 3), ('evaluation', 5),
                ('evaluation', 6), ('evaluation', 8), ('evaluation', 20), ('evaluation', 28)}
    if {(r['source_split'], r['episode']) for r in rows} != expected or len(rows) != 9:
        raise PermissionError('Only the nine specified pot/pitcher pairs are authorized')
    if set(queue['seed_crops']) != {'public-000021-water_pitcher', 'public-000042-white_pot', 'evaluation-000005-white_pot'}:
        raise PermissionError('Seed crops differ from authorized pairs')
    if queue['generation_seeds'] != [12346, 12347, 12348, 12349] or queue['max_generations'] != 12:
        raise PermissionError('Only twelve new generations are authorized')
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
        inputs.extend(ROOT/pair_inputs(row)[name] for name in ('crop', 'observed'))
    record = run_fields(ROOT, 'mesh_generative', 'insights', queue, inputs=inputs)
    record['sources'] += [input_file(ROOT, ROOT/p) for p in
                          ('scripts/track3/local_cdo.py', 'upstream/v2d_submission_kit/v2d_submission_kit/v2dlb/mesh_budget.py',
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
    done = {r['trial']: r for line in ledger.read_text().splitlines() if (r := json.loads(line))} if ledger.exists() else {}
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
                generation = None if seed == 12345 else generate(dict(mesh=mesh, seed=seed, crop=pair_inputs(row)['crop']), item)
                measured = score(row, mesh, item)
                residual = row['generated_fit']
                if seed != 12345:
                    with np.load(item/'budgeted.npz', allow_pickle=False) as a, np.load(ROOT/pair_inputs(row)['observed'], allow_pickle=False) as cloud:
                        residual = fit_residual(a['vertices'], a['faces'], cloud['points'], fit)
                ratio = residual['median_residual_m']/row['v4_residual_m']
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
        publish(directory, list(done.values()), queue, record)
        record.update(status='completed', completed_trials=len(done), finished_utc=now_utc())
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now_utc())
        raise
    finally:
        record['artifacts'] = {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob('*')
                               if p.is_file() and p.name != 'run.json'}
        write_json(path, record)
    return path


def publish(directory: Path, rows: list[dict], queue: dict, record: dict) -> None:
    """Write the post-hoc tables, descriptive seed statistics and two labelled figures."""
    import numpy as np
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    original = [r for r in rows if r['generation_seed'] == 12345]
    rho = spearmanr([r['ratio'] for r in original], [r['score']['change_cm'] for r in original])
    pot = [r['score']['generated_cm'] for r in original if r['object'] == 'white_pot']
    spreads = []
    for key in queue['seed_crops']:
        group = [r for r in rows if r['key'] == key]
        result = dict(key=key, n=len(group))
        for name, values in [('cdo_cm', [r['score']['generated_cm'] for r in group]), ('ratio', [r['ratio'] for r in group])]:
            result[name] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                                min=float(np.min(values)), max=float(np.max(values)))
        spreads.append(result)
    within = float(np.mean([r['cdo_cm']['sd']**2 for r in spreads if 'white_pot' in r['key']]))
    between = float(np.var(pot, ddof=1))
    packed = json.loads((ROOT/queue['packed_scores']).read_text())
    for r in original:
        if r['source_split'] == 'evaluation':
            pair = next(p for p in packed['pairs'] if p['episode'] == r['episode'] and p['object'] == r['object'])
            if abs(pair['v4']-r['score']['v4_cm']) > 1e-12:
                raise ValueError('Recomputed v4 differs from the recorded packed pair')
            if r['key'] in ('evaluation-000005-white_pot', 'evaluation-000020-water_pitcher') and abs(pair['candidate']-r['score']['generated_cm']) > 1e-12:
                raise ValueError('Recomputed generated mesh differs from shipped v5')
    counterfactual = [dict(r) for r in packed['pairs']]
    for pair in counterfactual:
        row = next((r for r in original if r['source_split'] == 'evaluation' and
                    r['episode'] == pair['episode'] and r['object'] == pair['object']), None)
        pair['all_generated'] = row['score']['generated_cm'] if row else pair['candidate']
        pair['combined_admission'] = (row['score']['generated_cm'] if row['historical_decision'] == 'generated'
                                     else row['score']['v4_cm']) if row else pair['candidate']
    objects = [dict(object=name, **{method: float(np.mean([r[method] for r in counterfactual if r['object'] == name]))
                                   for method in ('v4', 'candidate', 'combined_admission', 'all_generated')})
               for name in sorted({r['object'] for r in counterfactual})]
    means = {method: dict(pair_weighted_cm=float(np.mean([r[method] for r in counterfactual])),
                         equal_object_cm=float(np.mean([r[method] for r in objects])))
             for method in ('v4', 'candidate', 'combined_admission', 'all_generated')}
    generations = [r['generation'] for r in rows if r['generation']]
    resources = dict(wall_seconds=record['wall_seconds'], new_generations=len(generations),
                     peak_allocated_mib=max(g['peak_allocated_mib'] for g in generations),
                     peak_reserved_mib=max(g['peak_reserved_mib'] for g in generations),
                     peak_sampled_device_mib=max(g['gpu_lease']['peak_sampled_total_device_used_mib'] for g in generations))
    summary = dict(disclosure=queue['disclosure'], spearman=dict(rho=float(rho.statistic), pvalue=float(rho.pvalue), n=len(original)),
                   seed_spread=spreads, seed_sd_convention='sample SD (ddof=1), five generation seeds per unchanged crop',
                   pot_between_pair_variance_cm2=between, pot_pooled_within_crop_variance_cm2=within,
                   seed_fraction_of_between_pair_variance=within/between, means=means, objects=objects,
                   counterfactual_pairs=counterfactual, resources=resources, protected_identical=record['protected_identical'])
    write_json(directory/'summary.json', summary)
    lines = ['# Pretrained-shape diagnostics', '', queue['disclosure'], '',
             'Lower CD-O is better; change = generated minus v4. Public views average the recorded three development scoring seeds; evaluation views use their exact packed pair seed.',
             'A border contact is counted on the input mask in original frame coordinates, before square padding; near-invalid uses the frozen two-pixel rule. Handle pixels are unmeasured; notes are the recorded assignments.', '',
             '| Pair | Border mask px | Remap-invalid mask px | Near-invalid boundary px | Handle note | Ratio | Recorded decision | Generated cm | V4 cm | Change cm |',
             '|---|---:|---:|---:|---|---:|---|---:|---:|---:|']
    for r in original:
        v, s = r['visibility'], r['score']
        lines.append(f"| {r['key']} | {v['frame_border_mask_pixels']} | {v['remap_invalid_mask_pixels']} | {v['near_invalid_boundary_pixels']}/{v['boundary_pixels']} | {v['handle_note'].replace('|', '/')} | {r['ratio']:.5f} | {r['historical_decision']} | {s['generated_cm']:.8f} | {s['v4_cm']:.8f} | {s['change_cm']:+.8f} |")
    lines += ['', f"Spearman rho = {rho.statistic:.6f}, n = {len(original)}, nominal p = {rho.pvalue:.6f}. Repeated scans and post-pilot selection preclude an independent-sample interpretation.", '',
              '| All 26 covered evaluation pairs / 10 objects | Pair mean cm | Equal-object mean cm |', '|---|---:|---:|']
    lines += [f"| {name} | {v['pair_weighted_cm']:.8f} | {v['equal_object_cm']:.8f} |" for name, v in means.items()]
    lines += ['', 'candidate is shipped v5; combined_admission is the failed G3 candidate. all_generated admits every generated evaluation shape, including rejected shapes; missing-generation pairs and every other object keep v5.', '',
              '| Object | V4 cm | V5 cm | Combined admission cm | All generated cm |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['object']} | {r['v4']:.8f} | {r['candidate']:.8f} | {r['combined_admission']:.8f} | {r['all_generated']:.8f} |" for r in objects]
    lines += ['', '| Unchanged crop (five generation seeds) | CD-O mean cm | SD cm | Min cm | Max cm | Ratio mean | SD | Min | Max |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in spreads:
        c, a = r['cdo_cm'], r['ratio']
        lines.append(f"| {r['key']} | {c['mean']:.8f} | {c['sd']:.8f} | {c['min']:.8f} | {c['max']:.8f} | {a['mean']:.5f} | {a['sd']:.5f} | {a['min']:.5f} | {a['max']:.5f} |")
    lines += ['', f"Seed-only variance fraction = {within/between:.2%}: mean of the two pot crops' within-crop sample variances ({within:.8f} cm²), divided by the sample variance across seven original pot views ({between:.8f} cm²). This descriptive ratio is not a causal variance decomposition; only two pot crops were replicated.",
              'The scoring seeds are held fixed within each crop. They differ across public and evaluation views; cross-crop spread therefore also includes CD-O sampling variation.', '',
              f"Resources: {resources}. Historical files unchanged: {record['protected_identical']} ({len(record['protected_before'])} hashes).",
              'No pack, admission threshold, crop, original generation or historical decision was changed.', '',
              '![Admission and CD-O](admission_vs_cdo.png)', '', '![Generation seed spread](seed_spread.png)']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for r in original:
        admitted = r['historical_decision'] == 'generated'
        ax.scatter(r['ratio'], r['score']['change_cm'], color='0.25' if admitted else '0.6', marker='o' if admitted else 'x')
        label = f"{'Public' if r['source_split']=='public' else 'Eval'} {r['episode']} {'pitcher' if r['object']=='water_pitcher' else 'pot'}"
        offset = {('evaluation', 20): (-104, 8), ('public', 21): (5, -16),
                  ('public', 26): (5, -16)}.get((r['source_split'], r['episode']), (5, 7))
        ax.annotate(label, (r['ratio'], r['score']['change_cm']), xytext=offset, textcoords='offset points', fontsize=9)
    ax.axhline(0, color='0.7', linewidth=0.8)
    ax.axvline(1.5, color='0.6', linewidth=0.8, linestyle='--')
    ax.text(1.52, ax.get_ylim()[0]+0.03, 'Admission limit = 1.5', fontsize=9, color='0.35')
    ax.set(xlabel='Video residual ratio: generated / v4 (unitless)', ylabel='CD-O change from v4 (cm; lower is better)',
           title=f'Video fit and scan error are compared for nine generated shapes (Spearman ρ = {rho.statistic:.2f}).')
    ax.text(0.99, 0.02, 'Circles: admitted    Crosses: rejected\nReused public scans; post-hoc diagnostic', transform=ax.transAxes, ha='right', fontsize=9)
    ax.margins(x=0.18, y=0.2)
    fig.savefig(directory/'admission_vs_cdo.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for x, key in enumerate(queue['seed_crops']):
        group = sorted([r for r in rows if r['key'] == key], key=lambda r:r['generation_seed'])
        for ax, values in zip(axes, ([r['score']['generated_cm'] for r in group], [r['ratio'] for r in group])):
            for offset, r, value in zip(np.linspace(-0.14, 0.14, len(group)), group, values):
                ax.scatter(x+offset, value, color='0.25', s=20)
                ax.annotate(str(r['generation_seed']), (x+offset, value), xytext=(0, 7), textcoords='offset points',
                            fontsize=8, rotation=90, ha='center')
            ax.set_xticks(range(3), ['Public 21\npitcher', 'Public 42\npot', 'Evaluation 5\npot'])
            ax.set_xlabel('Unchanged input crop (five generation seeds each)')
            ax.margins(x=0.2, y=0.16)
    axes[0].set_ylabel('Public-scan CD-O (cm; lower is better)')
    axes[1].set_ylabel('Video residual ratio: generated / v4 (unitless)\nDashed line: admission limit = 1.5')
    axes[1].axhline(1.5, color='0.6', linewidth=0.8, linestyle='--')
    fig.suptitle('Changing only the generation seed changes scan error and video fit for the same crop.')
    fig.savefig(directory/'seed_spread.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    generation_worker(parser.parse_args().worker)
