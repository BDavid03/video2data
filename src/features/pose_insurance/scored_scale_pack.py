"""Compose accepted S1/v2 exports, pack with the kit and verify CPU reproduction."""
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
from src.paths import output_path
from src.provenance import run_fields, write_json, sha256_file, input_file
from src.runs import ROOT, read_json, runtime_flag, heavy_execution, recorded, source_hashes
from .scored_scale_report import report

FEATURE = Path(__file__).parent
OUTPUT = ROOT / 'data/outputs/pose_insurance/v3'
KIT = ROOT / 'upstream/v2d_submission_kit/v2d_submission_kit'


def compose(plans: dict, decisions: dict) -> tuple:
    """Copy twenty selected exports by content without altering any producer."""
    directory = output_path(ROOT, 'data/outputs/pose_insurance/results-v3')
    directory.mkdir(parents=True, exist_ok=True)
    exports, composition = [], dict(S1=[], v2=[])
    for ep in sorted(plans):
        decision, plan = decisions[ep], plans[ep]
        source = ROOT / (decision['trajectory'] if decision['accepted'] else plan['base_export'])
        expected = decision['trajectory_sha256'] if decision['accepted'] else plan['base_hash']
        if sha256_file(source) != expected:
            raise ValueError('Selected export changed')
        destination = directory / f'episode_{ep:06d}.parquet'
        if destination.exists() and sha256_file(destination) != expected:
            raise ValueError('An existing v3 export has different content')
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256_file(destination) != expected:
            raise ValueError('Export copy differs from the selected producer')
        exports.append(dict(path=str(destination.relative_to(ROOT)), source=str(source.relative_to(ROOT)), sha256=expected))
        composition['S1' if decision['accepted'] else 'v2'].append(ep)
    if set(directory.iterdir()) != {ROOT / e['path'] for e in exports} or len(exports) != 20:
        raise ValueError('The flat results must contain exactly twenty Parquets')
    return exports, composition


def runtime_sources() -> list:
    """Pin portable copies of the read-only FlashCHORD source used by the runtime."""
    runtime = ROOT / 'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord'
    inventory = read_json(OUTPUT.parent / 'runtime_inventory.json')
    for path, digest in inventory['files'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'Pinned runtime source differs: {path}')
    entries = []
    for folder in ('src', 'scripts', 'data/configs'):
        for path in sorted((runtime / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
                destination = OUTPUT / 'runtime_sources' / path.relative_to(runtime)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha256_file(destination) != sha256_file(path):
                    raise ValueError('Runtime snapshot changed')
                if not destination.exists():
                    shutil.copyfile(path, destination)
                entries.append(dict(input_file(ROOT, destination), runtime_origin=str(path.relative_to(ROOT))))
    return entries


def pack(plans: dict, decisions: dict, queue: dict) -> None:
    """Write an admitted v3 manifest and run the unchanged verifier and official kit."""
    runtime = runtime_flag(FEATURE, 'pack_authorized', 'D2 v3 packing')
    heavy_execution(dict(runtime, python=runtime['kit_python']))
    if not any(d['accepted'] for d in decisions.values()):
        result = dict(status='not packed', reason='No scored episode changed after no-op/validity/physical fallbacks')
        report(plans, decisions, result)
        write_json(OUTPUT / 'finished.json', dict(status='completed', package=result))
        return
    exports, composition = compose(plans, decisions)
    snapshots = runtime_sources()
    commit = subprocess.check_output(['git', 'rev-parse', 'bb8a3f0'], cwd=ROOT, text=True).strip()
    if commit != queue['code_commit']:
        raise ValueError('Requested code commit changed')
    url = f'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/{commit}'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')
    directory = output_path(ROOT, f'submission/{stamp}-d2-v3')
    csv_path = directory / 'track3_poses_v3.csv'
    commands = [[str(ROOT / runtime['kit_python']), str(KIT / 'tools/pack_submission.py'), '--track', 'track_3',
                 '--results', str(ROOT / 'data/outputs/pose_insurance/results-v3'), '--code-commit', url,
                 '--eval-script', str(KIT / 'flash_chord/scripts/evaluate_policy.py'), '--skip-crosscheck', '--out', str(csv_path)],
                [str(ROOT / runtime['kit_python']), str(KIT / 'tools/preflight.py'), '--track', 'track_3',
                 '--results', str(ROOT / 'data/outputs/pose_insurance/results-v3'), '--skip-kaggle']]
    parents = [d['run'] for d in decisions.values() if d['accepted']]
    parents += sorted({p['base_run'] for p in plans.values()})
    inputs = [ROOT / 'docs/chain/pose-scale-protocol.md', ROOT / 'docs/chain/pose-placement-protocol.md',
              FEATURE / 'queue_v3.json', FEATURE / 'runtime.json', FEATURE.parent / 'stereo_depth/runtime.json',
              ROOT / 'scripts/reproduce/d2_poses.sh', ROOT / 'scripts/reproduce/README.md', ROOT / 'scripts/reproduce/verify.sh',
              KIT / 'tools/pack_submission.py', KIT / 'tools/preflight.py', KIT / 'flash_chord/scripts/evaluate_policy.py',
              OUTPUT / 'plans.json', OUTPUT / 'decisions.json', OUTPUT / 'ledger.jsonl',
              ROOT / 'data/outputs/pose_insurance/runtime_inventory.json', *[ROOT / e['path'] for e in exports],
              *[ROOT / e['source'] for e in exports], *[ROOT / e['path'] for e in snapshots],
              *sorted(OUTPUT.glob('stereo_*.json'))]
    record = run_fields(ROOT, 'pose_insurance', 'pack', dict(version='v3', budget=queue['budget'], code_commit=url),
                        parents=parents, inputs=inputs)
    with recorded(directory, record, hash_artifacts=False):
        for command, name in zip(commands, ('pack.log', 'preflight.log')):
            with (directory / name).open('w') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        with (KIT / 'data/track_3_sample_submission.csv').open() as stream:
            roster = [r['row_id'] for r in csv.DictReader(stream)]
        if len(rows) != 9120 or [r['row_id'] for r in rows] != roster:
            raise ValueError('The v3 CSV must match all 9,120 official roster rows')
        if 'Ready to submit' not in (directory / 'preflight.log').read_text():
            raise ValueError('Kit preflight did not report Ready to submit')
        record['artifacts'] = {name: sha256_file(directory / name) for name in ('track3_poses_v3.csv', 'pack.log', 'preflight.log')}
    sources = list(record['sources'])
    sources += [input_file(ROOT, ROOT / p) for p in source_hashes(FEATURE.parent / 'stereo_depth')]
    manifest = dict(version='v3', file=str(csv_path.relative_to(ROOT)), sha256=sha256_file(csv_path), rows=len(rows),
                    commands=commands, budget=queue['budget'], results_dir='data/outputs/pose_insurance/results-v3',
                    exports=exports, composition=composition, code_commit_url=url, head=commit,
                    preflight_verdict='Ready to submit', source_hashes=sources, inputs=record['inputs'],
                    runtime_source_snapshots=snapshots, run_record=input_file(ROOT, directory / 'run.json'),
                    first_pose_and_drift={str(ep): d.get('checks') for ep, d in decisions.items()},
                    note='Owner placeholder URL at the adopted decision commit; working implementation source hashes are pinned. No upload.')
    write_json(directory / 'manifest.json', manifest)
    verify = ['bash', str(ROOT / 'scripts/reproduce/verify.sh'), str(directory / 'manifest.json')]
    with (directory / 'verify.log').open('w') as stream:
        subprocess.run(verify, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    verification_summary = (directory / 'verify.log').read_text().splitlines()[-1]
    command = ['bash', str(ROOT / 'scripts/reproduce/d2_poses.sh'), url, '--manifest', str(directory / 'manifest.json'), '--run']
    completed = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True, capture_output=True)
    (OUTPUT / 'reproduction.log').write_text(completed.stdout + completed.stderr)
    completed.check_returncode()
    reproduced_path = Path(completed.stdout.strip().splitlines()[-1])
    reproduced = read_json(reproduced_path)
    comparison = dict(identical=reproduced['sha256'] == manifest['sha256'], actual_sha256=reproduced['sha256'],
                      expected_sha256=manifest['sha256'], manifest=str(reproduced_path.relative_to(ROOT)))
    if not comparison['identical']:
        raise ValueError('CPU reproduction changed the packed CSV')
    write_json(OUTPUT / 'reproduction_comparison.json', comparison)
    result = dict(manifest, status='packed', manifest=str((directory / 'manifest.json').relative_to(ROOT)),
                  verification_summary=verification_summary, reproduction=comparison)
    report(plans, decisions, result)
    write_json(OUTPUT / 'finished.json', dict(status='completed', manifest=result['manifest'],
               sha256=manifest['sha256'], composition=composition, verification=verification_summary, reproduction=comparison))
    print(directory / 'manifest.json', flush=True)


if __name__ == '__main__':
    plans = {int(k): v for k, v in read_json(OUTPUT / 'plans.json')['plans'].items()}
    decisions = {int(k): v for k, v in read_json(OUTPUT / 'decisions.json').items()}
    pack(plans, decisions, read_json(FEATURE / 'queue_v3.json'))
