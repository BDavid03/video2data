"""Verify and reproduce v4 exports on CPU, with optional packing after the lead commits."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
from track3.paths import output_path
from track3.provenance import input_file, sha256_file, write_json
from track3.runs import ROOT, artifact, checked_entry, new_run_id, producer, read_json
from .scene import KIT
from .scored_scale_handler import inspect
from .scored_stereo_loop import BUDGET, EPISODES, UNTOUCHED

DEFAULT = 'outputs/pose_insurance/stereo-placement-scored/composition.json'


def verify(manifest: dict) -> None:
    """Admit every export against its policy record, physical evidence and exact source hash."""
    changed, fallback = manifest['composition']['T1'], manifest['composition']['v3']
    expected = set((*EPISODES, *UNTOUCHED))
    episodes = [int(Path(row['path']).stem.split('_')[1]) for row in manifest['exports']]
    if (manifest['version'] != 'v4' or manifest['budget'] != BUDGET or len(episodes) != 20
            or len(set(episodes)) != 20 or set(episodes) != expected or set(changed) & set(fallback)
            or set(changed + fallback) != expected or not set(changed).issubset(EPISODES)):
        raise ValueError('v4 composition must partition exactly the twenty scored episodes')
    for entry in [*manifest['inputs'], *manifest.get('gpu_runs', []), manifest['reproduction_script']]:
        checked_entry(entry, {'inputs': []})
    for row, ep in zip(manifest['exports'], episodes):
        source = (ROOT / row['source']).resolve()
        source.relative_to(ROOT)
        if ep in changed:
            valid = source.is_relative_to(ROOT / 'outputs/pose_insurance/stereo-placement-scored/runs') and source.name == 'trajectory.parquet'
        else:
            valid = source == ROOT / f'outputs/pose_insurance/results-v3/episode_{ep:06d}.parquet'
        if not valid or sha256_file(source) != row['sha256'] or sha256_file(ROOT / row['path']) != row['sha256']:
            raise ValueError(f'Export source or composed hash mismatch: {ep}')
        policy = producer(row['run'], 'pose_insurance.run')
        if policy[1]['episode_index'] != ep or policy[1]['split'] != 'evaluation' or policy[1]['request']['budget'] != BUDGET:
            raise ValueError('Export policy does not match its episode or budget')
        trajectory = artifact(policy, 'trajectory.parquet', {'inputs': []})
        build_entry = next(entry for entry in policy[1]['inputs'] if '/build' in entry['path'] and entry['path'].endswith('/run.json'))
        checked_entry(build_entry, {'inputs': []})
        build = producer(build_entry['path'], 'pose_insurance.build')
        if build[1]['scene'] != policy[1]['scene']:
            raise ValueError('Policy and task scene differ')
        if sha256_file(trajectory) != row['sha256'] or not inspect(trajectory, build[1])['accepted']:
            raise ValueError('Policy trajectory fails the existing physical/export check')
        if ep in changed and (policy[1]['request'].get('candidate') != 'T1'
                              or policy[1]['optimizer_updates'] < 1 or policy[1]['actor_adam_steps'] < 1):
            raise ValueError('T1 export has no admitted learned policy')


def committed_sources(manifest: dict, url: str) -> str:
    """Require the supplied real GitHub commit to contain the exact recorded computation sources."""
    match = re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/([0-9a-f]{40})', url)
    if match is None or 'PLACEHOLDER' in url:
        raise ValueError('Supply a real GitHub commit URL with a full forty-character SHA')
    sha = match[1]
    sources = dict(manifest['source_hashes'])
    sources[manifest['reproduction_script']['path']] = manifest['reproduction_script']['sha256']
    for path, digest in sources.items():
        content = subprocess.check_output(['git', 'show', f'{sha}:{path}'], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'The supplied commit differs from the recorded source: {path}')
    return sha


def reproduce(manifest: dict, directory: Path) -> Path:
    """Copy all twenty admitted exports into a fresh CPU reproduction directory."""
    results = directory / 'results'
    results.mkdir(parents=True, exist_ok=False)
    for row in manifest['exports']:
        target = results / Path(row['path']).name
        shutil.copyfile(ROOT / row['source'], target)
        if sha256_file(target) != row['sha256']:
            raise ValueError('Reproduced export differs from its admitted source')
    return results


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default=DEFAULT)
    parser.add_argument('--version', choices=('v4',), default='v4')
    parser.add_argument('--code-commit')
    parser.add_argument('--out')
    parser.add_argument('--run', action='store_true', help='compose verified exports and pack with a committed code URL')
    parser.add_argument('--compose-only', action='store_true', help='reproduce exports and run local preflight without packing')
    args = parser.parse_args()
    if args.run and args.compose_only:
        parser.error('Choose --run or --compose-only')
    if args.run and (not args.code_commit or not args.out):
        parser.error('--run requires --code-commit and --out')
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    manifest = read_json(args.manifest)
    verify(manifest)
    print('Verified 20 exports against admitted policies, source hashes and physical checks.', flush=True)
    if not args.run and not args.compose_only:
        return
    # Packing is deliberately a separate command after the lead's commit.
    sha = committed_sources(manifest, args.code_commit) if args.run else None
    if args.run and not manifest['composition']['T1']:
        raise ValueError('No accepted T1 changes: retain D2 v3 instead of packing v4')
    directory = output_path(ROOT, f'.runtime/tmp/t1-scored/reproduce/{new_run_id()}')
    results = reproduce(manifest, directory)
    output = output_path(ROOT, args.out) if args.run else None
    commands = []
    if output is not None:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        commands.append([str(ROOT / '.runtime/kit-env/bin/python'), str(KIT / 'tools/pack_submission.py'),
                         '--track', 'track_3', '--results', str(results), '--code-commit', args.code_commit,
                         '--eval-script', str(KIT / 'flash_chord/scripts/evaluate_policy.py'),
                         '--skip-crosscheck', '--out', str(output)])
    commands.append([str(ROOT / '.runtime/kit-env/bin/python'), str(KIT / 'tools/preflight.py'),
                     '--track', 'track_3', '--results', str(results), '--skip-kaggle'])
    for index, command in enumerate(commands):
        log = directory / ('preflight.log' if index == len(commands)-1 else 'pack.log')
        with log.open('w') as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    if 'Ready to submit.' not in (directory / 'preflight.log').read_text():
        raise ValueError('Kit preflight did not report Ready to submit')
    result = dict(manifest, results_dir=str(results.relative_to(ROOT)), commands=commands,
                  source_manifest=input_file(ROOT, args.manifest), preflight_verdict='Ready to submit (Kaggle checks skipped)',
                  reproduction_directory=str(directory.relative_to(ROOT)), status='reproduced; not packed')
    if output is not None:
        result.update(file=str(output.relative_to(ROOT)), sha256=sha256_file(output), head=sha,
                      code_commit_url=args.code_commit, status='packed; not uploaded')
        write_json(output.with_suffix('.manifest.json'), result)
    write_json(directory / 'reproduction.json', result)
    print(f'Reproduced 20 exports; preflight Ready to submit. Evidence: {directory / "reproduction.json"}')


if __name__ == '__main__':
    main()
