"""Pack frozen v5 poses and traced meshes with the unedited 2 November kit."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from src.paths import project_path
from src.provenance import sha256_file
from .combined_frames import ROOT, OUTPUT, D1, D2, KIT, entry, read

POSE = ['pos_' + axis for axis in 'xyz'] + ['quat_' + axis for axis in 'xyzw']
ARCHIVE_SHA = '7e1fdd00b50189191b4f1bf03158607f4f1a7bb5bfb0bf867b14eb06270a77f0'


def compare_poses(parquet):
    """The old kit prints six digits after the decimal; compare each scalar's rounding interval."""
    baseline = D2.with_name('track3_poses_v5_final.csv')
    with baseline.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if any(not re.fullmatch(r'-?\d+\.\d{6}', row[key]) for row in rows for key in POSE):
        raise ValueError('Frozen CSV precision differs from six decimal places')
    old = pd.DataFrame(rows).set_index('row_id').sort_index()
    new = pd.read_parquet(parquet).set_index('row_id').sort_index()
    if not old.index.equals(new.index) or not old.index.is_unique:
        raise ValueError('Pose row identity changed')
    a, b = old[POSE].to_numpy(float), new[POSE].to_numpy(float)
    difference = np.abs(a - b)
    # Half a unit in the last printed decimal, plus binary float representation error.
    tolerance = 0.5e-6 + 8 * np.finfo(float).eps * np.maximum(1., np.abs(b))
    if not np.isfinite(b).all() or np.any(difference > tolerance):
        raise ValueError(f'Pose changed beyond CSV printing: maximum {difference.max():.12g}')
    for column in ('eval_script_sha256', 'eval_code_sha256', 'checkpoint_sha256'):
        if not np.array_equal(old[column].to_numpy(), new[column].to_numpy()):
            raise ValueError(f'Frozen evaluator/checkpoint provenance changed: {column}')
    return dict(rows=len(new), pose_scalars=int(a.size), csv_decimal_places=6,
                tolerance='0.5e-6 plus 8 float64 eps times max(1,abs(value))',
                maximum_absolute_difference=float(difference.max()), passed=True)


def command(argv, log):
    result = subprocess.run([str(x) for x in argv], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                 OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}; see {log}')
    return dict(argv=[str(x) for x in argv], log=entry(log))


def extract_cd_o(parquet):
    """Call the uploader's own pure extraction path without authenticating or uploading."""
    sys.path.insert(0, str(KIT))
    from v2dlb.track3_pose import cd_o_submission
    roster = read(KIT / 'data/track_3_evaluation_objects.json')['episodes']
    sample = pd.read_parquet(KIT / 'data/track_3_mesh_sample_submission.parquet')
    frame = cd_o_submission(pd.read_parquet(parquet), sample, roster)
    target = parquet.with_name(parquet.stem + '.cd-o.parquet')
    frame.to_parquet(target, index=False)
    parent = pd.read_parquet(D1.with_name('track3_meshes_v5_final.parquet'))
    value_columns = ['x', 'y', 'z']
    left, right = frame.set_index('row_id').sort_index(), parent.set_index('row_id').sort_index()
    if not left.index.equals(right.index):
        raise ValueError('CD-O row IDs differ from D1')
    different = np.any(left[value_columns].to_numpy() != right[value_columns].to_numpy(), axis=1)
    identifiers = left.index[different].tolist()
    return dict(file=entry(target), changed_rows=int(different.sum()), total_rows=len(frame),
        changed_object_prefixes=sorted({s.rsplit('_', 2)[0] for s in identifiers}),
        note='Pot 5 and pitcher 20 keep v4 procedural shapes in place of Hunyuan. Episode 19 is rigidly rotated: '
             'ideal registered CD-O is invariant, but finite multi-start ICP and sampling can change it.')


def verify_manifest(path, release=False):
    manifest = read(path)
    if manifest.get('kind') != 'pose_insurance.combined_pack':
        raise ValueError('Not a combined-pack manifest')
    if release and manifest.get('dry'):
        raise ValueError('DRY pack is never eligible for release')
    def check(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                if value['path'] == 'AGENTS.md':
                    raise ValueError('Combined manifest must not pin AGENTS.md')
                if sha256_file(project_path(ROOT, value['path'])) != value['sha256']:
                    raise ValueError(f"Hash mismatch: {value['path']}")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(manifest)
    comparison = compare_poses(project_path(ROOT, manifest['file']['path']))
    if comparison != manifest['pose_comparison']:
        raise ValueError('Recorded comparison differs')
    if manifest['kit_archive']['sha256'] != ARCHIVE_SHA:
        raise ValueError('Unexpected kit archive')
    print(f"VERIFIED {'DRY; NOT RELEASE ELIGIBLE' if manifest['dry'] else 'combined pack'}: {manifest['file']['path']}")
    return manifest


def pack(url, output, dry=False):
    if not re.fullmatch(r'https://github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-f]{7,40}', url):
        raise ValueError('Expected a GitHub /commit/<sha> URL')
    output = project_path(ROOT, output)
    allowed = ROOT / ('data/.runtime/tmp' if dry else 'data/outputs/pose_insurance/combined-v5')
    output.relative_to(allowed)
    if not dry and re.fullmatch(r'0+', url.rsplit('/', 1)[-1]):
        raise ValueError('Placeholder URL requires --dry')
    output.mkdir(parents=True, exist_ok=True)
    archive = KIT.parent / 'kit.zip'
    if sha256_file(archive) != ARCHIVE_SHA:
        raise ValueError('Kit archive mismatch')
    frames_file = OUTPUT / 'body-frames.json'
    frames = read(frames_file)
    frozen = read(D2)
    results = output / 'results'
    results.mkdir(exist_ok=True)
    for export in frozen['exports']:
        path = project_path(ROOT, export['path'])
        if sha256_file(path) != export['sha256']:
            raise ValueError(f'Frozen export mismatch: {path}')
        shutil.copyfile(path, results / path.name)
    for row in frames['rows']:
        for key in ('combined_mesh', 'task', 'rollout', 'sim_mesh', 'urdf', 'd1_mesh', 'source_mesh'):
            item = row[key]
            if sha256_file(project_path(ROOT, item['path'])) != item['sha256']:
                raise ValueError(f"Body-frame trace changed: {item['path']}")
    target = output / ('track3_combined_v5_DRY.parquet' if dry else 'track3_combined_v5.parquet')
    python = ROOT / 'data/.runtime/kit-env/bin/python'
    commands = [command([python, '-B', KIT / 'tools/pack_submission.py', '--track', 'track_3',
        '--results', results, '--meshes', OUTPUT / 'meshes', '--code-commit', url,
        '--eval-script', KIT / 'flash_chord/scripts/evaluate_policy.py', '--skip-crosscheck', '-o', target], output / 'pack.log')]
    commands.append(command([python, '-B', KIT / 'tools/preflight.py', '--track', 'track_3',
        '--results', results, '--skip-kaggle'], output / 'preflight.log'))
    comparison = compare_poses(target)
    cd_o = extract_cd_o(target)
    manifest = dict(kind='pose_insurance.combined_pack', dry=dry,
        status='DRY; NEVER RELEASE' if dry else 'locally verified; lead release check and Kaggle preflight required',
        file=entry(target), code_commit_url=url, kit_archive=entry(archive),
        parents=[entry(D1), entry(D2)], body_frames=entry(frames_file),
        meshes=[row['combined_mesh'] for row in frames['rows']],
        exports=[entry(results / Path(e['path']).name) for e in frozen['exports']],
        sources=[entry(Path(__file__)), entry(Path(__file__).with_name('combined_frames.py'))],
        kit_files=[entry(p) for p in sorted(KIT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        commands=commands, pose_comparison=comparison, cd_o=cd_o,
        mesh_warnings=[line for line in (output / 'pack.log').read_text().splitlines() if 'MESH WARNING' in line],
        scoring_caveat='Local labelled measurements are in combined-v5/scoring-summary.json. '
            'Host reference anchors, physical symmetry axes and points are not published. '
            'Primitives can give ambiguous registration rotations. No accuracy gate is claimed.')
    path = target.with_suffix('.manifest.json')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    verify_manifest(path)
    print(json.dumps(dict(file=manifest['file'], pose_comparison=comparison, cd_o=cd_o), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--code-commit')
    parser.add_argument('--output', default=str(OUTPUT / 'release'))
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--verify-manifest')
    parser.add_argument('--release', action='store_true', help='reject a dry manifest during verification')
    args = parser.parse_args()
    if args.verify_manifest:
        verify_manifest(args.verify_manifest, args.release)
    elif args.code_commit:
        pack(args.code_commit, args.output, args.dry)
    else:
        parser.error('--code-commit or --verify-manifest is required')


if __name__ == '__main__':
    main()
