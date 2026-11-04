"""Extend the immutable combined-v5 packer with a re-admitted episode-38 rollout."""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.paths import project_path
from src.provenance import sha256_file
from . import pack_combined as base
from . import pack_moving as moving
from .combined_frames import ROOT, OUTPUT, D1, D2, KIT, entry, read

CHECKPOINT = 'b6145fcb70cbfed22aebf40e5c35cb6793c8a726a25f308791e35acc75a43cae'
KIND = 'pose_insurance.combined_moving_pack'


def admissions(request_file):
    """Re-admit the single finished video policy through every frozen moving gate."""
    request = read(request_file)
    if len(request['adoptions']) != 1 or request['adoptions'][0]['episode'] != 38:
        raise ValueError('Only episode 38 may replace a combined-v5 export')
    adoption, = request['adoptions']
    roster, objects = moving.official_roster()
    _, code = moving.evaluator_inputs()
    result = moving.admit(adoption, roster[38], objects[38].object_names, code)
    directory = ROOT / 'data/outputs/moving_policy/ep38-v1'
    if (moving.checked(result['trajectory']) != directory / 'score-66763776/trajectory.parquet'
            or result['checkpoint']['sha256'] != CHECKPOINT):
        raise ValueError('Moving variant requires the final official ep38-v1 export/checkpoint')
    return result


def mesh_trace(admission):
    """Verify the actual task mesh and identity URDF body frame against the shared asset."""
    policy = admission['policy']
    row = pq.read_table(moving.checked(policy['task'])).to_pylist()[0]
    source = ROOT / 'data/outputs/moving_policy/ep38-v1/mini_dust_pan.obj'
    urdf = source.with_suffix('.urdf')
    assets = policy['asset_hashes']
    for p in (source, urdf):
        if sha256_file(p) != assets[p.name]:
            raise ValueError('Moving mesh/URDF differs from shared simulator asset')
    task_mesh, = row['object_mesh_paths']
    task_urdf, = row['object_urdf_paths']
    if (sha256_file(moving.path(task_mesh)) != assets[source.name]
            or sha256_file(moving.path(task_urdf)) != assets[urdf.name]):
        raise ValueError('Moving asset differs from what the task simulated')
    link = ET.parse(urdf).getroot().find('link')
    if link.get('name') != 'mini_dust_pan':
        raise ValueError('Unexpected moving rigid-body name')
    for kind in ('visual', 'collision'):
        element = link.find(kind)
        origin = element.find('origin')
        if origin is not None and any(np.any(np.fromstring(origin.get(k, '0 0 0'), sep=' '))
                                      for k in ('xyz', 'rpy')):
            raise ValueError('Moving URDF must have identity visual and collision origins')
        mesh = element.find('geometry/mesh')
        if (mesh.get('filename') != source.name
                or not np.array_equal(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '), np.ones(3))):
            raise ValueError('Moving URDF must reference the unscaled simulated mesh')
    return dict(episode=38, object='mini_dust_pan', source_mesh=entry(source), urdf=entry(urdf),
                task=policy['task'], shared_asset_inputs=policy['shared_asset_inputs'],
                visual_origin_xyz=[0, 0, 0], visual_origin_rpy=[0, 0, 0], scale=[1, 1, 1],
                coordinate_rotation=np.eye(3).tolist(), coordinate_translation=[0, 0, 0])


def meshes(admission, destination):
    """Copy the 36-shape tree with only the simulated episode-38 pan substituted."""
    trace = mesh_trace(admission)
    frames = read(OUTPUT / 'body-frames.json')
    expected = {}
    for row in frames['rows']:
        old = moving.checked(row['combined_mesh'])
        changed = (row['episode'], row['object']) == (38, 'mini_dust_pan')
        source = moving.checked(trace['source_mesh']) if changed else old
        target = destination / old.relative_to(OUTPUT / 'meshes').with_suffix(source.suffix)
        expected[target] = source
    if destination.exists():
        if set(p for p in destination.rglob('*') if p.is_file()) != set(expected):
            raise ValueError('Existing moving mesh tree has a different file roster')
        for target, source in expected.items():
            if sha256_file(target) != sha256_file(source):
                raise ValueError('Existing moving mesh tree differs; use a fresh output')
    else:
        for target, source in expected.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    trace['combined_mesh'] = entry(destination / 'episode_000038/mini_dust_pan.obj')
    return [entry(p) for p in expected], trace


def compare_poses(parquet, admission):
    """Keep all nineteen static episodes and compare 38 exactly to its official export."""
    with D2.with_name('track3_poses_v5_final.csv').open(newline='') as stream:
        old = pd.DataFrame(list(csv.DictReader(stream))).set_index('row_id').sort_index()
    new = pd.read_parquet(parquet).set_index('row_id').sort_index()
    if not old.index.equals(new.index) or not new.index.is_unique:
        raise ValueError('Combined pose row identity changed')
    keys = new.index.to_series().str.extract(r'^track_3/e(\d+)/w(\d+)/f(\d+)/o(\d+)$').astype(int)
    keys.columns = ['episode_index', 'world_index', 'frame_index', 'object_slot']
    new = new.join(keys)
    static = new.episode_index != 38
    a, b = old.loc[static, base.POSE].to_numpy(float), new.loc[static, base.POSE].to_numpy(float)
    tolerance = .5e-6 + 8 * np.finfo(float).eps * np.maximum(1., np.abs(b))
    if not np.isfinite(b).all() or np.any(np.abs(a - b) > tolerance):
        raise ValueError('Unadopted pose changed beyond frozen CSV printing precision')
    for key in ('eval_script_sha256', 'eval_code_sha256', 'checkpoint_sha256'):
        if not np.array_equal(old.loc[static, key], new.loc[static, key]):
            raise ValueError(f'Unadopted provenance changed: {key}')
    sys.path.insert(0, str(KIT))
    from v2dlb.submission_parquet import read_trajectory, provenance
    trajectory = read_trajectory(moving.checked(admission['trajectory']))
    selected = new.loc[~static].sort_values(['world_index', 'frame_index', 'object_slot'])
    order = np.lexsort((trajectory.object_slot, trajectory.frame_index, trajectory.world_index))
    keys = selected[['world_index', 'frame_index', 'object_slot']].to_numpy()
    expected = np.column_stack((trajectory.world_index, trajectory.frame_index, trajectory.object_slot))[order]
    if not np.array_equal(keys, expected) or not np.array_equal(selected[base.POSE].to_numpy(), trajectory.pose_xyzw[order]):
        raise ValueError('Episode 38 packed keys/poses differ from the full official export')
    for key, value in provenance(trajectory).items():
        if key in selected and not selected[key].eq(value).all():
            raise ValueError(f'Episode 38 provenance differs from official export: {key}')
    if not selected.checkpoint_sha256.eq(CHECKPOINT).all():
        raise ValueError('Episode 38 checkpoint is not the final learned policy')
    return dict(rows=len(new), unchanged_rows=int(static.sum()), moving_rows=len(selected),
                unchanged_episodes=19, maximum_static_absolute_difference=float(np.abs(a - b).max()),
                moving_pose_equality='exact float64 values and complete official key grid',
                moving_checkpoint_sha256=CHECKPOINT, passed=True)


def sources():
    return [*moving.source_entries(), entry(Path(__file__)), entry(Path(base.__file__)),
            entry(Path(__file__).with_name('combined_frames.py'))]


def verify_kit():
    """Require the current kit's files to equal its frozen downloaded archive."""
    archive = KIT.parent / 'kit.zip'
    if sha256_file(archive) != base.ARCHIVE_SHA:
        raise ValueError('Kit archive mismatch')
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not item.is_dir() and (KIT.parent / item.filename).read_bytes() != bundle.read(item):
                raise ValueError(f'Current kit differs from its archive: {item.filename}')
    return archive


def pack(url, output, dry=False, adoptions=None):
    """Pack a re-admitted moving variant or delegate unchanged static packing."""
    if adoptions is None:
        return base.pack(url, output, dry)
    if not re.fullmatch(r'https://github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-f]{40}', url):
        raise ValueError('Moving pack requires a full-SHA GitHub commit URL')
    output = project_path(ROOT, output)
    output.relative_to(ROOT / ('data/.runtime/tmp' if dry else 'data/outputs/pose_insurance/combined-v5'))
    if output.exists():
        raise FileExistsError('Use a fresh moving release directory')
    admission = admissions(adoptions)
    inventory = sources()
    if not dry:
        if admission['abstract_disclosure']['path'] != 'docs/report/track3/main.tex':
            raise ValueError('Release admission must bind the promoted report')
        moving.committed(inventory, url)
    archive = verify_kit()
    if sha256_file(KIT / 'flash_chord/scripts/evaluate_policy.py') != sha256_file(moving.KIT / 'flash_chord/scripts/evaluate_policy.py'):
        raise ValueError('Current and producer kit evaluator scripts differ')
    frozen = moving.frozen_v5()
    output.mkdir(parents=True, exist_ok=False)
    results = output / 'results'
    results.mkdir()
    exports = []
    for original in frozen['exports']:
        episode = int(Path(original['path']).stem.split('_')[1])
        source = moving.checked(admission['trajectory'] if episode == 38 else original)
        target = results / f'episode_{episode:06d}.parquet'
        shutil.copyfile(source, target)
        exports.append(dict(episode=episode, **entry(target)))
    mesh_root = output / 'meshes-moving' if dry else OUTPUT / 'meshes-moving'
    mesh_entries, trace = meshes(admission, mesh_root)
    target = output / ('track3_combined_v5_DRY.parquet' if dry else 'track3_combined_v5.parquet')
    python = ROOT / 'data/.runtime/kit-env/bin/python'
    commands = [base.command([python, '-B', KIT / 'tools/pack_submission.py', '--track', 'track_3',
        '--results', output / 'results', '--meshes', mesh_root, '--code-commit', url,
        '--eval-script', KIT / 'flash_chord/scripts/evaluate_policy.py', '--skip-crosscheck', '-o', target], output / 'pack.log')]
    commands.append(base.command([python, '-B', KIT / 'tools/preflight.py', '--track', 'track_3',
        '--results', output / 'results', '--skip-kaggle'], output / 'preflight.log'))
    if 'Ready to submit.' not in (output / 'preflight.log').read_text():
        raise ValueError('Current kit preflight did not report Ready to submit')
    comparison = compare_poses(target, admission)
    # Admission imports the producer's old kit; isolate the current uploader's imports.
    extraction = ('import json,sys; from pathlib import Path; '
                  'from src.features.pose_insurance.pack_combined import extract_cd_o; '
                  'print(json.dumps(extract_cd_o(Path(sys.argv[1]))))')
    commands.append(base.command([python, '-B', '-c', extraction, target], output / 'cd-o.log'))
    cd_o = json.loads((output / 'cd-o.log').read_text())
    cd_o['moving_note'] = 'Episode 38 uses its actual simulated profile; its CD-O effect is unmeasured.'
    manifest = dict(kind=KIND, dry=dry, status='DRY; NEVER RELEASE' if dry else 'lead release review required',
        file=entry(target), code_commit_url=url, adoptions=entry(adoptions), admission=admission,
        kit_archive=entry(archive), parents=[entry(D1), entry(D2)], body_frames=entry(OUTPUT / 'body-frames.json'),
        meshes=mesh_entries, moving_mesh=trace, exports=exports, sources=inventory,
        kit_files=[entry(p) for p in sorted(KIT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        commands=commands, pose_comparison=comparison, cd_o=cd_o,
        mesh_warnings=[s for s in (output / 'pack.log').read_text().splitlines() if 'MESH WARNING' in s])
    manifest_file = target.with_suffix('.manifest.json')
    moving.write(manifest_file, manifest)
    print(json.dumps(dict(file=manifest['file'], pose_comparison=comparison, mesh_warnings=manifest['mesh_warnings']), indent=2))
    return manifest_file


def verify_manifest(name, release=False):
    """Re-hash and re-admit all moving evidence before comparing the packed output."""
    manifest = read(name)
    if manifest.get('kind') != KIND:
        return base.verify_manifest(name, release)
    if release and manifest['dry']:
        raise ValueError('DRY pack is never eligible for release')
    def visit(value):
        if isinstance(value, dict):
            if {'path', 'sha256'} <= value.keys():
                if Path(value['path']).name == 'AGENTS.md':
                    raise ValueError('Moving combined manifest must not pin AGENTS.md')
                moving.checked(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    verify_kit()
    if manifest['sources'] != sources() or manifest['kit_archive']['sha256'] != base.ARCHIVE_SHA:
        raise ValueError('Moving combined source or kit inventory changed')
    admission = admissions(moving.checked(manifest['adoptions']))
    if admission != manifest['admission'] or compare_poses(moving.checked(manifest['file']), admission) != manifest['pose_comparison']:
        raise ValueError('Moving combined re-admission or pose comparison changed')
    mesh_root = moving.checked(manifest['moving_mesh']['combined_mesh']).parent.parent
    mesh_entries, trace = meshes(admission, mesh_root)
    if mesh_entries != manifest['meshes'] or trace != manifest['moving_mesh']:
        raise ValueError('Moving mesh tree differs from the simulated asset and 35 frozen shapes')
    frozen = moving.frozen_v5()
    if len(manifest['exports']) != 20 or {e['episode'] for e in manifest['exports']} != {int(Path(e['path']).stem.split('_')[1]) for e in frozen['exports']}:
        raise ValueError('Moving combined export roster changed')
    for export in manifest['exports']:
        expected = admission['trajectory'] if export['episode'] == 38 else next(
            e for e in frozen['exports'] if int(Path(e['path']).stem.split('_')[1]) == export['episode'])
        if export['sha256'] != expected['sha256']:
            raise ValueError('Composed export differs from admitted or frozen source')
    if not manifest['dry']:
        if admission['abstract_disclosure']['path'] != 'docs/report/track3/main.tex':
            raise ValueError('Release requires the promoted report')
        moving.committed(manifest['sources'], manifest['code_commit_url'])
    print(f"VERIFIED {'DRY; NOT RELEASE ELIGIBLE' if manifest['dry'] else 'moving combined pack'}: {manifest['file']['path']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--adoptions')
    parser.add_argument('--code-commit')
    parser.add_argument('--output')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--verify-manifest')
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    if args.verify_manifest:
        verify_manifest(args.verify_manifest, args.release)
    elif args.code_commit:
        pack(args.code_commit, args.output or str(OUTPUT / ('release-moving' if args.adoptions else 'release')),
             args.dry, args.adoptions)
    else:
        parser.error('--code-commit or --verify-manifest is required')


if __name__ == '__main__':
    main()
