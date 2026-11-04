"""Trace frozen rollout assets and express procedural D1 shapes in their body frames.

No scan is loaded. Different reconstructions need not be congruent: the recorded
coordinate map is distinct from a rigid fit between their surfaces.
"""
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
import trimesh
from src.paths import project_path
from src.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'data/outputs/pose_insurance/combined-v5'
D2 = ROOT / 'submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json'
D1 = ROOT / 'submission/20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json'
KIT = ROOT / 'data/.runtime/kit-20261102/v2d_submission_kit'
V4_MESHES = ROOT / 'data/outputs/mesh_extents/build/20261102T051416.149421Z-f377e0a1/meshes'


def read(path):
    return json.loads(project_path(ROOT, path).read_text())


def entry(path):
    path = project_path(ROOT, path)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256_file(path))


def task_for(export):
    """Follow the packed export's recorded rollout, never a latest-file search."""
    run = read(export['run'])
    task_entry, = [e for e in run['inputs'] if e['path'].endswith('/task.parquet')]
    task_path = project_path(ROOT, task_entry['path'])
    if sha256_file(task_path) != task_entry['sha256']:
        raise ValueError(f'Task hash mismatch: {task_path}')
    return task_path, pq.read_table(task_path).to_pylist()[0]


def checked_asset(path):
    """Re-hash against the asset producer, not just the path currently in its task."""
    producer = read(path.parent / 'run.json')
    expected = producer['artifacts'][path.name]
    if isinstance(expected, dict):
        expected = expected['sha256']
    if sha256_file(path) != expected:
        raise ValueError(f'Frozen asset differs from producer: {path}')


def trace_and_build():
    """Use construction provenance for axes; shapes with no recorded frame keep their procedural v4 shape."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parent = read(D2)
    production = read(read(D1)['production']['path'])
    roster = read(KIT / 'data/track_3_evaluation_objects.json')['episodes']
    pairs = {(int(p['episode']), p['object']): p for p in production['pairs']}
    rows = []
    for export in parent['exports']:
        episode = int(Path(export['path']).stem.split('_')[1])
        task_path, task = task_for(export)
        build = read(task_path.parent / 'run.json')
        for slot, name in enumerate(roster[str(episode)]):
            sim = project_path(ROOT, task['object_mesh_paths'][slot])
            urdf = project_path(ROOT, task['object_urdf_paths'][slot])
            checked_asset(sim)
            checked_asset(urdf)
            visual = ET.parse(urdf).getroot().find('link/visual')
            origin = visual.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            if np.any(xyz) or np.any(rpy):
                raise ValueError('Unexpected nonidentity visual origin')
            if project_path(ROOT, visual.find('geometry/mesh').get('filename')) != sim:
                raise ValueError('URDF visual does not name task mesh')
            d1row = pairs[(episode, name)]
            source = project_path(ROOT, d1row['file'])
            if sha256_file(source) != d1row['sha256']:
                raise ValueError('D1 production mesh hash mismatch')
            rotation = np.eye(3)
            generated = d1row['changed']
            reason = ('Both procedural constructors use bounds-centred x=long, y=short, z=height; '
                      'scene.py applies positive per-axis scale only. Coordinate map I, translation 0; '
                      'different shapes/sizes are not rigidly congruent and no surface fit is asserted.')
            if generated:
                # Hunyuan raw.ply has no recorded scale, origin or axes in the rollout's frame, so this pair
                # keeps the same procedural v4 shape the other pairs use, built by the same constructor.
                source = V4_MESHES / f'episode_{episode:06d}' / f'{name}.ply'
                reason = 'D1 v5 Hunyuan shape has no recorded rollout-frame mapping; v4 procedural shape kept. ' + reason
            if episode == 19:
                evidence, = [e for e in build['equivalence'] if e['slot'] == slot]
                if not evidence['passed']:
                    raise ValueError('TW asset equivalence failed')
                rotation = np.asarray(evidence['geometry_rotation'])
                reason += ' Apply recorded TW geometry_rotation from task producer equivalence; v_new = R v_old.'
            folder = OUTPUT / 'meshes' / f'episode_{episode:06d}'
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (name + '.ply')
            if episode == 19:
                mesh = trimesh.load(source, force='mesh', process=False)
                mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
                # Double precision avoids changing the rigidly rotated geometry at float32 precision.
                target = folder / (name + '.obj')
                target.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17, include_texture=False))
            else:
                shutil.copyfile(source, target)
            rows.append(dict(episode=episode, slot=slot, object=name, task=entry(task_path),
                rollout=entry(export['run']), sim_mesh=entry(sim), urdf=entry(urdf),
                visual_origin_xyz=xyz.tolist(), visual_origin_rpy=rpy.tolist(),
                collision_origin=ET.parse(urdf).getroot().find('link/collision/origin').attrib
                    if ET.parse(urdf).getroot().find('link/collision/origin') is not None else None,
                d1_mesh=entry(project_path(ROOT, d1row['file'])), source_mesh=entry(source), d1_source=d1row['source'],
                byte_identical=sha256_file(sim) == sha256_file(source),
                coordinate_rotation=rotation.tolist(), coordinate_translation=[0., 0., 0.],
                decision='v4_procedural' if generated else 'rotated' if episode == 19 else 'identity_frame',
                evidence=reason, combined_mesh=entry(target)))
    if len(rows) != 36:
        raise ValueError('Expected exactly 36 episode/object pairs')
    result = dict(kind='pose_insurance.combined_frames', parents=[entry(D1), entry(D2)],
        note='Coordinate maps follow construction, not scan fitting. D1 pooled sizes are retained in metres; '
             'they can differ from the per-episode simulated bounds. No scan geometry is loaded.',
        evidence_sources=[entry(ROOT / p) for p in (
            'src/features/pose_insurance/scene.py', 'src/features/pose_insurance/scored_heading_assets.py',
            'src/features/mesh_extents/build.py', 'src/features/mesh_extents/_shapes.py')],
        counts={key: sum(row['decision'] == key for row in rows)
                for key in ('identity_frame', 'rotated', 'v4_procedural')}, rows=rows)
    (OUTPUT / 'body-frames.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Frozen v5 body-frame trace', '', result['note'], '',
        'Identity below means a coordinate map, not identical surfaces. Full SHA-256 and origins are in body-frames.json.', '',
        '| Episode | Object | Sim mesh (SHA-256) | Visual offset | D1 mesh (SHA-256) | Decision |',
        '|---:|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['episode']} | {row['object']} | {row['sim_mesh']['path']} ({row['sim_mesh']['sha256']}) | none | {row['d1_mesh']['path']} ({row['d1_mesh']['sha256']}) | {row['decision']} |")
    lines += ['', 'The two rotations and every task/rollout/URDF hash are recorded in JSON. '
              'All visual origins are identity. TW collision boxes are counter-rotated as recorded; '
              'that collision origin is not an extra visual-mesh transform.', '']
    (OUTPUT / 'body-frames.md').write_text('\n'.join(lines))
    print(json.dumps(result['counts']))
    return result


if __name__ == '__main__':
    trace_and_build()
