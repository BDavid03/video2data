"""Resumable, CPU-only public-label measurements with the replaced kit's scorer.

The host's geometry blob is not published. These are explicitly public-geometry
diagnostics: surface centroid anchors, sampled public scan points and estimated
physical symmetry axes, not predictions of confidential Kaggle scores.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import trimesh
from src.paths import project_path
from .combined_frames import ROOT, KIT, OUTPUT, D1, read, entry
from ._combined_metrics import registered_metrics, rotation_spread

ALLOWED = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)
TRAIN = ALLOWED[:8]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@lru_cache(maxsize=16)
def geometry(name):
    from v2dlb.vertices import mesh_path_for, sample_vertices
    dataset = next((ROOT / 'data').glob('datasets--*/snapshots/*/track_3/public'))
    path = mesh_path_for(dataset / 'mesh', name)
    mesh = trimesh.load(path, force='mesh', process=False)
    arrays = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
    anchor, _ = SCORER._surface_moments(*arrays)
    axis, order = np.zeros(3), 1
    if name in ('blue_cup', 'beige_cup', 'white_pot_lid'):
        # This estimate is a disclosed local proxy; the host's hand-declared axes are unavailable.
        surface = SCORER._sample_mesh_surface(*arrays, 10000, 0)
        _, eigenvectors = np.linalg.eigh(np.cov((surface - anchor).T))
        axis = eigenvectors[:, 0 if name == 'white_pot_lid' else -1]
        order = 2 if name == 'white_pot_lid' else 360
    return dict(mesh=arrays, anchor=anchor, points=sample_vertices(path), axis=axis, order=order,
                input=entry(path))


def budget(path):
    from v2dlb.mesh_budget import budget_mesh
    return budget_mesh(path, 4096, 4096)


def inputs(episode, mode, historical, baseline, candidates):
    if episode not in ALLOWED:
        raise ValueError('Excluded episode')
    names, ref = baseline.load_reference(episode)
    if mode.startswith('m2v'):
        names = ['mini_dust_pan']
        export = ROOT / 'data/outputs/moving_policy/ep39-v1/score-66763776/trajectory.parquet'
        df = pd.read_parquet(export).sort_values('reference_frame')
        columns = [f'achieved_position_{a}' for a in 'xyz'] + [f'achieved_quaternion_{a}' for a in 'xyzw']
        achieved = df[columns].to_numpy(dtype=float, copy=True)[:, None, None]
        mesh = ROOT / ('data/outputs/moving_policy/ep39-v1/mini_dust_pan.obj' if mode == 'm2v_simulated'
                       else 'data/outputs/moving_reference/ep39/candidate-c/canonical-asset.obj')
        explanation = 'Actual unchanged M2v evaluator export, with simulated mesh.'
        if mode == 'm2v_m1_tracking':
            # M2v poses are in the recorded scan-like simulator body, M1 vertices in asset frame.
            # world_T_asset = world_T_sim_body @ sim_body_T_asset, preserving physical placement.
            from scipy.spatial.transform import Rotation
            body_file = ROOT / 'data/outputs/moving_reference/ep39/candidate-c/body-convention.json'
            transform = np.asarray(read(body_file)['scan_T_asset'])
            rotation = Rotation.from_quat(achieved[:, 0, 0, 3:]).as_matrix()
            achieved[:, 0, 0, :3] += np.einsum('tij,j->ti', rotation, transform[:3, 3])
            achieved[:, 0, 0, 3:] = Rotation.from_matrix(rotation @ transform[:3, :3]).as_quat()
            explanation = 'Same M2v export re-expressed in M1 tracking-asset frame using recorded scan_T_asset; measurement only.'
        old = read(ROOT / 'data/outputs/moving_policy/ep39-v1/score-66763776/mocap_metrics_aligned.json')['kit']
        return names, achieved, ref, [mesh], old, explanation
    plan = historical['construction'][str(episode)]
    names = plan['names']
    ref_names, ref = baseline.load_reference(episode)
    ref = ref[:, :, [ref_names.index(name) for name in names]]
    scene = plan['scenes']['T1']
    positions, wxyz = np.asarray(scene['positions']), np.asarray(scene['quaternions'])
    start = np.c_[positions, wxyz[:, [1, 2, 3, 0]]]
    achieved = np.repeat(start[None, None], len(ref), axis=0)
    folder = OUTPUT / 'labelled-meshes' / mode / f'episode_{episode:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    if mode == 'primitive' or any(name not in candidates for name in names):
        # Reproduce scene.py's original primitive construction, not scan or D1 geometry.
        objects_dir = ROOT / 'data/outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff'
        objects = read(objects_dir / 'objects.json')['objects']
        for slot, name in enumerate(names):
            if mode != 'primitive' and name in candidates:
                paths.append(candidates[name])
                continue
            obj = objects[name]
            mesh = trimesh.load(objects_dir / obj['file'], force='mesh', process=False)
            mesh.apply_scale(np.asarray(scene['dimensions'][slot]) / obj['extents_m'])
            path = folder / (name + '.obj')
            path.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17, include_texture=False))
            paths.append(path)
    else:
        paths = [candidates[name] for name in names]
    old = historical['episodes']['T1'][str(episode)]['metrics']
    explanation = ('Frozen T1 static kinematic projection (the source of old train AUC 0.2767); '
                   'there is no eight-episode simulated train export to rescore. D1 procedural shape '
                   'in common centred long/short/height body coordinates; water_pitcher has no '
                   'non-Hunyuan D1 v5 source, so uses the original simulated primitive construction '
                   'at the frozen T1 dimensions, as the scored fallback does. No recipe selection.')
    return names, achieved, ref, paths, old, explanation


def publish(rows, disclosure):
    selected = [r for r in rows if r['seed_offset'] == 0 and r['mode'] == 'combined']
    mean = {key: float(np.mean([r['metrics'][key] for r in selected if r['episode'] in TRAIN]))
            for key in selected[0]['metrics']} if selected else {}
    spread = []
    metric_spread = []
    for mode, ep in sorted({(r['mode'], r['episode']) for r in rows}):
        group = [r for r in rows if r['mode'] == mode and r['episode'] == ep]
        metric_spread.append(dict(mode=mode, episode=ep, seeds=len(group),
            mean={key:float(np.mean([r['metrics'][key] for r in group])) for key in group[0]['metrics']},
            std={key:float(np.std([r['metrics'][key] for r in group])) for key in group[0]['metrics']}))
        for slot, name in enumerate(group[0]['names']):
            rotations = [np.asarray(r['registrations'][slot]['rotation']) for r in group]
            angles = rotation_spread(rotations)
            spread.append(dict(mode=mode, episode=ep, object=name, seeds=len(group),
                pairwise_rotation_degrees=angles, maximum_degrees=max(angles, default=0),
                flip_over_150_degrees=any(a > 150 for a in angles)))
    result = dict(disclosure=disclosure, train_mean=mean, rows=rows, registration_stability=spread,
                  metric_seed_spread=metric_spread)
    (OUTPUT / 'scoring-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# New-scorer public-geometry measurements', '', disclosure, '',
        '| Episode | Mode | Seed offset | New AUC | SP-SR | MP-SR | RPE cm | MPPE cm | Old AUC |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda r: (r['episode'], r['mode'], r['seed_offset'])):
        m = row['metrics']
        lines.append(f"| {row['episode']} | {row['mode']} | {row['seed_offset']} | {m['AUC']:.6f} | {m['SP_SR']:.6f} | {m['MP_SR']:.6f} | {m['RPE_cm']:.6f} | {m['MPPE_cm']:.6f} | {row['old_metrics']['AUC']:.6f} |")
    lines += ['', f'Train mean, scorer seeds: {mean}', '',
              '| Episode | Mode | Object | Seeds | Maximum rotation difference (deg) | >150 degree flip |',
              '|---:|---|---|---:|---:|---|']
    lines += [f"| {s['episode']} | {s['mode']} | {s['object']} | {s['seeds']} | {s['maximum_degrees']:.4f} | {s['flip_over_150_degrees']} |" for s in spread]
    lines += ['', 'Seeds measure registration sensitivity, not independent model replications. '
        'Symmetric boxes have exact half-turn ambiguities; near-spherical shapes may have effectively '
        'arbitrary body rotations. The scorer recognizes physical cup/lid symmetry only, not the extra '
        'symmetry of a poor reconstruction. For a single object, initial scene alignment cancels a '
        'constant orientation offset at frame zero, but not subsequent rotations about wrong body anchors.', '']
    (OUTPUT / 'scoring-summary.md').write_text('\n'.join(lines))


def run(queue_path):
    global SCORER
    sys.path.insert(0, str(KIT))
    SCORER = module('combined_official_auc', KIT / 'metric_code/track_3/AUC.py')
    baseline = module('combined_static_reference', ROOT / 'scripts/track3/static_pose_baseline.py')
    queue = read(queue_path)
    if tuple(queue['episodes']) != ALLOWED or queue['seed_offsets'] != [0, 1, 2]:
        raise ValueError('Keep fixed measurement episodes and seeds')
    historical_path = ROOT / 'data/outputs/pose_insurance/stereo-placement/summary.json'
    historical = read(historical_path)
    production = read(read(D1)['production']['path'])
    candidates = {p['object']: project_path(ROOT, p['file']) for p in production['pairs'] if not p['changed']}
    disclosure = ('Measurement only: labels from train 0/11/12/18/21/23/26/31 and tuning 39/42 are read; '
        'no selection, threshold or gate changes. No labels for 38, excluded/validation episodes never loaded. '
        'Exact confidential scores cannot be computed: host reference_geometry_b64 (anchors, physical axes, '
        'surface points and cleaned scan geometry) is unpublished. This diagnostic uses public visual scans, '
        'surface-centroid anchors, kit 500 vertex samples seed 0, and surface-PCA cup/lid axes. '
        'It runs the new scorer registration/conversion/scene-alignment/metric flow unchanged. '
        'T1 train/control rows are frozen static kinematic projections, not RL rollout exports; '
        'M2v rows use the real evaluator export. Existing M1 scan-derived frame provenance is disclosed.')
    ledger = OUTPUT / 'scoring-ledger.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    started = time.monotonic()
    active = []
    jobs = [('combined', ep) for ep in ALLOWED] + [('primitive', ep) for ep in queue['primitive_episodes']]
    jobs += [('m2v_' + mode, 39) for mode in queue['m2v_meshes']]
    for mode, episode in jobs:
        names, achieved, reference, meshes, old, explanation = inputs(episode, mode, historical, baseline, candidates)
        inventory = [entry(historical_path), entry(queue_path), entry(Path(__file__)),
                     entry(Path(__file__).with_name('_combined_metrics.py')),
                     entry(ROOT / 'scripts/track3/static_pose_baseline.py'),
                     entry(KIT / 'metric_code/track_3/AUC.py'), *[entry(p) for p in meshes],
                     *[geometry(name)['input'] for name in names]]
        dataset = next((ROOT / 'data').glob('datasets--*/snapshots/*/track_3/public'))
        inventory.append(entry(dataset / f'data/chunk-000/episode_{episode:06d}.parquet'))
        if mode.startswith('m2v'):
            inventory.append(entry(ROOT / 'data/outputs/moving_policy/ep39-v1/score-66763776/trajectory.parquet'))
        if mode == 'm2v_m1_tracking':
            inventory.append(entry(ROOT / 'data/outputs/moving_reference/ep39/candidate-c/body-convention.json'))
        for offset in queue['seed_offsets']:
            identity = hashlib.sha256(json.dumps([mode, episode, offset, inventory], sort_keys=True).encode()).hexdigest()
            previous = next((r for r in rows if r['identity'] == identity), None)
            if previous is not None:
                active.append(previous)
                continue
            if time.monotonic() - started > queue['max_seconds']:
                publish(active, disclosure)
                raise TimeoutError('Measurement budget ended; resume the same queue')
            metrics, registrations = registered_metrics(SCORER, achieved, reference,
                [budget(p) for p in meshes], [geometry(name) for name in names], episode, offset)
            row = dict(identity=identity, mode=mode, episode=episode, names=names, seed_offset=offset,
                inputs=inventory, metrics=metrics, registrations=registrations,
                old_metrics=old, explanation=explanation)
            with ledger.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            rows.append(row)
            active.append(row)
            publish(active, disclosure)
            print(json.dumps(dict(mode=mode, episode=episode, seed_offset=offset, metrics=metrics)), flush=True)
    publish(active, disclosure)
    (OUTPUT / 'scoring-finished.json').write_text(json.dumps(dict(status='completed',
        queue=entry(queue_path), rows=len(active), summary=entry(OUTPUT / 'scoring-summary.json')),
        indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default='src/features/pose_insurance/combined_score_queue.json')
    run(parser.parse_args().queue)
