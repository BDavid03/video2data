"""Build video-only T1 scenes, resume recorded CPU trials and publish the train verdict."""
import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
import sys
import tempfile
import time
import numpy as np
from src.paths import output_path
from src.provenance import input_file, read_run, run_fields, write_json
from src.runs import (ROOT, artifact, authorize_episode, checked_entry, heavy_execution,
                         new_run_id, producer, read_json, recorded, runtime_flag, source_hashes,
                         CACHE_FOLDERS)
from .placement_diagnostic import (FEATURE, ROSTER, D1, PlacementRequest, video_geometry,
                                   reference, baseline_module, label_paths)
from .scale_diagnostic import train_parents, tuning_geometry
from .length_diagnostic import clusters_for, load_stereo
from .scale_loop import append
from .shrink_loop import resume_ledger
from ._scale import scale_geometry
from ._shrink import EPISODES, CONTROLS
from ._stereo_placement import stereo_offset, placement, start_poses, unchanged_fields, direction_reading

OUTPUT = 'data/outputs/pose_insurance/stereo-placement'
PROTOCOL = 'docs/chain/pose-stereo-placement-protocol.md'


@dataclass(frozen=True)
class StereoPlacementRequest:
    episode_index: int
    candidate: str
    source_split: str = 'public'

    def __post_init__(self):
        if self.source_split != 'public' or self.episode_index not in (*EPISODES, *CONTROLS):
            raise PermissionError('T1 requires the fixed eight train episodes and two controls')
        if self.candidate not in ('construction', 'S1', 'T1'):
            raise ValueError('Unknown stereo-placement candidate')


def cpu_environment() -> dict:
    """Admit the existing CPU diagnostic flag and keep generated caches in task scratch."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, 'scale_diagnostic_authorized', 'Stereo placement CPU diagnostic')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    scratch = output_path(ROOT, 'data/.runtime/tmp/t1')
    for key in CACHE_FOLDERS:
        path = scratch / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    return runtime


def construction_inputs(episode: int, queue: dict, queue_path: str) -> tuple:
    """Resolve only video producers and fixed configuration for the candidate build."""
    scale_path = f'data/outputs/pose_insurance/scale/stereo_{episode:06d}.json'
    scale = read_json(scale_path)
    paths = [PROTOCOL, queue_path, FEATURE / 'runtime.json', FEATURE / 'placements_train.json',
             FEATURE.parent / 'mesh_extents/assignments.json', scale_path]
    if episode in EPISODES:
        measure, depth = train_parents(episode)
        parents = [D1, measure[0], depth[0], *scale['anchor_runs']]
    else:
        parents = [queue['tuning_builds'][str(episode)], *scale['anchor_runs']]
    return paths, parents, scale


def identity(request: StereoPlacementRequest, paths: list, parents: list) -> str:
    """Bind a resumable operation to exact inputs, producers and computation sources."""
    payload = dict(request=asdict(request), inputs=[input_file(ROOT, p) for p in (*paths, *parents)],
                   sources=source_hashes(FEATURE),
                   stereo_sources=source_hashes(FEATURE.parent / 'stereo_depth'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def observed_offset(episode: int, measure: tuple, depth: tuple, scale: dict, record: dict) -> tuple:
    """Read independent frame-zero clusters and admit stereo on their exact depth grid."""
    entries = read_json(FEATURE / 'placements_train.json')['episodes'][str(episode)]
    assignments = read_json(FEATURE.parent / 'mesh_extents/assignments.json')['episodes'][f'public:{episode}']['objects']
    with np.load(artifact(measure, 'labels_000000.npz', record)) as arrays:
        labels = arrays['label'].copy()
    _, masks, ids, defaults, merged = clusters_for(episode, entries, assignments, labels)
    details = dict(cluster_ids=ids, default_slots=defaults, merged_cluster=merged,
                   stereo_valid_pixels=[0, 0])
    if merged or any(mask is None for mask in masks):
        reason = 'default slot; merged cluster has no independent observed masks' if episode == 31 else (
            'merged cluster has no independent observed masks' if merged else 'slot without its own observed cluster')
        return None, reason, details
    try:
        stereo = load_stereo(episode, scale, depth, record)
    except FileNotFoundError:
        return None, 'no frame-zero stereo run', details
    with np.load(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])) as grid:
        intrinsics = grid['input_intrinsics_pixels'].copy()
    if stereo['rgb_depth_m'].shape != labels.shape or not np.allclose(stereo['rgb_intrinsics_pixels'], intrinsics, atol=1e-5):
        raise ValueError('Stereo and observed clusters do not share the exact MoGe grid')
    displacement, counts = stereo_offset(stereo, masks, np.array(measure[1]['frames'][0]['plane']['normal']))
    details['stereo_valid_pixels'] = counts
    return displacement, 'fewer than 300 valid stereo pixels on an object' if displacement is None else None, details


def construct(request: StereoPlacementRequest, queue: dict, paths: list, parents: list, scale: dict):
    """Record S1/T1 scenes and clearance factors without reading references or scans."""
    runtime = runtime_flag(FEATURE, 'scale_diagnostic_authorized', 'T1 construction')
    authorize_episode(runtime, request, 'T1 construction')
    record = run_fields(ROOT, 'pose_insurance', 'stereo_placement_build', asdict(request),
                        inputs=paths, parents=parents, allowed=('video_only',))
    record.update(split='public', episode_index=request.episode_index)
    directory = output_path(ROOT, f'{OUTPUT}/builds/{new_run_id()}')
    with recorded(directory, record):
        episode = request.episode_index
        if episode in EPISODES:
            measure, depth = train_parents(episode)
            geometry = video_geometry(PlacementRequest(episode, 'K1a' if episode in (0, 26) else 'v1'), record)
            names = ROSTER[episode]
            displacement, reason, observations = observed_offset(episode, measure, depth, scale, record)
        else:
            build = producer(queue['tuning_builds'][str(episode)], 'pose_insurance.build', request)
            geometry = tuning_geometry(build, record)
            names = build[1]['object_names']
            displacement, reason, observations = None, 'single-object control', {}
        s1 = scale_geometry(geometry, 'S1', scale['s'], scale['valid'])
        t1, clearance = placement(s1, displacement, reason)
        result = dict(episode=episode, names=names, clearance=clearance, observations=observations,
                      raw_offset_m=None if displacement is None else displacement.tolist(),
                      unchanged_fields=unchanged_fields(s1, t1),
                      scenes={name: {key: value.tolist() for key, value in scene.items()}
                              for name, scene in (('S1', s1), ('T1', t1))})
        write_json(directory / 'construction.json', result)
    return directory / 'run.json'


def resume_result(row: dict, operation: str, filename: str) -> dict:
    """Re-admit the completed operation, its consumed inputs and selected artifact."""
    record = read_run(ROOT, row['run'], kind=f'pose_insurance.{operation}')
    for entry in record['inputs']:
        checked_entry(entry, {'inputs': []})
    return read_json(artifact((ROOT / row['run'], record), filename, {'inputs': []}))


@lru_cache(maxsize=10)
def scoring_data(episode: int, names: tuple) -> tuple:
    """Load nearest-visible reference poses and deterministic seed-zero scan samples."""
    baseline = baseline_module()
    if episode in EPISODES:
        ref = reference(episode)
    else:
        ref_names, ref = baseline.load_reference(episode)
        ref = ref[:, :, [ref_names.index(name) for name in names]]
    return ref, np.stack([scan_vertices(name) for name in names])


@lru_cache(maxsize=16)
def scan_vertices(name: str) -> np.ndarray:
    """Reuse the existing baseline's deterministic 1,000-point scan sampler."""
    return baseline_module().vertices(name)


def scoring_inputs(episode: int, names: list) -> list:
    """List scoring-only labels, scan files, sampling settings and kit sources."""
    dataset = ROOT / baseline_module().ROOT
    meshes = []
    for name in names:
        path = dataset / f'mesh/{name}/{name}.glb'
        meshes.append(path if path.exists() else path.with_name(f'{name}_visual.glb'))
    return [PROTOCOL, FEATURE / 'runtime.json', FEATURE / 'diagnostic_sampling.json',
            ROOT / 'scripts/track3/static_pose_baseline.py', ROOT / baseline_module().KIT,
            *label_paths(episode), *meshes]


def diagnose(request: StereoPlacementRequest, plan: dict, build_path: str, paths: list):
    """Score one frozen scene with the existing kit integration in a recorded CPU trial."""
    runtime = runtime_flag(FEATURE, 'scale_diagnostic_authorized', 'T1 CPU diagnostic')
    authorize_episode(runtime, request, 'T1 CPU diagnostic')
    sampling = read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('T1 requires 1,000 surface samples per object, seed zero')
    build = producer(build_path, 'pose_insurance.stereo_placement_build', request)
    record = run_fields(ROOT, 'pose_insurance', 'stereo_placement_diagnostic', asdict(request),
                        inputs=paths, parents=[build[0]], provenance='public_label_diagnostic')
    record.update(split='public', episode_index=request.episode_index, object_names=plan['names'])
    artifact(build, 'construction.json', record)
    directory = output_path(ROOT, f'{OUTPUT}/runs/{new_run_id()}')
    with recorded(directory, record):
        scene = {key: np.array(value) for key, value in plan['scenes'][request.candidate].items()}
        ref, vertices = scoring_data(request.episode_index, tuple(plan['names']))
        baseline, start = baseline_module(), start_poses(scene)
        kit = baseline.load_kit()
        metrics = baseline.score(kit, np.repeat(start[None, None], len(ref), axis=0), ref, vertices)
        direction = direction_reading(scene, ref[0, 0], plan['raw_offset_m'], kit._ADD_THRESHOLDS_M) if len(start) > 1 else None
        result = dict(episode=request.episode_index, candidate=request.candidate, metrics=metrics,
                      direction=direction, frames=len(ref), sampling=sampling)
        record['metrics'] = metrics
        write_json(directory / 'metrics.json', result)
    return directory / 'run.json'


def operation(request, paths, parents, rows, ledger, execute, operation_name, filename) -> tuple:
    """Resume an exact completed operation or append one bounded recorded attempt."""
    key = identity(request, paths, parents)
    prior = next((row for row in reversed(rows) if row.get('identity') == key and row['status'] == 'completed'), None)
    if prior:
        return resume_result(prior, operation_name, filename), prior, False
    row = dict(identity=key, episode=request.episode_index, candidate=request.candidate,
               stage=operation_name, status='started')
    append(ledger, row)
    started = time.perf_counter()
    try:
        path = execute()
        row.update(status='completed', run=str(path.relative_to(ROOT)))
        result = resume_result(row, operation_name, filename)
    except Exception as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        if hasattr(error, 'feature_run_path'):
            row['run'] = error.feature_run_path
        raise
    finally:
        row['seconds'] = time.perf_counter() - started
        append(ledger, row)
    rows.append(row)
    return result, row, True


def run_queue(queue_path: str) -> None:
    """Freeze every video-only construction before any reference loading or scoring."""
    runtime, queue = cpu_environment(), read_json(queue_path)
    if (queue['episodes'] != list(EPISODES) or queue['controls'] != list(CONTROLS)
            or queue['candidates'] != ['S1', 'T1'] or not 1 <= queue['max_trials'] <= 20
            or not 1 <= queue['max_seconds'] <= 7200):
        raise ValueError('Queue must retain the fixed T1 roster, candidates and bounded budget')
    directory = output_path(ROOT, OUTPUT)
    directory.mkdir(parents=True, exist_ok=True)
    rows, ledger = resume_ledger(directory)
    started, plans, builds, selected = time.perf_counter(), {}, {}, []
    for episode in (*EPISODES, *CONTROLS):
        request = StereoPlacementRequest(episode, 'construction')
        authorize_episode(runtime, request, 'T1 queue')
        paths, parents, scale = construction_inputs(episode, queue, queue_path)
        plan, row, _ = operation(request, paths, parents, rows, ledger,
            lambda: construct(request, queue, paths, parents, scale), 'stereo_placement_build', 'construction.json')
        plans[episode], builds[episode] = plan, row['run']
        selected.append(row)
    append(ledger, dict(stage='all_constructions_frozen', status='completed', builds=builds,
                        clearance={ep: plan['clearance'] for ep, plan in plans.items()}))
    results, executed, resumed = {name: {} for name in queue['candidates']}, 0, 0
    for candidate in queue['candidates']:
        for episode, plan in plans.items():
            if time.perf_counter() - started >= queue['max_seconds']:
                raise RuntimeError('Time budget reached; resume unchanged queue')
            request = StereoPlacementRequest(episode, candidate)
            paths = [queue_path, *scoring_inputs(episode, plan['names'])]
            key = identity(request, paths, [builds[episode]])
            if executed >= queue['max_trials'] and not any(r.get('identity') == key and r['status'] == 'completed' for r in rows):
                raise RuntimeError('Trial budget reached; resume unchanged queue')
            result, row, fresh = operation(request, paths, [builds[episode]], rows, ledger,
                lambda: diagnose(request, plan, builds[episode], paths), 'stereo_placement_diagnostic', 'metrics.json')
            results[candidate][episode] = result
            selected.append(row)
            executed += int(fresh)
            resumed += int(not fresh)
            print(json.dumps(dict(episode=episode, candidate=candidate, resumed=not fresh, metrics=result['metrics'])), flush=True)
    from .stereo_placement_report import publish
    publish(results, plans, selected, executed, resumed, queue_path, directory)
    print(directory / 'summary.md', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', default='src/features/pose_insurance/stereo_placement_queue.json')
    run_queue(parser.parse_args().queue)


if __name__ == '__main__':
    main()
