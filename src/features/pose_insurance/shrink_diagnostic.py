"""Recorded CPU shrink trials using the existing scene and official metric integrations."""
from dataclasses import dataclass, asdict
from functools import lru_cache
import os
import sys
import numpy as np
import src.runs as runs
from src.paths import output_path
from src.provenance import run_fields, write_json
from .placement_diagnostic import (FEATURE, PlacementRequest, video_geometry, reference, baseline_module)
from .scale_diagnostic import train_parents, tuning_geometry, trial_inputs
from ._placement import transform_errors
from ._scale import scale_geometry
from ._shrink import EPISODES, CHANGED, CONTROLS, CANDIDATES, oracle_result

OUTPUT = 'data/outputs/pose_insurance/shrink'
PROTOCOL = runs.ROOT / 'docs/chain/pose-shrink-protocol.md'


@dataclass(frozen=True)
class ShrinkRequest:
    episode_index: int
    candidate: str
    factor: float
    source_split: str = 'public'

    def __post_init__(self):
        if self.source_split != 'public' or self.episode_index not in (*EPISODES, *CONTROLS):
            raise PermissionError('Shrink diagnostics require the fixed train/control episodes')
        if self.candidate not in (*CANDIDATES, 'oracle', 'H'):
            raise ValueError('Unknown registered shrink candidate')
        if not np.isfinite(self.factor) or not 0 <= self.factor <= 1.5:
            raise ValueError('Shrink factor outside the registered range')
        if self.candidate == 'oracle' and self.episode_index not in CHANGED:
            raise ValueError('Oracle is defined on the six changed episodes only')


def cpu_environment() -> dict:
    """Use the existing gate, interpreter check and default cache redirection."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runs.runtime_flag(FEATURE, 'scale_diagnostic_authorized', 'Shrink diagnostic')
    runs.heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    return runtime


@lru_cache(maxsize=10)
def episode_data(episode: int, build_path: str | None) -> tuple:
    """Cache admitted video geometry, filled references and deterministic scan samples."""
    record = {'inputs': []}
    baseline = baseline_module()
    if episode in EPISODES:
        # Resolve train parents through the established slice rather than a second loader.
        train_parents(episode)
        geometry = video_geometry(PlacementRequest(episode, 'K1a' if episode in (0, 26) else 'v1'), record)
        ref = reference(episode)
        from .placement_diagnostic import ROSTER
        names = ROSTER[episode]
    else:
        build = runs.producer(build_path, 'pose_insurance.build', ShrinkRequest(episode, 'I', 1.))
        geometry = tuning_geometry(build, record)
        names = build[1]['object_names']
        ref_names, ref = baseline.load_reference(episode)
        ref = ref[:, :, [ref_names.index(n) for n in names]]
    vertices = np.stack([scan_vertices(n) for n in names])
    return geometry, ref, vertices, names, record['inputs']


@lru_cache(maxsize=16)
def scan_vertices(name: str) -> np.ndarray:
    """Reuse the baseline's seed-zero scan surface sampler."""
    return baseline_module().vertices(name)


def inputs(request: ShrinkRequest, stereo: dict, queue: dict) -> tuple:
    """Extend the scale trial's provenance with the shrink protocol and complete assignments."""
    names, paths, parents = trial_inputs(request, stereo[request.episode_index], queue)
    paths += [PROTOCOL, FEATURE / 'shrink_queue.json',
              runs.ROOT / OUTPUT / 'assignments.json',
              runs.ROOT / 'data/outputs/pose_insurance/scale/summary.json']
    for ep in (*EPISODES, *CONTROLS):
        paths.append(runs.ROOT / f'data/outputs/pose_insurance/scale/stereo_{ep:06d}.json')
        parents += stereo[ep]['anchor_runs']
    if request.candidate == 'H':
        paths.append(runs.ROOT / OUTPUT / 'H_construction.json')
    return names, paths, list(dict.fromkeys(map(str, parents)))


def diagnose(request: ShrinkRequest, stereo: dict, queue: dict):
    """Score a fixed scene or the full registered MPPE oracle grid in one recorded CPU trial."""
    runtime = cpu_environment()
    runs.authorize_episode(runtime, request, 'shrink diagnostic')
    sampling = runs.read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('Shrink protocol requires 1,000 scan samples and seed zero')
    names, paths, parents = inputs(request, stereo, queue)
    record = run_fields(runs.ROOT, 'pose_insurance', 'shrink_diagnostic', asdict(request),
                        parents=parents, inputs=paths, provenance='public_label_diagnostic')
    record.update(split='public', episode_index=request.episode_index, object_names=names)
    directory = output_path(runs.ROOT, f'{OUTPUT}/runs/{runs.new_run_id()}')
    with runs.recorded(directory, record):
        geometry, ref, vertices, _, geometry_inputs = episode_data(
            request.episode_index, queue['tuning_builds'].get(str(request.episode_index)))
        for entry in geometry_inputs:
            runs.checked_entry(entry, record)
        baseline = baseline_module()
        kit = baseline.load_kit()
        if request.candidate == 'oracle':
            result = oracle(geometry, ref, kit)
        else:
            scene = scale_geometry(geometry, 'S1', request.factor, True)
            start = np.concatenate((scene['positions'], scene['quaternions'][:, [1, 2, 3, 0]]), axis=1)
            metrics = baseline.score(kit, np.repeat(start[None, None], len(ref), axis=0), ref, vertices)
            result = dict(metrics=metrics, scene={k: v.tolist() for k, v in scene.items()},
                          frame0=transform_errors(start, ref[0, 0]) if len(names) > 1 else None,
                          frames=len(ref), sampling=sampling)
            record['metrics'] = metrics
        result.update(episode=request.episode_index, candidate=request.candidate, factor=request.factor)
        write_json(directory / 'metrics.json', result)
    return directory / 'run.json'


def oracle(geometry: dict, ref: np.ndarray, kit) -> dict:
    """Evaluate every grid point with the kit's unmodified aligned MPPE function."""
    grid = np.arange(151) / 100.
    values = []
    for factor in grid:
        scene = scale_geometry(geometry, 'S1', factor, True)
        start = np.concatenate((scene['positions'], scene['quaternions'][:, [1, 2, 3, 0]]), axis=1)
        achieved = kit._warp_to_reference_start(np.repeat(start[None, None], len(ref), axis=0), ref)
        values.append(kit._mean_per_point_error_cm(achieved, ref))
    return oracle_result(grid, values, geometry['positions'], ref[0, 0])
