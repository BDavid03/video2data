"""Recorded CPU scale diagnostic using the pinned placement and kit integrations."""
from dataclasses import dataclass, asdict
import numpy as np
from track3.paths import output_path
from track3.provenance import run_fields, write_json, input_file
from track3.runs import (ROOT, read_json, producer, artifact, runtime_flag,
                         authorize_episode, heavy_execution, recorded, new_run_id)
from .placement_diagnostic import (FEATURE, ROSTER, PlacementRequest, video_geometry,
                                   reference, baseline_module, label_paths, KIT, D1)
from ._placement import transform_errors
from ._scale import scale_geometry

PROTOCOL = ROOT / 'docs/chain/pose-scale-protocol.md'
OUTPUT = 'outputs/pose_insurance/scale'


@dataclass(frozen=True)
class ScaleRequest:
    episode_index: int
    candidate: str
    source_split: str = 'public'

    def __post_init__(self):
        if self.source_split != 'public' or self.episode_index not in (*ROSTER, 39, 42):
            raise PermissionError('Scale diagnostics require the fixed train/tuning episodes')
        if self.candidate not in ('incumbent', 'S1', 'S2'):
            raise ValueError('Unknown fixed scale candidate')


def train_parents(episode):
    """Resolve the exact placement measurement and its monocular depth run."""
    entries = read_json(FEATURE / 'placements_train.json')['episodes'][str(episode)]
    measure = producer(entries['measure_run'], 'mesh_extents.measure')
    return measure, producer(measure[1]['request']['depth_run'], 'depth.generate')


def tuning_geometry(build, record):
    """Read the existing tuning scene without constructing another task."""
    scene = read_json(artifact(build, 'initial_scene.json', record))['scene']
    return dict(positions=np.array([r['position'] for r in scene]),
                dimensions=np.array([r['dimensions_m'] for r in scene]),
                quaternions=np.array([r['quaternion_wxyz'] for r in scene]))


def trial_inputs(request, stereo, queue):
    """Pin the protocol, stereo anchors, placement sources and scoring inputs."""
    baseline = baseline_module()
    if request.episode_index in ROSTER:
        names = ROSTER[request.episode_index]
        measure, depth = train_parents(request.episode_index)
        parents = [D1, measure[0], depth[0]]
    else:
        build = producer(queue['tuning_builds'][str(request.episode_index)], 'pose_insurance.build', request)
        names = build[1]['object_names']
        parents = [build[0], stereo['depth_run']]
    dataset = ROOT / baseline.ROOT
    meshes = []
    for name in names:
        path = dataset / f'mesh/{name}/{name}.glb'
        meshes.append(path if path.exists() else path.with_name(f'{name}_visual.glb'))
    inputs = [PROTOCOL, ROOT / 'docs/chain/pose-placement-protocol.md', FEATURE / 'runtime.json',
              FEATURE / 'diagnostic_sampling.json', FEATURE / 'placements_train.json',
              FEATURE.parent / 'mesh_extents/assignments.json', ROOT / D1,
              ROOT / 'scripts/track3/static_pose_baseline.py', KIT / 'metric_code/track_3/AUC.py',
              *label_paths(request.episode_index), *meshes]
    return names, inputs, [*parents, *stereo['anchor_runs']]


def diagnose(request, stereo, queue):
    """Apply one fixed scale transform and score a static world with kit functions."""
    runtime = runtime_flag(FEATURE, 'scale_diagnostic_authorized', 'Scale diagnostic')
    authorize_episode(runtime, request, 'scale diagnostic')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    sampling = read_json(FEATURE / 'diagnostic_sampling.json')
    if sampling['surface_points_per_object'] != 1000 or sampling['seed'] != 0:
        raise ValueError('Scale protocol requires 1,000 scan samples, seed zero')
    names, inputs, parents = trial_inputs(request, stereo, queue)
    record = run_fields(ROOT, 'pose_insurance', 'scale_diagnostic', asdict(request),
                        parents=parents, inputs=inputs, provenance='public_label_diagnostic')
    record.update(split='public', episode_index=request.episode_index, object_names=names)
    directory = output_path(ROOT, f'{OUTPUT}/runs/{new_run_id()}')
    with recorded(directory, record):
        if request.episode_index in ROSTER:
            placement = 'K1a' if request.episode_index in (0, 26) else 'v1'
            geometry = video_geometry(PlacementRequest(request.episode_index, placement), record)
            ref = reference(request.episode_index)
        else:
            build = producer(queue['tuning_builds'][str(request.episode_index)], 'pose_insurance.build', request)
            geometry = tuning_geometry(build, record)
            ref_names, ref = baseline_module().load_reference(request.episode_index)
            ref = ref[:, :, [ref_names.index(n) for n in names]]
        transformed = scale_geometry(geometry, request.candidate, stereo['s'], stereo['valid'])
        start = np.concatenate((transformed['positions'], transformed['quaternions'][:, [1, 2, 3, 0]]), axis=1)
        baseline = baseline_module()
        vertices = np.stack([baseline.vertices(n) for n in names])
        metrics = baseline.score(baseline.load_kit(), np.repeat(start[None, None], len(ref), axis=0), ref, vertices)
        result = dict(episode=request.episode_index, candidate=request.candidate, names=names,
                      metrics=metrics, scale={k: v for k, v in stereo.items() if k != 'frames'},
                      frame0=transform_errors(start, ref[0, 0]) if len(names) > 1 else None,
                      scene={k: v.tolist() for k, v in transformed.items()}, frames=len(ref),
                      mass_rule='Existing pose_insurance fixed mass 0.1 kg; recompute inertia from dimensions',
                      sampling=sampling)
        write_json(directory / 'metrics.json', result)
        record['metrics'] = metrics
    return directory / 'run.json'
