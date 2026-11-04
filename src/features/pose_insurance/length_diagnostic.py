"""Read admitted CPU artifacts and run the request-defined pose length diagnostic."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from src.paths import output_path, project_path
from src.provenance import input_file, run_fields, write_json
from src.runs import (ROOT, read_json, producer, artifact, runtime_flag,
                         authorize_episode, heavy_execution, recorded, new_run_id)

FEATURE = Path(__file__).parent
PROTOCOL = 'docs/chain/pose-length-diagnostic-protocol.md'
OUTPUT = 'data/outputs/pose_insurance/length'
REPRODUCE = ("PYTHONPATH=. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
             'data/.runtime/linux-cuda/bin/python -m src.features.pose_insurance '
             'length_diagnostic --request src/features/pose_insurance/length_request.json')


@dataclass(frozen=True)
class LengthRequest:
    episodes: tuple = field(default_factory=lambda: tuple(read_json(FEATURE/'length_request.json')['episodes']))
    source_split: str = 'public'
    max_seconds: int = 600
    preparation: tuple = ()
    assignments: str | None = None
    groups: dict = field(default_factory=dict)
    counted_episodes: tuple | None = None
    video: tuple = ()
    candidates: tuple = ()
    minimum_pixels: int = 300
    minimum_gain_cm: float = 5
    minimum_lower: int = 4
    maximum_mae_cm: float = 3
    default_offset_episodes: tuple = ()
    preparation_state: str | None = None
    presentation: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source_split != 'public' or not self.episodes or any(ep in (41, 43, 44) for ep in self.episodes):
            raise PermissionError('Length diagnostics require public episodes and exclude 41/43/44')
        if not 1 <= self.max_seconds <= 7200:
            raise ValueError('Length diagnostic requires a bounded CPU time budget')
        subsets = [*self.groups.values(), self.counted_episodes or ()]
        if len(set(self.episodes)) != len(self.episodes) or any(not set(group) <= set(self.episodes) for group in subsets):
            raise ValueError('Length subsets must belong to the requested roster')
        if self.counted_episodes is not None and not (self.assignments and self.preparation_state):
            raise ValueError('Confirmation requires frozen RGB assignments and preparation state')

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


def admit(parent: tuple, record: dict) -> None:
    """Add an already admitted parent to the consumer's hashed inputs."""
    entry = input_file(ROOT, parent[0])
    if entry not in record['inputs']:
        record['inputs'].append(entry)


def frame_arrays(parent: tuple, record: dict) -> dict:
    """Read the admitted frame-zero NPZ from an existing producer."""
    import numpy as np
    frame = next(r for r in parent[1]['frames'] if r['frame_index'] == 0 and r['status'] == 'completed')
    with np.load(artifact(parent, frame['artifact'], record, frame['sha256'])) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def clusters_for(episode: int, entries: dict, assignments: dict, labels: np.ndarray) -> tuple:
    """Resolve incumbent and observed masks with the existing placement conventions."""
    from .placement_diagnostic import ROSTER
    from ._length import cluster_masks
    return cluster_masks(ROSTER[episode], entries, assignments, labels, episode in (0, 26))


def load_stereo(episode: int, scale: dict, depth: tuple, record: dict, request=None) -> dict:
    """Read the existing calibrated stereo points on the exact monocular colour grid."""
    from .placement_diagnostic import PlacementRequest
    for path in scale['anchor_runs']:
        anchor = producer(path, 'stereo_depth.anchor')
        if any(r['frame_index'] == 0 for r in anchor[1]['frames']):
            break
    else:
        raise FileNotFoundError('No recorded frame-zero stereo anchor')
    if anchor[1]['request']['depth_run'] != scale['depth_run']:
        raise ValueError('Stereo anchor uses a different depth run')
    stereo = producer(anchor[1]['request']['stereo_run'], 'stereo_depth.generate', request or PlacementRequest(episode))
    admit(anchor, record)
    admit(stereo, record)
    if stereo[1]['grid'] != depth[1]['preprocessing_artifact_sha256']:
        raise ValueError('Stereo and MoGe pixel grids differ')
    arrays = frame_arrays(stereo, record)
    if int(arrays['frame_index']) != 0 or int(arrays['pts']) != 0:
        raise ValueError('Expected the static frame-zero stereo points')
    return arrays


def scan_centres(names: list, record: dict) -> np.ndarray:
    """Use exact scan vertices for AABB centres rather than sampled surface points."""
    import numpy as np
    from .placement_diagnostic import baseline_module
    import trimesh
    dataset = project_path(ROOT, baseline_module().ROOT)
    centres = []
    for name in names:
        path = dataset / f'mesh/{name}/{name}.glb'
        if not path.exists():
            path = path.with_name(f'{name}_visual.glb')
        record['inputs'].append(input_file(ROOT, path))
        centres.append(trimesh.load(path, force='mesh').bounds.mean(axis=0))
    return np.array(centres)


def read_grid(measure: tuple, depth: tuple, record: dict) -> dict:
    """Admit frame-zero labels, depths and their calibration without constructing geometry."""
    import numpy as np
    for parent in (measure, depth):
        admit(parent, record)
    with np.load(artifact(measure, 'labels_000000.npz', record)) as arrays:
        labels = arrays['label'].copy()
    with np.load(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])) as grid:
        intrinsics = grid['input_intrinsics_pixels'].copy()
    return dict(labels=labels, depth=frame_arrays(depth, record), intrinsics=intrinsics,
                normal=np.array(measure[1]['frames'][0]['plane']['normal']))


def load_episode(episode: int, record: dict, placements: dict, assignments: dict,
                 selected: list) -> dict:
    """Read the train scene and its recorded S1 comparator."""
    from .placement_diagnostic import ROSTER, PlacementRequest, video_geometry, reference
    from .scale_diagnostic import train_parents
    from ._scale import scale_geometry
    measure, depth = train_parents(episode)
    data = read_grid(measure, depth, record)
    geometry = video_geometry(PlacementRequest(episode, 'K1a' if episode in (0, 26) else 'v1'), record)
    scale = read_json(f'data/outputs/pose_insurance/scale/stereo_{episode:06d}.json')
    s1_path = next(row['run'] for row in selected if row.get('candidate') == 'S1' and row['episode'] == episode)
    s1 = producer(s1_path, 'pose_insurance.scale_diagnostic', PlacementRequest(episode),
                  allowed=('public_label_diagnostic',))
    admit(s1, record)
    return dict(data, geometry=geometry, scaled=scale_geometry(geometry, 'S1', scale['s'], scale['valid']),
                recorded_scene=read_json(artifact(s1, 'metrics.json', record))['scene'], scale=scale,
                masks=clusters_for(episode, placements[str(episode)], assignments[f'public:{episode}']['objects'], data['labels']),
                stereo=load_stereo(episode, scale, depth, record), poses=reference(episode)[0, 0],
                names=ROSTER[episode], scan_centres=scan_centres(ROSTER[episode], record))


def load_confirmation(episode: int, entry: dict, state: dict, record: dict) -> dict:
    """Read frozen RGB unions and their admitted stereo scale without an incumbent scene."""
    from .placement_diagnostic import baseline_module
    request = SimpleNamespace(episode_index=episode, source_split='public')
    measure = producer(entry['measure_run'], 'mesh_extents.measure', request)
    depth = producer(entry['depth_run'], 'depth.generate', request)
    data = read_grid(measure, depth, record)
    from ._length import confirmation_masks
    masks = confirmation_masks(entry, data['labels'])
    scale, stereo = None, None
    if 'scale' in state:
        parent = producer(state['scale'], 'pose_insurance.length_confirmation_scale')
        admit(parent, record)
        scale = read_json(artifact(parent, 'scale.json', record))
        stereo = load_stereo(episode, scale, depth, record, request)
    names, poses = baseline_module().load_reference(episode)
    return dict(data, scale=scale, stereo=stereo, masks=masks,
                poses=poses[0, 0, [names.index(name) for name in entry['names']]], names=entry['names'],
                scan_centres=scan_centres(entry['names'], record), merged_cluster_ids=entry['merged_cluster_ids'])


def episode_trial(episode: int, base: dict, data: dict, request: LengthRequest, directory) -> tuple:
    """Resume one row by the exact admitted input, request and source identity."""
    from ._length import build_row
    identity = hashlib.sha256(json.dumps(dict(inputs=base['inputs'], sources=base['sources'],
                                              request=asdict(request), episode=episode), sort_keys=True).encode()).hexdigest()
    ledger = directory/'ledger.jsonl'
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    prior = next((row for row in reversed(previous) if row['identity'] == identity), None)
    if prior:
        parent = producer(prior['run'], 'pose_insurance.length_diagnostic', allowed=('public_label_diagnostic',))
        return read_json(artifact(parent, 'episode.json', {'inputs': []})), prior['run'], True
    trial = directory/'episodes'/new_run_id()
    record = dict(base, kind='pose_insurance.length_diagnostic', episode_index=episode, split='public', identity=identity)
    counted = None if request.counted_episodes is None else episode in request.counted_episodes
    with recorded(trial, record):
        row = build_row(episode, data['names'], data, request.video, request.minimum_pixels, counted)
        write_json(trial/'episode.json', row)
    path = str((trial/'run.json').relative_to(ROOT))
    with ledger.open('a') as stream:
        stream.write(json.dumps(dict(identity=identity, episode=episode, status='completed', run=path))+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return row, path, False


def diagnostic_inputs(request: LengthRequest, request_path: str, publish: bool) -> tuple:
    """Resolve each protocol's input configuration after admission."""
    from .placement_diagnostic import label_paths, D1
    confirm = request.counted_episodes is not None
    paths = [PROTOCOL, request_path, FEATURE/'runtime.json', ROOT/'scripts/track3/static_pose_baseline.py',
             *[path for ep in request.episodes for path in label_paths(ep)]]
    if confirm:
        paths += [request.assignments, request.preparation_state]
        assignments = read_json(request.assignments)['episodes']
        state = read_json(request.preparation_state)
        if sorted(map(int, assignments)) != list(request.episodes):
            raise ValueError('Confirmation assignments do not match the requested roster')
        if any(entry['counted'] != (int(ep) in request.counted_episodes) for ep, entry in assignments.items()):
            raise ValueError('Frozen assignments and requested counted set differ')
        destination = f'data/outputs/pose_insurance/length-confirm/{new_run_id()}'
    else:
        paths += [FEATURE/'placements_train.json', FEATURE.parent/'mesh_extents/assignments.json',
                  ROOT/'data/outputs/pose_insurance/scale/finished.json',
                  *[ROOT/f'data/outputs/pose_insurance/scale/stereo_{ep:06d}.json' for ep in request.episodes]]
        placements = read_json(FEATURE/'placements_train.json')['episodes']
        assignments = read_json(FEATURE.parent/'mesh_extents/assignments.json')['episodes']
        selected = read_json('data/outputs/pose_insurance/scale/finished.json')['selected']
        destination = OUTPUT if publish else f'{OUTPUT}/reruns/{new_run_id()}'
    context = dict(assignments=assignments, state=state) if confirm else dict(
        placements=placements, assignments=assignments, selected=selected)
    return paths, [] if confirm else [D1], destination, context


def diagnose(request: LengthRequest, request_path: str, publish: bool = True):
    """Admit the request, then run one bounded measurement loop for either protocol."""
    runtime = runtime_flag(FEATURE, 'length_diagnostic_authorized', 'Pose length diagnostic')
    for episode in request.episodes:
        authorize_episode(runtime, SimpleNamespace(episode_index=episode, source_split=request.source_split), 'length diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    from ._length import summarize, summarize_confirmation
    from ._length_report import markdown, figure
    confirm = request.counted_episodes is not None
    paths, parents, destination, context = diagnostic_inputs(request, request_path, publish)
    base = run_fields(ROOT, 'pose_insurance', 'length_confirmation' if confirm else 'length_diagnostic',
                      asdict(request), inputs=paths, parents=parents, provenance='public_label_diagnostic')
    directory = output_path(ROOT, destination)
    if not confirm:
        directory.mkdir(parents=True, exist_ok=True)
    existing = [directory/name for name in ('run.json', 'summary.json', 'summary.md', 'lengths.png') if (directory/name).exists()]
    if existing:
        backup = output_path(ROOT, f'data/.runtime/tmp/pose-length/backups/{new_run_id()}')
        backup.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, backup/path.name)
    report_directory = directory if confirm else directory/'reports'/new_run_id()
    started = time.monotonic()
    trial_base = dict(base, inputs=base['inputs'].copy())
    with recorded(report_directory, base, hash_artifacts=False):
        rows, resumed = [], 0
        for episode in request.episodes:
            if time.monotonic() - started >= request.max_seconds:
                raise RuntimeError('Time budget reached; resume the same request')
            record = dict(trial_base, inputs=trial_base['inputs'].copy())
            data = (load_confirmation(episode, context['assignments'][str(episode)], context['state'][str(episode)], record) if confirm else
                    load_episode(episode, record, context['placements'], context['assignments'], context['selected']))
            data['default_half_m_offset'] = episode in request.default_offset_episodes
            row, path, reused = episode_trial(episode, record, data, request, directory)
            base['inputs'].append(input_file(ROOT, ROOT/path))
            rows.append(row)
            resumed += int(reused)
            print(json.dumps(dict(episode=episode, resumed=reused, lengths_cm=row['lengths_cm'])), flush=True)
        if confirm:
            report = summarize_confirmation(rows, request.counted_episodes, request.presentation['video'],
                                            request.maximum_mae_cm, request.minimum_gain_cm)
        else:
            report = dict(**summarize(rows, request.groups, request.video, request.candidates,
                                     request.minimum_gain_cm, request.minimum_lower),
                          resumed=resumed, executed=len(rows)-resumed, notes=NOTES,
                          s1_max_position_difference_m=max(row['s1_max_position_difference_m'] for row in rows))
        report.update(episodes=rows, reproduce=REPRODUCE.replace('length_request.json', 'length_confirm_request.json') if confirm else REPRODUCE)
        write_json(report_directory/'summary.json', report)
        (report_directory/'summary.md').write_text(markdown(report, request.presentation))
        figure(rows, report_directory/'lengths.png', request.presentation)
        base['artifacts'] = {name: input_file(ROOT, report_directory/name)['sha256'] for name in ('summary.json', 'summary.md', 'lengths.png')}
    if not confirm:
        for name in ('summary.json', 'summary.md', 'lengths.png', 'run.json'):
            shutil.copy2(report_directory/name, directory/name)
    return directory/'run.json'


NOTES = [
    '- References reuse the existing nearest-visible-filled placement reader at frame zero; these are descriptive public measurements. M_o/M_c use mocap world horizontal xy. M_c uses the exact scan-frame vertex AABB midpoint transformed by each mocap pose; no sampled-mesh approximation.',
    '- V_inc reuses video_geometry + scale_geometry(S1), checked against the admitted recorded S1 scene; full positions, not just their lengths, must agree within 1e-6 m.',
    '- V_moge uses unscaled observed MoGe cluster means; V_stereo uses the same observed masks with direct calibrated rgb_depth_m points and >=300 valid pixels per object, without MoGe depth or scale.',
    '- V_hull replaces each incumbent point mean by the midpoint of its min/max footprint bounds in the incumbent table basis, then applies the same S1 scale. Defaults and recorded separation shifts are retained (all shifts here are zero).',
    '- The fixed combination V_stereo_hull uses footprint-bounds midpoints on those same direct stereo points, with the same 300-pixel rule; no MoGe scaling is applied.',
    '- Episode 12 is in the protocol-defined clean five but S1 uses a slot-0 default: its border cup is observed cluster 1. V_moge/V_stereo use that explicitly recorded observed mask; V_hull retains the incumbent default. This is an assignment difference as well as a depth-source comparison.',
    '- Episodes 0/26 share one merged cluster and have zero S1 horizontal offset; 31 retains the 0.5 m pre-scale fallback. Independent centroid constructions are missing on all three, never split or substituted. All cluster pixel counts and assignment ids are in summary.json.',
    '- Along/cross errors use camera optical +z projected into the incumbent table plane. Reference vectors are transported by the kit slot-0 rigid alignment R_video,0 R_mocap,0^-1. They are conditional on that body alignment, not a measured camera-to-mocap yaw calibration; anchor up errors can tilt the reference plane. Aligned reference vectors are retained in JSON.',
    '- The reported M_o/M_c lengths remain world-horizontal; they need not equal the horizontal norms after the scoring alignment used only for ray errors. Thus ray errors include direction and body-frame error, not just scalar length error.',
    '- The earlier 14.713 cm S1 result used 3D relative lengths on six changed episodes. This protocol uses horizontal lengths and also reports that same six-episode subset for comparison.',
    '- This is one descriptive CPU measurement, with a resumable per-episode ledger. No scene, policy, package, model, or selection is changed; no excluded episode is read.',
]
