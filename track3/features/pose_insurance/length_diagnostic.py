"""Measure the fixed train-only pose length protocol from existing CPU-readable artifacts."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np
from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, read_json, producer, artifact, runtime_flag,
                         authorize_episode, heavy_execution, recorded, new_run_id)
from .placement_diagnostic import (FEATURE, ROSTER, PlacementRequest, video_geometry,
                                   reference, baseline_module, label_paths, D1)
from .scale_diagnostic import train_parents
from ._scene import camera_points
from ._scale import scale_geometry
from ._length import (VIDEO, footprint_centres, offset, reference_offsets, hull_offset,
                      measurements, ray_errors, summarize)

PROTOCOL = 'docs/chain/pose-length-diagnostic-protocol.md'
OUTPUT = 'outputs/pose_insurance/length'
REPRODUCE = ("PYTHONPATH=. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
             '.runtime/linux-cuda/bin/python -m track3.features.pose_insurance '
             'length_diagnostic --request track3/features/pose_insurance/length_request.json')


@dataclass(frozen=True)
class LengthRequest:
    episodes: tuple = tuple(ROSTER)
    source_split: str = 'public'
    max_seconds: int = 600
    preparation: tuple = ()
    assignments: str | None = None

    def __post_init__(self):
        if tuple(self.episodes) not in (tuple(ROSTER), (2, 13, 22, 24, 27)) or self.source_split != 'public':
            raise PermissionError('Length protocol requires the fixed train or Amendment 1 confirmation roster')
        if tuple(self.episodes) == (2, 13, 22, 24, 27) and self.assignments != 'track3/features/pose_insurance/length_confirm_assignments.json':
            raise ValueError('Confirmation requires its frozen RGB assignments')
        if not 1 <= self.max_seconds <= 7200:
            raise ValueError('Length diagnostic requires a bounded CPU time budget')

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
    frame = next(r for r in parent[1]['frames'] if r['frame_index'] == 0 and r['status'] == 'completed')
    with np.load(artifact(parent, frame['artifact'], record, frame['sha256'])) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def clusters_for(episode: int, entries: dict, assignments: dict, labels: np.ndarray) -> tuple:
    """Resolve incumbent and observed cluster masks from the recorded assignments."""
    incumbent, observed, ids, defaults = [], [], [], []
    for name in ROSTER[episode]:
        entry = entries['slots'].get(name)
        chosen = entry['cluster_ids'] if entry else assignments[name]
        seen = chosen if chosen is not None else entry.get('observed_cluster_ids')
        defaults.append(chosen is None)
        ids.append(dict(incumbent=chosen, observed=seen))
        incumbent.append(None if chosen is None else np.isin(labels, np.asarray(chosen) + 1))
        observed.append(None if seen is None else np.isin(labels, np.asarray(seen) + 1))
    merged = entries['merged_cluster_ids'] is not None
    if merged:
        observed = [None, None]
        if episode in (0, 26):
            mask = np.isin(labels, np.asarray(entries['merged_cluster_ids']) + 1)
            incumbent = [mask, mask]
    return incumbent, observed, ids, defaults, merged


def load_stereo(episode: int, scale: dict, depth: tuple, record: dict, request=None) -> dict:
    """Read the existing calibrated stereo points on the exact monocular colour grid."""
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


def load_episode(episode: int, record: dict, placements: dict, assignments: dict,
                 selected: list) -> dict:
    """Reuse placement and scale readers and admit the recorded S1 scene for comparison."""
    entries = placements[str(episode)]
    measure, depth = train_parents(episode)
    for parent in (measure, depth):
        admit(parent, record)
    geometry = video_geometry(PlacementRequest(episode, 'K1a' if episode in (0, 26) else 'v1'), record)
    scale = read_json(f'outputs/pose_insurance/scale/stereo_{episode:06d}.json')
    scaled = scale_geometry(geometry, 'S1', scale['s'], scale['valid'])
    s1_path = next(r['run'] for r in selected if r.get('candidate') == 'S1' and r['episode'] == episode)
    s1 = producer(s1_path, 'pose_insurance.scale_diagnostic', PlacementRequest(episode),
                  allowed=('public_label_diagnostic',))
    admit(s1, record)
    recorded_scene = read_json(artifact(s1, 'metrics.json', record))['scene']
    with np.load(artifact(measure, 'labels_000000.npz', record)) as arrays:
        labels = arrays['label'].copy()
    arrays = frame_arrays(depth, record)
    with np.load(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])) as grid:
        k = grid['input_intrinsics_pixels'].copy()
    points = camera_points(arrays['depth_m'].astype(float), k)
    masks = clusters_for(episode, entries, assignments[f'public:{episode}']['objects'], labels)
    stereo = load_stereo(episode, scale, depth, record)
    if stereo['rgb_depth_m'].shape != labels.shape or not np.allclose(stereo['rgb_intrinsics_pixels'], k, atol=1e-5):
        raise ValueError('Stereo labels or intrinsics differ from the incumbent grid')
    return dict(geometry=geometry, scaled=scaled, recorded_scene=recorded_scene, scale=scale,
                points=points, masks=masks, stereo=stereo,
                normal=np.array(measure[1]['frames'][0]['plane']['normal']),
                label_pixel_counts={str(int(k)-1): int(v) for k, v in zip(*np.unique(labels[labels > 0], return_counts=True))},
                poses=reference(episode)[0, 0], scan_centres=scan_centres(ROSTER[episode], record))


def measure_episode(episode: int, data: dict) -> dict:
    """Assemble array computations without invoking policy, scene writing or GPU handlers."""
    inc_masks, obs_masks, ids, defaults, merged = data['masks']
    clusters = [None if m is None else data['points'][m] for m in obs_masks]
    inc_clusters = [None if m is None else data['points'][m] for m in inc_masks]
    stereo = data['stereo']
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    stereo_points = camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels'])
    counts = [0 if m is None else int((m & valid).sum()) for m in obs_masks]
    stereo_clusters = [stereo_points[m & valid] if m is not None and n >= 300 else None
                       for m, n in zip(obs_masks, counts)]
    refs = reference_offsets(data['poses'], data['scan_centres'])
    positions = data['scaled']['positions']
    vectors = dict(refs, V_inc=positions[1, :2] - positions[0, :2],
                   V_moge=offset(footprint_centres(clusters, data['normal'])),
                   V_stereo=offset(footprint_centres(stereo_clusters, data['normal'])),
                   V_hull=hull_offset(data['geometry'], inc_clusters, data['normal'],
                                      data['scale']['s'] if data['scale']['valid'] else 1.),
                   V_stereo_hull=offset(footprint_centres(stereo_clusters, data['normal'], bounds=True)))
    mismatch = float(np.max(np.abs(positions - np.asarray(data['recorded_scene']['positions']))))
    if mismatch > 1e-6:
        raise ValueError(f'S1 scene reproduction failed: {mismatch} m')
    return dict(episode=episode, names=ROSTER[episode], **measurements(vectors),
                vectors_m={k: v.tolist() if v is not None else None for k, v in vectors.items()},
                ray=ray_errors(vectors, refs, data['poses'], data['geometry']['quaternions'][0], data['normal']),
                cluster_ids=ids, merged_cluster=merged, default_slots=defaults,
                default_half_m_offset=episode == 31,
                cluster_pixels=[sum(data['label_pixel_counts'][str(i)] for i in row['observed'])
                                if row['observed'] is not None else None for row in ids],
                label_pixel_counts=data['label_pixel_counts'],
                stereo_valid_pixels=counts, stereo_minimum_pixels=300,
                missing={k: ('merged cluster has no independent object pixels' if merged else
                             f'stereo cluster support {counts}; need 300 valid pixels in each')
                         for k in VIDEO if vectors[k] is None},
                scale=data['scale']['s'], separation_shifts_cm=data['geometry']['shifts'].tolist(),
                s1_max_position_difference_m=mismatch)


def markdown(report: dict) -> str:
    """Render plain-unit tables, missingness and the fixed authorization verdict."""
    def fmt(value):
        return 'missing' if value is None else f'{value:.3f}'
    quantities = ('M_o', 'M_c', *VIDEO)
    lines = ['# Train pose length diagnostic', '', report['authorization']['verdict'], '',
             'Frame 0 only; all lengths are horizontal table-plane distances in cm. Public mocap and scans are measurement only.', '',
             '| Episode | ' + ' | '.join(quantities) + ' |', '|---|' + '---:|' * len(quantities)]
    for row in report['episodes']:
        lines.append(f"| {row['episode']} | " + ' | '.join(fmt(row['lengths_cm'][k]) for k in quantities) + ' |')
    lines += ['', 'Ratios to M_o (unitless):', '', '| Episode | ' + ' | '.join(quantities[1:]) + ' |',
              '|---|' + '---:|' * (len(quantities) - 1)]
    for row in report['episodes']:
        lines.append(f"| {row['episode']} | " + ' | '.join(fmt(row['ratios_to_M_o'][k]) for k in quantities[1:]) + ' |')
    lines += ['', '| Subset | Video construction | Count | MAE vs M_o (cm) | MAE vs M_c (cm) |', '|---|---|---:|---:|---:|']
    for group, entries in report['means'].items():
        for name in VIDEO:
            r = entries[name]
            lines.append(f"| {group} | {name} | {r['count']} | {fmt(r['mae_vs_M_o_cm'])} | {fmt(r['mae_vs_M_c_cm'])} |")
    lines += ['', '| Episode | M_c − M_o (cm) | Absolute origin length difference (cm) | V_inc − M_c (cm) | Pixels slot 0/1 | Stereo valid slot 0/1 | Merged | Default slots 0/1 | 0.5 m fallback |', '|---|---:|---:|---:|---|---|---|---|---|']
    for r in report['episodes']:
        v = r['lengths_cm']
        lines.append(f"| {r['episode']} | {fmt(v['M_c']-v['M_o'])} | {fmt(abs(v['M_c']-v['M_o']))} | {fmt(v['V_inc']-v['M_c'])} | {r['cluster_pixels']} | {r['stereo_valid_pixels']} | {r['merged_cluster']} | {r['default_slots']} | {r['default_half_m_offset']} |")
    lines += ['', 'Signed decomposition is exact: V − M_o = (M_c − M_o) + (V − M_c); absolute terms do not generally add.', 'The JSON retains signed and absolute length decompositions for every video construction.', '',
              '| Subset | Mean absolute M_c − M_o (cm) |', '|---|---:|']
    lines += [f"| {g} | {fmt(v['origin_offset_mae_cm'])} |" for g, v in report['means'].items()]
    lines += ['', 'Along/cross-ray signed offset error (cm), after the kit slot-0 rigid alignment:', '',
              '| Episode | Construction | Along vs M_o | Cross vs M_o | Along vs M_c | Cross vs M_c |', '|---|---|---:|---:|---:|---:|']
    for row in report['episodes']:
        for name, r in row['ray']['errors'].items():
            vals = [r[ref][key] for ref in ('M_o', 'M_c') for key in ('along_cm', 'cross_cm')] if r else [None]*4
            lines.append(f"| {row['episode']} | {name} | " + ' | '.join(map(fmt, vals)) + ' |')
    lines += ['', '| Fixed candidate | Clean count | MAE gain vs V_inc (cm) | Episodes lower | Pass |', '|---|---:|---:|---:|---|']
    for name, r in report['authorization']['checks'].items():
        lines.append(f"| {name} | {r['count']} | {fmt(r['gain_cm'])} | {r['lower_episodes']} | {r['passed']} |")
    lines += ['', '## Definitions and limits', '', *report['notes'], '',
              'S1 maximum absolute position reproduction difference: ' + str(report['s1_max_position_difference_m']) + ' m.', '',
              '![Per-episode lengths and ray error](lengths.png)', '', 'Reproduce:', '', '```bash', REPRODUCE, '```', '']
    return '\n'.join(lines)


def figure(rows: list, path) -> None:
    """Plot all fixed lengths and aligned ray-error magnitudes in one labelled figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(rows))
    for index, name in enumerate(('M_o', 'M_c', *VIDEO)):
        values = [np.nan if r['lengths_cm'][name] is None else r['lengths_cm'][name] for r in rows]
        axes[0].bar(x + (index-3)*.11, values, .11, label=name)
    for index, name in enumerate(VIDEO):
        for component, hatch in (('along_cm', ''), ('cross_cm', '//')):
            values = [abs(r['ray']['errors'][name]['M_o'][component]) if r['ray']['errors'][name] else np.nan for r in rows]
            axes[1].bar(x + (index-2)*.17 + (-.04 if not hatch else .04), values, .08,
                        label=f'{name} {component[:-3]}', hatch=hatch, color=f'C{index+2}')
    for ax, title in zip(axes, ('Frame-zero horizontal lengths (zero/missing cases detailed in report)',
                              'Absolute horizontal offset-error components vs M_o; slot-0 rigid alignment')):
        ax.set(title=title, xlabel='Train episode', ylabel='cm', xticks=x,
               xticklabels=[str(r['episode']) for r in rows])
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
        ax.legend(ncol=5, fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def episode_trial(episode: int, base: dict, placements: dict, assignments: dict,
                  selected: list, directory, ledger) -> tuple:
    """Resume one measurement by hashed inputs and sources, recording each completed episode."""
    record = dict(base, inputs=base['inputs'].copy(), request=dict(episode_index=episode, source_split='public'))
    data = load_episode(episode, record, placements, assignments, selected)
    identity = hashlib.sha256(json.dumps(dict(inputs=record['inputs'], sources=record['sources'],
                                              request=record['request']), sort_keys=True).encode()).hexdigest()
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    prior = next((r for r in reversed(previous) if r['identity'] == identity and r['status'] == 'completed'), None)
    if prior:
        parent = producer(prior['run'], 'pose_insurance.length_diagnostic', allowed=('public_label_diagnostic',))
        return read_json(artifact(parent, 'episode.json', {'inputs': []})), prior['run'], True
    trial = directory / 'episodes' / new_run_id()
    record.update(episode_index=episode, split='public', identity=identity)
    with recorded(trial, record):
        result = measure_episode(episode, data)
        write_json(trial / 'episode.json', result)
    path = str((trial / 'run.json').relative_to(ROOT))
    with ledger.open('a') as stream:
        stream.write(json.dumps(dict(identity=identity, episode=episode, status='completed', run=path)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return result, path, False


def diagnose(request: LengthRequest, request_path: str, publish: bool = True):
    """Run the bounded fixed episode loop and write the descriptive report and provenance."""
    if tuple(request.episodes) == (2, 13, 22, 24, 27):
        return diagnose_confirmation(request, request_path)
    runtime = runtime_flag(FEATURE, 'length_diagnostic_authorized', 'Pose length diagnostic')
    for episode in request.episodes:
        authorize_episode(runtime, PlacementRequest(episode), 'length diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    paths = [PROTOCOL, request_path, FEATURE/'runtime.json', FEATURE/'placements_train.json',
             FEATURE.parent/'mesh_extents/assignments.json', ROOT/'scripts/track3/static_pose_baseline.py',
             ROOT/'outputs/pose_insurance/scale/finished.json',
             *[ROOT/f'outputs/pose_insurance/scale/stereo_{ep:06d}.json' for ep in request.episodes],
             *[p for ep in request.episodes for p in label_paths(ep)]]
    base = run_fields(ROOT, 'pose_insurance', 'length_diagnostic', asdict(request),
                      inputs=paths, parents=[D1], provenance='public_label_diagnostic')
    placements = read_json(FEATURE/'placements_train.json')['episodes']
    assignments = read_json(FEATURE.parent/'mesh_extents/assignments.json')['episodes']
    selected = read_json('outputs/pose_insurance/scale/finished.json')['selected']
    destination = OUTPUT if publish else f'{OUTPUT}/reruns/{new_run_id()}'
    directory = output_path(ROOT, destination)
    directory.mkdir(parents=True, exist_ok=True)
    existing = [directory/name for name in ('run.json', 'summary.json', 'summary.md', 'lengths.png') if (directory/name).exists()]
    if existing:
        backup = output_path(ROOT, f'.runtime/tmp/pose-length/backups/{new_run_id()}')
        backup.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, backup/path.name)
    started = time.monotonic()
    report_directory = directory / "reports" / new_run_id()
    trial_base = dict(base, inputs=base["inputs"].copy())
    with recorded(report_directory, base, hash_artifacts=False):
        rows, resumed = [], 0
        for episode in request.episodes:
            if time.monotonic() - started >= request.max_seconds:
                raise RuntimeError('Time budget reached; resume the same request')
            row, path, reused = episode_trial(episode, trial_base, placements, assignments, selected, directory, directory/'ledger.jsonl')
            base['inputs'].append(input_file(ROOT, ROOT/path))
            rows.append(row)
            resumed += int(reused)
            print(json.dumps(dict(episode=episode, resumed=reused, lengths_cm=row['lengths_cm'])), flush=True)
        report = dict(**summarize(rows), episodes=rows, resumed=resumed, executed=len(rows)-resumed,
                      s1_max_position_difference_m=max(r['s1_max_position_difference_m'] for r in rows),
                      notes=NOTES, reproduce=REPRODUCE)
        write_json(report_directory/'summary.json', report)
        (report_directory/'summary.md').write_text(markdown(report))
        figure(rows, report_directory/'lengths.png')
        base['artifacts'] = {name: input_file(ROOT, report_directory/name)['sha256'] for name in ('summary.json', 'summary.md', 'lengths.png')}
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


def confirmation_row(episode: int, entry: dict, state: dict, record: dict) -> dict:
    """Measure the frozen RGB object unions with the same readers and pure length functions."""
    from types import SimpleNamespace
    request = SimpleNamespace(episode_index=episode, source_split='public')
    measure = producer(entry['measure_run'], 'mesh_extents.measure', request)
    depth = producer(entry['depth_run'], 'depth.generate', request)
    for parent in (measure, depth):
        admit(parent, record)
    with np.load(artifact(measure, 'labels_000000.npz', record)) as arrays:
        labels = arrays['label'].copy()
    arrays = frame_arrays(depth, record)
    with np.load(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])) as grid:
        k = grid['input_intrinsics_pixels'].copy()
    points = camera_points(arrays['depth_m'].astype(float), k)
    normal = np.asarray(measure[1]['frames'][0]['plane']['normal'])
    masks = [None if entry['merged'] else np.isin(labels, np.array(entry['objects'][name])+1) for name in entry['names']]
    clusters = [None if m is None else points[m] for m in masks]
    scale, stereo_clusters, counts = None, [None, None], [0, 0]
    if 'scale' in state:
        parent = producer(state['scale'], 'pose_insurance.length_confirmation_scale')
        admit(parent, record)
        scale = read_json(artifact(parent, 'scale.json', record))
        stereo = load_stereo(episode, scale, depth, record, request)
        if stereo['rgb_depth_m'].shape != labels.shape or not np.allclose(stereo['rgb_intrinsics_pixels'], k, atol=1e-5):
            raise ValueError('Confirmation stereo and label grids differ')
        valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
        stereo_points = camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels'])
        counts = [0 if m is None else int((m & valid).sum()) for m in masks]
        stereo_clusters = [stereo_points[m & valid] if m is not None and n >= 300 else None for m,n in zip(masks,counts)]
    names, poses = baseline_module().load_reference(episode)
    poses = poses[0, 0, [names.index(n) for n in entry['names']]]
    refs = reference_offsets(poses, scan_centres(entry['names'], record))
    moge = offset(footprint_centres(clusters, normal))
    scaled = moge * scale['s'] if moge is not None and scale and scale['valid'] else None
    vectors = dict(refs, V_inc=None, V_moge=moge, V_s=scaled,
                   V_stereo=offset(footprint_centres(stereo_clusters, normal)))
    return dict(episode=episode, names=entry['names'], counted=entry['counted'], merged_cluster=entry['merged'],
                **measurements(vectors, ('V_inc','V_moge','V_s','V_stereo')),
                cluster_pixels=[int(np.isin(labels, np.array(entry['merged_cluster_ids'])+1).sum())]*2
                               if entry['merged'] else [int(m.sum()) for m in masks],
                stereo_valid_pixels=counts, scale=None if scale is None else scale['s'],
                scale_valid=None if scale is None else scale['valid'],
                missing_reason='No independent clusters; no split or fallback authorized' if entry['merged'] else
                               'Stereo needs at least 300 valid pixels per object' if min(counts)<300 else None,
                vectors_m={k: None if v is None else v.tolist() for k,v in vectors.items()})


def confirmation_markdown(report: dict) -> str:
    """Render the fixed counted set, missing measurements and all three checks."""
    def fmt(value):
        return 'missing' if value is None else f'{value:.3f}'
    lines = ['# Pose length confirmation — Amendment 1', '', report['verdict'], '',
             f"Counted episodes: {report['counted_episodes']}; measured offsets: {report['measured_episodes']}.",
             'Mandatory episode 13 stays counted even though its clusters merge. Optional merged episode 2 is excluded.', '',
             '| Episode | Counted | Merged | M_o cm | M_c cm | V_moge cm | V_s cm | V_stereo cm | Cluster pixels 0/1 | Stereo valid 0/1 | s |',
             '|---|---|---|---:|---:|---:|---:|---:|---|---|---:|']
    for row in report['episodes']:
        values = [fmt(row['lengths_cm'][k]) for k in ('M_o','M_c','V_moge','V_s','V_stereo')]
        lines.append(f"| {row['episode']} | {row['counted']} | {row['merged_cluster']} | " + ' | '.join(values) +
                     f" | {row['cluster_pixels']} | {row['stereo_valid_pixels']} | {fmt(row['scale'])} |")
    lines += ['', '| Construction | Available / counted | Counted MAE vs M_o cm | Available-subset MAE vs M_o cm | Available-subset MAE vs M_c cm |',
              '|---|---:|---:|---:|---:|']
    for name,r in report['means'].items():
        lines.append(f"| {name} | {r['count']}/{r['expected_count']} | {fmt(r['counted_mae_vs_M_o_cm'])} | {fmt(r['available_subset_mae_vs_M_o_cm'])} | {fmt(r['available_subset_mae_vs_M_c_cm'])} |")
    lines += ['', '| Amendment 1 check | Pass |', '|---|---|']
    lines += [f'| {key} | {value} |' for key,value in report['checks'].items()]
    lines += ['', 'The counted-set MAEs and gain are missing when a mandatory measurement is missing; available-subset means do not satisfy the registered test.',
              '', 'V_inc and incumbent V_hull are undefined here: no S1 scene exists. V_s is the raw MoGe centroid distance times the exact existing episode-scale estimator; invalid scale is never replaced.',
              'Every measured stereo cluster needs >=300 valid pixels. Identities/unions were frozen from frame-0 RGB overlays before any confirmation mocap was loaded.',
              'M_c uses exact scan AABB centres transformed by mocap poses; all scalar lengths use the same horizontal definitions as the train diagnostic.',
              'There is no incumbent body anchor for a kit-aligned ray-error split; no camera-to-mocap yaw calibration is inferred.',
              '', '![Confirmation lengths and scalar errors](lengths.png)', '', 'Reproduce:', '', '```bash',
              report['reproduce'], '```', '']
    return '\n'.join(lines)


def confirmation_figure(rows: list, path) -> None:
    """Plot validation horizontal lengths and scalar errors without filling missing values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    x = np.arange(len(rows))
    for index, name in enumerate(('M_o','M_c','V_moge','V_s','V_stereo')):
        values = [r['lengths_cm'][name] if r['lengths_cm'][name] is not None else np.nan for r in rows]
        axes[0].bar(x+(index-2)*.15, values, .15, label=name)
    for index,name in enumerate(('V_moge','V_s','V_stereo')):
        values = [abs(r['lengths_cm'][name]-r['lengths_cm']['M_o']) if r['lengths_cm'][name] is not None else np.nan for r in rows]
        axes[1].bar(x+(index-1)*.2, values, .2, label=name, color=f'C{index+2}')
    for ax,title in zip(axes, ('Frame-0 horizontal lengths; absent video bars = merged cluster', 'Absolute horizontal length error vs mocap origins')):
        ax.set(title=title, ylabel='cm', xlabel='Public validation episode', xticks=x,
               xticklabels=[str(r['episode'])+(' (excluded)' if not r['counted'] else '') for r in rows])
        ax.legend(ncol=5);ax.grid(axis='y',alpha=.2)
    fig.savefig(path,dpi=160);plt.close(fig)


def diagnose_confirmation(request: LengthRequest, request_path: str):
    """Record the fixed confirmation after RGB-only assignments and video preparation are complete."""
    from ._length import summarize_confirmation
    from ..stereo_depth._common import StereoDepthRequest
    runtime = runtime_flag(FEATURE, 'length_diagnostic_authorized', 'Length confirmation')
    for episode in request.episodes:
        authorize_episode(runtime, StereoDepthRequest(episode_index=episode), 'length confirmation')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.dont_write_bytecode=True
    heavy_execution(dict(runtime,python=runtime['diagnostic_python']))
    assignments=read_json(request.assignments)
    if sorted(map(int,assignments['episodes'])) != list(request.episodes):
        raise ValueError('Confirmation assignments do not match the fixed roster')
    state_path='.runtime/tmp/pose-length/confirmation_preparation.json'
    state=read_json(state_path)
    record=run_fields(ROOT,'pose_insurance','length_confirmation',asdict(request),
                      inputs=[PROTOCOL,request_path,request.assignments,state_path,FEATURE/'runtime.json',
                              ROOT/'scripts/track3/static_pose_baseline.py', *[p for ep in request.episodes for p in label_paths(ep)]],
                      provenance='public_label_diagnostic')
    directory=output_path(ROOT,f'outputs/pose_insurance/length-confirm/{new_run_id()}')
    with recorded(directory,record):
        rows=[]
        for ep in request.episodes:
            entry=assignments['episodes'][str(ep)]
            if entry['counted'] != (ep in (13,22,24) or not entry['merged']):
                raise ValueError('Amendment 1 counted-set rule changed')
            row=confirmation_row(ep,entry,state[str(ep)],record)
            rows.append(row)
            print(json.dumps(dict(episode=ep,counted=row['counted'],lengths_cm=row['lengths_cm'])),flush=True)
        report=dict(**summarize_confirmation(rows),episodes=rows,
                    reproduce=REPRODUCE.replace('length_request.json','length_confirm_request.json'))
        write_json(directory/'summary.json',report)
        (directory/'summary.md').write_text(confirmation_markdown(report))
        confirmation_figure(rows,directory/'lengths.png')
    print(report['verdict'],flush=True)
    return directory/'run.json'
