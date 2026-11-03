"""Admit frozen public inputs and run the CPU support-attribution diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, artifact, authorize_episode, checked_entry, heavy_execution,
                         new_run_id, producer, read_json, recorded, runtime_flag, save_npz)

FEATURE = Path(__file__).parent
PROTOCOL = 'docs/chain/pose-support-diagnostic-protocol.md'
OUTPUT = 'outputs/pose_insurance/support-diagnostic'
EPISODES = (0, 11, 12, 18, 21, 23, 26, 31, 22, 24, 27)
TRAIN = EPISODES[:8]
DATASET = 'datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public'
REPRODUCE = ('PYTHONPATH=. CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 '
             '.runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.support_diagnostic '
             '--request track3/features/pose_insurance/support_request.json')
CONFIGS = (PROTOCOL, 'track3/features/pose_insurance/runtime.json',
           'track3/features/pose_insurance/placements_train.json',
           'track3/features/mesh_extents/assignments.json',
           'track3/features/pose_insurance/length_confirm_assignments.json',
           'track3/features/pose_insurance/length_confirm_preparation.json',
           'outputs/pose_insurance/scale/finished.json', 'outputs/pose_insurance/scale/summary.json',
           'configs/calibration/ego.json', 'scripts/track3/static_pose_baseline.py')


@dataclass(frozen=True)
class SupportRequest:
    episodes: tuple
    source_split: str
    max_seconds: int

    def __post_init__(self):
        if tuple(self.episodes) != EPISODES or self.source_split != 'public' or self.max_seconds != 600:
            raise PermissionError('Support diagnostic requires the exact public roster and 600-second budget')


def serializable(value):
    """Convert arrays and explicit infinite errors into strict JSON values."""
    import numpy as np
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('Unexpected NaN outside saved image maps')
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def read_result(path):
    """Restore explicit infinite-error values from a recorded result."""
    return json.loads(path.read_text(), object_hook=lambda row: {
        key: float(value) if isinstance(value, str) and value in ('Infinity', '-Infinity') else value
        for key, value in row.items()})


def protocol_inputs() -> dict:
    """Resolve only the eleven exact producer paths printed in the protocol table."""
    rows = {}
    for line in project_path(ROOT, PROTOCOL).read_text().splitlines():
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) != 7 or not cells[1].isdigit():
            continue
        episode = int(cells[1])
        roots = ('mesh_extents', 'depth', 'stereo_depth', 'stereo_depth')
        rows[episode] = {name: f'outputs/{root}/public/episode_{episode:06d}/{run}/run.json'
                         for name, root, run in zip(('measure', 'depth', 'stereo', 'rectify'), roots, cells[2:6])}
    if tuple(rows) != EPISODES:
        raise ValueError('Protocol table does not contain the exact frozen roster')
    return rows


def admit_parent(path, kind, episode, record, labelled=False):
    """Admit a producer and its public episode identity before reading its artifacts."""
    from .length_diagnostic import admit
    allowed = ('public_label_diagnostic',) if labelled else ('video_only',)
    parent = producer(path, kind, allowed=allowed)
    row = parent[1]
    identity = row if 'episode_index' in row else row['request']
    split = identity.get('split', identity.get('source_split'))
    if identity['episode_index'] != episode or split != 'public':
        raise ValueError(f'{kind} belongs to a different public episode')
    admit(parent, record)
    return parent


def episode_parents(episode, pinned, context, record):
    """Resolve the frozen depth, stereo, scale and S1 scene chains."""
    parents = {name: admit_parent(pinned[name], f'{slice_name}.{operation}', episode, record)
               for name, slice_name, operation in (('measure', 'mesh_extents', 'measure'),
                   ('depth', 'depth', 'generate'), ('stereo', 'stereo_depth', 'generate'),
                   ('rectify', 'stereo_depth', 'rectify'))}
    if episode in TRAIN:
        scale_path = f'outputs/pose_insurance/scale/stereo_{episode:06d}.json'
        record['inputs'].append(input_file(ROOT, scale_path))
        scale = read_json(scale_path)
        selected = [row for row in context['finished']['selected']
                    if row.get('candidate') == 'S1' and row['episode'] == episode]
        if len(selected) != 1:
            raise ValueError('S1 selector is not unique for the train episode')
        parent = admit_parent(selected[0]['run'], 'pose_insurance.scale_diagnostic', episode, record, True)
        metrics = read_json(artifact(parent, 'metrics.json', record))
        scene = metrics['scene']
        if metrics['episode'] != episode or metrics['candidate'] != 'S1' or scene != context['summary']['episodes']['S1'][str(episode)]['scene']:
            raise ValueError('Admitted S1 metrics do not match the frozen summary')
    else:
        parent = admit_parent(context['preparation'][str(episode)]['scale'],
                              'pose_insurance.length_confirmation_scale', episode, record)
        scale = read_json(artifact(parent, 'scale.json', record))
        scene = None
    if scale['episode'] != episode or scale['depth_run'] != pinned['depth']:
        raise ValueError('Recorded scale does not use the pinned depth episode')
    for path in scale['anchor_runs']:
        row = read_json(path)
        if any(frame['frame_index'] == 0 for frame in row['frames']):
            anchor = admit_parent(path, 'stereo_depth.anchor', episode, record)
            break
    else:
        raise ValueError('No recorded frame-zero anchor')
    expected = dict(depth_run=pinned['depth'], stereo_run=pinned['stereo'])
    if any(anchor[1]['request'][key] != value for key, value in expected.items()):
        raise ValueError('Frame-zero anchor does not match the frozen producers')
    for key, expected_path in (('depth_run', pinned['depth']), ('rectification_run', pinned['rectify'])):
        if parents['stereo'][1]['request'][key] != expected_path:
            raise ValueError('Stereo generation does not match the protocol chain')
    if parents['measure'][1]['request']['depth_run'] != pinned['depth']:
        raise ValueError('Measurement used a different MoGe run')
    if parents['stereo'][1]['grid'] != parents['depth'][1]['preprocessing_artifact_sha256']:
        raise ValueError('Stereo and MoGe preprocessing hashes differ')
    return parents, scale, scene


def load_episode(episode, pinned, context, record):
    """Admit image arrays, masks, calibration and reference-file hashes for one episode."""
    from .length_diagnostic import frame_arrays
    from ._length import cluster_masks, confirmation_masks
    from ._support import check_grid
    from .placement_diagnostic import ROSTER
    import numpy as np
    parents, scale, scene = episode_parents(episode, pinned, context, record)
    measure, depth = parents['measure'], parents['depth']
    with np.load(artifact(measure, 'labels_000000.npz', record), allow_pickle=False) as arrays:
        labels = arrays['label'].copy()
    with np.load(artifact(depth, 'preprocessing.npz', record,
                          depth[1]['preprocessing_artifact_sha256']), allow_pickle=False) as arrays:
        grid = {key: arrays[key].copy() for key in ('input_valid', 'input_intrinsics_pixels')}
    moge, stereo = frame_arrays(depth, record), frame_arrays(parents['stereo'], record)
    if int(stereo['frame_index']) != 0 or int(stereo['pts']) != 0:
        raise ValueError('Stereo arrays are not frame zero at PTS zero')
    clocks = [next(frame for frame in parent[1]['frames'] if frame['frame_index'] == 0)
              for parent in (depth, parents['stereo'])]
    if clocks[0]['video_pts'] != clocks[1]['video_pts'] or clocks[0]['video_pts'] != 0:
        raise ValueError('Depth and stereo frame-zero PTS differ')
    with np.load(artifact(parents['rectify'], 'rectification.npz', record), allow_pickle=False) as arrays:
        for key in ('R1', 'R2', 'P1', 'P2', 'Q', 'A_T_C'):
            arrays[key]
    calibration = f'{DATASET}/meta/camera_calibration.json'
    entry = next(item for item in parents['rectify'][1]['inputs'] if item['path'] == calibration)
    checked_entry(entry, record)
    if entry['sha256'] != context['calibration']['source_sha256']:
        raise ValueError('Calibration differs from the audited policy')
    if episode in TRAIN:
        names = ROSTER[episode]
        masks = cluster_masks(names, context['placements'][str(episode)],
            context['assignments'][f'public:{episode}']['objects'], labels, episode in (0, 26))[1]
        if context['placements'][str(episode)]['measure_run'] != pinned['measure']:
            raise ValueError('Train assignments name a different measurement')
    else:
        entry = context['confirmation'][str(episode)]
        names, masks = entry['names'], confirmation_masks(entry, labels)[1]
        if entry['measure_run'] != pinned['measure'] or entry['depth_run'] != pinned['depth']:
            raise ValueError('Validation assignments name different producers')
    if episode in (0, 26, 31):
        masks = [None, None]
    for name in names:
        path = project_path(ROOT, f'{DATASET}/mesh/{name}/{name}.glb')
        if not path.exists():
            path = path.with_name(f'{name}_visual.glb')
        record['inputs'].append(input_file(ROOT, path))
    record['inputs'].append(input_file(ROOT, f'{DATASET}/data/chunk-000/episode_{episode:06d}.parquet'))
    plane = next(frame for frame in measure[1]['frames'] if frame['frame_index'] == 0)['plane']
    return dict(parents=parents, scale=scale, scene=scene, names=names, masks=masks, labels=labels,
                moge=moge, stereo=stereo, grid=grid, plane=plane,
                intrinsics_check=check_grid(moge, stereo, grid, labels))


def measure_episode(episode, data, record):
    """Compute frozen video measurements, then attach reference-only error columns."""
    from ._support import (assignment_contrast, episode_readings, reference_columns,
                           support_estimates, table_models)
    from ._length import reference_offsets
    from .length_diagnostic import scan_centres
    from .placement_diagnostic import baseline_module
    from scipy.spatial.transform import Rotation
    import numpy as np
    normal = np.asarray(data['plane']['normal'])
    scale = data['scale']['s'] if data['scale']['valid'] else None
    args = data['moge'], data['stereo'], data['grid']['input_valid'], data['grid']['input_intrinsics_pixels']
    result = support_estimates(*args, data['masks'], normal, scale)
    result.update(episode=episode, subset='train' if episode in TRAIN else 'spent_validation',
                  names=data['names'], recorded_scale=scale, intrinsics_check=data['intrinsics_check'],
                  plane=data['plane'], assignment=None)
    result['table'] = table_models(*args, data['labels'], normal, data['plane']['offset_m'],
                                  data['parents']['measure'][1]['request']['table_band_m'])
    if episode == 12:
        if scale is None or result['estimates']['stereo_full'] is None:
            raise ValueError('Episode 12 assignment contrast lacks its frozen inputs')
        result['assignment'] = assignment_contrast(data['scene'], result['moge_full_centres_table_m'][0][:2],
            scale, result['estimates']['stereo_full']['length_horizontal_cm'])
    result['readings'] = episode_readings(result)
    names, poses = baseline_module().load_reference(episode)
    poses = poses[0, 0, [names.index(name) for name in data['names']]]
    centres = scan_centres(data['names'], record)
    refs = reference_offsets(poses, centres)
    anchor = None if data['scene'] is None else Rotation.from_quat(
        np.asarray(data['scene']['quaternions'][0])[[1, 2, 3, 0]]).as_matrix()
    result.update(reference_columns(result['estimates'], refs,
                                    Rotation.from_quat(poses[0, 3:]).as_matrix(), anchor))
    return result


def save_episode(directory, row):
    """Persist image maps separately from the readable numeric episode record."""
    import numpy as np
    table = row['table']
    maps = {name: table.pop(name) for name in ('table_mask', 'ratio_map')}
    maps.update({f'{name}_residual_cm': model.pop('residual_cm')
                 for name, model in table['models'].items()})
    save_npz(directory / 'maps.npz', np, **maps)
    write_json(directory / 'episode.json', serializable(row))


def episode_trial(episode, request, pinned, context, base, root):
    """Resume a fully admitted episode by its input, request and source identity."""
    record = dict(base, inputs=base['inputs'].copy(), kind='pose_insurance.support_episode',
                  split='public', episode_index=episode)
    data = load_episode(episode, pinned, context, record)
    identity = hashlib.sha256(json.dumps(dict(inputs=record['inputs'], sources=record['sources'],
                          request=asdict(request), episode=episode), sort_keys=True).encode()).hexdigest()
    ledger = root / 'ledger.jsonl'
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    prior = next((row for row in reversed(previous) if row['identity'] == identity), None)
    if prior:
        parent = producer(prior['run'], 'pose_insurance.support_episode', allowed=('public_label_diagnostic',))
        row = read_result(artifact(parent, 'episode.json', {'inputs': []}))
        return row, parent, True
    directory = root / 'episodes' / new_run_id()
    record['identity'] = identity
    with recorded(directory, record):
        row = measure_episode(episode, data, record)
        save_episode(directory, row)
    path = (directory / 'run.json').relative_to(ROOT).as_posix()
    with ledger.open('a') as stream:
        stream.write(json.dumps(dict(identity=identity, episode=episode, run=path, status='completed')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return row, (directory / 'run.json', record), False


def format_number(value):
    """Format a nullable centimetre quantity without hiding missingness."""
    return 'missing' if value is None else f'{value:.4f}'


def markdown(report):
    """Present attribution, scale gaps, held-out fits and frozen readings."""
    lines = ['# Identical-support pose diagnostic', '',
        f"This invocation: **{report['executed']} new episodes; {report['resumed']} resumed episodes**.", '',
        'Descriptive only. Stereo is a calibrated estimate, not measured depth truth. No candidate or pack follows.',
        'All gaps and contributions below are signed horizontal lengths in cm. Depth = g_I; g_F = depth + support.', '']
    for title, roster in (('Clean train episodes', (11, 12, 18, 21, 23)),
                          ('Spent public validation', (22, 24, 27))):
        lines += [f'## {title}', '', '| Episode | g_F | g_I / depth | Support | MoGe support | Stereo support | MoGe×s full gap | MoGe×s common gap | MoGe×r_I full gap | MoGe×r_I common gap |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in report['episodes']:
            if row['episode'] not in roster:
                continue
            gaps = row['gaps']
            values = [gaps['native'][key] if gaps.get('native') else None
                      for key in ('g_F_cm', 'g_I_cm', 'support_cm', 'moge_support_cm', 'stereo_support_cm')]
            values += [gaps[name][key] if gaps.get(name) else None
                       for name in ('recorded', 'ratio') for key in ('g_F_cm', 'g_I_cm')]
            lines.append(f"| {row['episode']} | " + ' | '.join(map(format_number, values)) + ' |')
        lines.append('')
    lines += ['## Missing independent-object measurements', '']
    lines += [f"- Episode {row['episode']}: {row['missing_reason']}." for row in report['episodes'] if row['missing_reason']]
    lines += ['', '## Episode 12 assignment contrast', '',
              'Native MoGe full support and the recorded s are held fixed. Only default slot-0 xy is replaced.', '',
              '```json', json.dumps(serializable(report['assignment_episode_12']), indent=2), '```', '',
              '## Table models: spatially held-out depth MAE (cm)', '',
              '| Episode | Subset | Scale | Affine depth | Affine inverse | Plane MoGe AbsRel | Plane stereo AbsRel |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for row in report['episodes']:
        models = row['table']['models']
        values = [models[name]['mae_cm'] if name in models else None for name in ('scale', 'affine_depth', 'affine_inverse')]
        values += [models[name]['absolute_relative'] if name in models else None for name in ('plane_moge', 'plane_stereo')]
        lines.append(f"| {row['episode']} | {row['subset']} | " + ' | '.join(map(format_number, values)) + ' |')
    lines += ['', '## Frozen descriptive readings', '', '| Reading | Count / denominator | Required | Qualified |', '|---|---:|---:|---|']
    for name, reading in report['readings'].items():
        count = reading.get('count', int(reading['qualified']))
        lines.append(f"| {name} | {count}/{reading['denominator']} | {reading['required']} | {reading['qualified']} |")
    lines += ['', '## Spent validation readings (descriptive; no confirmation verdict)', '',
              '| Reading | Episode 22 | Episode 24 | Episode 27 | Count / denominator |', '|---|---|---|---|---:|']
    for name, reading in report['spent_validation_readings'].items():
        flags = ' | '.join(str(reading['episodes'].get(str(episode), 'missing')) for episode in (22, 24, 27))
        lines.append(f"| {name} | {flags} | {reading['count']}/3 |")
    lines += ['', '## Support counts', '', '| Episode | Mask pixels 0/1 | Common pixels 0/1 | MoGe-full pixels 0/1 | Stereo-full pixels 0/1 |', '|---:|---|---|---|---|']
    for row in report['episodes']:
        counts = None if row['support'] is None else row['support']['counts']
        lines.append(f"| {row['episode']} | " + ' | '.join(str(counts[name]) if counts else 'missing'
                    for name in ('mask', 'common', 'moge_full', 'stereo_full')) + ' |')
    lines += ['', '## Limits and artifacts', '',
        '- Masks and table bands are MoGe-conditioned. Spent validation is reported separately and is not fresh confirmation.',
        '- Per-pixel differences are along their own rays. Cross-ray centroid shifts do not establish a separate lateral warp.',
        '- Reference vector errors are conditional on the guessed S1 body frame; validation vector-error columns are missing.',
        '- Figure ratio colours saturate outside 0.5–1.5 and residual colours outside ±5 cm; numeric errors remain unclipped.',
        '- Missing values retain their original denominators. Infinite predictions are encoded as "Infinity" in strict JSON.',
        '- summary.json contains continuous aggregates, all vectors/errors, ratios, normal sensitivity, fold coefficients and eligibility.',
        '- Episode records and maps are immutable; ledger.jsonl is append-only. reports/ preserves each invocation.', '',
        '![Support attribution, coverage and table maps](support.png)', '', 'Reproduce:', '', '```bash', REPRODUCE, '```', '']
    return '\n'.join(lines)


def figure(rows, parents, path, record):
    """Render attribution, coverage and table residual maps in one labelled figure."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 3 + 1.8 * len(rows)), layout='constrained')
    grid = fig.add_gridspec(len(rows) + 1, 5)
    ax = fig.add_subplot(grid[0, :2])
    clean = [row for row in rows if row['episode'] in (11, 12, 18, 21, 23)]
    x = np.arange(len(clean))
    for shift, key, label in ((-.25, 'g_F_cm', 'Whole-support gap'), (0, 'depth_cm', 'Identical-pixel depth'), (.25, 'support_cm', 'Net support')):
        ax.bar(x + shift, [row['gaps']['native'][key] if row['gaps'].get('native') else np.nan for row in clean], width=.25, label=label)
    ax.set(xticks=x, xticklabels=[row['episode'] for row in clean], ylabel='Signed gap (cm)', title='Clean train: native MoGe minus stereo')
    ax.axhline(0, color='black', linewidth=.6)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[0, 2:])
    supported = [row for row in rows if row['support'] is not None]
    x = np.arange(len(supported))
    for slot, shift in ((0, -.18), (1, .18)):
        ax.bar(x + shift, [row['support']['fractions']['common'][slot] for row in supported], .35, label=f'Slot {slot}')
    ax.set(xticks=x, xticklabels=[row['episode'] for row in supported], ylim=(0, 1), ylabel='Common / mask pixels', title='Identical support; 22/24/27 are spent validation')
    ax.legend(fontsize=8)
    columns = [('ratio_map', 'Stereo / MoGe depth', .5, 1.5, 'viridis'),
               ('scale_residual_cm', 'Scale residual (cm)', -5, 5, 'coolwarm'),
               ('affine_depth_residual_cm', 'Affine depth residual (cm)', -5, 5, 'coolwarm'),
               ('affine_inverse_residual_cm', 'Affine inverse residual (cm)', -5, 5, 'coolwarm'),
               ('table_mask', 'Retained table pixels', 0, 1, 'gray')]
    for index, (row, parent) in enumerate(zip(rows, parents), 1):
        with np.load(artifact(parent, 'maps.npz', record), allow_pickle=False) as arrays:
            for col, (key, title, low, high, cmap) in enumerate(columns):
                ax = fig.add_subplot(grid[index, col])
                if key in arrays:
                    artist = ax.imshow(arrays[key], vmin=low, vmax=high, cmap=cmap)
                    if index == 1:
                        fig.colorbar(artist, ax=ax, shrink=.65)
                else:
                    ax.text(.5, .5, 'Missing: frozen fold support', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Episode {row['episode']}: {title}", fontsize=9)
                ax.set_axis_off()
    fig.suptitle('Descriptive support diagnostic; table support is MoGe-conditioned; residual = predicted stereo minus stereo', fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def diagnose(request, request_path):
    """Run the admitted eleven-episode loop and publish immutable invocation reports."""
    runtime = runtime_flag(FEATURE, 'support_diagnostic_authorized', 'Support diagnostic')
    for episode in request.episodes:
        authorize_episode(runtime, SimpleNamespace(source_split='public', episode_index=episode), 'support diagnostic')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    from ._support import summarize_rows
    started = time.monotonic()
    inputs = protocol_inputs()
    context = dict(placements=read_json(CONFIGS[2])['episodes'], assignments=read_json(CONFIGS[3])['episodes'],
                   confirmation=read_json(CONFIGS[4])['episodes'], preparation=read_json(CONFIGS[5]),
                   finished=read_json(CONFIGS[6]), summary=read_json(CONFIGS[7]), calibration=read_json(CONFIGS[8]))
    base = run_fields(ROOT, 'pose_insurance', 'support_diagnostic', asdict(request),
                      inputs=(*CONFIGS, request_path), provenance='public_label_diagnostic')
    root = output_path(ROOT, OUTPUT)
    root.mkdir(parents=True, exist_ok=True)
    report_directory = root / 'reports' / new_run_id()
    report_record = dict(base, inputs=base['inputs'].copy())
    with recorded(report_directory, report_record):
        rows, parents, resumed = [], [], 0
        for episode in request.episodes:
            if time.monotonic() - started >= request.max_seconds:
                raise RuntimeError('600-second budget reached; resume the unchanged request')
            row, parent, reused = episode_trial(episode, request, inputs[episode], context, base, root)
            rows.append(row)
            parents.append(parent)
            resumed += int(reused)
            report_record['inputs'].append(input_file(ROOT, parent[0]))
            artifact(parent, 'episode.json', report_record)
            print(json.dumps(dict(episode=episode, resumed=reused, missing=row['missing_reason'])), flush=True)
        report = dict(episodes=rows, executed=len(rows) - resumed, resumed=resumed,
                      reproduce=REPRODUCE, **summarize_rows(rows))
        write_json(report_directory / 'summary.json', serializable(report))
        (report_directory / 'summary.md').write_text(markdown(report))
        figure(rows, parents, report_directory / 'support.png', report_record)
        report_record.update(executed=report['executed'], resumed=resumed, episode_count=len(rows))
    for name in ('run.json', 'summary.json', 'summary.md', 'support.png'):
        shutil.copy2(report_directory / name, root / name)
    print(json.dumps(dict(executed=report['executed'], resumed=resumed, summary=f'{OUTPUT}/summary.md')), flush=True)
    return root / 'run.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    diagnose(SupportRequest(**read_json(args.request)), args.request)


if __name__ == '__main__':
    main()
