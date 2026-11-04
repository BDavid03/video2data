"""Evaluate the sealed position-only batch with the unchanged kit and centre diagnostics."""

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import project_path
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import position_track as h, kit_score
from ._kit_score import first_run
from ._depth_track import m1_errors

OUTPUT = h.OUTPUT
METRIC_NAMES = dict(AUC='add_auc', MPPE_cm='mppe_cm', SP_SR='spider_sr', MP_SR='maniptrans_sr')


@lru_cache(maxsize=32)
def body_registration(episode, slot, encoded_candidate, encoded_geometry):
    """Reuse the kit's deterministic body registration across the three variants."""
    kit = kit_score.scorer('AUC')
    geometry = kit._decode_archive(encoded_geometry, 16_000_000)
    mesh = kit._decode_object_mesh(encoded_candidate)
    reference = geometry[f'v{slot}'], geometry[f'f{slot}'].astype(np.int64)
    anchor = geometry[f'anchor{slot}']
    candidate_anchor, rotation = kit._register_object_body(mesh, reference, anchor, 1_000_003 * episode + 101 * slot)
    return candidate_anchor, rotation, anchor, geometry[f'points{slot}'] - anchor, geometry[f'axis{slot}'], int(geometry[f'order{slot}'])


def registered_arrays(directory):
    """Replay the unchanged kit's array conversions from its exact scored rows."""
    kit = kit_score.scorer('AUC')
    solution = pd.read_parquet(directory / 'local-solution.parquet')
    submission = pd.read_parquet(directory / 'local-submission.parquet')
    encoded_geometry = next(v for v in solution['reference_geometry_b64'] if v)
    merged = solution.merge(submission[['row_id', *kit._POSE_COLUMNS]], on='row_id', validate='one_to_one')
    achieved, reference, slots, names = kit._episode_tensors(merged)
    achieved = kit._normalise_quaternions(achieved, 'submission', ValueError)
    reference = kit._normalise_quaternions(reference, 'solution', ValueError)
    raw_reference = reference.copy(); points = []; axes = []; orders = []
    episode = int(solution['episode_index'].iloc[0])
    for slot in slots:
        encoded = submission.loc[submission['object_slot'] == slot, 'object_mesh_b64']
        encoded_candidate = next(v for v in encoded if v)
        ca, rotation, anchor, vertices, axis, order = body_registration(episode, int(slot), encoded_candidate, encoded_geometry)
        achieved[:, :, slot] = kit._convert_object_body(achieved[:, :, slot], ca, rotation)
        reference[:, :, slot, :3] += np.einsum('...ij,j->...i', kit._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        points.append(vertices); axes.append(axis); orders.append(order)
    achieved = kit._align_registered_scene(achieved, reference, axes)
    replay = {name: float(kit._registered_pose_metric(metric, achieved, reference, points, axes, orders))
              for name, metric in METRIC_NAMES.items()}
    original = h.read(directory / 'result.json')['metrics']
    difference = max(abs(replay[k] - original[k]) for k in replay)
    if difference > 1e-12:
        raise ValueError(f'Registered array replay differs from score(): {difference}')
    object_auc = {str(name): float(kit._add_auc(kit._symmetry_point_errors(
        achieved[:, :, slot], reference[:, :, slot], points[slot], axes[slot], orders[slot])[..., None]))
        for slot, name in enumerate(names)}
    return achieved, reference, raw_reference, names, np.sort(solution['frame_index'].unique()), replay, difference, object_auc


def motion_starts(episode, names):
    """Find mocap onset on the full original frame grid, before kit visibility drops."""
    import pyarrow.parquet as pq
    path = h.h.snapshot() / f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet'
    rows = pq.read_table(path, columns=['frame_index', 'observation.objects']).to_pylist()
    frames = np.asarray([r['frame_index'] for r in rows])
    if np.any(np.diff(frames) != 1):
        raise ValueError('Motion onset requires the full consecutive mocap frame grid')
    starts = {}
    for name in names:
        position = np.asarray([next(o for o in row['observation.objects'] if o['name'] == name)['pose'][:3] for row in rows])
        onset = first_run(np.linalg.norm(position - position[0], axis=1) > .005, 3)
        starts[str(name)] = None if onset is None else int(frames[onset])
    return starts


def evaluate():
    """Open evaluation only after the complete track batch and inputs are sealed."""
    h.h.verify_seal(OUTPUT / 'outputs-seal.json')
    queue = h.read(OUTPUT / 'queue.json')
    for clip in queue['clips']:
        h.hybrid.check(OUTPUT / f"ep{clip['item']['episode']}-{clip['item']['object']}")
    request = OUTPUT / 'kit-request.json'
    score_queue = h.read(request)
    kit_score.OUTPUT = OUTPUT / 'kit'
    kit_score.OUTPUT.mkdir(exist_ok=True)
    protocol = kit_score.OUTPUT / 'protocol.md'
    if not protocol.exists():
        protocol.write_bytes((OUTPUT / 'protocol.md').read_bytes())
    evaluation_request = OUTPUT / 'evaluation-request.json'
    if not evaluation_request.exists():
        write_json(evaluation_request, dict(utc=now_utc(), before_label_access=True,
            output_seal_sha256=sha256_file(OUTPUT / 'outputs-seal.json'),
            evaluator_source_sha256=sha256_file(Path(__file__)), score_request_sha256=sha256_file(request),
            post_onset='first of three frames displaced >5mm from frame zero; all subsequent frames',
            centres='kit registered body centres with shared scene alignment; M1 comparison retained separately'))
    elif h.read(evaluation_request)['evaluator_source_sha256'] != sha256_file(Path(__file__)):
        raise ValueError('Evaluator changed after its labelled measurement was frozen')
    kit_score.freeze(request)
    kit_score.run(request)
    objects = {}; differences = []
    for case in score_queue['cases']:
        episode = case['episode']
        starts = motion_starts(episode, [i['name'] for i in case['objects']])
        for variant in score_queue['variants']:
            directory = kit_score.OUTPUT / f'ep{episode}' / variant
            achieved, reference, raw_truth, names, frames, replay, difference, object_auc = registered_arrays(directory)
            differences.append(difference)
            error = np.linalg.norm(achieved[:, 0, :, :3] - reference[:, 0, :, :3], axis=-1)
            np.savez_compressed(directory / 'registered-centre-errors.npz', frame_index=frames,
                object_names=names, position_error_m=error, achieved_registered=achieved,
                reference_registered=reference)
            write_json(directory / 'array-replay.json', dict(metrics=replay, maximum_difference=difference, per_object_auc=object_auc))
            for slot, name in enumerate(names):
                onset = starts[str(name)]
                post = np.zeros(len(frames), bool) if onset is None else frames >= onset
                key = episode, str(name)
                record = objects.setdefault(key, dict(episode=episode, object=str(name), frames=len(frames),
                    motion_onset_frame=onset, post_onset_frames=int(post.sum()),
                    registered_median_cm={}, M1_median_cm={}, AUC={}))
                record['AUC'][variant] = object_auc[str(name)]
                record['registered_median_cm'][variant] = None if not post.any() else float(np.median(error[post, slot]) * 100)
                item = next(i for i in case['objects'] if i['name'] == name)
                source = project_path(h.ROOT, item['C' if variant == 'static' else variant])
                with np.load(source) as saved:
                    selected = np.searchsorted(saved['frame_index'], frames)
                    position = saved['position'][selected]; quaternion = saved['quaternion_xyzw'][selected]
                    if variant == 'static':
                        position = np.repeat(saved['position'][0:1], len(frames), axis=0)
                        quaternion = np.repeat(saved['quaternion_xyzw'][0:1], len(frames), axis=0)
                m1, _ = m1_errors(position, quaternion, raw_truth[:, 0, slot, :3], raw_truth[:, 0, slot, 3:], np.array([0., 0., 1.]), None)
                record['M1_median_cm'][variant] = None if not post.any() else float(np.median(m1[post]) * 100)
    write_json(OUTPUT / 'post-onset.json', dict(objects=list(objects.values()),
        kit_replay_maximum_difference=max(differences), utc=now_utc()))
    publish(list(objects.values()))


def publish(objects):
    """Publish the complete metric comparison and per-object position evidence."""
    scores = h.read(OUTPUT / 'kit/results.json')['rows']
    lines = ['# Position-only revision 6', '',
        'One fixed recipe; public train/tuning development only. All object outputs were sealed before this evaluation. '
        'Current kit score() entrypoints/flags unchanged; reference geometry is a disclosed public proxy, not the confidential host payload. '
        'All requested episodes now have complete object coverage. Episode 11 has identical cup masks and remains identity ambiguous.', '',
        '| Episode | Frames | Method | AUC | MPPE cm | SP-SR | MP-SR |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in scores:
        m = row['metrics']
        lines.append(f"| {row['episode']} | {row['frames']} | {row['variant']} | {m['AUC']:.6f} | {m['MPPE_cm']:.6f} | {m['SP_SR']:.0f} | {m['MP_SR']:.0f} |")
    lines += ['', '## Median centre error after motion onset', '',
        'Centres use the exact kit registration/body conversion/shared scene alignment. '
        'Onset is displacement >5mm for three successive reference frames; medians include every subsequent frame, irrespective of tracker validity. '
        'N/A means no such motion onset. Independent M1 first-pose-aligned medians are in post-onset.json.', '',
        '| Episode/object | Motion onset | Post-onset frames | Static cm | C cm | Position-only cm |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in objects:
        values = [row['registered_median_cm'][v] for v in ('static', 'C', 'position')]
        formatted = ['N/A' if v is None else f'{v:.3f}' for v in values]
        lines.append(f"| {row['episode']}/{row['object']} | {row['motion_onset_frame']} | {row['post_onset_frames']} | " + ' | '.join(formatted) + ' |')
    lines += ['', '## Label-free measurement and unchanged residual validity', '',
        '| Episode/object | Stereo observed | Fit valid | Fit IoU median (valid frames) | Max speed m/s | Ray offset cm |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in objects:
        directory = OUTPUT / f"ep{row['episode']}-{row['object']}"
        details = h.read(directory / 'object-diagnostics.json')
        gates = h.read(directory / 'gates.json')['gates']
        fmt = lambda v: 'N/A' if v is None else f'{v:.3f}'
        observed = np.mean([r['centre_observed'] for r in details])
        offset = h.read(directory / 'initialization.json')['signed_view_ray_offset_m']
        lines.append(f"| {row['episode']}/{row['object']} | {observed:.1%} | {gates['object_reference_valid_fraction']['value']:.1%} | " +
            f"{fmt(gates['reprojection_iou_median']['value'])} | {fmt(gates['object_speed_max_m_s']['value'])} | {offset*100:.3f} |")
    lines += ['', 'No policy/campaign admission or original accuracy-bar PASS is implied. '
        'The fit IoU here excludes the dilated hand mask; it is not the separate raw-hand campaign reprojection measurement. '
        'Existing tracker sources, masks, meshes, pass bars and two_object_* code were not edited. '
        'The newly needed C baselines use the existing campaign handler unchanged. '
        'Static/C scores for 12/26 are now whole-episode measurements and should not be compared to earlier object-only subsets.']
    (OUTPUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    plot_auc(scores)


def plot_auc(rows):
    """Plot the three methods on the complete public-geometry episode scores."""
    os.environ['MPLCONFIGDIR'] = str(OUTPUT / 'figure-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episodes = sorted({row['episode'] for row in rows})
    figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(episodes)); width = .25
    for index, variant in enumerate(('static', 'C', 'position')):
        values = [next(r['metrics']['AUC'] for r in rows if r['episode'] == ep and r['variant'] == variant) for ep in episodes]
        axis.bar(x + (index - 1) * width, values, width, label=variant)
    axis.set_xticks(x, [str(ep) + ('*' if ep == 11 else '') for ep in episodes])
    axis.set(xlabel='Public episode (*11: duplicate cup masks)', ylabel='AUC (higher is better)', ylim=(0, 1),
        title='Position-only revision 6: unchanged kit, public-geometry proxy')
    axis.legend(); axis.grid(axis='y', alpha=.2)
    figure.savefig(OUTPUT / 'auc-comparison.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    evaluate()
