"""Score sealed G/H arms, audit carried decisions, and compare five fixed methods."""

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import contact_probe as h, position_score as previous_eval, kit_score
from ._kit_score import first_run

OUTPUT = h.OUTPUT
VARIANTS = ('static', 'C', 'position', 'G', 'H')


def carried_truth(episode, names):
    """Evaluate the predeclared translation proxy on the full original mocap grid."""
    import pyarrow.parquet as pq
    path = h.h.snapshot() / f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet'
    rows = pq.read_table(path, columns=['frame_index', 'observation.objects']).to_pylist()
    if [r['frame_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Carried truth requires the original consecutive frame grid')
    result = {}
    for name in names:
        positions = np.asarray([next(o for o in r['observation.objects'] if o['name'] == name)['pose'][:3] for r in rows])
        radius = np.linalg.norm(positions - positions[0], axis=1)
        trigger = first_run(radius > h.probe.SETTINGS['radius_m'], h.probe.SETTINGS['consecutive_frames'])
        result[name] = dict(carried=trigger is not None, trigger_frame=trigger, maximum_radius_m=float(radius.max()))
    return result


def confusion(objects):
    """Count agreement with the fixed mocap translation proxy."""
    counts = {key: 0 for key in ('TP', 'FP', 'TN', 'FN')}
    for row in objects:
        key = ('TP' if row['mocap_carried'] else 'FP') if row['video_carried'] else ('FN' if row['mocap_carried'] else 'TN')
        counts[key] += 1
    return dict(**counts, objects=len(objects), accuracy=(counts['TP'] + counts['TN']) / len(objects))


def evaluate():
    """Require every arm seal before loading labels or invoking the kit."""
    h.h.verify_seal(OUTPUT / 'outputs-seal.json')
    queue = h.read(OUTPUT / 'queue.json')
    for clip in queue['clips']:
        directory = OUTPUT / f"ep{clip['item']['episode']}-{clip['item']['object']}"
        h.hybrid.check(directory)
        for arm in ('G', 'H'):
            h.hybrid.check(directory / arm)
    baseline = h.previous.OUTPUT
    old_finished = h.read(baseline / 'kit/finished.json')
    if old_finished['sources'] != kit_score.hashes():
        raise ValueError('Kit/adapter changed since the sealed baseline scores')
    request = OUTPUT / 'kit-request.json'
    record = OUTPUT / 'evaluation-request.json'
    source_digest = sha256_file(Path(__file__))
    if record.exists():
        if h.read(record)['source_sha256'] != source_digest:
            raise ValueError('Frozen contact evaluator changed')
    else:
        write_json(record, dict(utc=now_utc(), before_label_access=True, source_sha256=source_digest,
            output_seal_sha256=sha256_file(OUTPUT / 'outputs-seal.json'),
            baseline_results_sha256=sha256_file(baseline / 'kit/results.json'),
            baseline_medians_sha256=sha256_file(baseline / 'post-onset.json'),
            baseline_evaluator_sha256=sha256_file(Path(previous_eval.__file__))))
    kit_score.OUTPUT = OUTPUT / 'kit'; kit_score.OUTPUT.mkdir(exist_ok=True)
    protocol = kit_score.OUTPUT / 'protocol.md'
    if not protocol.exists():
        protocol.write_bytes((OUTPUT / 'protocol-H.md').read_bytes())
    kit_score.freeze(request); kit_score.run(request)
    baseline_rows = h.read(baseline / 'kit/results.json')['rows']
    new_rows = h.read(OUTPUT / 'kit/results.json')['rows']
    rows = baseline_rows + new_rows
    old_objects = h.read(baseline / 'post-onset.json')['objects']
    objects = {(r['episode'], r['object']): deepcopy(r) for r in old_objects}
    differences = []; compatibility = []
    for case in h.read(request)['cases']:
        episode = case['episode']; truth = carried_truth(episode, [i['name'] for i in case['objects']])
        old_solution = pd.read_parquet(baseline / f'kit/ep{episode}/static/local-solution.parquet')
        new_solution = pd.read_parquet(OUTPUT / f'kit/ep{episode}/G/local-solution.parquet')
        if not old_solution.equals(new_solution):
            raise ValueError('Baseline/new arm reference payloads or frame rows differ')
        compatibility.append(dict(episode=episode, reference_rows_and_geometry_exact=True))
        for item in case['objects']:
            row = objects[episode, item['name']]
            decision = h.read(OUTPUT / f"ep{episode}-{item['name']}/decision.json")
            row.update(mocap_carried=truth[item['name']]['carried'], mocap_carried_proxy=truth[item['name']],
                video_carried=decision['carried'], video_trigger_frame=decision['trigger_frame'],
                gate_correct=decision['carried'] == truth[item['name']]['carried'],
                H_attached_frames=decision['attached_frames'], H_events=decision['events'],
                palm_supported_frames_per_slot=decision['palm_supported_frames_per_slot'],
                missing_selected_palm_frames=decision['missing_selected_palm_frames'])
        for arm in ('G', 'H'):
            directory = OUTPUT / f'kit/ep{episode}' / arm
            achieved, reference, _, names, frames, replay, difference, auc = previous_eval.registered_arrays(directory)
            differences.append(difference)
            errors = np.linalg.norm(achieved[:, 0, :, :3] - reference[:, 0, :, :3], axis=-1)
            np.savez_compressed(directory / 'registered-centre-errors.npz', object_names=np.asarray(names, str),
                frame_index=frames, position_error_m=errors, achieved_registered=achieved, reference_registered=reference)
            write_json(directory / 'array-replay.json', dict(metrics=replay, maximum_difference=difference, per_object_auc=auc))
            for slot, name in enumerate(names):
                row = objects[episode, str(name)]; onset = row['motion_onset_frame']
                post = np.zeros(len(frames), bool) if onset is None else frames >= onset
                row['registered_median_cm'][arm] = None if not post.any() else float(np.median(errors[post, slot]) * 100)
                row['AUC'][arm] = auc[str(name)]
    objects = sorted(objects.values(), key=lambda r: (r['episode'], r['object']))
    lookup = {(r['episode'], r['variant']): r['metrics'] for r in rows}
    episodes = sorted({r['episode'] for r in rows})
    carried = [r for r in objects if r['mocap_carried']]
    outcomes = {}
    for arm in ('G', 'H'):
        deltas = {str(ep): lookup[ep, arm]['AUC'] - lookup[ep, 'static']['AUC'] for ep in episodes}
        outcomes[arm] = dict(AUC_deltas=deltas, wins=sum(v > 0 for v in deltas.values()),
            mean_AUC_delta=float(np.mean(list(deltas.values()))),
            excluding_ep11_wins=sum(v > 0 for ep, v in deltas.items() if ep != '11'),
            excluding_ep11_mean_AUC_delta=float(np.mean([v for ep, v in deltas.items() if ep != '11'])),
            tuning_mean_AUC_delta=float(np.mean([deltas['39'], deltas['42']])),
            carried_object_mean_post_onset_median_cm=float(np.mean([r['registered_median_cm'][arm] for r in carried])))
    result = dict(utc=now_utc(), objects=objects, rows=rows, outcomes=outcomes,
        gate_confusion_all=confusion(objects), gate_confusion_excluding_ep11=confusion([r for r in objects if r['episode'] != 11]),
        H_ever_attached_objects=sum(r['H_attached_frames'] > 0 for r in objects),
        video_carried_objects=sum(r['video_carried'] for r in objects),
        H_beats_G_carried_objects=sum(r['registered_median_cm']['H'] < r['registered_median_cm']['G'] for r in carried),
        G_beats_H_carried_objects=sum(r['registered_median_cm']['G'] < r['registered_median_cm']['H'] for r in carried),
        mocap_carried_objects=len(carried), kit_array_replay_maximum_difference=max(differences),
        reused_baseline_cases=30, new_kit_cases=20, reference_compatibility=compatibility, development_only=True)
    write_json(OUTPUT / 'results.json', result)
    publish(result); plot(result)
    print(json.dumps(dict(outcomes=outcomes, gate=result['gate_confusion_all'], replay_max_difference=max(differences))), flush=True)


def publish(result):
    """Write every episode metric and carried-object median beside all three baselines."""
    lookup = {(r['episode'], r['variant']): r for r in result['rows']}
    episodes = sorted({r['episode'] for r in result['rows']})
    lines = ['# Frozen motion-gate and hand-attachment arms', '',
        'Train/tuning development on all ten public episodes; no scored episode. '
        'All36 new trajectories and18 carried decisions were sealed before new label access. '
        'Unchanged kit code, public reference-geometry proxy. Static/C/rev6 rows reuse the30 sealed baseline cases; '
        'all20 G/H cases are new. Reference rows and geometry are identical in every episode. '
        'Episode11 has identical cup masks and remains identity ambiguous. No class-specific C shortcut.', '',
        '| Episode | Frames | Method | AUC | MPPE cm | SP-SR | MP-SR |',
        '| ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for episode in episodes:
        for variant in VARIANTS:
            row = lookup[episode, variant]; m = row['metrics']
            lines.append(f"| {episode} | {row['frames']} | {variant} | {m['AUC']:.6f} | {m['MPPE_cm']:.6f} | {m['SP_SR']:.0f} | {m['MP_SR']:.0f} |")
    lines += ['', '## Carried decision agreement', '',
        'Mocap truth here is the predeclared translation proxy (>5cm for five frames), not a semantic grasp annotation.', '',
        '| Episode/object | Video carried | Mocap proxy | Correct | H attached frames | Palm supported slot1 / slot2 |',
        '| --- | --- | --- | --- | ---: | --- |']
    for row in result['objects']:
        support = row['palm_supported_frames_per_slot']
        lines.append(f"| {row['episode']}/{row['object']} | {row['video_carried']} | {row['mocap_carried']} | {row['gate_correct']} | {row['H_attached_frames']} | {support[0]} / {support[1]} |")
    lines += ['', f"All objects: {result['gate_confusion_all']}. Excluding11: {result['gate_confusion_excluding_ep11']}.", '',
        '## Post-onset registered centre medians', '',
        'Original >5mm/three-frame onset; every subsequent scored frame. Bold object names satisfy the mocap carried proxy, '
        'including gate misses. N/A means no motion onset. Kit body registration and shared scene alignment are unchanged.', '',
        '| Episode/object | Static cm | C cm | Rev6 cm | G cm | H cm |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in result['objects']:
        name = f"{row['episode']}/{row['object']}"
        if row['mocap_carried']:
            name = f'**{name}**'
        values = [row['registered_median_cm'][v] for v in VARIANTS]
        lines.append('| ' + name + ' | ' + ' | '.join('N/A' if v is None else f'{v:.3f}' for v in values) + ' |')
    lines += ['', '## Outcome', '']
    for arm, values in result['outcomes'].items():
        lines.append(f"{arm} beats static AUC on {values['wins']}/10 episodes, mean change {values['mean_AUC_delta']:+.6f}; "
            f"excluding11: {values['excluding_ep11_wins']}/9, mean {values['excluding_ep11_mean_AUC_delta']:+.6f}; "
            f"tuning39/42 mean change {values['tuning_mean_AUC_delta']:+.6f}. "
            f"Mean of post-onset median errors across mocap-carried objects: {values['carried_object_mean_post_onset_median_cm']:.3f}cm.")
    lines += ['', f"On {result['mocap_carried_objects']} mocap-carried objects, H beats G on {result['H_beats_G_carried_objects']}, "
        f"G beats H on {result['G_beats_H_carried_objects']}; remaining ties are preserved. "
        f"H attaches on {result['H_ever_attached_objects']} of {result['video_carried_objects']} video-selected objects. "
        f"Maximum difference replaying all four kit metrics: {result['kit_array_replay_maximum_difference']:.3g}.", '',
        'This is disclosed development, not an unbiased assessment or a passed original accuracy bar. '
        'Original residual validity, quality thresholds and campaign admission remain unchanged. '
        'Palm means a thick interior-mask proxy; no anatomical landmarks are available. '
        'Continuous handovers without a no-contact gap are outside H. No masks, prior trackers or two_object_* files changed.']
    (OUTPUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def plot(result):
    """Compare the fixed arms with their baselines without choosing an oracle per episode."""
    os.environ['MPLCONFIGDIR'] = str(OUTPUT / 'figure-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episodes = sorted({r['episode'] for r in result['rows']})
    lookup = {(r['episode'], r['variant']): r['metrics']['AUC'] for r in result['rows']}
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(episodes)); width = .15
    for index, variant in enumerate(VARIANTS):
        axis.bar(x + (index - 2) * width, [lookup[ep, variant] for ep in episodes], width, label=variant)
    axis.set_xticks(x, [str(ep) + ('*' if ep == 11 else '') for ep in episodes])
    axis.set(xlabel='Public episode (*11: duplicate masks)', ylabel='AUC (higher is better)', ylim=(0, 1),
        title='Frozen G/H arms: unchanged kit, public-geometry proxy')
    axis.legend(ncol=5); axis.grid(axis='y', alpha=.2)
    figure.savefig(OUTPUT / 'auc-comparison.png', dpi=150); plt.close(figure)


if __name__ == '__main__':
    evaluate()
