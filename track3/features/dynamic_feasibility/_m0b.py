"""Pure M0b selection, progress arithmetic and frozen gates."""

import re

import numpy as np

PROGRESS = re.compile(r'M0B t=([\d.]+) updates=(\d+) replay=(\d+) credit=([\d.]+) training progress: ([\d,]+)/([\d,]+)')


def parse_progress(text):
    """Extract timestamped upstream progress and compilation time from a training log."""
    rows = [{'seconds': float(t), 'updates': int(u), 'replay_size': int(r),
             'credit': float(c), 'steps': int(s.replace(',', ''))}
            for t, u, r, c, s, _ in PROGRESS.findall(text)]
    compiled = re.search(r'compiled FlashSAC JAX executables .* in ([\d.]+)s', text)
    return rows, float(compiled[1]) if compiled else None


def throughput(rows, ring_full_seconds):
    """Measure a wall-clock window including compilation and checkpoint pauses."""
    if not rows:
        return {'steady_transitions_per_second': None, 'window': None}
    end = rows[-1]['seconds']
    start = end - 60 if ring_full_seconds is not None else 60.
    mode = 'final_60_seconds' if ring_full_seconds is not None else '60_seconds_to_end_ring_not_full'
    if end <= start or rows[0]['seconds'] > start:
        return {'steady_transitions_per_second': None, 'window': mode, 'window_start_seconds': start,
                'window_end_seconds': end, 'error': 'No complete measurement window'}
    steps = np.interp([start, end], [r['seconds'] for r in rows], [r['steps'] for r in rows])
    return {'steady_transitions_per_second': float((steps[1] - steps[0]) / (end - start)),
            'window': mode, 'window_start_seconds': start, 'window_end_seconds': end,
            'window_start_steps_interpolated': float(steps[0]), 'window_end_steps': int(steps[1])}


def select_recipe(rows, remaining=45188096, worlds=1024, maximum_peak=13824):
    """Apply the protocol's capacity-first rule, fitted times and fixed ratio order."""
    passing = {r['arm']: r for r in rows if r.get('admissible') is True
               and r.get('peak_sampled_device_mib') is not None
               and r['peak_sampled_device_mib'] <= maximum_peak
               and r.get('steady_transitions_per_second', 0) > 0}
    if not passing:
        return {'status': 'stopped', 'reason': 'no_arm_passed', 'predicted_minutes': {}}
    capacity = 1048576 if any(a in passing for a in ('c', 'd')) else 16384
    lo, hi = ('c', 'd') if capacity == 1048576 else ('a', 'b')
    times = {r: worlds / passing[a]['steady_transitions_per_second']
             for r, a in ((.5, lo), (2., hi)) if a in passing}
    fit = None
    if len(times) == 2:
        b = (times[2.] - times[.5]) / 1.5
        a = times[.5] - .5 * b
        fit = {'a_seconds': a, 'b_seconds': b}
        times[1.] = a + b
    predicted = {str(r): 1.25 * remaining / worlds * t / 60 for r, t in sorted(times.items())}
    result = {'capacity': capacity, 'fit': fit, 'predicted_minutes': predicted,
              'passing_arms': sorted(passing), 'planning_margin': 1.25}
    for ratio in (2., 1., .5):
        if str(ratio) in predicted and predicted[str(ratio)] <= 100:
            return {**result, 'status': 'selected', 'ratio': ratio, 'incomplete_fallback': False}
    if .5 in times:
        return {**result, 'status': 'selected', 'ratio': .5, 'incomplete_fallback': True}
    return {**result, 'status': 'stopped', 'reason': 'no_passing_half_ratio_fallback'}


def smoke_checks(before, after, first_steps, maximum_seconds):
    """Apply every frozen smoke check to CPU checkpoint summaries."""
    return {'environment_steps': after['environment_steps'] == 21584896,
            'global_update_step': after['global_update_step'] == 2633,
            'update_credit': after['update_credit'] == .25,
            'curriculum_stage': after['curriculum_stage'] == 5,
            'voc_scale': after['voc_scale'] == .05,
            'initial_progress': first_steps == 21575680,
            'actor_adam_advanced': after['actor_adam_steps'] > before['actor_adam_steps'],
            'critic_adam_advanced': after['critic_adam_steps'] > before['critic_adam_steps'],
            'actor_parameters_advanced': after['actor_parameters_sha256'] != before['actor_parameters_sha256'],
            'critic_parameters_advanced': after['critic_parameters_sha256'] != before['critic_parameters_sha256'],
            'finite_state': after['finite'], 'under_two_gpu_minutes': maximum_seconds < 120}


def thresholds(rows):
    """Set M0b's raw gates using all five fixed M0 baseline arms."""
    baseline = max(rows[name]['T'] for name in ('Z', 'L1-seed0', 'L1-seed1', 'D1p-seed0', 'D1p-seed1'))
    static = rows['S']['mean_position_cm']
    return {'GO-1': {'T': .5, 'Q': .5, 'E': .5 * static},
            'SIGNAL': {'T': max(.2, baseline + .1), 'Q': .1, 'E': static}}


def final_gate(row, checks, controls, gates):
    """Gate only the final policy and preserve all M0 validity requirements."""
    from ._metrics import meets_m0
    if not all(item['passed'] for item in checks.values()):
        return 'INVALID'
    for name in ('GO-1', 'SIGNAL'):
        if meets_m0(row, gates[name]):
            return f'METRIC-NONDISCRIMINATING ({name})' if controls[name] else name
    return 'NULL'
