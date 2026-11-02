"""Pure registered shrink assignments, integrity controls and geometric diagnostics."""
from itertools import permutations
import numpy as np
from scipy.stats import spearmanr

EPISODES = (0, 11, 12, 18, 21, 23, 26, 31)
CHANGED = (11, 12, 18, 21, 23, 31)
CONTROLS = (39, 42)
CANDIDATES = ('I', 'S1', 'G', *(f'P{k}' for k in range(1, 8)), 'C0', 'C25', 'C50', 'C75')
METRICS = ('AUC', 'SP_SR', 'MP_SR', 'RPE_cm', 'MPPE_cm')
TOL = 1e-9


def assignments(stereo: dict) -> dict:
    """Assign the recorded scales in the exact registered cyclic order."""
    scales = np.array([stereo[e]['s'] for e in EPISODES])
    if not all(np.isfinite(stereo[e]['s']) and stereo[e]['valid'] for e in (*EPISODES, *CONTROLS)):
        raise ValueError('The protocol supplies no invalid-scale fallback')
    s_bar = float(scales.mean())
    if abs(s_bar - .856952) > 1e-6:
        raise ValueError('Recomputed global scale disagrees with the registered value')
    result = {c: {} for c in CANDIDATES}
    for e in (*EPISODES, *CONTROLS):
        for c, factor in [('I', 1.), ('S1', stereo[e]['s']), ('G', s_bar),
                          ('C0', 0.), ('C25', .25), ('C50', .5), ('C75', .75)]:
            result[c][e] = float(factor)
        for k in range(1, 8):
            index = (EPISODES.index(e) + k) % 8 if e in EPISODES else k
            result[f'P{k}'][e] = float(scales[index])
    return dict(s_bar=s_bar, scales=scales.tolist(), factors=result)


def comparison(before: dict, after: dict) -> dict:
    """Compare all five finite metrics at the registered absolute tolerance."""
    delta = {k: float(after[k] - before[k]) for k in METRICS}
    return dict(passed=all(np.isfinite(v) and abs(v) <= TOL for v in delta.values()),
                differences=delta)


def means(results: dict, episodes: tuple) -> dict:
    """Average the complete fixed denominator, refusing non-finite metrics."""
    values = np.array([[results[e]['metrics'][k] for k in METRICS] for e in episodes])
    if not np.isfinite(values).all():
        raise ValueError('Missing or non-finite comparator; denominator cannot be reduced')
    return dict(zip(METRICS, map(float, values.mean(axis=0))))


def best_factor(scores: dict, factors: dict) -> str:
    """Resolve MPPE ties within 1e-9 cm in favour of the larger factor."""
    minimum = min(scores[c]['MPPE_cm'] for c in factors)
    return max((c for c in factors if scores[c]['MPPE_cm'] - minimum <= TOL), key=factors.get)


def verdicts(scores: dict, eligible: dict, reproduced: bool, s_bar: float) -> dict:
    """Compute verdicts 1 and 2 without changing any registered margin."""
    def available(names):
        return reproduced and all(eligible.get(c, False) and c in scores for c in names)
    first = dict(label='blocked: missing, non-finite or disqualified comparator / reproduction')
    second = dict(label=first['label'])
    if available(('S1', 'G', *(f'P{k}' for k in range(1, 8)))):
        dm = scores['G']['MPPE_cm'] - scores['S1']['MPPE_cm']
        dr = scores['G']['RPE_cm'] - scores['S1']['RPE_cm']
        wins = sum(scores['S1']['MPPE_cm'] <= scores[f'P{k}']['MPPE_cm'] + TOL for k in range(1, 8))
        if dm >= .25 and dr >= .5 and wins >= 6:
            label = 'Calibration adds information'
        elif abs(dm) < .25:
            label = 'Difference below the registered practical margin'
        elif -dm >= .25:
            label = 'Global shrink is better'
        else:
            label = 'mixed'
        first = dict(label=label, global_minus_S1_mppe_cm=dm, global_minus_S1_rpe_cm=dr,
                     S1_no_worse_than_shifts=wins, shifts_total=7)
    factors = dict(C0=0., C25=.25, C50=.5, C75=.75, G=s_bar, I=1.)
    if available(('S1', *factors)):
        best = best_factor(scores, factors)
        gain = scores['S1']['MPPE_cm'] - scores[best]['MPPE_cm']
        label = ('Further shrinkage helps' if factors[best] <= .5 and gain >= 1.
                 else 'Registered shrinkage criterion not met')
        second = dict(label=label, C_best=best, c_best=factors[best], S1_minus_best_mppe_cm=gain)
    return {'1': first, '2': second}


def oracle_reading(rows: dict, stereo: dict) -> dict:
    """Describe the six grid optima and stereo/true-length rank correlation."""
    factors = [rows[e]['c_star'] for e in CHANGED]
    ratios = [rows[e]['length_ratio'] for e in CHANGED]
    scales = [stereo[e]['s'] for e in CHANGED]
    rho = None if np.ptp(scales) == 0 or np.ptp(ratios) == 0 else float(spearmanr(scales, ratios).statistic)
    return dict(median_c_star=float(np.median(factors)), range_c_star=[min(factors), max(factors)],
                spearman_s_length_ratio=rho,
                episodes={e: dict(c_star=rows[e]['c_star'], s=stereo[e]['s'], length_ratio=rows[e]['length_ratio'])
                          for e in CHANGED})


def oracle_result(grid: np.ndarray, values: list, positions: np.ndarray, reference_start: np.ndarray) -> dict:
    """Select the larger tied grid factor and compute the registered full-distance ratio."""
    if not np.isfinite(values).all():
        raise ValueError('Non-finite oracle grid; descriptive reading blocked')
    optimum = max(i for i, value in enumerate(values) if value - min(values) <= TOL)
    offset = positions[1] - positions[0]
    distance = float(np.linalg.norm(reference_start[1, :3] - reference_start[0, :3]))
    return dict(c_star=float(grid[optimum]), grid=grid.tolist(), MPPE_cm=values,
                length_ratio=float(distance / np.linalg.norm(offset)),
                incumbent_offset_m=offset.tolist(), mocap_distance_m=distance,
                note='Mocap oracle only; never a candidate scene or input to H')


def length_reading(offsets: np.ndarray, distances: np.ndarray, plan: dict) -> tuple:
    """Enumerate every scale assignment using full 3D lengths after planar scaling."""
    xy2 = (offsets[:, :2] ** 2).sum(axis=1)
    z2 = offsets[:, 2] ** 2
    def error(factors):
        return np.abs(np.sqrt(np.asarray(factors) ** 2 * xy2 + z2) - distances) * 100.
    per_episode = {c: error([plan['factors'][c][e] for e in CHANGED]) for c in ('I', 'S1', 'G')}
    all_assignments = np.array(list(permutations(range(8))), dtype=np.int16)
    indices = [EPISODES.index(e) for e in CHANGED]
    scales = np.array(plan['scales'])
    errors = error(scales[all_assignments[:, indices]]).mean(axis=1)
    actual = float(per_episode['S1'].mean())
    rank = int(np.count_nonzero(errors < actual)) + 1
    tied = int(np.count_nonzero(errors == actual))
    gain = float(per_episode['G'].mean() - actual)
    report = dict(means_cm={c: float(v.mean()) for c, v in per_episode.items()},
                  per_episode_cm={c: dict(zip(CHANGED, map(float, v))) for c, v in per_episode.items()},
                  S1_rank=rank, exact_ties=tied, assignments=len(errors),
                  rank_fraction=rank / len(errors), global_minus_S1_cm=gain,
                  label=('S1 carries length information' if gain >= .5 and rank <= .05 * len(errors)
                         else 'Length information not shown'),
                  rank_rule='Competition rank: 1 + strictly lower errors; exact ties share rank')
    return report, dict(assignments=all_assignments, mean_error_cm=errors)


def footprint_clearance(positions: np.ndarray, dimensions: np.ndarray, quaternions: np.ndarray) -> float:
    """Return the greatest projected gap on the rectangles' four unit edge normals."""
    angles = 2 * np.arctan2(quaternions[:, 3], quaternions[:, 0])
    bases = np.array([[[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]] for a in angles])
    axes = bases.reshape(-1, 2)
    radius = sum(np.abs(axes @ basis.T) @ (dims[:2] / 2) for basis, dims in zip(bases, dimensions))
    return float(np.max(np.abs(axes @ (positions[1, :2] - positions[0, :2])) - radius))


def separation_factor(geometry: dict) -> dict:
    """Bisect the registered clearance predicate and retain its feasible upper endpoint."""
    positions = geometry['positions']
    offset = positions[1, :2] - positions[0, :2]
    if not np.any(offset):
        return dict(status='unchanged: zero planar offset', c_sep=None)
    def gap(c):
        points = positions.copy()
        points[1, :2] = points[0, :2] + c * offset
        return footprint_clearance(points, geometry['dimensions'], geometry['quaternions'])
    if gap(1.) < .01:
        return dict(status='fallback: no feasible endpoint', c_sep=None, gap_at_one_m=gap(1.))
    lower, upper = 0., 1.
    while upper - lower > 1e-4:
        middle = (upper + lower) / 2
        if gap(middle) >= .01:
            upper = middle
        else:
            lower = middle
    return dict(status='separated', c_sep=upper, infeasible_lower=lower, width=upper-lower,
                clearance_m=gap(upper), lower_clearance_m=gap(lower))
