"""Pure fixed-protocol scale estimation, scene transforms and adoption gates."""
import numpy as np
from ._scene import box_properties


def estimate_scale(frames):
    """Apply the protocol's median estimator and four validity conditions."""
    observed = [f for f in frames if f.get('scale_status') == 'observed']
    s = float(np.median([f['scale'] for f in observed])) if observed else None
    support = float(np.median([f['support_share'] for f in observed])) if observed else None
    mad = float(np.median([f['log_ratio_mad'] for f in observed])) if observed else None
    checks = dict(observed_frames=len(observed) >= 3,
                  support=support is not None and support >= .10,
                  log_ratio_mad=mad is not None and mad <= .30,
                  scale=s is not None and .40 <= s <= 1.20)
    return dict(s=s, observed_frames=len(observed), requested_frames=len(frames),
                median_support_share=support, median_log_ratio_mad=mad,
                valid=all(checks.values()), checks=checks)


def scale_geometry(geometry, candidate, s, valid):
    """Scale planar offsets alone or the whole scene about slot zero at table height."""
    result = {key: np.array(value, copy=True) for key, value in geometry.items()}
    if candidate not in ('incumbent', 'S1', 'S2'):
        raise ValueError('Unknown fixed scale candidate')
    if candidate != 'incumbent' and valid:
        origin = result['positions'][0].copy()
        origin[2] = 0.
        if candidate == 'S1':
            result['positions'][1:, :2] = origin[:2] + s * (result['positions'][1:, :2] - origin[:2])
        else:
            result['positions'] = origin + s * (result['positions'] - origin)
            result['dimensions'] *= s
    result['masses_kg'] = np.full(len(result['positions']), .1)
    result['inertia_diagonal_kg_m2'] = np.array([box_properties(d, .1)[0] for d in result['dimensions']])
    return result


def candidate_gates(incumbent, candidate):
    """Compute exactly the three registered equal-episode adoption gates."""
    mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
    rpe_gain = mean(incumbent, 'RPE_cm') - mean(candidate, 'RPE_cm')
    mppe_gain = mean(incumbent, 'MPPE_cm') - mean(candidate, 'MPPE_cm')
    loss = max(a['AUC'] - b['AUC'] for a, b in zip(incumbent, candidate))
    mppe_worse = max(b['MPPE_cm'] - a['MPPE_cm'] for a, b in zip(incumbent, candidate))
    guarded = {key: mean(candidate, key) - mean(incumbent, key) for key in ('AUC', 'SP_SR', 'MP_SR')}
    checks = dict(mean_errors=rpe_gain >= 2. and mppe_gain >= 1.,
                  guarded_means=all(v >= -1e-12 for v in guarded.values()),
                  episode_safety=loss <= .02 and mppe_worse <= .5)
    return dict(checks=checks, passed=all(checks.values()), rpe_gain_cm=rpe_gain,
                mppe_gain_cm=mppe_gain, guarded_changes=guarded,
                worst_auc_loss=loss, worst_mppe_increase_cm=mppe_worse)


def adoption(gates, means, controls_passed, reproduction_passed):
    """Select the registered winner only after all integrity controls pass."""
    if not controls_passed or not reproduction_passed:
        return 'blocked: integrity control failed'
    passing = [c for c in ('S1', 'S2') if gates[c]['passed']]
    if not passing:
        return 'keep D2 v2'
    if len(passing) == 1:
        return passing[0]
    return 'S2' if means['S1']['MPPE_cm'] - means['S2']['MPPE_cm'] >= 1. else 'S1'
