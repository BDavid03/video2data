"""Pure frozen panel constructions, frame-zero limits and clause arithmetic."""
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from ._heading_h3 import METRICS, errors, table_world
from ._placement import relative_transform, reexpress_rotations
from ._position_oracle import EPISODES
from ._scene import table_basis

ARMS = ('T1', 'P', 'P-free', 'TW4', 'AN4', 'TW5', 'AN5',
        'P-keep', 'TW4-keep', 'AN4-keep')
NEW_PAIRS = (0, 26, 31)


def admitted(row: dict) -> bool:
    """Apply the panel's mask, scan and three-subsample stability admission."""
    stability = row.get('stability_deg') or []
    return bool(row.get('rotation') is not None and row.get('stereo_valid_points', 0) > 0
                and len(stability) == 3 and np.isfinite(stability).all()
                and max(stability) <= 10)


def positions(start: np.ndarray, episode: int, arm: str, stage_a: dict) -> np.ndarray:
    """Apply amended full-XYZ displacement or the original descriptive contact rule."""
    result = start.copy()
    if episode in NEW_PAIRS and arm != 'T1' and stage_a['admitted']:
        if arm.endswith('-keep'):
            result[:, :3] = stage_a['placement']['positions_m']
        else:
            result[1, :3] = result[0, :3] + np.asarray(stage_a['d_table_m'])
    return result


def construction(start: np.ndarray, episode: int, arm: str, stage_a: dict,
                 registrations: list, normal: np.ndarray, inertia_diagonal: np.ndarray) -> dict:
    """Enumerate selected/rival rotations, preserving positions and pair fallbacks."""
    baseline = positions(start, episode, arm, stage_a)
    rotation_arm = arm.startswith(('TW', 'AN'))
    targeted = episode in (*NEW_PAIRS, 23) or (episode == 18 and arm in ('TW5', 'AN5'))
    slots = [0, 1] if arm.startswith('TW') else [0]
    input_admitted = bool(episode not in NEW_PAIRS or stage_a['admitted'])
    allowed = bool(rotation_arm and targeted and input_admitted
                   and len(registrations) == 2 and all(admitted(registrations[s]) for s in slots))
    if rotation_arm and targeted and not allowed:
        baseline = start.copy()
    choices = [[False, True] if registrations[s].get('rival_rotation') is not None else [False]
               for s in slots] if allowed else [[]]
    combinations = list(product(*choices)) if allowed else [()]
    variants = []
    for rival_flags in combinations:
        pose = baseline.copy()
        if allowed:
            rotations = np.asarray([np.eye(3) if row.get('rotation') is None else row['rotation'] for row in registrations])
            for slot, rival in zip(slots, rival_flags):
                if rival:
                    rotations[slot] = registrations[slot]['rival_rotation']
            # AN consumes only slot zero; fill the unused rotation so TW's invariant
            # checks remain meaningful even when slot one's admission failed.
            if arm.startswith('AN') and not admitted(registrations[1]):
                rotations[1] = np.eye(3)
            tw, _ = table_world(baseline, rotations, normal)
            pose[slots, 3:] = tw[slots, 3:]
        if any(np.allclose(pose[:, :3], np.asarray(v['poses'])[:, :3], atol=1e-14, rtol=0)
               and np.all((Rotation.from_quat(pose[:, 3:]).inv() *
                           Rotation.from_quat(np.asarray(v['poses'])[:, 3:])).magnitude() <= 1e-12)
               for v in variants):
            continue
        physical = reexpress_rotations(Rotation.from_quat(start[:, 3:]), Rotation.from_quat(pose[:, 3:]),
                                       np.asarray([np.diag(v) for v in inertia_diagonal]))
        variants.append(dict(poses=pose.tolist(), rival_flags=list(rival_flags),
                             counter_rotation=physical['geometry_rotation'].tolist(),
                             body_inertia_kg_m2=physical['inertia'].tolist(),
                             physical_orientation_residual=physical['physical_rotation_residual']))
    return dict(positions_baseline=baseline.tolist(), rotation_admitted=allowed,
                required_slots=slots if rotation_arm and targeted else [],
                targeted=bool(rotation_arm and targeted), variants=variants)


def frame_errors(start: np.ndarray, result: np.ndarray, reference: np.ndarray) -> dict:
    """Reuse H3 errors, explicitly making truly zero offsets angular failures."""
    _, b = relative_transform(result)
    _, ref_b = relative_transform(reference)
    with np.errstate(invalid='ignore', divide='ignore'):
        row = errors(start, result, reference)
    row['direction_defined_3d'] = bool(np.linalg.norm(b) > 0 and np.linalg.norm(ref_b) > 0)
    if not row['direction_defined_3d']:
        row['b_error_deg'] = None
    row['offset_length_error_m'] = float(abs(np.linalg.norm(b) - np.linalg.norm(ref_b)))
    return row


def limits(frame: dict, admission: bool, kind: str) -> dict:
    """Evaluate all fixed pair limits with missing/nonfinite values failing."""
    def finite(key):
        value = frame.get(key)
        return value is not None and bool(np.isfinite(value))
    checks = dict(input_admission=bool(admission),
                  offset_5cm=finite('offset_error_m') and frame['offset_error_m'] <= .05,
                  b_angle_30deg=(frame.get('direction_defined_3d', False) and finite('b_error_deg')
                                 and frame['b_error_deg'] < 30))
    if kind == 'TW':
        checks['C_30deg'] = finite('C_error_deg') and frame['C_error_deg'] < 30
    three = finite('offset_error_m') and frame['offset_error_m'] <= .03
    return dict(checks=checks, passed=all(checks.values()), offset_3cm=three,
                passed_3cm=all(checks.values()) and three)


def number(value) -> float:
    """Propagate missing numerical inputs as nonfinite failures without dropping rows."""
    return np.nan if value is None else float(value)


def mean_metrics(rows: list) -> dict:
    """Keep the immutable equal-episode denominator, including nonfinite failures."""
    return {key: float(np.mean([number(row.get(key)) for row in rows])) for key in METRICS}


def clause(value: float, comparison: str, threshold: float, units: str = '') -> dict:
    """Describe one numeric clause and fail nonfinite operands."""
    finite = value is not None and bool(np.isfinite(value))
    passed = finite and {'>=': lambda: value >= threshold, '<=': lambda: value <= threshold,
                         '>': lambda: value > threshold}[comparison]()
    return dict(value=value, comparison=comparison, threshold=threshold, units=units, passed=bool(passed))


def assess(arm: str, results: dict, baseline: dict, changed: list) -> dict:
    """Report every gate clause against original T1, including descriptive keep arms."""
    rows = results[arm]
    means = mean_metrics([rows[str(ep)]['metrics'] for ep in EPISODES])
    t1 = mean_metrics([baseline[str(ep)]['metrics'] for ep in EPISODES])
    delta = {k: means[k] - t1[k] for k in METRICS}
    kind = 'P' if arm.startswith('P') else 'TW' if arm.startswith('TW') else 'AN'
    checks = {}
    if kind == 'P':
        checks['1_RPE_drop'] = clause(delta['RPE_cm'], '<=', -1., 'cm')
        checks['2_MPPE_drop'] = clause(delta['MPPE_cm'], '<=', -.5, 'cm')
        for key in ('AUC', 'SP_SR', 'MP_SR'):
            checks[f'3_{key}_not_down'] = clause(delta[key], '>=', -1e-12)
        scope, max_mppe = list(EPISODES), .5
    else:
        passed_pairs = [ep for ep in NEW_PAIRS if rows[str(ep)]['limits'][kind]['passed']]
        checks['1_frame0'] = dict(clause(len(passed_pairs), '>=', 2, 'pairs'),
                                  denominator=3, passed_pairs=passed_pairs,
                                  note='2/3 required; 0 failed input admission',
                                  previously_inspected_23=rows['23']['limits'][kind],
                                  admissible_supplementary=dict(passed=sum(ep in passed_pairs for ep in (26, 31)),
                                                               denominator=2))
        checks['2_AUC_gain'] = clause(delta['AUC'], '>=', .02 if kind == 'TW' else .01)
        if kind == 'TW':
            checks['2_SP_SR_positive'] = clause(means['SP_SR'], '>', 0.)
        checks['3_RPE_not_worse'] = clause(delta['RPE_cm'], '<=', 0., 'cm')
        checks['3_MPPE_not_worse'] = clause(delta['MPPE_cm'], '<=', 0., 'cm')
        scope, max_mppe = changed, 2.
    episode_limits = {}
    for ep in scope:
        key = str(ep)
        episode_limits[key] = dict(
            AUC_loss=clause(number(baseline[key]['metrics'].get('AUC')) - number(rows[key]['metrics'].get('AUC')), '<=', .02),
            MPPE_gain=clause(number(rows[key]['metrics'].get('MPPE_cm')) - number(baseline[key]['metrics'].get('MPPE_cm')),
                             '<=', max_mppe, 'cm'))
    checks['4_episode_limits'] = dict(episodes=episode_limits,
        passed=all(c['passed'] for row in episode_limits.values() for c in row.values()))
    if kind == 'TW':
        changes = {str(ep): clause(100 * (number(rows[str(ep)]['frame0'].get('offset_error_m')) -
                                          number(baseline[str(ep)]['frame0'].get('offset_error_m'))), '<=', .5, 'cm')
                   for ep in EPISODES}
        checks['5_original_T1_offset'] = dict(episodes=changes, baseline='immutable original T1',
                                             passed=all(row['passed'] for row in changes.values()))
    if kind != 'P':
        expected_rows, stress_limits, counts = [], {}, {}
        for ep in EPISODES:
            key = str(ep)
            variants = rows[key]['flip_stress'] if ep in changed else [baseline[key]]
            counts[key] = len(variants)
            expected_rows.append(mean_metrics([v['metrics'] for v in variants]))
            stress_limits[key] = [dict(
                AUC_loss=clause(number(baseline[key]['metrics'].get('AUC')) - number(v['metrics'].get('AUC')), '<=', .02),
                MPPE_gain=clause(number(v['metrics'].get('MPPE_cm')) - number(baseline[key]['metrics'].get('MPPE_cm')), '<=', 2., 'cm'))
                for v in variants]
        expected = mean_metrics(expected_rows)
        expected_delta = {k: expected[k] - t1[k] for k in METRICS}
        stress_checks = dict(AUC_gain=clause(expected_delta['AUC'], '>=', .01),
                             RPE_not_worse=clause(expected_delta['RPE_cm'], '<=', 0., 'cm'),
                             MPPE_not_worse=clause(expected_delta['MPPE_cm'], '<=', 0., 'cm'),
                             every_combination=dict(passed=all(c['passed'] for row in stress_limits.values()
                                 for v in row for c in v.values()), episodes=stress_limits))
        checks['6_flip_stress'] = dict(expected_means=expected, changes=expected_delta,
                                      distinct_combinations_per_episode=counts, denominator=8,
                                      checks=stress_checks, passed=all(c['passed'] for c in stress_checks.values()))
    return dict(kind=kind, descriptive=arm.endswith('-keep'), adoption_route=not arm.endswith('-keep'),
                baseline='immutable original T1', denominator=8, means=means, changes=delta,
                checks=checks, passed=all(c['passed'] for c in checks.values()))


def bias_reading(start: np.ndarray, reference: np.ndarray, stage_a: dict,
                 registrations: list, normal: np.ndarray) -> dict:
    """Separate offset length evidence from registration-dependent centroid biases."""
    displaced = start.copy()
    displaced[1, :3] = start[0, :3] + stage_a['d_table_m']
    tw, _ = table_world(displaced, np.asarray([r['rotation'] for r in registrations]), normal)
    basis = table_basis(normal)
    biases = []
    for centroid, registration in zip(stage_a['centroids_camera_m'], registrations):
        translation = registration.get('translation_m')
        bias = None if centroid is None or translation is None else np.asarray(centroid) - translation
        biases.append(dict(slot=registration['slot'], name=registration['name'],
                           camera_m=None if bias is None else bias.tolist(),
                           table_m=None if bias is None else (basis @ bias).tolist(),
                           norm_cm=None if bias is None else float(np.linalg.norm(bias) * 100)))
    t1_error = frame_errors(start, start, reference)
    d_t1_error = frame_errors(displaced, displaced, reference)
    registered_error = frame_errors(displaced, tw, reference)
    return dict(T1=t1_error, d_with_T1_rotation=d_t1_error, d_with_registered_rotation=registered_error,
                stereo_centroid_minus_registered_body_origin=biases,
                bias_caveat='Origins are fitted scan translations, not measured mocap camera extrinsics.',
                position_only_lower_bound_cm=100 * registered_error['offset_length_error_m'],
                position_necessarily_fails_5cm=registered_error['offset_length_error_m'] > .05,
                rotation_C_fails_30deg=registered_error['C_error_deg'] >= 30,
                attribution='Offset length mismatch is position evidence; C is rotation-only. Offset angle mixes centroid displacement direction and anchor rotation, inseparable without independent camera-to-mocap extrinsics.')
