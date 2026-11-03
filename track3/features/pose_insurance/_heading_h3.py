"""Pure H3 table-world construction, rival combinations and protocol gates."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from ._heading import frame0_checks, oracle
from ._placement import relative_transform
from ._scene import table_basis

METRICS = ('AUC', 'RPE_cm', 'MPPE_cm', 'SP_SR', 'MP_SR')


def basis_checks(normal: np.ndarray) -> dict:
    """Measure orthonormality, handedness and the signed up mapping."""
    basis = table_basis(normal)
    checks = dict(orthogonality_max=float(np.max(np.abs(basis @ basis.T - np.eye(3)))),
                  determinant=float(np.linalg.det(basis)),
                  up_max_error=float(np.max(np.abs(basis @ normal - [0., 0., 1.]))))
    checks['passed'] = (checks['orthogonality_max'] <= 1e-12 and
                        abs(checks['determinant'] - 1) <= 1e-12 and checks['up_max_error'] <= 1e-12)
    return checks


def table_world(start: np.ndarray, rotations: np.ndarray, normal: np.ndarray) -> tuple:
    """Place scan-to-camera rotations in T1's table frame without moving positions."""
    checks = basis_checks(normal)
    if not checks['passed']:
        raise ValueError('T1 plane does not define a signed proper orthonormal basis')
    result = start.copy()
    result[:, 3:] = Rotation.from_matrix(table_basis(normal) @ rotations).as_quat()
    actual_c, _ = relative_transform(result)
    checks.update(positions_unchanged=bool(np.array_equal(result[:, :3], start[:, :3])),
                  relative_rotation_max_error=float(np.max(np.abs(
                      actual_c.as_matrix() - rotations[0].T @ rotations[1]))))
    if not checks['positions_unchanged'] or checks['relative_rotation_max_error'] > 1e-12:
        raise ValueError('TW changed positions or the registered relative rotation')
    return result, checks


def descriptive_b(start: np.ndarray, registered: np.ndarray) -> np.ndarray:
    """Apply the disclosed yaw-only anchor construction to registered b and C."""
    return oracle(start, registered, 'O2')


def has_rival(row: dict) -> bool:
    """Select the recorded distant rival only within the fixed five-percent margin."""
    return (row.get('rival_index') is not None and row['rival_relative_margin'] <= .05
            and row['rival_separation_deg'] > 30)


def combinations(predictions: list) -> list:
    """Enumerate selected and all nonempty combinations of qualifying recorded rivals."""
    choices = [[row['best_index'], row['rival_index']] if has_rival(row)
               else [row['best_index']] for row in predictions]
    return [dict(variant='H3-sel' if index == 0 else ('H3-flip' if index == 1 else f'H3-flip-{index}'),
                 indices=list(indices)) for index, indices in enumerate(product(*choices))]


def errors(start: np.ndarray, result: np.ndarray, reference: np.ndarray) -> dict:
    """Report full three-dimensional body-offset and relative-rotation errors."""
    checks = frame0_checks(start, result, reference)
    c, b = relative_transform(result)
    ref_c, ref_b = relative_transform(reference)
    checks.update(b_error_deg=float(np.degrees(np.arccos(np.clip(
        b @ ref_b / (np.linalg.norm(b) * np.linalg.norm(ref_b)), -1., 1.)))),
        C_error_deg=float(np.degrees((ref_c.inv() * c).magnitude())),
        b_m=b.tolist(), C=c.as_matrix().tolist())
    return checks


def relative_change(start: np.ndarray, result: np.ndarray) -> dict:
    """Compare the TW relative pose with the recorded D2 v4 scene."""
    old_c, old_b = relative_transform(start)
    new_c, new_b = relative_transform(result)
    return dict(T1=dict(C=old_c.as_matrix().tolist(), b_m=old_b.tolist()),
                TW=dict(C=new_c.as_matrix().tolist(), b_m=new_b.tolist()),
                change=dict(C_deg=float(np.degrees((old_c.inv() * new_c).magnitude())),
                            b_m=(new_b - old_b).tolist(), b_norm_m=float(np.linalg.norm(new_b - old_b))))


def assess(results: dict, baseline: dict, predictions: list) -> dict:
    """Apply H3's fixed G1/G2/G3 arithmetic with equal episode and basin weights."""
    means = {'T1': {key: float(np.mean([r[key] for r in baseline.values()])) for key in METRICS}}
    for name, row in results.items():
        means[name] = {key: float(np.mean([row['metrics'][key] if ep == '18' else values[key]
                                           for ep, values in baseline.items()])) for key in METRICS}
    t1, selected = means['T1'], means['H3-sel']
    delta = {key: selected[key] - t1[key] for key in METRICS}
    worst = max(0., baseline['18']['AUC'] - results['H3-sel']['metrics']['AUC'])
    g1 = dict(auc_gain=delta['AUC'] >= .010, rpe_not_worse=delta['RPE_cm'] <= 0,
              mppe_not_worse=delta['MPPE_cm'] <= 0, sp_sr_not_lower=delta['SP_SR'] >= 0,
              mp_sr_not_lower=delta['MP_SR'] >= 0, worst_episode_loss=worst <= .02)
    variants = [name for name in results if name.startswith('H3-')]
    expected = {key: float(np.mean([means[name][key] for name in variants])) for key in METRICS}
    expected_delta = {key: expected[key] - t1[key] for key in METRICS}
    flip_losses = {name: baseline['18']['AUC'] - results[name]['metrics']['AUC']
                   for name in variants if name != 'H3-sel'}
    g2 = dict(expected_auc_gain=expected_delta['AUC'] >= .010,
              expected_rpe_not_worse=expected_delta['RPE_cm'] <= 0,
              expected_mppe_not_worse=expected_delta['MPPE_cm'] <= 0,
              every_flip_episode_loss=all(value <= .02 for value in flip_losses.values()))
    slots = []
    for row in predictions:
        stability = row.get('stability_deg', [])
        slots.append(dict(slot=row['slot'], mask=row.get('stereo_valid_points', 0) > 0,
                          loadable_scan='rotation' in row, completed_registration='rotation' in row,
                          stable=len(stability) == 3 and bool(np.isfinite(stability).all())
                          and max(stability) <= 10))
    g3 = len(slots) == 2 and all(all(value for key, value in slot.items() if key != 'slot') for slot in slots)
    required = any(has_rival(row) for row in predictions) or not g3
    passed = all(g1.values()) and g3 and (not required or all(g2.values()))
    return dict(means=means, G1=dict(checks=g1, changes=delta, worst_auc_loss=worst, passed=all(g1.values())),
                G2=dict(checks=g2, expected_means=expected, changes=expected_delta,
                        equal_weight_variants=variants, flip_episode_auc_losses=flip_losses,
                        required=required, passed=all(g2.values())),
                G3=dict(slots=slots, passed=g3), gates_passed=passed,
                verdict='H3 passes the required gates; owner approval and physical check still required' if passed
                else 'D2 v4 stands: H3 failed a required gate')
