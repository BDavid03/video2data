"""Pure W1.0 arm constructions, invariant residuals and descriptive calculations."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._heading import oracle
from ._placement import relative_transform

EPISODES = (0, 11, 12, 18, 21, 23, 26, 31)
TREATED = (0, 26, 31)
ARMS = ('T1', 'O2', 'O3', 'PO', 'PO-W1', 'PL-W1', 'RC')
METRICS = ('AUC', 'RPE_cm', 'MPPE_cm', 'SP_SR', 'MP_SR')


def construct(start: np.ndarray, reference: np.ndarray, arm: str,
              episode: int, direction: int = 0) -> np.ndarray:
    """Construct a protocol arm in T1's world without modifying its input."""
    if arm in ('T1', 'O2', 'O3'):
        return oracle(start, reference, arm)
    result = start.copy()
    relative, offset = relative_transform(reference)
    anchor = Rotation.from_quat(start[0, 3:])
    if arm == 'RC':
        result[1, 3:] = (anchor * relative).as_quat()
    elif arm == 'PO' or (arm == 'PO-W1' and episode in TREATED):
        result[1, :3] = start[0, :3] + anchor.apply(offset)
    elif arm == 'PL-W1' and episode in TREATED:
        alignment = Rotation.from_quat(reference[0, 3:]) * anchor.inv()
        yaw = Rotation.from_euler('z', direction * 15, degrees=True)
        delta = reference[1, :3] - reference[0, :3]
        result[1, :3] = start[0, :3] + alignment.inv().apply(yaw.apply(delta))
    elif arm not in ('PO-W1', 'PL-W1'):
        raise ValueError('Unknown position-oracle arm')
    return result


def residuals(start: np.ndarray, result: np.ndarray, reference: np.ndarray,
              warped: np.ndarray, arm: str, episode: int) -> dict:
    """Measure the protocol's applicable frame-zero invariants in metres or radians."""
    checks = {}
    if arm == 'PO':
        checks.update(anchor_bitwise=result[0].tobytes() == start[0].tobytes(),
                      rotations_bitwise=result[:, 3:].tobytes() == start[:, 3:].tobytes(),
                      anchor_max_change_m=float(np.max(np.abs(result[0] - start[0]))),
                      rotations_max_change=float(np.max(np.abs(result[:, 3:] - start[:, 3:]))),
                      aligned_slot1_error_m=float(np.linalg.norm(warped[1, :3] - reference[1, :3])))
    if arm == 'PL-W1' and episode in TREATED:
        actual = warped[1, :3] - warped[0, :3]
        expected = reference[1, :3] - reference[0, :3]
        checks.update(height_error_m=float(abs(actual[2] - expected[2])),
                      horizontal_radius_error_m=float(abs(np.linalg.norm(actual[:2]) - np.linalg.norm(expected[:2]))))
    if arm == 'RC':
        actual, _ = relative_transform(result)
        expected, _ = relative_transform(reference)
        checks.update(positions_bitwise=result[:, :3].tobytes() == start[:, :3].tobytes(),
                      anchor_bitwise=result[0].tobytes() == start[0].tobytes(),
                      positions_max_change_m=float(np.max(np.abs(result[:, :3] - start[:, :3]))),
                      anchor_max_change=float(np.max(np.abs(result[0] - start[0]))),
                      relative_rotation_error_rad=float((expected.inv() * actual).magnitude()))
    return checks


def mean_metrics(rows: list[dict]) -> dict:
    """Average each metric with equal row weights."""
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def calculations(results: dict) -> dict:
    """Calculate eight-episode means, both marginal orders and fixed route scenarios."""
    means = {arm: mean_metrics([results[arm][str(ep)]['metrics'] for ep in EPISODES]) for arm in ARMS}
    pairs = [('position first: PO - T1', 'PO', 'T1'), ('then O3 - PO', 'O3', 'PO'),
             ('rotation first: RC - T1', 'RC', 'T1'), ('then O3 - RC', 'O3', 'RC'),
             ('O2 - T1 (direction and rotation)', 'O2', 'T1')]
    marginals = {label: {key: means[a][key] - means[b][key] for key in METRICS} for label, a, b in pairs}
    routes = [('W1, length with unsolved yaw', 'PL-W1', TREATED, 4, 3),
              ('W1, direction solved', 'PO-W1', TREATED, 4, 3),
              ('Rotation (W2a or H3)', 'O2', tuple(ep for ep in EPISODES if ep != 18), 8, 4.5),
              ('Rotation and offset (W2b)', 'O3', tuple(ep for ep in EPISODES if ep != 18), 8, 4)]
    scenarios = []
    for label, arm, episodes, reach, hours in routes:
        gains = mean_metrics([{key: results[arm][str(ep)]['metrics'][key] - results['T1'][str(ep)]['metrics'][key]
                               for key in METRICS} for ep in episodes])
        scenarios.append(dict(route=label, arm=arm, treated_episodes=list(episodes), r=reach, h=hours,
                              hours_unverified=arm == 'O3', g=gains,
                              formula={key: value * reach / (20 * hours) for key, value in gains.items()}))
    return dict(means=means, marginals=marginals, scenarios=scenarios)
