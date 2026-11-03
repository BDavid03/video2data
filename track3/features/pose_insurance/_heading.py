"""Pure frame-zero heading oracles and signed body-up priors."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._placement import relative_transform


def oracle(start: np.ndarray, reference: np.ndarray, variant: str) -> np.ndarray:
    """Construct O1/O2/O3 in the predicted world, retaining the anchor position."""
    result = start.copy()
    if variant == 'T1':
        return result
    if variant not in ('O1', 'O2', 'O3'):
        raise ValueError('Unknown heading oracle')
    _, predicted_b = relative_transform(start)
    reference_c, reference_b = relative_transform(reference)
    if np.linalg.norm(predicted_b[:2]) > 1e-12:
        yaw = np.arctan2(predicted_b[1], predicted_b[0]) - np.arctan2(reference_b[1], reference_b[0])
        rotations = Rotation.from_rotvec([0., 0., yaw]) * Rotation.from_quat(start[:, 3:])
        result[:, 3:] = rotations.as_quat()
    anchor = Rotation.from_quat(result[0, 3:])
    if variant in ('O2', 'O3'):
        result[1, 3:] = (anchor * reference_c).as_quat()
    if variant == 'O3':
        result[1, :3] = result[0, :3] + anchor.apply(reference_b)
    return result


def body_ups(poses: np.ndarray) -> np.ndarray:
    """Express world positive z in each object's body frame."""
    return Rotation.from_quat(poses[:, 3:]).as_matrix()[:, 2, :]


def fit_priors(occurrences: dict, consistency: dict) -> dict:
    """Fit normalised signed means only for mutually consistent up vectors."""
    result = {}
    for name, rows in sorted(occurrences.items()):
        vectors = np.asarray([row['up'] for row in rows], float).reshape(-1, 3)
        mean = vectors.mean(axis=0) if len(rows) else np.zeros(3)
        norm = float(np.linalg.norm(mean))
        angle = float(np.degrees(np.arccos(np.clip(vectors @ vectors.T, -1, 1))).max()) if len(rows) else None
        accepted = bool(len(rows) and angle <= consistency['max_pairwise_angle_deg']
                        and norm >= consistency['min_mean_norm'])
        result[name] = dict(occurrences=len(rows), episodes=[row['episode'] for row in rows],
                            up_vectors=vectors.tolist(), max_pairwise_angle_deg=angle, mean_norm=norm,
                            prior=accepted, up=(mean / norm).tolist() if accepted else None)
    return result


def up_axis_rotation(up: np.ndarray, yaw: float) -> Rotation:
    """Take body-up to world z by the shortest arc, then apply the box yaw."""
    tilt, _ = Rotation.align_vectors([[0., 0., 1.]], [up])
    return Rotation.from_rotvec([0., 0., yaw]) * tilt


def apply_priors(start: np.ndarray, names: list, priors: dict) -> np.ndarray:
    """Re-express object rotations using priors while retaining all T1 positions."""
    result = start.copy()
    for slot, name in enumerate(names):
        prior = priors[name]
        if prior['prior']:
            yaw = 2 * np.arctan2(start[slot, 5], start[slot, 6])
            result[slot, 3:] = up_axis_rotation(prior['up'], yaw).as_quat()
    return result


def cup_oracle(start: np.ndarray, reference: np.ndarray, names: list,
               priors: dict | None = None) -> np.ndarray:
    """Apply the cup-covered oracle after optional priors on uncovered objects."""
    covered = [name in ('blue_cup', 'beige_cup') for name in names]
    result = start.copy()
    if priors is not None:
        fitted = apply_priors(start, names, priors)
        result[np.logical_not(covered)] = fitted[np.logical_not(covered)]
    if covered[0]:
        return oracle(result, reference, 'O2' if covered[1] else 'O1')
    if covered[1]:
        reference_c, _ = relative_transform(reference)
        result[1, 3:] = (Rotation.from_quat(result[0, 3:]) * reference_c).as_quat()
    return result


def selected_registration(start: np.ndarray, registered: np.ndarray) -> np.ndarray:
    """Map registered b direction and C into T1 while retaining every position."""
    relative, offset = relative_transform(registered)
    delta = start[1, :3] - start[0, :3]
    anchor, _ = Rotation.align_vectors([delta / np.linalg.norm(delta)],
                                       [offset / np.linalg.norm(offset)])
    result = start.copy()
    result[:, 3:] = Rotation.concatenate([anchor, anchor * relative]).as_quat()
    actual_c, actual_b = relative_transform(result)
    if (np.linalg.norm(actual_b / np.linalg.norm(actual_b) - offset / np.linalg.norm(offset)) > 1e-12
            or (actual_c.inv() * relative).magnitude() > 1e-12):
        raise ValueError('Selected-registration mapping changed its relative pose')
    return result


def frame0_checks(start: np.ndarray, result: np.ndarray, reference: np.ndarray) -> dict:
    """Measure invariant residuals and reference-relative errors at the first frame."""
    old_c, old_b = relative_transform(start)
    new_c, new_b = relative_transform(result)
    ref_c, ref_b = relative_transform(reference)
    horizontal = np.linalg.norm(old_b[:2]) > 1e-12
    direction = None
    if np.linalg.norm(new_b[:2]) > 1e-12 and np.linalg.norm(ref_b[:2]) > 1e-12:
        direction = float(abs(np.arctan2(np.sin(np.arctan2(new_b[1], new_b[0]) - np.arctan2(ref_b[1], ref_b[0])),
                                        np.cos(np.arctan2(new_b[1], new_b[0]) - np.arctan2(ref_b[1], ref_b[0])))))
    return dict(positions_max_change_m=float(np.max(np.abs(result[:, :3] - start[:, :3]))),
                anchor_position_change_m=float(np.linalg.norm(result[0, :3] - start[0, :3])),
                horizontal_direction_defined=bool(horizontal), horizontal_direction_error_rad=direction,
                relative_rotation_change_rad=float((old_c.inv() * new_c).magnitude()),
                relative_rotation_error_rad=float((ref_c.inv() * new_c).magnitude()),
                offset_error_m=float(np.linalg.norm(new_b - ref_b)),
                relative_offset_change_m=float(np.linalg.norm(new_b - old_b)),
                body_ups=body_ups(result).tolist())


def analyze(results: dict, queue: dict, reproduction: dict, metric_names: list) -> dict:
    """Apply T1 reproduction, Stage A and Amendment 1's fixed U1 gates."""
    episodes, variants = queue['episodes'], queue['variants']
    descriptive, gates = queue['descriptive_variants'], queue['gates']
    means = {v: {key: float(np.mean([results[v][ep]['metrics'][key] for ep in episodes]))
                  for key in metric_names} for v in variants}
    t1, loo = means['T1'], means['U1-LOO']
    changes = {key: loo[key] - t1[key] for key in metric_names}
    worst_loss = max(results['T1'][ep]['metrics']['AUC'] - results['U1-LOO'][ep]['metrics']['AUC'] for ep in episodes)
    checks = dict(auc_gain=changes['AUC'] >= gates['auc_gain_min'], rpe_not_worse=changes['RPE_cm'] <= 0,
                  mppe_not_worse=changes['MPPE_cm'] <= 0, sp_sr_not_lower=changes['SP_SR'] >= 0,
                  mp_sr_not_lower=changes['MP_SR'] >= 0, worst_episode_auc_loss=worst_loss <= gates['worst_episode_auc_loss_max'])
    stage_a_gain = means['O2']['AUC'] - t1['AUC']
    coverage = {ep: [name for name in results['T1'][ep]['names'] if name in ('blue_cup', 'beige_cup')] for ep in episodes}
    return dict(means=means, ranking=sorted(variants, key=lambda v: means[v]['AUC'], reverse=True),
                descriptive_variants=descriptive, cup_coverage=coverage,
                cup_coverage_episodes=[ep for ep, names in coverage.items() if names],
                cup_auc_gains={v: means[v]['AUC'] - t1['AUC'] for v in descriptive if v.startswith('O2-cups')},
                reproduction=reproduction,
                stage_a=dict(auc_gain=stage_a_gain, passed=stage_a_gain >= gates['auc_gain_min']),
                u1_gate=dict(changes=changes, worst_auc_loss=worst_loss, checks=checks,
                             passed=all(checks.values())),
                h2s_descriptive=selected_effects(results, queue, means, metric_names))


def selected_effects(results: dict, queue: dict, means: dict, metric_names: list) -> dict:
    """Describe selected-only HR2 effects without applying any adoption rule."""
    effects = {}
    for key in queue['h2s']['registrations']:
        ep = int(key)
        delta = {k: results['H2S-desc'][ep]['metrics'][k] - results['T1'][ep]['metrics'][k]
                 for k in metric_names}
        effects[ep] = dict(episode_changes=delta,
                           eight_episode_mean_changes={k: v / len(queue['episodes']) for k, v in delta.items()})
    return dict(label=queue['h2s']['label'], hr2_rule='failed', candidate_evaluation=False,
                changed_episodes=list(effects), effects=effects,
                mean_changes={k: means['H2S-desc'][k] - means['T1'][k] for k in metric_names},
                identity_caveat='Episode 23 wooden-piece identity is documented as arbitrary; no swaps performed.')
