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


def fit_priors(occurrences: dict) -> dict:
    """Fit normalised signed means only for mutually consistent up vectors."""
    result = {}
    for name, rows in sorted(occurrences.items()):
        vectors = np.asarray([row['up'] for row in rows], float).reshape(-1, 3)
        mean = vectors.mean(axis=0) if len(rows) else np.zeros(3)
        norm = float(np.linalg.norm(mean))
        angle = float(np.degrees(np.arccos(np.clip(vectors @ vectors.T, -1, 1))).max()) if len(rows) else None
        accepted = bool(len(rows) and angle <= 30. and norm >= .8)
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
