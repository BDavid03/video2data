"""Pure motion measurements for the frozen M2 Amendment 1 adoption rule."""

import numpy as np
from scipy.spatial.transform import Rotation
from src.features.dynamic_feasibility._metrics import motion_score, transport


def tracking_metrics(achieved: np.ndarray, reference: np.ndarray) -> list[dict]:
    """Measure each object's raw tracking, held-still baseline and initial errors."""
    results = []
    for slot in range(achieved.shape[1]):
        a, r = achieved[:, slot], reference[:, slot]
        theta = np.degrees((Rotation.from_quat(a[:, 3:]) * Rotation.from_quat(r[:, 3:]).inv()).magnitude())
        theta_s = np.degrees((Rotation.from_quat(np.repeat(r[:1, 3:], len(r), axis=0)) *
                              Rotation.from_quat(r[:, 3:]).inv()).magnitude())
        position = np.linalg.norm(a[:, :3] - r[:, :3], axis=-1)
        # T and Q are undefined for an exactly stationary positional reference.
        # Such a reference supplies no evidence for adopting a moving policy.
        displacement = r[:, :3] - r[:1, :3]
        increments = r[15:, :3] - r[:-15, :3]
        moving = len(r) > 15 and np.sum(displacement ** 2) > 0 and np.sum(increments ** 2) > 0
        results.append(dict(slot=slot,
                            T=transport(a[:, :3], r[:, :3]) if moving else None,
                            Q=motion_score(a[:, :3], r[:, :3]) if moving else None,
                            E_m=float(position.mean()),
                            E_S_m=float(np.linalg.norm(displacement, axis=-1).mean()),
                            Theta_deg=float(theta.mean()), Theta_S_deg=float(theta_s.mean()),
                            first_position_m=float(position[0]), first_orientation_deg=float(theta[0])))
    return results


def translation_errors(source: np.ndarray, moved: np.ndarray, translation: np.ndarray) -> dict:
    """Measure whether two full pose sequences differ by the recorded pure translation."""
    expected = source.copy()
    expected[..., :3] += translation
    position = np.linalg.norm(moved[..., :3] - expected[..., :3], axis=-1)
    angle = np.degrees((Rotation.from_quat(moved.reshape(-1, 7)[:, 3:]) *
                        Rotation.from_quat(expected.reshape(-1, 7)[:, 3:]).inv()).magnitude())
    return dict(maximum_translation_error_m=float(position.max()),
                maximum_orientation_error_deg=float(angle.max()),
                first_translation_error_m=float(position.reshape(len(position), -1)[0].max()))


def pose_matrices(poses: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Map task-world xyzw poses back into the frozen M1 world."""
    matrices = np.repeat(np.eye(4)[None], len(poses), axis=0)
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    matrices[:, :3, 3] = poses[:, :3] - translation
    return matrices


def reprojection_summary(ious: np.ndarray, usable: np.ndarray) -> dict:
    """Summarize finite per-frame achieved silhouette overlap on usable video frames."""
    use = np.asarray(usable, bool) & np.isfinite(ious)
    values = np.asarray(ious)[use]
    return dict(median_iou=float(np.median(values)) if len(values) else None,
                p10_iou=float(np.percentile(values, 10)) if len(values) else None,
                usable_fraction=float(use.mean()), frames=len(ious), usable_frames=int(use.sum()))
