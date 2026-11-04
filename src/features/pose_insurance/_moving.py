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
