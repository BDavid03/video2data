"""Pure S1 scene editing and physical acceptance checks."""
from copy import deepcopy
import numpy as np
from ._scale import scale_geometry
from ._scene import pose_statistics, initial_pose_error


def scaled_scene(scene: list, scale: float, valid: bool) -> list:
    """Apply the registered S1 transform to an existing scene without clearance changes."""
    geometry = dict(positions=np.array([r['position'] for r in scene], float),
                    dimensions=np.array([r['dimensions_m'] for r in scene], float),
                    quaternions=np.array([r['quaternion_wxyz'] for r in scene], float))
    result = deepcopy(scene)
    transformed = scale_geometry(geometry, 'S1', scale, valid)
    for row, position in zip(result, transformed['positions']):
        row['position'] = position.tolist()
    return result


def physical_checks(poses: np.ndarray, intended: np.ndarray) -> dict:
    """Measure first-pose error and drift using the existing geometry primitives."""
    stats = pose_statistics(poses)
    first_cm, first_deg = initial_pose_error(poses[0, 0], intended)
    passed = (stats['finite'] and stats['unit_quaternions'] and first_cm <= 1.
              and first_deg <= 2. and stats['drift_cm'] < 1. and stats['drift_deg'] < 2.)
    return dict(stats, first_pose_cm=first_cm, first_pose_deg=first_deg, physical_passed=bool(passed))
