"""Array operations for the C-initialized depth tracker and hybrid selection."""

import numpy as np
from . import full_c


def c_start(frame, camera_pose, mesh, k, canonical_to_campaign):
    """Use C's exact canonical frame-zero initializer, then the recorded asset map."""
    canonical = full_c.initialize(frame, camera_pose, mesh, k)
    mapping = np.eye(4)
    mapping[:3, :3] = canonical_to_campaign
    return canonical@np.linalg.inv(mapping), canonical


def select_c(c_pose, c_valid, fitted_pose, fitted_valid):
    """Preserve C exactly whenever it is valid."""
    if c_valid:
        return np.array(c_pose, copy=True), True, 'C'
    return np.array(fitted_pose, copy=True), bool(fitted_valid), 'depth'
