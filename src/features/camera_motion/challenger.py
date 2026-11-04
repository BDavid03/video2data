"""Normalize explicitly declared offline challenger poses without fitting a gauge."""


def normalize_optical_pose(matrix, direction, np):
    """Return world_T_camA; preserve the declared world frame and translation units.

    Only OpenCV optical coordinates are admitted. In particular this function
    does not align an arbitrary model world to the incumbent table world.
    """
    pose = np.asarray(matrix, dtype=np.float64)
    if direction not in ("camera_to_world", "world_to_camera"):
        raise ValueError("Unsupported extrinsic direction")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Extrinsics must be finite 4x4 matrices")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
        raise ValueError("Extrinsics must be rigid SE(3), without scale or reflection")
    return np.linalg.inv(pose) if direction == "world_to_camera" else pose.copy()
