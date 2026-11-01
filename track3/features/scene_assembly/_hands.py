"""Scene-owned completion of MANO21 skeletons without Cartesian bone collapse."""

# Wrist followed by thumb, index, middle, ring and pinky chains.
PARENTS = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)


def contract():
    return {"method": "parent_frame_spherical_bones", "joint_order": "MANO21",
            "parents": list(PARENTS), "wrist_translation": "linear",
            "joint_rotations": "shortest-path SO(3) interpolation",
            "bone_directions": "spherical interpolation in the corresponding endpoint parent frames",
            "bone_lengths": "linear between endpoint lengths; equal lengths remain constant",
            "antipodal_rule": "deterministic perpendicular to the first direction using its least-aligned coordinate axis",
            "scope": "Filled reference geometry only; no new observations, MANO solve or collision-free guarantee"}


def _direction(a, b, fraction, np):
    cosine = float(np.clip(np.dot(a, b), -1, 1))
    tangent = b - cosine * a
    sine = float(np.linalg.norm(tangent))
    if sine > 1e-10:
        tangent /= sine
        angle = np.arctan2(sine, cosine)
        result = np.cos(fraction * angle) * a + np.sin(fraction * angle) * tangent
    elif cosine >= 0:
        result = (1 - fraction) * a + fraction * b
    else:
        axis = np.eye(3)[int(np.argmin(np.abs(a)))]
        tangent = axis - np.dot(axis, a) * a
        tangent /= np.linalg.norm(tangent)
        result = np.cos(fraction * np.pi) * a + np.sin(fraction * np.pi) * tangent
    return result / np.linalg.norm(result)


def blend(a, b, rotations_a, rotations_b, fraction, np):
    """Interpolate a complete hand along its tree; never average world bones."""
    from scipy.spatial.transform import Rotation

    if not 0 <= fraction <= 1:
        raise ValueError("Hand interpolation fraction must be in [0, 1]")
    if fraction == 0:
        return a.copy(), rotations_a.copy()
    if fraction == 1:
        return b.copy(), rotations_b.copy()
    start = Rotation.from_matrix(rotations_a)
    rotations = (start * Rotation.from_rotvec((start.inv() * Rotation.from_matrix(rotations_b)).as_rotvec() * fraction)).as_matrix()
    joints = np.empty((21, 3), dtype=np.float64)
    joints[0] = (1 - fraction) * a[0] + fraction * b[0]
    for index, parent in enumerate(PARENTS[1:], 1):
        # Row vectors transform from world into the endpoint parent's frame.
        left = (a[index] - a[parent]) @ rotations_a[parent]
        right = (b[index] - b[parent]) @ rotations_b[parent]
        left_length, right_length = float(np.linalg.norm(left)), float(np.linalg.norm(right))
        if not np.isfinite([left_length, right_length]).all() or min(left_length, right_length) <= 1e-8:
            raise ValueError("Cannot complete a hand with a zero-length or nonfinite endpoint bone")
        direction = _direction(left / left_length, right / right_length, fraction, np)
        length = (1 - fraction) * left_length + fraction * right_length
        joints[index] = joints[parent] + (length * direction) @ rotations[parent].T
    return joints, rotations
