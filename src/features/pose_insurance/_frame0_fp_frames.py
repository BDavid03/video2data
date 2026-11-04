"""Pure pose conversions for the frame-zero diagnostic."""
import numpy as np
from scipy.spatial.transform import Rotation


def matrix(pose):
    """Convert xyz plus xyzw to a homogeneous transform."""
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    value[:3, 3] = pose[:3]
    return value


def pose(value):
    """Convert a homogeneous transform to xyz plus xyzw."""
    return np.r_[value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_quat()]


def registered_transform(candidate_anchor, rotation):
    """Return candidate_T_registered, matching the kit's body conversion."""
    value = np.eye(4)
    value[:3, :3] = rotation.T
    value[:3, 3] = candidate_anchor
    return value


def correct_pair(start, camera_T_scan, scan_anchors, candidate_T_registered):
    """Reanchor both converted FP poses onto the unchanged candidate A pose."""
    camera_T_candidate = []
    for fp, anchor, conversion in zip(camera_T_scan, scan_anchors, candidate_T_registered):
        scan_T_registered = np.eye(4)
        scan_T_registered[:3, 3] = anchor
        camera_T_candidate.append(fp @ scan_T_registered @ np.linalg.inv(conversion))
    relative = np.linalg.inv(camera_T_candidate[0]) @ camera_T_candidate[1]
    result = start.copy()
    result[1] = pose(matrix(start[0]) @ relative)
    return result, relative


def relative_errors(predicted, reference):
    """Measure relative translation and orientation residuals in the same frame."""
    p = np.linalg.inv(matrix(predicted[0])) @ matrix(predicted[1])
    r = np.linalg.inv(matrix(reference[0])) @ matrix(reference[1])
    return dict(offset_cm=float(100 * np.linalg.norm(p[:3, 3] - r[:3, 3])),
                rotation_deg=float(np.degrees(Rotation.from_matrix(p[:3, :3] @ r[:3, :3].T).magnitude())))
