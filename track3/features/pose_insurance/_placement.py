"""Pure box placement, body conventions and relative-pose diagnostics."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._scene import box_pose, scaled_dimensions, separate_boxes, box_properties


def place_boxes(proportions, measurements, clusters, normal, movable,
                merged=None, k1=None, stacked_slot=1):
    """Compute v1 placements, optionally replacing merged defaults with K1."""
    dimensions, positions, quaternions = [], [], []
    for slot, (shape, measured, points) in enumerate(zip(proportions, measurements, clusters)):
        dims = scaled_dimensions(shape, measured)
        if points is None:
            pos = np.array([slot * .5, 0., dims[2] / 2])
            quat = np.array([1., 0., 0., 0.])
        else:
            pos, quat = box_pose(points, normal, dims)
        if k1 and merged is not None and points is None:
            pos, quat = box_pose(merged, normal, dims)
        dimensions.append(dims)
        positions.append(pos.tolist())
        quaternions.append(quat.tolist())
    dimensions = np.array(dimensions)
    if k1 == "K1a" and merged is not None:
        support = 1 - stacked_slot
        positions[stacked_slot][2] = (positions[support][2] + dimensions[support, 2] / 2
                                     + dimensions[stacked_slot, 2] / 2)
    before = np.array(positions)
    vertically_disjoint = (k1 == "K1a" and merged is not None and
                           abs(before[0, 2] - before[1, 2]) >= dimensions[:, 2].sum() / 2 - 1e-12)
    if vertically_disjoint:
        separated, shifts = before.copy(), np.zeros(len(before))
    else:
        separated, shifts = separate_boxes(positions, dimensions, quaternions, movable)
    return dict(dimensions=dimensions, positions=separated, quaternions=np.array(quaternions),
                before=before, shifts=shifts)


def convention_estimate(rotations):
    """Estimate signed body-up from frame-zero reference rotations, with singleton fallback."""
    if not len(rotations):
        return np.array([0., 0., 1.])
    matrices = Rotation.from_quat(rotations).as_matrix()
    mean = np.mean(matrices[:, 2, :], axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        raise ValueError("Signed body-up mean is undefined; protocol supplies no cancellation fallback")
    return mean / norm


def convention_rotation(up, yaw, normal=(0., 0., 1.)):
    """Apply the minimal body-up rotation followed by yaw about the table normal."""
    tilt, _ = Rotation.align_vectors(np.array([normal]), np.array([up]))
    return Rotation.from_rotvec(np.asarray(normal) * yaw) * tilt


def apply_conventions(quaternions_wxyz, dimensions, ups):
    """Return new body poses and geometry/inertia expressed in those body frames."""
    old = Rotation.from_quat(np.asarray(quaternions_wxyz)[:, [1, 2, 3, 0]])
    yaw = 2 * np.arctan2(np.asarray(quaternions_wxyz)[:, 3], np.asarray(quaternions_wxyz)[:, 0])
    new = Rotation.concatenate([convention_rotation(up, angle) for up, angle in zip(ups, yaw)])
    inertia = np.array([np.diag(box_properties(dims)[0]) for dims in dimensions])
    return reexpress_rotations(old, new, inertia)


def reexpress_rotations(old: Rotation, new: Rotation, inertia: np.ndarray) -> dict:
    """Express geometry and COM inertia in explicit new body rotations without moving them."""
    geometry = (new.inv() * old).as_matrix()
    return dict(quaternions=new.as_quat()[:, [3, 0, 1, 2]], geometry_rotation=geometry,
                inertia=np.einsum("bij,bjk,blk->bil", geometry, inertia, geometry),
                physical_rotation_residual=float(np.max(np.abs(new.as_matrix() @ geometry - old.as_matrix()))))


def relative_transform(poses):
    """Express slot one's rotation and displacement in slot zero's body frame."""
    rotations = Rotation.from_quat(poses[:, 3:])
    return rotations[0].inv() * rotations[1], rotations[0].inv().apply(poses[1, :3] - poses[0, :3])


def transform_errors(achieved, reference):
    """Decompose frame-zero relative errors and gravity-observable body-up errors."""
    ra, ta = relative_transform(achieved)
    rr, tr = relative_transform(reference)
    la, lr = np.linalg.norm(ta), np.linalg.norm(tr)
    direction = np.degrees(np.arccos(np.clip(ta @ tr / (la * lr), -1, 1))) if la * lr else None
    ua = Rotation.from_quat(achieved[:, 3:]).as_matrix()[:, 2, :]
    ur = Rotation.from_quat(reference[:, 3:]).as_matrix()[:, 2, :]
    return dict(offset_error_cm=float(np.linalg.norm(ta - tr) * 100),
                achieved_offset_m=ta.tolist(), reference_offset_m=tr.tolist(),
                distance_ratio=float(la / lr) if lr else None, direction_error_deg=float(direction) if direction is not None else None,
                relative_rotation_error_deg=float(np.degrees((rr.inv() * ra).magnitude())),
                achieved_body_up=ua.tolist(), reference_body_up=ur.tolist(),
                up_axis_error_deg=np.degrees(np.arccos(np.clip(np.sum(ua * ur, axis=1), -1, 1))).tolist(),
                direct_heading_error="unobservable: no camera-to-mocap yaw calibration")


def oracle_replacements(achieved, reference):
    """Replace offset direction, length, relative rotation, or all three at frame zero."""
    ra, ta = relative_transform(achieved)
    rr, tr = relative_transform(reference)
    la, lr = np.linalg.norm(ta), np.linalg.norm(tr)
    if min(la, lr) == 0:
        raise ValueError("Offset direction oracle is undefined at zero displacement")
    anchor = Rotation.from_quat(achieved[0, 3:])
    variants = {}
    for key, rot, offset in (("direction", ra, tr / lr * la), ("length", ra, ta / la * lr),
                             ("rotation", rr, ta), ("joint", rr, tr)):
        poses = achieved.copy()
        poses[1, :3] = poses[0, :3] + anchor.apply(offset)
        poses[1, 3:] = (anchor * rot).as_quat()
        variants[key] = poses
    return variants


def flip_anchor(poses):
    """Turn only slot zero's heading by pi about the table normal."""
    result = poses.copy()
    result[0, 3:] = (Rotation.from_rotvec([0., 0., np.pi]) * Rotation.from_quat(poses[0, 3:])).as_quat()
    return result
