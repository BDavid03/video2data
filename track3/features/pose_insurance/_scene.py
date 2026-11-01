"""Pure table-frame geometry and static robot reference arrays."""
import numpy as np


def extents_with_axes(points, height, normal):
    """Return the original extents unchanged, plus the corresponding camera-space PCA axes."""
    e1 = np.array([1., 0, 0]) - normal[0] * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.stack((points @ e1, points @ e2), 1)
    uv -= uv.mean(0)
    pca = np.linalg.svd(uv, full_matrices=False)[2]
    low, high = np.percentile(uv @ pca.T, [2, 98], axis=0)
    original = {"in_plane_m": (high-low).tolist(), "height_m": float(np.percentile(height, 98)),
                "height_max_m": float(height.max())}
    axes = pca @ np.stack((e1, e2))
    # Fix SVD sign ambiguity and make the returned frame right handed.
    if axes[0, np.argmax(np.abs(axes[0]))] < 0:
        axes[0] *= -1
    axes[1] = np.cross(normal, axes[0])
    return original, axes


def table_basis(normal):
    x = np.array([1., 0, 0]) - normal[0] * normal
    x /= np.linalg.norm(x)
    return np.stack((x, np.cross(normal, x), normal))


def box_pose(points, normal, dims):
    """Seat the box on z=0 using its cluster's projected centre and PCA heading."""
    basis = table_basis(normal)
    _, axes = extents_with_axes(points, points @ normal, normal)
    heading = basis @ axes[0]
    yaw = np.arctan2(heading[1], heading[0])
    centre = basis @ points.mean(0)
    centre[2] = dims[2] / 2
    return centre, np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])


def scaled_dimensions(proportions, measured):
    """Keep D1 proportions and restore the measured geometric-mean metric size."""
    dims = np.asarray(proportions, float)
    return dims * np.exp(np.log(np.asarray(measured, float)).mean() - np.log(dims).mean())


def parked_robot(template, count):
    """Repeat one shipped robot pose after translating each hand away from the scene."""
    result = {}
    for side, x in (("left", -3.), ("right", 3.)):
        position = np.asarray(template[f"robot_{side}_wrist_position"][0], float)
        shift = np.array([x, -3., 2.]) - position
        for suffix in ("wrist_position", "wrist_wxyz", "finger_joints", "frames"):
            value = np.asarray(template[f"robot_{side}_{suffix}"][0], float).copy()
            if suffix == "wrist_position":
                value += shift
            if suffix == "frames":
                value[..., :3] += shift
            result[f"robot_{side}_{suffix}"] = np.repeat(value[None], count, axis=0).tolist()
        for suffix in ("finger_joint_names", "frame_names"):
            result[f"{side}_robot_{suffix}"] = template[f"{side}_robot_{suffix}"]
        for suffix in ("positions", "normals"):
            result[f"mano_{side}_object_contact_{suffix}"] = np.zeros((count, 16, 3)).tolist()
        result[f"mano_{side}_object_contact_part_ids"] = np.zeros((count, 16), dtype=int).tolist()
    return result


def drift(poses):
    """Maximum translation and geodesic rotation from the first exported pose."""
    translation = np.linalg.norm(poses[..., :3] - poses[:1, ..., :3], axis=-1)
    q = poses[..., 3:] / np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
    dots = np.abs(np.sum(q * q[:1], axis=-1)).clip(0, 1)
    return float(translation.max() * 100), float(np.degrees(2 * np.arccos(dots)).max())


def camera_points(z, k):
    """Unproject metric camera-z depth into a camera-space point grid."""
    v, u = np.indices(z.shape)
    return np.stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z), axis=-1)


def box_properties(dimensions, mass=.1):
    """Principal inertia and enclosing radius of one solid metric box."""
    dims = np.asarray(dimensions, float)
    return mass / 12 * (np.sum(dims**2) - dims**2), float(np.linalg.norm(dims) / 2)


def rectangle_translation(first, second, first_dims, second_dims, first_quat, second_quat):
    """Minimum separating translation of the first oriented footprint, or zero if disjoint."""
    angles = [2 * np.arctan2(q[3], q[0]) for q in (first_quat, second_quat)]
    bases = [np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) for a in angles]
    axes = np.concatenate(bases)
    delta = np.asarray(first)[:2] - np.asarray(second)[:2]
    radius = sum(np.abs(axes @ basis.T) @ (np.asarray(dims)[:2] / 2)
                 for basis, dims in zip(bases, (first_dims, second_dims)))
    penetration = radius - np.abs(axes @ delta)
    if (penetration <= 0).any():
        return np.zeros(2)
    index = int(penetration.argmin())
    direction = 1 if axes[index] @ delta >= 0 else -1
    return direction * axes[index] * (penetration[index] + 1e-5)


def separate_boxes(positions, dimensions, quaternions, movable):
    """Separate actual footprint overlaps, preferentially moving an uncertain placement."""
    result = np.array(positions, float)
    for first in range(len(result)):
        for second in range(first + 1, len(result)):
            i, j = (first, second) if movable[first] else (second, first)
            shift = rectangle_translation(result[i], result[j], dimensions[i], dimensions[j],
                                          quaternions[i], quaternions[j])
            result[i,:2] += shift
    distances = np.linalg.norm(result - np.asarray(positions), axis=1) * 100
    return result, distances


def pose_statistics(poses):
    """Finite/unit-pose checks and drift, without any file or admission state."""
    values = np.asarray(poses, float)
    finite = bool(np.isfinite(values).all())
    unit = bool(np.allclose(np.linalg.norm(values[...,3:], axis=-1), 1, atol=1e-4))
    cm, deg = drift(values) if finite and unit else (None, None)
    return {"finite":finite, "unit_quaternions":unit, "drift_cm":cm, "drift_deg":deg}
