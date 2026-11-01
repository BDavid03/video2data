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


def separate_fallbacks(positions, radii, fallback):
    """Move only uncertain fallback boxes to clear all other boxes' bounding spheres."""
    result = np.array(positions, float)
    for i in np.flatnonzero(fallback):
        # Moving past the furthest right boundary clears every existing object in one pass.
        others = np.arange(len(result)) != i
        distances = np.linalg.norm(result[others,:2] - result[i,:2], axis=-1)
        if np.any(distances < np.asarray(radii)[others] + radii[i] + .03):
            result[i,0] = np.max(result[others,0] + np.asarray(radii)[others]) + radii[i] + .03
    return result
