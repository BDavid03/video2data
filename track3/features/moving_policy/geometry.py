"""Pure M2 geometry, with no paths, records, admission or file I/O."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def matrix(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]]).as_matrix()


def wxyz(rotation):
    return Rotation.from_matrix(rotation).as_quat()[..., [3, 0, 1, 2]]


def transform_poses(poses, rotation, translation):
    poses = np.asarray(poses, float)
    result = poses.copy()
    result[..., :3] = poses[..., :3] @ rotation.T + translation
    result[..., 3:] = wxyz(rotation @ matrix(poses[..., 3:]))
    return result


def follow_object(local_poses, positions, rotations):
    local_poses = np.asarray(local_poses)
    return np.concatenate((np.einsum('tij,kj->tki', rotations, local_poses[:, :3]) + positions[:, None],
                           wxyz(rotations[:, None] @ matrix(local_poses[:, 3:])[None])), axis=-1)


def template_frame(task):
    """First maximum contact-coverage sample after the template rises 5 cm."""
    height = np.asarray(task['object_body_position'])[:, 0, 2]
    coverage = sum(np.count_nonzero(task[f'mano_{side}_object_contact_part_ids'], axis=1)
                   for side in ('left', 'right'))
    coverage = np.where(height - height[0] > .05, coverage, -1)
    if coverage.max() < 0:
        raise ValueError('The shipped template has no 5 cm carry segment')
    return int(np.argmax(coverage))


def rim_grasp(groups, cloud, triangles, side, repair=False):
    """Place captured middle fingertips at opposite longitudinal exterior faces.

    Two hands hold the scoop's front lip and the distal grip. Four fixed rolls are compared using only
    geometry, before Stage 0. The sole permitted repair increases exterior clearance.
    This does not synthesize any object trajectory or change the captured joints.
    """
    bounds = np.array([triangles.min((0, 1)), triangles.max((0, 1))])
    sign = -1. if side == 'left' else 1.
    inward = np.array([-sign, 0., 0.])
    direction = groups[3] - groups[0]
    direction /= np.linalg.norm(direction)
    across = groups[1] - groups[3]
    across -= across @ direction * direction
    across /= np.linalg.norm(across)
    source = np.stack((direction, across, np.cross(direction, across)), axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centers = triangles.mean(axis=1)
    exterior = ((normals[:, 0] * sign > .9)
                & (np.abs(centers[:, 0] - bounds[int(sign > 0), 0]) < .01))
    if not exterior.any():
        raise ValueError('No longitudinal exterior face for the fixed grasp rule')
    selected = np.flatnonzero(exterior)[np.argmin(np.abs(centers[exterior, 1]))]
    target = centers[selected].copy()
    target[0] += sign * (.008 if repair else .003)
    equations = ConvexHull(triangles.reshape(-1, 3)).equations
    sample = cloud[np.linspace(0, len(cloud) - 1, min(2048, len(cloud)), dtype=int)]
    candidates = []
    for roll in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
        transverse = Rotation.from_rotvec(inward * roll).apply([0., 1., 0.])
        destination = np.stack((inward, transverse, np.cross(inward, transverse)), axis=1)
        rotation = destination @ source.T
        translation = target - rotation @ groups[3]
        transformed = sample @ rotation.T + translation
        depth = np.maximum(0., -(transformed @ equations[:, :3].T + equations[:, 3]).max(axis=1))
        # A conservative full-asset hull keeps the palm/fingers out of the scoop.
        cost = (float(depth.max()), float(depth.sum()), roll)
        candidates.append((cost, rotation, translation))
    cost, rotation, translation = min(candidates, key=lambda row: row[0])
    return rotation, translation, {'side': side, 'tip_group': 3, 'target_m': target.tolist(),
                                 'roll_rad': cost[2], 'hull_sample_penetration_m': cost[0]}


def deepest_piece_penetration(hand_pieces, object_pieces):
    """Exact LP depth of hand solids inside each simulated convex object piece.

    The meshes are convex collision shapes in Newton/MuJoCo. For each pair,
    maximize the minimum signed distance to the object's supporting planes over
    the intersection. This includes triangle interiors, not just sampled tips.
    """
    maximum, intersections = 0., 0
    hand_eq = [ConvexHull(vertices).equations for vertices in hand_pieces]
    object_eq = [ConvexHull(vertices).equations for vertices in object_pieces]
    for vertices, h in zip(hand_pieces, hand_eq, strict=True):
        for object_vertices, o in zip(object_pieces, object_eq, strict=True):
            if np.any(vertices.max(0) < object_vertices.min(0)) or np.any(vertices.min(0) > object_vertices.max(0)):
                continue
            a = np.concatenate((np.column_stack((h[:, :3], np.zeros(len(h)))),
                                np.column_stack((o[:, :3], np.ones(len(o))))))
            b = -np.concatenate((h[:, 3], o[:, 3]))
            result = linprog([0., 0., 0., -1.], A_ub=a, b_ub=b,
                             bounds=[(None, None)] * 3 + [(0., None)], method='highs')
            if result.success:
                intersections += 1
                maximum = max(maximum, float(result.x[3]))
    return {'maximum_m': maximum, 'intersecting_shape_pairs': intersections,
            'method': 'LP over full convex collision solids; maximum per-piece penetration depth'}


def motion_summary(positions, times):
    delta = np.diff(positions, axis=0)
    speed = np.linalg.norm(delta, axis=1) / np.diff(times)
    distance = np.linalg.norm(positions - positions[0], axis=1)
    return {'frames': len(times), 'duration_s': float(times[-1] - times[0]),
            'maximum_displacement_m': float(distance.max()), 'path_length_m': float(np.linalg.norm(delta, axis=1).sum()),
            'maximum_speed_m_s': float(speed.max()), 'maximum_height_rise_m': float((positions[:, 2] - positions[0, 2]).max()),
            'peak_frame': int(np.argmax(distance))}
