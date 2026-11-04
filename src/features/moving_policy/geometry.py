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


def rim_grasp(groups, cloud, triangles, side, repair=False, pair="longitudinal", head_fraction=.6):
    """Place captured middle fingertips at opposite longitudinal exterior faces.

    Two hands hold the scoop's front lip and the distal grip. Four fixed rolls are compared using only
    geometry, before Stage 0. The sole permitted repair increases exterior clearance.
    This does not synthesize any object trajectory or change the captured joints.
    """
    bounds = np.array([triangles.min((0, 1)), triangles.max((0, 1))])
    sign = -1. if side == 'left' else 1.
    axis = 0 if pair == 'longitudinal' else 1
    inward = np.zeros(3); inward[axis] = -sign
    direction = groups[3] - groups[0]
    direction /= np.linalg.norm(direction)
    across = groups[1] - groups[3]
    across -= across @ direction * direction
    across /= np.linalg.norm(across)
    source = np.stack((direction, across, np.cross(direction, across)), axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    centers = triangles.mean(axis=1)
    exterior = ((normals[:, axis] * sign > .8)
                & ((np.abs(centers[:, 0] - bounds[int(sign > 0), 0]) < .01) if axis == 0 else np.ones(len(centers), bool)))
    if not exterior.any():
        raise ValueError('No longitudinal exterior face for the fixed grasp rule')
    desired_axis = 0. if axis == 0 else bounds[0, 0] + head_fraction * np.ptp(bounds[:, 0]) / 2
    selected = np.flatnonzero(exterior)[np.argmin(np.abs(centers[exterior, 1 - axis] - desired_axis))]
    target = centers[selected].copy()
    target[axis] += sign * (.008 if repair else .003)
    equations = ConvexHull(triangles.reshape(-1, 3)).equations
    sample = cloud[np.linspace(0, len(cloud) - 1, min(2048, len(cloud)), dtype=int)]
    candidates = []
    for roll in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
        transverse = Rotation.from_rotvec(inward * roll).apply([0., 1., 0.] if axis == 0 else [1., 0., 0.])
        destination = np.stack((inward, transverse, np.cross(inward, transverse)), axis=1)
        rotation = destination @ source.T
        translation = target - rotation @ groups[3]
        transformed = sample @ rotation.T + translation
        depth = np.maximum(0., -(transformed @ equations[:, :3].T + equations[:, 3]).max(axis=1))
        # A conservative full-asset hull keeps the palm/fingers out of the scoop.
        cost = (float(depth.max()), float(depth.sum()), roll)
        candidates.append((cost, rotation, translation))
    cost, rotation, translation = min(candidates, key=lambda row: row[0])
    return rotation, translation, {'side': side, 'pair': pair, 'tip_group': 3, 'target_m': target.tolist(),
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


def acquisition_rule(poses, times, seconds=.5):
    """Permit an approach only while the reference is stationary at the fixed limits."""
    use = times - times[0] <= seconds
    translation = np.linalg.norm(poses[use, :3] - poses[0, :3], axis=1)
    angle = np.degrees((Rotation.from_matrix(matrix(poses[use, 3:])) *
                        Rotation.from_matrix(matrix(poses[0, 3:])).inv()).magnitude())
    return bool(translation.max() <= .001 and angle.max() <= 1.), float(translation.max()), float(angle.max())


def cylinder_support(pieces, position, rotation):
    """Return a footprint-covering thin cylinder from imported collision solids."""
    vertices = np.concatenate(pieces) @ rotation.T + position
    center = (vertices[:, :2].min(0) + vertices[:, :2].max(0)) / 2
    radius = float(np.linalg.norm(vertices[:, :2] - center, axis=1).max())
    top = float(vertices[:, 2].min() - .001)
    return center.tolist(), radius, top


def orientation_errors(achieved_xyzw, reference_wxyz):
    """Return per-frame shortest orientation error in degrees."""
    return np.degrees((Rotation.from_quat(achieved_xyzw) *
                       Rotation.from_matrix(matrix(reference_wxyz)).inv()).magnitude())


def amended_verdict(t, q, error, theta, e_static, theta_static, auc):
    """Apply Amendment 1's frozen final-policy thresholds to raw metrics."""
    if t >= .5 and q >= .5 and error <= min(.5 * e_static, .03) and theta <= min(.5 * theta_static, 15.) and auc >= .45:
        return 'GO'
    if t >= .2 and q >= .1 and error <= min(e_static, .06) and theta <= min(.9 * theta_static, 30.) and auc >= .20:
        return 'SIGNAL'
    return 'NULL'


def diagnostic_feedforward_residual(positions, quaternions_wxyz, fps, mass, inertia):
    """Reference-only soft-wrist inverse PD; never a policy or an exportable rollout."""
    dt = 1. / fps
    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    rotation = Rotation.from_matrix(matrix(quaternions_wxyz))
    omega = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
    omega = np.concatenate((omega, omega[-1:]))
    alpha = np.gradient(omega, dt, axis=0)
    # Target held over a control interval; compensate the midpoint sample as well
    # as damping and inertia, using reference derivatives alone.
    position = (6. / 25. + .5 * dt) * velocity + mass / 25. * acceleration
    body_omega = rotation.inv().apply(omega)
    body_alpha = rotation.inv().apply(alpha)
    torque = body_alpha @ inertia.T + np.cross(body_omega, body_omega @ inertia.T) + 6. * body_omega
    advance = torque / 25. + .5 * dt * body_omega
    # Warp quat_rpy uses extrinsic xyz angles in the reference's child frame.
    euler = Rotation.from_rotvec(advance).as_euler('xyz')
    return np.column_stack((np.clip(position, -.2, .2), np.clip(euler, -1., 1.)))


def inverse_residual_input(desired, previous, scale, clip, ema, mapping, input_scale):
    """Invert upstream EMA and its input map within the configured action bounds."""
    mapped = (np.clip(desired, -clip, clip) - ema * previous) / ((1. - ema) * scale)
    if mapping == 'identity':
        return mapped
    if mapping == 'linear':
        return np.clip(mapped / input_scale, -1., 1.)
    return np.clip(mapped / (1. + np.abs(mapped) * (1. - 1. / input_scale)), -1., 1.)
