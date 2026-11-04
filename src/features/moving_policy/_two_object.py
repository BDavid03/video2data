"""Pure task composition, video role selection and resting-body measurements."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation


def joint_row(row: dict, active_name: str, names: list[str], companion: dict) -> dict:
    """Add one constant free-body reference in the official object-slot order."""
    if len(names) != 2 or active_name not in names or len(set(names)) != 2:
        raise ValueError('A joint task requires two distinct official object names')
    active = names.index(active_name)
    resting = 1 - active
    if companion['name'] != names[resting] or len(row['object_body_names']) != 1:
        raise ValueError('Companion identity or active-task body count differs')
    result = copy.deepcopy(row)
    poses = np.asarray(row['object_body_position'], float)
    quaternions = np.asarray(row['object_body_wxyz'], float)
    start = np.asarray(companion['pose_wxyz'], float)
    count = len(poses)
    positions = np.empty((count, 2, 3))
    rotations = np.empty((count, 2, 4))
    positions[:, active] = poses[:, 0]
    rotations[:, active] = quaternions[:, 0]
    positions[:, resting] = start[:3]
    rotations[:, resting] = start[3:]
    result.update(object_name='+'.join(names), safe_object_name='+'.join(names),
                  object_body_names=list(names), safe_object_body_names=list(names),
                  object_body_position=positions.tolist(), object_body_wxyz=rotations.tolist(),
                  object_root_position=positions[:, 0].tolist(),
                  object_root_axis_angle=Rotation.from_quat(rotations[:, 0, [1, 2, 3, 0]]).as_rotvec().tolist())
    for field, key in [('object_mesh_paths', 'mesh'), ('object_urdf_paths', 'urdf'),
                       ('object_mesh_radius', 'radius')]:
        values = [None, None]
        values[active] = row[field][0]
        values[resting] = companion[key]
        result[field] = values
    for side in ('left', 'right'):
        ids = np.asarray(row[f'mano_{side}_object_contact_part_ids'], int)
        result[f'mano_{side}_object_contact_part_ids'] = np.where(ids > 0, active + 1, 0).tolist()
    return result


def select_active(centres: np.ndarray, background_steps: np.ndarray, width: int) -> dict:
    """Choose the larger 95th-percentile background-corrected mask displacement."""
    centres = np.asarray(centres, float)
    steps = np.asarray(background_steps, float)
    if centres.ndim != 3 or centres.shape[1:] != (2, 2) or steps.shape != (len(centres), 2):
        raise ValueError('Role measurement arrays have incompatible shapes')
    if not np.isfinite(centres[0]).all() or not np.isfinite(steps).all():
        raise ValueError('No frame-zero object centre or valid background flow')
    corrected = centres - np.cumsum(steps, axis=0)[:, None]
    distances = np.linalg.norm(corrected - corrected[0], axis=2) / width
    scores = np.nanpercentile(distances, 95, axis=0)
    if not np.isfinite(scores).all():
        raise ValueError('Missing video motion statistic')
    active = int(np.argmax(scores))
    return dict(active_slot=active, resting_slot=1-active, displacement_p95_image_width=scores.tolist())


def resting_metrics(poses_xyzw: np.ndarray, intended_wxyz: np.ndarray) -> dict:
    """Measure the panel's first-pose and whole-rollout physical limits."""
    poses = np.asarray(poses_xyzw, float)
    intended = np.asarray(intended_wxyz, float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Resting body needs a finite full pose trajectory')
    rotation = Rotation.from_quat(poses[:, 3:])
    target = Rotation.from_quat(intended[[4, 5, 6, 3]])
    error = np.linalg.norm(poses[:, :3] - intended[:3], axis=1)
    angle = np.degrees((rotation * target.inv()).magnitude())
    drift = np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1)
    turn = np.degrees((rotation * rotation[0].inv()).magnitude())
    result = dict(first_translation_m=float(error[0]), first_rotation_deg=float(angle[0]),
                  maximum_translation_from_start_m=float(error.max()), maximum_rotation_from_start_deg=float(angle.max()),
                  maximum_translation_drift_m=float(drift.max()), maximum_rotation_drift_deg=float(turn.max()))
    result['passed'] = bool(error[0] <= .01 and angle[0] <= 2 and error.max() < .01
                            and angle.max() < 2 and drift.max() < .01 and turn.max() < 2)
    return result
