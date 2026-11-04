"""Object correspondence, silhouette initialization and fixed pose smoothing."""

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .camera import apply, corners, flow, lift, project, rigid


def render(vertices, faces, camera_T_asset, k, shape):
    points = apply(camera_T_asset, vertices)
    uv = project(points, k)
    mask = np.zeros(shape, np.uint8)
    for face in faces:
        if np.all(points[face, 2] > .05) and np.isfinite(uv[face]).all():
            triangle = np.rint(np.clip(uv[face], -10000, 10000)).astype(np.int32)
            cv2.fillConvexPoly(mask, triangle, 1)
    return mask.astype(bool)


def iou(a, b, hands):
    a, b = a & ~hands, b & ~hands
    union = (a|b).sum()
    return float((a&b).sum()/union) if union else 0.


def initial_pose(xyz, visible, object_mask, hands, world_T_camera, vertices, faces, k):
    cloud = xyz[visible & np.isfinite(xyz).all(axis=2)]
    cloud = apply(world_T_camera, cloud)
    center = np.median(cloud, axis=0)
    _, vectors = np.linalg.eigh(np.cov(cloud[:, :2].T))
    angle = np.arctan2(vectors[1, -1], vectors[0, -1])
    # Initial height comes from stereo, not from forcing contact with the table.
    center[2] = np.percentile(cloud[:, 2], 15) - np.percentile(vertices[:, 2], 15)
    camera_T_world = np.linalg.inv(world_T_camera)
    candidates = []
    for yaw in angle + np.arange(4)*np.pi/2:
        def pose_at(values):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('z', values[0]).as_matrix()
            pose[:3, 3] = values[1:]
            return pose

        def objective(values):
            pose = pose_at(values)
            predicted = render(vertices, faces, camera_T_world@pose, k, object_mask.shape)
            return 1-iou(predicted, object_mask, hands) + .1*np.linalg.norm(values[1:]-center)

        result = minimize(objective, np.r_[yaw, center], method='Powell',
                          bounds=[(yaw-.6, yaw+.6), *((v-.06, v+.06) for v in center)],
                          options={'maxiter': 35, 'xtol': .0003, 'ftol': .0001})
        pose = pose_at(result.x)
        score = iou(render(vertices, faces, camera_T_world@pose, k, object_mask.shape), object_mask, hands)
        candidates.append((score, pose))
    candidates.sort(key=lambda row:row[0], reverse=True)
    winner = candidates[0]
    opposite = [row for row in candidates[1:]
                if Rotation.from_matrix(winner[1][:3,:3].T@row[1][:3,:3]).magnitude()>np.pi*.75]
    margin = winner[0]-max((row[0] for row in opposite), default=0)
    # A tied opposite-facing solution is invalid, irrespective of the later track.
    return camera_T_world@winner[1], dict(
        method='Stereo cloud PCA heading; four silhouette fits about the stereo table normal; stereo initial height; no world-axis heading inherited',
        candidate_ious=[float(row[0]) for row in candidates], opposite_iou_margin=float(margin),
        opposite_resolved=bool(margin>.02), initial_world_T_asset=winner[1].tolist())


def pair(previous_gray, current_gray, previous_xyz, current_xyz, previous_mask, current_mask):
    pixels = corners(previous_gray, previous_mask)
    nxt, good = flow(previous_gray, current_gray, pixels, previous_mask, current_mask)
    a, va = lift(pixels, previous_xyz)
    b, vb = lift(nxt, current_xyz)
    good &= va & vb
    transform, kept = rigid(a[good], b[good])
    return transform, int(kept.sum())


def object_step(previous, current, keyframes, previous_pose):
    gray, xyz, mask = current
    transform, count = pair(*previous[:1], gray, previous[1], xyz, previous[2], mask)
    backward, backward_count = pair(gray, previous[0], xyz, previous[1], mask, previous[2])
    fb_translation, fb_rotation = None, None
    if transform is not None and backward is not None:
        closure = backward@transform
        fb_translation = float(np.linalg.norm(closure[:3, 3]))
        fb_rotation = float(np.rad2deg(Rotation.from_matrix(closure[:3, :3]).magnitude()))
    sources, targets, keyframe_ids = [], [], []
    for index, old_gray, old_xyz, old_mask, old_pose in keyframes:
        pixels = corners(old_gray, old_mask)
        nxt, good = flow(old_gray, gray, pixels, old_mask, mask)
        a, va = lift(pixels, old_xyz)
        b, vb = lift(nxt, xyz)
        good &= va & vb
        if good.sum() >= 6:
            sources.append(apply(np.linalg.inv(old_pose), a[good]))
            targets.append(b[good]); keyframe_ids.append(index)
    if sources:
        refined, kept = rigid(np.concatenate(sources), np.concatenate(targets))
    else:
        refined, kept = None, np.zeros(0, bool)
    measured = refined is not None
    pose = refined if measured else (transform@previous_pose if transform is not None else previous_pose.copy())
    return pose, dict(correspondences=count, backward_correspondences=backward_count,
                      keyframe_correspondences=int(kept.sum()), keyframes=keyframe_ids,
                      keyframe_refined=measured, fb_translation_m=fb_translation, fb_rotation_deg=fb_rotation,
                      observed=bool(transform is not None and measured))


def smooth(poses, valid):
    result = poses.copy()
    groups = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid))>1)+1)
    for ids in groups:
        if len(ids)<5:
            continue
        vectors = Rotation.from_matrix(poses[ids, :3, :3]).as_rotvec()
        for index in range(1, len(vectors)):
            angle = np.linalg.norm(vectors[index])
            if angle>1e-9:
                axis = vectors[index]/angle
                choices = vectors[index] + np.arange(-2,3)[:, None]*2*np.pi*axis
                vectors[index] = choices[np.argmin(np.linalg.norm(choices-vectors[index-1], axis=1))]
        result[ids, :3, 3] = savgol_filter(poses[ids, :3, 3], 5, 2, axis=0, mode='interp')
        result[ids, :3, :3] = Rotation.from_rotvec(savgol_filter(vectors, 5, 2, axis=0, mode='interp')).as_matrix()
    return result


def body_axes(bounds_transform, extents, points):
    """Orient bounding axes by the broad scoop end and the dense floor side."""
    axes = np.linalg.inv(bounds_transform)
    order = np.argsort(extents)[::-1]
    rotation = axes[:3, :3][:, order]
    local = (points-axes[:3, 3])@rotation
    width = []
    for sign in (-1, 1):
        part = local[sign*local[:, 0]>.3*extents[order[0]]]
        width.append(np.ptp(part[:, 1]) if len(part) else 0)
    if width[1]>width[0]:
        rotation[:, 0] *= -1
    # The thin floor carries more area than the raised rim; its side is -z.
    if np.median((points-axes[:3, 3])@rotation[:, 2])>0:
        rotation[:, 2] *= -1
    rotation[:, 1] = np.cross(rotation[:, 2], rotation[:, 0])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, axes[:3, 3]
    return result, dict(ordered_bounding_extents_m=np.asarray(extents)[order].tolist(), end_widths_m=width)
