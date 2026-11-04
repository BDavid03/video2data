"""Robust stereo centres and a fixed own-mesh surface correction."""

import cv2
import numpy as np

from .camera import apply, project


def centre_observation(frame, minimum_points=30):
    """Take a median after a robust two-sigma camera-depth cut, excluding hands."""
    inner = cv2.erode(frame['object_mask'].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hands = cv2.dilate(frame['hands'].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    good = inner & ~hands & frame['depth_valid'] & np.isfinite(frame['xyz']).all(2)
    good &= frame['xyz'][..., 2] > .05
    cloud = frame['xyz'][good].astype(float)
    sigma = None
    if len(cloud):
        depth = np.median(cloud[:, 2])
        sigma = max(1.4826 * np.median(abs(cloud[:, 2] - depth)), .001)
        retained = cloud[abs(cloud[:, 2] - depth) <= 2 * sigma]
    else:
        retained = cloud
    supported = len(retained) >= minimum_points
    centre = np.median(retained, axis=0) if supported else None
    detail = dict(points_before_cut=len(cloud), points_after_cut=len(retained),
                  depth_sigma_m=None if sigma is None else float(sigma), centre_observed=supported,
                  hand_overlap_pixels=int((inner & hands).sum()))
    return centre, detail


def surface_offset(samples, camera_T_mesh, k, shape):
    """Estimate the signed ray offset from frontmost mesh pixels to its origin."""
    points = apply(camera_T_mesh, samples)
    points = points[np.isfinite(points).all(1) & (points[:, 2] > .05)]
    pixels = np.rint(project(points, k)).astype(int)
    inside = (pixels >= 0).all(1) & (pixels[:, 0] < shape[1]) & (pixels[:, 1] < shape[0])
    points, pixels = points[inside], pixels[inside]
    keys = pixels[:, 1] * shape[1] + pixels[:, 0]
    order = np.lexsort((points[:, 2], keys))
    _, first = np.unique(keys[order], return_index=True)
    visible = points[order[first]]
    if not len(visible):
        raise ValueError('No own-mesh surface projects into the frame-zero image')
    centroid = visible.mean(0)
    ray = centroid / np.linalg.norm(centroid)
    offset = float((camera_T_mesh[:3, 3] - centroid) @ ray)
    return offset, centroid, len(visible)


def position_step(centre, world_T_camera, offset, history, previous, initial):
    """Apply the fixed ray correction and causal median, carrying unsupported outputs."""
    if centre is None:
        raw = np.array(history[-1] if len(history) else initial, copy=True)
        return raw, np.array(previous if previous is not None else initial, copy=True)
    camera_position = centre + offset * centre / np.linalg.norm(centre)
    raw = apply(world_T_camera, camera_position[None])[0]
    filtered = np.median(np.vstack([*history, raw]), axis=0)
    return raw, filtered
