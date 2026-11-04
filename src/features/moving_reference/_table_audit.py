"""Deterministic robust planes and a training-only table footprint."""
import cv2
import numpy as np


def robust_plane(points, tolerance=.015, trials=1000):
    """Fit the largest plane consensus, then refit its inliers by centred SVD."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return None, None, np.zeros(len(points), bool)
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(trials):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        keep = abs(points@normal-a@normal) <= tolerance
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < 3:
        return None, None, best
    for _ in range(3):
        center = points[best].mean(axis=0)
        _, _, axes = np.linalg.svd(points[best]-center, full_matrices=False)
        normal = axes[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -float(center@normal)
        best = abs(points@normal+offset) <= tolerance
        if best.sum() < 3:
            return None, None, best
    return normal, offset, best


def footprint(shape, training_pixels):
    """Rasterise the convex hull of independently selected planar training points."""
    mask = np.zeros(shape, np.uint8)
    if len(training_pixels) >= 3:
        hull = cv2.convexHull(np.rint(training_pixels).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def inside(mask, pixels):
    """Look up a fixed image region without accepting out-of-image coordinates."""
    xy = np.rint(pixels).astype(int)
    h, w = mask.shape
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return valid & mask[np.clip(xy[:, 1], 0, h-1), np.clip(xy[:, 0], 0, w-1)]


def normal_angle(first, second):
    """Return the unsigned angle of two consistently upward plane normals."""
    if first is None or second is None:
        return None
    return float(np.degrees(np.arccos(np.clip(first@second, -1, 1))))


def table_view_pixels(pixels, intrinsics, pose, first_pose, normal, offset):
    """Map image rays through the training table plane into the frame-0 view."""
    rays = np.c_[pixels, np.ones(len(pixels))]@np.linalg.inv(intrinsics).T
    rays = rays@pose[:3,:3].T
    denominator = rays@normal
    distance = -(pose[:3,3]@normal+offset)/denominator
    world = pose[:3,3]+rays*distance[:,None]
    first = (world-first_pose[:3,3])@first_pose[:3,:3]
    projected = first@intrinsics.T
    uv = projected[:,:2]/projected[:,2:3]
    valid = (distance>0)&(first[:,2]>0)&np.isfinite(uv).all(axis=1)
    return uv, valid
