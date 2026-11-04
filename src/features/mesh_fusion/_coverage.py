"""Camera-ray diversity and exact 3 mm object surface-voxel support."""

import numpy as np
from scipy.spatial import cKDTree


def angular_span(centres, object_centre):
    rays = np.asarray(object_centre) - np.asarray(centres).reshape(-1, 3)
    if len(rays) < 2:
        return 0.
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))


def surface_voxels(points, origin, pitch=0.003):
    return np.unique(np.floor((np.asarray(points).reshape(-1, 3) - origin) / pitch).astype(np.int64), axis=0)


def supported_surface(reference, accepted, pitch=0.003):
    """Use a fixed origin; numerator is fused voxels absent from the opening observation."""
    origin = np.floor(reference.min(axis=0) / pitch) * pitch
    opening = surface_voxels(reference, origin, pitch)
    fused = surface_voxels(np.concatenate([reference, *accepted]), origin, pitch)
    old = set(map(tuple, opening))
    additional = sum(tuple(voxel) not in old for voxel in fused)
    return dict(pitch_m=pitch, origin_m=origin.tolist(), opening_voxels=len(opening),
                fused_voxels=len(fused), additional_voxels=additional,
                additional_supported_surface_percent=100. * additional / len(fused))


def object_region(reference, normal, offset):
    """Grow the opening stereo seed by 30 mm, above its local support plane."""
    return dict(lower=np.percentile(reference, 0.5, axis=0) - 0.03,
                upper=np.percentile(reference, 99.5, axis=0) + 0.03,
                normal=np.asarray(normal), offset=float(offset), seed=reference)


def select_object(points, valid, pose, region):
    world = points @ pose[:3, :3].T + pose[:3, 3]
    inside = (valid & (world >= region['lower']).all(-1) & (world <= region['upper']).all(-1)
              & (world @ region['normal'] - region['offset'] > 0.006))
    selected = world[inside]
    # Limit growth to the seed's neighbourhood; no newly touching hand can grow the ROI indefinitely.
    if len(selected):
        selected = selected[cKDTree(region['seed']).query(selected)[0] <= 0.03]
    return selected


def verdict(accepted_count, angle, additional_percent):
    reasons = []
    if accepted_count < 3:
        reasons.append('fewer than 3 accepted frames')
    if angle < 10 and additional_percent < 10:
        reasons.append('angular span < 10 deg AND additional supported surface < 10%')
    return dict(verdict='KILL' if reasons else 'PASS', reasons=reasons,
                frozen_rule='kill if accepted frames < 3, OR (angle < 10 deg AND additional surface < 10%)')
