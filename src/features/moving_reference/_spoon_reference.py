"""Final camera measurements on the frozen M1 held-out observations."""

import numpy as np

from . import camera


def camera_measurements(data, poses, rows):
    """Use first-sight anchors and the sealed perpendicular residual instrument."""
    observations = data['rows']
    selected = observations[data['heldout'][observations[:, 1].astype(int)]]
    anchors = {}
    perpendicular = []
    old = []
    for frame, pose in enumerate(poses):
        observed = selected[selected[:, 0] == frame]
        points = camera.apply(pose, observed[:, 6:9])
        for index, point in zip(observed[:, 1].astype(int), points):
            anchors.setdefault(int(index), point.copy())
        errors = points - np.asarray([anchors[int(i)] for i in observed[:, 1]]).reshape(-1, 3)
        rays = points - pose[:3, 3]
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        parallel = np.einsum('ij,ij->i', errors, rays)
        perpendicular.extend(np.linalg.norm(errors - parallel[:, None]*rays, axis=1))
        old.extend(np.linalg.norm(errors, axis=1))
    reprojection = [v for row in rows for v in row['heldout_reprojection_px640']]
    values = dict(
        camera_heldout_reprojection_median_px640=float(np.median(reprojection)),
        camera_heldout_reprojection_p95_px640=float(np.percentile(reprojection, 95)),
        camera_heldout_world_residual_median_m=float(np.median(perpendicular)),
        camera_heldout_tracks_min=min(row['heldout_count'] for row in rows),
        camera_heldout_regions_min=min(row['heldout_regions'] for row in rows),
        camera_coverage_fraction=float(np.mean([row['observed'] for row in rows])))
    return values, float(np.median(old))
