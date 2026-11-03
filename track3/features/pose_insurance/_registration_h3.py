"""Pure extraction of scored-stage masks and calibrated stereo observations."""
import numpy as np
from ._length import stereo_clusters
from ._scene import camera_points


def observations(stereo: dict, labels: np.ndarray, intrinsics: np.ndarray,
                 cluster_ids: list, normal: list, minimum_pixels: int) -> dict:
    """Construct independent object clouds and background validity on the admitted grid."""
    masks = [np.isin(labels, np.array(ids) + 1) for ids in cluster_ids]
    clusters, counts = stereo_clusters(stereo, masks, labels, intrinsics, minimum_pixels)
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    return dict(clusters=clusters, counts=counts, labels=labels, valid=valid,
                points=camera_points(stereo['rgb_depth_m'].astype(float), intrinsics),
                normal_hint=np.array(normal))
