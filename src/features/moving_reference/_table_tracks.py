"""Project a fixed table footprint and summarise independent stereo tracks."""
import cv2
import numpy as np

from ._table_audit import table_view_pixels


def projected_region(region, intrinsics, pose, first_pose, normal, offset):
    """Carry a frame-0 image footprint through its training-only plane."""
    h,w=region.shape
    corners=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    uv,_=table_view_pixels(corners,intrinsics,pose,first_pose,normal,offset)
    transform=cv2.getPerspectiveTransform(corners,uv.astype(np.float32))
    return cv2.warpPerspective(region.astype(np.uint8),transform,(w,h),
        flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP).astype(bool)


def point_metrics(points, anchors, normal, offset):
    """Measure every admitted point without residual-based rejection."""
    if not len(points):
        return None,None
    errors=points-anchors
    errors-=np.einsum('ij,j->i',errors,normal)[:,None]*normal
    return float(np.median(abs(points@normal+offset))),float(np.median(np.linalg.norm(errors,axis=1)))
