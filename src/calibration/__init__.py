"""Calibration checks (NumPy); coordinate changes are always explicit.

``A_T_B`` acts on column vectors in B to return coordinates in A. Factory
calibration is preserved: rotation metadata alone never rotates decoded images.
"""
from .geometry import (load_calibration, validate_calibration, matmul, rigid_inverse,
                       transform_point, pixel_transform, transformed_intrinsics,
                       rotate_camera_frame_180, DISTORTION_ORDER)

__all__ = ["load_calibration", "validate_calibration", "matmul", "rigid_inverse",
           "transform_point", "pixel_transform", "transformed_intrinsics",
           "rotate_camera_frame_180", "DISTORTION_ORDER"]
