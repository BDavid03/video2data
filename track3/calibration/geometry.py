"""Calibration geometry: validation of the released schema and the few transforms we need.

Conventions: ``A_T_B`` maps column vectors in frame B to frame A. Pixel coordinates are
pixel centres. Rotation metadata never rotates decoded images implicitly.
"""
import json
import math
from pathlib import Path

import numpy as np

DISTORTION_ORDER = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
                    "s1", "s2", "s3", "s4", "tau_x", "tau_y")
TOLERANCE = 1e-5


def _array(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected a finite {shape} array")
    return array


def matmul(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("incompatible or nonfinite matrices")
    return (a @ b).tolist()


def rotation_quality(rotation):
    r = _array(rotation, (3, 3))
    return {"orthogonality_max_abs": float(np.abs(r.T @ r - np.eye(3)).max()),
            "determinant": float(np.linalg.det(r))}


def _rigid(transform):
    t = _array(transform, (4, 4))
    quality = rotation_quality(t[:3, :3])
    if (np.abs(t[3] - [0, 0, 0, 1]).max() > TOLERANCE or quality["orthogonality_max_abs"] > TOLERANCE
            or abs(quality["determinant"] - 1) > TOLERANCE):
        raise ValueError("transform is not a proper rigid 4x4 matrix")
    return t


def rigid_inverse(transform):
    t = _rigid(transform)
    inverse = np.eye(4)
    inverse[:3, :3] = t[:3, :3].T
    inverse[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inverse.tolist()


def transform_point(transform, point):
    t = _rigid(transform)
    return (t[:3, :3] @ _array(point, (3,)) + t[:3, 3]).tolist()


def pixel_transform(width, height, *, rotation_deg=0, output_size=None):
    """Homography on pixel centres: rotate by 0/180 first, then resize with half-pixel offsets."""
    dims = (width, height, *(output_size or ()))
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dims) or len(dims) not in (2, 4):
        raise ValueError("image dimensions must be positive integers")
    if rotation_deg not in (0, 180):
        raise ValueError("only explicit 0/180 degree rotation is supported")
    h = np.eye(3)
    if rotation_deg == 180:
        h = np.array([[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
    if output_size is not None:
        sx, sy = output_size[0] / width, output_size[1] / height
        h = np.array([[sx, 0.0, (sx - 1) / 2], [0.0, sy, (sy - 1) / 2], [0.0, 0.0, 1.0]]) @ h
    return h.tolist()


def transformed_intrinsics(intrinsics, width, height, **kwargs):
    return (np.asarray(pixel_transform(width, height, **kwargs)) @ _array(intrinsics, (3, 3))).tolist()


def rotate_camera_frame_180(transform):
    """Re-express both endpoint optical frames with S = diag(-1, -1, 1): a basis change, not image processing."""
    s = np.diag([-1.0, -1.0, 1.0, 1.0])
    return (s @ _rigid(transform) @ s).tolist()


def load_calibration(path):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_calibration(calibration)
    return calibration


def _camera_evidence(name, camera, devices):
    width, height = camera["resolution"]
    pixel_transform(width, height, rotation_deg=camera["rotation_deg"])
    if camera["rotation_deg"] != devices[camera["camera"]]["rotation_deg"]:
        raise ValueError(f"{name}: device/camera rotation disagreement")
    k = _array(camera["intrinsic_matrix"], (3, 3))
    expected = {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]}
    if k[0, 0] <= 0 or k[1, 1] <= 0 or k[2].tolist() != [0.0, 0.0, 1.0]:
        raise ValueError(f"{name}: invalid factory intrinsics")
    if any(not math.isclose(camera[field], value, abs_tol=1e-9) for field, value in expected.items()):
        raise ValueError(f"{name}: redundant intrinsic mismatch")
    if camera["distortion_model"] != "Perspective":
        raise ValueError("only the released Perspective model is supported")
    distortion = _array(camera["distortion_coefficients"], (14,))
    entry = {"resolution": [width, height], "rotation_metadata_deg": camera["rotation_deg"],
             "distortion_coefficient_count": 14, "nonzero_tilt": bool(distortion[12:].any())}
    if "extrinsics_to_stereo_left" in camera:
        t = _rigid(camera["extrinsics_to_stereo_left"])
        entry["extrinsic_rotation"] = rotation_quality(t[:3, :3])
        entry["translation_norm_m"] = float(np.linalg.norm(t[:3, 3]))
    return entry


def _stereo_evidence(pair, cameras):
    left, right = pair["left"], pair["right"]
    if left == right or any(cameras[n]["camera"] != pair["camera"] for n in (left, right)):
        raise ValueError("invalid stereo membership")
    if cameras[left]["resolution"] != cameras[right]["resolution"]:
        raise ValueError("stereo resolution mismatch")
    if not math.isfinite(pair["baseline_m"]) or pair["baseline_m"] <= 0:
        raise ValueError("baseline must be positive and finite")
    t = _rigid(pair["left_to_right"])
    norm = float(np.linalg.norm(t[:3, 3]))
    entry = {"left": left, "right": right, "nominal_baseline_m": pair["baseline_m"],
             "extrinsic_translation_norm_m": norm, "baseline_difference_m": norm - pair["baseline_m"],
             "rotation": rotation_quality(t[:3, :3])}
    rectification = []
    for field in ("left_rectification_rotation", "right_rectification_rotation"):
        r = _array(pair[field], (3, 3))
        quality = rotation_quality(r)
        if quality["orthogonality_max_abs"] > TOLERANCE or abs(quality["determinant"] - 1) > TOLERANCE:
            raise ValueError("invalid rectification rotation")
        entry[field] = quality
        rectification.append(r)
    residual = rectification[1] @ t[:3, :3] @ rectification[0].T
    entry["stored_rectification_relative_rotation_max_abs_from_identity"] = float(np.abs(residual - np.eye(3)).max())
    entry["stored_rectification_translation"] = (rectification[1] @ t[:3, 3]).tolist()
    return entry


def validate_calibration(calibration):
    """Validate the released schema and return numeric evidence (consistency, not image alignment)."""
    if calibration.get("translation_units") != "meters":
        raise ValueError("expected translation_units=meters")
    cameras = calibration["cameras"]
    return {"cameras": {name: _camera_evidence(name, camera, calibration["devices"]) for name, camera in cameras.items()},
            "stereo_pairs": {pair["camera"]: _stereo_evidence(pair, cameras) for pair in calibration["stereo_pairs"]}}
