"""Recompute all stereo maps and projection matrices in one calibration call."""

from src.provenance import sha256_file
from ._common import authorize, is_rigid, operation_run, read_json, save_npz, select_episode


def _camera(camera, np):
    k = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    d = np.asarray(camera["distortion_coefficients"], dtype=np.float64)
    size = tuple(camera["resolution"])
    if (camera["distortion_model"] != "Perspective" or k.shape != (3, 3) or d.shape != (14,)
            or not np.isfinite(k).all() or not np.isfinite(d).all() or min(*size, k[0, 0], k[1, 1]) <= 0):
        raise ValueError("Expected factory K and all 14 Perspective distortion coefficients")
    return k, d, size


def _rigid(value, np):
    matrix = np.asarray(value, dtype=np.float64)
    if not is_rigid(matrix, np):
        raise ValueError("Invalid metric rigid calibration transform")
    return matrix


def _inside(x, y, size, np):
    return np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= size[0] - 1) & (y >= 0) & (y <= size[1] - 1)


def rectify(request):
    authorize("rectify", request)
    selection = select_episode(request)
    with operation_run("rectify", request, selection) as (directory, record):
        import cv2
        import numpy as np

        # select_episode checked the C-left/B-right policy; B_T_C holds only for the audited file.
        calibration = read_json(selection["calibration"])
        if sha256_file(selection["calibration"]) != read_json("data/configs/calibration/ego.json")["source_sha256"]:
            raise ValueError("Calibration file differs from the audited one in data/configs/calibration/ego.json")
        if calibration.get("translation_units") != "meters":
            raise ValueError("Stereo calibration translations must be metres")
        kl, dl, size = _camera(calibration["cameras"]["ego_cam_c"], np)
        kr, dr, right_size = _camera(calibration["cameras"]["ego_cam_b"], np)
        if right_size != size:
            raise ValueError("Stereo cameras must have the same source resolution")
        pairs = [pair for pair in calibration["stereo_pairs"] if pair["camera"] == "ego"]
        if len(pairs) != 1:
            raise ValueError("Exactly one ego stereo calibration is required")
        b_t_c = _rigid(pairs[0]["left_to_right"], np)
        b_t_a = _rigid(calibration["cameras"]["ego_cam_a"]["extrinsics_to_stereo_left"], np)
        a_t_c = np.linalg.inv(b_t_a) @ b_t_c
        ratio = min(1.0, request.max_image_side / max(size))
        output_size = tuple(round(value * ratio) for value in size)
        # OpenCV needs a contiguous 3x3 R and a 3x1 (not row-vector) T.
        rotation = np.ascontiguousarray(b_t_c[:3, :3])
        translation = np.ascontiguousarray(b_t_c[:3, 3].reshape(3, 1))
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            kl, dl.reshape(14, 1), kr, dr.reshape(14, 1), size, rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=output_size)
        baseline = -float(p2[0, 3] / p2[0, 0])
        if baseline <= 0 or abs(p2[1, 3]) > 1e-8:
            raise ValueError("Expected positive-disparity horizontal C-left/B-right rectification")
        # Remap takes the 3x3 rectified intrinsics; the full 3x4 P1/P2 are saved below.
        left_k, right_k = np.ascontiguousarray(p1[:, :3]), np.ascontiguousarray(p2[:, :3])
        lx, ly = cv2.initUndistortRectifyMap(kl, dl.reshape(14, 1), r1, left_k, output_size, cv2.CV_32FC1)
        rx, ry = cv2.initUndistortRectifyMap(kr, dr.reshape(14, 1), r2, right_k, output_size, cv2.CV_32FC1)
        save_npz(directory / "rectification.npz", np, R1=r1, R2=r2, P1=p1, P2=p2, Q=q,
                 map_left_x=lx, map_left_y=ly, map_right_x=rx, map_right_y=ry,
                 left_valid=_inside(lx, ly, size, np), right_valid=_inside(rx, ry, size, np), A_T_C=a_t_c,
                 source_size=np.asarray(size, dtype=np.int64))
        record.update(baseline_m=baseline, focal_px=float(p1[0, 0]),
                      output_size_width_height=list(output_size), additional_rotation_degrees=0)
    return directory / "run.json"
