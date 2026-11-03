"""Pure stereo offsets, bounded placement, integrity gates and direction readings."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._length import footprint_centres, offset
from ._scene import camera_points
from ._shrink import footprint_clearance


def stereo_offset(stereo: dict, masks: list, normal: np.ndarray) -> tuple:
    """Measure the diagnostic's exact valid-pixel centroid displacement."""
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    points = camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels'])
    counts = [0 if mask is None else int((mask & valid).sum()) for mask in masks]
    clusters = [points[mask & valid] if mask is not None and count >= 300 else None
                for mask, count in zip(masks, counts)]
    return offset(footprint_centres(clusters, normal)), counts


def placement(s1: dict, displacement: np.ndarray | None, reason: str | None) -> tuple:
    """Edit only slot-one XY, returning a feasible clearance endpoint or the incumbent."""
    scene = {key: np.array(value, copy=True) for key, value in s1.items()}
    if len(scene['positions']) == 1:
        return scene, dict(status='unchanged', reason='single-object control', factor=None)
    if reason is not None:
        return scene, dict(status='fallback', reason=reason, factor=None)
    length = float(np.linalg.norm(displacement))
    if not np.isfinite(length) or length == 0:
        return scene, dict(status='fallback', reason='zero or non-finite stereo offset', factor=None)
    upper = float(np.linalg.norm(s1['positions'][1, :2] - s1['positions'][0, :2]) / length)
    scene['positions'][1, :2] = scene['positions'][0, :2] + displacement
    initial_gap = footprint_clearance(scene['positions'], scene['dimensions'], scene['quaternions'])
    details = dict(raw_clearance_m=initial_gap, upper_bound=upper, raw_length_m=length)
    if initial_gap >= .01:
        return scene, dict(details, status='changed', reason='raw stereo offset clears', factor=1.,
                           clearance_m=initial_gap)
    if upper < 1:
        return s1, dict(details, status='fallback', reason='empty clearance interval: upper bound below one', factor=None)
    scene['positions'][1, :2] = scene['positions'][0, :2] + upper * displacement
    upper_gap = footprint_clearance(scene['positions'], scene['dimensions'], scene['quaternions'])
    if upper_gap < .01:
        return s1, dict(details, status='fallback', reason='no feasible clearance upper endpoint',
                        factor=None, upper_clearance_m=upper_gap)
    lower = 1.
    while upper - lower > 1e-4:
        middle = (lower + upper) / 2
        scene['positions'][1, :2] = scene['positions'][0, :2] + middle * displacement
        if footprint_clearance(scene['positions'], scene['dimensions'], scene['quaternions']) >= .01:
            upper = middle
        else:
            lower = middle
    scene['positions'][1, :2] = scene['positions'][0, :2] + upper * displacement
    return scene, dict(details, status='changed', reason='expanded stereo offset for clearance', factor=upper,
                       infeasible_lower=lower, width=upper-lower,
                       clearance_m=footprint_clearance(scene['positions'], scene['dimensions'], scene['quaternions']))


def start_poses(scene: dict) -> np.ndarray:
    """Express the scene as the kit's XYZ plus XYZW initial poses."""
    return np.concatenate((scene['positions'], scene['quaternions'][:, [1, 2, 3, 0]]), axis=1)


def unchanged_fields(s1: dict, t1: dict) -> bool:
    """Require every field except slot-one XY to remain byte-equivalent as arrays."""
    copied = {key: np.array(value, copy=True) for key, value in t1.items()}
    if len(copied['positions']) > 1:
        copied['positions'][1, :2] = s1['positions'][1, :2]
    return all(np.array_equal(value, copied[key]) for key, value in s1.items())


def offset_comparison(actual, expected) -> dict:
    """Compare raw vectors at absolute tolerance, including registered missingness."""
    if actual is None or expected is None:
        return dict(passed=actual is None and expected is None, missing=actual is None,
                    expected_missing=expected is None, max_difference_m=None)
    difference = float(np.max(np.abs(np.asarray(actual) - expected)))
    return dict(passed=bool(np.isfinite(difference) and difference <= 1e-9),
                missing=False, expected_missing=False, max_difference_m=difference)


def direction_reading(scene: dict, ref_start: np.ndarray, raw_offset, thresholds: np.ndarray) -> dict:
    """Describe applied horizontal offsets in the same aligned frame as ray_errors."""
    alignment = (Rotation.from_quat(scene['quaternions'][0, [1, 2, 3, 0]]) *
                 Rotation.from_quat(ref_start[0, 3:]).inv())
    target = alignment.apply(ref_start[1, :3] - ref_start[0, :3])[:2]
    predicted = scene['positions'][1, :2] - scene['positions'][0, :2]
    target_length, predicted_length = float(np.linalg.norm(target)), float(np.linalg.norm(predicted))
    cross = target[0] * predicted[1] - target[1] * predicted[0]
    angle = float(np.degrees(np.arctan2(cross, target @ predicted))) if target_length * predicted_length else None
    error = float(np.linalg.norm(predicted - target) * 100)
    tolerance = float(np.degrees(2 * np.arcsin(min(1., .09 / (2 * target_length))))) if target_length else None
    return dict(signed_angle_deg=angle, horizontal_error_cm=error,
                aligned_reference_xy_m=target.tolist(), applied_offset_xy_m=predicted.tolist(),
                reference_length_cm=target_length * 100, applied_length_cm=predicted_length * 100,
                inside_position_thresholds={f'{t * 100:.0f}': bool(error < t * 100) for t in thresholds},
                raw_exact_direction_error_cm=None if raw_offset is None else
                    float(abs(np.linalg.norm(raw_offset) - target_length) * 100),
                applied_exact_direction_error_cm=abs(predicted_length - target_length) * 100,
                equal_length_angle_tolerance_deg=tolerance)


def scene_placement(scene: list, displacement, reason: str | None) -> tuple:
    """Apply the same T1 rule to scored scene rows while preserving all other metadata."""
    from copy import deepcopy
    geometry = dict(positions=np.array([row['position'] for row in scene], float),
                    dimensions=np.array([row['dimensions_m'] for row in scene], float),
                    quaternions=np.array([row['quaternion_wxyz'] for row in scene], float))
    transformed, clearance = placement(geometry, displacement, reason)
    result = deepcopy(scene)
    if len(scene) > 1:
        result[1]['position'][:2] = transformed['positions'][1, :2].tolist()
    return result, clearance
