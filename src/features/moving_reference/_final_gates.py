"""Pure final M1 gate composition under the recorded sealed-pass condition."""

from copy import deepcopy
import math

RULE = ("Only when the sealed reference and allowance both pass, gate the perpendicular world residual "
        "at the unchanged 10 mm; retain old 3D residual descriptively and change no other limit.")
KIND = 'moving_reference.final_gates.v1'
WORLD = 'camera_heldout_world_residual_median_m'

FINAL_LIMITS = {
    'camera_heldout_reprojection_median_px640': ('le', .5),
    'camera_heldout_reprojection_p95_px640': ('le', 2.),
    'camera_heldout_world_residual_median_m': ('le', .01),
    'camera_heldout_tracks_min': ('ge', 30), 'camera_heldout_regions_min': ('ge', 4),
    'camera_coverage_fraction': ('ge', .98), 'table_residual_rms_m': ('le', .005),
    'table_disjoint_patch_normal_max_deg': ('le', 3.), 'mask_usable_fraction': ('ge', .9),
    'mask_longest_unusable_interval_frames': ('le', 2),
    'object_30_stereo_correspondence_fraction': ('ge', .9),
    'object_fb_translation_median_m': ('le', .01), 'object_fb_rotation_median_deg': ('le', 5.),
    'object_speed_max_m_s': ('le', 1.5), 'object_angular_rate_max_deg_s': ('le', 540.),
    'object_reference_valid_fraction': ('ge', .9),
    'reprojection_iou_median': ('ge', .6), 'reprojection_iou_p10': ('ge', .3),
    'reprojection_coverage_fraction': ('ge', .9), 'frame0_base_gap_m': ('le', .005),
    'frame0_penetration_m': ('le', .005),
    'table_offset_max_frame_median_m': ('le', .01), 'table_offset_p95_frame_medians_m': ('le', .015),
    'table_late_normal_change_deg': ('le', 2.), 'table_drift_used_fraction': ('ge', .80),
    'table_landmark_max_frame_median_m': ('le', .01),
    'table_landmark_p95_frame_medians_m': ('le', .02), 'table_landmark_used_fraction': ('ge', .80),
}


def compose_final(source: dict, sealed: dict) -> dict:
    """Copy the last amended record and apply only the conditionally authorized instrument."""
    if sealed.get('reference_passed') is not True or sealed.get('allowance_passed') is not True:
        raise ValueError('Sealed reference_passed and allowance_passed must both be true')
    result = deepcopy(source)
    gates = result['gates']
    if set(gates) != set(FINAL_LIMITS):
        raise ValueError('Final gate roster differs from the fixed 28 gates')
    world = gates[WORLD]
    value = world.get('perpendicular_median_m')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Missing or nonfinite perpendicular world residual')
    world['value'] = value
    world['instrument'] = 'perpendicular under sealed-pass clause 2'
    for name, (comparison, threshold) in FINAL_LIMITS.items():
        gate = gates[name]
        if gate.get('threshold') != threshold or gate.get('comparison') != comparison:
            raise ValueError(f'Final gate limit differs from the recorded rule: {name}')
        value = gate.get('value')
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        fraction = 'fraction' in name or 'iou' in name
        passed = bool(finite and value >= 0 and (not fraction or value <= 1)
                      and (value <= threshold if comparison == 'le' else value >= threshold))
        if name == WORLD:
            gate['passed'] = passed
        elif gate.get('passed') is not passed:
            raise ValueError(f'Recorded gate verdict disagrees with its unchanged measurement: {name}')
    result['passed'] = all(gate['passed'] for gate in gates.values())
    return result
