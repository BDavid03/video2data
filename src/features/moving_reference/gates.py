"""Tabulate every frozen M1 gate, failing closed on unmeasured evidence."""

import numpy as np
from scipy.spatial.transform import Rotation


def _amended_check(value, threshold, comparison='le'):
    finite=value is not None and np.isfinite(value)
    return dict(value=float(value) if finite else None,threshold=threshold,comparison=comparison,
                passed=bool(finite and (value<=threshold if comparison=='le' else value>=threshold)))


def table_drift_gate(medians, normal_degrees, used_fraction):
    values=np.asarray(medians,float);values=values[np.isfinite(values)]
    return dict(table_offset_max_frame_median_m=_amended_check(float(values.max()) if len(values) else None,.01),
                table_offset_p95_frame_medians_m=_amended_check(float(np.percentile(values,95)) if len(values) else None,.015),
                table_late_normal_change_deg=_amended_check(normal_degrees,2),
                table_drift_used_fraction=_amended_check(used_fraction,.8,'ge'))


def long_baseline_gate(medians, used_fraction):
    values=np.asarray(medians,float);values=values[np.isfinite(values)]
    return dict(table_landmark_max_frame_median_m=_amended_check(float(values.max()) if len(values) else None,.01),
                table_landmark_p95_frame_medians_m=_amended_check(float(np.percentile(values,95)) if len(values) else None,.02),
                table_landmark_used_fraction=_amended_check(used_fraction,.8,'ge'))


def perpendicular_world_gate(perpendicular_median, old_median, control):
    explained=bool(control['pairs']>=100 and control['frames']>=3 and
                   control['norm_median_m'] is not None and control['norm_median_m']>=old_median and
                   control['along_median_m']>=3*control['perpendicular_median_m'])
    result=_amended_check(perpendicular_median if explained else old_median,.01)
    result.update(old_3d_median_m=old_median,perpendicular_median_m=perpendicular_median,
                  noise_floor_explains_old=explained,instrument='perpendicular' if explained else 'original 3D retained',control=control)
    return result


def stereo_pixel_gate(counts,klt_counts):
    result=_amended_check(float(np.mean(np.asarray(counts)>=30)),.9,'ge')
    result.update(original_klt_fraction=float(np.mean(np.asarray(klt_counts)>=30)),minimum_stereo_pixels=int(min(counts)))
    return result


def amended_reprojection_gate(median,p95):
    return dict(camera_heldout_reprojection_median_px640=_amended_check(median,.5),
                camera_heldout_reprojection_p95_px640=_amended_check(p95,2))


def amended_patch_normal_gate(value):
    return _amended_check(value,3)


def evaluate(table, camera_rows, object_rows, times, poses, valid, ious, usable, first_base, support):
    result = {}

    def gate(name, value, limit, direction='le', note=None):
        finite = value is not None and np.isfinite(value)
        passed = bool(finite and (value<=limit if direction=='le' else value>=limit))
        result[name] = dict(value=float(value) if finite else None, threshold=limit,
                            comparison=direction, passed=passed, note=note)

    def median(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(values)) if values else None

    reprojection = [v for row in camera_rows for v in row['heldout_reprojection_px640']]
    positions = [v for row in camera_rows for v in row['heldout_world_residual_m']]
    gate('table_residual_rms_m', table['residual_rms_m'], .005)
    gate('table_disjoint_patch_normal_max_deg', table['patch_normal_max_deg'], 2)
    gate('camera_heldout_reprojection_median_px640', median(reprojection), 1/3)
    gate('camera_heldout_reprojection_p95_px640', float(np.percentile(reprojection,95)) if reprojection else None, 1)
    gate('camera_heldout_world_residual_median_m', median(positions), .01)
    gate('camera_heldout_tracks_min', min(row['heldout_count'] for row in camera_rows), 30, 'ge')
    gate('camera_heldout_regions_min', min(row['heldout_regions'] for row in camera_rows), 4, 'ge')
    gate('camera_coverage_fraction', np.mean([row['observed'] for row in camera_rows]), .98, 'ge')
    gate('mask_usable_fraction', np.mean(usable), .90, 'ge',
         'Operational usability: eroded non-hand object has >=30 stereo-supported texture points; semantics checked only in overlays.')
    missing = np.flatnonzero(~np.asarray(usable))
    chunks = np.split(missing, np.flatnonzero(np.diff(missing)>1)+1)
    gate('mask_longest_unusable_interval_frames', max(map(len, chunks)), 2)
    gate('object_30_stereo_correspondence_fraction', np.mean([row['correspondences']>=30 for row in object_rows]), .90, 'ge')
    gate('object_fb_translation_median_m', median([row['fb_translation_m'] for row in object_rows[1:]]), .01)
    gate('object_fb_rotation_median_deg', median([row['fb_rotation_deg'] for row in object_rows[1:]]), 5)
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)/np.diff(times)
    rotation = poses[:-1, :3, :3].transpose(0,2,1)@poses[1:, :3, :3]
    angular = np.rad2deg(Rotation.from_matrix(rotation).magnitude())/np.diff(times)
    gate('object_speed_max_m_s', float(speed.max()), 1.5,
         note='All written frames, including invalid fallbacks; validity is gated separately.')
    gate('object_angular_rate_max_deg_s', float(angular.max()), 540,
         note='All written frames, including invalid fallbacks; validity is gated separately.')
    gate('object_reference_valid_fraction', np.mean(valid), .90, 'ge', 'Includes resolved initial facing, observed camera and keyframe refinement.')
    measured = np.asarray(ious)[np.isfinite(ious)]
    gate('reprojection_iou_median', float(np.median(measured)) if len(measured) else None, .60, 'ge')
    gate('reprojection_iou_p10', float(np.percentile(measured,10)) if len(measured) else None, .30, 'ge')
    gate('reprojection_coverage_fraction', len(measured)/len(ious), .90, 'ge')
    gate('frame0_base_gap_m', max(0., first_base), .005)
    gate('frame0_penetration_m', max(0., -first_base), .005)
    gate('zero_action_support_translation_m', support['maximum_translation_m'], .01,
         note='CPU evaluator, 1 second, static frame-0 reference, hands cleared; '+support['status'])
    gate('zero_action_support_rotation_deg', support['maximum_rotation_deg'], 2,
         note='CPU evaluator, 1 second, static frame-0 reference, hands cleared; '+support['status'])
    return dict(passed=all(row['passed'] for row in result.values()), gates=result,
                speed_all_frames_max_m_s=float(speed.max()), angular_all_frames_max_deg_s=float(angular.max()),
                label_free=True, mocap_scored=False)
