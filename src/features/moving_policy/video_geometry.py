"""Pure self-reference motion and orientation metrics for M2v."""

import numpy as np

from .geometry import orientation_errors
from src.features.dynamic_feasibility.partial_eval import motion_metrics


def profile_body_transform(body, profile_axes):
    """Remove a tracking-mesh convention using M1's already recorded scan bounding axes."""
    scan_axes = np.asarray(body.get('original_scan_T_asset', body['scan_T_asset']))
    if 'asset_T_bounding_body' in body:
        composed = np.asarray(body['scan_T_asset']) @ np.asarray(body['asset_T_bounding_body'])
        if not np.allclose(composed, scan_axes, atol=1e-12, rtol=0):
            raise ValueError('Recorded scan bounding convention is inconsistent')
    return scan_axes @ np.linalg.inv(profile_axes)


def reference_metrics(achieved, reference):
    """Measure translation, rotation and frozen held-still bounds without a label source."""
    result = motion_metrics(achieved[:, :3], reference[:, :3])
    quaternions = reference[:, [6, 3, 4, 5]]
    result.update(E_m=float(np.linalg.norm(achieved[:, :3]-reference[:, :3], axis=1).mean()),
                  E_S_m=float(np.linalg.norm(reference[:, :3]-reference[:1, :3], axis=1).mean()),
                  Theta_deg=float(orientation_errors(achieved[:, 3:], quaternions).mean()),
                  Theta_S_deg=float(orientation_errors(np.repeat(reference[:1, 3:], len(reference), axis=0), quaternions).mean()))
    result['tracking_bounds_passed'] = bool(result['T'] >= .5 and result['Q'] >= .5
            and result['E_m'] <= min(.5*result['E_S_m'], .03)
            and result['Theta_deg'] <= min(.5*result['Theta_S_deg'], 15.))
    return result
