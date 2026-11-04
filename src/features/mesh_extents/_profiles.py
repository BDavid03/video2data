"""Fit pooled three-knot vessel profiles and construct shells from array inputs."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax


def table_coordinates(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Express camera points in an orthonormal plane basis with height above support."""
    normal = np.asarray(normal, dtype=float)
    e1 = np.array([1., 0., 0.]) - normal[0] * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return np.column_stack((points @ e1, points @ e2, points @ normal + offset))


def prepare_frame(points: np.ndarray, handled: bool, recipe: dict) -> dict:
    """Normalize by measured height and remove the fixed handle sector before fitting."""
    height = float(np.percentile(points[:, 2], recipe['height_percentile']))
    xy = points[:, :2] - np.mean(points[:, :2], axis=0)
    axes = np.linalg.svd(xy, full_matrices=False)[2]
    xy = xy @ axes.T
    # The handle produces a long positive tail; the sign is set before any profile fit.
    if np.mean((xy[:, 0] - np.median(xy[:, 0])) ** 3) < 0:
        xy[:, 0] *= -1
    low, high = np.percentile(xy, recipe['footprint_percentiles'], axis=0)
    radius = (high[1] - low[1]) / 2
    centre = (low + high) / 2
    if handled:
        centre[0] = low[0] + radius
    else:
        xy[:, 0] = centre[0] + (xy[:, 0] - centre[0]) * (high[1] - low[1]) / (high[0] - low[0])
    xyz = np.column_stack(((xy - centre) / height, points[:, 2] / height))
    keep = (xyz[:, 2] >= 0) & (xyz[:, 2] <= 1)
    if handled:
        angle = np.arctan2(xyz[:, 1], xyz[:, 0])
        keep &= np.abs(angle) >= np.deg2rad(recipe['handle_half_angle_degrees'])
    return {'points': xyz[keep], 'height_m': height, 'radius': float(radius / height),
            'raw_points': len(points), 'retained_points': int(keep.sum()),
            'handle_axis_table': axes[0].tolist()}


def height_support(frames: list[dict], recipe: dict) -> dict:
    """Measure occupied height bins and support around each fixed knot."""
    h = np.concatenate([f['points'][:, 2] for f in frames])
    counts, _ = np.histogram(h, np.linspace(0, 1, recipe['height_bins'] + 1))
    threshold = max(recipe['minimum_bin_points'], int(np.ceil(recipe['minimum_bin_fraction'] * len(h))))
    windows = recipe['knot_support_windows']
    knot_counts = [int(((h >= a) & (h <= b)).sum()) for a, b in windows]
    occupied = counts >= threshold
    span = float(np.percentile(h, 98) - np.percentile(h, 2))
    passed = (span >= recipe['minimum_height_span'] and
              all(c >= threshold for c in knot_counts) and
              occupied.mean() >= recipe['minimum_occupied_height_fraction'])
    return {'counts': counts.tolist(), 'knot_window_counts': knot_counts, 'threshold_points': threshold,
            'occupied_fraction': float(occupied.mean()), 'p02_p98_span': span, 'pass': bool(passed)}


def _radii(parameters: np.ndarray) -> np.ndarray:
    return 3 * softmax(np.r_[parameters, 0.])


def radial_residual(points: np.ndarray, centre: np.ndarray, scale: float,
                    radii: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return signed radial point-to-surface distance in frame-height units."""
    return np.linalg.norm(points[:, :2] - centre, axis=1) - scale * np.interp(points[:, 2], knots, radii)


def _sample(points: np.ndarray, count: int) -> np.ndarray:
    return points[np.linspace(0, len(points) - 1, count, dtype=int)]


def fit_profile(frames: list[dict], recipe: dict, fixed_radii: np.ndarray | None = None) -> dict:
    """Fit shared mean-one radii with independent frame centres and radial scales."""
    free = fixed_radii is None
    count = min(recipe['fit_points_per_frame'], min(len(f['points']) for f in frames))
    points = [_sample(f['points'], count) for f in frames]
    refs = np.array([f['radius'] for f in frames])
    initial = np.column_stack((np.zeros((len(frames), 2)), refs)).ravel()
    lower = np.column_stack((-refs, -refs, recipe['radius_lower_factor'] * refs)).ravel()
    upper = np.column_stack((refs, refs, recipe['radius_upper_factor'] * refs)).ravel()
    if free:
        initial = np.r_[0., 0., initial]
        lower, upper = np.r_[-recipe['logit_bound'] * np.ones(2), lower], np.r_[recipe['logit_bound'] * np.ones(2), upper]
    knots = np.asarray(recipe['height_knots'])

    def residual(x):
        radii = _radii(x[:2]) if free else np.asarray(fixed_radii)
        nuisance = x[2 if free else 0:].reshape(-1, 3)
        return np.concatenate([radial_residual(p, n[:2], n[2], radii, knots) / ref
                               for p, n, ref in zip(points, nuisance, refs)])

    fitted = least_squares(residual, initial, bounds=(lower, upper), loss=recipe['loss'],
                           f_scale=recipe['loss_scale_radius'], max_nfev=recipe['max_nfev'],
                           ftol=recipe['optimizer_tolerance'], xtol=recipe['optimizer_tolerance'],
                           gtol=recipe['optimizer_tolerance'])
    return {'radii': (_radii(fitted.x[:2]) if free else np.asarray(fixed_radii)).tolist(),
            'nuisance': fitted.x[2 if free else 0:].reshape(-1, 3).tolist(),
            'success': bool(fitted.success), 'nfev': int(fitted.nfev), 'cost': float(fitted.cost)}


def predict_frame(frame: dict, radii: list[float], recipe: dict) -> dict:
    """Calibrate nuisance parameters on even pixels and score odd pixels only."""
    calibration = {**frame, 'points': frame['points'][::2]}
    fit = fit_profile([calibration], recipe, fixed_radii=np.asarray(radii))
    centre_x, centre_y, scale = fit['nuisance'][0]
    residual = radial_residual(frame['points'][1::2], (centre_x, centre_y), scale,
                               np.asarray(radii), np.asarray(recipe['height_knots'])) / frame['radius']
    return {'mean_absolute_residual_radius': float(np.abs(residual).mean()),
            'coverage': float((np.abs(residual) <= recipe['coverage_tolerance_radius']).mean()),
            'scored_points': len(residual), 'calibration_fit': fit}


def evaluate_gate(frames: list[dict], recipe: dict) -> dict:
    """Fit alternating disjoint subsets and cross-predict their omitted frames."""
    subsets = [frames[::2], frames[1::2]]
    if not all(subsets):
        return {'decision': 'FAIL', 'reason': 'fewer than two admitted frames'}
    support = [height_support(s, recipe) for s in subsets]
    fits = [fit_profile(s, recipe) for s in subsets]
    predictions = []
    for i, heldout in enumerate(subsets):
        for frame in heldout:
            predictions.append({'id': frame['id'], 'trained_subset': 1 - i,
                                'profile': predict_frame(frame, fits[1 - i]['radii'], recipe),
                                'constant': predict_frame(frame, [1., 1., 1.], recipe)})
    differences = np.abs(np.array(fits[0]['radii']) - fits[1]['radii'])
    profile_residual = np.mean([p['profile']['mean_absolute_residual_radius'] for p in predictions])
    constant_residual = np.mean([p['constant']['mean_absolute_residual_radius'] for p in predictions])
    coverage = {m: float(np.mean([p[m]['coverage'] for p in predictions])) for m in ('profile', 'constant')}
    improvement = 1 - profile_residual / constant_residual
    checks = {'a': bool(np.all(differences < recipe['maximum_subset_difference_radius'])),
              'b': bool(improvement >= recipe['minimum_residual_reduction']),
              'c': bool(coverage['profile'] >= coverage['constant']),
              'height': all(s['pass'] for s in support),
              'optimizer': all(f['success'] for f in fits) and all(
                  p[m]['calibration_fit']['success'] for p in predictions for m in ('profile', 'constant'))}
    pooled = fit_profile(frames, recipe)
    checks['optimizer'] &= pooled['success']
    return {'decision': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'subset_ids': [[f['id'] for f in s] for s in subsets], 'height_support': support,
            'subset_fits': fits, 'knot_differences_radius': differences.tolist(),
            'profile_residual_radius': float(profile_residual), 'constant_residual_radius': float(constant_residual),
            'residual_reduction': float(improvement), 'coverage': coverage, 'predictions': predictions,
            'pooled_fit': pooled}


def profile_shell(extents: np.ndarray, radii: np.ndarray, handle: str = 'none'):
    """Revolve an open shell inside the v4 body bounds and attach the original handle."""
    import trimesh
    from ._shapes import _attach_vessel_handle

    long, short, height = np.asarray(extents, dtype=float)
    outer = np.asarray(radii, dtype=float) / np.max(radii) * short / 2
    wall = 0.10 * short / 2
    if np.min(outer) <= wall or height <= wall:
        raise ValueError('Profile leaves no positive cavity')
    heights = np.array([0., .5, 1.]) * height - height / 2
    inner_bottom = np.interp(-height / 2 + wall, heights, outer) - wall
    outline = [(0., -height / 2), *zip(outer, heights),
               (outer[-1] - wall, heights[-1]), (outer[1] - wall, heights[1]),
               (inner_bottom, -height / 2 + wall), (0., -height / 2 + wall), (0., -height / 2)]
    body = trimesh.creation.revolve(outline, sections=64)
    if handle == 'none':
        body.apply_scale((long / short, 1, 1))
        return body
    return _attach_vessel_handle(body, extents, rim_flat=handle == 'rim_flat')


def reconstruct_frame(z, valid, intrinsics, masks, name, plane, original, request,
                      recipe, admitted, frame_id):
    """Reconstruct one cached video frame and require its original mask/extent geometry."""
    from ._geometry import camera_points, table_component, _local_plane, extents
    handled = name in ('white_pot', 'water_pitcher')
    shared = sum(m.astype(np.int8) for m in masks.values()) > 1
    mask = masks[name] & ~shared
    pts = camera_points(z, intrinsics)
    ok = valid & np.isfinite(pts).all(-1) & (z > 0)
    normal, offset = np.array(plane['normal']), plane['offset_m']
    table = table_component(pts @ normal + offset, ok, request['table_band_m'])
    _, normal, offset, local = _local_plane(pts, table, mask, normal, offset, request['ring_pixels'])
    points = pts[mask & ok]
    geometry = extents(points, points @ normal + offset, normal)
    error = max(abs(geometry['height_m'] - original['height_m']),
                float(np.max(np.abs(np.array(geometry['in_plane_m']) - original['in_plane_m']))))
    equal = len(points) == original['pixels'] and error < 1e-10
    if not equal:
        raise ValueError(f'Reconstructed geometry differs from original: {frame_id} error={error}')
    prepared = prepare_frame(table_coordinates(points, normal, offset), handled, recipe)
    accept = admitted and local and prepared['retained_points'] >= recipe['minimum_retained_points']
    diagnostic = {'id': frame_id, 'frame': int(frame_id.rsplit(':', 1)[1]), 'admitted': accept, 'mask_pixels': int(mask.sum()),
                  'local_plane': bool(local), 'geometry_max_abs_error_m': error,
                  'table_pixels': int(table.sum()), 'mask_table_overlap': int((mask & table).sum()),
                  'below_plane_points': int(((points @ normal + offset) < 0).sum()),
                  **{key: value for key, value in prepared.items() if key != 'points'}}
    return prepared, diagnostic


def saved_checks(document):
    """Recompute the frozen gate arithmetic and geometry equivalence from saved values."""
    kill, control = document['results']
    gate = kill['gate']
    differences = np.abs(np.array(gate['subset_fits'][0]['radii'])-gate['subset_fits'][1]['radii'])
    residuals = {model: float(np.mean([row[model]['mean_absolute_residual_radius'] for row in gate['predictions']]))
                 for model in ('profile', 'constant')}
    coverage = {model: float(np.mean([row[model]['coverage'] for row in gate['predictions']]))
                for model in ('profile', 'constant')}
    reduction = 1-residuals['profile']/residuals['constant']
    assert np.allclose(differences, gate['knot_differences_radius'], rtol=0, atol=1e-15)
    assert abs(reduction-gate['residual_reduction']) < 1e-15
    assert coverage == gate['coverage']
    assert document['decision'] == 'R10 dropped' and not gate['checks']['a']
    assert control['fit_performed'] is False
    frames = [row for result in document['results'] for row in result['frames']]
    assert len(frames) == 9 and max(row['geometry_max_abs_error_m'] for row in frames) == 0
    assert all(row['below_plane_points'] == 0 for row in frames)
    result = dict(geometry_equivalence_frames=len(frames), geometry_max_abs_error_m=0.,
                  maximum_subset_difference_radius=float(max(differences)), residual_reduction=reduction,
                  coverage=coverage, candidate_plys=0, fitting_repeated=False)
    return result
