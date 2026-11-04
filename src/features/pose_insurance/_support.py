"""Pure common-support offsets, attribution and spatially withheld depth fits."""

import numpy as np

from ._scene import camera_points, table_basis


def describe_offset(vector: np.ndarray | None, basis: np.ndarray) -> dict | None:
    """Describe an offset in camera and fixed table coordinates."""
    if vector is None:
        return None
    table = basis @ vector
    return dict(camera_m=vector, table_m=table, horizontal_m=table[:2],
                length_3d_cm=float(np.linalg.norm(vector) * 100),
                length_horizontal_cm=float(np.linalg.norm(table[:2]) * 100))


def gap_decomposition(m_full: np.ndarray, s_full: np.ndarray,
                      m_common: np.ndarray, s_common: np.ndarray,
                      basis: np.ndarray) -> dict:
    """Retain exact signed length and vector depth/support decompositions."""
    lengths = [float(np.linalg.norm((basis @ d)[:2]) * 100)
               for d in (m_full, s_full, m_common, s_common)]
    mf, sf, mi, si = lengths
    depth, support_m, support_s = mi - si, mf - mi, -(sf - si)
    total = mf - sf
    vectors = dict(total=m_full - s_full, depth=m_common - s_common,
                   moge_support=m_full - m_common, stereo_support=-(s_full - s_common))
    return dict(g_F_cm=total, g_I_cm=depth, depth_cm=depth,
                moge_support_cm=support_m, stereo_support_cm=support_s,
                support_cm=support_m + support_s,
                scalar_residual_cm=total - depth - support_m - support_s,
                vectors_m=vectors,
                vector_residual_m=vectors['total'] - vectors['depth']
                                  - vectors['moge_support'] - vectors['stereo_support'],
                opposing_contributions=bool(depth * (support_m + support_s) < 0))


def ray_decomposition(m_points: np.ndarray, s_points: np.ndarray,
                      rays: np.ndarray, common: list, factor: float) -> dict | None:
    """Project each common-support centroid difference onto its mean pixel ray."""
    if any(int(mask.sum()) < 300 for mask in common):
        return None
    objects = []
    for mask in common:
        ray = rays[mask].mean(0)
        ray /= np.linalg.norm(ray)
        difference = factor * m_points[mask] - s_points[mask]
        delta = difference.mean(0)
        along = ray * (ray @ delta)
        cross = delta - along
        objects.append(dict(mean_ray=ray, centroid_difference_m=delta,
                            along_m=along, cross_m=cross,
                            along_signed_cm=float(ray @ delta * 100),
                            cross_norm_cm=float(np.linalg.norm(cross) * 100),
                            point_cross_max_m=float(np.linalg.norm(
                                np.cross(difference, rays[mask]), axis=1).max())))
    along = objects[1]['along_m'] - objects[0]['along_m']
    cross = objects[1]['cross_m'] - objects[0]['cross_m']
    return dict(objects=objects, offset_along_m=along, offset_cross_m=cross,
                offset_difference_m=objects[1]['centroid_difference_m']
                                    - objects[0]['centroid_difference_m'])


def projection_sensitivity(normal: np.ndarray, vectors: dict, gaps: dict) -> dict:
    """Reproject fixed offsets under four prescribed five-degree normal changes."""
    basis = table_basis(normal)
    rows = []
    for axis_name, axis in zip(('table_x', 'table_y'), basis[:2]):
        for degrees in (-5, 5):
            angle = np.deg2rad(degrees)
            changed = normal * np.cos(angle) + np.cross(axis, normal) * np.sin(angle)
            changed += axis * (axis @ normal) * (1 - np.cos(angle))
            projected = table_basis(changed)
            row = dict(axis=axis_name, degrees=degrees, gaps={})
            for name, baseline in gaps.items():
                if baseline is None:
                    continue
                measured = gap_decomposition(vectors[f'{name}_full'], vectors['stereo_full'],
                                             vectors[f'{name}_common'], vectors['stereo_common'], projected)
                row['gaps'][name] = {key: measured[key] - baseline[key]
                                     for key in ('g_F_cm', 'g_I_cm')}
            rows.append(row)
    return dict(perturbations=rows, native_max_gap_change_cm=max(
        (abs(value) for row in rows for value in row['gaps'].get('native', {}).values()),
        default=None))


def support_estimates(moge: dict, stereo: dict, input_valid: np.ndarray,
                      intrinsics: np.ndarray, masks: list, normal: np.ndarray,
                      scale: float | None) -> dict:
    """Measure the frozen estimators without reference poses or scan geometry."""
    zm, zs = moge['depth_m'].astype(float), stereo['rgb_depth_m'].astype(float)
    vm = moge['valid'] & input_valid & np.isfinite(zm) & (zm > 0)
    vs = stereo['rgb_valid'] & input_valid & np.isfinite(zs) & (zs > 0)
    pm, ps = camera_points(zm, intrinsics), camera_points(zs, intrinsics)
    rays = camera_points(np.ones_like(zm), intrinsics)
    basis = table_basis(normal)
    if any(mask is None for mask in masks):
        return dict(missing_reason='Merged/default episode has no independent observed masks',
                    support=None, ratio_I=None, object_ratios=None, estimates={}, gaps={}, rays={},
                    projection_sensitivity=None)
    full_m, full_s = [mask & vm for mask in masks], [mask & vs for mask in masks]
    common = [mask & vm & vs for mask in masks]
    counts = {name: [int(mask.sum()) for mask in items] for name, items in
              (('mask', masks), ('moge_full', full_m), ('stereo_full', full_s), ('common', common))}
    fractions = {name: [count / size if size else None for count, size in zip(values, counts['mask'])]
                 for name, values in counts.items() if name != 'mask'}
    fractions.update({f'common_over_{name}': [c / f if f else None
                      for c, f in zip(counts['common'], counts[name])]
                      for name in ('moge_full', 'stereo_full')})
    ratios = [float(np.median(zs[mask] / zm[mask])) if mask.sum() >= 300 else None for mask in common]
    ratio = float(np.median(ratios)) if all(r is not None for r in ratios) else None
    factors = dict(native=1., recorded=scale, ratio=ratio)
    vectors, missing = {}, {}
    for support_name, items in (('common', common), ('full', full_s)):
        name = f'stereo_{support_name}'
        vectors[name] = ps[items[1]].mean(0) - ps[items[0]].mean(0) if min(map(np.sum, items)) >= 300 else None
        if vectors[name] is None:
            missing[name] = 'Fewer than 300 supported pixels in at least one object'
    for name, factor in factors.items():
        for support_name, items in (('common', common), ('full', full_m)):
            key = f'{name}_{support_name}'
            vectors[key] = factor * (pm[items[1]].mean(0) - pm[items[0]].mean(0)) if (
                factor is not None and min(map(np.sum, items)) >= 300) else None
            if vectors[key] is None:
                missing[key] = 'Unavailable scale or fewer than 300 supported pixels per object'
    gaps = {name: gap_decomposition(vectors[f'{name}_full'], vectors['stereo_full'],
                                    vectors[f'{name}_common'], vectors['stereo_common'], basis)
            if all(vectors[key] is not None for key in
                   (f'{name}_full', 'stereo_full', f'{name}_common', 'stereo_common')) else None
            for name in factors}
    return dict(missing_reason=None, support=dict(counts=counts, fractions=fractions),
                ratio_I=ratio, object_ratios=ratios,
                moge_full_centres_table_m=[basis @ pm[mask].mean(0) if mask.sum() >= 300 else None
                                          for mask in full_m],
                stereo_support_duplicate=all(np.array_equal(a, b) for a, b in zip(common, full_s)),
                estimates={name: describe_offset(vector, basis) for name, vector in vectors.items()},
                missing=missing, gaps=gaps,
                rays={name: ray_decomposition(pm, ps, rays, common, factor)
                      if factor is not None else None for name, factor in factors.items()},
                projection_sensitivity=projection_sensitivity(normal, vectors, gaps))


def assignment_contrast(scene: dict, observed_slot_zero_xy: np.ndarray,
                        scale: float, stereo_length_cm: float, thresholds: dict) -> dict:
    """Replace only the default pre-separation slot-zero xy while retaining shifts."""
    before = np.asarray(scene['before'], float)[:, :2]
    distances = np.asarray(scene['shifts'], float)
    if np.any(distances != 0):
        raise ValueError('Assignment contrast needs directions for nonzero recorded separation distances')
    shifts = np.zeros_like(before)
    positions = np.asarray(scene['positions'], float)[:, :2]
    reproduced = scale * (before[1] - before[0] + shifts[1] - shifts[0])
    recorded = positions[1] - positions[0]
    mismatch = float(np.max(np.abs(reproduced - recorded)))
    if mismatch > 1e-6:
        raise ValueError(f'S1 offset reproduction failed: {mismatch} m')
    observed = reproduced + scale * (before[0] - observed_slot_zero_xy)
    change = float((np.linalg.norm(observed) - np.linalg.norm(recorded)) * 100)
    denominator = float(np.linalg.norm(recorded) * 100 - stereo_length_cm)
    return dict(recorded_xy_m=recorded, reproduced_xy_m=reproduced,
                observed_xy_m=observed, change_xy_m=observed - recorded,
                length_change_cm=change, separation_term_xy_m=scale * (shifts[1] - shifts[0]),
                recorded_separation_distances_cm=distances,
                reproduction_max_difference_m=mismatch, denominator_cm=denominator,
                change_over_recorded_stereo_gap=change / denominator
                    if abs(denominator) >= thresholds['assignment_min_denominator_cm'] else None,
                material=abs(change) >= thresholds['assignment_min_change_cm'])


def cell_errors(predicted: np.ndarray, target: np.ndarray, cells: np.ndarray) -> dict:
    """Average depth errors within cells and then equally across cells."""
    if not np.isfinite(predicted).all() or np.any(predicted <= 0):
        return dict(mae_cm=float('inf'), absolute_relative=float('inf'),
                    invalid_predictions=int((~np.isfinite(predicted) | (predicted <= 0)).sum()))
    error = np.abs(predicted - target)
    groups = [cells == cell for cell in np.unique(cells)]
    return dict(mae_cm=float(np.mean([error[group].mean() for group in groups]) * 100),
                absolute_relative=float(np.mean([(error[group] / target[group]).mean() for group in groups])),
                invalid_predictions=0)


def table_folds(shape: tuple) -> tuple:
    """Assign the frozen spatial folds and two-pixel cell-edge guards."""
    height, width = shape
    v, u = np.indices(shape)
    row, col = 8 * v // height, 8 * u // width
    top, left = np.ceil(row * height / 8), np.ceil(col * width / 8)
    bottom, right = np.ceil((row + 1) * height / 8), np.ceil((col + 1) * width / 8)
    interior = (v >= top + 2) & (v < bottom - 2) & (u >= left + 2) & (u < right - 2)
    return (row + 2 * col) % 5, row * 8 + col, interior


def table_models(moge: dict, stereo: dict, input_valid: np.ndarray,
                 intrinsics: np.ndarray, labels: np.ndarray,
                 normal: np.ndarray, plane_offset: float, band: float) -> dict:
    """Fit the three depth models and two plane models on spatially withheld table patches."""
    zm, zs = moge['depth_m'].astype(float), stereo['rgb_depth_m'].astype(float)
    valid = moge['valid'] & stereo['rgb_valid'] & input_valid
    valid &= np.isfinite(zm) & np.isfinite(zs) & (zm > 0) & (zs > 0)
    padded = np.pad(labels != 0, 5)
    excluded = np.lib.stride_tricks.sliding_window_view(padded, (11, 11)).any(axis=(-2, -1))
    table = valid & ~excluded & (np.abs(camera_points(zm, intrinsics) @ normal + plane_offset) <= band)
    folds, cells, interior = table_folds(zm.shape)
    selected = table & interior
    counts = [dict(fold=i, train_pixels=int((selected & (folds != i)).sum()),
                   test_pixels=int((selected & (folds == i)).sum()),
                   test_cells=len(np.unique(cells[selected & (folds == i)]))) for i in range(5)]
    ratio_map = np.full(zm.shape, np.nan)
    ratio_map[selected] = zs[selected] / zm[selected]
    result = dict(table_pixels=int(table.sum()), retained_pixels=int(selected.sum()), fold_counts=counts,
                  eligible=all(c['train_pixels'] >= 1000 and c['test_pixels'] >= 200
                               and c['test_cells'] >= 4 for c in counts),
                  table_mask=selected, ratio_map=ratio_map, models={})
    if not result['eligible']:
        result['missing_reason'] = 'At least one spatial fold fails the frozen support requirements'
        return result
    v, u = np.indices(zm.shape)
    spatial = np.stack((np.ones_like(zm), u / zm.shape[1], v / zm.shape[0]), axis=-1)
    for name in ('scale', 'affine_depth', 'affine_inverse', 'plane_moge', 'plane_stereo'):
        rows, residual = [], np.full(zm.shape, np.nan)
        for fold in range(5):
            train, test = selected & (folds != fold), selected & (folds == fold)
            if name.startswith('plane_'):
                target = zm if name == 'plane_moge' else zs
                coefficients = np.linalg.lstsq(spatial[train], 1 / target[train], rcond=None)[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    predicted = 1 / (spatial[test] @ coefficients)
            else:
                inverse = name == 'affine_inverse'
                x, y = (1 / zm[train], 1 / zs[train]) if inverse else (zm[train], zs[train])
                design = x[:, None] if name == 'scale' else np.column_stack((x, np.ones_like(x)))
                coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
                xt = 1 / zm[test] if inverse else zm[test]
                design_test = xt[:, None] if name == 'scale' else np.column_stack((xt, np.ones_like(xt)))
                predicted = design_test @ coefficients
                if inverse:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        predicted = 1 / predicted
                target = zs
            errors = cell_errors(predicted, target[test], cells[test])
            rows.append(dict(fold=fold, coefficients=coefficients, **errors))
            residual[test] = (predicted - target[test]) * 100
        result['models'][name] = dict(folds=rows, residual_cm=residual,
                                     mae_cm=float(np.mean([r['mae_cm'] for r in rows])),
                                     absolute_relative=float(np.mean([r['absolute_relative'] for r in rows])))
    return result


def episode_readings(row: dict, thresholds: dict) -> dict:
    """Apply the frozen descriptive predicates to one episode without reference errors."""
    gap = row['gaps'].get('native')
    eligible = gap is not None and abs(gap['g_F_cm']) >= thresholds['min_gap_cm']
    readings = dict(depth_dominates=bool(eligible and gap['g_I_cm'] * gap['g_F_cm'] > 0
        and abs(gap['support_cm']) <= thresholds['dominance_fraction'] * abs(gap['g_F_cm'])),
        support_dominates=bool(eligible and gap['support_cm'] * gap['g_F_cm'] > 0
        and abs(gap['g_I_cm']) <= thresholds['dominance_fraction'] * abs(gap['g_F_cm'])))
    for name in ('recorded', 'ratio'):
        scaled = row['gaps'].get(name)
        readings[f'{name}_scale_removes_gap'] = bool(gap is not None and scaled is not None
            and abs(gap['g_I_cm']) >= thresholds['min_gap_cm']
            and abs(scaled['g_I_cm']) <= thresholds['scale_max_fraction'] * abs(gap['g_I_cm'])
            and abs(scaled['g_I_cm']) <= thresholds['scale_max_gap_cm'])
    models = row.get('table', {}).get('models', {})
    for name in ('affine_depth', 'affine_inverse'):
        scale, model = models.get('scale'), models.get(name)
        readings[f'{name}_table_correction'] = bool(scale and model and np.isfinite(model['mae_cm'])
            and model['mae_cm'] <= thresholds['table_max_fraction'] * scale['mae_cm']
            and scale['mae_cm'] - model['mae_cm'] >= thresholds['table_min_improvement_cm'])
    sensitivity = row.get('projection_sensitivity')
    change = None if sensitivity is None else sensitivity['native_max_gap_change_cm']
    readings['projection_sensitive'] = change is not None and change >= thresholds['projection_min_change_cm']
    return readings


def aggregate_readings(rows: list, clean_episodes: tuple, thresholds: dict) -> dict:
    """Aggregate cached episode readings with the fixed clean-roster denominator."""
    clean = [row['readings'] for row in rows if row['episode'] in clean_episodes]
    labels = ('depth_dominates', 'support_dominates', 'recorded_scale_removes_gap',
              'ratio_scale_removes_gap', 'affine_depth_table_correction',
              'affine_inverse_table_correction', 'projection_sensitive')
    result = {}
    for name in labels:
        count = sum(bool(readings[name]) for readings in clean)
        required = thresholds['projection_required_count' if name == 'projection_sensitive' else 'required_count']
        result[name] = dict(count=count, denominator=len(clean_episodes), required=required,
                            qualified=count >= required)
    return result


def check_grid(moge: dict, stereo: dict, grid: dict, labels: np.ndarray) -> dict:
    """Check the common camera grid and quantify the unused model-K discrepancy."""
    k = grid['input_intrinsics_pixels']
    if not all(a.shape == labels.shape for a in (moge['depth_m'], moge['valid'],
               stereo['rgb_depth_m'], stereo['rgb_valid'], grid['input_valid'])):
        raise ValueError('Depth, validity and labels use different grids')
    if not all(np.allclose(k, other, atol=1e-5, rtol=0) for other in
               (moge['input_intrinsics_pixels'], stereo['rgb_intrinsics_pixels'])):
        raise ValueError('Input/stereo intrinsics differ beyond the protocol tolerance')
    difference = moge['model_intrinsics_pixels'].astype(float) - k
    return dict(model_minus_input_K=difference,
                model_input_K_max_difference_px=float(np.max(np.abs(difference))))


def reference_columns(estimates: dict, references: dict, ref_rotation_zero: np.ndarray,
                      anchor_rotation: np.ndarray | None) -> dict:
    """Attach reference lengths and conditional aligned errors to frozen estimators."""
    alignment = None if anchor_rotation is None else anchor_rotation @ ref_rotation_zero.T
    reference_rows, errors = {}, {}
    for name, vector in references.items():
        aligned = None if alignment is None else alignment @ vector
        reference_rows[name] = dict(world_vector_m=vector,
            world_horizontal_length_cm=float(np.linalg.norm(vector[:2]) * 100),
            length_3d_cm=float(np.linalg.norm(vector) * 100), aligned_table_vector_m=aligned,
            aligned_horizontal_length_cm=None if aligned is None else float(np.linalg.norm(aligned[:2]) * 100))
    for name, estimate in estimates.items():
        errors[name] = None if estimate is None else {}
        if estimate is None:
            continue
        for ref_name, ref in reference_rows.items():
            horizontal = estimate['length_horizontal_cm'] - ref['world_horizontal_length_cm']
            length_3d = estimate['length_3d_cm'] - ref['length_3d_cm']
            vector = None if alignment is None else estimate['table_m'] - ref['aligned_table_vector_m']
            errors[name][ref_name] = dict(horizontal_length_signed_cm=horizontal,
                horizontal_length_absolute_cm=abs(horizontal), length_3d_signed_cm=length_3d,
                length_3d_absolute_cm=abs(length_3d), aligned_vector_error_m=vector,
                aligned_vector_error_3d_cm=None if vector is None else float(np.linalg.norm(vector) * 100))
    return dict(references=reference_rows, reference_errors=errors,
        centre_minus_origin_horizontal_length_cm=reference_rows['M_c']['world_horizontal_length_cm']
                                                - reference_rows['M_o']['world_horizontal_length_cm'])


def summarize_rows(rows: list, clean_episodes: tuple, validation_episodes: tuple,
                   thresholds: dict) -> dict:
    """Summarize continuous quantities with explicit counts and fixed-subset readings."""
    groups = dict(clean=clean_episodes, spent_validation=validation_episodes)
    means = {}
    for group, episodes in groups.items():
        selected = [row for row in rows if row['episode'] in episodes]
        means[group] = dict(expected_count=len(episodes), gaps={}, table_models={})
        for name in ('native', 'recorded', 'ratio'):
            gaps = [row['gaps'][name] for row in selected if row['gaps'].get(name) is not None]
            means[group]['gaps'][name] = dict(count=len(gaps), **{
                key: float(np.mean([g[key] for g in gaps])) if gaps else None
                for key in ('g_F_cm', 'g_I_cm', 'support_cm', 'depth_cm')})
        for name in ('scale', 'affine_depth', 'affine_inverse', 'plane_moge', 'plane_stereo'):
            models = [row['table']['models'][name] for row in selected if name in row['table']['models']]
            means[group]['table_models'][name] = dict(count=len(models),
                mae_cm=float(np.mean([m['mae_cm'] for m in models])) if models else None,
                absolute_relative=float(np.mean([m['absolute_relative'] for m in models])) if models else None)
    assignment = next((row.get('assignment') for row in rows if row['episode'] == 12), None)
    readings = aggregate_readings(rows, clean_episodes, thresholds)
    readings['material_assignment_effect'] = dict(qualified=bool(assignment and assignment['material']),
                                                 denominator=1, required=1)
    validation = [row for row in rows if row['episode'] in groups['spent_validation']]
    validation_readings = {name: dict(denominator=len(validation_episodes),
        count=sum(bool(row['readings'][name]) for row in validation),
        episodes={str(row['episode']): bool(row['readings'][name]) for row in validation})
        for name in readings if name != 'material_assignment_effect'}
    return dict(means=means, readings=readings, assignment_episode_12=assignment,
                spent_validation_readings=validation_readings)
