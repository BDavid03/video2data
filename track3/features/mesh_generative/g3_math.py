"""Pure frame-mask measurements and the frozen G3 gate calculations."""

import numpy as np
from scipy import ndimage


def largest_component(mask):
    """Keep the largest eight-connected foreground component, first label on ties."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())[1:]
    return labels == int(sizes.argmax() + 1)


def mask_checks(mask, remap_valid, observed_valid, rule):
    """Count inner eight-neighbour boundary pixels near borders or invalid remap pixels."""
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    y, x = np.indices(mask.shape)
    border_distance = np.minimum.reduce((x, y, mask.shape[1]-1-x, mask.shape[0]-1-y))
    invalid_distance = (ndimage.distance_transform_edt(remap_valid) if (~remap_valid).any()
                        else np.full(mask.shape, np.inf))
    near = np.minimum(border_distance, invalid_distance) <= rule['boundary_margin_pixels']
    total = int(boundary.sum())
    near_count = int((boundary & near).sum())
    area, points = int(mask.sum()), int((mask & observed_valid).sum())
    fraction = near_count / total if total else 0.0
    checks = dict(area=rule['minimum_pixels'] <= area <= rule['maximum_pixels'],
                  boundary=fraction <= rule['maximum_border_fraction'],
                  observed_depth=points >= rule['minimum_observed_points'])
    reasons = [name for name, passed in checks.items() if not passed]
    return dict(mask_pixels=area, boundary_pixels=total, near_invalid_boundary_pixels=near_count,
                border_fraction=fraction, observed_points=points, checks=checks,
                accepted=not reasons, reason='mask accepted' if not reasons else 'mask rejection: '+', '.join(reasons))


def iou(mask, reference):
    """Measure binary intersection over union, with zero for an empty union."""
    union = int((mask | reference).sum())
    return float((mask & reference).sum() / union) if union else 0.0


def gates(object_mean, pair_mean, changes, limits, has_new_pair):
    """Apply strict mean improvements, the frozen regression cap and a nonempty change."""
    checks = dict(new_pair=has_new_pair, equal_object_mean=object_mean < limits['object_mean_cm'],
                  pair_weighted_mean=pair_mean < limits['pair_mean_cm'],
                  no_object_regresses=all(change <= 0.15 for change in changes))
    return dict(gates=checks, verdict='PASS' if all(checks.values()) else 'FAIL')
