"""Pure basin matching, triangle silhouettes and fixed S1 diagnostic scores."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from ._registration import geodesic_deg, up_error_deg, vector_error_deg


def representatives(rotations: np.ndarray, losses: np.ndarray, separation: float,
                    relative_margin: float | None = None) -> np.ndarray:
    """Select loss-ranked representatives with inclusive angular membership."""
    remaining = np.argsort(losses, kind='stable')
    if relative_margin is not None:
        remaining = remaining[losses[remaining] <= losses.min() * (1 + relative_margin)]
    chosen = []
    while len(remaining):
        index = int(remaining[0])
        chosen.append(index)
        remaining = remaining[geodesic_deg(rotations[remaining], rotations[index]) > separation]
    return np.asarray(chosen, dtype=int)


def match(reference: np.ndarray, other: np.ndarray, maximum_deg: float,
          exhaustive: bool = True) -> tuple[list, str | None]:
    """Require unambiguous, distinct matches, optionally exhausting both sets."""
    if not len(reference) or not len(other) or (exhaustive and len(reference) != len(other)):
        return [], 'missing_or_extra_basin'
    eligible = geodesic_deg(reference[:, None], other[None, :]) <= maximum_deg
    if not np.all(eligible.sum(axis=1) == 1):
        return [], 'missing_or_ambiguous_basin_match'
    indices = eligible.argmax(axis=1)
    if len(np.unique(indices)) != len(indices):
        return [], 'non_bijective_basin_match'
    return indices.tolist(), None


def candidate_sets(searches: list, thresholds: dict, control: bool) -> tuple[list, list, list]:
    """Build near-tied sets or a best/farthest control pair and repeated matches."""
    separation = thresholds['basin_separation_deg']
    groups = [representatives(s['rotations'], s['losses_m'], separation,
              None if control else thresholds['relative_loss_margin']) for s in searches]
    if control:
        primary = groups[0]
        angles = geodesic_deg(searches[0]['rotations'][primary], searches[0]['rotations'][primary[0]])
        order = np.lexsort((primary, searches[0]['losses_m'][primary], -angles))
        groups[0] = np.asarray([primary[0], primary[order[0]]]) if len(primary) > 1 else primary
    mappings = [list(range(len(groups[0])))]
    reasons = ['fewer_than_two_candidates'] if len(groups[0]) < 2 else []
    reference = searches[0]['rotations'][groups[0]]
    for number in range(1, len(searches)):
        mapping, reason = match(reference, searches[number]['rotations'][groups[number]],
                                thresholds['matching_max_deg'], exhaustive=not control)
        mappings.append(mapping)
        if reason:
            reasons.append(f'search_{number}:{reason}')
        if control and not reason:
            groups[number] = groups[number][mapping]
            mappings[number] = list(range(len(mapping)))
        elif control:
            groups[number] = np.array([], dtype=int)
    return groups, mappings, reasons


def touches_border(mask: np.ndarray) -> bool:
    """Return whether a binary image occupies any outermost image pixel."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def raster(vertices: np.ndarray, faces: np.ndarray, rotation: np.ndarray,
           translation: np.ndarray, intrinsics: np.ndarray, shape: tuple) -> tuple:
    """Project scan triangles and fill each separately into their binary union."""
    camera = vertices @ rotation.T + translation
    result = np.zeros(shape, np.uint8)
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        return result.astype(bool), 'nonfinite_or_nonpositive_vertex_depth'
    projected = camera @ intrinsics.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:])
    if not np.isfinite(pixels).all() or np.abs(pixels).max() > np.iinfo(np.int32).max:
        return result.astype(bool), 'invalid_projected_coordinates'
    pixels = pixels.astype(np.int32)
    for triangle in pixels[faces]:
        cv2.fillConvexPoly(result, triangle, 1, lineType=cv2.LINE_8)
    reason = 'empty_render' if not result.any() else 'render_touches_border' if touches_border(result) else None
    return result.astype(bool), reason


def mask_variants(mask: np.ndarray, kernel_pixels: int) -> list:
    """Return original, once-eroded and once-dilated binary object regions."""
    binary = mask.astype(np.uint8)
    kernel = np.ones((kernel_pixels, kernel_pixels), np.uint8)
    return [mask.astype(bool), cv2.erode(binary, kernel, iterations=1).astype(bool),
            cv2.dilate(binary, kernel, iterations=1).astype(bool)]


def outer_contour(mask: np.ndarray) -> np.ndarray:
    """Rasterise external boundaries, retaining all disconnected components."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(result, contours, -1, 1, thickness=1, lineType=cv2.LINE_8)
    return result.astype(bool)


def scores(render: np.ndarray, variants: list, cap_px: float) -> tuple[list, list]:
    """Compute IoU and symmetric capped outer-contour chamfer in pixels."""
    contour = outer_contour(render)
    if not contour.any():
        return [None] * len(variants), [None] * len(variants)
    distance = distance_transform_edt(~contour)
    ious, chamfers = [], []
    for mask in variants:
        target = outer_contour(mask)
        ious.append(float(np.count_nonzero(render & mask) / np.count_nonzero(render | mask)))
        chamfers.append(None if not target.any() else float(.5 * (
            np.minimum(distance[target], cap_px).mean()
            + np.minimum(distance_transform_edt(~target)[contour], cap_px).mean())))
    return ious, chamfers


def choose(ious: np.ndarray, margin: float) -> dict:
    """Select one identity only if it wins every search and variant by the margin."""
    if ious.shape[1] < 2 or not np.isfinite(ious).all():
        return dict(selected=None, minimum_margin=None, reason='incomplete_scores')
    winners = []
    margins = []
    for candidate in range(ious.shape[1]):
        others = np.delete(ious, candidate, axis=1)
        minimum = float(np.min(ious[:, candidate, :] - np.max(others, axis=1)))
        margins.append(minimum)
        if minimum >= margin:
            winners.append(candidate)
    if len(winners) == 1:
        selected = winners[0]
        return dict(selected=selected, minimum_margin=margins[selected], reason=None)
    return dict(selected=None, minimum_margin=max(margins), reason='no_common_winner_with_margin')


def evaluate(searches: list, vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray,
             intrinsics: np.ndarray, thresholds: dict, control: bool) -> tuple:
    """Measure every saved candidate and apply the fixed matching and margin rules."""
    groups, mappings, reasons = candidate_sets(searches, thresholds, control)
    variants = mask_variants(mask, thresholds['kernel_pixels'])
    if not mask.any():
        reasons.append('empty_mask')
    if touches_border(mask):
        reasons.append('mask_touches_border')
    if any(not m.any() for m in variants):
        reasons.append('empty_mask_variant')
    mask_invalid = bool(reasons and any(r in reasons for r in
                        ('empty_mask', 'mask_touches_border', 'empty_mask_variant')))
    rows, renders, matched_scores, search_decisions = [], {}, [], []
    for number, (search, indices) in enumerate(zip(searches, groups)):
        candidates = []
        for local, index in enumerate(indices):
            render, reason = raster(vertices, faces, search['rotations'][index],
                                    search['translations'][index], intrinsics, mask.shape)
            ious, chamfer = scores(render, variants, thresholds['chamfer_cap_px'])
            if reason:
                reasons.append(f'search_{number}:index_{index}:{reason}')
            candidates.append(dict(index=int(index), loss_ratio=float(search['losses_m'][index]
                / search['losses_m'].min()), iou=ious, contour_chamfer_px=chamfer, invalid=reason))
            renders[f'search_{number}_candidate_{local}'] = render
        rows.append(candidates)
        if mappings[number] and not mask_invalid and not any(c['invalid'] for c in candidates):
            values = np.array([candidates[i]['iou'] for i in mappings[number]], dtype=float)
            matched_scores.append(values)
            search_decisions.append(choose(values[None], thresholds['iou_margin']))
        else:
            search_decisions.append(dict(selected=None, minimum_margin=None, reason='invalid_or_unmatched_candidates'))
    decision = dict(selected=None, minimum_margin=None, reason='; '.join(reasons))
    if not reasons and len(matched_scores) == 4:
        decision = choose(np.array(matched_scores), thresholds['iou_margin'])
    reasons = list(dict.fromkeys(reasons + ([decision['reason']] if decision['reason'] and not reasons else [])))
    result = dict(searches=rows, mappings=mappings, decision=decision, abstain_reasons=reasons,
                  search_decisions=search_decisions, variant_order=['original', 'eroded', 'dilated'],
                  mask_pixels=int(mask.sum()), control=control)
    return result, dict(mask=mask, **renders)


def label_errors(rotations: np.ndarray, normal: np.ndarray, reference: np.ndarray,
                 canonical: np.ndarray | None = None, partner: np.ndarray | None = None,
                 partner_reference: np.ndarray | None = None) -> dict:
    """Measure saved candidate orientations against frame-zero body-frame labels."""
    if partner is not None:
        target = reference.T @ partner_reference
        return dict(C_error_deg=geodesic_deg(np.swapaxes(rotations, -1, -2) @ partner, target).tolist())
    return dict(up_error_deg=[up_error_deg(r, normal, reference) for r in rotations],
                canonical_up_error_deg=[vector_error_deg(r.T @ normal, canonical) for r in rotations])
