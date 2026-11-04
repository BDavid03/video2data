"""Pure frame-zero overlap, stereo centroid and placement measurements."""
import numpy as np

from track3.features.mesh_generative.g3_math import iou
from ._shrink import footprint_clearance


def box_iou(boxes: list) -> float | None:
    """Measure detector-box overlap when both detections exist."""
    if any(box is None for box in boxes):
        return None
    a, b = np.asarray(boxes, float)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    areas = np.maximum(0, np.asarray(boxes)[:, 2:] - np.asarray(boxes)[:, :2]).prod(axis=1)
    return float(intersection / (areas.sum() - intersection))


def resolve_overlap(masks: list) -> tuple:
    """Assign shared pixels to the smaller original mask, slot zero on ties."""
    result = [mask.copy() for mask in masks]
    overlap = result[0] & result[1]
    owner = 0 if result[0].sum() <= result[1].sum() else 1
    result[1 - owner] &= ~overlap
    return result, dict(overlap_pixels=int(overlap.sum()), overlap_owner=owner,
                        original_areas_pixels=[int(mask.sum()) for mask in masks])


def mask_reading(mask: np.ndarray, valid: np.ndarray, detected: bool,
                 reference: np.ndarray | None) -> dict:
    """Measure support and the fixed per-object admission predicate."""
    count = int((mask & valid).sum())
    reasons = ([] if detected else ['no_detector_box'])
    if count < 300:
        reasons.append('fewer_than_300_valid_stereo_pixels')
    return dict(area_pixels=int(mask.sum()), area_fraction=float(mask.mean()),
                valid_stereo_pixels=count, admitted=not reasons, reasons=reasons,
                control_cluster_iou=None if reference is None else iou(mask, reference))


def centroid_offset(points: np.ndarray, valid: np.ndarray, masks: list,
                    basis: np.ndarray) -> dict:
    """Keep all three coordinates of the two visible-surface mean displacement."""
    centres = [points[mask & valid].mean(0) if (mask & valid).any() else None for mask in masks]
    displacement = None if any(c is None for c in centres) else basis @ (centres[1] - centres[0])
    return dict(centroids_camera_m=[None if c is None else c.tolist() for c in centres],
                d_table_m=None if displacement is None else displacement.tolist(),
                horizontal_length_m=None if displacement is None else float(np.linalg.norm(displacement[:2])),
                height_m=None if displacement is None else float(displacement[2]))


def placement_reading(scene: dict, displacement: list | None, admitted: bool) -> dict:
    """Check the frozen one-centimetre footprint rule without expansion."""
    original = np.asarray(scene['positions'], float)
    if displacement is None:
        return dict(clearance_m=None, clear=None, classification='unavailable',
                    positions_m=original.tolist(), reason='missing_centroid_offset')
    proposed = original.copy()
    proposed[1] = proposed[0] + displacement
    clearance = footprint_clearance(proposed, np.asarray(scene['dimensions']), np.asarray(scene['quaternions']))
    clear = clearance >= .01
    return dict(clearance_m=clearance, clear=clear, classification='clear' if clear else 'contact',
                positions_m=(proposed if clear and admitted else original).tolist(),
                reason=('stage_a_failed_keep_incumbent' if not admitted else
                        'raw_offset_clears' if clear else 'contact_keep_incumbent'))
