"""Pure mask measurements, hand-box assignment and direction selection."""
import numpy as np


def smallest_box_index(boxes, scores, threshold):
    """Index the smallest-area xyxy box strictly above the fixed detector threshold."""
    candidates = np.flatnonzero(np.asarray(scores) > threshold)
    if not len(candidates):
        return None
    boxes = np.asarray(boxes, dtype=float)
    areas = np.maximum(0, boxes[:, 2:4] - boxes[:, :2]).prod(axis=1)
    return int(candidates[np.argmin(areas[candidates])])


def box_iou(a, b):
    """Intersection over union for two xyxy boxes."""
    intersection = np.maximum(0, np.minimum(a[2:4], b[2:4]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:4] - a[:2]).prod() + np.maximum(0, b[2:4] - b[:2]).prod() - intersection
    return float(intersection / union) if union else 0.0


def hand_boxes(boxes, scores, previous, maximum=2, nms_iou=0.7):
    """NMS then assign two anonymous hand slots by minimum box-center displacement."""
    from itertools import permutations
    kept = []
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        box = np.asarray(boxes[index], dtype=float)
        if not any(box_iou(box, other) > nms_iou for other in kept):
            kept.append(box)
        if len(kept) == maximum:
            break
    if not kept:
        return {}
    center = lambda box: (np.asarray(box)[:2] + np.asarray(box)[2:]) / 2
    if not previous:
        return {slot + 1: box.tolist() for slot, box in enumerate(sorted(kept, key=lambda b: center(b)[0]))}
    choices = list(permutations(range(1, maximum + 1), len(kept)))
    slots = min(choices, key=lambda assignment: sum(
        np.linalg.norm(center(box) - center(previous[slot])) if slot in previous else 1e6
        for box, slot in zip(kept, assignment)))
    return {slot: box.tolist() for box, slot in zip(kept, slots)}


def choose_directions(forward, reverse, index, anchors):
    """Use the nearer hand seed's direction; preserve forward-only pan and measure IoU."""
    selected = forward.copy()
    directions = np.zeros(len(forward), dtype=np.uint8)
    agreement = np.full(len(forward), -1.0, dtype=np.float32)
    for slot, (a, b) in enumerate(zip(forward, reverse)):
        union = np.count_nonzero(a | b)
        agreement[slot] = np.count_nonzero(a & b) / union if union else 1.0
        if slot == 0 or not anchors.get(str(slot)):
            continue
        left = min((index - x for x in anchors[str(slot)] if x <= index), default=float('inf'))
        right = min((x - index for x in anchors[str(slot)] if x >= index), default=float('inf'))
        if right < left:
            selected[slot] = b
            directions[slot] = 1
    return selected, directions, agreement


def track_statistics(areas, timestamps):
    """Summarize nonempty-mask coverage and missing runs, without claiming mask accuracy."""
    areas = np.asarray(areas, dtype=np.int64)
    missing = areas == 0
    padded = np.r_[False, missing, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    longest = int(np.max(stops - starts)) if len(starts) else 0
    location = int(np.argmax(stops - starts)) if len(starts) else None
    interval = None if location is None else [int(starts[location]), int(stops[location] - 1)]
    return dict(nonempty_frames=int((~missing).sum()), coverage_fraction=float((~missing).mean()),
                longest_missing_frames=longest, longest_missing_interval=interval,
                longest_missing_seconds=float(longest * np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0,
                area_pixels=dict(min=int(areas.min()), max=int(areas.max()), median=float(np.median(areas)),
                                 mean=float(areas.mean()), p05=float(np.percentile(areas, 5)), p95=float(np.percentile(areas, 95))))
