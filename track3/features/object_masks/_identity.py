"""Mask-feature box association; named identities never come from left/right sorting."""

from itertools import product


def iou(a, b, np):
    overlap = np.maximum(0, np.minimum(a[2:4], b[2:4]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[2:4] - a[:2]) + np.prod(b[2:4] - b[:2]) - intersection)
    return intersection / union if union > 0 else 0.0


def suppress(boxes, shape, request, np):
    kept = []
    for box in boxes[np.argsort(-boxes[:, 4], kind="stable")]:
        box = box.copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, shape[1])
        box[[1, 3]] = np.clip(box[[1, 3]], 0, shape[0])
        if box[4] < request.minimum_box_score or (box[2:4] <= box[:2]).any():
            continue
        if any(iou(box, old, np) >= request.nms_iou for old in kept):
            continue
        kept.append(box)
        if len(kept) == request.maximum_boxes_per_prompt:
            break
    return np.asarray(kept, dtype=np.float64).reshape(-1, 5)


def assign(boxes, slots, previous, index, shape, anonymous, request, np):
    if len(slots) > 4:
        raise ValueError("At most four instances may share one detection prompt")
    history = {slot: previous.get(slot) for slot in slots}
    history = {slot: value if value is not None and index - value["frame"] <= request.maximum_identity_gap else None
               for slot, value in history.items()}
    if anonymous and not any(history.values()):
        # These are anonymous track slots, never anatomical left/right identities.
        selected = sorted(range(min(len(boxes), len(slots))), key=lambda i: boxes[i, 0])
        return {slot: boxes[i] for slot, i in zip(slots, selected)}
    candidates = []
    diagonal = float(np.hypot(*shape))
    for choices in product([None, *range(len(boxes))], repeat=len(slots)):
        ids = [value for value in choices if value is not None]
        if len(set(ids)) != len(ids):
            continue
        score, allowed = 0.0, True
        for slot, choice in zip(slots, choices):
            if choice is None:
                continue
            box, old = boxes[choice], history[slot]
            if old is None:
                if len(slots) > 1 and not anonymous:
                    allowed = False
                    break
                score += float(box[4])
            else:
                distance = float(np.linalg.norm((box[:2] + box[2:4] - old["box"][:2] - old["box"][2:4]) / 2) / diagonal)
                if distance > request.maximum_center_distance:
                    allowed = False
                    break
                score += float(box[4]) + iou(box, old["box"], np) - distance
        if allowed:
            candidates.append((score, choices))
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, best = candidates[0]
    near = [choices for value, choices in candidates if score - value < request.assignment_margin]
    return {slot: boxes[choice] for position, (slot, choice) in enumerate(zip(slots, best))
            if choice is not None and all(alternative[position] == choice for alternative in near)}
