"""Video-only motion gates and separate-hand translation attachment."""

import cv2
import numpy as np

from .camera import apply
from ._kit_score import first_run

SETTINGS = dict(radius_m=.05, consecutive_frames=5, contact_fraction=.05,
                contact_dilation=11, palm_fraction=.5, minimum_palm_points=30,
                median_window=5, minimum_grasp_motion_m=.01, motion_disagreement_m=.03)


def resize_maps(left_maps, raw_shape, source_wh):
    width, height = source_wh
    return ((left_maps[0] + .5) * raw_shape[1] / width - .5,
            (left_maps[1] + .5) * raw_shape[0] / height - .5)


def rectify_slots(raw, ids, maps):
    """Preserve SAM2 IDs with the existing half-pixel nearest-neighbour mapping."""
    if not np.array_equal(ids, [0, 1, 2]):
        raise ValueError('Expected object0 and separate SAM2 hand slots1/2')
    return np.stack([cv2.remap(mask.astype(np.uint8), *maps, cv2.INTER_NEAREST).astype(bool) for mask in raw])


def palm_observations(frame, slots, world_camera):
    """Measure a stereo centre inside each thick hand-mask region separately."""
    obj = frame['object_mask'].astype(bool)
    near = cv2.dilate(obj.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    finite = frame['depth_valid'] & np.isfinite(frame['xyz']).all(2) & (frame['xyz'][..., 2] > .05)
    centres = np.full((2, 3), np.nan); counts = np.zeros(2, int); contact = np.zeros(2)
    for hand, mask in enumerate(slots):
        denominator = min(int(obj.sum()), int(mask.sum()))
        contact[hand] = (near & mask).sum() / denominator if denominator else 0.
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        palm = mask & (distance >= .5 * distance.max()) & ~obj
        points = frame['xyz'][palm & finite].astype(float)
        if len(points):
            depth = np.median(points[:, 2])
            sigma = max(1.4826 * np.median(abs(points[:, 2] - depth)), .001)
            points = points[abs(points[:, 2] - depth) <= 2 * sigma]
        counts[hand] = len(points)
        if len(points) >= SETTINGS['minimum_palm_points']:
            centres[hand] = apply(world_camera, np.median(points, axis=0)[None])[0]
    return centres, counts, contact


def smooth_palms(centres):
    """Use supported samples in a causal five-frame window without filling missing frames."""
    result = np.full_like(centres, np.nan)
    for frame in range(len(centres)):
        for hand in range(2):
            if np.isfinite(centres[frame, hand]).all():
                window = centres[max(0, frame - 4):frame + 1, hand]
                result[frame, hand] = np.median(window[np.isfinite(window).all(1)], axis=0)
    return result


def motion_gate(position, contact):
    """Declare carried only after a sustained excursion with simultaneous hand contact."""
    radius = np.linalg.norm(position - position[0], axis=1)
    qualifies = (radius > SETTINGS['radius_m']) & (contact.max(1) >= SETTINGS['contact_fraction'])
    trigger = first_run(qualifies, SETTINGS['consecutive_frames'])
    return dict(carried=trigger is not None, trigger_frame=trigger, maximum_radius_m=float(radius.max()))


def hand_attachment(position, initial, palms, contact, carried):
    """Attach one confirmed hand, hold on missing depth, and release with hysteresis."""
    count = len(position); k = SETTINGS['consecutive_frames']; threshold = SETTINGS['contact_fraction']
    output = np.repeat(np.asarray(initial)[None], count, axis=0)
    selected = np.zeros(count, np.int8); states = np.full(count, 'supported', dtype='<U12')
    events = []; active = None; armed = True; clear = 0; losses = [0, 0, 0]
    anchor_frame = None; anchor_position = None; last_release = -1
    supported = np.isfinite(palms).all(2)
    if not carried:
        return output, selected, states, events
    for frame in range(count):
        if frame:
            output[frame] = output[frame - 1]
        clear = clear + 1 if np.all(contact[frame] < threshold) else 0
        if active is not None:
            losses[0] = losses[0] + 1 if contact[frame, active] < threshold else 0
            losses[1] = losses[1] + 1 if not supported[frame, active] else 0
            previous = frame - k + 1
            mismatch = (previous >= 0 and supported[frame, active] and supported[previous, active] and
                np.linalg.norm((position[frame] - position[previous]) -
                               (palms[frame, active] - palms[previous, active])) > SETTINGS['motion_disagreement_m'])
            losses[2] = losses[2] + 1 if mismatch else 0
            if max(losses) >= k:
                reason = ['contact_lost', 'palm_depth_missing', 'motion_disagreement'][int(np.argmax(losses))]
                events.append(dict(kind='release', frame=frame, hand_slot=active + 1, reason=reason))
                active = None; armed = False; last_release = frame
                states[frame] = 'released'
            else:
                selected[frame] = active + 1; states[frame] = 'carried'
                if supported[frame, active]:
                    output[frame] = anchor_position + palms[frame, active] - palms[anchor_frame, active]
        else:
            if last_release >= 0:
                states[frame] = 'released'
            if clear >= k:
                armed = True
            start = frame - k + 1
            if not armed or start <= last_release or start < 0:
                continue
            displacement = position[frame] - position[start]
            if np.linalg.norm(displacement) < SETTINGS['minimum_grasp_motion_m']:
                continue
            candidates = []
            for hand in range(2):
                window = slice(start, frame + 1)
                if (contact[window, hand] >= threshold).all() and supported[window, hand].all():
                    disagreement = np.linalg.norm(displacement - (palms[frame, hand] - palms[start, hand]))
                    if disagreement <= SETTINGS['motion_disagreement_m']:
                        candidates.append((float(contact[window, hand].mean()), -hand))
            if candidates:
                _, negative_hand = max(candidates); active = -negative_hand
                anchor_frame = start; anchor_position = position[start].copy(); losses = [0, 0, 0]
                output[start:frame + 1] = anchor_position + palms[start:frame + 1, active] - palms[start, active]
                selected[start:frame + 1] = active + 1; states[start:frame + 1] = 'carried'
                events.append(dict(kind='grasp', frame=start, confirmed_frame=frame, hand_slot=active + 1))
    return output, selected, states, events
