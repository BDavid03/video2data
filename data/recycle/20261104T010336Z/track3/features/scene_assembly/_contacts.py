"""Scene-owned contact queries in the pinned upstream MANO hand-link order."""

# Read-only reference: robotic_grounding/retarget/params.py at aee95f25,
# MANO_HAND_LINKS. These are links, not the 21 MANO joint-name slots.
LINKS = (
    ("link_palm", (0, 1, 5, 9, 13, 17)),
    ("link_thumb1", (1, 2)), ("link_thumb2", (2, 3)), ("link_thumb3", (3, 4)),
    ("link_index1", (5, 6)), ("link_index2", (6, 7)), ("link_index3", (7, 8)),
    ("link_middle1", (9, 10)), ("link_middle2", (10, 11)), ("link_middle3", (11, 12)),
    ("link_ring1", (13, 14)), ("link_ring2", (14, 15)), ("link_ring3", (15, 16)),
    ("link_pinky1", (17, 18)), ("link_pinky2", (18, 19)), ("link_pinky3", (19, 20)),
)


def contract():
    return {"schema_version": 1, "joint_order": "MANO21", "body_ids": "one_based_object_slot_zero_absent",
            "links": [{"name": name, "joint_indices": list(indices)} for name, indices in LINKS],
            "method": "nearest_object_surface_to_sampled_joint_links",
            "finger_samples_per_segment": 5,
            "palm_samples": "six defining joints, fifteen pair midpoints and their centroid",
            "scope": "Geometric proximity proxy; neither observed contact nor full hand-surface collision"}


def queries(joints, np):
    """Return bounded query points and each link's slice in the query axis.

    Include original joints first so the same triangle traversal supplies the
    existing 21-joint penetration diagnostic. Link samples follow independently.
    """
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3) or not np.isfinite(joints).all():
        raise ValueError("Contact queries require complete finite MANO21 hands")
    groups, blocks, offset = [], [joints], 21
    for name, indices in LINKS:
        if name == "link_palm":
            palm = joints[:, :, indices, :]
            midpoints = [(palm[:, :, a] + palm[:, :, b]) * 0.5
                         for a in range(len(indices)) for b in range(a + 1, len(indices))]
            points = np.concatenate((palm, np.stack(midpoints, axis=2), palm.mean(axis=2, keepdims=True)), axis=2)
        else:
            fraction = np.linspace(0, 1, 5)[None, None, :, None]
            a, b = (joints[:, :, index:index + 1] for index in indices)
            points = a * (1 - fraction) + b * fraction
        blocks.append(points)
        groups.append((offset, offset + points.shape[2]))
        offset += points.shape[2]
    return np.concatenate(blocks, axis=2), groups


def reduce_links(distance, position, normal, groups, np):
    """Keep the nearest sampled point's surface location and normal per link."""
    distances, positions, normals = [], [], []
    for start, stop in groups:
        nearest = distance[:, :, start:stop].argmin(axis=2)
        distances.append(np.take_along_axis(distance[:, :, start:stop], nearest[..., None], axis=2)[..., 0])
        positions.append(np.take_along_axis(position[:, :, start:stop], nearest[..., None, None], axis=2)[..., 0, :])
        normals.append(np.take_along_axis(normal[:, :, start:stop], nearest[..., None, None], axis=2)[..., 0, :])
    return np.stack(distances, axis=2), np.stack(positions, axis=2), np.stack(normals, axis=2)
