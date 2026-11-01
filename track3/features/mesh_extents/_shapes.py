"""Closed procedural shapes using only table-relative video extents."""

import numpy as np
import trimesh


def cylinder(extents) -> trimesh.Trimesh:
    """Make a capped vertical cylinder with the measured elliptical footprint."""
    mesh = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=64)
    mesh.apply_scale(extents)
    return mesh


def vessel_with_handle(extents, height_fraction: float = 0.6) -> trimesh.Trimesh:
    """Put the long-axis excess into a closed bar overlapping a circular body."""
    long, short, height = np.asarray(extents, dtype=float)
    body = cylinder((short, short, height))
    excess = max(0.0, long - short)
    if excess == 0:
        return body
    overlap = 0.05 * short
    handle = trimesh.creation.box(extents=(excess + overlap, 0.15 * short, height_fraction * height))
    handle.apply_translation((short / 2 + (excess - overlap) / 2, 0, 0))
    mesh = trimesh.util.concatenate((body, handle))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def head_and_handle(extents, head_fraction: float = 0.35) -> trimesh.Trimesh:
    """Join a full-width box head to a narrow box handle along the long axis."""
    long, short, height = np.asarray(extents, dtype=float)
    head_length = head_fraction * long
    head = trimesh.creation.box(extents=(head_length, short, height))
    head.apply_translation((-long / 2 + head_length / 2, 0, 0))
    overlap = 0.02 * long
    handle_length = long - head_length + overlap
    handle = trimesh.creation.box(extents=(handle_length, 0.2 * short, height))
    handle.apply_translation((long / 2 - handle_length / 2, 0, 0))
    return trimesh.util.concatenate((head, handle))
