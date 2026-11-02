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


def _attach_vessel_handle(body, extents, rim_flat: bool) -> trimesh.Trimesh:
    """Attach a closed bar with R2 overlap and center the combined bounds."""
    long, short, height = np.asarray(extents, dtype=float)
    excess = max(0.0, long - short)
    if excess == 0:
        return body
    overlap = 0.05 * short
    bar_height = (0.15 if rim_flat else 0.6) * height
    bar = trimesh.creation.box(extents=(excess + overlap, 0.15 * short, bar_height))
    bar.apply_translation((short / 2 + (excess - overlap) / 2, 0,
                           (height - bar_height) / 2 if rim_flat else 0))
    mesh = trimesh.util.concatenate((body, bar))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def rim_handle(extents) -> trimesh.Trimesh:
    """Place the R2 long-axis excess in a thin bar flush with the vessel rim."""
    _, short, height = np.asarray(extents, dtype=float)
    return _attach_vessel_handle(cylinder((short, short, height)), extents, rim_flat=True)


def vessel_shell(extents, wall_radius_fraction: float, handle: str = "none") -> trimesh.Trimesh:
    """Revolve one connected closed shell with an open mouth and optional bar."""
    long, short, height = np.asarray(extents, dtype=float)
    radius = short / 2
    wall = wall_radius_fraction * radius
    if not 0 < wall < min(radius, height):
        raise ValueError("Wall thickness must leave a positive cavity radius and depth")
    profile = [(0, -height / 2), (radius, -height / 2), (radius, height / 2),
               (radius - wall, height / 2), (radius - wall, -height / 2 + wall),
               (0, -height / 2 + wall), (0, -height / 2)]
    body = trimesh.creation.revolve(profile, sections=64)
    if handle == "none":
        body.apply_scale((long / short, 1, 1))
        return body
    return _attach_vessel_handle(body, extents, rim_flat=handle == "rim_flat")


def peg_rack(extents, rows: int, count_per_row: int) -> trimesh.Trimesh:
    """Place evenly spaced upright box pegs overlapping a full-footprint base."""
    long, short, height = np.asarray(extents, dtype=float)
    base_height = 0.25 * height
    base = trimesh.creation.box(extents=(long, short, base_height))
    base.apply_translation((0, 0, -height / 2 + base_height / 2))
    spacing, width = long / count_per_row, 0.3 * short
    overlap = 0.01 * height
    peg_height = height - base_height + overlap
    xs = (np.arange(count_per_row) + 0.5) * spacing - long / 2
    ys = np.linspace((-short + width) / 2, (short - width) / 2, rows) if rows > 1 else [0]
    parts = [base]
    for x in xs:
        for y in ys:
            peg = trimesh.creation.box(extents=(0.5 * spacing, width, peg_height))
            peg.apply_translation((x, y, height / 2 - peg_height / 2))
            parts.append(peg)
    return trimesh.util.concatenate(parts)
