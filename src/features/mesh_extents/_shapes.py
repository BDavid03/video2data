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


def _bounded_box(lower, upper) -> trimesh.Trimesh:
    """Create a closed box between two corners."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    mesh = trimesh.creation.box(extents=upper - lower)
    mesh.apply_translation((upper + lower) / 2)
    return mesh


def compartment_rack(extents, profile: dict) -> trimesh.Trimesh:
    """Build the video tray, open compartment and lower pegs around the v3 floor."""
    dims = np.asarray(extents, dtype=float)
    p = profile
    floor, overlap = p["floor_height_fraction"], p["overlap_height_fraction"]
    wall_x = p["wall_short_fraction"] * dims[1] / dims[0]
    wall_y = p["wall_short_fraction"]
    parts = [_bounded_box((0, 0, 0), (1, 1, floor))]
    for length, width, cx, cy, top in (
        (1, 1, 0.5, 0.5, p["rim_top_fraction"]),
        (p["compartment_length_fraction"], p["compartment_width_fraction"],
         *p["compartment_center_fraction"], p["compartment_top_fraction"]),
    ):
        x0, x1, y0, y1 = cx - length / 2, cx + length / 2, cy - width / 2, cy + width / 2
        for lo, hi in (((x0, y0), (x1, y0 + wall_y)), ((x0, y1 - wall_y), (x1, y1)),
                       ((x0, y0), (x0 + wall_x, y1)), ((x1 - wall_x, y0), (x1, y1))):
            parts.append(_bounded_box((*lo, floor - overlap), (*hi, top)))
    for x in p["peg_centers_long_fraction"]:
        for y in p["peg_centers_short_fraction"]:
            half_x, half_y = p["peg_long_fraction"] / 2, p["peg_short_fraction"] / 2
            parts.append(_bounded_box((x - half_x, y - half_y, floor - overlap),
                                      (x + half_x, y + half_y, p["peg_top_fraction"])))
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation((-0.5, -0.5, -0.5))
    mesh.apply_scale(dims)
    return mesh


def _profile_prism(outline, lower: float, upper: float) -> trimesh.Trimesh:
    """Extrude a convex outline into a closed solid using the existing convex hull."""
    xy = np.asarray(outline, dtype=float)
    return trimesh.convex.convex_hull(np.vstack([np.column_stack((xy, np.full(len(xy), z)))
                                                for z in (lower, upper)]))


def _shallow_bowl(length: float, inner_radius: float, floor: float) -> trimesh.Trimesh:
    """Revolve a closed shallow paraboloid shell and scale its elliptical footprint."""
    radii = np.linspace(0, 1, 7)
    outer = [(r, (1 - floor) * r * r) for r in radii]
    inner = [(inner_radius * r, floor + (1 - floor) * r * r) for r in radii[::-1]]
    mesh = trimesh.creation.revolve([*outer, (1, 1), *inner, outer[0]], sections=48)
    mesh.apply_scale((length / 2, 0.5, 1))
    mesh.apply_translation((length / 2, 0, 0))
    return mesh


def thin_tool_profile(extents, profile: dict) -> trimesh.Trimesh:
    """Join a measured broad head and grip in the tool's observed longitudinal plane."""
    p = profile
    head, neck = p["head_fraction"], p["neck_width_fraction"]
    outline = np.array([(0, -0.5), (head, -neck / 2), (head, neck / 2), (0, 0.5)])
    if p["kind"] == "bowl":
        parts = [_shallow_bowl(head, p["bowl_inner_radius_fraction"], p["floor_thickness_fraction"])]
    elif p["kind"] == "brush":
        parts = [_profile_prism(outline, 0, 1)]
    else:
        parts = [_profile_prism(outline, 0, p["floor_thickness_fraction"])]
        wall = p["wall_width_fraction"]
        for first, second in ((0, 1), (2, 3)):
            a, b = outline[first], outline[second]
            direction = b - a
            inward = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction) * wall
            parts.append(_profile_prism(np.clip([a, b, b + inward, a + inward],
                                                (0, -0.5), (head, 0.5)), 0, 1))
        parts.append(_profile_prism([(head - wall, -neck / 2), (head, -neck / 2),
                                     (head, neck / 2), (head - wall, neck / 2)], 0, 1))
    start, width = p["grip_start_fraction"], p["grip_width_fraction"]
    if p["kind"] == "bowl":
        grip_outline = [(start, -neck / 2), (1, -width / 2), (1, width / 2), (start, neck / 2)]
    else:
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        grip_outline = np.column_stack(((1 + start) / 2 + (1 - start) / 2 * np.cos(angles),
                                        width / 2 * np.sin(angles)))
    thickness = p["grip_thickness_fraction"]
    lower = 1 - thickness if p["kind"] == "bowl" else 0
    parts.append(_profile_prism(grip_outline, lower, lower + thickness))
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation((-0.5, 0, -0.5))
    if p["axis"] == "z":
        # Proper rotation maps longitudinal u to height, transverse v to long, thickness w to short.
        mesh.vertices = mesh.vertices[:, [1, 2, 0]]
    mesh.apply_scale(extents)
    return mesh
