"""Exact exterior triangles for unions of axis-aligned boxes."""

import numpy as np


def box_bounds(mesh) -> np.ndarray:
    """Recover exact bounds from closed axis-aligned box components."""
    boxes = []
    for part in mesh.split(only_watertight=False):
        if (part.vertices.shape != (8, 3) or part.faces.shape != (12, 3)
                or not part.is_watertight or not part.is_winding_consistent):
            raise ValueError("Expected closed 8-vertex, 12-face box components")
        bounds = part.bounds
        corners = np.array(np.meshgrid(*bounds.T, indexing="ij")).reshape(3, -1).T
        on_plane = (np.all(part.triangles == bounds[0], axis=1)
                    | np.all(part.triangles == bounds[1], axis=1))
        if (not np.array_equal(np.unique(part.vertices, axis=0), np.unique(corners, axis=0))
                or not on_plane.any(axis=1).all() or not (part.area_faces > 0).all()
                or not np.isclose(part.volume, np.prod(bounds[1] - bounds[0]), rtol=1e-12, atol=0)):
            raise ValueError("Component does not exactly describe an axis-aligned box")
        boxes.append(bounds)
    return np.asarray(boxes)


def mesh_for(shape: str, rule: dict, name: str, extents, parameter, attributes: dict):
    """Apply the R11 exterior boundary to the existing rack constructor only."""
    from ._rule_scores import mesh_for as base_mesh_for

    mesh = base_mesh_for(shape, rule, name, extents, parameter, attributes)
    if rule.get("rack_union") and name == "plastic_dish_rack":
        import trimesh

        if shape != "compartment_rack":
            raise ValueError("R11 requires the frozen R7 compartment rack")
        vertices, faces = union_boundary(box_bounds(mesh))
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def occupied_grid(boxes) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Return distinct face coordinates and occupied cells for (N, 2, 3) corners."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not len(boxes)
            or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= boxes[:, 0])):
        raise ValueError("Expected nonempty finite boxes with strictly increasing corners")
    axes = tuple(np.unique(boxes[:, :, axis]) for axis in range(3))
    occupied = np.zeros(tuple(len(axis) - 1 for axis in axes), dtype=bool)
    for box in boxes:
        limits = [np.searchsorted(axis, box[:, i]) for i, axis in enumerate(axes)]
        occupied[tuple(slice(lo, hi) for lo, hi in limits)] = True
    return axes, occupied


def _rectangles(mask):
    """Partition a face mask into greedily maximal disjoint rectangles."""
    remaining = mask.copy()
    while remaining.any():
        u0, v0 = np.argwhere(remaining)[0]
        v1 = v0 + 1
        while v1 < remaining.shape[1] and remaining[u0, v1]:
            v1 += 1
        u1 = u0 + 1
        while u1 < remaining.shape[0] and remaining[u1, v0:v1].all():
            u1 += 1
        remaining[u0:u1, v0:v1] = False
        yield u0, u1, v0, v1


def _boundary_rectangles(occupied):
    """Merge exposed cells on each oriented coordinate plane."""
    rectangles = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        cells = occupied.transpose(axis, u, v).astype(np.int8)
        jumps = np.diff(np.pad(cells, ((1, 1), (0, 0), (0, 0))), axis=0)
        for plane in range(len(jumps)):
            for sign in (-1, 1):
                # Entering solid has an outward normal in the negative direction.
                for u0, u1, v0, v1 in _rectangles(jumps[plane] == -sign):
                    corners = np.zeros((4, 3), dtype=np.int64)
                    corners[:, axis] = plane
                    corners[:, u] = (u0, u1, u1, u0)
                    corners[:, v] = (v0, v0, v1, v1)
                    rectangles.append(corners if sign > 0 else corners[::-1])
    return rectangles


def _triangulate(rectangles, axes):
    """Weld rectangle corners and split shared edges before triangulating patches."""
    corners = np.unique(np.concatenate(rectangles), axis=0)
    positions = np.column_stack([axes[i][corners[:, i]] for i in range(3)])
    vertices = positions.tolist()
    faces = []
    for rectangle in rectangles:
        boundary = []
        for start, end in zip(rectangle, np.roll(rectangle, -1, axis=0)):
            axis = np.flatnonzero(start != end)[0]
            fixed = [i for i in range(3) if i != axis]
            on_edge = ((corners[:, fixed] == start[fixed]).all(axis=1)
                       & (corners[:, axis] >= min(start[axis], end[axis]))
                       & (corners[:, axis] <= max(start[axis], end[axis])))
            indices = np.flatnonzero(on_edge)
            indices = indices[np.argsort(corners[indices, axis])]
            if start[axis] > end[axis]:
                indices = indices[::-1]
            boundary.extend(indices[:-1].tolist())
        if len(boundary) == 4:
            a, b, c, d = boundary
            faces.extend(((a, b, c), (a, c, d)))
        else:
            center = len(vertices)
            vertices.append(positions[boundary].mean(axis=0).tolist())
            faces.extend((center, a, b) for a, b in zip(boundary, boundary[1:] + boundary[:1]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def union_boundary(boxes) -> tuple[np.ndarray, np.ndarray]:
    """Return welded vertices and outward triangles, without coordinate snapping."""
    axes, occupied = occupied_grid(boxes)
    return _triangulate(_boundary_rectangles(occupied), axes)
