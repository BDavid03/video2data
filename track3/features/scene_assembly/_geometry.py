"""Bounded triangle geometry; no spatial index or trimesh proximity calls."""


def closest(points, triangles, np, chunk_size=1024, point_chunk_size=64):
    distances = np.full(len(points), np.inf)
    positions = np.zeros_like(points, dtype=np.float64)
    normals = np.zeros_like(points, dtype=np.float64)
    winding = np.zeros(len(points), dtype=np.float64)
    for start in range(0, len(triangles), chunk_size):
        tri = triangles[start:start + chunk_size]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        lengths = np.linalg.norm(normal, axis=1)
        if (lengths <= 1e-12).any():
            raise ValueError("invalid_collision_mesh: degenerate triangle")
        normal /= lengths[:, None]
        d00, d01, d11 = np.sum(ab * ab, axis=-1), np.sum(ab * ac, axis=-1), np.sum(ac * ac, axis=-1)
        denominator = d00 * d11 - d01 * d01
        for begin in range(0, len(points), point_chunk_size):
            end = min(begin + point_chunk_size, len(points))
            query = points[begin:end]
            delta = query[:, None] - a
            projected = query[:, None] - np.sum(delta * normal, axis=-1)[..., None] * normal
            v = projected - a
            d20, d21 = np.sum(v * ab, axis=-1), np.sum(v * ac, axis=-1)
            beta = (d11 * d20 - d01 * d21) / denominator
            gamma = (d00 * d21 - d01 * d20) / denominator
            inside = (beta >= 0) & (gamma >= 0) & (beta + gamma <= 1)
            candidates = [projected]
            for left, right in ((a, b), (b, c), (c, a)):
                edge = right - left
                t = np.clip(np.sum((query[:, None] - left) * edge, axis=-1) / np.sum(edge * edge, axis=-1), 0, 1)
                candidates.append(left + t[..., None] * edge)
            candidates = np.stack(candidates, axis=2)
            squared = np.sum((candidates - query[:, None, None])**2, axis=-1)
            squared[:, :, 0] = np.where(inside, squared[:, :, 0], np.inf)
            which = squared.argmin(axis=2)
            per_triangle = np.take_along_axis(squared, which[..., None], axis=2)[..., 0]
            best = per_triangle.argmin(axis=1)
            rows = np.arange(len(query))
            distance = np.sqrt(per_triangle[rows, best])
            better = distance < distances[begin:end]
            positions[begin:end][better] = candidates[rows[better], best[better], which[rows[better], best[better]]]
            normals[begin:end][better] = normal[best[better]]
            distances[begin:end][better] = distance[better]
            va, vb, vc = (tri[None, :, i] - query[:, None] for i in range(3))
            la, lb, lc = (np.linalg.norm(value, axis=-1) for value in (va, vb, vc))
            numerator = np.sum(va * np.cross(vb, vc), axis=-1)
            solid_denominator = (la * lb * lc + np.sum(va * vb, axis=-1) * lc +
                                 np.sum(vb * vc, axis=-1) * la + np.sum(vc * va, axis=-1) * lb)
            winding[begin:end] += np.sum(2 * np.arctan2(numerator, solid_denominator), axis=1)
    return distances, positions, normals, abs(winding) > 2 * np.pi


def seating_offset(vertices, support_triangles, np):
    if support_triangles is None:
        return -float(vertices[:, 2].min())
    correction = -np.inf
    for start in range(0, len(support_triangles), 512):
        tri = support_triangles[start:start + 512]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        denominator = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        usable = abs(denominator) > 1e-12
        for begin in range(0, len(vertices), 256):
            points = vertices[begin:begin + 256]
            delta = points[:, None, :2] - a[None, :, :2]
            safe = np.where(usable, denominator, 1)
            beta = (delta[..., 0] * ac[:, 1] - delta[..., 1] * ac[:, 0]) / safe
            gamma = (ab[:, 0] * delta[..., 1] - ab[:, 1] * delta[..., 0]) / safe
            inside = usable & (beta >= 0) & (gamma >= 0) & (beta + gamma <= 1)
            height = a[:, 2] + beta * (b[:, 2] - a[:, 2]) + gamma * (c[:, 2] - a[:, 2])
            if inside.any():
                correction = max(correction, float(np.max(np.where(inside, height - points[:, None, 2], -np.inf))))
    if not np.isfinite(correction):
        raise ValueError("Declared supporting object does not overlap the resting mesh")
    return correction
