"""Table-relative object clusters and their extents in one depth frame (NumPy, OpenCV, SciPy)."""

import cv2
import numpy as np

from src.features.object_masks.opening_clusters import _clusters


def camera_points(z, k):
    v, u = np.indices(z.shape)
    return np.stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1)


def _ransac_plane(points, threshold, rng, iterations=800):
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        count = int((np.abs((points - a) @ normal) < threshold).sum())
        if count > best_count:
            best, best_count = (normal, -a @ normal), count
    return best


def _fit(points, reference):
    centre = points.mean(0)
    normal = np.linalg.svd(points - centre, full_matrices=False)[2][2]
    normal = normal if normal @ reference > 0 else -normal
    return normal, float(-centre @ normal)


def support_plane(pts, ok, rng, low=0.2, high=1.2):
    """Floor = largest plane in the lower 70 % of the image; table = sharpest height peak low-high m above it.

    Returns (normal, offset, table_height_above_floor); the normal points away from the floor.
    """
    start = int(0.3 * ok.shape[0])
    lower = pts[start:][ok[start:]]
    sample = lower[rng.choice(len(lower), min(30000, len(lower)), replace=False)]
    normal, offset = _ransac_plane(sample, 0.01, rng)
    above_floor = sample @ normal + offset
    if np.median(above_floor) < 0:
        normal, offset, above_floor = -normal, -offset, -above_floor
    bins = np.arange(low, high, 0.005)
    counts, _ = np.histogram(above_floor, bins)
    peak = float(bins[counts.argmax()] + 0.0025)
    near = sample[np.abs(above_floor - peak) < 0.015]
    table_normal, table_offset = _fit(near, normal)
    near = sample[np.abs(sample @ table_normal + table_offset) < 0.006]
    table_normal, table_offset = _fit(near, normal)
    return table_normal, table_offset, peak


def _local_plane(pts, table, layer, normal, offset, ring_px):
    """Refit the support plane on table pixels in a ring around one cluster (MoGe tables are not flat)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)
    region = cv2.dilate(layer.astype(np.uint8), kernel).astype(bool)
    ring = pts[region & table & ~layer]
    if len(ring) < 500:
        return region, normal, offset, False
    local_normal, local_offset = _fit(ring, normal)
    return region, local_normal, local_offset, True


def extents(points, height, normal):
    """In-plane PCA extents (2-98 %), height (98th percentile above the plane) and the in-plane axes."""
    e1 = np.array([1.0, 0, 0]) - normal[0] * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.stack((points @ e1, points @ e2), 1)
    uv -= uv.mean(0)
    axes = np.linalg.svd(uv, full_matrices=False)[2]
    projected = uv @ axes.T
    low, high = np.percentile(projected, [2, 98], axis=0)
    return {"in_plane_m": (high - low).tolist(), "height_m": float(np.percentile(height, 98)),
            "height_max_m": float(height.max())}


def table_component(height, ok, band):
    """Select the largest four-connected component within the table-height band."""
    table = ok & (np.abs(height) < band)
    count, labels = cv2.connectedComponents(table.astype(np.uint8), connectivity=4)
    sizes = np.bincount(labels[table], minlength=count)
    sizes[0] = 0
    return labels == sizes.argmax()


def frame_clusters(z, valid, k, request, rng):
    """Label image (0 = none, i + 1 = cluster i, largest first) and one summary row per cluster."""
    pts = camera_points(z, k)
    ok = valid & np.isfinite(pts).all(-1) & (z > 0)
    normal, offset, table_height = support_plane(pts, ok, rng)
    height = pts @ normal + offset
    table = table_component(height, ok, request.table_band_m)
    flat = lambda p: p - np.outer(p @ normal + offset, normal)  # noqa: E731
    table_points = flat(pts[table])
    centre = table_points.mean(0)
    radius = float(np.percentile(np.linalg.norm(table_points - centre, axis=1), 98))
    radial = np.linalg.norm(flat(pts.reshape(-1, 3)) - centre, axis=1).reshape(ok.shape)
    keep = (ok & (height > request.minimum_height_m) & (height < request.maximum_height_m)
            & (radial < radius + request.overhang_m))
    layers = _clusters(pts, keep, request.neighbour_distance_m, request.minimum_cluster_pixels, np)
    border = np.zeros_like(ok)
    border[:3], border[-3:], border[:, :3], border[:, -3:] = True, True, True, True
    label = np.zeros(ok.shape, np.int16)
    rows = []
    found = []
    for layer in layers:
        # Re-cut against the plane refitted around the cluster; depth bleed can bridge separate objects,
        # so every part that came from this cluster becomes a cluster of its own.
        region, local_normal, local_offset, local = _local_plane(pts, table, layer, normal, offset, request.ring_pixels)
        local_height = pts @ local_normal + local_offset
        parts = _clusters(pts, region & ok & (local_height > request.minimum_height_m)
                          & (local_height < request.maximum_height_m), request.neighbour_distance_m, 50, np)
        # A cluster with no part above the refitted plane was table bleed, not an object: it is dropped.
        found += [(part, local_height, local_normal, local) for part in parts if (part & layer).sum() > 0.5 * part.sum()]
    found.sort(key=lambda item: -int(item[0].sum()))
    for layer, local_height, local_normal, local in found:
        layer = layer & ~(label > 0)
        if layer.sum() < request.minimum_cluster_pixels:
            continue
        label[layer] = len(rows) + 1
        ys, xs = np.nonzero(layer)
        rows.append({"cluster": len(rows), "pixels": int(layer.sum()), "touches_border": bool((layer & border).any()),
                     "local_plane": local, "centroid_camera_m": pts[layer].mean(0).tolist(),
                     "bbox_rows": [int(ys.min()), int(ys.max())], "bbox_cols": [int(xs.min()), int(xs.max())],
                     **extents(pts[layer], local_height[layer], local_normal)})
    plane = {"normal": normal.tolist(), "offset_m": offset, "table_above_floor_m": table_height,
             "table_radius_m": radius, "table_pixels": int(table.sum())}
    scene = {"pts": pts, "ok": ok, "table": table, "normal": normal, "offset": offset}
    return label, rows, plane, scene


def object_mask(label, reference, dilation_px, share=0.5):
    """Union of this frame's clusters that lie mostly inside the dilated reference (frame-0) object mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1,) * 2)
    grown = cv2.dilate(reference.astype(np.uint8), kernel).astype(bool)
    mask = np.zeros_like(reference)
    for value in np.unique(label[label > 0]):
        cluster = label == value
        if (cluster & grown).sum() > share * cluster.sum():
            mask |= cluster
    return mask


def object_extents(scene, mask, request):
    """Extents of one object's (possibly multi-cluster) mask against the support plane refitted around it."""
    _, normal, offset, local = _local_plane(scene["pts"], scene["table"], mask, scene["normal"], scene["offset"],
                                            request.ring_pixels)
    points = scene["pts"][mask]
    row = extents(points, points @ normal + offset, normal)
    row.update(pixels=int(mask.sum()), local_plane=local)
    return row
