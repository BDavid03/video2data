"""Bounded TSDF fusion and observed-surface extraction; unknown grid points never form surface."""

import math


class OpeningVolume:
    def __init__(self, cloud, request, np):
        self.np = np
        self.spacing = request.voxel_size_m
        self.truncation = request.truncation_voxels * self.spacing
        self.origin = np.floor((cloud.min(axis=0) - self.truncation) / self.spacing) * self.spacing
        size = np.ceil((cloud.max(axis=0) + self.truncation - self.origin) / self.spacing) + 1
        if not np.isfinite(size).all() or (size < 2).any():
            raise ValueError("Opening TSDF has invalid bounds")
        self.shape = tuple(int(value) for value in size)
        self.count = math.prod(self.shape)
        if self.count > request.maximum_grid_points:
            raise ValueError("Opening TSDF exceeds maximum_grid_points; explicitly choose a coarser voxel size")
        self.total = np.zeros(self.count, dtype=np.float64)
        self.weight = np.zeros(self.count, dtype=np.uint32)
        self.frames = []
        self.minimum_views = request.minimum_fusion_views
        self.maximum_faces = request.maximum_surface_faces

    def coordinates(self, ids):
        np = self.np
        return np.column_stack(np.unravel_index(ids, self.shape)) * self.spacing + self.origin

    def integrate(self, index, depth, mask, valid, intrinsics, world_t_camera):
        np = self.np
        height, width = depth.shape
        updated = 0
        # One nearest-pixel observation per grid point per frame. Positive values
        # face the observing camera; negative values lie just behind the surface.
        for start in range(0, self.count, 65536):
            ids = np.arange(start, min(start + 65536, self.count))
            camera = (self.coordinates(ids) - world_t_camera[:3, 3]) @ world_t_camera[:3, :3]
            forward = camera[:, 2] > 0
            ids, camera = ids[forward], camera[forward]
            pixels = camera @ intrinsics.T
            uv = pixels[:, :2] / pixels[:, 2:3]
            inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
            ids, camera, uv = ids[inside], camera[inside], np.rint(uv[inside]).astype(np.int64)
            u, v = uv[:, 0], uv[:, 1]
            observed = mask[v, u] & valid[v, u]
            signed = depth[v, u] - camera[:, 2]
            observed &= np.isfinite(signed) & (signed >= -self.truncation)
            ids, signed = ids[observed], signed[observed]
            self.total[ids] += np.clip(signed / self.truncation, -1, 1)
            self.weight[ids] += 1
            updated += len(ids)
        self.frames.append({"frame_index": index, "integrated_grid_points": updated})

    def surface(self, trimesh):
        np = self.np
        known = self.weight >= self.minimum_views
        field = np.zeros(self.count, dtype=np.float64)
        np.divide(self.total, self.weight, out=field, where=self.weight > 0)
        # Six tetrahedra share cube diagonal 0--6. Neighbouring cubes therefore
        # use the same triangulation on their shared face.
        offsets = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        tetrahedra = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                      (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
        vertices, faces, edges = [], [], {}

        def crossing(a, b):
            a, b = int(a), int(b)
            if field[a] == 0:
                key = (a, a)
            elif field[b] == 0:
                key = (b, b)
            else:
                key = tuple(sorted((a, b)))
            if key not in edges:
                if len(vertices) >= 3 * self.maximum_faces:
                    raise ValueError("Opening TSDF exceeds its surface vertex budget; choose an explicit coarser grid")
                points = self.coordinates(np.array([a, b]))
                fraction = field[a] / (field[a] - field[b])
                edges[key] = len(vertices)
                vertices.append(points[0] + fraction * (points[1] - points[0]))
            return edges[key]

        def triangle(indices, outward):
            if len(set(indices)) != 3:
                return
            a, b, c = (vertices[index] for index in indices)
            normal = np.cross(b - a, c - a)
            if np.linalg.norm(normal) <= self.spacing ** 2 * 1e-12:
                return
            if np.dot(normal, outward) < 0:
                indices = (indices[0], indices[2], indices[1])
            if len(faces) >= self.maximum_faces:
                raise ValueError("Opening TSDF exceeds maximum_surface_faces; choose an explicit coarser grid")
            faces.append(indices)

        cube_shape = tuple(value - 1 for value in self.shape)
        for start in range(0, math.prod(cube_shape), 16384):
            base = np.column_stack(np.unravel_index(
                np.arange(start, min(start + 16384, math.prod(cube_shape))), cube_shape))
            corners = base[:, None, :] + offsets
            ids = np.ravel_multi_index(tuple(corners[:, :, axis] for axis in range(3)), self.shape)
            values = field[ids]
            active = known[ids].all(axis=1) & (values < 0).any(axis=1) & (values >= 0).any(axis=1)
            for cube in ids[active]:
                for tetrahedron in tetrahedra:
                    nodes = cube[list(tetrahedron)]
                    negative, positive = nodes[field[nodes] < 0], nodes[field[nodes] >= 0]
                    if not len(negative) or not len(positive):
                        continue
                    # The affine TSDF gradient sets consistent outward winding.
                    points = self.coordinates(nodes)
                    outward = np.linalg.solve(points[1:] - points[0], field[nodes[1:]] - field[nodes[0]])
                    if len(negative) == 1 or len(positive) == 1:
                        single, other = (negative, positive) if len(negative) == 1 else (positive, negative)
                        triangle(tuple(crossing(single[0], node) for node in other), outward)
                    else:
                        a = crossing(negative[0], positive[0])
                        b = crossing(negative[0], positive[1])
                        c = crossing(negative[1], positive[0])
                        d = crossing(negative[1], positive[1])
                        triangle((a, b, d), outward)
                        triangle((a, d, c), outward)
        if not faces:
            raise ValueError("insufficient_views: no observed TSDF zero surface at the requested grid and view count")
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        mesh.remove_unreferenced_vertices()
        return mesh, {"grid_origin_world_m": self.origin.tolist(), "grid_shape": list(self.shape),
                      "grid_points": self.count, "voxel_size_m": self.spacing,
                      "maximum_surface_vertices": 3 * self.maximum_faces,
                      "truncation_m": self.truncation, "minimum_views": self.minimum_views,
                      "observed_grid_points": int(known.sum()), "frames": self.frames,
                      "extraction": "Marching tetrahedra on cubes with all eight corners observed",
                      "unseen_surface_completion": False, "depth_weighting": "Equal per observed frame"}
