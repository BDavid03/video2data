"""Bounded point-weighted voxel accumulation for opening reconstruction."""


class OpeningPoints:
    def __init__(self, spacing, maximum_voxels, np):
        self.np = np
        self.spacing = spacing
        self.maximum_voxels = maximum_voxels
        self.cells = np.empty((0, 3), dtype=np.int64)
        self.totals = np.empty((0, 3), dtype=np.float64)
        self.counts = np.empty(0, dtype=np.int64)
        self.input_points = 0

    def add(self, points):
        np = self.np
        # Keep this bound even if a future caller supplies a whole frame.
        for start in range(0, len(points), 65536):
            batch = np.asarray(points[start:start + 65536], dtype=np.float64)
            if batch.ndim != 2 or batch.shape[1] != 3 or not np.isfinite(batch).all():
                raise ValueError("Opening points must be finite three-dimensional measurements")
            cells = np.floor(batch / self.spacing)
            if not np.isfinite(cells).all() or (np.abs(cells) >= 2**62).any():
                raise ValueError("Opening voxel coordinates exceed the supported integer range")
            keys, inverse, counts = np.unique(cells.astype(np.int64), axis=0,
                                              return_inverse=True, return_counts=True)
            totals = np.zeros((len(keys), 3), dtype=np.float64)
            np.add.at(totals, inverse, batch)
            keys, inverse = np.unique(np.concatenate((self.cells, keys)), axis=0, return_inverse=True)
            if len(keys) > self.maximum_voxels:
                raise ValueError("Opening points exceed maximum_point_voxels; explicitly choose a coarser voxel size")
            merged_totals = np.zeros((len(keys), 3), dtype=np.float64)
            merged_counts = np.zeros(len(keys), dtype=np.int64)
            np.add.at(merged_totals, inverse, np.concatenate((self.totals, totals)))
            np.add.at(merged_counts, inverse, np.concatenate((self.counts, counts)))
            if not np.isfinite(merged_totals).all() or (merged_counts <= 0).any():
                raise ValueError("Opening point accumulation overflowed")
            self.cells, self.totals, self.counts = keys, merged_totals, merged_counts
            self.input_points += len(batch)

    def add_depth(self, depth, keep, intrinsics, world_t_camera):
        np = self.np
        v, u = np.nonzero(keep)
        for start in range(0, len(u), 65536):
            x, y = u[start:start + 65536], v[start:start + 65536]
            z = depth[y, x]
            points = np.column_stack(((x - intrinsics[0, 2]) * z / intrinsics[0, 0],
                                      (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z))
            self.add(points @ world_t_camera[:3, :3].T + world_t_camera[:3, 3])

    def centroids(self):
        return self.totals / self.counts[:, None]

    def summary(self):
        return {"input_points": self.input_points, "occupied_voxels": len(self.cells),
                "maximum_point_voxels": self.maximum_voxels, "voxel_size_m": self.spacing,
                "chunk_points": 65536, "weighting": "One weight per measured pixel across all opening frames",
                "overflow_policy": "Fail; never drop points or change voxel size automatically"}
