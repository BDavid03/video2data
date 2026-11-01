# mesh_extents

Video-measured object proportions, and the closed primitive (ellipsoid or box) built from them, for
the Track 3 CD-O mesh file (D1).

## Why this slice exists

CD-O (kit `metric_code/track_3/CD-O.py`) scales each candidate by the ratio of surface RMS radii and
registers it rigidly to the scan before a symmetric Chamfer. Absolute scale and pose cost nothing;
proportions do. Context 01 measured, on the 10 public scans, that a primitive with the scan's own
principal extents scores 1.17 cm against 2.52 cm for a cube (analysis only). This slice measures those
proportions from the egocentric video instead, so no scan geometry enters a submitted mesh (the
challenge FAQ requires participants' own reconstructions).

## Operations

| Operation | Produces | GPU |
|---|---|---|
| `measure` | Per frame: support plane, table-relative clusters (label image, extents); per assigned object: extents per frame and their median; `overlay_<frame>.png` of the first frame | No |
| `build` | One primitive per object pooled over measure runs, and `meshes/episode_XXXXXX/<object>.ply` for every scored pair, ready for the kit's `pack_reconstruction.py track3-meshes` | No |

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python scripts/track3/d1_measure.py --out .runtime/tmp/d1-runs.json
PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_extents build --request <build.json>
```

Inputs: one completed `depth.generate` run per episode (MoGe2, ego_cam_a, opening frames 0-19).
No stereo anchor is used: a per-frame scale multiplies all three extents alike, so proportions do
not depend on it.

## Method

Per frame, in camera coordinates:

1. **Support plane.** RANSAC the largest plane in the lower 70 % of the image (the floor), then take
   the sharpest histogram peak of heights 0.2-1.2 m above it (the round table top) and refit there.
   Context 02 found that the camera_motion world plane can be the floor, which made the whole table
   one "object" on public episode 39.
2. **Clusters.** Points 8 mm-0.5 m above the table and within the table radius + 15 cm horizontally
   (objects overhang the rim), joined by 4-connected metric edges of at most 2 cm, at least 300 px.
3. **Local plane.** MoGe2 table tops are not flat: depth bleeds a few mm upwards around objects. Each
   cluster's plane is refitted on table pixels in a 40 px ring around it and the cluster re-cut.
4. **Extents.** In-plane PCA extents (2-98 % range) and height (98th percentile above the local
   plane): `[in_plane_long, in_plane_short, height]`.

An object's clusters are named once, on the first measured frame, in `assignments.json` (a list, so an
object split by depth gaps is reunited). Later frames take every cluster lying mostly inside the
object's first-frame mask dilated by 15 px. Only clean, isolated, fully visible objects are assigned;
merged objects (a pot on the rack) are left out of that episode. `build` divides each episode's
median by its geometric mean, takes the per-axis median over episodes and restores the median size.

## Known limits

- Undersides are never seen; height is the visible top above the table, so hollow objects become solid.
- The first-frame assignment is a recorded visual judgement by the chain context (see
  `assignments.json`), not a detector.
- Scale is MoGe2's, not metric. Simulation (D2) needs a metric rescale before these meshes are used.

## Evidence

`outputs/mesh_eval/d1-ctx02/` (local CD-O against the public scans, measurement only).
