# Camera motion

Pipeline role: a support tool, not in the submitted entry. It estimates the head camera's motion; the mesh_fusion test
(evidence only) reads it. No step of the main pipeline depends on it.

A slice is one folder under `src/features/` with its own code, request files and README. An episode is one recorded
video clip from the challenge dataset.

## What it does

It gives the colour camera's pose at every frame, in metres, in a world fixed to the table. Object pose in the world is
camera pose times object-in-camera pose, so camera drift becomes object drift one for one. The scorer removes one rigid
transform at the first frame, so later drift counts in full.

## Run

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.camera_motion estimate --request src/features/camera_motion/estimate.json
```

`runtime.json` holds the switch (`estimate_authorized`) and the episode list (public 39 only). CPU only, OpenCV and
NumPy. Runtime and RAM are not measured. The acceptance checks (still objects stay still, table plane fixed, landmark
reprojection at start, middle and end) have not been run, so accuracy is unknown.

Inputs: `ego_cam_a` remapped to a pinhole image, stereo depth, monocular depth scaled by `anchor.json` where stereo has
holes, and hand and object masks to drop moving pixels. No pose label is read.

## Method

1. Drop pixels inside dilated hand and object masks.
2. Detect features on the rest; get their 3D position from metric depth.
3. Track frame to map with PnP inside RANSAC. Add landmarks when enough new static area shows.
4. Fit the table as the largest near-horizontal RANSAC plane.
5. World z is the table normal. The origin is a given first-frame point, or where the first camera's centre ray meets
   the table. x is the first viewing direction on the table.
6. A frame with too few inliers gets `valid=false` and a NaN pose. Nothing is filled in.

It assumes a level table. The recordings have no inertial data to check this.

## Output

`data/outputs/camera_motion/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Meaning |
| --- | --- | --- |
| `trajectory.npz` | `world_T_camA` | float64 Tx4x4, NaN where invalid |
| | `valid`, `observed` | bool T; equal, since no pose is filled |
| | `inliers`, `reprojection_px` | RANSAC inliers, median reprojection error |
| | `frame_index`, `pts` | Frame identity |
| `world.json` | `up_axis`, `support_plane`, `origin_rule`, `heading_rule`, `scale_source` | World definition |
| `run.json` | | Request, input hashes, status, timing |

Camera axes: x right, y down, z forward (`data/configs/calibration/ego.json`). Failures record `tracking_lost`,
`no_support_plane` or `insufficient_depth` and keep the poses solved so far.

`challenger.normalize_optical_pose` is a small helper for the depth slice's disabled challenger import. It accepts rigid
OpenCV 4x4 camera poses, keeps units, and estimates no scale. See [depth](../depth/README.md).
