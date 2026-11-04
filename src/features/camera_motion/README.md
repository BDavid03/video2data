# Camera motion

Estimates the head-mounted colour camera's pose at every frame in a fixed, metric, table-aligned world. One run
exists, on the opening of public episode 39. Independent accuracy and full-clip reliability are not established.

`world_T_object = world_T_camera * camera_T_object`, so camera drift becomes object drift one for one. The scorer
removes one rigid transform from the first frame of slot 0, so later drift is scored in full. Origin and heading cost
nothing, but a tilted world makes objects slide in simulation.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.camera_motion estimate --request src/features/camera_motion/estimate.json
```

`runtime.json` limits this to the episodes it allows. CPU only, using OpenCV and NumPy. Runtime and RAM are unmeasured.

Inputs: `ego_cam_a` remapped to the pinhole grid; stereo depth; monocular depth scaled by `anchor.json` where stereo
has holes; and hand and object masks to exclude moving pixels.

Method:

1. Dilate hand and object masks and discard those pixels.
2. Detect features on the rest, with 3D positions from metric depth.
3. Track frame to map using PnP inside RANSAC, adding landmarks when a frame sees enough new static area.
4. Fit the table with RANSAC, keeping the largest near-horizontal plane.
5. World z is the table normal. The origin is a supplied first-frame slot-0 point, or the first camera's principal ray
   meeting the table. x is the first viewing direction on the table. Units are metres.
6. A frame with too few inliers has `valid=false` and a NaN pose. Nothing is interpolated here.

This assumes a level table, which the recordings cannot confirm because they carry no inertial data. A learned SLAM such
as DROID-SLAM would be a new dependency and a build risk.

Output: `data/outputs/camera_motion/<split>/episode_XXXXXX/<run-id>/`.

| File | Field | Meaning |
| --- | --- | --- |
| `trajectory.npz` | `world_T_camA` | float64 Tx4x4, NaN where invalid |
| | `valid`, `observed` | bool T; equal here, since no pose is filled |
| | `inliers`, `reprojection_px` | RANSAC inliers, median reprojection error |
| | `frame_index`, `pts` | Frame identity |
| `world.json` | `up_axis`, `support_plane`, `origin_rule`, `heading_rule`, `scale_source` | World definition |
| `run.json` | | Request, input hashes, status, timing |

The camera frame is x right, y down, z forward (`data/configs/calibration/ego.json`). Failures record
`tracking_lost`, `no_support_plane` or `insufficient_depth` and keep the poses solved so far. No pose label is read.

Open questions: whether there is enough static texture when hands cover the image centre, and whether the table is level
in every recording.

`challenger.normalize_optical_pose` is a pure helper for depth's disabled challenger ingestion. It admits rigid OpenCV
camera-to-world or world-to-camera 4x4 matrices and keeps translation units and gauge. It estimates no scale and fills no
invalid poses. See [depth](../depth/README.md).
