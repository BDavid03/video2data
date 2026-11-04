# Camera motion

Status: the bounded opening-39 camera stage completed in the canonical checkout;
independent accuracy and full-clip reliability remain unestablished. See the
[evidence inventory](../../../docs/experiments/01-evidence.md) and
[camera/depth challenger dispositions](../../../docs/experiments/05-camera-depth.md).
No new camera execution or checks accompanied this backlog review.

The camera is mounted on the demonstrator's head and moves throughout every clip.
This slice estimates where the colour camera is in a fixed, metric, gravity-aligned
world at every frame. Without it, an object pose seen from the camera cannot be
told apart from the head turning.

## Role in scoring

The scored quantity is the object's pose in a world frame, over time
(**Official**, `KIT/tracks/track_3/README.md:10-22`). Object pose is composed as
`world_T_object = world_T_camera * camera_T_object`, so camera drift becomes
object drift one for one. The scorer derives one rigid transform from the first
frame of object slot 0 and applies it to every object and every frame
(**Official**, `KIT/v2dlb/object_pose.py:122-163`), so drift after the first frame
is scored in full.

The choice of world origin and heading costs nothing, because the scorer removes
one rigid transform. The direction of gravity is different: the scorer would
absorb a tilted world, but the simulator would not. Objects in a tilted scene
slide off the table before the policy touches them.

## Operations

Disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `estimate` | Per-frame camera pose and the world definition for one episode | Perception environment | No for the baseline |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `ego_cam_a` clip, remapped to the pinhole grid | Dataset, [depth](../depth/README.md) | Frames, grid, intrinsics |
| `data/outputs/stereo_depth/.../frames` | [stereo_depth](../stereo_depth/README.md) | Metric depth on the colour grid, validity |
| `data/outputs/depth/.../frames` and `anchor.json` | [depth](../depth/README.md), stereo_depth | Dense monocular depth scaled to metres, where stereo has holes |
| `data/outputs/object_masks/.../frames` | [object_masks](../object_masks/README.md) | Hand and object masks, to exclude moving pixels |

## Method

Baseline, using only packages already installed:

1. **Static pixels.** Dilate every hand and object mask and discard what they
   cover. What remains is table, wall and furniture.
2. **Landmarks.** Detect and describe features on static pixels. Give each
   feature a 3D position in the camera frame from metric depth. Features without
   valid depth are dropped.
3. **Frame-to-map tracking.** Initialise the map from the first frame. For each
   later frame, match features to the map and solve the camera pose with
   perspective-n-point inside a RANSAC loop. Add landmarks from a frame when it
   sees enough new static area. Tracking against a map, and not only against the
   previous frame, limits drift when the head returns to a view it had before,
   which a seated tabletop demonstration does often.
4. **Support plane.** Fit a plane to the static 3D points with RANSAC and keep the
   largest near-horizontal plane in the camera's lower field. This is the table.
5. **World frame (Design).** The z axis is the table normal, pointing up. The
   origin is the table projection of an explicitly supplied first-frame slot-0
   point when provided. Otherwise the implemented opening pass uses the first
   camera's principal-ray intersection with the table, avoiding a dependency on
   object initialisation that has not happened yet. `world.json` records the
   actual origin rule and source. The x axis is the first camera's viewing
   direction projected onto the table. Units are metres.
6. **Gaps.** A frame with too few inliers has `valid=false` and a NaN pose.
   No gap is interpolated here. object_pose composes only where both factors
   are valid; hand_motion also propagates missing camera poses. scene_assembly
   bridges the resulting object and hand world-pose gaps, within its own maximum
   gap, and records every fill (D10).

**Assumption (Design).** The table is level, so its normal is the direction of
gravity. The recordings carry no inertial data to check this.

**Upgrade path.** If measured drift is a leading error, replace steps 2 and 3
with a learned stereo SLAM system. The candidate in the upstream baseline,
DROID-SLAM, compiles CUDA extensions whose build recipe does not list the
RTX 5080's architecture (source reading in `data/reports/model-readiness/README.md`). That
is a new dependency and a build risk, so it is not the starting point.

## Outputs

`data/outputs/camera_motion/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Type and shape | Meaning |
|---|---|---|---|
| `trajectory.npz` | `world_T_camA` | float64 T×4×4 | Maps colour-camera optical coordinates into the world; NaN where `valid=false` |
| | `valid` | bool T | A pose was solved from this frame's own matches |
| | `observed` | bool T | Equal to `valid`; this slice never fills a pose |
| | `inliers` | int32 T | RANSAC inlier count |
| | `reprojection_px` | float32 T | Median inlier reprojection error |
| | `frame_index`, `pts` | int64 T | Frame identity |
| `world.json` | `up_axis` | | `+z` |
| | `support_plane` | | Plane in world coordinates, inlier count, extent of inliers |
| | `origin_rule`, `heading_rule` | | The rules in step 5, as text |
| | `scale_source` | | The stereo_depth run that supplied metres |
| `run.json` | | | Request, input hashes, status, timing |

T equals the clip's frame count. The camera optical frame is x right, y down,
z forward, matching `data/configs/calibration/ego.json`.

## Configuration and authorization

`runtime.json` holds `estimate_authorized: false`. The request names the episode,
feature budget, inlier floor and mask dilation. Gap limits belong to scene_assembly.

## Dependencies

Baseline: OpenCV and NumPy, already present. No new package.

## Resources

CPU only for the baseline. Runtime and RAM are unmeasured.

## Failure handling

The run record distinguishes `tracking_lost`, `no_support_plane` and
`insufficient_depth`. A failed episode keeps the poses solved so far.

## Separation of public labels

No pose label is read. The public labels carry no camera pose, so they cannot
check this slice directly. On development episodes, a static object's labelled
world pose is constant, so the spread of `world_T_camA * camA_T_object` over the
frames before first contact is a label-free consistency check on camera motion
and object pose together.

## Acceptance evidence

Once runs are authorized: a static object stays still in world coordinates while
the head moves; the table plane stays fixed; and re-projected map landmarks stay
on their image features at the start, middle and end of a clip.

## Open questions

| Question | Who can answer |
|---|---|
| Is there enough static texture on the table and background for feature tracking when hands cover the centre of the image? | Development runs |
| Is the table level in every recording? | Organisers, or a consistency check that resting objects do not slide in simulation |

## Offline challenger pose normalization

`challenger.normalize_optical_pose` is a camera-owned pure integration primitive
used by the disabled depth challenger-ingestion operation. It admits explicit
OpenCV optical camera-to-world or world-to-camera rigid 4×4 matrices, inverts the
latter, and preserves translation units and world gauge. It never estimates
scale, supplies a table-world alignment, fills invalid poses or launches ViPE/DA3.
The owning depth handler records invalid pose support, immutable source/weight/
license receipts and the complete calibrated grid/PTS lineage. Relative poses
remain `world_T_camA_relative`, separate from metric production trajectories.
See `../depth/README.md` for the full offline export contract and false gate.
