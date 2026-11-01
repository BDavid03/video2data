# Scene assembly

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

Perception produces honest, incomplete measurements: object poses with gaps, hands
that leave the image, small jitter everywhere. The simulator needs a complete and
physically possible scene on every frame. This slice is where one becomes the
other. It is the only place in the project where a missing measurement is filled,
and every fill is flagged.

## Role in scoring

The scene bundle is the demonstration the policy learns to reproduce. Three of its
properties reach the score directly.

| Property | Effect |
|---|---|
| The first-frame pose of object slot 0 | The scorer anchors its alignment there (**Official**, `KIT/v2dlb/object_pose.py:122-163`). This slice takes that pose from the quiet opening of the clip, where it is measured best. |
| The relative placement of the two objects | RPE measures it in every frame (**Official**, `KIT/v2dlb/object_pose.py:194-224`). |
| A stable starting state | The rollout starts from the reference's first frame. An object that starts slightly inside the table, or floating above it, moves before the policy acts, and that motion is scored. |

## Operations

Disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `assemble` | The scene bundle for one episode | Perception environment | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `poses.npz`, `opening.json` | [object_pose](../object_pose/README.md) | Camera-relative and world poses, `valid` and `world_valid`, confidence, opening averages |
| `hands.npz` | [hand_motion](../hand_motion/README.md) | Joint positions and rotations, `world_valid` |
| `world.json`, `trajectory.npz` | [camera_motion](../camera_motion/README.md) | Support plane, camera positions |
| `<object>.glb`, `mesh.json` | [object_mesh](../object_mesh/README.md) | Geometry, body frame record |

## Method

1. **Frame grid.** One scene frame per video frame, same index. The frame rate is
   declared as exactly 20.0 (see [policy_rollout](../policy_rollout/README.md)
   for why).
2. **Put the table at zero height.** Shift the world along z so the support
   plane is z = 0. The simulator's ground plane is at z = 0 and upstream drops
   any support surface at or below 5 cm as already covered by it (**Official**,
   `RG/source/robotic_grounding/robotic_grounding/retarget/support_recon.py:39,741-757`).
   With floating hands there is no robot body that needs a floor below the table,
   so the ground plane can serve as the table top (**Design**).
3. **Label each object's frames as resting or moving.** An object is resting
   when no hand is near it and its measured pose is not changing beyond noise.
   The thresholds are request fields, set on development episodes.
4. **Resting segments become constant.** Replace the poses in each resting
   segment with one robust average. A resting object does not move, so any
   measured motion there is camera drift or tracking noise.
5. **Seat resting objects.** Lower or raise each resting pose so the mesh's
   lowest point touches its support, either the table or the object it rests on.
   Record the correction. A correction above a configured limit fails the
   episode, because it means the pose or the scale is wrong and should not be
   hidden.
6. **Moving segments are smoothed and bridged.** Smooth with weights from the
   tracker's confidence. Read object `world_valid` and hand `world_valid` before
   world arrays, so missing camera poses remain visible as gaps. Bridge a gap
   by interpolating between its valid ends,
   up to the maximum gap length. A longer gap fails the episode.
7. **Complete the hands.** Upstream requires both hands on every frame and has
   no validity field (**Official**,
   `RG/scripts/retarget/ego_recon_to_sharpa.py:307-312,521-535`).

   | Situation | Rule |
   |---|---|
   | Short gap between valid frames | Interpolate, flag as filled |
   | Hand leaves the image and returns | Move smoothly to a parked pose, hold, and return |
   | Hand never seen in the episode | Parked for the whole episode |

   The parked pose is open-handed, above the table, and outside the region the
   objects occupy during the episode, so a parked hand cannot touch anything.
8. **Contacts.** For each of the 16 hand links upstream defines, find the
   nearest point on any object mesh using explicit closest-point calculations
   over triangles in bounded NumPy chunks. Keep the winning triangle's surface
   normal. No trimesh spatial-index or signed-distance query is called. Within the contact distance, record the
   point, the surface normal and the body's index; otherwise record none. The
   upstream format needs these fields present, accepts zeros, and numbers bodies
   from 1 with 0 meaning no contact (**Official**, `support_recon.py:322-337`).
9. **Report.** Hand-object penetration depth per frame, the share of frames
   filled per object and per hand, and the seating corrections. For a closed,
   consistently oriented mesh, classify inside/outside by the triangle solid-angle
   winding sum and use the triangle distance from step 8 as penetration depth.
   Invalid or open simulation meshes fail with `invalid_collision_mesh`; do not
   invent a signed distance. This baseline needs no `rtree`. Its cost and
   geometric accuracy remain unmeasured.

## Outputs

`outputs/scene_assembly/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Type and shape | Meaning |
|---|---|---|---|
| `scene.json` | `fps` | | 20.0 |
| | `frame_count` | | Equal to the clip's frame count |
| | `objects` | | Names in official slot order, mesh file, body frame record, mass estimate |
| | `world` | | Z up, table at z = 0, origin and heading rule |
| | `provenance` | | Inherited from inputs |
| `trajectories.npz` | `world_T_object` | float64 T×B×4×4 | Complete, no NaN |
| | `object_state` | uint8 T×B | 0 resting, 1 moving |
| | `object_filled` | bool T×B | True where the pose was interpolated |
| | `hand_joints_world` | float64 T×2×21×3 | Complete |
| | `hand_joint_rotations_world` | float64 T×2×21×3×3 | Complete |
| | `hand_source` | uint8 T×2 | 0 observed, 1 interpolated, 2 parked or in transit to parking |
| | `contact_position`, `contact_normal` | float32 T×2×16×3 | In world coordinates |
| | `contact_body` | int32 T×2×16 | 1-based body index, 0 for none |
| `assets/<object>.glb` | | | Unchanged copies of the meshes used |
| `report.json` | | | The numbers in step 9 |
| `run.json` | | | Request, input hashes, status |

This bundle and its assets are the forward interface from Windows perception
to Linux, where all retargeting operations run. Rollout Parquet, evaluator JSON
and run/inspection records later return from Linux to Windows submission; see
[artifact contracts](../../../docs/design/artifact-contracts.md).

## Configuration and authorization

`runtime.json` holds `assemble_authorized: false`. The request names the input
runs and sets the resting thresholds, the maximum gap, the seating limit, the
contact distance and the parked pose.

## Dependencies

NumPy, SciPy and `trimesh`, all in the locked MoGe2 environment. Geometry and
mesh loading use these packages, but contact distance and inside/outside use
the explicit NumPy triangle calculations in steps 8 and 9. The MoGe2 lock has
no `rtree`; no spatial-index-dependent trimesh query is assumed available.
No new package is proposed for this baseline.

## Resources

CPU only. Unmeasured.

## Failure handling

`assemble` fails, and says which rule failed, when a gap exceeds the maximum, a
seating correction exceeds its limit, or the opening pose of object slot 0 is
missing. A scene is never completed by guessing past a limit.

## Separation of public labels

No label is read. On development episodes, the assembled object trajectory can be
scored against public poses with the kit's metric code, as in
[object_pose](../object_pose/README.md). Comparing the score before and after
assembly shows whether the resting and seating rules help or hurt. That
comparison carries the `public_label_diagnostic` tag.

## Acceptance evidence

Once runs are authorized: replaying the bundle without physics shows objects that
stay still until touched, hands that arrive before the object moves, and no hand
passing through an object. Placing the objects in the simulator at frame 0 and
stepping with no robot leaves them where they are.

## Open questions

| Question | Who can answer |
|---|---|
| Does using the ground plane as the table change contact behaviour compared with a separate support surface? | A simulator run |
| How far from the objects must a parked hand be so the policy's reward does not pull it into the scene? | The training objective's terms, then a training run |
| Is the mass of each object published anywhere? The public URDFs describe their mass as a placeholder (**Official**, dataset `track_3/public/urdf/*.urdf`). | Organisers |
