# Scene assembly

Status: implemented, never run

Perception produces honest, incomplete measurements: object poses with gaps, hands
that leave the image, small jitter everywhere. The simulator needs a complete and
physically possible scene on every frame. This slice is where one becomes the
other. It is the only place in the project where a missing measurement is filled,
and every fill is flagged.

The uniform 20 Hz export also requires consecutive RGB PTS values to differ by
exactly 512 ticks at the admitted `1/10240` timebase. Matching nominal FPS and
frame count alone is insufficient. Irregular clocks are rejected rather than
silently exported as uniform FlashCHORD reference samples. This admission guard
was implemented by source review only; no assembly or verification ran.

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
   for why). Names and pixel grid alone are insufficient: the pose run's mesh
   hashes (visual and collision) must match the resolve run, and its recorded
   inputs must contain the exact camera run path/hash and, in full mode, the
   exact hand run path/hash. Object_pose `compose` can bind that `hand_run`.
   A scans-only spoon assignment pass cannot be assembled.
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
   | Short gap between valid frames | Blend the wrist, parent-frame bone directions and anatomical rotations; flag as filled |
   | Hand leaves the image and returns | Move smoothly to a parked pose, hold, and return |
   | Hand never seen in the episode | Parked for the whole episode |

   The parked pose is open-handed, above the table, and outside the region the
   objects occupy during the episode, so a parked hand cannot touch anything.
   Both wrist anchors are on the camera side (below the objects' minimum world
   x); the left hand is beyond their maximum world y and the right beyond their
   minimum y. World x is the first camera's heading projected onto the table.

   `_hands.py` completes the MANO21 tree from wrist to fingertips. The wrist
   translates linearly and each anatomical joint rotation follows its shortest
   SO(3) interpolation. Bone directions interpolate on the unit sphere in the
   corresponding endpoint parent frames, then transform through the blended
   parent frame. Bone lengths interpolate between endpoint lengths; equal lengths
   therefore stay constant through a rotation instead of shrinking as they can
   when world joint positions are averaged independently. The same method handles
   transitions to and from parking. A template with different hand proportions
   causes explicit length interpolation during that transition.

   Exactly opposing local directions use a recorded deterministic perpendicular
   direction. Degenerate endpoint bones fail interpolation. This is a reference
   completion rule, not a MANO forward solve or an observed motion. It does not
   enforce joint limits or guarantee collision-free transit. `hand_source` keeps
   the existing observed/interpolated/parked distinction, and `hand_completion` in
   the scene and report records the rule. Input hands must declare metric MANO21
   anatomical frames, and boolean world validity must be a subset of camera
   validity. The implementation has not been executed or tested.
8. **Contacts.** Preserve the pinned upstream `MANO_HAND_LINKS` order: palm,
   then three links each for thumb, index, middle, ring and pinky. These are
   **16 link slots**, separate from the 21 joint slots. `_contacts.py` owns the
   scene's geometric proxy: five evenly spaced points per finger segment, plus
   the palm's six defining joints, fifteen pair midpoints and centroid. For each
   link, find the nearest object triangle to those samples and retain that
   triangle's surface point and unit normal. Compare candidates across objects
   and retain the closest body within the configured contact distance.
   Calculations use bounded NumPy chunks; no spatial index or signed-distance
   query is called. Body IDs are one-based, with zero position/normal fields and
   body ID zero for absent contacts. This is sampled geometric proximity, not
   observed contact or exact full-surface hand collision. Its accuracy and cost
   remain unmeasured.

   `scene.json` and `report.json` record `contact_layout`, including the ordered
   link names, joint groups and sampling rule. `trajectories.npz` records the
   corresponding `contact_link_names`. The parked template supplies geometry
   only; it no longer selects arbitrary joints as contact slots. The referenced
   upstream `retarget/params.py` is hashed into the scene's source record.
9. **Report.** Hand-object penetration depth per frame, the share of frames
   filled per object and per hand, and the seating corrections. For a closed,
   consistently oriented mesh, classify inside/outside by the triangle solid-angle
   winding sum and use the triangle distance from step 8 as penetration depth.
   Invalid or open simulation meshes fail with `invalid_collision_mesh`; do not
   invent a signed distance. This baseline needs no `rtree`. Its cost and
   geometric accuracy remain unmeasured.

Geometry calculations use the recorded collision file where available, with
the visual mesh as the fallback. A geometry-only in-memory copy merges equal
vertex positions across UV seams and removes degenerate triangles before
closedness, winding and volume checks. Seating, contacts and penetration all
use this same copy. Visual and collision asset files remain unchanged and are
both copied into the scene bundle.

Seating is computed once per resting segment and broadcast over it. In resting
mode parking and contact/penetration checks use one frame and broadcast their
results. In full mode contact queries are grouped by each object's resting or
moving segment, transformed into the fixed mesh body frame, and deduplicated.
The triangle data is traversed once per segment in bounded triangle/point
chunks; moving hands still get their own distances. Hand-nearness queries for
candidate resting segments use the same grouped calculation. A constant
object pose does not imply constant contacts when a hand moves.

### The `resting` mode (Design)

A request with `mode: resting` builds the scene for the resting entry
([entry tiers](../../../docs/design/entry-tiers.md)). It reads `opening.json`,
the meshes and `world.json`, and nothing per-frame.

| Step | Resting mode |
|---|---|
| 1, 2 | As above |
| 3, 4 | Every object is resting on every frame, at its opening average |
| 5 | As above. The seating limit still applies. |
| 6 | Skipped. There is no moving segment and nothing to bridge. |
| 7 | Both hands parked on every frame, from the selected `hand_motion.parked_template` artifact |
| 8 | No contacts. The arrays are present, sized for 16 links, and zero. |
| 9 | The seating corrections and each object's `rotation_ambiguous` flag |

`scene.json` records `mode` and `opening_quality`, which is `measured` or
`degraded`. A degraded scene was built from a registration above the residual
limit, for an episode where the opening route failed and a submission still
needed a rollout. `hand_source` is 2 on every frame. A resting bundle
carries the same provenance as its inputs and is a normal input to retargeting.

Both modes require `parked_hand_run`, an explicit completed
[`hand_motion.parked_template`](../hand_motion/README.md) producer. The old null
`parked_hand.json` beside this README is retained as historical configuration
but is no longer read. Template generation remains separately disabled and
requires approved MANO provisioning plus user-provided licensed assets.

The template contains per-side MANO21 joints relative to a zero wrist, anatomical
rotations and 778 surface vertices. It declares metres, anatomical unmirrored
frames, palms down and `observed: false`. Its model-asset scope has no episode or
split, so it can be reused without pretending to be an observed hand from another
episode. All episode-derived scene inputs still require matching dataset
revision, episode and provenance.

Assembly checks the selected template and raw-model artifact hashes and records
the backend/source identity. It copies `parked_hand.json` into the scene bundle
and records both producer and artifact references in `scene.json` and `run.json`.
Parking bounds include the full neutral hand surface as well as its joints.
This bounds the model geometry; it is not evidence of robot-link clearance after
retargeting. Template generation, orientation and scene behavior remain unrun.

## Outputs

`outputs/scene_assembly/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Type and shape | Meaning |
|---|---|---|---|
| `scene.json` | `fps` | | 20.0 |
| | `frame_count` | | Equal to the clip's frame count |
| | `objects` | | Names in official slot order, mesh file, body frame record, mass estimate |
| | `world` | | Z up, table at z = 0, origin and heading rule |
| | `provenance` | | Inherited from inputs |
| | `hand_completion` | | Skeleton interpolation contract, or constant parked-template rule in resting mode |
| `trajectories.npz` | `world_T_object` | float64 T×B×4×4 | Complete, no NaN |
| | `object_state` | uint8 T×B | 0 resting, 1 moving |
| | `object_filled` | bool T×B | True where the pose was interpolated |
| | `hand_joints_world` | float64 T×2×21×3 | Complete |
| | `hand_joint_rotations_world` | float64 T×2×21×3×3 | Complete |
| | `hand_source` | uint8 T×2 | 0 observed, 1 interpolated, 2 parked or in transit to parking |
| | `contact_link_names` | 16 strings | Exactly the upstream MANO hand-link order |
| | `contact_position`, `contact_normal` | float32 T×2×16×3 | In world coordinates |
| | `contact_body` | int32 T×2×16 | 1-based body index, 0 for none |
| `assets/<object>.glb` | | | Unchanged copies of the meshes used |
| `assets/<object>_collision.glb` | | | Unchanged collision assets; paths and hashes also recorded in `scene.json` |
| `parked_hand.json` | | | Exact copy of the selected sourced neutral template |
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
| Does the neutral MANO template preserve the intended palms-down anatomical convention through retargeting? | Future authorized template generation and retargeting evidence; licensed assets and runtime provisioning remain outstanding |
| Is the mass of each object published anywhere? The public URDFs describe their mass as a placeholder (**Official**, dataset `track_3/public/urdf/*.urdf`). | Organisers |

## Bounded joint candidate (`refine`, source implemented, unrun)

`RefineSceneRequest` and `refine.py` add a separately disabled CPU operation.
The illustrative `refine.json` needs exact completed input identities before use;
it is not an admitted experiment. `refine_authorized` remains false and
`refine_episode_indices` remains empty. Artifacts can only be written from the
canonical Z: checkout. No solver, import, check or model was executed to implement
this operation. Existing assembly requests and defaults retain their behavior.

The handler retains unchanged independent, sequential camera→object→hand, and
joint candidates, with the same admitted observation records and family noise
scales. Sequential and joint each have at most 15 total iterations/window,
four 16-frame windows/episode with four-frame overlap, 600 CPU seconds per arm,
768 variables/window and 256 admitted records/frame across all source artifacts.
A record may produce multiple scalar residuals; counts are reported separately.
The 600-second ceiling is cooperative between residual/Jacobian evaluations;
one SciPy least-squares call cannot be interrupted by this handler. Total output
has a 2 GiB ceiling. These are bounds, not measured resource estimates.

The solver uses finite-difference Gauss–Newton with Huber weights and damping,
through the existing NumPy/SciPy dependencies. It fits right-local rigid-pose
increments and supported world joint displacements. Rotation-vector and
translation-vector norms are bounded separately (0.1 rad/0.025 m defaults).
Original world gauge, calibration and metric scale, stereo baseline, mesh body
origins and hand shape remain fixed. The earliest supported camera of each window
is fixed; the earlier window owns every overlap frame and later overlap variables
stay fixed. Both pre/post window candidates and all attempted optimizer steps
are retained. Rank deficiency stops an arm before an arbitrary correction;
objective increase, invalid geometry and resource ceilings stop it explicitly.

Implemented residuals are calibrated reprojection, optical-z depth, world-point
and persistent-correspondence displacement, endpoint bone-length distance and
explicitly supported contact distance. Changes to consecutive supported-state
velocity form a timestamp-aware prior; measured rapid motion is preserved rather
than penalized toward zero velocity. Bone-length distance is only a length prior,
not a MANO anatomical model. Occluded, unsupported or rejected measurements are
recorded as rejected; support is never dropped in response to a trial step. Huber
objectives normalize by each family's original scalar residual count. The source
noise scales/defaults have no tuning evidence and must be frozen before any
future comparison. The handler does not select or promote an arm. Objective
reduction is a fit statistic; independent reserved evidence remains separate.

Silhouette distance, MANO pose/shape fitting, joint limits, signed penetration,
contact sliding and temporal acceleration are explicitly absent. Competing
symmetry/contact candidates require separately frozen input receipts and runs;
the handler does not average their hypotheses. Current inferred MANO FK joints
have `joint_observed=false`, so they cannot become optimized observed joints.
Current hand-contact phase candidates have no confirmed physical contact and
cannot create accepted contact constraints. Sparse hand-refinement intervals
need an explicit source-clock adapter; this handler deliberately requires full
aligned source arrays. This limitation is not concealed by filling joints.

The input camera and object estimates must share the exact completed camera run,
revision, source RGB frame indices/PTS and calibrated input grid. The camera's
exact `depth.generate` parent supplies frozen pixel intrinsics. Optional hands
must share the same camera and complete dense frame grid. Only the frozen public
training/tuning episodes are accepted, never 41, 43 or 44. Diagnostic labels,
reference scans and diagnostic provenance are refused.

Observation sources are explicit `{run, artifact, sha256}` entries; the named
artifact must be declared by that completed `video_only` producer. Each artifact
is JSON with `schema: scene_refinement_observations_v1`, `units: metres`,
`clock: camA_rgb_pts`, `time_base: [1,10240]`, `grid_sha256`, exact
`pose_run_sha256`/`camera_run_sha256`, and `observations`. Each row contains:

- `frame_index`, `video_pts`, `role: optimization`, `family`, `sigma` and
  `motion_class: static|dynamic`.
- `entity: world|camera|object:<slot>|hand:<side>:<joint>`, `local_point_m`,
  `target` and explicit boolean `observed`, `visible`, `accepted`, `occluded`.
  A world entity's point is already in world coordinates; hand points must be
  zero so they refer to joint centres. Object points use the frozen mesh frame.
- Reprojection/depth rows have `intrinsics_pixels` exactly equal to frozen input
  calibration. Depth adds `depth_valid`; correspondence/contact adds
  `identity_supported`. Contact additionally requires
  `physical_contact_supported: true`.
- Paired correspondence/bone/contact rows add `other_entity`,
  `other_frame_index`, `other_video_pts`, `other_local_point_m`, and explicit
  `other_observed`, `other_visible`, `other_occluded` endpoint masks.

`point3d`/`correspondence` targets use world metres; reprojection uses two pixels;
depth is positive optical-z metres; bone/contact targets are nonnegative metres.
Source path/hash/row index is attached by the consumer, never supplied as proof
by a request. Producer adapters must preserve measured body-point origins,
reserved-point separation and support identity. The separately disabled
`prepare_refinement` operation concretely adapts item07's stereo-anchored pose
receipt and item06's correspondence artifacts into this unified artifact. It
reuses the object-pose producer's admission helper to recheck accepted query
identity, both origin/current clocks, measured body-point backprojection and
same-pixel direct metric depth. It skips unresolved object symmetry and cameras
without observed support; it emits reprojection and supported optical-z rows,
not guessed hands, contact or static-background evidence. Source receipts,
completed parents and mesh origins remain hash-bound. At most 512 observed source frames,
128 points per object/frame, 256 total residual records/frame and 64 MiB JSON
are accepted; exceeding a bound fails rather than silently subsampling. The
illustrative `prepare_refinement.json` requires exact source identities.
`prepare_refinement_authorized` remains false. Actual admitted receipts/producer
runs, canonical storage and independent assessment remain runtime prerequisites.

Each candidate NPZ retains full `camera`, `object`, `hand` arrays and source `pts`;
these are research candidates, not an assemble-compatible promoted scene.
`original.npz`, arm outputs and pre/post window files preserve unchanged gaps.
`refinement.json` contains admitted/rejected support, fixed-objective statistics,
rank ambiguity and stopping reasons; `steps.jsonl` retains accepted/rejected
trial vectors with variable identities. `run.json` binds exact parent/source
hashes. No calibrated uncertainty or independent improvement is claimed.

Assembly can explicitly consume `object_pose.refine` and `hand_motion.refine`
when all original producer schema/grid/mesh/clock and exact camera/hand bindings
remain satisfied. Refined hands must retain their exact lifted baseline and
camera lineage. Refined poses retain their exact compose baseline. Their opening
summary is intentionally unchanged; assembly rejects a refined-pose run in
`resting` mode and requires the compose baseline for that route. No new candidate
is automatically selected by assembly.
