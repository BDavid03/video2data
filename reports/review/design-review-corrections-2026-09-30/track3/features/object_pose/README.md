# Object pose

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

This slice estimates the six-degree-of-freedom pose of every tracked object in
every frame, first relative to the colour camera and then in the world. Its output
is the demonstration the robot policy is trained to reproduce. What the policy
reproduces is what gets scored.

## Role in scoring

All five pose metrics compare the rolled-out object trajectory with a confidential
reference (**Official**, `KIT/tracks/track_3/README.md:60-71`). The policy can
at best reproduce the trajectory this slice gives it, so this slice sets the
ceiling for the entry.

Three properties of the scorer shape the design. All are **Official**, from
`KIT/v2dlb/object_pose.py`.

| Property | Lines | Consequence |
|---|---|---|
| One rigid transform is fitted from object slot 0 at frame 0 and applied to everything. | 122-163 | The first-frame pose of slot 0 is free of cost in itself, and decides the cost of everything else. An error in its orientation rotates the whole episode about that object: a 10 degree error moves a point 30 cm away by about 5 cm. |
| AUC uses plain average distance over fixed object points, with no symmetry handling. | 80-104 | A cup rotated about its own axis looks the same and scores as wrong. Rotation about a symmetry axis must be tracked from texture or handles. |
| MP-SR requires every object to stay under 3 cm and 30 degrees on average. SP-SR averages over objects, under 10 cm and 0.5 rad. | 173-191 | The second object counts as much as the first for MP-SR. |

RPE compares the position of object 1 in object 0's body frame, so it is
unaffected by world alignment and directly measures the relative layout this
slice reconstructs (lines 194-224).

## Operations

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `initialise` | The pose of each object over the opening frames, before any contact | Perception environment | Yes |
| `track` | Per-frame camera-relative poses for one episode | Perception environment | Yes |
| `compose` | World poses, from camera-relative poses and camera motion | Perception environment | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `ego_cam_a` clip on the pinhole grid | Dataset, [depth](../depth/README.md) | Colour frames, intrinsics |
| `outputs/object_mesh/.../<object>.glb`, `mesh.json` | [object_mesh](../object_mesh/README.md) | Geometry, texture, body frame record |
| `outputs/object_masks/.../frames` | [object_masks](../object_masks/README.md) | Object masks, hand masks |
| Depth in metres | [depth](../depth/README.md), [stereo_depth](../stereo_depth/README.md) | Dense depth and validity |
| `outputs/camera_motion/.../trajectory.npz` | [camera_motion](../camera_motion/README.md) | `world_T_camA`, validity |

## Method

1. **Initialise from the quiet opening.** Before the hands arrive, the objects
   rest on the table and are fully visible. Estimate each object's pose on every
   opening frame, express it in the world, and take a robust average. The object
   is not moving, so the spread across these frames is a direct, label-free
   measure of combined pose and camera error. The end of the opening is the first
   frame in which a hand mask touches the object's mask.
2. **Track.** From the opening onward, refine the pose frame to frame against
   the mesh, using colour, depth and the mask. When the hand covers most of the
   object, the track continues on what remains visible and the confidence drops.
3. **Recover.** When confidence falls below the floor, mark the pose invalid,
   and re-initialise at the next frame where the object is sufficiently visible.
   Invalid frames stay invalid here. Filling them is
   [scene_assembly](../scene_assembly/README.md)'s job.
4. **Compose.** `world_T_object = world_T_camA * camA_T_object`, for frames where
   both factors are valid. Write `world_valid = valid AND camera_motion.valid`
   per object; retain camera-relative estimates when only the camera is missing.

**Model choice (Design).** A model-based tracker that takes a mesh, colour, depth
and a mask. FoundationPose is the candidate in the upstream baseline and the
strongest known option for textured, scanned objects. It has two costs recorded
in `reports/model-readiness/README.md`: its licence limits use to research and
evaluation, and it depends on compiled components whose build recipes do not list
the RTX 5080's architecture. The fallback is geometric registration of the mesh
to the masked depth points, which needs no new model but cannot resolve rotation
about a symmetry axis.

**Order of work (Design).** Start with the ten scored episodes that contain only
scanned objects: 1, 8, 10, 14, 19, 20, 25, 28, 30 and 38. Their meshes and body
frames are known. The ten `wooden_spoon` episodes follow, once a spoon mesh
exists.

## Outputs

`outputs/object_pose/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Type and shape | Meaning |
|---|---|---|---|
| `poses.npz` | `object_names` | string B | Slot order, equal to the official order |
| | `camA_T_object` | float64 T×B×4×4 | Object body frame into colour-camera optical frame |
| | `world_T_object` | float64 T×B×4×4 | After `compose`; NaN where `world_valid=false` |
| | `valid` | bool T×B | A camera-relative pose was estimated from this frame; `camA_T_object` is NaN otherwise |
| | `world_valid` | bool T×B | Object `valid` AND camera_motion `valid`; mask for `world_T_object` |
| | `confidence` | float32 T×B | Tracker score, not a calibrated probability |
| | `visible_fraction` | float32 T×B | Share of the rendered object not covered by a hand mask |
| | `reinitialised` | bool T×B | True on a frame where tracking restarted |
| | `frame_index`, `pts` | int64 T | Frame identity |
| `opening.json` | per object | | Opening frame range, averaged world pose, spread in metres and degrees |
| `run.json` | | | Request, input hashes, mesh body-frame record, status, timing |

Matrices are used inside the project. Quaternions appear only at the two external
interfaces: scalar-first in the upstream motion format and scalar-last in the
rollout file. Conversion happens once at each, in the slice that owns the
interface.

## Configuration and authorization

`runtime.json` holds `initialise_authorized`, `track_authorized` and
`compose_authorized`, all false. The request names the episode, the mesh run, the
confidence floor and the visibility floor for re-initialisation.

## Dependencies

Proposed, not installed. Each needs the user's confirmation.

| Item | Licence | Note |
|---|---|---|
| FoundationPose code and two weight sets | NVIDIA Source Code License, research and evaluation use | Sizes and hashes unknown; compiled components |
| A differentiable or offscreen renderer | Varies | Required by the tracker |

The challenge's release terms say pretrained models under an incompatible licence
need not be released with a winning entry
(**Official**, `KIT/config/tracks.json:335`). Whether a research-only tracker may
be used at all is the organisers' call.

## Resources

One GPU stage at a time through the shared lock. Memory and runtime unmeasured.

## Failure handling

`initialise` fails the episode when an object is never found in the opening
frames, because every later step depends on it. `track` keeps going with invalid
frames and records their count.

## Separation of public labels

Evaluation runs read no label. On development episodes, comparing
`world_T_object` with the public poses is the main accuracy check for the whole
perception stack. That comparison uses the scorer's own alignment rule, carries
the `public_label_diagnostic` tag, and its results never become inputs to a
later run. Validation episodes are touched only after settings are frozen.

## Acceptance evidence

Once runs are authorized, on development episodes and using the kit's metric
code: the reconstructed trajectory, scored as if it were a rollout, gives AUC,
MPPE and RPE values. These are the numbers a perfect policy would receive, and
they tell the project whether perception or control is the weaker half.

## Open questions

| Question | Who can answer |
|---|---|
| Do reference poses cover frames where the object was invisible to motion capture, and how are those rows scored? Every roster frame is scored as far as the public code shows. | Organisers |
| Do `blue_cup` and `beige_cup` have handles or texture that fix rotation about their axis? | Inspecting the public scans |
| May a research-only pose tracker be used? | Organisers |
