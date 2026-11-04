# Object pose

Status: implemented, never run

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
| `initialise` | The pose of each object over the opening frames, before any contact | Perception environment | No, geometric method |
| `track` | Per-frame camera-relative poses for one episode | Perception environment | No, geometric method |
| `compose` | World poses, from camera-relative poses and camera motion | Perception environment | No |

### Implemented geometric tracking

`track.py` implements the planned geometric backend, exposed through `track.json`
and `python -m track3.features.object_pose track`. This command is reference
documentation only: `track_authorized` remains false, as do evaluation and
validation permissions, and the episode allowlists remain empty. It has never
been imported, executed or tested. No new dependencies were added.

The request names completed opening, mesh, full-video mask, depth, stereo, anchor
and camera producers. Opening and tracking must use the exact same mesh, depth,
anchor and camera manifests. All producers must be video-only, on the same
episode, revision and pinhole grid. The full mask producer must write instances
in the exact object slot order followed by `hand_0`, `hand_1`. Opening clusters
cannot substitute for named full-video masks. The mask producer is implemented
but its optional model backend remains unprovisioned and unexecuted.

Tracking retains the measured opening camera poses and processes every subsequent
RGB frame. For each object it removes dilated hand masks and overlapping object
mask pixels, then registers the body mesh against the remaining metric points.
Observed per-frame anchor scales supply metric monocular depth; direct valid
stereo overrides those pixels. Missing scale can use direct stereo but never an
episode median or an interpolated scale.

The last observed pose is moved into the current camera frame using the selected
camera trajectory when both camera poses are valid. A local trimmed ICP update
must satisfy residual, sampled-surface support, confidence and motion limits.
After a gap, missing camera compensation, or a rejected local update, recovery
tries the previous seed and 24 proper axis permutations, translated to the
observed cloud and refined with ICP. Recovery requires more surface support and
rejects competing poses that are geometrically indistinguishable and unresolved
by colour. This coarse recovery search does not guarantee a globally correct
pose. Continuous tracking relies on geometric registration and its temporal seed;
rotation about a featureless symmetry axis remains a limitation.

`confidence` is geometric inlier share multiplied by
`exp(-mean_residual / residual_limit)`, not a calibrated model probability.
`surface_support` measures the fraction of projected sample pixels supported by
the visible object mask and metric depth, retaining the nearest sample at each
pixel. It is not rendered visibility: `visible_fraction` remains NaN with
`visible_fraction_valid=false`. The NPZ additionally records
`registration_residual_m`, `surface_support` and `confidence_valid` per frame and
object. Copied opening poses have no invented tracking confidence.

Tracking loss leaves camera poses NaN/invalid. Recovery sets `reinitialised` only
for an accepted new observation. World arrays remain NaN until `compose`, which
now accepts the tracker and its full-video mask lineage. A reconstructed mesh's
opening orientation defines the body frame without restricting later motion to
the opening's 90-degree orientation prior.

The run records every object's outcome and observed/lost frame counts. Failure
or interruption during decoding/tracking writes the partial arrays with a failed
or interrupted run status; downstream consumers require a completed producer.
Missing artifacts or mismatched timestamps fail the operation rather than being
interpreted as an ordinary tracking loss. A completed tracking run can contain
gaps; scene assembly still decides whether those gaps are bridgeable.

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

**Initialising without a new model (Design).** The resting entry
([entry tiers](../../../docs/design/entry-tiers.md)) needs opening poses before
any tracker is approved. `initialise` therefore has a geometric method, selected
in the request, that uses only installed packages:

1. Read the clusters from object_masks `opening-clusters` and lift each to a
   metric point cloud in the world, fused over the opening frames.
2. For each scan the episode lists, and each cluster, register the scan's
   surface points to the cluster's points: several starting rotations that stand
   the object on the table in each of its stable resting orientations, then
   iterative closest point. Stable poses use the convex hull's centre of mass,
   because the visual scan may be open. Only the camera-facing side of the object is seen,
   so the fit is scored one way, from observed points to the scan.
3. Assign names to clusters by the assignment with the lowest total residual
   over the episode's objects. A cluster that no listed scan fits is
   `wooden_spoon` when the episode lists it. This requires exactly one leftover
   cluster and a fit under the residual limit for each listed scan; a leftover
   that another listed scan also fits is refused. Cluster layers already have
   stable physical identities from `opening-clusters` and are not reordered here.
4. For an object whose fit is nearly unchanged by rotation about an axis, search
   that rotation by comparing the scan's vertex colours with the image colours
   at the projected vertices. Starting at the best coarse candidate, sweep the
   geometrically flat axis in one-degree increments with colour alone deciding
   among geometrically equivalent poses. Unmeasured colour receives the worst
   measured colour cost, never a free zero cost. If colour leaves alternatives
   more than two degrees apart within `ambiguity_color_margin`, or a distant
   alternative has no colour measurement, set `rotation_ambiguous` and record
   the axis. A colour-refined opening orientation is held during each frame's
   translation refinement so ICP cannot undo the colour decision.
5. When there are fewer clusters than objects, two objects started in contact.
   Register the larger listed scan to the merged cluster, remove the points it
   explains, and register the next scan to what is left. The result is accepted
   only when both residuals are under the request's limit. Otherwise
   `initialise` fails the episode with `opening_objects_inseparable`. A spoon
   needs its own leftover cluster; merged scan/spoon openings are refused.
6. Average over the opening frames as in step 1 of the main method.

Two cups of the same shape, as in evaluation episode 10, differ only in colour.
The assignment in step 3 then uses the colour comparison from step 4.

**Order of work (Design).** Start with the ten scored episodes that contain only
scanned objects: 1, 8, 10, 14, 19, 20, 25, 28, 30 and 38. Their meshes and body
frames are known. The ten `wooden_spoon` episodes follow, once a spoon mesh
exists. Spoon episodes use two passes, without manual cluster assignments:

1. Run object_mesh `resolve` with `scans_only: true`, then `initialise` with
   that mesh run and the opening masks, depth, anchor and camera runs. It fits
   the listed scans first and records the leftover as `wooden_spoon` in
   `opening.json.object_assignments`. `pending_objects` contains the spoon,
   `initialisation_pass` is `scans_only`, and its pose remains NaN/invalid.
   A spoon-only episode likewise assigns its sole cluster without inventing a pose.
2. Run object_mesh `reconstruct` with this `pose_run` and the same masks, depth,
   anchor and camera runs. With `objects` empty it selects the pending spoon.
   The existing spoon frame-convention review requirement still applies.
3. Resolve again with `scans_only: false` and `reconstruction_run` set, then
   initialise again with the complete mesh run and the same camera. Registration
   of the same-episode hull starts at its recorded `world_T_object`; rotations
   more than 90 degrees from that frame are rejected, including during refinement.

The assignment pass is a completed assignment artifact, not a complete opening
pose estimate. `compose` and scene assembly refuse it until the second pass.
`compose` requires the camera run recorded by its pose input. For a full scene,
its optional `hand_run` binds the selected completed hand-motion run into the
pose lineage and requires that hand run to use the same camera. Scene assembly
then requires these exact camera/hand run paths and hashes, plus matching mesh
hashes. No hand-motion implementation or execution is added here.

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
| `opening.json` | per object | | Opening frame range, averaged world pose, spread in metres and degrees, method (`tracker` or `geometric`), registration residual, `rotation_ambiguous` and its axis |
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

Except for the explicitly pending spoon in the scans-only assignment pass,
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

## Bounded correspondence refinement (source only)

`refine.py`/`refine.json` add the disabled `refine` operation. It accepts a completed
`object_pose.compose`, exact camera trajectory, accepted
`object_masks.correspondences` and direct `stereo_depth.generate` metric evidence.
The correspondence and stereo producers must share their depth.generate parent.
No model, renderer, new dependency, scan or label is loaded. The input receipt
is an explicit admission boundary, not an implemented receipt generator. Missing
receipts/identity/stereo support remain blockers; empty support never creates a
successful correction.

The separately SHA256-pinned observations JSON uses `schema_version: 1`,
`provenance: "video_only"`, `body_point_origin: "measured_depth_backprojection"`,
`split`, `episode_index`, `grid`, exact `mesh_body_frames`, and `parents` equal to
ordered `{path, sha256}` input_file entries for pose, camera, correspondences and
stereo runs. `time_base` is the exact source `[numerator, denominator]`.
`symmetry_alternatives` maps every object name to a list of proper 4×4 body-frame
alternative transforms (empty only when none are admitted). Unresolved alternatives
and the opening's ambiguous rotations preserve the original trajectory.

Each `observations` row identifies `frame_index`, `video_pts`, `object_name`,
`query_ids`, `object_points_m` N×3, `image_points_px` N×2, `depth_m` N (positive
metres or null), `intrinsics_pixels` 3×3, `visible`/`supported` boolean N, and
positive `pixel_sigma`/`depth_sigma_m`. Each point's `origin_evidence` gives
`depth_run_sha256`, `pose_run_sha256`, `observed: true`, `metric_valid: true`,
`frame_index`, `video_pts`, and integer `pixel_xy`. Origins must match the accepted
track query's origin. The handler reads the pinned stereo artifact and recomputes
the body point using the original accepted camera-relative pose, checking equality;
current depth residuals similarly require direct same-pixel stereo evidence.
Intrinsics must equal the selected metric grid. Visible current tracks must be
geometrically accepted and agree with the receipt pixels. Six spatially
non-collinear correspondences are required; no synthetic visibility is inferred.

Each disjoint window optimizes right body-frame pose increments with robust
calibrated reprojection and supported metric-depth residuals, correction bounds,
and timestamp-aware changes to the measured velocity. Actual measured motion is
retained as the temporal reference. Loss, reinitialization, unsupported frames,
clock gaps and rapid measured translation/rotation break windows. There is no gap
fill, constant-motion assumption or silhouette renderer. Converged corrections
must improve image evidence; exhausted/failed fits preserve the baseline. Bounds
are conservative source defaults, not tuned or experimentally established values.
Disjoint windows may leave boundary discontinuities; this is not a globally
optimal trajectory or independent held-out evidence.

`poses.npz` retains originals under `unrefined_` for world/camera poses and validity,
adds `refinement_supported` T×B and `correction_object_se3` T×B×6. The latter uses
rotation-vector first and translation second, with `refined = original @ delta`;
translation is the right increment, not a world displacement or coupled SE(3)
exponential coordinate. Unsupported corrections remain zero and never enable
invalid frames. `refinement.json` retains robust before/after costs, residual
counts, solver status, budgets, rejected windows and symmetry alternatives.
`opening.json` intentionally retains the original opening summary, explicitly
marked in the refinement report. Consumers must explicitly admit the research
candidate; existing scene production is not silently redirected.

All flags, including `refine_authorized`, remain false. The request admits only
training/tuning episodes 0/11/12/18/21/23/26/31/39/42. Canonical Z: output admission
remains enforced. FoundationPose and BundleSDF launches/imports/weight provisioning
remain deferred pending approved pins, dependencies and actual geometric evidence.
No runtime outputs, imports, tests or verification commands were produced.

Composition also accepts `hand_motion.refine` with the same camera/grid and the
exact original `hand_motion.lift` parent; both parent links are checked. This adds
explicit lineage admission and leaves its existing authorization gates unchanged.

The public producer-owned `admit_refinement_observations(request, parents, result,
motion, record, np)` helper shares receipt admission with scene preparation. It
performs no optimization or artifact writes, returns the receipt plus supported
rows keyed by `(frame_index, object_slot)`, and appends consumed artifact hashes
to the caller's record. Parents are ordered compose/camera/correspondences/stereo;
callers retain their own disabled authorization, exact producer/clock checks and
canonical output boundary. It never mutates the loaded pose/camera arrays.
