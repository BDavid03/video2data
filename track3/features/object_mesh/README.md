# Object mesh

Status: scan resolution, opening hull and opening TSDF implemented, never run;
scan registration and generative reconstruction remain proposed.

Moving-object observed point/TSDF fusion is source implemented in
[`fuse.py`](fuse.py), independently disabled by `fuse_authorized: false`.
This is unrun source; no new geometry or accuracy evidence exists.

This slice supplies a triangle mesh for every tracked object. The mesh has three
consumers with different needs: the pose tracker needs the right shape and the
right body frame, the simulator needs a closed shape at the right size, and the
CD-O leaderboard scores the shape itself.

## Role in scoring

| Consumer | Metric affected | What matters |
|---|---|---|
| CD-O leaderboard | CD-O | Surface shape only. The scorer rescales the submitted mesh to the reference's size and registers it before measuring, so units, origin and orientation are free (**Official**, `KIT/v2dlb/object_mesh.py:170-216`). |
| [object_pose](../object_pose/README.md) | AUC, SP-SR, MP-SR, RPE, MPPE | Shape, metric size and body frame. The scorer's alignment removes a world-side offset and never a body-side one, so a mesh whose origin or axes differ from the reference object's frame has its offset cancelled at frame 0 for slot 0, then can produce error as that object moves; for other slots the offset is never removed and is scored in full from frame 0 (**Official**, `KIT/v2dlb/object_pose.py:122-163`; derivation in the [kit reading](../../../reports/contract-evidence/2026-09-30/submission-kit.md), section 3). |
| Simulator | All pose metrics, through policy behaviour | A closed, correctly sized shape the policy can grasp |

## The two kinds of object

Eleven object names appear in the 20 scored episodes (**Official**,
`KIT/data/track_3_evaluation_objects.json`). Ten of them have a textured scan in
the public release. One does not.

| Object | Public scan | Scored episodes | Episode-object pairs |
|---|---|---|---|
| `white_pot` | Yes | 1, 3, 5, 6, 8, 28 | 6 |
| `blue_cup` | Yes | 10, 14, 20, 32, 33 | 5 |
| `plastic_dish_rack` | Yes | 14, 15, 16, 28, 30 | 5 |
| `beige_cup` | Yes | 8, 10 | 2 |
| `mini_sweeper` | Yes | 19, 30 | 2 |
| `mini_dust_pan` | Yes | 19, 38 | 2 |
| `white_pot_lid` | Yes | 1 | 1 |
| `water_pitcher` | Yes | 20 | 1 |
| `wooden_piece_1` | Yes | 25 | 1 |
| `wooden_piece_2` | Yes | 25 | 1 |
| `wooden_spoon` | **No** | 3, 5, 6, 15, 16, 32, 33, 35, 36, 37 | 10 |

Ten of the 20 scored episodes contain `wooden_spoon`, and in three of them it is
slot 0, the object the scorer's alignment is anchored on. In the other seven
episodes it is slot 1, so its body-frame offset is scored from frame 0. Public poses are
expressed "in that scan's own frame" (**Official**, dataset `track_3/README.md`).
For `wooden_spoon` there is no scan, so its reference body frame cannot be known
from published material. This is the largest open question in the project.

## Operations

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `resolve` | For each object of an episode, a mesh record naming its source | Perception environment | No |
| `reconstruct` | An opening hull or observed TSDF surface built from one episode's depth | Perception environment | No |
| `fuse` | Moving-object body-frame observed points, TSDF and surface support | Perception environment | No |

`register` is a proposed diagnostic operation, not an implemented CLI command.

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `track_3/public/mesh/<name>/<name>.glb` | Dataset | Geometry and texture of the ten scanned objects |
| `outputs/object_masks/.../frames` | [object_masks](../object_masks/README.md) | Object and hand masks |
| `outputs/depth/...`, `outputs/stereo_depth/.../anchor.json` | [depth](../depth/README.md), [stereo_depth](../stereo_depth/README.md) | Dense depth in metres |
| `outputs/camera_motion/.../trajectory.npz` | [camera_motion](../camera_motion/README.md) | Camera pose, to combine views of a resting object |

## Method

**Mesh for pose tracking and simulation (Design).**

- Scanned objects: use the public scan unchanged. Its frame is the frame the
  reference poses use, which no reconstruction can match as well. `resolve`
  also reads the public URDF's collision-mesh reference and copies that file
  unchanged, recording `collision_file` and `collision_sha256` separately from
  visual `file` and `sha256`. This includes the separate `_collision.glb`
  assets for `mini_sweeper`, `wooden_piece_1` and `wooden_piece_2`. A nonidentity
  collision origin or scale is refused rather than silently changing frames.
- `wooden_spoon`: use the reconstructed mesh, with a body frame set by a written
  rule until the organisers publish the scan or its frame. The provisional rule:
  origin at the centroid of the surface, x along the longest principal axis from
  bowl to handle end, z along the shortest. The rule is a placeholder. A wrong
  guess costs accuracy on every `wooden_spoon` episode.

**Mesh for the CD-O submission (Design).** Submit a mesh reconstructed from the
evaluation video for all 36 pairs. Submitting the organisers' own scans back to
them would not be a reconstruction, and the kit's instruction is to "Reconstruct
one triangle mesh" per pair (**Official**, `KIT/tracks/track_3/README.md:40`).
Whether the public scans may be submitted is left as an open question and the
design does not depend on the answer.

**Reconstruction.** Available and proposed methods are intended for future
comparison on development episodes where a reference scan is available. No
comparison or reconstruction has run:

| Method | How | Strength | Weakness |
|---|---|---|---|
| Depth fusion | While the object rests, the head moves around it. Fuse the masked metric depth from many frames into one volume using camera poses, then extract a surface. | Metric size comes directly from measurement. No new model. | Sees only the visible side. Thin parts such as the spoon are near the depth noise level. |
| Image-to-3D model | Give a generative model one clear, unoccluded crop of the object and take its complete mesh. Set metric size by fitting the mesh to the fused depth. | Completes the unseen side. Handles thin shapes. | New model and weights. The shape is plausible and not measured. |
| Opening hull | Fuse the object's opening-frame cluster points, add their table footprints, and take a closed hull. | No new model or package. Closed with a bottom on the table. | Fills every concavity. Crude for pots, racks and cups. |

The opening hull is the mesh the resting entry uses for `wooden_spoon` in
simulation and for all 36 pairs in its first CD-O submission
([entry tiers](../../../docs/design/entry-tiers.md)). It is recorded with
`method: opening_hull`.

The masks omit the bottom 1.5 cm above the table. Before taking the hull,
each voxel-fused point therefore contributes another point with the same x/y
and z = 0 in the table world. `frame_rule.bottom_completion` records this
assumption. The spoon centroid, principal axes and end-width sign decision
are computed from the completed hull's surface, including its bottom.

Both reconstruction methods accumulate opening measurements in chunks of at most
65,536 points. `_points.py` retains a coordinate sum and pixel count per occupied
voxel, so repeated observations keep their point weighting without retaining
every frame's cloud. `maximum_point_voxels` defaults to 500,000 per object. A run
that exceeds it fails and requires an explicit resolution or limit change; no
measurement is silently dropped. This bounds retained voxel state, not the cost
of loading one depth/mask frame or the downstream hull algorithm. `point_fusion`
records raw point counts, occupied voxels, spacing, cap and weighting in the run
record and each mesh record. Chunked floating-point summation can differ slightly
from the earlier all-at-once reduction.

**Observed depth fusion.** `reconstruct` also accepts `method: opening_tsdf`;
[`reconstruct-tsdf.json`](reconstruct-tsdf.json) is an explicit request template
with unset producer paths. It uses the same stationary opening clusters,
observed stereo scales, camera transforms, PTS and object assignments as the hull.
It does not fuse moving-object frames or use public scan geometry.

The private `_fusion.py` constructs one bounded metric grid per object. Grid
points project into each masked depth frame; the signed optical-axis distance
is clipped to a truncation band and averaged with equal weight per frame.
Samples more than one band behind the visible surface are excluded. Extraction
uses marching tetrahedra only on cubes whose eight corners each have at least
`minimum_fusion_views` observations. Shared edge vertices and field-gradient
winding keep adjacent extracted triangles consistent. Unknown corners never
create an inferred back surface.

Defaults are 2 mm grid spacing, a four-voxel truncation band, two contributing
frames per corner, at most two million grid points and 200,000 output faces.
Generated surface vertices are also capped at three times the face limit,
including vertices later discarded with degenerate triangles.
Exceeding a bound fails with a request for an explicitly coarser grid;
the implementation does not silently reduce resolution. Fusion reads frames
again per object instead of retaining every depth image. Grid projection and
extraction use chunks. Opening point collection now retains bounded voxel sums
and counts; input frames, mesh storage and library working memory still contribute
to memory use. Total cost remains unmeasured.

This method can retain observed concavities, but its surface can be open,
fragmented or noisy. Frame count does not establish independent views or adequate
coverage. It does not complete bottoms, correct depth bias, infer unseen geometry,
or establish an accuracy improvement. Empty extracted surfaces fail. TSDF meshes
use the observed surface centroid and world axes for CD-O, which removes rigid
frame and scale; they do not claim the spoon's provisional collision frame.
`resolve` continues to require `opening_hull` for the spoon used in simulation.
Campaign queue requests can explicitly choose `submission_mesh_method:
opening_tsdf`; spoon collision tasks always specify `opening_hull`. The default
submission method remains the hull pending separately authorized evidence.

**Two-pass spoon workflow.** Resolve with `scans_only: true` to publish the
listed scans and `pending_objects: ["wooden_spoon"]`. This can be an empty
mesh list in a spoon-only episode. Object_pose `initialise` registers the scans
first and assigns the unique unexplained cluster to the spoon. Pass that
initialisation's `pose_run` to `reconstruct` with the same masks, depth, anchor
and camera runs: their paths and hashes must match the pose run's recorded
inputs. Leave `objects` empty to reconstruct the pending spoon automatically;
no manual `cluster_assignments` are needed. Cluster layer identity is consumed
as recorded by opening-clusters, without another centroid reassignment.
Resolve again with `scans_only: false` and this `reconstruction_run`, then
initialise again to obtain all opening poses. The hull records its camera run
and `world_T_object`; the second pass preserves that frame's sign with a
90-degree orientation bound. `scans_only` does not bypass any execution flag.

**The spoon's provisional frame.** The rule above fixes the axes only up to
sign, and a wrong sign is a 180° error
([error budget](../../../docs/design/error-budget.md)). Before the rule is used,
the `frame-convention` report in
[public_diagnostics](../public_diagnostics/README.md) is read to see where the
ten published scans put their origin and which axis points up at rest. If they
share a convention, `sources.json` records it for `wooden_spoon` in place of the
centroid-and-principal-axes rule, with a note naming the report.

The upstream baseline uses an image-to-3D model whose weights are gated
(source reading in `reports/model-readiness/README.md`). Any such model is a new
dependency and needs the user's confirmation.

**Reuse across episodes (Design).** The same physical object appears in several
scored episodes. Each episode's pair is scored separately. The design
reconstructs and submits per episode by default. Reusing the best reconstruction
for other episodes of the same object remains disabled until the organisers
explicitly permit it in Q06. If permitted, record its source split, episode and
frames; no pose trajectory is reused.

## Outputs

`outputs/object_mesh/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Meaning |
|---|---|---|
| `<object>.glb` | | Triangle mesh, metres, in the object's body frame |
| `<object>_collision.glb` | | Unchanged public URDF collision mesh, when resolving scans |
| `mesh.json` | `source` | `public_scan` or `reconstructed` |
| | `file`, `sha256`, `collision_file`, `collision_sha256` | Visual asset and optional separate collision asset identities |
| | `pending_objects` | Top-level list; only the spoon may be pending in scans-only resolve |
| | `body_frame` | `scan_frame`, `registered_to_scan` or `provisional_rule` |
| | `method`, `source_split`, `source_episode`, `source_frames` | For reconstructed meshes; submission enforces the Q06 reuse rule |
| | `fusion` | For TSDF: bounds, spacing, truncation, contributing-frame counts, extraction rule and no-completion statement; null for hulls |
| | `point_fusion` | Raw measurement count, occupied voxel count, point-weighted reduction and explicit capacity for either method |
| | `vertices`, `faces`, `closed` | Counts and whether the surface is closed |
| | `extent_m` | Bounding-box size, as a sanity value for consumers |
| `run.json` | | Request, input hashes, status, timing |

The submission packer reduces each mesh to its budget, so this slice does not
decimate. The config states 4,096 vertices and 4,096 faces
(**Official**, `KIT/config/tracks.json:280-281`).

## Configuration and authorization

`runtime.json` holds `resolve_authorized`, `reconstruct_authorized` and
`register_authorized`, all false. `sources.json` maps each object name to its
allowed source, so a run cannot quietly fall back from a scan to a reconstruction.

## Dependencies

Hull and TSDF reconstruction use NumPy and `trimesh` from the locked MoGe2
environment (`configs/perception/moge/pylock.python313.toml`); convex hulls also
use its SciPy dependency. TSDF surface extraction is implemented locally with
marching tetrahedra and adds no package. An image-to-3D model remains a proposed
new dependency with weights.

## Resources

Unmeasured. Depth fusion is CPU work. An image-to-3D model needs the GPU and
takes the shared lock.

## Failure handling

`resolve` fails when an object has no allowed source, except for a pending spoon
in explicitly selected scans-only mode. `reconstruct` records
`insufficient_views` when the object is never seen unoccluded.

## Separation of public labels

Public scans are assets, not pose labels, and the design uses them on both
splits. `register` and the development comparison of reconstructed shape against
scan carry the `public_asset_diagnostic` tag. No pose label is read.

## Acceptance evidence

The TSDF implementation and campaign selection are source changes only. No tests,
smoke checks, validation commands, project imports, reconstruction, model execution
or installation ran. All authorization flags remain false.

Once runs are authorized: on development episodes, the reconstructed mesh of a
scanned object, scored against its scan with the kit's CD-O code, gives a number
to compare methods by. `public_diagnostics.mesh` can compare a named baseline
reconstruction against the candidate for the same episode and object set. It
reports raw-mesh distances and deltas, without submission-budget decimation.
The spoon has no reference scan and is explicitly unscorable in that report;
its length still needs comparison with metric depth measurements.

## Moving observed fusion (source only)

`MovingFusionRequest` and [`fuse.json`](fuse.json) declare the independently
disabled `fuse` CLI operation. The handler requires a completed
`object_pose.track` and its exact named-mask, depth, anchor and camera parents,
including hashes, pixel-grid identity, full frame-index/PTS agreement and
`video_only` provenance. It reads no reference scan, generator, reserved pose
labels or weights. Prior mesh metadata identifies the tracking body's origin
and axes; mesh geometry is not loaded. It retains that frame without recentring
the observed surface. This does not remove bias inherited from the tracker.

For each admitted frame, `world_T_object = world_T_camA @ camA_T_object` and
`object_T_camA = inverse(world_T_object) @ world_T_camA`. Point accumulation
and TSDF projection use this transform, so moving measurements share one
object frame. Valid measured poses, finite pose confidence, sampled surface
support, finite registration residual and observed metric scale must pass
request thresholds. No interpolation is admitted. Hand pixels and overlapping
object identities are excluded. Missing support produces a recorded exclusion;
too few admitted views or observed voxels fails without completion or fallback.
Historical opening confidence arrays are invalid and cannot silently pass.

The default interval samples at most 120 frames, stride five over frames
0–599, clipped to clip length. Only training episodes 0/11/12/18/21/23/26/31
and tuning episodes 39/42 are accepted. Episode 41, consumed holdouts 43/44,
other public episodes and evaluation are rejected by the request. Thresholds
are proposals to freeze using real tuning evidence, not fitted values. Existing
authorization for `reconstruct` does not enable `fuse`. A future invocation is
`python -m track3.features.object_mesh fuse --request <explicit-request.json>`;
it remains disabled and has not been invoked here.

The handler reuses `_points.py` and `_fusion.py`. Observed voxel centroids form
the point-fusion baseline; marching tetrahedra supplies the TSDF candidate
without neural or generative completion. Retained point, grid and mesh bounds
match the opening request and apply one object at a time. The mesh preserves
`mesh.json` fields `name`, `file`, `sha256`, units, source episode, source frames,
extent and camera identity. `body_frame` and `inherited_body_frame` retain the
tracking convention; `world_T_object` names the first admitted reference frame.
This method does not enter the existing spoon hull collision route or promote
submission.

Each object additionally names a hashed `<name>_support.npz`, also listed in
the run's artifact map. It stores `centroids_object_m`, `voxel_cells`,
`point_counts`, `tsdf_weight`, `tsdf_sum`, `grid_origin_object_m` and
`voxel_size_m`. `point_counts` counts measured pixels; `tsdf_weight` counts
contributing frames per grid point. These are not independent view coverage,
complete-shape coverage or calibrated uncertainty. Extraction requires every
cube corner to meet the minimum frame count; unknown regions remain unknown.
`completion: false` and `collision_ready: false` separate observed surfaces
from filled collision geometry. Per-object/per-frame admission and exclusion
reports persist in `run.json`, including failures.

Pose noise, mask contamination, depth bias, repeated viewing angles and
unresolved symmetry can still smear or duplicate surfaces. No view-diversity
estimator, free-space consistency rejection or per-face uncertainty calibration
is claimed. FoundationPose and BundleSDF remain deferred pending admitted
geometry, licensing/assets and dependency approval. No package, test, check,
project import, model, fusion run or runtime output was executed for this build.

## Open questions

| Question | Who can answer |
|---|---|
| Will the `wooden_spoon` scan be released, or its body frame described? | Organisers |
| May the public scans be used at evaluation for tracking and simulation? | Organisers |
| May the public scans be submitted to the CD-O leaderboard? | Organisers |
| May one reconstruction be submitted for several episodes that contain the same object? | Organisers |
