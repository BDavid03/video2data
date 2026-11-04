# Object mesh

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

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
| `reconstruct` | A mesh built from one episode's video | Perception environment | Yes |
| `register` | A rigid transform from a reconstructed mesh to a public scan of the same object | Perception environment | No |

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
  reference poses use, which no reconstruction can match as well.
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

**Reconstruction.** Two methods, compared on development episodes where the
public scan is available as a reference shape:

| Method | How | Strength | Weakness |
|---|---|---|---|
| Depth fusion | While the object rests, the head moves around it. Fuse the masked metric depth from many frames into one volume using camera poses, then extract a surface. | Metric size comes directly from measurement. No new model. | Sees only the visible side. Thin parts such as the spoon are near the depth noise level. |
| Image-to-3D model | Give a generative model one clear, unoccluded crop of the object and take its complete mesh. Set metric size by fitting the mesh to the fused depth. | Completes the unseen side. Handles thin shapes. | New model and weights. The shape is plausible and not measured. |

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
| `mesh.json` | `source` | `public_scan` or `reconstructed` |
| | `body_frame` | `scan_frame`, `registered_to_scan` or `provisional_rule` |
| | `method`, `source_split`, `source_episode`, `source_frames` | For reconstructed meshes; submission enforces the Q06 reuse rule |
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

Depth fusion: NumPy, SciPy, OpenCV and `trimesh`, all in the locked MoGe2
environment (`configs/perception/moge/pylock.python313.toml`). Surface extraction
from a fused volume needs a marching-cubes routine, which that environment does
not list; adding one is a new dependency. An image-to-3D model is a new
dependency with weights.

## Resources

Unmeasured. Depth fusion is CPU work. An image-to-3D model needs the GPU and
takes the shared lock.

## Failure handling

`resolve` fails when an object has no allowed source. `reconstruct` records
`insufficient_views` when the object is never seen unoccluded.

## Separation of public labels

Public scans are assets, not pose labels, and the design uses them on both
splits. `register` and the development comparison of reconstructed shape against
scan carry the `public_asset_diagnostic` tag. No pose label is read.

## Acceptance evidence

Once runs are authorized: on development episodes, the reconstructed mesh of a
scanned object, scored against its scan with the kit's CD-O code, gives a number
to compare methods by. The reconstructed spoon's length matches its length
measured in metric depth.

## Open questions

| Question | Who can answer |
|---|---|
| Will the `wooden_spoon` scan be released, or its body frame described? | Organisers |
| May the public scans be used at evaluation for tracking and simulation? | Organisers |
| May the public scans be submitted to the CD-O leaderboard? | Organisers |
| May one reconstruction be submitted for several episodes that contain the same object? | Organisers |
