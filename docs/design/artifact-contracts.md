# Artifact contracts

What every run writes, and what a consumer must check before it uses another run's output. Exact
fields per operation are in each slice's README and source.

## Where outputs go

Every generated file stays inside the checkout. `src/paths.py` (`output_path`) refuses writes
unless the checkout is listed under `artifact_roots` in `data/configs/project.json`, and refuses
writes into `upstream/`, `.git` or the dataset snapshot.

Outputs live under `data/outputs/<slice>/...`. Episode outputs usually sit under
`data/outputs/<slice>/<split>/episode_XXXXXX/<run-id>/`. Policy and pack slices document their own layout.

## The run record (`run.json`)

`src/runs.py` (`recorded`) writes `run.json` before and after the work. On failure it marks the run
`failed` or `interrupted` and closes any running frame rows. At the end it hashes every file in the
run folder into `artifacts`. `src/provenance.py` (`run_fields`) fills the shared fields:

| Field | Meaning |
|---|---|
| `kind` | Slice and operation, such as `mesh_extents.measure` |
| `status` | `running`, then `completed`, `failed` or `interrupted`; only `completed` can feed another run |
| `request` | The resolved request |
| `inputs` | Project-relative paths and SHA-256 hashes, including parent run records |
| `sources` | Hashes of the slice's Python files, the shared modules, the source lock and the split |
| `provenance` | One of the three tags below |

Slices add timings, frame outcomes, backend identity and resource use.

**Consumers check bytes, not names.** Name the parent `run.json` explicitly. `read_run` checks its
kind, status and provenance. Then re-hash each artifact you use against the parent's recorded hash.
A matching file name or a newer time stamp is not lineage.

A completed perception run can still hold invalid frames: its validity arrays say which values
exist. JSON uses finite numbers or explicit nulls. Arrays may use NaN under their validity masks.

## Provenance tags

| Tag | Meaning |
|---|---|
| `video_only` | Built from video only; allowed for production, subject to each slice's split rules |
| `public_asset_diagnostic` | Used a public scan or asset; diagnostic only, never submitted |
| `public_label_diagnostic` | Used public pose labels; diagnostic only, never used to build a submitted result |

A child takes the strongest tag of its parents. Missing or unknown tags are refused. Copying a file
does not change its tag. Submitted meshes must also say `source = reconstructed` with the right
episode: a public scan is never a submitted mesh.

## Frames, clocks and units

Consumers must agree on dataset revision, split, episode, object order, producer and image grid.
Equal image size does not mean equal intrinsics. The depth slice owns the pinhole grid.

| Quantity | Meaning |
|---|---|
| `frame_index` | Video frame number; array rows follow the declared grid |
| `pts` / `video_pts` | Integer MP4 presentation time stamp |
| Reference/rollout clock | Exact 20 Hz match used by the policy and scorer |
| `world_T_camA` | Colour-camera optical frame to metric world |
| `camA_T_object` | Object body frame to camera frame (use the producer's exact field name) |
| `world_T_object` | `world_T_camA` composed with `camA_T_object` |

Optical frame: x right, y down, z forward. World +z is table-up; the table is assumed level. Mesh
body axes are separate from world axes. All metric geometry is in metres. Convert between matrix,
axis-angle and quaternion only at named boundaries. Upstream hand export uses scalar-first
quaternions. Never guess quaternion order from four numbers.

## Perception outputs

`T`, `H`, `W` = frame count, grid height, grid width.

| Slice | Main output | How to read it |
|---|---|---|
| [depth](../../src/features/depth/README.md) | Per-frame depth, validity, preprocessing metadata | Depth and intrinsics belong to the recorded grid |
| [stereo_depth](../../src/features/stereo_depth/README.md) | `rectification.npz`, per-frame disparity and depth | Rectified depth and colour-grid `rgb_depth_m` are different frames |
| stereo anchor | `anchor.json` | Per-frame scale, support and spread; only measured scales make monocular depth metric |
| [object_masks](../../src/features/object_masks/README.md) | Per-frame masks and instance records | Named object slots, then anonymous hand slots |
| [camera_motion](../../src/features/camera_motion/README.md) | `trajectory.npz`, `world.json` | `world_T_camA` float64 `T×4×4`; `valid`, `observed` bool `T`; invalid poses are NaN |

Direct stereo beats scaled monocular depth. A frame with no scale is not given an interpolated one.
Tracker confidence is a heuristic, not a probability.

## Meshes

Mesh records give source type, body-frame rule, visual and collision file hashes, method, and the
source episode and frames. Never swap a visual mesh for a collision mesh because the names look alike.
Point fusion fails on overflow instead of silently coarsening.

## Packed files

We run the kit's packer unchanged, then check what it wrote:

- Mesh Parquet: exact roster and commit, contiguous indices, finite vertices, in-range integer
  faces, at least one positive-area triangle per mesh.
- Pose rows: exact roster and columns, values matching the staged rollouts after six-decimal
  rounding, nonzero quaternions, and the selected checkpoint and source hashes.

The pack manifest (`submission/<stamp>/*.manifest.json`) records the packed file's hash and every
input hash. `scripts/reproduce/verify.sh <manifest>` re-hashes them. This proves the bytes, not the
pose accuracy or mesh quality.

The object-pose, hand, scene, training-checkpoint, scratch-depth and submission-plan contracts
belonged to slices deleted on 4 Nov 2026 (object_pose, object_mesh, hand_motion, scene_assembly,
retargeting, policy_rollout, scratch_depth, submission). They are in Git history before the cleanup
commit (parent `ef884fd`).

## Changing a contract

Change the producer, every consumer and the docs together, and say what breaks. Keep old records
unchanged. A new required field means old outputs must be rebuilt; never backfill a value that was
not measured. A new optional field must not change the meaning of validity, provenance, clocks or
object order. See the [feature template](feature-template.md).
