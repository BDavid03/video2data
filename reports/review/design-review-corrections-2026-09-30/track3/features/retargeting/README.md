# Retargeting

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

This slice turns the scene bundle into the training-ready sequence the policy
runtime reads: robot hand motion that follows the human hands, the object
trajectories, and the object and support assets. It writes the upstream input
format and then runs the upstream retargeter unchanged. `RG/` means
`upstream/video_to_data-flashchord-aee95f2/robotic_grounding/` and `FC/` means its
`flash_chord/` folder. The source reading is
[upstream-retargeting.md](../../../reports/contract-evidence/2026-09-30/upstream-retargeting.md).

## Role in scoring

This is one of the two boundaries where a mistake produces a valid file and a bad
score (see [artifact contracts](../../../docs/design/artifact-contracts.md)). Units,
quaternion order, object order and frame count all change hands here.

## Embodiment

**Design: floating Sharpa hands, robot name `sharpa_wave`.**

| Reason | Source |
|---|---|
| It is the only egocentric route that accepts two tracked objects. The arm-mounted route raises an error unless there is exactly one. | **Official**, `RG/scripts/retarget/ego_recon_to_dexmate_sharpa.py:1068-1069` |
| Sixteen of the 20 scored episodes have two objects. | **Official**, `KIT/data/track_3_evaluation_objects.json` |
| Two independent rigid objects already pass through this route upstream, for the TACO dataset. | **Official**, `RG/scripts/retarget/taco_to_sharpa.py:58-70` |
| Floating hands have no arm, so no demonstration is out of reach. | **Design** |

No published source names a required embodiment for Track 3 (**Open**).

## Operations

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `export` | The upstream "loaded" Parquet, with hand and object columns filled and robot columns empty | Policy environment, Linux | No |
| `assets` | One mesh and one rigid URDF per object | Policy environment, Linux | No |
| `retarget` | The "processed" Parquet, by running the upstream script unchanged | Policy environment, Linux | No, see Resources |
| `support` | A support-surface file, by running the upstream script unchanged | Policy environment, Linux | No |
| `inspect` | `inspect.json`: findings from opening the processed Parquet | Policy environment, Linux | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `scene.json`, `trajectories.npz`, `assets/` | [scene_assembly](../scene_assembly/README.md) | Everything |
| `RG/scripts/retarget/ego_recon_to_sharpa.py` and its package | Upstream | Run unchanged |
| Sharpa hand model files | Upstream `robotic_grounding/assets`, not yet fetched | Read by the upstream script |

## Method

1. **Export.** Write one Parquet row in the unversioned `ManoSharpaData` format
   (**Official**, `RG/source/robotic_grounding/robotic_grounding/retarget/data_logger.py:24-357`).

   | Column | Value |
   |---|---|
   | `fps` | 20.0 |
   | `mano_{side}_joints` | T×21×3, world, metres |
   | `mano_{side}_joints_wxyz` | T×21×4, scalar-first, from the rotation matrices |
   | `mano_{side}_trans` | T×3. The upstream script takes its frame count from the right hand's column, so it is always filled. |
   | `mano_{side}_global_orient`, `finger_pose`, `betas` | Carried from hand_motion |
   | `mano_{side}_object_contact_positions`, `normals`, `part_ids` | From the scene bundle's contacts |
   | `object_name` | The body names joined by an underscore |
   | `object_body_names` | The official object names in official slot order |
   | `object_body_position`, `object_body_wxyz` | T×B×3 and T×B×4, scalar-first, unit length |
   | `object_mesh_paths`, `object_urdf_paths` | One per body, as seen from the policy environment |
   | `object_mesh_radius` | Largest vertex distance from the body origin, per body |
   | `object_articulation` | Zeros |
   | `object_root_position`, `object_root_axis_angle` | Body 0's pose |

   The partition directory is `sequence_id=episode_XXXXXX/robot_name=sharpa_wave`.
   The upstream script copies `robot_name` from the loaded file and never sets it
   (`RG/scripts/retarget/ego_recon_to_sharpa.py:573-612`).
2. **Assets.** For each object, write the mesh as OBJ in metres in its body
   frame, and a rigid URDF with a single link. File names never end in `_cm`,
   because upstream scales such files by 0.01
   (`retarget/support_recon.py:120-121`).

   Upstream's generator uses 0.3 kg and a fixed inertia for every object and sets
   no friction (`RG/scripts/generate_rigid_urdfs.py:94-133`). **Design:** this
   slice writes the URDF itself, in the same form, with mass from the scene
   bundle's estimate and inertia computed from the mesh at that mass. The
   simulator decomposes the collision mesh into convex parts when
   `scene.decompose_objects` is true, which is the default
   (`FC/src/flash_chord/configs/scene/train.yaml`).
3. **Retarget.** Run the upstream script on the loaded file. It solves inverse
   kinematics for both hands on every frame and fills the robot columns. Hand
   smoothing and penetration handling are options of that script and are off by
   default; the request chooses them. The script's penetration handling treats
   each object as its bounding box, which is exact only for box-like objects
   (`retarget/object_collision.py:12-15`), so it stays off for pots, cups and the
   spoon unless a run shows it helps.
4. **Support.** Run the upstream support script. It drops supports at or below
   5 cm, and its phantom filter also removes the disk of a body resting on
   another tracked body (`RG/source/robotic_grounding/robotic_grounding/retarget/support_recon.py:538-541,758-766`).
   With the table at z = 0, many episodes will therefore have no support shapes.
   Upstream writes an empty stage in that case (`support_recon.py:787-799`).
   The runtime rejects such a stage: it requires at least one Cylinder or Cube
   (`FC/src/flash_chord/scene/support.py:47-48,90-91`). When there are no shapes,
   record `scene.support=false` as required training configuration. This disables
   only the extra support stage; the ground plane remains on
   (`FC/V2D_INTEGRATION.md:80-81`, `scene/builder.py:447-448,506-507`).
   When support is enabled, the file must exist at the path the runtime searches
   (`scene/setup.py:62-66`), with boxes or cylinders placed by translation and
   no rotation (`scene/support.py:67-104`).
5. **Inspect.**

   | Check | Expected |
   |---|---|
   | Exactly one Parquet file in the partition directory, one row | The consumer rejects anything else (`FC/src/flash_chord/data/parquet.py:24-65`) |
   | Frame count of every time series | The clip's frame count |
   | `fps` | 20.0 |
   | `object_body_names` | Official names, official order |
   | Object quaternions | Unit length within 1e-4, the consumer's tolerance |
   | `object_urdf_paths`, `object_mesh_paths` | Exist in the policy environment |
   | Support configuration | File exists at the searched path and contains at least one Cylinder or Cube, or `scene.support=false` is required and recorded in `inspect.json` |
   | Contact arrays | Present for both sides with 16 slots; the runtime treats an empty array as missing (`FC/src/flash_chord/data/mano_sharpa.py:154-158,279-295`) |
   | Each object URDF | One free body, mass and inertia present, at least one collision shape (`FC/src/flash_chord/scene/objects.py:263-293,411-413`) |
   | Robot columns | Filled and finite for both sides |
   | Inverse-kinematics position error per task | Reported; large values listed by frame |

## Outputs

`outputs/retargeting/<split>/episode_XXXXXX/<run-id>/assets/human_motion_data/ego_recon/`

| Path | Content |
|---|---|
| `loaded/sequence_id=episode_XXXXXX/robot_name=sharpa_wave/<file>.parquet` | Step 1 |
| `processed/sequence_id=episode_XXXXXX/robot_name=sharpa_wave/<file>.parquet` | Step 3, the file the policy runtime trains on |
| `processed/<object>.obj`, `<object>_rigid.urdf` | Step 2 |
| `reconstructed_stage/episode_XXXXXX_support.usda` | Step 4 |

Two properties of this path are deliberate. It contains a directory named
`assets`, because the policy runtime can re-root stored paths that contain
`/assets/` when the tree is mounted elsewhere (**Official**,
`FC/V2D_INTEGRATION.md`). It contains `episode_XXXXXX` and no other text of that
form, because the evaluator reads the episode index from the first such match in
the reference path (**Official**, `FC/src/flash_chord/evaluation/harness.py:154-156`).
The run id is a timestamp and cannot match.

## Configuration and authorization

`runtime.json` holds one flag per operation, all false. The request names the
scene run and the upstream script's options.

## Dependencies

All operations run in the Linux policy environment. `export`, `assets` and
`inspect` use its locked `pyarrow` (`FC/pyproject.toml:18`) and geometry packages;
no extra perception-environment installation is proposed. The policy environment
installation itself still needs approval.

`retarget` and `support` need the upstream retargeting stack: Python 3.11,
Pinocchio, Pink, a quadratic-programming solver, a USD library, and the Sharpa
hand model files (**Official**, `RG/workflow/Dockerfile:13-104`). The simulator is
not needed for retargeting (`RG/workflow/data_pipeline.md:173-184`). These are new
dependencies in the Linux policy environment and need the user's confirmation.

## Resources

Inverse kinematics runs on the CPU through Pinocchio. The upstream script's
device option defaults to the GPU but moves values to CPU arrays before solving
(`retarget/hand_kinematics.py:341-345`). Whether it runs with the device set to
CPU is unverified. Runtime unmeasured.

## Failure handling

`export` refuses a scene bundle with any NaN. `retarget` failure keeps the loaded
file and the upstream script's output. `inspect` findings are blocking or
advisory, and [policy_training](../policy_training/README.md) refuses a sequence
with a blocking finding or one whose recorded support requirement is not
applied to the resolved training configuration.

## Separation of public labels

Evaluation sequences inherit `video_only`. A sequence built from public poses, to
test control separately from perception, carries `public_label_diagnostic` and is
written under `outputs/retargeting/public/`.

## Acceptance evidence

Once runs are authorized: the policy runtime's scene viewer loads the processed
sequence, shows both hands and every object at frame 0, and its kinematic replay
follows the video.

## Open questions

| Question | Who can answer |
|---|---|
| Is an embodiment required for Track 3? | Organisers |
| Which licence covers the Sharpa hand model files? | Organisers |
| What mass and friction do the organisers' reference scenes use? | Organisers |
