# Retargeting

Status: implemented, never run

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

## Contact layout at the scene boundary

The scene's contact arrays use the 16 names and joint groups from the pinned
upstream `MANO_HAND_LINKS`: palm, then three links for each finger. Export checks
`scene.json.contact_layout` and `trajectories.npz.contact_link_names` against the
installed, source-pinned `retarget/params.py`, and writes those 16 names into
`mano_link_names`. Hand joints and orientations retain their independent MANO21
order. Older scene artifacts without an explicit contact layout must be rebuilt;
21 joint labels cannot describe 16 contact-link slots.

Inspection checks the processed motion's link order, body ID bounds, active unit
normals, and zero position/normal fields for absent contacts. Scene contacts are
sampled geometric proximity estimates, not measured physical contact. These
changes were implemented from source reading only; no export, inspection,
retargeting, validation command or test was run.

### Optional supported interval transfer

`RetargetRequest.contact_run` names a completed `hand_motion.contacts` run.
It requires the separate `contact_transfer_authorized=false` gate and public
training/tuning episodes 0/11/12/18/21/23/26/31/39/42. Defaults preserve the
existing export. The scene and new evidence must be `video_only`; the contacts'
hashed `hand_run`, `pose_run` and `mesh_run` must occur in the scene's inputs. Different fits cannot
silently supply one another's contacts.

The adapter requires complete matching frame indices, integer PTS and
`time_base="1/10240"`. Each interval uses exclusive `end_frame` and the inclusive
last sample's `end_pts`. It admits contact slots only when the interval explicitly
reports `accepted=true`, `physical_contact_confirmed=true`, phase contact/grasp/lift,
`support.contact_supported=true`, no ambiguity, and finite confidence at least
`minimum_contact_confidence` (default 0.75). That threshold is an admission rule,
not a calibration claim. Missing confidence stays null. Rejected overlapping
intervals suppress the corresponding object/side frames conservatively.

Current hand interval hypotheses are unconfirmed and unaccepted; they therefore
cannot create physical contacts. Selected transfer zeros unsupported body IDs,
positions and normals in the existing 16-slot upstream fields. It never inserts
contacts absent from scene geometry. `contact-admission.json` retains the original
interval evidence, confidence, ambiguity, timing, source identity, reasons and
before/after slot counts. It is copied through assets, retarget and support;
inspect exposes its hashed identity. This is conservative evidence admission,
not proof of contact feasibility, measured force, robot reachability or IK success.

### Measured-error randomization proposals

`uncertainty_run` optionally names a completed video-only run with a hashed
`scene-errors.json`. Its separate `uncertainty_proposal_authorized` flag stays
false. No current producer supplies this independent evidence; joint-refinement
residuals, Hessian conditioning and ensemble disagreement do not qualify.

The input schema has `schema_version=1`, `provenance="video_only"`, matching
`split`/`episode_index`, `evidence_kind="independent_measured_scene_errors"`,
`coordinate_frame="world"`, and
`error_definition="prediction_minus_independent_measurement"`. Each `components`
entry identifies `object_name`, `quantity` (`translation_error` or
`rotation_error`), `axis` (x/y/z), `units` (m/rad), signed finite `samples`, and a
parallel `support` list. Rotational errors must be world-frame rotation-vector
components. Each support identifies `frame_index`, `pts`, nonempty `method`, and
`source_artifact` with path/SHA256, which must match a recorded producer input.
At least two distinct support records are required per component.

The feature derives empirical extrema and rejects components exceeding the
predeclared absolute caps (default 0.03 m and 0.15 rad). It does not clip poor
geometry into acceptable evidence. `scene-randomization-proposal.json` records
accepted/rejected envelopes, physical units, support and source hashes, with
`distribution=null`, `sampling_enabled=false` and calibrated uncertainty false.
The proposal is copied with the bundle and exposed by inspect. It does not alter
training configuration, create a probability distribution or infer mass/friction.
Empirical extrema from sparse or correlated errors have no established coverage.

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

The implemented CLI is `python -m track3.features.retargeting <operation>
--request <file>`. Operations run in order: export, assets, retarget, support,
inspect. Export takes `scene_run`; each later request takes the previous
operation's `input_run`. Each operation writes a new timestamped run directory.
Asset-bearing stages copy the previous bundle and rebase its stored mesh/URDF
paths, leaving the earlier run unchanged. Inspect reports the final support
run's processed Parquet and its hash, including `scene.support` for training.
Paths contain only the selected episode index, repeated in the episode directory
and sequence partition as required by the upstream layout.

Resting scenes use the same path: both parked hands and all 16 contact slots
per side are exported, including zero contact arrays. The current scene contract
does not include MANO betas or finger parameters. Those optional, unused-by-IK
columns remain empty when absent; if supplied in the scene arrays with their
upstream names, they are carried through. Joint rotations are never guessed
from positions. The assets stage fills mesh radii and writes separate visual
and collision OBJ files plus a one-link rigid URDF with the scene mass and the
collision mesh's centre of mass and inertia. The export-only intermediate has
planned asset paths; it is not training-ready until assets, retarget, support
and inspect finish.

The retargeter runs on CPU, with no viewer. Smoothing, surface projection and
adaptive offsets are explicit request fields and default false. IK errors are
reported per task and frame as advisories above `ik_position_error_m`; malformed
or missing robot data block training. Support reconstruction uses the upstream
`--dataset ego_recon --ground_threshold 0.05` options. Empty stages are accepted
only with the recorded requirement `scene.support=false`.

All operations require the dedicated `video2data-policy` WSL distribution,
its separate retargeting Python 3.11 environment, and a completed pinned
installation record. See [provisioning](../../../scripts/wsl/README.md).

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
