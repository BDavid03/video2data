# FlashCHORD runtime: contract evidence for Track 3 (Egocentric)

Date: 2026-09-30. Source: `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/` at commit
`aee95f25ae0943f335d8bb42fa95b6a9acfa266f` ("Publish FlashCHORD trajectory export for V2D leaderboard (#181)",
2026-09-28, from a read-only `git log -1`).

## Scope and method

This is a read-only source reading. Nothing was executed: no Python, `uv`, `pip`, Docker, tests, smoke checks,
validation commands or dependency resolution. Nothing was installed or downloaded. The only commands were file
listings, `cat`/`grep`/`sed`/`wc`/`find`, `git log -1` and `git sparse-checkout list` (both read-only, with lazy
fetch disabled). Unless stated otherwise, every citation is `path:line` relative to the FlashCHORD root
(`robotic_grounding/flash_chord/`). Paths that start with `../` point to sibling files under `robotic_grounding/`.
`../../LICENSE` is the file at the root of the checkout. Instructions aimed at AI agents in the upstream tree, such
as `../CLAUDE.md`, were treated as data and not followed.

Two observations about the checkout:

1. **The sparse checkout grew while this reading was underway.** When the reading started, `robotic_grounding/`
   held only `flash_chord/` and top-level files. At about 07:28, `docs/`, `scripts/`, `workflow/` and
   `source/robotic_grounding/robotic_grounding/{motion_schema,retarget}` appeared, and `git sparse-checkout list`
   now shows them. This report did not make that change; it came from another process. Those upstream files were
   used only to cross-check producer dtypes for Question 1 (`motion_schema/README.md`, `motion_schema/schema.py`,
   `retarget/data_logger.py`), and they are labelled as upstream where cited.
2. **Every binary asset is an unresolved Git LFS pointer.** The pointers are:
   - `src/flash_chord/assets/meshes/sharpawave`: 346 of 346 files;
   - `meshes/vega_sharpa`: 148 of 148;
   - `meshes/g1`: 63 of 63;
   - `policies/sonic/*.onnx`: 2 files;
   - `docs/media/*.gif`: 8 files.

   The total of 559 under `assets/` matches `SOURCE.json:5` (`lfs_files: 559`). URDF XML files are real. No robot
   meshes or SONIC weights are present locally, and `README.md:20-21` requires `git lfs pull`.

---

## 1. Input data contract for one demonstration

### 1.1 Container, discovery and dispatch

- **Path.** `task.parquet` is mandatory and has no default (`src/flash_chord/configs/task/default.yaml:1-3`;
  `src/flash_chord/configs/README.md:31`). It may be a file, or a directory that holds exactly one `.parquet`
  anywhere beneath it. The search is recursive (`rglob`) (`src/flash_chord/data/parquet.py:24-39`).
- **One row.** The file must contain exactly **one row**; each cell holds a full per-frame nested array
  (`data/parquet.py:46-49`; `data/mano_sharpa.py:6-8`).
- **Hive partitions.** Path components such as `key=value` (`sequence_id=…`, `robot_name=…`) are parsed as
  partition values and must agree with same-named physical columns (`data/parquet.py:55-65,68-82`). Hydra values
  that contain `=` must be quoted as `"task.parquet='${PARQUET}'"` (`V2D_INTEGRATION.md:78-79`).
- **Dispatch.** The loader dispatches as follows (`data/loader.py:45-70`):
  - `schema_version == "motion_v1"` goes to the `motion_v1` loader (`loader.py:54-60`).
  - Any other non-empty `schema_version` is rejected (`loader.py:61-62`).
  - An unversioned file goes to `ManoSharpaData` only if **all** signature columns are present
    (`loader.py:14-42,63-66`). Otherwise it is rejected (`loader.py:67-70`).
- **Two supported contracts.** Only these two exist: unversioned `ManoSharpaData` (floating Sharpa retargeters)
  and `motion_v1/single_robot` (Dexmate/Vega and G1 writers) (`V2D_INTEGRATION.md:28-31`). `motion_v1/dual_hand`
  is intentionally unsupported (`V2D_INTEGRATION.md:44-46`). Objectless motions are unsupported
  (`V2D_INTEGRATION.md:48-50`).
- **Global conventions.**
  - Metres and **wxyz** quaternions in the Parquet/`Reference`; Newton/Warp use xyzw (`CONVENTIONS.md:28-29`;
    `data/reference.py:10-11`).
  - Arrays are time-major, `T` first (`reference.py:10-11`).
  - Contact part IDs are 1-indexed object-body indices, with 0 meaning inactive (`reference.py:182-184`).
  - At least two frames are required (`data/resampling.py:97-102`).
  - Frames are assumed uniformly spaced at `fps`. There is no per-frame timestamp column
    (`resampling.py:14-18,34-41`).

### 1.2 `ManoSharpaData` (unversioned; the only format accepted by the floating-Sharpa embodiment)

`schema_version` must be absent or empty. All signature columns must exist (`data/loader.py:14-42`). Validation is
in `data/mano_sharpa.py:161-352`.

| Column(s) | Shape / type as validated | Notes |
|---|---|---|
| `fps` | scalar > 0 (used as `float`) | `mano_sharpa.py:175`. Upstream writer type `float32` (upstream `retarget/data_logger.py`, BASE_FIELDS) |
| `object_name` | non-empty string | `mano_sharpa.py:193-201` |
| `object_body_names` | list[str], length `B`, unique | `:194-201`. Defines the object/body order used everywhere (see Question 3) |
| `object_body_position` | `(T,B,3)` finite, m | `:177-182` |
| `object_body_wxyz` | `(T,B,4)`, unit within 1e-4, **wxyz** | `:183-190` |
| `object_mesh_paths` | `B` non-empty strings | `:202-204`. Required, but used only by the articulated ARCTIC/Synthbox adapter (`assets/registry.py:77-105`) |
| `object_urdf_paths` | optional; if present, exactly `B` paths (one rigid URDF per body) | `:205`; `registry.py:179-194` |
| `object_mesh_radius` | `(B,)` positive finite | `:206-212`. Bounding-ball radius, used as the wrench-support torque scale (`reference.py:207-209`) |
| `object_articulation` | optional `(T,A)`. With per-body URDFs only a single all-zero placeholder column is allowed | `:213-223` |
| `robot_{left,right}_wrist_position` | `(T,3)` finite, m, **both sides required** | `:253-258` |
| `robot_{left,right}_wrist_wxyz` | `(T,4)` unit within 1e-4 | `:259-269` |
| `robot_{side}_finger_joints` + `{side}_robot_finger_joint_names` | `(T,N)` plus N unique names | `:270-274`. Names must include every simulated finger joint (`embodiments/base.py:586-598`). Sharpa has 22 revolute finger joints per hand (`assets/urdfs/sharpawave/right_sharpawave.urdf`; upstream writer fixes 22) |
| `mano_{side}_object_contact_positions` / `_normals` | `(T,K,3)` each, finite; active normals unit | `:279-295`. Cannot be zero-size: an empty array reads as "missing" (`:154-158`) and triggers the error "required together" (`:282-283`). Upstream writer uses K = 16 |
| `mano_{side}_object_contact_part_ids` | `(T,K)` integers in `[0,B]` | `:286-292` |
| `robot_{side}_frames` + `{side}_robot_frame_names` | `(T,F,7)` pos + wxyz, F unique names | `:301-310`. Shape-validated; only positions are kept, and this reading found no runtime consumer |
| optional metadata | `source_dataset`, `sequence_id`, `robot_name`, `raw_motion_file`, `coord_frame`, `motion_kind` | `:224,321-332`. `source_dataset` falls back to the path component after `human_motion_data` (`:359-365`). `coord_frame` is **not validated** for this format |

The loader reads the whole row but consumes nothing else. MANO fields (`mano_*_trans`, `_joints` and so on) are
not used by FlashCHORD.

### 1.3 `motion_v1/single_robot` (required by `dexmate_sharpa` and `g1_dex3`)

Validation is in `data/motion_v1.py:180-334`. Columns read: `motion_v1.py:17-47`.

| Column(s) | Rule |
|---|---|
| `schema_version` | `"motion_v1"` (`:196-198`) |
| `motion_kind` | `"single_robot"` (`:199-201`) |
| `coord_frame` | `"world"` or `"robot_base_z_up"`; anything else is rejected (`:202-208`) |
| `fps` | finite, > 0 (`:210`) |
| `robot_joint_names`, `robot_joint_positions` | unique non-empty names; `(T,J)` finite (`:211-217`) |
| `robot_root_position` / `robot_root_wxyz` | `(T,3)` / `(T,4)` unit (`:218-219`) |
| `ee_link_names` + `ee_pose_w` | `(T,E,7)` as `[xyz, wxyz]` (`:221-227`; `V2D_INTEGRATION.md:35`) |
| `hand_sides`, `hand_frame_names`, `hand_frames_w` | optional; per-side `(T,K_side,7)`, appended after the EE frames; names globally unique (`:229-241,337-387`) |
| object columns | same meaning as in 1.2 (`:242-252`); `object_mesh_paths` and `object_mesh_radius` required (`:405-416`) |
| contacts `hand_object_contact_{positions,normals,part_ids}` | optional as a group; per side `(T,K_side,3)` / `(T,K_side)`; `K_left != K_right` allowed (`:490-550`; `V2D_INTEGRATION.md:39`) |
| `hand_contact_active` | optional `(S,T)` binary; otherwise derived from part IDs (`:571-588`) |

Upstream producer dtypes come from the now-present upstream `motion_schema/schema.py:47-123`: strings, `fps`
`float32`, poses `list<float32>[7]`, `hand_object_contact_part_ids` `int32`. FlashCHORD itself checks only
"numeric and finite", not an exact dtype (`motion_v1.py:435-443`).

### 1.4 Timing and resampling

- The reference is resampled to the control rate at load. Positions and joints use `lerp`, quaternions use SLERP
  and contacts use nearest-neighbour (`mano_sharpa.py:166-170,235-300`; `motion_v1.py:271-296`;
  `resampling.py:44-74`).
- **Half-speed Sharpa and Dexmate.** With `motion_speed != 1`, the target samples are
  `linspace(0, duration, int(duration*control_fps/motion_speed))` in source seconds (`resampling.py:21-41`). The
  Sharpa and Dexmate recipes use `motion_speed: 0.5` at a control rate of `fps: 20` (`configs/experiment/sharpa_flash_sac.yaml:19-23`;
  `configs/sim/rl.yaml:2`; `configs/experiment/dexmate_sharpa_flash_sac.yaml:21-25,67-68`), so a clip of D seconds
  becomes about `40·D` control frames.
- **G1.** `source_frame_playback: true` means one source frame per 50 Hz control step, without resampling; the
  reference's fps is then reported as the control fps (`configs/experiment/g1_recon_body_flash_sac.yaml:15-19`;
  `motion_v1.py:271-276,312`). `ManoSharpaData` rejects source-frame playback (`loader.py:64-65`).
- **Frame window.** `task.motion_start_frame` and `task.motion_end_frame` select a half-open source-frame window
  `[start, end)` (`data/__init__.py:40-44`; `reference.py:220-233`).

### 1.5 Object assets (mesh, URDF, collision, mass/inertia, scale)

- **One URDF per rigid body.** Each rigid body needs its own URDF. Without per-body URDFs, only
  `source_dataset ∈ {arctic, synthbox}` articulated objects are accepted, and they need root body `bottom` and one
  `rotation` joint (`assets/registry.py:163-241`, especially `:179-200,214-237`). A custom Track 3 object must
  therefore supply `object_urdf_paths`.
- **Import.** Each URDF is imported with `floating=True`, `collapse_fixed_joints=True` and
  `ignore_inertial_definitions=False` (`scene/objects.py:386-393`). Mass, COM and inertia therefore come from the
  URDF `<inertial>`; the mesh must be referenced inside the URDF.
- **Structural rules.** The URDF must yield exactly one world-parented free joint on its root body
  (`objects.py:263-293`) and at least one active collision shape (`objects.py:411-413`).
- **Initial pose and body frames.** The object's free-joint pose is set to the reference frame-0 pose of its root
  body (`objects.py:400-404`). The URDF body frame must therefore coincide with the frame in which
  `object_body_position/wxyz` are expressed. Exported "achieved" poses are body-frame poses of that same body
  (`evaluation/rollout.py:207-227`).
- **Convex decomposition.** When `scene.decompose_objects: true` (the default in every scene config,
  `configs/scene/train.yaml:3`), collision meshes are split with CoACD (`objects.py:159-210`; threshold default
  0.05 in `assets/mesh.py:62-70`). Results are cached under `FLASH_CHORD_CACHE_DIR`, which defaults to
  `src/flash_chord/assets/.cache/convex` inside the checkout (`assets/mesh.py:32-34`).
- **Scale.** No scale column exists. The optional randomization `scene.object_scale_{min,max,seed}` scales geometry
  and inertia while keeping mass fixed; world 0 stays nominal (`docs/configuration.md:40-48`;
  `scene/builder.py:279-299,347-418`). Every recipe uses 1.0 (`configs/scene/train.yaml:4-5`).

### 1.6 Support surface (USDA) and ground

- **Default and failure.** `scene.support: true` is the default (`configs/scene/train.yaml:2`, `eval.yaml:2`). If
  no support file resolves, setup raises `FileNotFoundError` (`scene/setup.py:62-66`). Override with
  `scene.support=false`; the OSMO templates pass `scene.support={{ include_support }}`
  (`workflow/train_flash_sac.yaml:38`).
- **Lookup order** (`assets/registry.py:114-160`). For a parquet under
  `…/<dataset>/<stage>/sequence_id=<seq>/robot_name=<robot>/…`:
  1. `…/<dataset>/reconstructed_stage/<seq>_<robot>_support.usda`, then `<seq>_support.usda` (`registry.py:126-131`);
  2. FlashCHORD's vendored `assets/human_motion_data/<dataset>/reconstructed_stage/` (`registry.py:136-160`).

  The same order appears in `V2D_INTEGRATION.md:42`.
- **Content.** Only `Cylinder` prims (`radius`, `height`, Z axis) and `Cube` prims (`size` = full edge) are read,
  with `xformOp:translate`. Rotation and scale xformOps are ignored, because shapes are added with an identity
  quaternion on one static kinematic body (`scene/support.py:1-8,67-89,95-104`).
- **Ground plane.** A ground plane is added when `collision.ground` is true (`scene/builder.py:506-507`). A scene
  without ground requires a support file (`builder.py:447-448`).
- **Z-up.** The source assumes Z-up: VOC gravity compensation acts along +z (`runtime/object_control.py:36`) and
  support cylinders lie along Z. No explicit up-axis is set; Newton's default is used, and that default is not
  determined by this source.

### 1.7 Asset path resolution

- **Reference parquet.** The parquet path is resolved against the invocation directory
  (`configuration.py:36-41`; `data/parquet.py:26`).
- **Asset paths stored in the parquet** (`assets/registry.py:43-74`):
  1. An existing file is kept.
  2. Otherwise the suffix after the last `"/assets/"` is re-rooted onto the parquet's nearest `assets` ancestor
     directory, then onto FlashCHORD's vendored `src/flash_chord/assets/`.
  3. A path containing `"/data/object_assets/"` is re-rooted onto the vendored assets.
  4. Anything else is returned unchanged (`V2D_INTEGRATION.md:40`).

  The markers are forward-slash strings.
- **Container mount.** `docker/run.sh` mounts `V2D_ASSETS_DIR`, or the sibling
  `../source/robotic_grounding/robotic_grounding/assets`, read-only at `/v2d/assets`
  (`docker/run.sh:29-39`; `README.md:67-82`; `V2D_INTEGRATION.md:54-56`).
- **`source_root` (evaluation).** `evaluation.source_root` re-roots only a saved parquet path under `/workspace`
  or a relative path (`evaluation/harness.py:283-290`).

### 1.8 Which fields come from reconstruction and which from retargeting

- **ManoSharpaData.** Robot wrist and finger fields exist only in retargeted (`_processed`) parquets. `_loaded`
  parquets carry MANO, object and contact data but leave the robot fields empty (`data/mano_sharpa.py:6-8`;
  `V2D_INTEGRATION.md:76-77`).
  - **From human reconstruction and the loader:** `mano_*`, `object_body_position/wxyz`, object metadata and
    contacts (`mano_{side}_object_contact_*`).
  - **From retargeting:** `robot_{side}_wrist_position/wxyz`, `robot_{side}_finger_joints`, `robot_{side}_frames`
    and the name lists.

  `V2D_INTEGRATION.md:34` says V2D's wrist, finger, object, contact and named-frame fields "remain
  authoritative".
- **motion_v1.** `robot_joint_positions`, `robot_root_*`, `ee_pose_w` and `hand_frames_w` are retarget outputs.
  Object poses and contacts come from reconstruction. Robot joints are authoritative
  (`V2D_INTEGRATION.md:35-36`).
- **Is a retargeter included?** **No retargeter is in the FlashCHORD tree.** FlashCHORD "does not import
  `robotic_grounding`" (`V2D_INTEGRATION.md:3-5`). The expected upstream producers are named in the top-level
  files:
  - the ego path is `result.npz` from "the upstream ego reconstruction pipeline, which lives outside this repo",
    followed by the loader image and then `scripts/retarget/ego_recon_to_sharpa.py`,
    `scripts/generate_rigid_urdfs.py` and `scripts/reconstruct_support_surfaces.py` (`../README.md:214-250`);
  - the loader image is `v2d_task_library_loader` in the `reconstruction` repo (`../CLAUDE.md:53-62`);
  - the motion schema is `robotic_grounding.motion_schema`, and the dataset registry is
    `source/robotic_grounding/robotic_grounding/retarget/dataset_registry.py` (`../CLAUDE.md:147-152`).

  The `ManoSharpaData` writer class is `robotic_grounding.retarget.data_logger.ManoSharpaData` (upstream
  `retarget/data_logger.py`, `create_data_logger_class("ManoSharpaData", …)`).
- **Monocular properties that matter.** For monocular ego data the top-level README warns that fingers often start
  inside the object, that contacts are noisy (it recommends `force_closure` over `contact_wrench_support`), and
  that the world may carry a gravity tilt (`../README.md:259-275,486-493`).

---

## 2. Embodiments and recipes

| Config (`configs/embodiment/`) | Class / robot | Base | Actuated DOF (from URDF and layout code) | Required reference | Action group |
|---|---|---|---|---|---|
| `sharpa_hands` (plus variants `sharpa_hands_primitive`, `sharpa_hands_stiff`) | `SharpaHands`: two floating SharpaWave hands (`embodiments/sharpa_hands.py:78-85`) | floating: per hand, 3 prismatic + 1 ball wrist chain (`:89-146`) | per hand 3 + 3 wrist + 22 finger revolute joints (`urdfs/sharpawave/*_sharpawave.urdf`: 22 revolute) | `HandPoseReference`, i.e. **ManoSharpaData** only (`:292-293`) | `residual_hand_pose`: per hand 6 + fingers, so 56 dims (`runtime/actions/hand_pose.py:159-186`) |
| `dexmate_sharpa` | `VegaSharpa`: Dexmate Vega torso, two 7-DOF arms, Sharpa hands (`embodiments/vega_sharpa.py:1-8,156-163`) | **fixed** (`floating=False`, `:173-180`). The reference root must stay at identity within 1e-5 (`:419-441`) | 7 arm + 22 finger per side = 58 (`:286-287`; `urdfs/vega_sharpa/v2/SOURCE.json:2-61`) | **motion_v1** `NamedRobotReference`, with palm/fingertip or DP frames validated against FK to 5e-4 m (`:336-363,444-549`) | `residual_joint_position`: 58 dims (`runtime/actions/joint_position.py:322`), 2-physics-step delay (`configs/action/residual_joint_position.yaml:19-23`) |
| `g1_dex3` | `G1Dex3`: Unitree G1 plus two Dex3 hands (`embodiments/g1_dex3.py:3,183-190`) | floating pelvis (`:205`) | 29 body + 14 Dex3 = 43 scalar joints (`:48-85,293`) | **motion_v1** with `left/right_hand_palm_link` frames to 1e-4 (`:336-368,451-478`) | `sonic_joint_residual`: 29 residuals on the SONIC ONNX prior plus 14 direct finger joints = 43 dims (`runtime/actions/sonic.py:702`; `configs/action/sonic_joint_residual.yaml`) |

**Released recipes.** `configs/experiment/` holds five recipes (`docs/configuration.md:10-16`;
`configs/README.md:3-13`):

| Recipe | Algorithm | Embodiment | Worlds | Budget |
|---|---|---|---|---|
| `sharpa_ppo` | PPO | sharpa_hands | 4096 | 20,000 iterations × 24 steps |
| `sharpa_flash_sac` | FlashSAC | sharpa_hands | 4096 | 250,003,456 env steps |
| `dexmate_sharpa_flash_sac` | FlashSAC | dexmate_sharpa | 2048 | 250,009,600 env steps |
| `g1_recon_body_ppo` | PPO | g1_dex3 | 4096 | 20,000 iterations |
| `g1_recon_body_flash_sac` | FlashSAC | g1_dex3 | 2048 | 349,999,104 env steps |

Sources: `experiment/sharpa_flash_sac.yaml:25-33`, `experiment/dexmate_sharpa_flash_sac.yaml:72-81`,
`experiment/g1_recon_body_flash_sac.yaml:21-52`, `experiment/g1_recon_body_ppo.yaml:21-55`,
`experiment/sharpa_ppo.yaml:25-26`, `configs/training/reference_ppo.yaml:4-5`.

**Single-hand and bimanual manipulation with one or two objects.**

- **`sharpa_hands`.** The generic scene/objective code handles several independent rigid bodies. The upstream test
  `tests/scene/test_objects.py:45-72` spawns two HOT3D rigid components for Sharpa. There is **no single-hand
  mode**: both sides' wrist, finger, frame and contact data are mandatory (`data/mano_sharpa.py:253-310`), the
  action covers both hands, and the idle hand is still tracked and terminated at 0.15 m wrist error
  (`configs/termination/tracking.yaml:2-5`; `lifecycle/termination.py:181-196`).
- **`dexmate_sharpa`.** Bimanual, fixed-base. The generic code allows multiple object bodies; see
  `tests/scene/test_objects.py:163` ("two object imports preserve vega palm mounts").
- **`g1_dex3` (ReconBody).** Limited to **exactly one rigid object** (`lifecycle/termination.py:713-714`;
  `envs/recon_body.py:345-347`).

**No embodiment is mandated for Track 3 anywhere in this tree.** Searches for Track, Kaggle, challenge, leaderboard
and similar terms in `flash_chord/` found no such mandate. The only related trace is the commit subject
"…trajectory export for V2D leaderboard". The README's worked Sharpa example is an ego-reconstruction sequence,
`ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave` (`README.md:70-78`). That shows the Sharpa
recipe was validated on egocentric data, but it is not a mandate.

---

## 3. Multi-object support

- **Ordering and slots.** Reference body index = position in `object_body_names`
  (`scene/objects.py:473`; `runtime/command.py:106-111`).
  - With per-body URDFs, each body is its own "object" whose asset name equals the body name
    (`assets/registry.py:179-194`).
  - `ObjectLayout` maps body to owning object, and each object's root body drives VOC
    (`command.py:39-122`).
  - Every reference body must be bound exactly once (`objects.py:457-463`; `command.py:92-96`).
- **Commands.** Per-world, per-body targets are gathered at the current reference frame
  (`command.py:151-209`). With object scaling, a per-object root offset is added (`command.py:186-191`).
- **Observations.**
  - Per-body blocks: object position `3B`, orientation `4B`, optional velocity `6B`, command relative position
    `3B` and orientation `4B` (`envs/observation.py:323-349`).
  - Contact slots per side are `num_objects × contact_link_count`, where `num_objects` is the number of bodies
    (`observation.py:265-276,353-370`).

  Observation dimension therefore grows with the number of bodies.
- **Rewards.**
  - Object keypoints: six unit-axis keypoints per body, summed over bodies, then divided by `num_bodies*6`. The
    result is one scalar, with no per-object weights (`objectives/keypoints.py:24-48`;
    `objectives/tracking.py:184-200`).
  - Wrench support is computed per hand × body × basis (`objectives/tracking.py:299-351`).
- **Termination.** The per-world flag uses the **max** error over all bodies; any body over the threshold
  terminates the world (`lifecycle/termination.py:198-212`).
- **Metrics.** ADD-AUC is per body and then averaged; MPPE takes the max over bodies; ManipTrans requires every
  object to pass; RPE is pairwise between object roots (`evaluation/metrics.py:281-323,384-404`).
- **Exported order.** Rows are ordered `step, environment_id, object_body_id`, with `object_body_id` in reference
  (`object_body_names`) order, plus `object_id` and `object_body_name` columns (`evaluation/rollout.py:67,172-199`;
  `tests/evaluation/test_object_trajectory_export.py:43-49`).
- **Limits.** ReconBody/G1 allows only one rigid object (see Question 2). For articulated objects, only the
  ARCTIC/Synthbox adapter exists (`registry.py:196-241`).

---

## 4. Simulation stack

- **Physics.**
  - Newton `SolverMuJoCo`, which is MuJoCo-Warp; there is "no separate physics engine to select"
    (`runtime/sim.py:1-9,56-69`).
  - MuJoCo contacts are mandatory (`envs/base.py:123-124`).
  - Solver `newton`, integrator `implicitfast`, pyramidal cone (`configs/sim/rl.yaml:3-6`).
- **Rates.**
  - RL: 20 Hz control × 5 substeps = 100 Hz physics (`sim/rl.yaml:2-3`). Dexmate uses 10 substeps
    (`experiment/dexmate_sharpa_flash_sac.yaml:67-70`).
  - ReconBody: 50 Hz × 4 substeps (`sim/recon_body.yaml:2-3`).
  - Replay: 50 Hz × 4 substeps (`sim/replay.yaml:2-3`).
- **Pins.** `newton==1.2.0`, `warp-lang==1.14.0`, `mujoco==3.8.1`, `mujoco-warp==3.8.1`, `onnxruntime-gpu==1.28.0`,
  `usd-core==26.5`, `coacd==1.0.11` (`pyproject.toml:6-26`). The train extra adds `jax[cuda13]==0.10.2`,
  `flax==0.12.7`, `optax==0.2.8`, `distrax==0.1.9`, `safetensors==0.8.0` and `wandb==0.22.2` (`pyproject.toml:31-39`).
  The lock resolves CUDA-13 `nvidia-*` wheels (for example `uv.lock:773-925`).
- **Python.** `>=3.11,<3.13` (`pyproject.toml:5`). The image uses CPython 3.11.15 (`docker/Dockerfile:49-52`).
- **Docker.**
  - Base `nvidia/cuda:13.0.2-devel-ubuntu24.04` (`Dockerfile:11,22`), uv 0.11.21 (`Dockerfile:12,17`), install with
    `uv sync --frozen --extra train --extra dev` (`Dockerfile:60-62,72-74`).
  - The Dockerfile comment says runtime CUDA comes from the PyPI wheels ("warp-lang/newton/mujoco bundle CUDA 12")
    plus the host driver, and that the CUDA 13 base only supplies nvcc (`Dockerfile:8-10`).
  - `docker/run.sh` requires `--runtime=nvidia` (`docker/run.sh:86-92`) and uses `--network host`, X11 and EGL
    (`run.sh:144-149,167-185`).
- **Driver and OS.** "Linux and NVIDIA driver **580 or newer**", Docker with NVIDIA Container Toolkit, and Git LFS
  (`README.md:11-13`; `Dockerfile:10`).
- **Platform facts in the lock.**
  - The `jax-cuda13-plugin` and `jax-cuda13-pjrt` wheels are **manylinux only** (`uv.lock:518-535`), and the CUDA
    `nvidia-*` extras are marked `sys_platform == 'linux'` (`uv.lock` jax-cuda13-plugin `with-cuda` block).
  - `jaxlib`, `warp-lang`, `mujoco` and `onnxruntime-gpu` do have `win_amd64` wheels (`uv.lock:571,691,958,1664`).

  The locked environment therefore gives GPU JAX only on Linux, which includes WSL2.
- **Windows- or Unix-specific code.** Both training scripts call `time.tzset()` after forcing
  `TZ=PST8PDT` (`scripts/train_flash_sac.py:35-36`; `scripts/train_rl.py:32-33`). Python documents `tzset` as
  Unix-only; that is outside this source.
- **Windows, WSL, Blackwell, RTX 50-series and sm_120.** **None of these appears anywhere in the FlashCHORD tree.**
  The only architecture statement is the SONIC model card: "Supported Hardware Microarchitecture Compatibility:
  Ampere, Hopper, Lovelace, Jetson" and "Supported Operating System(s): Linux" (`src/flash_chord/assets/policies/sonic/MODEL_CARD.md:77-84`).
  Whether the pinned Warp, JAX, mujoco-warp and ONNX Runtime builds support compute capability 12.0 is not
  determined by source.

---

## 5. Memory and scale knobs

| Knob | Defaults / recipe values | Where |
|---|---|---|
| `scene.world_count` (training) | Base 4096 (`configs/scene/train.yaml:1`). `train_flash_sac.yaml:19-20` sets 1024, but the recipes override it: Sharpa 4096, Dexmate 2048, G1 FlashSAC 2048, G1 PPO 4096, Sharpa PPO 4096 | experiment files (Question 2) |
| `evaluation.world_count` | 4096 (`configs/evaluation/flash_sac.yaml:6`; README example `README.md:170-171`; OSMO `workflow/*:evaluation_worlds 4096`). Any positive value is accepted (`training/flash_sac/evaluation.py:45-46`). The actor restore is world-count independent (`flash_sac/evaluation.py:137`; PPO `training/checkpoint.py:127-134`) | |
| Constraint/contact capacity per world (`njmax`/`nconmax`) | RL 512/256 (`sim/rl.yaml:7-8`); Dexmate njmax 1024 (`experiment/dexmate_sharpa_flash_sac.yaml:70`); ReconBody 1024/512 with the note "32-world probe peaked at 114 constraints and 28 contacts … at least 8x headroom" (`sim/recon_body.yaml:7-10`); replay 16384/4096 (`sim/replay.yaml:7-8`; `SimConfig` defaults `runtime/sim.py:32-33`). Contact buffer = `solver.get_max_contact_count()` (`sim.py:72-80`) | |
| Replay buffer (FlashSAC) | **Device-resident** (`training/flash_sac/replay.py:3,209-230`). Base capacity 1,000,000, minimum 100,000, batch 2048, n-step 3, obs storage float32 (`configs/training/reference_flash_sac.yaml:59-67`). Sharpa: 4,194,304, float32 by default (`experiment/sharpa_flash_sac.yaml:45-47`). Dexmate and G1: 10,000,000, **float16** obs (`experiment/dexmate_sharpa_flash_sac.yaml:93-96`; `experiment/g1_recon_body_flash_sac.yaml:71-76`) | |
| Replay bytes formula | `bytes_per_transition = 2*obs_dim*obs_bytes + 4*(action_dim + objective_terms + 2*critic_ctx) + 2`, ring = `capacity × that` (`replay.py:184-200`). `obs_dim` is known only at runtime and grows with the number of object bodies and contact slots (Question 3), so no number is stated here | |
| Mixed precision | `mixed_precision: true`, `compute_dtype: float16` (`reference_flash_sac.yaml:8-9`) | |
| Networks (FlashSAC) | Base actor 2 blocks × 128, critic 2 blocks × 256, expansion 4, two critics, 101 atoms (`reference_flash_sac.yaml:19-36`). Recipes use actor 256 / critic 512 (for example `experiment/sharpa_flash_sac.yaml:37-39`) | |
| Networks and batch (PPO) | actor/critic `[1024,512,256,128]`, 24 rollout steps, 4 mini-batches, 5 epochs (`configs/training/reference_ppo.yaml:4,18-34`). Rollout batch = `world_count × rollout_steps` (`training/ppo/config.py:131-134`) | |
| `updates_per_collection` | 8 (Sharpa), 2 (Dexmate), 4 (G1) (experiment files) | |
| JAX preallocation | `XLA_PYTHON_CLIENT_PREALLOCATE=false` via `os.environ.setdefault`. In `scripts/evaluate_policy.py:12` and `scripts/view_policy.py:11` this happens before `import jax`. In `scripts/train_flash_sac.py:34` it happens after `import jax` (`:13`) but before any computation. **`scripts/train_rl.py` does not set it**; only the OSMO templates export it (`workflow/train_rl.yaml:26`) | |
| Mesh/collision resolution | CoACD on by default (`scene/*.yaml:3`); concavity threshold 0.05, unbounded hull count (`assets/mesh.py:62-70`). Hull count adds collision shapes and contacts; the magnitude is not determined by source | |
| Evaluation buffers | Device buffers `step_count × world_count × bodies × 7` float32 poses plus errors (`evaluation/rollout.py:385-390`). The host copy is float64 (`rollout.py:427`) | |

**The 48 GiB / 46,000 MiB requirement is documentation only.** It appears solely at `README.md:14` ("Full
training: 48 GiB GPU with ≥46,000 MiB free, as validated in EVT-07"), next to a manual `nvidia-smi` query
(`README.md:24`). The tree has **no admission check in code**: no code queries free memory (searches for
`mem_get_info`, `memory.free`, `MiB` and similar found only the README). "EVT-07" is not defined anywhere else in the
tree beyond `docs/media/README.md:8`.

**What a participant could change for 16 GB.** The source offers only these override hooks, not guidance:

- `scene.world_count=…` (README example `README.md:215`), subject to the constraints below;
- `training.replay.capacity`, `training.replay.observation_storage_dtype=float16`, `training.replay.batch_size`;
- network sizes;
- `sim.njmax`/`sim.nconmax`;
- `scene.decompose_objects`;
- `evaluation.world_count`;
- `training.checkpoint.save_replay=false` (the default).

Constraints that apply when changing world count:

- `total_environment_steps` must be divisible by `world_count` (`training/flash_sac/config.py:326-330`). The
  recipe budgets are multiples of 2048 or 4096.
- Replay capacity must be at least `world_count` (`config.py:339-342`).
- The FlashSAC curriculum is keyed to **environment steps** (`training/flash_sac/runner.py:437-446`). Fewer worlds
  means more collections and optimizer updates for the same budget
  (`total_optimizer_updates = collections × updates_per_collection`, `config.py:358-366`).
- The PPO curriculum is keyed to **iterations** (`training/ppo/runner.py:291`), so fewer worlds means fewer
  environment steps per stage.
- The Dexmate curriculum comment ties its thresholds to its "2,048-world collection cadence"
  (`configs/curriculum/dexmate_tracking_250m_mixed_reset.yaml:5-7`).

**The source says nothing about training quality with fewer worlds.** Its only related statements are that
experiment files own the "validated … world-count, learner, replay" selections and that individual values should
be overridden "only for an intentional ablation" (`configs/README.md:15-17`). The effect of a lower world count on
quality is not determined by source.

---

## 6. Training

- **Algorithms.** PPO and FlashSAC are peer learners on a shared Warp/JAX boundary (`training/README.md:1-4`).
  FlashSAC is a JAX port of Holiday Robotics' FlashSAC (`training/flash_sac/NOTICE:1-5`). Warp arrays are shared
  with JAX as zero-copy views (`training/environment.py:55-56,129-177`; `runtime/jax.py`).
- **FlashSAC defaults** (`configs/training/reference_flash_sac.yaml:1-90`):
  - discount 0.99;
  - learning rate 3e-4 falling to 1.5e-4;
  - Adam with loss scaling;
  - actor updated every 2 steps;
  - target τ 0.01;
  - distributional critic, 101 atoms, return bound ±5;
  - target σ 0.07;
  - reward normalization on.

  The recipes override σ, network size, replay and `updates_per_collection`, as in Question 5.
- **PPO defaults** (`configs/training/reference_ppo.yaml`): clip 0.1, entropy 0.001, learning rate 1e-3 adaptive
  (desired KL 0.005), GAE λ 0.95, gradient norm 1.0, 20,000 iterations. The G1 recipe overrides several of these
  (`experiment/g1_recon_body_ppo.yaml:53-86`).
- **Step budgets.** Question 2 lists them. `README.md:153-156` gives the terminal checkpoints
  `policy_250003456` (Sharpa) and `policy_349999104` (G1).
- **Curriculum.** A fixed stage schedule that reduces Virtual Object Control (VOC) assistance.
  - `tracking_250m_mixed_reset`: VOC 1.0 falling to 0 over 11 stages; object-keypoint weight 20; final stage
    starting at 200,007,680 steps uses first-frame reset probability 0.5 (`configs/curriculum/tracking_250m_mixed_reset.yaml:1-53`;
    `docs/configuration.md:29-38`).
  - ReconBody variants mix in "immediate first-frame" resets (`configs/curriculum/recon_body_flash_sac.yaml:1-50`).
- **Reset modes during training.**
  - `ReferenceResetConfig`: a uniform random start frame in `[0, T-2]`; frame 0 with probability 0.1 once VOC is
    below 0.1; random finger opening up to 0.7; a 20-step VOC=1 settling hold (`configs/reset/reference.yaml`;
    `lifecycle/reset.py:228-256,344-382,404-433`).
  - ReconBody: a 50-step freeze and 10-step VOC decay (`configs/reset/recon_body.yaml`; `reset.py:609-621`).
- **Termination.**
  - Tracking: wrist position 0.15 m, object position 0.10 m, object orientation 0.50 rad, plus reference-end
    truncation (`configs/termination/tracking.yaml`).
  - ReconBody: pelvis 0.70 m / 1.5 rad, palm 0.15 m / 1.5 rad, object 0.10 m / 1.5 rad, masked for 50 steps after
    reset (`configs/termination/recon_body.yaml`; `lifecycle/termination.py:636-646`).
- **Observation for Sharpa and Dexmate** (`configs/observation/policy.yaml`; `envs/observation.py:23-43,290-351`).
  - Proprioception: wrist position and wxyz orientation in the environment frame; wrist velocity in the body frame;
    finger joint position (normalized) and velocity.
  - Object: position and orientation per body; optional object velocity.
  - Targets: reference wrist pose relative to the current wrist; finger deltas; object position and orientation
    relative to the current object.
  - Actions: raw and processed.
  - Contacts: contact positions and force directions in the wrist frame.
  - Dexmate adds arm joint position, velocity and reference delta (`configs/observation/articulated_arm.yaml`).
  - The reference is read at the current `timestep` only (`observation.py:496-500`). **There is no multi-frame
    future window** for Sharpa and Dexmate.
- **Observation for G1.** A fixed 486-dimensional vector with future reference offsets of 0, 5 and 10 frames
  (`envs/recon_body.py:22-43`).
- **Critic-only context.** VOC state, reference phase and object-body velocity (`envs/rl.py:36-74`;
  `configs/train_flash_sac.yaml:26-31`).
- **Action spaces.** Question 2 lists them. For Sharpa, the residuals are added to the reference wrist pose and
  finger joints, scaled by 0.05 m, 0.15 rad and 0.15 rad, with EMA 0.3 (`configs/action/residual_hand_pose.yaml`;
  `runtime/actions/hand_pose.py:209-313`).
- **Checkpoints.**
  - FlashSAC writes three files: `policy_<steps>.safetensors` (actor params and batch statistics, used for
    evaluation), `state_<steps>.safetensors` (resume) and optionally `replay_<steps>.safetensors`
    (`training/flash_sac/checkpoint.py:114-146`; `training/README.md:45-54`). PPO writes
    `model_<iteration>.safetensors` (`training/ppo/runner.py:329-330`).
  - The format is safetensors plus a JSON manifest, without pickle (`training/checkpoint.py:23-57`).
  - Metadata includes `resolved_config_json` (the full resolved Hydra config), `policy_schema_json` and
    `critic_schema_json` (`training/checkpoint_metadata.py:19-21,203-215`). It also includes environment steps,
    global update step, curriculum stage and VOC scale, and algorithm/kind/set ID (`flash_sac/runner.py:617-641`;
    `flash_sac/checkpoint.py:32-36`).
  - **Yes, the checkpoint embeds the resolved config and the task parquet path.** The training scripts overwrite
    `task.parquet` with the resolved absolute path and add
    `reference = {parquet, num_frames, playback_fps, object_articulations}`
    (`scripts/train_flash_sac.py:80-87`; `checkpoint_metadata.py:97-104`).
  - No hash of the parquet's contents is stored.
- **Logging.**
  - **W&B defaults to `mode: online`, entity `nvidia-isaac`, project `v2d-flash-chord`, with
    `upload_checkpoints: true`** (`configs/logging/wandb.yaml:1-8`).
  - The training scripts create a W&B run unless `logging.mode=disabled` (`scripts/train_flash_sac.py:89-98`;
    `scripts/train_rl.py:81-90`). With it enabled, policy files are uploaded as artifacts
    (`flash_sac/runner.py:202-224`).
  - The README commands pass `logging.mode=disabled` (`README.md:144-150`). The evaluation root defaults to
    disabled (`configs/evaluate_policy.yaml:8-9`).
  - `docker/run.sh` injects `WANDB_API_KEY` from the environment or from `~/.wandb_api_key` (`run.sh:157-165`).
- **Training duration and hardware claims.** Beyond the 48 GiB statement, none are stated. The GIF actors are "EVT-07"
  seed-42 terminal actors (`docs/media/README.md:8-14`).

---

## 7. Evaluation and trajectory export

**Call trace.**

- `scripts/evaluate_policy.py:47-80` reads the `EvaluationConfig` from `configs/evaluation/flash_sac.yaml` and calls
  `evaluate_checkpoint(…, eval_script=Path(__file__))` (`:54-64`).
- `harness.evaluate_checkpoint` (`evaluation/harness.py:360-519`):
  1. reads the checkpoint config (`:377-378`);
  2. identifies the algorithm as FlashSAC or PPO (`:293-303`);
  3. forces the evaluation composition (`:216-251`);
  4. rebuilds the scene from the saved `task.parquet` (`:391-420`);
  5. sets the curriculum stage saved in the checkpoint, with VOC 0 (`:429-438`, `:265-272`);
  6. restores the policy (PPO `:442-447`; FlashSAC through `training/flash_sac/evaluation.py:128-157` at `:457-465`);
  7. records the rollout (`evaluation/rollout.py:346-451`);
  8. scores it (`:488-492`) and optionally exports Parquet (`:493-499`).

**`reset_mode=explicit`, the default** (`configs/evaluation/flash_sac.yaml:8`; `docs/configuration.md:99-101`):

- `auto_reset=False`, `voc_scale=0`, VOC `max_force` and `max_torque` = 0, `always_reset_to_first_frame=True`,
  `voc_decay_steps=0`, reset seed 42 (`harness.py:228-237`).
- Failure termination terms are disabled, or their thresholds set to 1e6 for ReconBody; reference-end truncation is
  required (`:238-250`).
- All worlds are reset to `start_frame` (default 0) (`:449-450`), at the exact reference pose, with no finger
  opening and zero velocity (`lifecycle/reset.py:79-122,386-400`).
- `sampled_settled` instead uses the checkpoint's sampled resets plus the ReconBody settling preamble
  (`:321-357,451-476`).

**Determinism.**

- Actions are the deterministic policy mean regardless of the `evaluation.deterministic` flag, which is hardcoded
  `True` (`harness.py:313,463`).
- The config flags `deterministic`, `use_checkpoint_config` and `validate_policy_schema` are parsed but not
  consulted by the harness. The harness always rebuilds from the checkpoint config and performs no explicit policy-schema
  check; checkpoint leaves must still match shape and dtype (`training/checkpoint.py:112-116`).
- Per-world randomness remains only where the saved config contains it:
  - object-scale ranges (`scene/builder.py:279-299`);
  - the G1 SONIC body-actuator delay, sampled uniformly from 0–2 physics steps per world with a seed
    (`configs/action/sonic_joint_residual.yaml:35-39`; `runtime/delay.py`).
- Bitwise GPU reproducibility is not determined by source.

**Horizon and alignment.**

- `step_count = reference.num_frames - start_frame`, counted on the **control-rate** reference after resampling and
  frame windowing (`harness.py:421-423`).
- Each step gathers targets at `timestep = t`, runs the physics substeps, evaluates errors against frame `t`, then
  increments `t` (`envs/base.py:289-335`).
- The pose recorded at step `s` is the post-step state compared with reference frame `start_frame + s`
  (`rollout.py:392-402,283-290`).
- The last step truncates at `timestep >= num_frames - 1` (`lifecycle/termination.py:328-343`).

**Control rate against reference rate.**

- Sharpa: 20 Hz control. With `motion_speed` 0.5, the trajectory has about 40 control steps per source second and
  plays at half speed.
- G1: 50 Hz with one source frame per step (Question 1.4).
- `time_s = step / control_fps` is simulation time, not source time (`rollout.py:190`).
- `reference_frame = step + start_frame` indexes the resampled reference, not the source frame (`rollout.py:189`).
- The export metadata omits `motion_speed` and source fps. Those live only in the checkpoint config.

**Early termination does not truncate the export.** Failure termination is disabled, so every world runs every step.
The first completion is latched without stopping the rollout (`rollout.py:248-267`). Any termination makes scoring
raise an error (`harness.py:580-581`; `metrics.py:213-217`). Worlds with non-finite poses are zero-filled and flagged
`environment_valid=false` (`rollout.py:427-432,158-160`).

**Exported Parquet** (`evaluation.object_trajectories_output`, `rollout.py:59-204`).

- Written atomically with zstd compression and dictionary encoding; one row per `(step, environment_id,
  object_body_id)`, in that order (`:67,171-199`).
- **Columns**, all non-nullable:
  - `environment_id int32`, `step int32`, `reference_frame int32`, `time_s float64`;
  - `object_id int32`, `object_body_id int32`, `object_body_name dictionary<int32,string>`,
    `environment_valid bool`;
  - `achieved_position_{x,y,z}`, `achieved_quaternion_{x,y,z,w}`, `reference_position_{x,y,z}` and
    `reference_quaternion_{x,y,z,w}`, all `float32` (`:80-103`).
- **Schema metadata** (`:63-74`): `flash_chord.schema = flash_chord_object_trajectories_v1`,
  `position_unit = meter`, `quaternion_order = xyzw`, `row_order`, `control_fps`, `start_frame`, `step_count`,
  `world_count`, `object_body_count` and `pose_width`. Non-empty provenance values are added as `flash_chord.<key>`
  (`:75-78`; `tests/evaluation/test_object_trajectory_export.py:80-117`).
- **Provenance** (`harness.py:115-157`):
  - `checkpoint_sha256`: SHA-256 of the evaluated checkpoint file.
  - `eval_code_sha256`: SHA-256 over sorted `name + \0 + bytes` for `evaluation/__init__.py`, `harness.py`,
    `metrics.py` and `rollout.py` **only** (`:124-135`). It does not cover `envs/`, `runtime/`, configs or
    dependencies.
  - `eval_script_sha256`: SHA-256 of the entry script, which is `scripts/evaluate_policy.py` through `__file__`
    (`:150-153`; `scripts/evaluate_policy.py:63`).
  - `reference_parquet`: the resolved path string.
  - `episode_index`: taken from the regex `episode_(\d+)` on that path (`:154-156`).
- **World frame.** Poses are Newton `body_q` in the simulation world. That is the same frame as the input parquet's
  `object_body_position/wxyz`, because objects spawn and are commanded at the reference coordinates
  (`objects.py:400-404`; `command.py:186-192`). Under object scaling, roots receive a bottom-height z offset
  (`builder.py:364-373`).
- **Quaternion order.** xyzw in both columns; the reference is converted from wxyz (`rollout.py:283-290`).

**PPO support.** The harness evaluates both PPO and FlashSAC checkpoints (`harness.py:293-303,306-318,442-447`), and
the export is algorithm-independent. The template for evaluation JSON reports is `write_evaluation_report`
(`harness.py:619-636`).

**Metrics** (`evaluation/metrics.py`; poses are xyzw, metres and radians).

| Metric | Implementation |
|---|---|
| ADD-AUC (`add_auc`) | ADD = mean distance over 500 body-frame vertices sampled with seed 0 from the simulator's visible object meshes (`metrics.py:18-19,222-258`; `rollout.py:293-343`). Thresholds 0.01 to 0.09 m in 0.01 steps, i.e. `np.arange(0.01, 0.10, 0.01)` (`metrics.py:20`). AUC is the trapezoid of the fraction of (frame, world) samples below each threshold on x = linspace(0,1,9), computed per body and then averaged (`:281-295`) |
| SP-SR (`spider_sr_uncentered`) | Per world, the mean over frames and bodies of position error ≤ 0.1 m **and** orientation error ≤ 0.5 rad, uncentered (`:21-22,360-373`) |
| MP-SR (`maniptrans_sr`) | Per world, per object (bodies averaged within the object): position < 0.03 m and orientation < 30°. Every object must pass (`:23-24,384-404`) |
| RPE (`relative_position_error_cm`) | Pairwise object-root position of the second object expressed in the first object's frame, compared with the reference, in cm, averaged over pairs, worlds and frames. Returns 0.0 for fewer than two objects (`:298-323`) |
| MPPE (`mppe_cm`) | Six keypoints at ±5 cm on X, Y and Z. Mean distance per body, **maximum over bodies**, then mean over frames and valid worlds, in cm (`:25,261-278`; `README.md:176`; invalid worlds excluded at `harness.py:545-555`). A separate `mean_per_point_error_cm` averages over bodies instead (`:336-350`) |
| CHORD-SR (`chord_sr`) | Reached the reference end, finite, and never crossed any task threshold (wrist or palm, object, and pelvis for G1) (`harness.py:488-492,559-606`) |

---

## 8. OSMO, cluster and cloud assumptions

- **OSMO templates.** `workflow/train_flash_sac.yaml`, `train_rl.yaml` and `evaluate_policy.yaml` request 1 GPU, 15
  CPUs, 120 Gi memory and 100 Gi storage (`workflow/train_flash_sac.yaml:7-12`).
  - They default to `logging_mode: online` and `upload_checkpoints: true` (`train_flash_sac.yaml:78-92`).
  - They use a W&B credential `wandb-cred`.
  - Evaluation downloads the checkpoint from a W&B artifact under entity `nvidia-isaac` (`evaluate_policy.yaml:39-48,69-81`).
  - `image` and `output_url` have no defaults (`README.md:226-247`).
- **Local operation.** "The local Docker workflow above needs no OSMO service or remote storage" (`README.md:246-247`).
- **Local-only overrides.**
  - `logging.mode=disabled` for training; the default is online (`configs/logging/wandb.yaml:1`).
  - Avoid `~/.wandb_api_key` injection (`docker/run.sh:157-165`).
  - Set `V2D_ASSETS_DIR` for the `/v2d/assets` mount (`run.sh:29-39`; `README.md:81-82`).
  - Pass explicit `task.parquet`, `output_dir` and `evaluation.metrics_output`/`object_trajectories_output`.
  - Set `FLASH_CHORD_CACHE_DIR` to control the CoACD cache location (`assets/mesh.py:32-34`).
  - Hydra run directories default to `logs/rl/...` and `outputs/...` relative to the current directory
    (`configs/train.yaml:22-26`; `configs/evaluate_policy.yaml:11-15`).
  - `docker/run.sh` writes passwd and group files to `${HOME}/.cache/flash_chord/<container>` (`run.sh:131`) and
    calls `xhost` (`run.sh:140`); both act outside the project directory.
- **Container paths.** `/workspace` is the container repo mount (`run.sh:180`). Checkpoints trained in the container
  record `/workspace/...` or `/v2d/assets/...` parquet paths. `evaluation.source_root` re-roots only `/workspace`
  paths (`harness.py:283-290`).

---

## 9. Licences

- **FlashCHORD source: Apache-2.0.** "Licensed under the Apache License, Version 2.0" (`README.md:258-260`). Every
  `src/**/*.py` file carries `SPDX-License-Identifier: Apache-2.0`; for example `src/flash_chord/__init__.py:2`, and
  no file lacks the header. The checkout-root `../../LICENSE:1-2` is dual: CC-BY-4.0 for documentation and skills,
  Apache-2.0 for source code.
- **FlashSAC port.** Derived from Holiday-Robot/FlashSAC under the MIT licence, and the notice must be retained
  (`src/flash_chord/training/flash_sac/NOTICE:1-29`).
- **G1 and Dex3 URDF.** BSD 3-Clause, © Unitree Robotics, "from or derived from" Unitree assets
  (`assets/urdfs/g1/NOTICE:1-8`; `assets/urdfs/g1/LICENSE`). `meshes/g1/LICENSE` and `NOTICE` are LFS pointers
  locally, and their hashes are recorded in `urdfs/g1/SOURCE.json:16-21`.
- **SONIC weights** (`assets/policies/sonic/*.onnx`). **NVIDIA Open Model License**, not Apache
  (`policies/sonic/NOTICE:1-11`; `SOURCE.json:25`). The LICENSE file is dual (Apache for code, Open Model License
  for weights). Redistribution requires including the agreement and the attribution "Licensed by NVIDIA Corporation
  under the NVIDIA Open Model License" (`policies/sonic/LICENSE` Section 3). This matters only if the G1 path is
  used.
- **Sharpa (`urdfs/sharpawave`, `meshes/sharpawave`) and Vega (`urdfs/vega_sharpa`, `meshes/vega_sharpa`).** These
  directories contain **no licence or notice file**; Vega has only provenance in `urdfs/vega_sharpa/v2/SOURCE.json:183-194`.
  Their licence beyond the repo-level LICENSE is not determined by source.
- **Upstream retargeting package.** Outside FlashCHORD, `../pyproject.toml:1-7` carries a proprietary "all rights
  reserved … strictly prohibited" header, which conflicts with the checkout-root LICENSE. MANO is licence-gated,
  and the loader image is described as "the MANO/GPL stage" (`../CLAUDE.md:60,143`; `../README.md:45-50`).

---

## 10. Risks and gaps for local 16 GB Blackwell under WSL2 and a custom reconstruction pipeline

1. **Memory.** The only documented operating point is 48 GiB with at least 46,000 MiB free (`README.md:14`). No
   tuned reduced-memory recipe or quality guidance exists (Question 5). World count, budget divisibility, replay
   size and curriculum cadence must be re-derived by hand, and the result is unvalidated.
2. **Platform.**
   - The docs say Linux only (`README.md:11`).
   - GPU JAX is locked to manylinux wheels (`uv.lock:518-535`).
   - The training scripts call `time.tzset()` (`scripts/train_flash_sac.py:36`).
   - `run.sh` assumes `--runtime=nvidia`, `--network host` and X11 (`run.sh:86-92,167-185`).

   Native Windows is excluded, so WSL2 is the path. Whether Docker Desktop or the WSL2 NVIDIA Container Toolkit
   honours `--runtime=nvidia` is not determined by source.
3. **Blackwell (sm_120).** Never mentioned. SONIC lists only Ampere, Hopper, Lovelace and Jetson
   (`MODEL_CARD.md:77-81`). The Dockerfile says Warp, Newton and MuJoCo bundle CUDA 12 while JAX uses CUDA 13
   (`Dockerfile:8-10`; `pyproject.toml:35`). Compatibility with CUDA graphs (`envs/rl.py:487-493`) and ONNX CUDA
   graphs (`runtime/actions/sonic.py:524-533`) on sm_120 is not determined by source. The driver minimum is 580
   (`README.md:11`); the machine's Windows driver is 596.21.
4. **Assets missing locally.** All 559 LFS assets are pointers, including every Sharpa mesh, so nothing can run
   without `git lfs pull`. No reference corpora are shipped (`configs/README.md:31`; `docs/package_structure.md:32-33`).
5. **Converting a custom reconstruction.**
   - **Structure.** One row per file. The floating Sharpa hands need `ManoSharpaData` with **both hands** fully
     populated. Contacts must be non-empty (K ≥ 1) per side. The 22 Sharpa finger joint names must be present.
   - **Objects.** One URDF per rigid object with correct inertial data, and a body frame equal to the reconstruction
     object frame.
   - **World frame.** Z-up, gravity along -z, with a floor or support consistent with the ground plane at the world
     origin. The support USDA must use axis-aligned `Cube`/`Cylinder` prims with translation only.
   - **Retargeting.** A retargeter must still produce `robot_*` wrist and finger targets; FlashCHORD has none
     (Question 1.8).
   - **Episode files.** A LeRobot-style per-frame `episode_*.parquet` (the test fixture naming at
     `tests/evaluation/test_object_trajectory_export.py:90`) would fail the one-row rule (`parquet.py:46-49`).
     Whether the Kaggle episodes are single-row V2D motions is not determined by source.
6. **Single-hand Track 3 episodes.** The idle Sharpa hand is simulated and tracked. It can terminate episodes at a
   0.15 m wrist error and counts toward CHORD-SR (Question 2). Two tracked objects work for Sharpa and Vega but not
   for G1/ReconBody (Question 3).
7. **Monocular noise.** Upstream recommends `force_closure` over `contact_wrench_support` for ego data
   (`../README.md:486-493`), but FlashCHORD's Sharpa recipe keeps `contact_wrench_support` at weight 10 and
   `force_closure` disabled (`configs/objective/tracking.yaml:21-27,57-61`). Frame-0 finger penetration and gravity
   tilt are warned about upstream (`../README.md:259-275`).
8. **Export alignment.** `time_s` and `reference_frame` are on the control-rate, speed-scaled timeline. For the
   Sharpa recipes the export runs at half speed (Question 7). The export carries neither source frame indices nor
   `motion_speed`, so mapping back to source frames requires the checkpoint config.
   - Exports at 4096 worlds have `T × 4096 × B` rows.
   - `reference_parquet` metadata leaks local absolute paths.
   - `eval_code_sha256` covers only four evaluation modules.
9. **Evaluation flags.** `deterministic`, `use_checkpoint_config` and `validate_policy_schema` in the evaluation
   config have no effect in `evaluate_policy.py` (Question 7).
10. **Local-only operation.** W&B defaults to online with checkpoint upload (`configs/logging/wandb.yaml:1,7`).
    Every training command must pass `logging.mode=disabled`, and `~/.wandb_api_key` should be absent or kept out
    of the container. The CoACD cache writes into the checkout (`assets/mesh.py:32-34`) unless
    `FLASH_CHORD_CACHE_DIR` is set. `docker/run.sh` writes under `$HOME` (`run.sh:131`).
11. **`train_rl.py` (PPO)** does not disable XLA preallocation locally; only the OSMO templates do
    (`workflow/train_rl.yaml:26`). JAX's default preallocation behaviour then applies, which is not determined by
    this source.
12. **Import gate, not a training gate.** The integration note says the port passed "import/scene gates, not the
    final training-reproduction gate" (`V2D_INTEGRATION.md:83-84`).
