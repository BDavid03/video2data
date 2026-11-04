# Upstream retargeting contract (NVIDIA `video_to_data`, `robotic_grounding`) — evidence report

Date: 2026-09-30. Subject: what the upstream **retargeting** stage consumes and produces, so that a Track 3
("Egocentric") team can build an adapter from its own reconstruction to the retarget → FlashCHORD stack.

## Scope and method

This report comes from reading source files only. **Nothing was executed**: no Python, `uv`, `pip`, Docker,
shell scripts, tests, validation commands, dependency resolution, network access or `git`. The material is the
sparse checkout of `nvidia-isaac/video_to_data` at commit `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, located
at `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/`. Every `path:line` citation below is relative
to that `robotic_grounding/` directory. Citations into the consumer use `flash_chord/src/flash_chord/...`, and
those files were read only to confirm what the consumer reads (another analyst covers `flash_chord/` in depth).
Instructions aimed at agents in `CLAUDE.md`, such as "run inside the container", were read as data and not
followed.

**Referenced but not available in this checkout** (nothing below guesses at their content):
- The whole **`reconstruction/` repo**. That includes the Stage-1 loader
  `reconstruction/modules/v2d_task_library_loader/lib/{ego_recon_loader,run_loader,loader_registry,read_mano,level_result_bundle}`,
  the default ego bundle `reconstruction/data/tissue_box/result_leveled`, and the upstream "ego reconstruction
  pipeline" that writes `result.npz` (`README.md:216-220`, `scripts/retarget/ego_recon_to_dexmate_sharpa.py:43-45,62-64`).
- `robotic_grounding.assets`, both the `ASSET_DIR` module (`retarget/support_recon.py:26`,
  `scripts/validate_training_assets.py:31`) and every robot/object asset: `assets/xmls/sharpawave/{left,right}_sharpawave.xml`
  (`retarget/retarget_utils.py:43`), `assets/urdfs/vega_sharpa/vega_sharpa_reduced{,_fixed}.urdf`
  (`scripts/retarget/mano_to_dexmate_sharpa.py:23`, `ego_recon_to_dexmate_sharpa.py:69`),
  `assets/urdfs/sharpawave/*_primitive.urdf` (`scripts/filter_penetrations.py:68-74`), `assets/urdfs/g1/dex3_*.urdf`,
  and the shipped `assets/human_motion_data/ego_recon/processed/` clip `tissue_box_simple` (`README.md:237-239`).
- `robotic_grounding.tasks` (`SceneConfig`, `tracking_command.py`, the `v2d`/`v2d_whole_body` envs),
  `robotic_grounding.planner` (`scripts/filter_penetrations.py:57-63`, optional import), and `robotic_grounding.rendering`.
- `source/robotic_grounding/config/extension.toml` (read by `source/robotic_grounding/setup.py:12-15`),
  `workflow/data_storage.md`, `motion_schema/motion_schema.md` (`motion_schema/README.md:12-13`), `tests/`, `ci/`,
  `visualizer/`, `groot_finetune/`, `scripts/sync_css_data.py`, `scripts/setup_css_env.sh`,
  `fetch_object_assets.py`/`upload_object_assets.py`, the monorepo `.claude/skills/*`, and the data-quality checks
  `fitting_error`/`ik_task_error` that are named in `scripts/data_assessor.py:21-28` but have no file in
  `scripts/data_quality_checks/`.

Throughout, "motion_schema" means `source/robotic_grounding/robotic_grounding/motion_schema/` and "retarget/" means
`source/robotic_grounding/robotic_grounding/retarget/`.

---

## Q1. The `ego_recon` input contract

### 1.1 Two ego retarget routes, each reading different things

| Route | Script | Robot | Reads | Writes |
|---|---|---|---|---|
| A. Floating dual Sharpa hands | `scripts/retarget/ego_recon_to_sharpa.py` (registered: `retarget/dataset_registry.py:267`) | `sharpa_wave` | **Only** the loaded `ManoSharpaData` Parquet (`ego_recon_to_sharpa.py:3-10,327-330`) | `ManoSharpaData` (unversioned) |
| B. Fixed-base Dexmate/Vega + Sharpa | `scripts/retarget/ego_recon_to_dexmate_sharpa.py` (standalone, not in the registry) | `vega_sharpa` | **`result.npz` + `mesh.obj` bundle always**, plus either the loaded Parquet (`--loaded_dir`) or MANO models (`--mano_model_dir`, legacy) (`ego_recon_to_dexmate_sharpa.py:5-7,1014,1034-1120`) | `motion_v1/single_robot` |

The documented chain for ego video is: an upstream ego reconstruction, outside this repo, produces a `result.npz`
bundle; the reconstruction repo's loader image turns that into the standard loaded Parquet; this repo retargets from
there (`README.md:216-220`). Stage 1 (load) is the GPL/MANO stage and lives in `reconstruction`
(`workflow/data_pipeline.md:43-49,55-59`).

### 1.2 Directory layout

- **Loaded (intermediate) Parquet for ego_recon:** `INTERMEDIATE_DATA_DIR/ego_recon/loaded/`
  (`retarget/dataset_registry.py:123-135,262-266`). `INTERMEDIATE_DATA_DIR` defaults to `<robotic_grounding>/.cache`
  and can be overridden with the env var `ROBOTIC_GROUNDING_INTERMEDIATE_DIR` (`retarget/__init__.py:16-19`;
  `README.md:229-233`). It is the default `--input_dir` (`ego_recon_to_sharpa.py:46-49,88`). The README contradicts
  itself here: one code comment says "Needs `<HMD>/ego_recon/loaded/`" (`README.md:244`).
- **Hive partitions:** `sequence_id=<seq>/robot_name=<robot>/` (`retarget/retarget_utils.py:23`;
  `workflow/data_pipeline.md:81-82`). Sequences are discovered by globbing `sequence_id=*` directories
  (`retarget/data_logger.py:544-562`).
- **The output keeps the loaded `robot_name`.** `ego_recon_to_sharpa.py` builds the output row from `data.to_dict()`
  and never sets `robot_name` (`ego_recon_to_sharpa.py:573-612`). The loaded partition must therefore already say
  `robot_name=sharpa_wave` for the output to land at `processed/sequence_id=<seq>/robot_name=sharpa_wave/`
  (`README.md:242-246`). The synthetic fixture writes `robot_name="sharpa_wave"` already at the loaded stage
  (`scripts/make_synthbox_fixtures.py:53,384,474-476`). What value the real ego_recon loader writes is not
  determined by source.
- **Processed output and per-clip assets:** `HUMAN_MOTION_DATA_DIR/ego_recon/processed/`. The mesh, material,
  collision STL and rigid URDF all sit in that one directory next to the partitions (`README.md:222-227`;
  `scripts/generate_rigid_urdfs.py:307-327`). `ManoSharpaData.from_parquet` uses the pyarrow dataset API precisely so
  those sibling `.obj`/URDF files are skipped (`retarget/data_logger.py:482-493`).
- **Loaded dir for route B:** `--loaded_dir` points at the same `ManoSharpaData` root
  (`ego_recon_to_dexmate_sharpa.py:891-896`). `--bundle_dir` must contain `result.npz` (`:255-257`) and `mesh.obj`
  (`:746-750`).

### 1.3 The loaded Parquet (`ManoSharpaData`, robot fields empty)

Yes, a loaded Parquet sits between reconstruction and retargeting. Its schema is `ManoSharpaData`
(`retarget/data_logger.py:639-642`) with `mano_*` and `object_*` filled and the `robot_*` time series left empty. The
synthbox fixture describes it as "ManoSharpaData with mano_* + object_* filled, robot_* empty"
(`scripts/make_synthbox_fixtures.py:17-19,460-476`), and the full column list is in Q2. Format details: one row per
sequence, zstd-compressed, written with `pq.write_to_dataset` under the partition columns
(`retarget/data_logger.py:433-460`).

**What route A actually reads from the loaded Parquet** (`ego_recon_to_sharpa.py`):

| Field | Shape / dtype | Used for | Cite |
|---|---|---|---|
| `mano_right_trans` | (T,3) f32 | **Frame count only**: `num_frames = len(data.mano_right_trans)` | `:331` |
| `mano_{right,left}_joints` | (T,21,3) f32, world, metres | IK position targets; per-finger push-out | `:485-496,462-481` |
| `mano_{right,left}_joints_wxyz` | (T,21,4) f32, **wxyz**, world | IK orientation targets; initial wrist pose | `:497-519` |
| `object_mesh_paths` | list[str] per body | OBB projector (only with `--surface_project`/`--adaptive_offset`), viser | `:343-350,414-420` |
| `object_body_position`, `object_body_wxyz` | (T,B,3), (T,B,4) wxyz | projector per frame, viser | `:462-463,391-400` |
| `object_root_position` | (T,3) | outward-offset direction (non-zero offset only) | `:224-243,355-372` |
| `object_body_names` | list[str] | viser only | `:540-547` |

Every other loaded column passes through unchanged into the processed file (`ego_recon_to_sharpa.py:573`). That
includes `fps`, the object metadata and the MANO contact fields. The consumer then requires many of them
(Q2 §2.3), so the adapter must populate them in the **loaded** Parquet.

**Hand representation.** The input is **21 joint positions plus 21 joint orientations per hand**, in MANO order:
wrist, then thumb/index/middle/ring/pinky 1–4 (`retarget/params.py:7-29`). The IK uses:

- the wrist (`hand_C_MC`, position cost 0.2, orientation cost 0.2);
- five fingertips (`*_tip_site` ← `thumb4/index4/middle4/ring4/pinky4`, position cost 1.0 or 0.5, orientation cost
  0.05–0.1);
- MP/MCP sites (thumb2, index1, middle1, ring1, pinky1; position cost 0.1, orientation cost 0)
  (`retarget/params.py:196-219`).

Orientations therefore matter for the **wrist and the fingertips**. The wrist target orientation is the MANO wrist
rotation post-multiplied by `inv(R(0.5,-0.5,0.5,0.5))` (wxyz) (`retarget/params.py:191-194`;
`retarget/hand_kinematics.py:452-457,484-488,236-251`). This hard-wires the **MANO wrist-frame convention**. Finger
positions are scaled about the wrist by `mano_to_robot_scale` (default 1.2; `ego_recon_to_sharpa.py:93`;
`retarget/hand_kinematics.py:220-234`). Both hands are always solved (`ego_recon_to_sharpa.py:307-312,521-535`).
There is no per-hand side flag beyond the column prefixes and no per-frame validity column.

**MANO parameters vs joints.** The IK never reads `mano_*_global_orient`, `mano_*_finger_pose` or `mano_*_betas`.
They are carried through only. MANO **model files are not needed at retarget time** on route A: the loader does MANO
forward kinematics (manotorch, GPL-3.0, license-gated `.pkl`) and robotic_grounding "never imports manotorch"
(`workflow/data_pipeline.md:43-49,55-59`; `docs/ARCHITECTURE.md:205-207`). The per-joint **orientations** in MANO's
kinematic convention are still required, however, and upstream obtains them from MANO FK (Q10).

**Object pose format.** `object_body_position` is (T,B,3) and `object_body_wxyz` is (T,B,4), both **wxyz**, in the
world frame, in metres (`retarget/data_logger.py:345-356`). Support reconstruction reads them as `w,x,y,z`
(`retarget/support_recon.py:366-374`). Mesh vertices are in the object/body frame and are transformed by that pose
(`retarget/object_collision.py:304-309`; `retarget/support_recon.py:729-738`). `object_root_position` is (T,3) and
`object_root_axis_angle` is (T,3), a rotation vector. `object_articulation` is (T,) float
(`retarget/data_logger.py:332-344`).

**Object mesh.** The loaded Parquet names "a single metric mesh per body" through `object_mesh_paths`
(`ego_recon_to_sharpa.py:9-10`). For ego_recon, `object_scale` is baked in by the loader and
`mesh_vertex_scale=1.0` (`retarget/dataset_registry.py:256`; `scripts/generate_rigid_urdfs.py:313-317`).

**World frame and gravity.** Route A does no frame conversion; the IK runs in whatever world the loaded Parquet uses.
Downstream code assumes **Z-up with gravity along −Z**: the outward offset treats XY as "the horizontal plane"
(`ego_recon_to_sharpa.py:227-243`), and support surfaces are Z-axis disks with a floor at z≈0
(`retarget/support_recon.py:39,480-491,503-504`). The README requires a gravity-aligned bundle and warns about
residual tilt. The shipped clip needed an 8.4° levelling done with `level_result_bundle` and loaded with
`--no_ground_align` (`README.md:268-275`); that tool is not available.

**Frame rate and timestamps.** There is a single scalar `fps` (`retarget/data_logger.py:24-29`). It passes through
from the loaded file, and no timestamps exist. The registry says ego_recon is 30 fps
(`retarget/dataset_registry.py:252`), but no retarget script in this checkout reads `DatasetConfig.fps`. The
uniform-rate assumption is implicit.

**Validity and visibility.** Route A has none. Route B checks optional `camera_is_valid`, `hand_left_is_valid`,
`hand_right_is_valid` and `object_is_valid` masks in `result.npz`. It only **warns**, noting that invalid frames
"typically hold finite stale/identity poses", and then uses them anyway. It rejects non-finite arrays
(`ego_recon_to_dexmate_sharpa.py:249-254,293-310`).

**Contact labels.** Route A computes no contacts. The `mano_{side}_{link,object}_contact_*` columns (16 MANO links
per hand) come from the loader ("Contact data (optional)", `workflow/data_pipeline.md:71-72`) and pass through.
The consumer requires `mano_{side}_object_contact_positions/normals/part_ids` to be present (Q2 §2.3). The synthbox
fixture fills them with zeros (`scripts/make_synthbox_fixtures.py:325-335`). Route B computes its own contacts
(`scripts/retarget/contact_labels.py`; see Q2 §2.2).

### 1.4 The `result.npz` bundle as read by route B

From `ego_recon_to_dexmate_sharpa.py:246-310`:

| Key | Shape (from usage) | Meaning / use |
|---|---|---|
| `gravity_alignment_applied` | bool | **Must be True**, otherwise `ValueError` (`:260-265`) |
| `camera_to_world_transform` | (T,4,4) | Per-frame camera pose. A moving camera is supported. T is taken from this key (`:267`) |
| `object_to_world_transform` | (T,4,4) | Object pose. The 3×3 block may embed a **uniform scale**, which must be constant across frames within 1e-3; it is SVD-orthonormalised and folded into the mesh vertex scale (`:1127-1149,174-187`) |
| `object_scale` | scalar (optional, default 1.0) | Mesh vertex scale factor (`:289-291,1144-1145`) |
| `hand_{left,right}_betas` | not determined (passed to MANO `betas`) | legacy FK only (`:278,345-348`) |
| `hand_{side}_wrist_orient_in_camera` | (T,3) axis-angle | MANO global orient in the **camera** frame (`:279-281`) |
| `hand_{side}_finger_pose` | (T,15,3) → (T,45) | MANO finger axis-angles (`:282-284,350`) |
| `hand_{side}_wrist_to_world_transform` | (T,4,4) | Only the **translation** is used. It is the MANO `transl` anchor (WiLoR `cam_t`), **not the wrist joint**; world joints = `R_c2w @ joints_cam + t` (`:361-390`) |
| `hand_{side}_scale` | (T,) | Mean feeds the optional `--scene_scale auto` (`:1018-1027`) |
| `{camera,hand_left,hand_right,object}_is_valid` | (T,) bool, optional | Warning only |

Further conventions on route B:

- **Left-hand params are stored in right-hand space** (WiLoR/HaMeR). The legacy route negates the y,z axis-angle
  components before running MANO's left layer (`:313-326`).
- The bundle's `mesh.obj` is parsed as text OBJ `v` lines (`:746-775`).
- With `--loaded_dir`, the loaded object tracks must match the raw `object_to_world` tracks up to a **constant
  translation** within 1e-4. The loader must therefore have run with `--no_ground_align`
  (`:393-435`).
- `--source_fps` defaults to 30 and is not read from the bundle (`:941`).

---

## Q2. What retargeting produces

### 2.1 `ManoSharpaData` (unversioned): floating Sharpa route A, full field list

Source: `retarget/data_logger.py:24-357,639-642`. All floats are float32. "(T,…)" means a time series stored as a
list per row, with T as the leading axis. Fixed-size inner lists are pyarrow fixed-size lists, so the writer
enforces them. B is ragged.

| Group | Field | Shape / dtype | Notes |
|---|---|---|---|
| Base | `sequence_id`, `robot_name` | str | Hive partition columns; stripped from the file body on write and backfilled from the path on read (`data_logger.py:501-521`) |
| | `raw_motion_file` | str | |
| | `fps` | f32 | uniform rate |
| MANO meta | `mano_flat_hand_mean` | bool | |
| | `mano_center_idx` | int32 | ego_recon uses `center_idx=None` (`dataset_registry.py:255`) |
| | `mano_to_robot_scale` | f32 | **not updated** by route A even though `--mano_to_robot_scale` is used (`ego_recon_to_sharpa.py:573-608`) |
| | `mano_{right,left}_betas` | list[f32] | 10 in the fixture (`make_synthbox_fixtures.py:390-391`) |
| | `mano_link_names` | list[str] | 16 links (`make_synthbox_fixtures.py:191-208`; `retarget/params.py:253-272`) |
| MANO per side (`right`,`left`) | `mano_{s}_trans` | (T,3) | frame count source in route A |
| | `mano_{s}_global_orient` | (T,3) | axis-angle |
| | `mano_{s}_finger_pose` | (T,45) | |
| | `mano_{s}_joints` | (T,21,3) | world, m |
| | `mano_{s}_joints_wxyz` | (T,21,4) | wxyz |
| | `mano_{s}_fitting_err` | (T,) | |
| | `mano_{s}_tips_distance` | (T,5) | fingertip→object distance, m (`scripts/retarget/verify_tips_distance.py:50-62`) |
| | `mano_{s}_link_contact_positions/normals` | (T,16,3) | |
| | `mano_{s}_object_contact_positions/normals` | (T,16,3) | |
| | `mano_{s}_object_contact_part_ids` | (T,16) int32 | **1-based body index, 0 = no contact** (`retarget/support_recon.py:322-337`) |
| Robot meta | `{s}_robot_finger_joint_names` | list[str] | 22 |
| | `{s}_robot_frame_names` | list[str] | all Pinocchio model frames, 67 (`hand_kinematics.py:73-77`; `SHARPA_NUM_FRAMES=67`, `data_logger.py:172`) |
| | `{s}_robot_frame_task_names` | list[str] | 11 IK tasks |
| Robot per side | `robot_{s}_wrist_position` | (T,3) | **free-flyer root** `q[:3]`, world, m (`ego_recon_to_sharpa.py:553`) |
| | `robot_{s}_wrist_wxyz` | (T,4) | `q[3:7]` reordered xyzw→**wxyz** (`:554-556`) |
| | `robot_{s}_finger_joints` | (T,22) | `q[7:]`, rad |
| | `robot_{s}_frames` | (T,67,7) | `[x,y,z,qw,qx,qy,qz]` world pose of every model frame (`hand_kinematics.py:404-415`) |
| | `robot_{s}_frame_task_errors` | (T,11) | final **position** error norm per task, m (`hand_kinematics.py:391-399,420-422`) |
| | `robot_{s}_num_optimization_iterations` | (T,) int32 | |
| Object | `object_name`, `safe_object_name` | str | one per sequence |
| | `object_body_names`, `safe_object_body_names`, `object_mesh_paths`, `object_urdf_paths` | list[str] (B) | |
| | `object_mesh_radius` | list[f32] (B) | max vertex norm from the object origin (compare `contact_labels.py:38-52`, `ego_recon_to_dexmate_sharpa.py:794`) |
| | `object_articulation` | (T,) | 0 for rigid |
| | `object_root_axis_angle`, `object_root_position` | (T,3) | rotation vector / m |
| | `object_body_position` | (T,B,3) | world, m |
| | `object_body_wxyz` | (T,B,4) | wxyz |

The file is one row per `sequence_id=<seq>/robot_name=sharpa_wave/<uuid>.parquet`. Any existing partition directory
is deleted before writing (`data_logger.py:443-460`). There is no schema-version tag or metadata:
`schema_version` is absent, which is how consumers tell this format apart (`retarget/support_recon.py:168-171`;
`flash_chord/src/flash_chord/data/loader.py:53-70`).

### 2.2 `motion_v1` (`single_robot`): Dexmate/Vega route B

The schema has these field groups (`motion_schema/schema.py:52-136`):

- **METADATA:** `schema_version`="motion_v1", `sequence_id`, `robot_name`, `motion_kind`, `source_dataset`,
  `raw_motion_file`, `fps` (f32), `coord_frame`.
- **ROBOT:** `robot_joint_names`, `robot_root_position` (T,3), `robot_root_wxyz` (T,4), `robot_joint_positions` (T,J).
- **EE:** `ee_link_names` (E), `ee_pose_w` (T,E,7) `[x,y,z,qw,qx,qy,qz]`.
- **HAND:** `hand_sides`, `hand_frame_names`, `hand_frames_w` per side (T,K,7), `hand_finger_joint_names`,
  `hand_finger_joints` per side (T,J_f).
- **OBJECT:** the same names as ManoSharpaData, with (T,B,3)/(T,B,4).
- **CONTACT:** per side: `hand_contact_link_names`, `hand_link_contact_{positions,normals}`,
  `hand_object_contact_{positions,normals,part_ids}`, `hand_contact_active` (T,).
- **SOURCE:** `source_kind`, `source_payload` (bytes), `source_joint_names`.
- **DIAGNOSTICS:** `ik_error_per_frame`, `ik_num_iterations`, `frame_task_errors`.

Conventions: wxyz everywhere, pose-7 packing, T leading, world frame unless `coord_frame` says otherwise
(`motion_schema/README.md:121-130`).

**Required fields.**

- **Common:** `schema_version`, `sequence_id`, `robot_name`, `motion_kind`, `fps`>0, `ee_link_names`, `ee_pose_w`.
- **`single_robot`:** adds `robot_joint_names`, `robot_root_position`, `robot_root_wxyz`, `robot_joint_positions`.
  The object group is **optional but all-or-nothing** (`object_body_names`, `object_body_position`,
  `object_body_wxyz`).
- **`dual_hand`:** additionally requires the hand group and the object group
  (`motion_schema/schema.py:151-213,245-257`).

Enforcement: the writer checks these rules and unit-norm quaternions, but **only on `robot_root_wxyz`**
(`motion_schema/writer.py:173-235,269-271`). The reader enforces the version and required fields
(`motion_schema/reader.py:173-251`). The writer uses Hive partitions `sequence_id=/robot_name=` with an optional
fixed file name (`motion_schema/writer.py:246-302`).

**What route B writes** (`ego_recon_to_dexmate_sharpa.py:1481-1527`):

- Tags: `robot_name="vega_sharpa"`, `motion_kind="single_robot"`, `coord_frame="robot_base_z_up"`,
  `source_dataset="ego_recon"`, file `data.parquet`, `fps=--target_fps` (default **50**).
- Robot: `robot_joint_positions` in fixed-URDF joint order, clamped 1e-4 rad inside limits.
  `robot_root_position`=0 and `robot_root_wxyz`=identity (fixed base).
- End effectors: `ee_link_names=["L_arm_l7","R_arm_l7"]`.
- Hands: `hand_sides=["left","right"]`, with `hand_frames_w` holding the 5 `<side>_<finger>_DP` frames per side
  (`:112-118,1396-1409`), plus `hand_finger_joints`.
- Contacts: `hand_contact_active` and `hand_object_contact_{positions,normals,part_ids}` (11 probes per side).
- Object: one body with `object_articulation`=0, and the root pose equal to the body pose.

Resampling from 30 to 50 fps is linear for joints and positions and slerp for object quaternions (`:1360-1388`). All
world poses are in the **reduced-URDF placement world**. `T_reduced_to_fixed` is deliberately *not* applied
(`:12-18,1528-1532`). The sidecar `.npz`, report `.json` and MP4 go to `RG_ROOT/out`, not the asset tree
(`:66-68,1534-1598`).

### 2.3 Required vs optional, judged by what the consumer rejects

**FlashCHORD dispatch.** A file with `schema_version=="motion_v1"` goes to `load_motion_v1`. An unversioned file must
contain the `ManoSharpaData` signature columns (`flash_chord/src/flash_chord/data/loader.py:14-70`). The
`motion_v1` loader accepts **only `motion_kind='single_robot'`** and `coord_frame ∈ {world, robot_base_z_up}`
(`flash_chord/src/flash_chord/data/motion_v1.py:196-208`). `dual_hand` is intentionally unsupported
(`flash_chord/V2D_INTEGRATION.md:44-46`). Therefore **floating Sharpa hands must be delivered as `ManoSharpaData`**.

**For a rigid 1–2-object `ManoSharpaData` sequence,** the consumer requires the following
(`flash_chord/src/flash_chord/data/mano_sharpa.py:173-311`; `flash_chord/src/flash_chord/data/parquet.py:24-65`):

- exactly **one** `.parquet` file per directory and exactly one row;
- `fps`;
- `object_body_position` (T,B,3) and `object_body_wxyz` (T,B,4), finite and unit-norm within 1e-4;
- a non-empty `object_name`;
- `object_body_names`: unique, length B;
- `object_mesh_paths`: non-empty, length B;
- `object_mesh_radius`: positive, length B;
- **`object_urdf_paths` of length B.** If absent, only `arctic`/`synthbox` articulated adapters are accepted
  (`flash_chord/src/flash_chord/assets/registry.py:179-203`);
- `object_articulation` either empty or a single all-zero column;
- per side: `robot_{s}_wrist_position` (T,3), unit `robot_{s}_wrist_wxyz` (T,4), `robot_{s}_finger_joints` (T,N) with
  matching unique `{s}_robot_finger_joint_names`, and `robot_{s}_frames` (T,F,7) with matching unique
  `{s}_robot_frame_names`;
- per side: **`mano_{s}_object_contact_positions/normals/part_ids` all present**, with part IDs integral in [0,B]
  and active normals unit length.

**Optional or unused by the consumer:** MANO params, `mano_{s}_joints*` (needed only by the retargeter),
`tips_distance`, `link_contact_*`, `fitting_err`, `frame_task_errors`, iteration counts, `raw_motion_file`,
`object_root_*` (used by viser playback, `retarget/viser_playback.py:207-223`), and `safe_*` names.

### 2.4 Companion files

- **Object mesh and rigid URDF (ego_recon, route A).** These are `processed/<name>.obj`, `processed/<name>_rigid.urdf`
  and `<name>_visual.stl`. The name must equal the loader's `make_usd_safe(object_name)` so that the stored URDF path
  resolves (`scripts/generate_rigid_urdfs.py:307-327,53-69`; `retarget/naming.py:11-16`).
- **Route B URDF.** It writes `<object_name>.obj` (vertex-scaled copy plus prefixed `.mtl`/texture copies) and
  `<object_name>.urdf`, or with `--reuse_object_assets` it reuses `<name>.obj` + `<name>_rigid.urdf`
  (`ego_recon_to_dexmate_sharpa.py:693-818`). Paths are stored as container-absolute
  `/workspace/video_to_data/...` (`:165-171,1347-1348`).
- **Support surface USDA.** For ego_recon this is `HUMAN_MOTION_DATA_DIR/ego_recon/reconstructed_stage/<seq>_support.usda`
  (`retarget/dataset_registry.py:137-145`; `scripts/reconstruct_support_surfaces.py:106-109,156-157`). The consumer
  prefers `<sequence>_<robot>_support.usda`, then `<sequence>_support.usda`, and treats support as optional unless
  the scene config requires it (`flash_chord/V2D_INTEGRATION.md:42`).
- **SDF (`<mesh>.sdf.npz`).** This is only for the offline penetration QA (`scripts/build_object_sdfs.py:4-12`;
  `scripts/object_sdf.py:142-157`). It is not a training input on this path.

---

## Q3. Multiple objects

- **Schemas:** both formats carry **B bodies**, with per-body name, mesh, URDF and radius lists, (T,B,3) positions
  and (T,B,4) quaternions (`retarget/data_logger.py:323-357`; `motion_schema/schema.py:86-100`). There is one
  sequence-level `object_name` string. There is one `object_root_position`/`object_root_axis_angle` (T,3), which is
  not per body.
- **Two independent rigid objects already flow through the ManoSharpaData/`sharpa_wave` path upstream.** TACO is
  "MANO hands + two rigid objects, `tool` and `target`" (`docs/TACO_SETUP.md:3`). Its retargeter treats
  `object_name` as `"{tool_id}_{target_id}"` and `object_body_names` as the two bodies
  (`scripts/retarget/taco_to_sharpa.py:58-70`). The registry says `has_articulated_objects=False`
  (`retarget/dataset_registry.py:153-166`).
- **Articulated single object** (ARCTIC, synthbox): bodies `bottom`/`top` plus a scalar `object_articulation`
  (`scripts/make_synthbox_fixtures.py:265-289,394-400`).
- **Route A (`ego_recon_to_sharpa.py`) does not care how many bodies there are.** The IK ignores objects; the OBB
  projector builds one box per body and projects per body, which the code comments call "multi-body safe"
  (`ego_recon_to_sharpa.py:340-350,460-481`; `retarget/object_collision.py:272-325,370-392`). The one exception is
  `_outward_offset`, which uses the single `object_root_position` for both hands (`ego_recon_to_sharpa.py:224-243`).
- **Generation and QA handle multiple objects:**
  - `generate_rigid_urdfs.py --dataset ego_recon` makes one URDF per `*.obj` in `processed/`
    (`scripts/generate_rigid_urdfs.py:318-327`).
  - Support reconstruction handles multi-body scenes, with a phantom-support filter and cross-body consolidation
    (`retarget/support_recon.py:539-602,759-780`).
  - The penetration filter loops over every body mesh (`scripts/filter_penetrations.py:449-460`).
  - Contact `part_ids` index the body (1-based) (`retarget/support_recon.py:322-337`).
- **The consumer supports independent rigid objects** when `object_urdf_paths` has one URDF per body. Each body
  becomes its own asset (`flash_chord/src/flash_chord/assets/registry.py:179-194`).
- **Route B (Dexmate/Vega) supports exactly one object.** It raises
  `"Vega ego reconstruction expects exactly one object body"` (`ego_recon_to_dexmate_sharpa.py:1068-1069`). It reads
  one `object_to_world_transform` and one `mesh.obj` (`:274-276,746-750`). Its contact labeller takes one mesh
  (`scripts/retarget/contact_labels.py:105-141`), and it writes a single body (`:1342-1350,1504-1522`).
- **The `result.npz` bundle format, as read in this checkout, holds one object.** Whether the unavailable ego_recon
  loader can emit two bodies is not determined by source.

**Conclusion.** The route to `white_pot` + `wooden_spoon` is to write a loaded `ManoSharpaData` with B=2 bodies and
two URDFs (TACO precedent), then run route A to `sharpa_wave`. The `vega_sharpa` route and the `result.npz`
contract as implemented handle **one** object only.

---

## Q4. Robots/embodiments, IK, parameters, limits, quality checks

**Selectable for ego input.**

- Registry: `ego_recon.retarget_scripts = {"sharpa_wave": ...}` only (`retarget/dataset_registry.py:267`). The generic
  dispatcher `run_retarget.py` refuses robots not in that map (`scripts/retarget/run_retarget.py:86-100`).
- `vega_sharpa` is the standalone route B.
- Dex3 exists only for arctic/taco (`dataset_registry.py:162-165,176-179`).
- G1 whole-body retargeting takes SOMA inputs (`soma_params.npz`, `poses.npy`, `output_aligned.glb`), not the ego
  hand bundle (`README.md:300-309`).
- Hands: both routes are **bimanual only**. Route A solves left and right every frame
  (`ego_recon_to_sharpa.py:307-312,521-535`), and route B places the scene using the midpoint of both wrists
  (`ego_recon_to_dexmate_sharpa.py:462-482`).

**Route A IK (floating Sharpa).**

- Model: Pinocchio `BuildFromMJCF` of `{side}_sharpawave.xml` with a free-flyer root (`retarget/hand_kinematics.py:459-464`;
  `retarget/retarget_utils.py:26-52`).
- Solver: Pink `solve_ik` with `daqp`, 11 `FrameTask`s at `lm_damping=1.0` (`hand_kinematics.py:295-312,373-383`).
- Limits: `ConfigurationLimit` + `VelocityLimit`, with `dt=1/200` s from the rate limiter
  (`hand_kinematics.py:79-93`; `retarget_utils.py:28`).
- Iterations: `max_iter=100`. It stops once every task's position-error change is below 1e-6
  (`hand_kinematics.py:385-402`; `ego_recon_to_sharpa.py:307-312`).
- Warm start: the previous frame's q. Frame 0 is initialised from the MANO wrist pose, and ego/WiLoR uses no
  link-to-site offset (`ego_recon_to_sharpa.py:54-55,504-535`; `retarget_utils.py:126-138,159-191`).
- Collisions: **none inside the IK.** Hand–object handling is pre-IK and opt-in:
  - `--surface_project` pushes penetrating MANO keypoints out of the per-body **OBB** by `--surface_margin` 5 mm,
    rigidly per `finger` (default), `hand` or `joint`;
  - `--hand_object_offset` applies a constant horizontal outward shift;
  - `--adaptive_offset` scales the shift by Savitzky–Golay-smoothed penetration depth (alpha 1.2, cap 0.03 m,
    window 9);
  - `--smooth_reference` applies Savitzky–Golay (9, 2) to joints and hemisphere-aligned quaternions before IK.
  All of these default to **off** (`ego_recon_to_sharpa.py:93-185,190-221,246-294`).

**Route B IK (Dexmate/Vega fixed base).**

- Model: Pinocchio URDF `vega_sharpa_reduced.urdf` with no root joint (`scripts/retarget/mano_to_dexmate_sharpa.py:23,418-422`).
- Tasks: 22 frame tasks (the same 11 per side, costs `:72-84`) plus a `PostureTask` (cost 1e-2) pulling towards a
  reference pose with arm joints preset (`:96-103,498-499`).
- Solver settings: `lm_damping=1e-3`, `dt=1e-2`, tolerances 5e-4 m / 2e-3 rad, `max_iters` 100, and 5 extra
  first-frame solves (`:388-414,603-634`; `ego_recon_to_dexmate_sharpa.py:945-946,1208-1224`).
- Collisions: **12 `BodySphericalBarrier`s** between `torso_l3` and `{L,R}_arm_l2..l7`, `d_min=0.3` m
  (`mano_to_dexmate_sharpa.py:183-190,478-486`).
- Joint limits: configurations are clamped 1e-6 rad inside the limits on every iterate
  (`:535-542,686-691`; `retarget/joint_limits.py:10-55`) and 1e-4 rad at output, because Isaac Lab rejects values
  exactly on the limit (`ego_recon_to_dexmate_sharpa.py:89-92,1292-1294`).
- Placement: yaw (default 180°) plus a translation that maps the scene's mean wrist midpoint onto the robot's
  reference wrist midpoint (`:462-482,943-944`).
- Targets: the wrist gets the inverse Sharpa wrist offset, and fingertips get `TIP_CORRECTION_WXYZ`
  (`:81-87,485-531`).

**Quality checks available.**

- IK diagnostics: per-frame task errors and iteration counts in route A. Route B produces a JSON report with
  converged %, position-error stats for wrists, fingertips and MPs, and active orientation error
  (`ego_recon_to_dexmate_sharpa.py:1240-1284`).
- `scripts/filter_penetrations.py`: robot capsule/sphere primitives (from `*_primitive.urdf`, not available) against
  the object SDF or a convex-hull fallback, plus hand–hand checks. The default rejection threshold is 2 cm, and very
  concave hulls are skipped (`:1-33,388-460,553-580`). It is wrapped as the data-assessor check `hand_penetration`
  (`scripts/data_quality_checks/hand_penetration.py:1-21`).
- `scripts/data_assessor.py` auto-discovers checks. Present: `hand_penetration`, `arctic_support_disks`,
  `dummy_agent_success`.
- `scripts/retarget/verify_tips_distance.py` is **hard-coded to ARCTIC `box_grab`**. It checks that
  `mano_*_tips_distance` is in [0,1) m and not all zero (`:14,50-62`).
- `scripts/validate_training_assets.py` checks only that the URDF and mesh paths stored in the Parquet exist
  (`:79-89`).

---

## Q5. Ground/support alignment

**`retarget/ground_alignment.py`** is used **only by the whole-body `soma_to_g1.py`**
(`scripts/retarget/soma_to_g1.py:19-23,370-406,455-457`). It is not on the ego hand path.

- Inputs: a `FirstPassResult` holding robot, EE, object, hand-contact and ankle trajectories
  (`ground_alignment.py:65-93`), and optionally the reconstruction's `ground_plane.json` = `{"plane":[a,b,c,d]}` **in
  OpenCV camera coordinates** (X right, Y down, Z forward). That plane is transformed CV→source→first-frame
  anchor→robot (`:996-1066`); `soma_to_g1` falls back to a frame-0 body-mesh plane when it is missing
  (`soma_to_g1.py:370-399`).
- Outputs: numeric corrections only, **no USDA**:
  - a leveling rigid transform that maps the fitted plane to z=0 (`:170-209`);
  - per-frame Z offsets that drag the lowest contact onto the plane, with a rolling median of 11 (`:264-289,360-438`);
  - an object trajectory corrected with rule 1 (follow the robot's Z during interaction) and rule 2 (hold at the
    adjacent contact pose otherwise) (`:22-38,526-632`);
  - a capped (1 cm) frame-0 object lift (`:212-260`).
- Assumptions: the robot world is Z-up, the plane is near-horizontal (`normal_z` must be non-zero), and the result is
  forced to z=0 (`:96-167`). It uses no depth or point clouds itself; the plane JSON is produced upstream, and how is
  not determined by source.

**`retarget/support_recon.py`** (driven by `scripts/reconstruct_support_surfaces.py`) works as follows.

- Input: a Parquet whose schema (motion_v1 vs mano_sharpa) is auto-detected by the presence of `schema_version`
  (`:154-171`), plus object meshes via `object_mesh_paths`, re-rooted if stale (`:55-123`). With several embodiments
  it needs `robot_name`, because "their placement worlds differ" (`:126-151`).
- Algorithm:
  - A body counts as "still" when it moves less than 1 mm/frame and less than 0.01 rad/frame for at least 5
    consecutive frames (`:228-307`).
  - For each still segment, the mesh is posed at the segment's first frame and a disk is built: center = centre of
    the XY AABB, z = lowest vertex, radius = half the larger XY span (`:377-392,726-739`).
  - Overlapping disks are merged. Disks with **z ≤ 0.05 m are dropped**, since "the sim ground plane already covers
    them" (`:39,741-757`). hot3d/oakink2 lower that threshold to -0.02 (`scripts/reconstruct_support_surfaces.py:117-129`).
  - Phantom supports (body resting on another body) are filtered, colliding disks are consolidated, and articulated
    objects use only their root body (`:669-684,759-780`).
  - An optional hand-release gate uses contact `part_ids` or `tips_distance`; it is only switched on for h2o
    (`:314-350`; `reconstruct_support_surfaces.py:144-150`).
- Output: USDA with Z-axis cylinders of height 0.01, `CollisionAPI`, `metersPerUnit=1`, `upAxis=Z`, under
  `/support_surfaces`. An empty stage is written if nothing is needed (`:474-522,782-804`).
- Assumptions: world Z-up, gravity −Z, the sim floor at z=0, and **no height offset applied**
  (`_compute_height_offset` returns 0.0, `:530-536`). It needs **only object poses and meshes**: no depth, point
  clouds or scene mesh.
- For `--dataset ego_recon` the default input is the **loaded** directory, and the output goes to the committed
  `reconstructed_stage/` (`scripts/reconstruct_support_surfaces.py:100-109,156-157`).

---

## Q6. Object assets

**`scripts/generate_rigid_urdfs.py`.**

- Discovery for ego_recon: `processed/*.obj`, with URDFs written beside them (`:307-327`).
- Mesh loading: `trimesh.load(force="mesh", process=False)` then `vertices *= config.mesh_vertex_scale`, which is 1.0
  for ego_recon. The result is exported as `<name>_visual.stl`, to avoid Isaac's OBJ parser issues (`:72-91,373-395`).
- URDF contents: a single link `object`, origin 0, **mass 0.3 kg, inertia diag 0.001**, and the **same STL for
  visual and collision**, plus a MuJoCo `<compiler balanceinertia="true">` (`:94-133`).
- Friction: none. Convex decomposition: none (collision is the raw mesh). How the simulator approximates the
  collision mesh is not determined by source.
- The script is idempotent according to the docs (`workflow/data_pipeline.md:127-129`). The code simply rewrites
  existing files.
- Articulated datasets are skipped (`:429-440`).

**Route B's own URDF.** Mass 0.3 kg; box inertia from the bounding-box extents with the **inertial origin at the
bounding-box centre**; visual = collision = the OBJ; no friction (`ego_recon_to_dexmate_sharpa.py:150,786-817`).

**`retarget/object_collision.py`** (pre-IK de-penetration only).

- Builds one **minimum-volume OBB per body** from the mesh vertices (rotating calipers via
  `trimesh.bounds.oriented_bounds`, with a PCA fallback) (`:117-147,362-392`).
- The code says the OBB is exact for boxes. For other shapes it approximates the whole body as its OBB.
- An `"sdf"` method is "reserved" and not implemented (`:12-15,379-386`).

**`scripts/build_object_sdfs.py` + `scripts/object_sdf.py`** (QA only).

- Pipeline: VHACD convex decomposition (with coarser fallbacks, then the hull), solid voxelisation at 1 mm pitch with
  2 cm padding, and an EDT → float16 SDF (positive inside) in `<mesh>.sdf.npz` (`object_sdf.py:32-139`).
- Grids over 200 M voxels are rejected as "likely mis-scaled" (`:39-41,104-111`).
- `_cm.obj` meshes are scaled ×0.01 (`:44-56`).
- The dataset list excludes ego_recon, but `--processed-dir` discovers meshes from any Parquet
  (`build_object_sdfs.py:44-45,76-98`).

**Mesh requirements** (implied, or stated where cited).

- **Units:** metres, since the scale is 1.0 for ego_recon. Do not name meshes `*_cm.obj`: support reconstruction and
  the SDF builder scale those by 0.01 (`retarget/support_recon.py:120-121`; `scripts/object_sdf.py:54-55`).
- **Origin:** the object frame whose pose is tracked, because vertices are transformed directly by the body pose
  (Q1). `object_mesh_radius` is measured from that origin.
- **Format:** anything trimesh reads for route A. Route B and `contact_labels.mesh_radius_from_obj` parse **text OBJ**
  `v` lines (`ego_recon_to_dexmate_sharpa.py:715-719,755-771`; `contact_labels.py:38-52`).
- **Watertightness:** not required by any retarget-stage code read here. Not determined by source for the simulator.

---

## Q7. Dependencies and environment

- **Python 3.11** (`>=3.11,<3.12`; `source/robotic_grounding/setup.py:38`; `README.md:696-697`). Linux is assumed:
  Docker/NVIDIA Container Toolkit prerequisites (`README.md:10-12`), pyright `pythonPlatform = "Linux"`
  (`pyproject.toml:80`), and bash wrappers.
- **Image:** `FROM nvcr.io/nvidia/isaac-lab:2.3.2`. Packages include:
  - `pin==3.7.0`, `pin-pink==4.2.0`, `coal==3.0.1`, `eigenpy==3.10.3`, the pinned `cmeel-*` closure, `mujoco`,
    `usd-core`;
  - `qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0`, `viser==1.0.15`, `scipy`, `pyrender`, `py-soma-x==0.2.1`;
  - `numpy<2.0.0`.

  torch, trimesh, pyarrow and `loop_rate_limiters` (imported at `retarget/hand_kinematics.py:10`) are not pinned
  there, and their source is not determined by this checkout (`workflow/Dockerfile:13-104`). The Isaac Sim version
  classifier is 5.1.0 (`source/robotic_grounding/setup.py:42`).
- **Isaac Sim is not needed for retargeting itself.** "retarget is pinocchio IK, no sim"; the local recipe overrides
  the Isaac entrypoint (`workflow/data_pipeline.md:173-184`). Isaac Sim is needed for Stage 5 video and for RL
  (`workflow/data_pipeline.md:229-246,272-276`).
- **Non-Isaac requirements per stage:**
  - Support reconstruction needs `pxr` (usd-core) (`retarget/support_recon.py:23`).
  - Route B's video uses pyrender/EGL, headless (`ego_recon_to_dexmate_sharpa.py:821-878`).
  - `contact_labels.py` needs only numpy/scipy/trimesh (`:11-12`).
  - Route A imports `viser` at module load even without `--visualize` (`ego_recon_to_sharpa.py:26`;
    `hand_kinematics.py:9`).
- **GPU:**
  - Route A defaults to `--device cuda:0` (`ego_recon_to_sharpa.py:90`), but torch tensors only carry values that are
    moved to CPU numpy for Pinocchio (`hand_kinematics.py:341-345`). Whether `--device cpu` works was not verified,
    since nothing was run.
  - Route B defaults to `cpu` (`ego_recon_to_dexmate_sharpa.py:951`).
  - The recommended driver is 580.126.09 with CUDA 13.0 (`README.md:36`).
- **Local scripts:**
  - `run_retarget_local.sh`/`run_pipeline_local.sh` run **inside the robotic-grounding container**.
    `run_load_local.sh` runs on the host with the loader image and needs `MANO_DIR/models/MANO_{LEFT,RIGHT}.pkl` +
    `BMC/*.npy` and the `reconstruction/` checkout (`run_load_local.sh:15,27-36`; `run_retarget_local.sh:15`).
  - **All three hard-code `{dataset}_loaded`/`{dataset}_processed`** (`run_retarget_local.sh:37-42`), as do
    `run_pipeline_docker.py` (`scripts/run_pipeline_docker.py:252,353-359`) and OSMO `workflow/retarget.yaml:116-118`.
    That conflicts with ego_recon's `loaded`/`processed` overrides and `.cache` location, so ego_recon has to be run
    with the direct commands in `README.md:241-257`.
  - `ASSETS_DIR` is package-relative and not overridable by env; only `HUMAN_MOTION_DATA_DIR` is
    (`docs/TACO_SETUP.md:115-118`).

---

## Q8. Dataset registry

- `DatasetConfig` fields (`retarget/dataset_registry.py:32-145`):
  - identity and data: `name`, `fps`, `mano_kwargs`, `mesh_vertex_scale`, `mesh_format`;
  - capability flags: `has_articulated_objects`, `has_contact_data`, `has_support_surfaces`;
  - IK: `link_to_site_quat_wxyz`;
  - storage: `loaded_suffix`/`processed_suffix`/`loaded_dir`/`processed_dir`/`loaded_in_intermediate`,
    `css_raw_prefix`;
  - scripts: a deprecated `loader_script` and `retarget_scripts{robot: path}`.

  `get_dataset_config` raises for unknown names (`:277-286`). No retarget script in this checkout reads `fps`,
  `mesh_format`, `link_to_site_quat_wxyz`, `has_contact_data` or `has_support_surfaces`; grep finds no reader
  outside the registry. `mesh_vertex_scale` *is* read (`generate_rigid_urdfs.py:374-375`;
  `ego_recon_to_sharpa.py:345`).
- **Adding a dataset** takes three steps (`workflow/data_pipeline.md:389-400`):
  1. add a `DatasetConfig` entry;
  2. write a loader in `reconstruction/.../lib/<name>_loader.py` and register it in `loader_registry.py` (not
     available here);
  3. write `scripts/retarget/<name>_to_sharpa.py`.

  After that, "everything else" is registry-driven. `generate_rigid_urdfs.py` also needs a `_discover_<name>_objects`
  function (`scripts/generate_rigid_urdfs.py:330-351`).
- **Registered:** taco, arctic, synthbox, oakink2, hot3d, h2o, grab, dexycb, **ego_recon**
  (`retarget/dataset_registry.py:152-269`). ego_recon is `fps=30`, `mano_kwargs={"flat_hand_mean": True,
  "center_idx": None}`, scale 1.0, `obj`, rigid, `loaded_dir="loaded"`, `processed_dir="processed"`,
  `loaded_in_intermediate=True`, with `sharpa_wave` only (`:250-268`).
- **The challenge's Track 3 data is not registered.** No registry entry, doc or script mentions it. The only ego clip
  referenced is `tissue_box_simple` (`README.md:237-239`).

---

## Q9. Licences and licence-gated inputs

- **Code:** "source code under Apache-2.0", documentation/skills under CC-BY-4.0 (top-level `LICENSE:1-2`; SPDX
  headers throughout). There is an **inconsistency**: `workflow/Dockerfile:1-7` and `pyproject.toml:1-7` carry an NVIDIA
  proprietary header ("without an express license agreement … strictly prohibited").
- **MANO:** license-gated; users download `MANO_{LEFT,RIGHT}.pkl` from mano.is.tue.mpg.de themselves, and it is never
  committed (`README.md:47-50`; `docs/ARCHITECTURE.md:205-207,316`).
  - It is used **only at Stage 1 (load)** through manotorch, which is **GPL-3.0**, and is contained in the separate
    loader image (`workflow/data_pipeline.md:43-49,55-59`; `workflow/Dockerfile:44-46`).
  - Route A needs no MANO.
  - Route B needs MANO + manotorch + `read_mano` only for the legacy `--mano_model_dir` fallback
    (`ego_recon_to_dexmate_sharpa.py:897-902,1085-1099`).
  - The joint orientations route A consumes are MANO-FK products (Q10).
- **SMPL-X:** mentioned only for the GRAB dataset (`workflow/data_pipeline.md:363`); not on this path.
- **SOMA:** `py-soma-x` is for whole-body G1 only (`workflow/Dockerfile:48-54`).
- **Robot assets** (Sharpa Wave MJCF, Vega/Dexmate URDFs): not available, and their licence is not determined by
  source.
- **Base image and datasets:** the terms for `nvcr.io/nvidia/isaac-lab:2.3.2` are not determined by source. Datasets
  are "per the dataset's own license/terms" (`workflow/data_pipeline.md:419-421`). The ego path mentions
  WiLoR/HaMeR hand estimation (`ego_recon_to_sharpa.py:54,137-141`; `ego_recon_to_dexmate_sharpa.py:316-320`); their
  licences are not determined by source.

---

## Q10. Gaps and risks for a team-built reconstruction

**Hard to get from video alone**

1. **Per-joint orientations in MANO convention.** Route A needs `mano_*_joints_wxyz` (21×4) as well as positions. The
   wrist and fingertip orientations are active IK targets, and the wrist target applies a fixed MANO→Sharpa offset
   (`retarget/params.py:191-219`; `retarget/hand_kinematics.py:236-251`). Joint positions alone are not enough. A
   non-MANO hand model has to reproduce MANO's joint-frame convention, which upstream gets from MANO FK (licence and
   GPL, Q9).
2. **Both hands on every frame.** Neither ManoSharpaData nor route A has a hand-validity field. In egocentric video one
   hand is often off-screen, and there is no upstream strategy in this checkout for missing hands. Route B uses
   invalid (stale) frames after only a warning (`ego_recon_to_dexmate_sharpa.py:249-254,293-304`).
3. **Metric scale.** The contract is that bundles are metric (`--scene_scale` default 1.0; `auto` is an "escape
   hatch"; `ego_recon_to_dexmate_sharpa.py:928-939`). Object meshes must be metric in the tracked object frame.
4. **Gravity-aligned Z-up world.** Residual tilt makes objects rest on an edge (`README.md:268-275`). The upstream
   levelling tool is unavailable. How the reconstruction world relates to the sim floor at z=0 is not specified by
   any retarget code. Support disks at z≤0.05 m are silently dropped, and no height offset is applied
   (`retarget/support_recon.py:530-536,745-757`).
5. **Contacts.** For route A the consumer needs `mano_*_object_contact_*` (16 slots per side). Upstream they come from
   the loader's MANO-surface computation, which is unavailable. Zeros are schema-valid (synthbox) but leave contact
   rewards without signal. The README recommends the `force_closure` reward for monocular data because reconstructed
   contacts are noisy (`README.md:265-267,488-521`).
6. **A known object mesh per body.** The mesh must be metric, in object-frame coordinates, and named consistently
   with `make_usd_safe(object_name)` so the stored `<name>_rigid.urdf` path resolves (`scripts/generate_rigid_urdfs.py:307-317`).

**Implicit assumptions**

7. A **uniform frame rate** and no timestamps (Q1). Route B's `--source_fps` is a CLI default of 30 and is not read
   from the data.
8. **One object** on route B and in the `result.npz` bundle format (Q3).
9. The floating-hand penetration handling relies on an **OBB proxy** that is "exact for box-like objects"
   (`retarget/object_collision.py:12-15`). A pot's OBB fills its interior, so fingers inside the rim get pushed out;
   a spoon's OBB is a slab. The SDF projector is unimplemented.
10. `_outward_offset` uses **one** `object_root_position` for both hands. With two objects it depends on whichever
    root the adapter writes (`ego_recon_to_sharpa.py:224-243`).
11. The frame count comes from `mano_right_trans` (`ego_recon_to_sharpa.py:331`). If that column is empty, zero frames
    are solved without error.
12. The output `robot_name` is inherited from the loaded partition (`ego_recon_to_sharpa.py:573-612`).
13. The ego_recon directory overrides are ignored by `run_*_local.sh`, `run_pipeline_docker.py` and `retarget.yaml`
    (Q7).

**Places where upstream silently defaults or passes things through**

14. `from_parquet` fills missing columns with nulls; if several rows match, it prints a message and uses row 0
    (`retarget/data_logger.py:501-531`). The IK's convergence flag is not recorded in route A, only iterations and
    errors.
15. Route A keeps the loaded `mano_to_robot_scale` even though it used a different CLI value
    (`ego_recon_to_sharpa.py:573-608`).
16. Physics defaults: URDF mass 0.3 kg, inertia 0.001, no friction, collision = visual mesh
    (`scripts/generate_rigid_urdfs.py:111-128`).
17. Route B writes an **objectless** Parquet (only a warning) if the object assets cannot be written
    (`ego_recon_to_dexmate_sharpa.py:1330-1358`).
18. The `motion_v1` writer validates only `robot_root_wxyz` quaternions (`motion_schema/writer.py:269-271`). Unit-norm
    object quaternions are enforced later by the consumer (`flash_chord/src/flash_chord/data/mano_sharpa.py:185-189`).
19. Parquets bake absolute or container paths into `object_mesh_paths`/`object_urdf_paths`
    (`ego_recon_to_dexmate_sharpa.py:165-171`; `docs/TACO_SETUP.md:119-124`). The consumer re-roots paths that contain
    `/assets/` (`flash_chord/V2D_INTEGRATION.md:40`).
20. `MotionData.num_frames()` reads `robot_root_position`. For a `dual_hand` file that field can be `None`, which makes
    `trim`/`resample` fail (`motion_schema/schema.py:507-512,602-604`). The consumer does not accept `dual_hand`
    anyway.

**Documentation vs code discrepancies noticed**

- `workflow/data_pipeline.md:8-14` and `retarget/data_logger.py:635-638` say the Dex3 producers write `motion_v1`, but
  `scripts/retarget/arctic_to_dex3.py:281-282` writes `ManoDex3Data`.
- `README.md:244` gives the loaded dir as `<HMD>/ego_recon/loaded/`, while `README.md:232` and
  `retarget/dataset_registry.py:266` put it in `.cache`.
- `scripts/reconstruct_support_surfaces.py:125-128` prints "-0.01" after setting -0.02.
