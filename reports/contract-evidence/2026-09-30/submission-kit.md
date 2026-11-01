# V2D Submission Kit: Track 3 contract evidence

## Scope and method

This report comes from reading source code only. I executed no code. I did not run Python,
any kit script, tests, `pip`, dependency resolution, the preflight, the uploader, or any
network call. I did not open the binary Parquet samples; their schema is inferred from the
packer, reader and scorer source. I inspected `data/track_3_sample_submission.csv` with text
tools only (`head`, `tail`, `awk`, `wc`, `grep`, `od`). I did not verify file hashes against
`KIT_MANIFEST.json`, because that would be a validation step.

Kit root, abbreviated `KIT/` below:
`Z:\hugging-face\video2data\upstream\v2d_submission_kit\v2d_submission_kit\`.
Citations use `path:line` relative to `KIT/`. Where the files do not settle a point, the report
says **not determined by the kit**.

Files read in full: `tools/pack_submission.py`, `tools/pack_reconstruction.py`,
`tools/preflight.py`, `submit/v2d_submit.py`, `v2dlb/{object_pose,report_schema,
submission_parquet,object_mesh,mesh_budget,mesh_common,dataset,vertices,trajectory}.py`,
`metric_code/track_3/{AUC,CD-O}.py` (the other four pose scorers were diffed against `AUC.py`),
`metric_code/README.md`, `config/tracks.json`, `data/track_3_evaluation_objects.json`,
`tracks/track_{2,3}/README.md`, `README.md`, `LEADERBOARDS.md`, `KIT_MANIFEST.json`,
`skills/v2d-submit/SKILL.md`, `flash_chord/scripts/evaluate_policy.py`, the three Hydra YAMLs,
`examples/run_policy_evaluation.py` and `requirements.txt`.

---

## 1. Evaluator trajectory parquet (`episode_XXXXXX.parquet`)

### 1.1 Discovery and episode identity

- The packer loads **every** `*.parquet` under `--results` **recursively** (`rglob`) and parses
  each one as a trajectory (`tools/pack_submission.py:57-64`). If any parquet is unreadable,
  the whole run aborts (`v2dlb/submission_parquet.py:115-118`). If two files resolve to the same
  episode, the run also aborts (`tools/pack_submission.py:59-63`).
  The preflight differs: it only globs `episode_*.parquet` at the **top level** and reads the
  episode number from the filename stem (`tools/preflight.py:197,205`).
- Episode index resolution, first match wins (`v2dlb/submission_parquet.py:89-108`):
  1. metadata `flash_chord.episode_index`
  2. metadata `episode_index`
  3. regex `episode_(\d+)` on metadata `flash_chord.reference_parquet`
  4. the same regex on metadata `reference_parquet`
  5. the same regex on the file name

  The packer requires no particular number of digits. The SKILL says the names must be 6-digit
  (`skills/v2d-submit/SKILL.md:49-50`).
- Parquets for episodes outside the scored roster are read and validated, then ignored with a
  note (`tools/pack_submission.py:231-233`).

### 1.2 Accepted layouts and required columns

**A. FlashCHORD schema.** This layout is selected when any column name starts with `achieved_`
(`v2dlb/submission_parquet.py:136`). The declared schema comes from the module docstring
(`v2dlb/submission_parquet.py:6-18`). The table's "Enforced by reader" column shows what the
reader actually checks.

| Column | Declared dtype | Meaning | Enforced by reader |
|---|---|---|---|
| `environment_id` | int32 | parallel evaluation world, used as `world_index` | **required** (`:140-144`), cast to int64 (`:148`) |
| `step` | int32 | 0-based rollout step | fallback frame index when `reference_frame` is absent (`:145-147`) |
| `reference_frame` | int32 | `step + start_frame` | used as `frame_index` when present (`:145-147`) |
| `time_s` | float64 | `step / control_fps` | not read |
| `object_id` | int32 | object a body belongs to | optional, defaults to `object_body_id` (`:150-151`); **not used by the Track 3 packer or scorer** |
| `object_body_id` | int32 | rigid body, in reference-body order | **required** (`:140-144`); becomes the **object slot** (`:149`) |
| `object_body_name` | dict<str> | body name | optional (`:152-153`); if present, must equal the reference object order (see 1.5) |
| `environment_valid` | bool | false where the world recorded a non-finite pose | optional, defaults to all true (`:154-155`) |
| `achieved_position_{x,y,z}`, `achieved_quaternion_{x,y,z,w}` | float32 | achieved pose | **all 7 required** (`:77-86`, `:137`) |
| `reference_position_{x,y,z}`, `reference_quaternion_{x,y,z,w}` | float32 | evaluator's reference | optional. If any `reference_position*` column exists, all 7 are required (`:138-139`). **Not used at all for Track 3** (`tools/pack_submission.py:321`) |

- The docstring says rows are ordered step, environment, body (`:17`). The reader does not
  depend on row order; the packer re-indexes by (world, frame, slot)
  (`tools/pack_submission.py:262-297`).
- Dtypes are not enforced. Every numeric column is cast to int64 or float64
  (`v2dlb/submission_parquet.py:86,145-155`).
- The shape is long format: N rows, one per (step, environment, body). The packer turns it into
  N×7 float64 (`v2dlb/submission_parquet.py:57`).

**B. Legacy layout.** The docstring says this layout is kept for "hand-made fixtures and any
earlier exports" (`v2dlb/submission_parquet.py:25-26`, code at `:156-193`). Its requirements:
- The pose is either the 7 columns `pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w`, or a
  single width-7 list column named `pose`, `achieved_pose_w` or `pose_xyzw`.
- `frame_index` is required.
- `world_index` is optional (defaults to 0).
- Either `object_slot` or `object_name` is required. With only `object_name`, slots are assigned
  in first-appearance order (`:180-189`).
- No reference pose and no validity flag are read.

### 1.3 Units, quaternion order and frame

- **Quaternion order.** The reader takes metadata `flash_chord.quaternion_order`, then
  `quaternion_layout`, and defaults to `"xyzw"`. Only `xyzw` and `wxyz` are accepted. `wxyz`
  input is reordered to xyzw (`v2dlb/submission_parquet.py:126-134,195-199`). The rest of the
  pipeline uses xyzw (`v2dlb/object_pose.py:9-12,46-47`). The released HF dataset parquets use
  `[x,y,z,qw,qx,qy,qz]` (w-first), converted in `v2dlb/trajectory.py:13-15,43-49`. This does not
  apply to the submission file.
- **Units.** Metres are assumed and **never checked**. The docstring says positions are metres
  and that "both [are] declared in the metadata rather than assumed"
  (`v2dlb/submission_parquet.py:17-18`). However, only the quaternion order key is read.
  `flash_chord.position_unit` is written by the test helper `write_example` (`:250`) and is never
  read.
- **Coordinate frame.** For Track 3 the packer applies **no transform**. The achieved pose is
  copied into the CSV as-is (`tools/pack_submission.py:297,345-346`). The only frame handling is
  the scorer's initial-pose alignment (section 3). By contrast, Track 2 converts into the
  released frame using the evaluator's recorded reference (`tools/pack_submission.py:82-130,
  321-334`).
- **Finite values.** Any NaN or Inf in any achieved pose, in any world, rejects the file
  (`v2dlb/submission_parquet.py:201-206`).
- **Quaternion norm.** The Track 3 packer does not check or normalise quaternions. The uploader
  rejects norms below 1e-8 (`submit/v2d_submit.py:161-163`). The scorer normalises and rejects
  norms below 1e-8 (`metric_code/track_3/AUC.py:209-216,418`).

### 1.4 Frame and world indexing

- `frame_index` is `reference_frame` (`step + start_frame`), or `step` when that column is
  absent (`v2dlb/submission_parquet.py:8,145-147`). The evaluator default is `start_frame: 0`
  (`flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml:7`). The example runner does
  not override it (`examples/run_policy_evaluation.py:71-80`).
- The schema has one row per `step` and a `reference_frame = step + start_frame`. This implies
  exactly one control step per reference frame. `time_s = step / control_fps` suggests the
  control rate equals the reference frame rate. The kit does not state the Track 3 reference
  frame rate. **Not determined by the kit.**
- `world_index` is taken from `environment_id`. Track 3 scores **only world 0**, because the
  public roster lists only `w000` (section 2) and `worlds_per_episode` is 1
  (`config/tracks.json:229`). The Hydra default is `world_count: 4096` (`flash_sac.yaml:6`); the
  example runner's default is `--world-count 1` (`examples/run_policy_evaluation.py:42`). Extra
  worlds are silently dropped by the packer (`tools/pack_submission.py:295-297`). However, the
  `environment_valid` check covers **all** worlds in the file, so one bad world rejects the
  episode even when it is not submitted (`:266-273`).

### 1.5 Object slots

- The slot is `object_body_id`. The expected order is
  `data/track_3_evaluation_objects.json` (`tools/pack_submission.py:148-161`). This order is
  "NOT always sorted by name" (`data/track_3_evaluation_objects.json:4-5`). In the scored set,
  e10 (`blue_cup, beige_cup`) and e19 (`mini_sweeper, mini_dust_pan`) are not alphabetical.
- If `object_body_name` is present, the names selected for the roster rows must equal the
  reference names in slot order, or the episode is rejected
  (`tools/pack_submission.py:299-312`).
- If names are absent, **no slot/name check happens**. A body-order mismatch would then
  silently swap objects.
- If an object is made of several bodies, `object_body_id` would no longer equal the object
  slot. `object_id` is read but unused for Track 3 (the scorer uses the slot as object id;
  `metric_code/track_3/AUC.py:194,206`).

### 1.6 Schema-metadata keys read

| Key (either form) | Where | Use |
|---|---|---|
| `flash_chord.schema` | `v2dlb/submission_parquet.py:63-65` | `schema_version`; only affects Track 2 (`tools/pack_submission.py:314-319`) |
| `flash_chord.quaternion_order` / `quaternion_layout` | `submission_parquet.py:126-134` | quaternion order |
| `flash_chord.episode_index` / `episode_index` | `:90-95` | episode id |
| `flash_chord.reference_parquet` / `reference_parquet` | `:96-103` | episode id fallback |
| `flash_chord.eval_script_sha256` / `eval_script_sha256` | `:223-234` | provenance |
| `flash_chord.eval_code_sha256` / `eval_code_sha256` | same | provenance |
| `flash_chord.checkpoint_sha256` / `checkpoint_sha256` | same | provenance, per episode |

Written by `write_example` but **never read**: `flash_chord.world_count` and
`flash_chord.position_unit` (`submission_parquet.py:248-250`).

Hash rules in the packer:
- `eval_script_sha256` must be non-empty somewhere. If no parquet records it,
  `--eval-script <file>` makes the packer hash that file instead
  (`tools/pack_submission.py:354-361`).
- `eval_code_sha256` has **no fallback**. At least one parquet must record it, or the run fails
  with "nothing records 'eval_code_sha256'" (`:380-388`). The hint printed for that failure
  talks about the metrics JSON and `checkpoint_sha256`, so it is wrong for this key.
- Both evaluator digests must be uniform across episodes. Blanks are discarded first, so a mix
  of "recorded" and "not recorded" passes (`:363-376`).
- `checkpoint_sha256` may differ between episodes (`:236-238,367-379`). Every scored episode
  must have one (`:431-445`).

### 1.7 Optional `episode_XXXXXX.json`

- The packer loads every `*.json` under `--results` recursively. Unparseable files are skipped.
  The episode is found from the JSON's `"parquet"` value, or else from its file name, via
  `episode_(\d+)` (`tools/pack_submission.py:65-73`). Later files silently overwrite earlier
  ones for the same episode. A JSON whose top level is not an object would crash on
  `report.get` (`:70`).
- Keys the kit reads:
  - `checkpoint_sha256`, used only when the parquet metadata lacks it (`:251`).
  - `metrics.{add_auc, spider_sr_uncentered, maniptrans_sr, relative_position_error_cm,
    mean_per_point_error_cm}` (`v2dlb/report_schema.py:58-64`), compared with a local
    recomputation at `--tolerance` 1e-4 (`tools/pack_submission.py:176-179,518-543`).
- **For Track 3 the metric cross-check never runs.** It needs the reference, which is held out
  (`tools/pack_submission.py:447-451`). On Track 3 the JSON therefore only matters as a
  checkpoint-hash fallback.
- The JSON is written by `write_evaluation_report` in the FlashCHORD runtime
  (`flash_chord/scripts/evaluate_policy.py:67-68`). Its full contents are **not determined by
  the kit**.

---

## 2. Pose submission CSV

### 2.1 Columns and format

- Exact header, in this order (`tools/pack_submission.py:44-45,426`; sample line 1):
  `row_id,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,eval_script_sha256,eval_code_sha256,checkpoint_sha256,packer_script_sha256,uploader_script_sha256,code_commit_url`
- Floats are written with `float_format="%.6f"` (`tools/pack_submission.py:454`), i.e.
  1 µm position resolution and 1e-6 per quaternion component.
- There is no index column. The sample file has LF line endings, no CR, and a trailing newline.
- Provenance values: the packer's own SHA-256 (`:421`), the SHA-256 of `submit/v2d_submit.py`
  on disk (`:422-425`), the commit URL (`:420`), the uniform evaluator digests (`:416-419`) and
  the per-episode checkpoint hash (`:347`).

### 2.2 `row_id` grammar

`track_{track}/e{episode:06d}/w{world:03d}/f{frame:06d}/o{slot}` (`tools/pack_submission.py:50-51`),
parsed with `^track_3/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)$` (`:139`). Example:
`track_3/e000001/w000/f000000/o0`. Every sample `row_id` is 31 characters long.

### 2.3 Roster (derived from `data/track_3_sample_submission.csv` with text tools)

- 9,121 lines: 1 header plus **9,120 data rows**. There are 20 episodes, all with world `w000`
  only.
- Every episode's frames are **contiguous from `f000000` to N−1**.
- Rows are sorted by (episode, frame, slot) with no order breaks.
- Every sample row is a placeholder: pose `0,0,0,0,0,0,1`, 64-zero hashes, and
  `https://github.com/your-org/your-repo/commit/000…0`. That URL matches the commit regex.

| Episode | Slots (slot 0, slot 1) | Frames (0..N−1) | Rows |
|---|---|---|---|
| 1 | white_pot, white_pot_lid | 335 | 670 |
| 3 | white_pot, wooden_spoon | 313 | 626 |
| 5 | white_pot, wooden_spoon | 288 | 576 |
| 6 | white_pot, wooden_spoon | 243 | 486 |
| 8 | beige_cup, white_pot | 219 | 438 |
| 10 | blue_cup, beige_cup | 161 | 322 |
| 14 | blue_cup, plastic_dish_rack | 158 | 316 |
| 15 | plastic_dish_rack, wooden_spoon | 166 | 332 |
| 16 | plastic_dish_rack, wooden_spoon | 173 | 346 |
| 19 | mini_sweeper, mini_dust_pan | 235 | 470 |
| 20 | blue_cup, water_pitcher | 202 | 404 |
| 25 | wooden_piece_1, wooden_piece_2 | 356 | 712 |
| 28 | plastic_dish_rack, white_pot | 220 | 440 |
| 30 | mini_sweeper, plastic_dish_rack | 263 | 526 |
| 32 | blue_cup, wooden_spoon | 259 | 518 |
| 33 | blue_cup, wooden_spoon | 348 | 696 |
| 35 | wooden_spoon | 421 | 421 |
| 36 | wooden_spoon | 407 | 407 |
| 37 | wooden_spoon | 151 | 151 |
| 38 | mini_dust_pan | 263 | 263 |

Totals: 5,181 frames. The 16 two-object episodes contribute 3,939 frames (7,878 rows); the 4
single-object episodes contribute 1,242 rows. The episode list matches
`config/tracks.json:206-227`. Object names come from `data/track_3_evaluation_objects.json:8-85`.
The CSV holds only slot numbers, not names.

Holdout episodes 4, 7, 9, 17 and 34 have ground truth but no released video, so they are not
scored (`config/tracks.json:248`).

### 2.4 Mapping rollout frames to roster frames

- Matching is by **exact integer equality** of (world, frame, slot) against the roster grid
  (`tools/pack_submission.py:256-297`). There is no interpolation, resampling or time
  alignment.
- **Rollout shorter than the roster, or any missing (w, f, o):** the episode is rejected with a
  diagnostic giving submitted and expected frame ranges (`:277-293`). The message notes that the
  evaluator's frames are contiguous (`step + start_frame`), while released references can drop
  invisible frames, and says any disagreement "needs resolving with the organisers" (`:288-291`).
  For Track 3 the roster is contiguous from 0, so this conflict does not show up in the roster.
- **Rollout longer, extra worlds or extra bodies:** the extra rows are silently ignored. Only
  roster rows are selected (`:295-297`).
- **Duplicate (world, frame, slot) anywhere in the file:** rejected (`:274-276`).
- Output row order equals the roster order: episodes in ascending order, grid order within each
  episode (`:198-207,242,256-257,339-348,415`).

### 2.5 Rejection conditions

**Packer (`tools/pack_submission.py`)**
- The sample CSV or objects JSON is missing (`:134-137`).
- A roster `row_id` is invalid or duplicated (`:140-141`).
- The roster episodes differ from `scored_episodes` (`:145-147`).
- An episode has no object names (`:151-153`).
- A roster episode grid is incomplete (`:159-160`).
- Any scored episode has no trajectory (`:226-230`).
- Two files claim one episode (`:59-63`).
- Any `TrajectoryError`: unreadable file, zero rows, unrecognised layout, partial FlashCHORD
  columns, missing `environment_id`/`object_body_id`, unknown quaternion order, non-finite
  achieved pose, or undeterminable episode (`v2dlb/submission_parquet.py:77-206`).
- Any `environment_valid == false` row (`:266-273`).
- Duplicate grid rows (`:274-276`).
- Missing grid rows (`:277-293`).
- Object-name order mismatch (`:299-312`).
- Evaluator digests not uniform (`:372-376`), or `eval_script_sha256`/`eval_code_sha256`
  entirely absent (`:380-388`).
- `--code-commit` does not match the pattern (`:405-413`).
- Any blank provenance cell (`:431-445`). See the crash note in section 8.

**Uploader (`submit/v2d_submit.py:124-218`)**
- The file is not `.csv`, or cannot be read.
- Missing columns: extra columns are allowed, missing ones are not (`:131-137`).
- Zero rows, or duplicate `row_id`.
- A `row_id` does not start with `track_3/` (`:143-147`).
- `row_id` values **or order** differ from the sample (`:148-156`).
- A non-finite pose (`:158-160`), or a quaternion norm below 1e-8 (`:161-163`).
- The commit does not match the pattern (`:165-171`).
- A blank provenance cell (`:173-183`).
- A uniform column is non-uniform (`:188-194`).
- The per-episode checkpoint check runs only if the file has an `episode_index` column. Packer
  output does not, so the check is effectively skipped locally (`:196-206`).
- The allowlist check runs only if `accepted_*` is non-empty (`:208-217`). It is empty
  (`config/tracks.json:27-28`).

**Scorer (`metric_code/track_3/AUC.py`; identical in the other four)**
- A required column is missing, or a pose column is non-numeric or non-finite (`:296-309`).
- Duplicate `row_id` (`:310-314`).
- Any solution row has no submission row (`:396-405`). The merge is
  `how="left", validate="one_to_one"`, so **extra submission rows are ignored by the metric**.
  Whether Kaggle's platform rejects them before scoring is **not determined by the kit**.
- Provenance is blank, not uniform, or has more than one checkpoint within an episode; the
  commit regex fails; or the allowlist is non-empty and the digest is not on it (`:317-371`).
- The episode grid is not rectangular (`:189-206,411-416`).
- A degenerate quaternion (`:418`).

---

## 3. Initial-pose alignment (`align_initial_pose: true`)

- **Where it is applied.** Only in the **scorer**. `_ALIGN_INITIAL_POSE = True` in all five
  Track 3 pose scorers (`metric_code/track_3/AUC.py:221,424-425`), and it is the only
  difference from the Track 2 copies (Track 2 has `False`; diff of `track_2/AUC.py` against
  `track_3/AUC.py`). The switch comes from `config/tracks.json:246`.
- The packer **does not** apply it, and the CSV carries raw simulator-frame poses. The
  packer's local `crosscheck()` would apply it through
  `op.episode_metric(..., align_initial_pose=...)` (`tools/pack_submission.py:505-506`), but
  that function never runs for Track 3 (`:447-451`).
- **What is fitted** (`v2dlb/object_pose.py:122-163`; public copy `AUC.py:113-127`). Per
  evaluation world *w*, the transform comes from the **first roster frame** (index 0 after
  sorting by frame) and **object slot 0**. Quaternions are normalised first (`AUC.py:418`):
  - `q_align = q_ref[0, slot0] ⊗ conj(q_ach[0, w, slot0])`, normalised
  - `R_align = R(q_align)`
  - `t_align = p_ref[0, slot0] − R_align · p_ach[0, w, slot0]`
  - It is then applied to every frame and every object: `p' = R_align p + t_align`,
    `q' = q_align ⊗ q`.
- **It is rigid, not a similarity.** "No scale is fitted, so a submission cannot shrink its way
  to a smaller error" (`object_pose.py:141-142`). The Track 3 README says "Trajectory scale error
  remains scored" (`tracks/track_3/README.md:34-36`).
- **What remains scored:**
  - every frame after the first, including drift;
  - every object other than slot 0, with the inter-object relationship preserved;
  - scale.

  Object 0's frame-0 error becomes exactly zero (`object_pose.py:148-150`). RPE is invariant
  to the warp; the other four metrics change (`:144-146`).
- **What this implies for the simulator world frame:**
  - The simulator may express poses in **any right-handed world frame that differs from the
    reference world by one proper rigid transform**. This includes a different origin, a
    different up-axis or gravity direction, and a different yaw.
  - Positions must be in **metres**, because no scale is removed, and the frame must be
    consistent across all objects and frames of an episode.
  - A reflection (a left-handed frame) cannot be absorbed.
  - The transform is estimated from the *achieved* object-0 pose at frame 0, not from a
    reference-to-reference comparison as in Track 2 (`tools/pack_submission.py:82-130`). Any
    error in object 0's initial pose is therefore treated as a world-frame offset and
    propagated to everything else. An orientation error at frame 0 becomes a rotation of the
    whole trajectory about object 0's start. That creates lever-arm position errors for object 1
    and for later frames.
  - Only a **world-side** transform is removed. Suppose a participant's object body frame
    (mesh origin and axes) differs from the reference object's body frame by a constant
    `B`, so that `T_ach(t) = W · T_ref(t) · B`. The alignment does not cancel `B`. For
    object 0, the residual grows as the object rotates away from its frame-0 orientation. For
    object 1, `B₁` is never removed.
  - The kit does **not** define the body-frame convention (origin and axes) of Track 3's
    confidential reference objects. **Not determined by the kit.** This is the main open
    question for any pipeline whose objects come from its own reconstructed meshes.

---

## 4. Metric formulas as implemented

Source of truth: the CORE block in `v2dlb/object_pose.py:22-293`, inlined into each scorer.
The five Track 3 pose scorers are byte-identical except for `_METRIC`/`_METRIC_DISPLAY` at
lines 222-223 (diffed). Tensors are achieved `(T, W, B, 7)` and reference `(T, 1, B, 7)`. For
Track 3, W = 1 and B is 1 or 2.

| Metric | Formula (code) | Thresholds | Aggregation within episode | Direction |
|---|---|---|---|---|
| **AUC** (`add_auc`) | `ADD[t,w,b] = mean_v ‖(R_a − R_r) v + (p_a − p_r)‖` over the object's point set `v` (`object_pose.py:80-93`). This is plain ADD, not ADD-S, so no symmetry handling. For each body: accuracy `mean_{t,w}(ADD < τ)` at each τ, then `np.trapezoid` over `x = linspace(0,1,9)` (`:96-104`). | τ = `np.arange(0.01, 0.10, 0.01)` m, i.e. 0.01 to 0.09 m, strict `<` (`:25`) | mean over bodies (`:104`) | higher is better |
| **SP-SR** (`spider_sr`) | per body: frame-mean position error (m) and frame-mean geodesic angle (rad). **Averaged over bodies**, then success if `pos ≤ 0.1` **and** `ang ≤ 0.5` (`:166-178`) | 0.1 m, 0.5 rad, `≤` | fraction of worlds (0 or 1 with W = 1) | higher is better |
| **MP-SR** (`maniptrans_sr`) | per object id (the **slot**; `AUC.py:206`): frame-mean position `< 0.03` m **and** frame-mean angle `< 30°`; **every** object must pass (`:181-191`) | 0.03 m, 30°, strict `<` | fraction of worlds | higher is better |
| **RPE** (`rpe_cm`) | for each pair i<j: `‖R_ia^T (p_ja − p_ia) − R_ir^T (p_jr − p_ir)‖`, mean over frames, worlds and pairs, ×100 (`:194-224`). Only object j's **position** relative to object i's frame counts; j's orientation does not. | none | a single-object episode returns **0.0** (`:211-213`) | lower is better |
| **MPPE** (`mppe_cm`) | 6 keypoints `±0.05 m` along object axes: `‖(p_a + R_a k) − (p_r + R_r k)‖`, mean over T, W, B and the 6 points, ×100 (`:36-44,227-253`) | none | pooled mean | lower is better |

- The geodesic angle is `2·atan2(√(1−d²), d)` with `d = |⟨q_a, q_r⟩|` clipped to [0, 1]
  (`:73-77`), so the quaternion sign does not matter.
- **Across episodes:** an unweighted `np.mean` of per-episode values over the solution's
  episodes (`AUC.py:409-447`). Episodes count equally regardless of length. Inside an episode,
  frames count equally.
- **Single-object episodes** (35, 36, 37, 38):
  - RPE = 0, which pulls the 20-episode mean toward zero. The docstring acknowledges this
    (`object_pose.py:206-209`), and so does the README (`tracks/track_3/README.md:69`).
  - AUC, SP-SR, MP-SR and MPPE use the single body.
- **NaN/invalid values:**
  - A non-finite submission value is a hard error (`AUC.py:302-309`).
  - A quaternion norm below 1e-8 is a hard error (`:209-216,418`).
  - A non-finite final result raises `ValueError` (`:444-446`).
  - Unlike `episode_metric`'s docstring, the scorer has no per-episode "NaN, skip" path
    (`object_pose.py:296-298`; `tools/pack_submission.py:507-508`).
- **Missing or failed episodes:** the kit has no partial credit. Any missing roster row fails
  the whole submission (`AUC.py:399-405`). The packer refuses to build a file without all 20
  episodes (`tools/pack_submission.py:226-230`).
- **AUC point set:**
  - The public scorer's `_VERTICES_B64 = ""` (`AUC.py:239`). The production scorer takes the
    points either from an embedded blob or from a single `auc_vertex_samples_b64` payload in
    the solution (`:242-263,380-384`).
  - `v2dlb/vertices.py:35-60` defines the canonical set: 500 mesh **vertices** (not surface
    samples), chosen with `default_rng(0).choice(n, 500, replace=n<500)` from
    `<name>_visual.glb` or `<name>.glb`, stored float32 (`:65`). For Track 3 these would come
    from the confidential meshes (`config/tracks.json:205`).
  - The builder that produced the Track 3 blob (`metrics/build_kaggle_metrics.py`) is not in the
    kit. **Not determined by the kit.**
  - `vertices.py:10-15` says the evaluator's own sampler uses simulation-model shapes. The
    evaluator's JSON AUC can therefore differ slightly from Kaggle's.
- **What the public copies omit** (`metric_code/README.md:3-6`; `AUC.py:1`):
  - the solution: reference poses `ref_*`, `episode_index`/`world_index`/`frame_index`/
    `object_slot`/`object_name`, `Usage`;
  - the AUC vertex payload;
  - for CD-O, the reference meshes;
  - all comments and docstrings, replaced by blank padding (e.g. `AUC.py:16-18,24-29`).

  `_ACCEPTED` is present with empty lists (`AUC.py:224`). Whether production uses the same
  lists is **not determined by the kit**.

---

## 5. Mesh submission (`track_3_mesh`, CD-O)

- **Input layout.** `<meshes>/episode_{episode:06d}/<object_name>.{glb|obj|ply|stl|off}`. The
  first existing suffix wins, in that order (`tools/pack_reconstruction.py:29-37,58,72-76,198`).
  The object list comes from the **sample parquet's row ids**, not from the objects JSON
  (`:185-199`). From the objects JSON there are **36 (episode, object) pairs** (16×2 + 4×1)
  covering 11 distinct names. The same physical object is scored once per episode.
- **Row id.** `t{track}m_{episode:06d}_{object_name}_{array}_{point:06d}` (`v2dlb/object_mesh.py:44-52`),
  parsed with `^t\d+m_(\d{6})_(.+)_(\d)_(\d{6})$` (`:56`). `array` 0 is
  `object_mesh_vertices` and 1 is `object_mesh_faces` (`:38`). The greedy `(.+)` handles names
  containing underscores and digits, such as `wooden_piece_1`.
- **Budget:**
  - The packer reads the budget from the sample's row counts for the first (episode, object)
    (`tools/pack_reconstruction.py:79-89,194`). The config says 4096 faces and 4096 vertices
    (`config/tracks.json:280-281`); the parquet itself was not opened.
  - If every pair uses 4096 + 4096 rows, the sample has 36 × 8,192 = 294,912 rows. This is
    derived, not verified.
  - `budget_mesh` (`v2dlb/mesh_budget.py:15-40`) loads with
    `trimesh.load(force="mesh", process=False)`, welds with `process=True`, then decimates with
    `fast_simplification.simplify(target_count=faces)` only if faces exceed the budget. It
    raises `ValueError` if the result still exceeds either budget, then pads vertices with
    copies of vertex 0 and faces with `(0,0,0)` (zero-area faces, never sampled).
  - Face indices are stored as floats in x/y/z.
- **Units and scale.** No unit is required. The scorer rescales the candidate so its
  area-weighted RMS surface radius equals the reference's, about the candidate's area-weighted
  centroid (`object_mesh.py:170-204`). The result is reported "in centimetres at the
  REFERENCE's scale" (`:16`), i.e. `100 ×` reference units.
- **Registration and metric** (`object_mesh.py:59-216`; public `metric_code/track_3/CD-O.py:39-191`):
  1. Scale is **fixed analytically** by the RMS-radius ratio, not optimised. Only the
     **candidate** is transformed; the reference never moves. The README's "separate Sim(3)
     registration of each shape" (`tracks/track_3/README.md:71`) is loose wording for this.
  2. Surface sampling is area-weighted with square-root barycentric coordinates, with the
     **same seed on both sides**: 10,000 points for the metric (seed `s`) and 3,000 for
     alignment (seed `s+1`), where `s = 1,000,003·episode + 101·slot` (`:62-63,205-216`).
  3. Rotation and translation: up to 25 starts (identity plus proper PCA-axis mappings; no
     reflections), each running up to 40 ICP iterations with 90% trimmed Kabsch. The best start
     is chosen by symmetric mean distance on the alignment samples (`:86-167`).
  4. The score is `100 × 0.5 × (mean_c min_r ‖c−r‖ + mean_r min_c ‖r−c‖)`: symmetric,
     **non-squared** L2, on the 10k samples (`:131-136,210-211`). It uses `scipy.spatial.cKDTree`.
     `scipy` is not in `requirements.txt`, which only matters for running the scorer locally.
  5. The track score is the unweighted mean over all (episode, object) pairs (`CD-O.py:289-309`).
- **Parquet schema:**
  - Columns are exactly `["row_id","x","y","z","code_commit_url"]`, in order. The uploader
    compares the list exactly (`submit/v2d_submit.py:106-108`); the packer writes it
    (`tools/pack_reconstruction.py:103`).
  - The `object_mesh.py:21-23` docstring describes the **solution** layout (`episode_index`,
    `object_slot`, `object_name`, `array_name`, `point_index`). A submission must *not* carry
    those key columns: the scorer does `solution.merge(submission)` on **all** columns
    (`CD-O.py:270-272`), so colliding names would get `_x`/`_y` suffixes and break
    `groupby("episode_index")`.
- **Provenance.** Only `code_commit_url` (`config/tracks.json:291-293`; `CD-O.py:200-203`). It
  must be non-empty, uniform and match the regex (`CD-O.py:208-247`).
- **Rejections:**
  - **Packer:** the sample's row ids do not match the pattern (`pack_reconstruction.py:186-189`);
    a mesh file is missing (`:76`); the budget is exceeded (a `ValueError` traceback, from
    `mesh_budget.py:34-36`); any sample row is unfilled (`:96-98`); a non-finite value
    (`:99-100`); a bad commit (`:93-94`, whose message says "40-hex sha" although the regex
    allows 7-40).
  - **Uploader:** the file is not `.parquet`; the columns are wrong; `row_id` values or order
    differ from `data/track_3_mesh_sample_submission.parquet`; non-finite coordinates; the
    commit URL is not single-valued or does not match (`v2d_submit.py:99-122`).
  - **Scorer:** missing columns; duplicate `row_id`; a missing row or blank coordinate;
    non-numeric or non-finite values (`CD-O.py:260-285`); a `point_index` not running
    `0..n-1`; a face index that is not an integer (tolerance 1e-6) or falls outside `[0, V)`;
    zero surface area or zero extent (`CD-O.py:13-21,49-63,150-169`).

---

## 6. Provenance and rule-like constraints

**Commit URL.** The pattern is
`^https://[\w.-]*github[\w.-]*/[\w.-]+/[\w.-]+/commit/[0-9a-fA-F]{7,40}/?$`
(`v2dlb/report_schema.py:55`). The host only has to contain "github", so GitHub Enterprise
passes, and the SHA can be 7-40 hex characters.
- **Enforced:** the format, in the packer, uploader and scorer.
- **Not enforced:** that the commit exists, is reachable, or is in "a repo shared with the V2D
  account" (`tools/pack_submission.py:180-183`; `SKILL.md:71-72`). Nothing in the kit makes a
  network call to check.
- The requirement is attributed to "Section 2.9.a of the rules" (`report_schema.py:107-109`).
  Section 2.9.e requires winners to release the Submission and code under Apache-2.0
  (`config/tracks.json:334-337`). The rules text itself is not in the kit.

**Uniformity** (`report_schema.py:18-46`; scorer `AUC.py:317-348`):
- `eval_script_sha256`, `eval_code_sha256`, `packer_script_sha256`, `uploader_script_sha256`
  and `code_commit_url` must be identical on every row.
- `checkpoint_sha256` must be identical within each episode and may differ between episodes.
- None of them may be blank.

**Allowlists.** `accepted_eval_script_sha256` and `accepted_eval_code_sha256` are both `[]`
(`config/tracks.json:27-28`), and the scorer's `_ACCEPTED` is empty (`AUC.py:224`). "An empty list
disables that check" (`config/tracks.json:17-22`). The same comment says hashes may be appended
later, that this needs a new metric version, and that it "does NOT rescore existing
submissions". The packer never checks allowlists.

**Script hashes.** The packer hashes its own bytes and `submit/v2d_submit.py`
(`pack_submission.py:421-425`). Nothing in the kit compares these with `KIT_MANIFEST.json`
(`:12-13`). Whether the host does is **not determined by the kit**. A byte-level edit to either
file, including a line-ending conversion, changes the recorded hash. The files ship with LF
endings.

**Is the official FlashCHORD evaluator required?**
- **What the READMEs and docs state:**
  - "run the included `evaluate_policy.py` in a compatible FlashCHORD environment"
    (`tracks/track_3/README.md:12-15`);
  - "Run it with the public FlashCHORD runtime at this revision … commit aee95f25…"
    (`README.md:29-37`; `KIT_MANIFEST.json:3-9`);
  - "Run the organiser-provided evaluator" (`pack_submission.py:4`);
  - "Tracks 2 and 3 submit a policy rollout produced by the organiser's evaluator"
    (`report_schema.py:101-102`);
  - "`--results` is a directory of the organiser evaluator's output" (`SKILL.md:41`);
  - the scorer's allowlist message says "Re-run with the official evaluate_policy.py"
    (`AUC.py:369-370`).
- **What the code enforces:** nothing that distinguishes the official evaluator from another
  producer.
  - The reader accepts a legacy, hand-made layout (`submission_parquet.py:25-26,156-193`).
  - Hashes are opaque, non-empty strings.
  - `--eval-script` hashes whatever file is passed (`pack_submission.py:354-361`).
  - For Track 3 the evaluator's recorded reference is not compared with anything
    (`:321`).
  - The allowlists are empty.
  - The only evaluator-specific checks are schema shape, `environment_valid`, and name/slot
    order.
- The provenance columns exist for later verification: "verification can still ask for the
  weights behind any given episode" (`pack_submission.py:367-371`). Using a non-official
  simulator is a rules question for the organisers, not a question the code answers. Whether the
  rules require the official evaluator is **not determined by the kit**.

---

## 7. Preflight and uploader (read, not run)

**Environment:**
- Python ≥ 3.10 (`requirements.txt:1`; `tools/preflight.py:57-62`).
- Packages: `numpy>=1.24`, `pandas>=2.0`, `pyarrow>=14.0`, `kaggle>=2.0`, `kagglesdk`,
  `trimesh>=4.0`, `fast_simplification>=0.1` (`requirements.txt:2-8`).
- The preflight requires numpy, pandas, pyarrow and kaggle (major version ≥ 2). For
  `object_mesh`/`mhr` it also requires trimesh and fast_simplification; otherwise trimesh is
  optional (`preflight.py:28-34,65-86`). It does **not** check `kagglesdk`.
- Token sources:
  - `KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`, are accepted.
  - A legacy `~/.kaggle/kaggle.json` alone is a FAIL (`preflight.py:97-111`).
  - The uploader also mentions `kaggle auth login` (`v2d_submit.py:29-30,62-79`).

**Preflight checks, in order** (`preflight.py:216-251`):
1. Python version and packages.
2. The container check, which is informational only (`:89-94`). Its precedence parses as
   `(dockerenv or cgroup) if /proc/1/cgroup exists else False`, so it always reports "host" on
   Windows.
3. Inputs. For `track_3`: the sample CSV and objects JSON exist, and there is one top-level
   `episode_*.parquet` per scored episode, **by filename only**. Parquets are not opened, and
   roster, frame and provenance are not checked (`:164-213`). For `track_3_mesh`: only that the
   sample parquet exists (`:154-163`).
4. The token.
5. Only if nothing is blocking so far: Kaggle authentication, then `get_competition` per
   competition slug, reading `user_has_entered` (`:114-147`).

`--skip-kaggle` skips steps 4-5 (`:223-235`). "Ready to submit" does not mean the packer will
succeed.

**Uploader flow** (`submit/v2d_submit.py:426-524`):
1. Local `validate_submission` (section 2.5 and section 5).
2. `build_api()`.
3. `preflight()`. For each competition it calls `get_competition` (entered?),
   `competition_submissions(page_size=1)` (team/user identity) and
   `competition_get_submission_limits` (`num_allowed_now`). It stops before any upload if
   rules are not accepted, a slug is missing, another error occurs, or identities differ across
   competitions (`:273-372`). An exhausted quota is only a warning (`:369-371`).
4. Sequential `competition_submit(path, message, slug)` per competition. Only rate limiting is
   retried: 5 attempts with 5 s, 10 s, 20 s, 40 s backoff (`:56-57,375-389`).
5. On partial failure it prints `--only <metrics>` to retry just the failed legs
   (`:510-518`). `--only` accepts metric keys such as `add_auc` or `rpe_cm` (`:450-454`).
6. `--wait` polls every 20 s for up to 1,800 s (`:58-59,392-423`).

`--dry-run` still authenticates and makes the read-only API calls, but does not submit
(`:496-498`). `--check-entry` only reports entry status (`:456-474`).

**Quota:**
- One pose CSV goes to **5** competitions, so it consumes one submission on each. The mesh
  file goes to 1.
- "A rejected submission still costs quota" (`SKILL.md:16`; `v2d_submit.py:14-16,96-97`).
- The rules say **5 per week**, with unlimited submissions in the final 3 days
  (`config/tracks.json:304,333`). The Kaggle configuration is `max_daily_submissions: 5`
  (`:325`). The uploader's "left today" figure therefore reflects the daily Kaggle mechanism,
  not the weekly rule. The config notes that the website policy prevails (`:313-316`).

**Outward-facing behaviour:**
- All Kaggle calls above, and `competition_submit` uploads the file.
- Importing `kaggle` in `check_packages` (`preflight.py:72`) may authenticate as a side effect
  (`v2d_submit.py:65-66`), even with `--skip-kaggle`. Whether that makes a network call is
  **not determined by the kit**.
- The evaluator's W&B publish happens only when `logging.mode != disabled`
  (`evaluate_policy.py:26-44,71-80`). The root config sets it to `disabled`
  (`evaluate_policy.yaml:8-9`), and the example runner forces `logging.mode=disabled`
  (`run_policy_evaluation.py:79`).

**Agent-addressed instructions.** `skills/v2d-submit/SKILL.md` is a skill file addressed to an
AI agent. It tells the agent to:
- run the preflight first, "every time", and not proceed while it reports a blocking problem
  (`:16-28`);
- ask the user to accept competition rules manually and "not try to automate it" (`:32-34`);
- not hand-build the CSV, not cross-submit tiers, and not resubmit whole files (`:167-174`).

These are described here only. None were acted on.

---

## 8. Surprises and risks for a participant

1. **The blank-provenance error path crashes.** In `pack_submission.py:436-439`, `int(row.split("_")[1])`
   on `track_3/e000001/...` evaluates `int("3/e000001/…")` and raises `ValueError`. So a
   missing checkpoint hash produces a traceback instead of the intended message. No file is
   written, so the outcome is still a rejection.
2. **Wrong hint** for a missing `eval_code_sha256` (`pack_submission.py:381-387`). Only the
   parquet metadata can supply that hash.
3. **Position units are never checked**, despite the docstring claim (`submission_parquet.py:17-18`
   vs `:126-134`). Scale is not removed by the alignment, so a unit mistake (e.g. cm) is scored
   in full.
4. **Initial-pose alignment is anchored on the achieved object-0 pose at frame 0.** It removes
   only a world-side rigid offset. Body-frame (mesh-origin or axes) mismatches and frame-0
   orientation errors of object 0 propagate. The reference object-frame convention is not
   documented (section 3).
5. **Silent truncation.** Extra frames, worlds and bodies are dropped without a message. Only
   *missing* rows fail (`pack_submission.py:277-297`).
6. **Silent slot swap** when the parquet lacks `object_body_name`: no name check happens
   (`:299`). A multi-body object also breaks the `object_body_id` → slot assumption.
7. **Track 3 has no local score.** The metric cross-check and the reference-frame check are
   Track 2 only (`:321,447-451`). The evaluator's JSON metrics cannot be compared locally, and
   its AUC point set may differ from Kaggle's (`vertices.py:10-15`).
8. **Recursive discovery in the packer vs top-level discovery in the preflight**
   (`pack_submission.py:57,65`; `preflight.py:197`). A stray or backup parquet anywhere under
   `results/` aborts packing, while the preflight can still pass.
9. **One invalid world anywhere in the parquet rejects the episode**, even though only `w000`
   is submitted (`pack_submission.py:266-273`). The Hydra default `world_count: 4096`
   (`flash_sac.yaml:6`) makes this more likely if `evaluate_policy.py` is run directly.
10. **The uploader enforces row order; the scorer does not.** The uploader requires the exact
    sample order (`v2d_submit.py:155`). The scorer ignores extra rows (`AUC.py:396-398`).
11. **SP-SR averages errors over both objects** (`object_pose.py:176`). The README table
    describes it as "mean position error", which hides that one bad object can pass if the
    other is good (`tracks/track_3/README.md:67`). MP-SR uses "every object"
    (`object_pose.py:186-190`). With W = 1, both SR metrics are 0/1 per episode.
12. **RPE counts single-object episodes as 0.** 4 of 20 Track 3 episodes do this, so the RPE
    leaderboard is 0.8 × the two-object mean (`object_pose.py:206-213`).
13. **Quota:** the weekly rule (5/week) differs from the Kaggle daily configuration (5/day)
    (`config/tracks.json:304,325,333`). A missing `kagglesdk` makes `entry_status` return
    `None`, which the preflight reports as "registered on all" (`preflight.py:134-147`;
    `v2d_submit.py:249-260`).
14. **Misleading preflight message.** If `import kaggle` raises during its import-time
    authentication (e.g. no token), the preflight reports "kaggle missing"
    (`preflight.py:79-80`).
15. **Host-only and build-tree paths:**
    - `mirror: /data/v2d_challenge_data_public` (`config/tracks.json:26`);
    - `/data/v2d_track1_gt/...` (`:111-112`);
    - `build/kaggle/...` fallbacks (`pack_submission.py:199-205`; `preflight.py:157-170`;
      `v2d_submit.py:109-111,149-151`);
    - references to `metrics/build_kaggle_metrics.py` and `tests/*.py`, which are not shipped
      (`object_pose.py:3-4,14-15,146`; `AUC.py:292`).

    The `pack_submission.py` docstring passes `--dataset-root` for Track 3 (`:7-9`), but
    Track 3 never uses it (`:197-207`).
16. **Decision markers:**
    - "NEEDS DECISION" rendering for null rule fields (`config/tracks.json:308-311`), with
      `prize_breakdown: null` (`:328`) and `total_prizes: "To be announced (TBA)."` (`:327`);
    - the MPPE radius "confirmed by Shalin on 2026-09-18 … do not change it without a new
      decision" (`object_pose.py:30-36`);
    - the evaluator AUC point-set alignment "Until that lands" (`vertices.py:13-15`);
    - allowlists that may be appended later without rescoring (`config/tracks.json:17-22`).
17. **README wording:**
    - `tracks/track_3/README.md:44-45` has a garbled sentence ("This file has a commit URL;
      policy and evaluator hashes identify the pose rollout…");
    - the "separate Sim(3) registration of each shape" wording (`:71`) differs from the code,
      which moves only the candidate and fixes scale analytically.
18. **Commit-error text** in `pack_reconstruction.py:94` asks for a 40-hex SHA; the regex
    accepts 7-40.
19. **The sample CSV passes every format check.** Its placeholder commit URL matches the regex
    and its hashes are non-empty. Format validation therefore says nothing about whether a
    submission is real.

## Open questions (not determined by the kit)

- The body-frame convention (origin and axes) of Track 3 reference object poses, and how a
  participant is expected to match it from reconstructed meshes.
- Track 3 reference frame rate, and how roster frame 0..N−1 maps to video frames.
- The production values of `_ACCEPTED` and of the AUC vertex payload; whether the host checks
  the packer and uploader hashes against the manifest.
- Whether Kaggle rejects extra rows before the metric runs.
- The full contents of the evaluator metrics JSON (`write_evaluation_report` is not in the kit).
- Whether the rules require the official FlashCHORD evaluator, as opposed to any producer of a
  compatible parquet.
- The exact face/vertex budget and row count of `track_3_mesh_sample_submission.parquet`
  (config says 4096/4096; not opened).
