# Pose support diagnostic protocol

Fixed on 2026-11-03, before this diagnostic loads arrays or computes a value.
The lead commits this file before implementation or measurement.

## Question and prior knowledge

How much of the MoGe-2 versus direct-stereo offset difference comes from depth values,
pixel support, and the default-to-observed slot assignment in episode 12?
The background is IDEAS 44-45 and `docs/chain/pose-length-diagnostic-protocol.md`.
The already-read `outputs/pose_insurance/length/summary.md` reports clean-five horizontal
length MAE against mocap origins of 0.581 cm for stereo and 11.754 cm for S1.
Those results motivated this protocol. They are not new or independent evidence.
This is descriptive only. No candidate, scene change, policy run or pack follows from it.
Stereo is a calibrated estimate, not measured ground-truth depth.

## Decisions frozen before data

- Read frame zero only of train 0/11/12/18/21/23/26/31 and spent validation 22/24/27.
- The primary measurable set is exactly train 11/12/18/21/23, with equal episode weight.
- Keep 0/26/31 as missing merged/default cases. Do not split, impute or replace them.
- Report validation separately. It cannot confirm a fresh hypothesis independently.
- Do not use controls 39/42: single-object controls add no offset information here.
- Never read episode 41, 43 or 44, or any evaluation episode's data.
- Freeze the recorded observed assignments. Do not relabel from depth, mocap or scans.
- Use the recorded measurement plane for every primary projection in an episode.
- Freeze the support, estimators, fits, folds and reading thresholds below.
- Mocap and scans enter reference/error columns only, after video estimators are fixed.
- All labels below describe this sample. None is a model-selection or adoption gate.

## Frozen input chain

All paths are relative to the repository. No newest-run search is allowed.
Train assignments come from `track3/features/pose_insurance/placements_train.json`
(`episodes[ep].slots`, including `observed_cluster_ids`) and
`track3/features/mesh_extents/assignments.json` (`public:<ep>.objects`).
Use the existing length diagnostic's object order and observed-mask resolution.
Validation assignments and object order come from
`track3/features/pose_insurance/length_confirm_assignments.json`, `episodes[ep]`.
Its preparation pointers are in `track3/features/pose_insurance/length_confirm_preparation.json`.

The following run directory IDs fix every episode's inputs.
For episode e formatted with six digits, M is under `outputs/mesh_extents/public/episode_e/`,
D under `outputs/depth/public/episode_e/`, and S/R under `outputs/stereo_depth/public/episode_e/`.
Append the table's ID and `/run.json` to obtain each producer record.

| Episode | M: measurement | D: MoGe | S: stereo generation | R: rectification |
|---:|---|---|---|---|
| 0 | 20261101T150512.015735Z-5ff23e7a | 20261029T020915.737479Z | 20261102T000929.987328Z-de029371 | 20261102T000929.127060Z-a4bb077e |
| 11 | 20261101T103701.928174Z-f63cc238 | 20261029T020928.345436Z | 20261102T000952.439543Z-5de74a8d | 20261102T000952.260922Z-80cb523f |
| 12 | 20261101T103703.949004Z-546c56c2 | 20261029T020937.002419Z | 20261102T001021.465628Z-6bed174e | 20261102T001020.917490Z-55d2e763 |
| 18 | 20261101T103706.195611Z-8e4c4ca5 | 20261029T020946.139727Z | 20261102T001045.006993Z-0015ba94 | 20261102T001044.790223Z-0828b999 |
| 21 | 20261101T103707.151959Z-05bf6bac | 20261029T020955.740261Z | 20261102T001102.465540Z-4ee2053e | 20261102T001102.285015Z-3be57f78 |
| 23 | 20261101T103708.924989Z-52ed3be1 | 20261029T021005.260544Z | 20261102T001116.902086Z-a3ddeae5 | 20261102T001116.708104Z-59fba1fd |
| 26 | 20261101T150527.858368Z-51cea8f6 | 20261029T021014.452728Z | 20261102T001134.651961Z-375d9d68 | 20261102T001134.462780Z-c30d1a8f |
| 31 | 20261101T150530.131074Z-2d008e1e | 20261029T021023.736359Z | 20261102T001149.510307Z-0ca4bcb5 | 20261102T001149.323926Z-31202910 |
| 22 | 20261102T163445.927917Z-d36d08b3 | 20261102T163436.427079Z | 20261102T163608.919918Z-d1cc13ff | 20261102T163608.743794Z-8d5fba30 |
| 24 | 20261102T163458.120337Z-abdb6d9b | 20261102T163447.053558Z | 20261102T163626.034336Z-f27229a4 | 20261102T163625.852773Z-7faa9d31 |
| 27 | 20261101T225816.131017Z-ebd30c8f | 20261101T225805.331133Z | 20261102T163653.712522Z-14e03ebc | 20261102T163653.532269Z-37f07dee |

- M: `labels_000000.npz`, key `label`; zero is background, cluster IDs add one.
  Plane normal n and offset b are `frames[0].plane.normal` and `.offset_m` in M's record.
- D: completed frame-zero entry's `artifact` and `sha256`, normally `frames/frame_000000.npz`.
  Read `depth_m`, `valid`, `model_intrinsics_pixels`, `input_intrinsics_pixels`.
  Read sibling `preprocessing.npz`: `input_intrinsics_pixels` and `input_valid`.
- S: completed frame-zero entry's `artifact` and `sha256`, normally `frames/frame_000000.npz`.
  Read `rgb_depth_m`, `rgb_valid`, `rgb_intrinsics_pixels`, `frame_index` and `pts`.
  These depths already use camera A's grid and optical-z frame. Do not transform them again.
- R: `rectification.npz`, including `R1`, `R2`, `P1`, `P2`, `Q`, `A_T_C`.
  Calibration policy is `configs/calibration/ego.json`. The source file is
  `datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public/meta/camera_calibration.json`.
- Train S1 scale: `outputs/pose_insurance/scale/stereo_<six-digit-episode>.json`, fields `s`, `valid`, `anchor_runs`, `depth_run`.
  Require its frame-zero anchor to point to S and D above; S must point to R above.
  Validation uses the preparation file's `scale` run and its `scale.json`, with the same chain.
  Call validation's value the recorded episode ratio, not an actual S1 scene scale.
- Recorded S1 scenes: `outputs/pose_insurance/scale/summary.json` selects the S1 run per train episode;
  its `metrics.json.scene` supplies positions, quaternions, `before` positions and shifts.
  Resolve only those train rows. Do not load any evaluation record.

Admit every producer with `read_run` and the existing producer/artifact admission helpers.
Require completed status, matching public episode and operation, and re-hash every artifact used.
Hash the protocol, request, assignment files, scale records and input run records too.
Require equal grid shapes and input/stereo K agreement with absolute tolerance 1e-5, zero relative tolerance.
Record model-K versus input-K differences; retain input K for all estimators regardless.
Require matching preprocessing hash and frame-zero identity/PTS as in the length diagnostic.
An admission failure stops the run; it cannot silently substitute a producer or shrink the roster.

## Supports and estimators

Let M_j be the frozen observed object mask for slot j. Let K be D's preprocessing input K.
V_M is D.valid intersect input_valid and finite positive depth_m.
V_S is S.rgb_valid intersect input_valid and finite positive rgb_depth_m.
Set I_j = M_j intersect V_M intersect V_S, F_Mj = M_j intersect V_M, F_Sj = M_j intersect V_S.
Require at least 300 pixels per object on each support used for an offset.
Insufficient support makes that estimator missing, with its count and reason retained.
Report mask counts, all support counts, support/mask fractions and I/source-full fractions.
Masks originate in the MoGe measurement pipeline; this conditions every interpretation.

Unproject optical depth with P(u,v) = z(u,v) K^-1 [u,v,1]^T. Use unweighted point means.
Each offset d is slot-1 centroid minus slot-0 centroid. Preserve camera-space 3D vectors.
On I_j compute stereo, MoGe native, MoGe times recorded episode s, and MoGe times r_I.
Native depth_m already includes MoGe's metric scale. Never apply that scale a second time.
The factors s and r_I are explicitly labelled stereo calibrations of native depth.
Define r_I as the median of the two per-object medians of z_S/z_M on I_j.
This gives each object equal weight and uses no reference labels. Report both object medians.
Missing/invalid s makes only the recorded-scale estimator missing. No substitution is allowed.
Keep r_I fixed when restoring full supports; never re-estimate it for a support comparison.
On F_Mj compute the three MoGe estimators; on F_Sj compute stereo.
If I_j equals F_Sj, identical-support and full-support stereo are explicitly duplicates.

Use B = table_basis(n) from the existing scene convention, with n's recorded sign unchanged.
Report B d in 3D and its first two coordinates as the primary horizontal vector.
No plane is refitted for a source, scale, support or assignment variant.
Let L be the norm of that horizontal vector. Keep signed gaps and absolute gaps in cm.
For native depth, write g_F = L_MF - L_SF and g_I = L_MI - L_SI.
Report the exact scalar decomposition g_F = g_I + (L_MF-L_MI) - (L_SF-L_SI).
Also report its exact vector counterpart d_MF-d_SF = (d_MI-d_SI)+(d_MF-d_MI)-(d_SF-d_SI).
Repeat the decomposition for each scaled MoGe estimator. Do not add absolute errors as contributions.

## Assignment, rays, projection and references

Episode 12 has a recorded default slot 0 and an observed cluster 1 for that slot.
Its assignment contrast holds native MoGe depth, full support, s, B and slot 1 fixed.
Replace only the pre-separation default slot-0 xy by its observed MoGe centroid xy.
Keep heights, rotations and recorded separation shifts fixed. Report the resulting offset change.
Reproduce the final recorded S1 xy offset within 1e-6 m before interpreting the contrast.
Report any separation term explicitly. Do not call observed MoGe times s the old default S1.
The observed estimator, common-support depth contrast and assignment contrast remain separate rows.

For object j on I_j, set r_j to the mean K^-1[u,v,1]^T, normalized to unit length.
For delta_j = centroid_Mj - centroid_Sj, report r_j(r_j dot delta_j) and its orthogonal remainder.
Subtract slot-0 components from slot-1 components to reconstruct the offset difference.
Do this for native and both scaled MoGe estimators, retaining signed vectors and norms.
Every identical-pixel point difference equals (z_M-z_S)K^-1[u,v,1]^T and lies along its own ray.
Cross-ray centroid shifts relative to r_j reflect spatially varying depth differences across rays.
They do not identify an independent lateral point-map warp or camera-to-mocap heading error.

For projection sensitivity only, rotate n by +5/-5 degrees about each of primary B's two in-plane axes.
Reproject the same fixed 3D offsets with these four normals. Do not change support or refit depth models.
Report the maximum change in g_F and g_I in cm, separately from primary results.

Reference inputs use the same dataset revision, public `data/chunk-000/episode_<six-digit-episode>.parquet`,
and `mesh/<object>/<object>.glb` (or the existing `<object>_visual.glb` fallback).
Use the length diagnostic's nearest-visible-filled frame-zero poses and exact scan vertex-AABB midpoints.
Never use trimesh.split(), component mesh copies, or sampled bounds. Hash these reference files.
Report origin and transformed-centre reference 3D vectors, world-horizontal lengths and their difference.
For every estimator report 3D length error and signed/absolute horizontal length errors against both references.
For train vector-error columns only, transport references by the frozen recorded S1 slot-0 full rotation,
exactly as _length.ray_errors does, then compare in B's table coordinates; retain all three coordinates.
Also report transported reference horizontal lengths beside world-horizontal lengths.
This alignment is conditional on the guessed body frame, not a measured camera-to-mocap transform.
Validation has no recorded S1 anchor: vector-error columns are missing; invariant 3D/length columns remain.
Mocap and scan geometry never set masks, scales, planes, fits, assignments or descriptive readings.

## Table depth models and spatial withholding

Select table pixels outside all nonzero labels and a five-pixel Chebyshev dilation of those labels.
Intersect V_M and V_S with abs(n dot P_M + b) <= M.request.table_band_m.
This is a frozen MoGe-conditioned table band, not an independently annotated table or ground truth.
Do not change it after viewing ratios. Save its pixel count and footprint in the figure.
Partition the image into 8 by 8 cells using row=floor(8v/H), col=floor(8u/W).
Remove pixels within two pixels of every cell edge. Set fold=(row+2col) modulo 5.
Hold out each fold once and fit on the other four; no random pixel split or fold tuning.
Require each fold to have >=1,000 training pixels and >=200 test pixels in >=4 test cells.
Otherwise mark the episode's table comparison missing without relaxing the requirements.

Fit by NumPy least squares, on training pixels only:

1. Constant scale: zhat_S = a z_M.
2. Affine depth: zhat_S = a z_M + b_depth.
3. Affine inverse depth: 1/zhat_S = a/z_M + b_inverse.

Nonpositive or nonfinite test predictions make that model's fold error infinite, never dropped or clipped.
Report held-out depth MAE in cm and absolute-relative error, averaging pixels within cells,
then cells within folds, folds within episodes, and episodes equally. Retain coefficients and all fold errors.
Plot the uncorrected z_S/z_M map and held-out residual maps on the same fixed table pixels.
For plane context, fit 1/z_M and 1/z_S separately to [1,u/W,v/H] with the same folds.
Report their held-out relative depth residuals. A plane is affine in inverse depth over image coordinates.
An affine ratio or a winning correction model alone does not establish a physical planar warp.

## Pre-declared descriptive readings

The following labels require the stated count out of all five clean train episodes.
Missing values do not reduce five. A native abs(g_F) below 1 cm is too small for attribution labels.

- **Depth dominates:** g_I has g_F's sign and abs(g_F-g_I) <= 25% of abs(g_F), on >=4/5.
  Thus identical-support depth retains at least 75% of the signed whole-support length gap.
- **Support dominates:** g_F-g_I has g_F's sign and abs(g_I) <= 25% of abs(g_F), on >=4/5.
- Otherwise report **mixed or unresolved**. Flag episodes where depth and net support contributions oppose.
  Ratios above 100% indicate cancellation, not additional independent error explained.
- **Scale largely removes the common-support gap:** for s and r_I separately, abs(g_I_native) >=1 cm,
  abs(g_I_scaled) <=50% of abs(g_I_native), and abs(g_I_scaled) <=1 cm, on >=4/5.
- **Material assignment effect:** episode 12's frozen assignment contrast changes horizontal length by >=1 cm.
  Report its signed magnitude and ratio to the recorded-S1 versus observed-stereo gap; do not generalize n=1.
  If that denominator's magnitude is below 1 cm, report the ratio as undefined.
- **Nonconstant table correction supported:** affine depth or affine inverse depth, each considered separately,
  lowers held-out MAE versus constant scale by >=20% and >=0.5 cm on >=4/5 clean episodes.
  An infinite model error cannot qualify. This labels predictive discrepancy, not causal MoGe error.
- **Projection sensitive:** one of the four fixed normal perturbations changes g_F or g_I by >=1 cm on >=2/5.

Report validation episode readings individually with denominator three, without a confirmation verdict.
Report every continuous quantity even when no label qualifies. Do not change thresholds after seeing values.

## Implementation and outputs

Use NEW files only in `track3/features/pose_insurance/`: `_support.py`, `support_diagnostic.py`,
and `support_request.json`. The library uses NumPy only and has no I/O, paths, gates or run records.
The handler owns admission, reference I/O, reporting and a resumable episode loop with a ledger.
The request fixes the eleven-episode roster and a 600-second CPU budget; resume the same request on timeout.
Expose a module entry point in the new handler; do not edit the existing dispatcher or any existing .py.
The lead adds the NEW `support_diagnostic_authorized` flag to `runtime.json` in a later step.
`track3/features/pose_insurance/runtime.json` is absent from
`submissions/20261103T0645Z-d2-v4/track3_poses_v4_final.manifest.json`, checked before writing this protocol.
All existing pose_insurance .py files remain unchanged because the D2 v4 manifest pins them.
Use CPU only and existing dependencies. Reuse existing readers without invoking writing diagnostics.
Write under `outputs/pose_insurance/support-diagnostic/`, with run.json, summary.md, summary.json,
one labelled figure `support.png`, and per-episode records plus append-only ledger.jsonl.
Resume by input/protocol/request/source identity. Preserve earlier records; never overwrite a different identity.
Summaries show all eleven episodes, the fixed clean-five aggregate, and spent validation separately.
The figure shows attribution, support coverage and table ratio/residual panels with units and missingness.
Record reproduction command, hashes, failures, eligibility counts and limitations in the report.
No GPU run, package change, upload or model choice is authorized by this protocol.

## Amendments

### Amendment 1 (2026-11-03 07:24 UTC, before any array was loaded or offset computed)

The S1 selector named above is wrong: `outputs/pose_insurance/scale/summary.json` stores scenes under
`episodes.S1` and holds no producer run paths. Select recorded S1 producer runs from
`outputs/pose_insurance/scale/finished.json`, using `selected` entries with candidate `S1` and an episode in the
frozen train roster. Admit each completed producer and re-hash its `metrics.json`. Require its scene to equal
`outputs/pose_insurance/scale/summary.json`, `episodes.S1[episode].scene`. Hash both selector files. Do not resolve
producers outside the frozen roster. Nothing else changes.
