# Track 3 public diagnostic protocol v1

Frozen 2026-09-24 against dataset revision `5f68335f3acc802033d1e80728c1633197521de8`. **Official scoring, success thresholds and evaluator equivalence are unverified.** This is an engineering protocol, not a challenge scoring implementation. No reconstruction scores, public trajectory evaluation or evaluation-label access were used to prepare it. The accompanying code supplies only tested pose/coverage primitives; aggregation, alignment orchestration and scene diagnostics remain to be implemented.

## Split and evidence

The authoritative configuration is `configs/splits/public-v1.json`, split ID `track3-public-dev13-val7-v1`. It pins both metadata SHA-256 values. Local inspection found the same 20 IDs in public episode metadata, `info.json`'s explicit `episode_indices`, 20 parquet filenames and each of the three 20-clip camera directories. Agent 03 independently confirmed the file inventory. No episode is inferred from the stale `train: 0:46` range.

| Task | Development episode IDs | Validation episode IDs |
|---|---|---|
| Pot_with_lid | 0 | 2 |
| Cup_Stack | 11 | — |
| Cup_In_Dish_Rack | 12 | 13 |
| Dust_Brush | 18 | — |
| Water_Can_Blue_Cup | 21 | 22 |
| Planter_Stand | 23 | 24 |
| Pan_In_Dishrack | 26 | 27 |
| Brush_Dishrack | 31 | — |
| Dustpan_Solo | 39 | 40 |
| Pan_Solo | 41, 42, 43, 44 | 45 |

To reproduce the allocation, group the released IDs by metadata task and reserve the largest numeric ID from each group with at least two members. All others are development. The rule uses no random generator, video inspection, pose quality or model performance. It gives 13 development and 7 validation episodes across seven multi-example tasks. It is frozen but provisional: a later official contract may require a separately versioned replacement, preserving this version and its history.

Split keys are `(dataset_revision, source_split, episode_index)`. All camera streams (`ego_cam_a/b/c`), frames, tracked objects, crops, reconstructions and augmented/derived artifacts of an episode inherit its assignment. Shared object assets and rig calibration are common dataset metadata, with their use at official evaluation still subject to the contract. `evaluation` is a different namespace; an overlapping integer episode ID cannot turn an evaluation episode into development data.

Development supports debugging, fitting and parameter selection. Validation supports prespecified checks after configurations are frozen; do not use its trajectories, hand labels, fitted alignment or diagnostic reports as training targets, inference inputs or adaptive per-episode tuning. If repeated selection is made from validation results, record it and describe that set as used for model selection; do not claim it remains an untouched estimate. Public ground-truth control experiments must carry a distinct `public_label_diagnostic` provenance and cannot enter the video-only evaluation branch. Evaluation has no distributed pose labels and receives no fabricated proxy score.

`Cup_Stack`, `Dust_Brush` and `Brush_Dishrack` are singleton tasks and have no within-task validation. The split is not participant/session independent: those grouping metadata are unavailable. Largest-ID selection may have capture-order bias, and just one held-out example per covered task does not support broad generalization claims. Development has four `Pan_Solo` episodes, so pooled results alone would overweight that task.

## Coordinate and row contract

The pinned [Track 3 README](../datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/README.md) declares pose rows `[x,y,z,qw,qx,qy,qz]`, metres, scalar-first quaternions and `world_T_object` in the scanned mesh's frame. The transform maps object-frame column vectors into world coordinates: `p_world = R_world_object p_object + t_world_object`. Composition is `world_T_object = world_T_camera * camera_T_object`; the inverse has the opposite direction. Do not substitute a mocap rigid-body frame or an object's geometric centroid for the mesh origin.

Each episode has its own horizontal origin: x/y are centred on visible object positions, z is not centred, and the centring offset is not distributed. There is no shared cross-episode translation gauge or camera-to-mocap-world extrinsic. A world-coordinate estimate therefore needs an explicit compatible frame or a clearly labelled diagnostic alignment before absolute pose errors are meaningful. Image-coordinate comparisons require the separately validated camera transforms, orientation and distortion model; pose metrics do not prove those are correct.

Rows are joined by `(dataset_revision, source_split, episode_index, frame_index, object_name)` with an exact identity match; no nearest-object matching. Public object names identify mesh folders. Each eligible frame-object is counted once even when three cameras contributed. Preserve capture timestamp and frame index independently, verify pairing through the data/calibration audit, reject duplicate row keys and report missing/extra rows. Equal clip counts alone do not establish synchrony. Do not use global parquet `index` as an episode-local frame index.

## Validity, normalization and coverage

Apply the public `visible` flag before pose conversion. Invisible public poses contain seven zeros; a zero quaternion is invalid, never an identity or a training target. A visible object at position `(0,0,0)` with a valid quaternion is valid. Mocap `visible` is only the dataset's tracking-validity signal; it does not establish visibility in a particular ego image, contact state or occlusion category.

For both labels and predictions, require seven finite numeric components and a quaternion norm within `1 +/- 0.001`, then normalize it to unit length. This numerical tolerance is an engineering integrity threshold, not a success threshold. Count normalization corrections and retain the original norm in future audit outputs. A flagged-valid zero/nonfinite/grossly nonunit quaternion is malformed; never silently repair it. Quaternions `q` and `-q` represent the same rotation. The primitives reject boolean/string components and malformed flagged-valid rows; `Pose.from_row(..., valid=False)` filters before conversion.

Report these counts separately per episode/object/task and pooled: expected frame-object rows, visible source labels, invisible source labels, malformed visible labels, eligible valid labels, valid matched predictions, missing predictions, malformed predictions, explicitly invalid predictions, and unsupported extra predictions. A malformed visible label is a data-quality failure to investigate; exclude it from valid-label error support only with that count prominently retained. Missing/invalid/malformed predictions remain in the eligible-label coverage denominator. No label at an occluded row means the pose error is unobservable, not zero.

Primary coverage is `matched_valid_predictions / eligible_valid_labels`. An empty denominator produces null/undefined and support zero. With eligible labels but no valid predictions, coverage is zero and pose-error summaries are null. Error summaries use valid matched rows and always accompany coverage and counts. Confidence thresholds must be configured on development and frozen before validation; report any threshold and its effect on coverage. Observed estimates and interpolated estimates retain separate flags, counts and errors. A future interpolation configuration must fix a maximum time gap and preserve raw missingness; this protocol does not authorize filling labels or count interpolation as direct observation.

## Pose errors and one episode alignment

Translation error is `||t_prediction - t_reference||_2` in metres. Rotation error is the SO(3) geodesic angle `2 atan2(||xyz(q_reference^-1 q_prediction)||, |w(q_reference^-1 q_prediction)|)` in radians internally; reports may include degrees with the conversion explicitly named. This handles quaternion sign equivalence and remains accurate near zero. The base rotation metric has no symmetry reduction.

Report raw absolute errors only when provenance establishes a compatible world gauge; otherwise mark them `not_comparable_gauge`. The provisional labelled-alignment diagnostic uses exactly one rigid `SE(3)` transform per episode, shared by every object, camera and frame. No scale fitting, reflection, per-object alignment, or per-frame label-fitted transform is allowed.

For the fixed `first_valid_pose_anchor` mode, choose the earliest frame with an eligible reference and valid prediction; break ties by lexicographic object name. Set `A = reference_T_object(anchor) * inverse(prediction_T_object(anchor))`, then apply `A` on the left to every prediction in the episode. Record the anchor key, both source poses, `A`, units and provenance. With no anchor, aligned diagnostics are unavailable and coverage still reports failures. The anchor is a label-assisted diagnostic, never an inference transform. It makes the anchor error zero by construction, removes its constant orientation discrepancy, and is sensitive to that anchor's noise. Report metrics including and excluding the anchor, and explicitly disclose that limitation. More robust fit modes require a new named protocol and declared fitting support; they must not silently replace this frozen rule. No alignment fitting is implemented in the first batch.

## Symmetries

No object symmetry group has been verified in this batch. For all ten public object names (`beige_cup`, `blue_cup`, `mini_dust_pan`, `mini_sweeper`, `plastic_dish_rack`, `water_pitcher`, `white_pot`, `white_pot_lid`, `wooden_piece_1`, `wooden_piece_2`), symmetry status is **unknown**, not verified asymmetric and not verified symmetric. Shape resemblance, task names and apparent cylindrical silhouettes do not establish texture/contact-aware equivalence.

Report the unsymmetrized mesh-frame errors as the default. A future symmetry-aware metric is a separate supplementary column: require object/asset hash, verified group elements in the mesh frame, geometric/texture/contact rationale and versioned reviewer evidence. For a verified rotational symmetry `S`, it compares against `reference_T_object * S` and minimizes over the declared group. Unknown symmetry is never silently assigned continuous axial symmetry or automatically discounted. Continuous symmetries need a specified analytic quotient/approximation and uncertainty report; they are unsupported here.

## Relative motion, drift and aggregation

Alongside absolute/aligned errors, compare `Delta_reference(i,j) = inverse(T_reference(i)) * T_reference(j)` with the equivalent predicted relative transform. Apply the same translation/geodesic primitives to these deltas. This diagnostic is invariant to one fixed left world transform and does not fit alignment to each pair. It still depends on correct object frames and does not remove scale error. Use frame lags 1 and 20 on the original episode frame-index grid, report actual timestamp deltas, and require all intervening frame-object rows to have eligible labels and valid predictions; report candidate/usable pair counts. Never close a dropout by renumbering retained rows. Lag 20 is approximately one second at the nominal rate, not exactly one second.

For longer-term drift, additionally compare relative transforms between the first and last valid matched rows of each object. Record endpoint keys, elapsed seconds, endpoint support, intervening missing-label/prediction counts and predicted/reference path lengths on valid adjacent segments. Gaps make this an endpoint-only diagnostic and do not prove continuous tracking. Raw drift is metres and radians; normalized drift per metre is reported only when reference path length is positive, with that denominator attached. A stationary object yields null per-metre drift rather than division by zero. This endpoint statistic and lagged errors remain necessary even if aligned absolute error is small.

For each metric report count, mean, median, RMSE and nearest-rank p90 where applicable, with null for zero support. Pool frame-object samples for the micro result; also publish the complete per-task table and equal-task macro mean of task means over tasks with support, listing how many tasks contribute. Compute micro coverage from pooled counts and macro coverage from per-task ratios. Never average percentages without naming the weights. Retain per-episode and per-object rows so `Pan_Solo` weighting and absent tasks are visible. Development and validation are separate reports. For any uncertainty interval, resample whole episodes, not correlated frames; seven validation episodes are very limited evidence.

Contact phase and ego-image visibility strata require independently sourced annotations with provenance; do not infer them from pose validity. Reprojection residuals require validated extrinsics and measured image correspondences, and must state distorted/undistorted pixel space. Temporal acceleration uses physical timestamps and valid consecutive triplets, not interpolated dropouts. Hand-object penetration requires validated hand/object meshes, signed-distance convention, scale and contact tolerances. These diagnostics are specified as later work and have no measured values in this batch. No public object-pose error is a substitute for robot task success, policy reload, simulation or official scoring.

## First-batch verification

`python -m unittest discover -s tests -p 'test_metrics*.py' -v` exercises analytic composition direction, both inverses, quaternion signs, tiny/half-turn angles, invalid zero/nonfinite poses, normalization, relative-gauge invariance, injected endpoint drift and coverage edge cases. `python -m unittest discover -s tests -p 'test_splits*.py' -v` verifies the exact 13/7 partition, singleton handling, deterministic selection, metadata hashes/tasks and grouped camera/trajectory filename inventory. The snapshot integration test explicitly skips when the pinned cache is absent; a skip is not dataset validation. Neither suite reads trajectory values or evaluation files.

On 2026-09-24 the local Python run passed all 10 metrics tests (0.001 s) and both split tests (0.002 s), with zero skips. Commands exited successfully; wall time of the combined test command was 0.55 s. No dependency installs or GPU work occurred. Only small source/config/report files and local Python bytecode caches were produced. These tests cannot verify official metric equivalence or reconstruction quality.
