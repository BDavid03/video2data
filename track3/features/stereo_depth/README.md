# Stereo depth and metric scale

Status: baseline opening-39 geometry stages completed in the canonical checkout;
independent accuracy remains unestablished. See the completed opening records in
[evidence reconciliation](../../../experiments/01-evidence.md). The runtime flags
reflect prior episode39 research authorization, not permission to run checks.

This slice turns the calibrated mono stereo pair into metric depth, and uses it to
fix the scale of the monocular depth from the [depth](../depth/README.md) slice.
The entry needs it because trajectory scale is scored and nothing downstream can
repair a wrong scale.

## Role in scoring

The scorer removes one rigid rotation and translation per episode and no scale:
"Trajectory scale error remains scored" (**Official**,
`KIT/tracks/track_3/README.md:34-36`). A reconstruction that is 5% too large moves
every object 5% too far, and AUC, SP-SR, MP-SR, RPE and MPPE all pay for it. The
calibrated stereo baseline is the only measured ruler that does not depend on
object identity. Known public scan dimensions are a cross-check and the fallback
scale anchor for episodes with a scanned object under D07. Evaluation episodes
have no pose labels or camera-to-world transform.

## Operations

The baseline flags retain the previously authorized episode39 campaign state.
New research operations are disabled.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `rectify` | Rectification maps and projection matrices for one processing resolution | Perception environment | No |
| `generate` | Per-frame disparity and metric depth | Perception environment | No for the baseline matcher |
| `anchor` | Per-frame scale between a monocular depth run and this slice's depth | Perception environment | No |
| `correspondences` | Sparse reciprocal independently triangulated image queries | Perception environment | No |
| `report` | Reserved fixed-scale residuals, coverage and paired comparison | Perception environment | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `ego_cam_c` and `ego_cam_b` clips | Dataset | Decoded frames and integer MP4 PTS |
| `track_3/public/meta/camera_calibration.json` | Dataset | K, all 14 distortion coefficients, pair rotation and translation |
| `configs/calibration/ego.json` | Calibration audit | Left and right assignment, transform directions, orientation policy |
| `outputs/depth/.../run.json` and frames | [depth](../depth/README.md) | Monocular depth and validity, for `anchor` only |
| `outputs/object_masks/.../frames` | [object_masks](../object_masks/README.md) | Hand and object masks, for `anchor` only, optional |

The calibration file ships under `public/`. The evaluation split has videos only.
This design assumes the same rig recorded both splits (**Open**, see below).

## Method

1. **Rectify.** Camera C is left and B is right (**Measured**,
   `reports/calibration/README.md`). Recompute `R1, R2, P1, P2, Q` in one
   `stereoRectify` call from K, the full 14-coefficient distortion and the pair's
   rotation and translation, with zero-disparity alignment. The rectification
   rotations stored in the calibration file fail their own closure test by 0.039
   and are not used (**Measured**, same report).
2. **Match.** Baseline matcher: semi-global block matching from the OpenCV build
   already installed, with a left-right consistency check and speckle removal.
   It needs no new dependency. Its search is one-dimensional along the row.
3. **Convert.** `depth = P1[0,0] * baseline / disparity`, with focal length and
   baseline taken from the same `stereoRectify` call. The nominal 75 mm baseline is
   never mixed with the calibrated 75.47 mm translation (**Measured**, same report).
4. **Transfer to the colour camera.** Back-project valid pixels in the rectified
   left frame, rotate them into camera C's optical frame with `R1ᵀ` from the
   same `stereoRectify` call, then transform them with
   `A_T_C = inverse(C_T_A)`. Project them into the
   colour pinhole grid defined by the depth slice, and keep the nearest point per
   pixel. Holes stay invalid. Nothing is inpainted.
5. **Anchor.** For each frame, take pixels valid in both the monocular depth and
   the transferred stereo depth, excluding hand and object masks when available.
   Estimate one scale factor as the median of `stereo / monocular`. Record the
   support count and the spread. Report a per-episode scale as the median over
   frames, and keep the per-frame values so drift is visible. Source-only backlog
   additions record grid/support share, ratio quantiles, log-ratio MAD, frame
   support share and episode log-scale MAD. Unmasked support is explicitly
   unsegregated; excluding supplied masks does not prove static background.
   The scale estimator is unchanged; these new fields have not been executed.

**Known weakness.** The sampled rectified pairs show a median vertical residual of
about 2 px (**Measured**, two frames of public episode 41). A row-wise matcher
loses accuracy when corresponding points sit on different rows. Two remedies are
designed, and neither is chosen until a run shows which is needed:

| Remedy | Cost |
|---|---|
| Refine the pair rotation from static-background matches in the images themselves. No pose label is involved, so it may run on evaluation episodes. | No new dependency |
| Replace the matcher with a learned stereo network that tolerates small vertical offsets. | New dependency and weights, needs the user's confirmation and a licence check |

## Outputs

`outputs/stereo_depth/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Type and shape | Meaning |
|---|---|---|---|
| `rectification.npz` | `R1, R2, P1, P2, Q` | float64 | From one `stereoRectify` call |
| | `map_left_x/y`, `map_right_x/y` | float32 H×W | Remap tables at the processing resolution |
| `frames/frame_XXXXXX.npz` | `disparity_px` | float32 H×W | Left-referenced disparity, NaN when invalid |
| | `depth_m` | float32 H×W | Camera-z in the rectified left frame, metres |
| | `valid` | bool H×W | Consistency check passed, finite, inside the configured range |
| | `rgb_depth_m`, `rgb_valid` | float32, bool | The same depth on the colour pinhole grid |
| | `rgb_intrinsics_pixels` | float64 3×3 | Must equal the depth slice's `input_intrinsics_pixels` |
| `anchor.json` | per frame | | Scale, support pixel count, median absolute deviation |
| | per episode | | Median scale, frame count used |
| `run.json` | | | Request, input hashes, per-frame status, timing |

A consumer must check `rgb_intrinsics_pixels` against the grid it uses. Frame
identity is the frame index and integer PTS; no other clock is read.

## Configuration and authorization

[anchor-comparison.json](anchor-comparison.json) is a bounded **research design**,
not a CLI request. Its proposed independent support and comparisons are described
in [item04](../../../experiments/04-metric-anchor.md); no comparison ran here.

`runtime.json` retains the prior `rectify_authorized`, `generate_authorized` and
`anchor_authorized` state for episode39. New research flags are false. Request files name the episode, frame range,
processing resolution, disparity range and the depth limits.

## Dependencies

Baseline: OpenCV and NumPy, already present in `.runtime/moge-python313`. No new
package. A learned matcher would be a new production dependency.

## Resources

CPU only for the baseline. Runtime, RAM and disk are unmeasured.

## Failure handling

A frame with too few valid pixels is recorded as `insufficient_support` and gets no
scale. `anchor` fails the episode when fewer than a configured share of frames
have a scale, because a guessed scale is worse than a visible gap.

## Separation of public labels

No pose label is read in any operation. Public pose labels could check the scale
on development episodes, since object height above the table is physical there.
That check belongs to a separate diagnostic with the `public_label_diagnostic`
tag and never feeds `anchor`.

## Acceptance evidence

Once runs are authorized: the per-frame scale is stable within an episode, the
transferred depth agrees with monocular depth shape on static background, and a
known dimension of a public scanned object measured in the depth matches the scan.

## Open questions

| Question | Who can answer |
|---|---|
| Does the public calibration apply to the evaluation recordings? | Organisers |
| May scored inference use the stereo streams and the calibration? The kit says to reconstruct "from the released egocentric videos", which include all three streams. | Organisers |
| Is the 2 px vertical residual calibration error, timing or rolling shutter? | A wider image-only check on development episodes |

## Independent geometry implementation (unexecuted)

`correspondences` and `report` are new, separately disabled operations. Existing
rectify/generate/anchor flags and episode authorization are preserved. New
operations and non-default anchor variants accept only public training/tuning
0/11/12/18/21/23/26/31/39/42, never 41/43/44. They require canonical output access;
no result was produced in this implementation checkout. Source was read without
running checks, imports, models, or compiler/dependency commands.

The sparse producer uses the existing OpenCV ORB and Hamming matcher, a fixed
2000-feature ceiling, reciprocal ratio-tested descriptors (ratio .75), the saved
joint rectification, positive triangulation, configurable vertical admission
(default 3px) and both-eye triangulation reprojection admission (default 2px).
It reads the depth producer's **grid only**, never predicted depth values. Its
integer PTS is the same audited stereo/source clock as the baseline producer.
Rows retain eligible left queries, reciprocal matches, admitted queries and
rejections; textureless areas remain missing. Reciprocal matching and positive
triangulation are a visibility proxy, not proof that occlusions have disappeared.
The report therefore cannot establish full surface or textureless accuracy.

Each accepted triangulated point is mapped into the colour grid and assigned
`a:{row}:{column}`; collisions keep one deterministic best descriptor match.
The query partition hashes UTF-8 `episode/frame/query_id` with SHA256, takes the
first eight bytes as an unsigned big-endian integer, and reserves residue zero
modulo five. `anchor.holdback_modulus=5` excludes those exact colour pixels from
fitting. `holdback_modulus=0` preserves the incumbent's fitting support. The new
`anchor_research_authorized=false` gate protects non-default fitting/aggregation.
`episode_aggregation` offers incumbent `median`, `log_median`, and a fixed
20-iteration Huber log-location challenger; per-frame medians remain unchanged.
There is no selection or promotion in this slice.

Sparse producer NPZ arrays are `query_ids` (N strings), `rgb_uv` (N×2 integers),
`stereo_depth_m` (N optical-z metres), `left_uv/right_uv` (N×2 rectified pixels),
`left_projection/right_projection` (3×4 mapping A optical points to rectified
C/B pixels), and `input_intrinsics_pixels` (3×3). Identity and exact integer PTS
are stored in each artifact and its hashed producer frame row. The producer kind
is `stereo_depth.correspondences` and its declared contract is
`independent-image-correspondences-v1`. At most 32 selected frames, maximum side
384 and 1200 CPU wall seconds are admitted; archive payloads are bounded by the
2000-feature ceiling. Source decoding before the selected window counts against
the wall budget, and no model runs in this operation.

`report.json` supplies candidate and optional baseline depth runs, the completed
sparse producer, optional regions/motion, and an explicit frozen protocol. The
[protocol template](geometry-protocol.example.json) intentionally has null hashes
and scales and cannot be admitted unchanged. The protocol binds the candidate,
baseline, query producer and complete frame list before residual outcome access.
One metric scale is frozen per candidate. Native metric input must use exactly
1; any correction must point to a completed anchor with the same deterministic
fitting-support holdback. `depth.ingest_challenger` metric optical-z outputs are
admitted on their unchanged grid; relative optical-z outputs need that explicit
frozen metric anchor. No per-frame normalization occurs. Anchoring a challenger
also requires holdback5, which invokes the disabled research-anchor gate.

The `geometry.json` report includes fixed-scale signed log-depth, metre and
relative residuals, both-eye candidate reprojection distances, and independent
rectified vertical residual. Each stratum retains absolute-error numerator,
admitted denominator, eligible support, coverage, median/p90/maximum and missing
reason. Paired candidates additionally retain support intersection, union,
candidate-only, baseline-only, both missing, and absolute-error differences on
the intersection. Per-query values and worst-error maxima remain available for
inspection; none becomes an automatic candidate ranking.

Strata are unknown/static/hands/objects (integer labels 0/1/2/3). An optional
`object_masks.generate` producer supplies hands and objects; overlapping masks
and the complement stay unknown. Explicit static masks require a completed
same-grid video-only producer declaring `region_contract=geometry-regions-v1`
and frame NPZ `region_labels`; no such static-region producer is claimed to
exist. Mask names are semantic strata, not accepted persistent object identities.
Object/event IDs and identity-failure flags therefore remain null in query rows.

Optional temporal evidence accepts existing `camera_motion.estimate` hashed
`trajectory.npz` with exact observed frame/PTS poses, or a completed same-grid
producer declaring `motion_contract=calibrated-camera-motion-v1` with frame NPZ
`current_from_previous`, `previous_frame_index`, `previous_pts`. It uses calibrated
backprojection, saved rigid transforms, a forward nearest-depth buffer and
backward visible support, with explicit out-of-field, missing and occlusion
counts. It compares only admitted static support. Dynamic/unknown support is
excluded, never assumed rigid. The default 3% depth visibility threshold is
reported as conditional admission and its rejection count retained. With only
semantic masks, static support is unavailable, so temporal scores remain null.
Accepted object-motion transforms, true independent unused-view observations,
rigidity, edges, events and achieved behaviour are not inferred by this report.

`geometry.json` frame query rows carry A-grid `rgb_uv` plus observed/predicted
left/right endpoints. Those arrows belong on the **rectified C/B views**, not
on colour A. Invalid candidate projections are null. `run.json` hashes the
report, request/protocol, source files, all producer runs and consumed artifacts.
The selection slice may ingest an explicit receipt binding this report; missing
channels/events prevent promotion. New output paths use the existing canonical
Z: guard, and all new operation flags remain false.

Research requests may optionally specify increasing unique `frame_indices`
inside `[start_frame, stop_frame)` for the predeclared sparse sample centres.
The correspondence/report budget counts selected frames, not interval span;
source sequential decoding still counts against the wall ceiling. Nonconsecutive
samples explicitly have unavailable temporal consistency. Anchor supports the
same frame identities behind its research gate, retaining the incumbent default
contiguous selection.

Scale-drift diagnostics retain each frame/stratum's median observed log stereo
ratio, one fixed first observed origin, median, p90 absolute spread,
first-to-last change and exact missing frame identities. These ratios never
renormalize predictions or refit the frozen metric scale.

Correspondence frame rows also preserve sequential rejection reasons and vertical
median/p90/maximum over **all reciprocal matches before geometric admission**.
The report's admitted vertical statistic is conditional on its configured gate;
use the producer's pre-admission distribution and rejection counts when examining
calibration tails rather than mistaking thresholded support for the full sample.

### Matching quality for original-learning inputs

The existing SGBM generator now also owns `left_right_residual_px` on its valid
rectified grid and `rgb_match_confidence` on the colour grid. Matching quality is
`clip(1 - abs(left_disparity + sampled_right_disparity) / LR_tolerance, 0, 1)`;
it is a continuous residual diagnostic, not a binary validity mask, calibrated
probability, visibility probability or error bound. Invalid pixels remain NaN.
Quality transfers from the same nearest visible stereo point as metric depth;
exact depth ties use the first row-major source point. It never averages or
maximizes quality across overlapping surfaces. Existing matcher parameters,
depth values and validity policy are preserved.

The producer declares `confidence_field=rgb_match_confidence`,
`confidence_semantics=uncalibrated_matching_quality_not_probability`, and
`confidence_origin=linear_left_right_disparity_residual`. New frame archives are
also hash-owned in `run.artifacts` for the explicit learning input adapter.
Historical archives lack this confidence field and are not backfilled or relabelled.
No confidence calibration claim or informative-value result exists before an
actual admitted run. This source addition remains unexecuted.

The independent report also admits `scratch_depth.research_predict` only for
39/42 with declared `depth_kind=metric_optical_z_estimate`, exact copied
calibration grid, `depth_m`, `valid`, and `input_intrinsics_pixels`. These are
candidate estimates to assess, not established metric accuracy or automatic
promotion.
