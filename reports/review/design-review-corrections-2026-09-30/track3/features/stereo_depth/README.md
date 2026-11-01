# Stereo depth and metric scale

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

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

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `rectify` | Rectification maps and projection matrices for one processing resolution | Perception environment | No |
| `generate` | Per-frame disparity and metric depth | Perception environment | No for the baseline matcher |
| `anchor` | Per-frame scale between a monocular depth run and this slice's depth | Perception environment | No |

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
   frames, and keep the per-frame values so drift is visible.

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

`runtime.json` holds `rectify_authorized`, `generate_authorized` and
`anchor_authorized`, all false. Request files name the episode, frame range,
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
