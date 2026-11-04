# Stereo depth and metric scale

Turns the calibrated mono stereo pair into metric depth and fixes the scale of monocular depth. The scorer removes one rigid
transform per episode and no scale, so a reconstruction 5% too large moves every object 5% too far. The calibrated baseline
is the only measured ruler that does not depend on object identity. Baseline runs are under `data/outputs/stereo_depth/`;
independent accuracy is not established.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth rectify --request src/features/stereo_depth/rectify.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth generate --request src/features/stereo_depth/generate.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth anchor --request src/features/stereo_depth/anchor.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth correspondences --request src/features/stereo_depth/correspondences.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth report --request src/features/stereo_depth/report.json
```

All CPU, using the OpenCV and NumPy already installed. Runtime, RAM and disk are unmeasured. `runtime.json` keeps the flags and episode lists.
`correspondences`, `report` and non-default anchor variants are disabled by default, never accept episodes 41, 43 or 44, and have
never been run. No pose label is read by any operation.

## Baseline method

1. Rectify. Camera C is left and B is right. `R1, R2, P1, P2, Q` come from one `stereoRectify` call with all 14 distortion
   coefficients. The stored rectification rotations fail their closure test by 0.039 and are not used.
2. Match with semi-global block matching, a left-right check and speckle removal.
3. `depth = P1[0,0] * baseline / disparity`. The nominal 75 mm baseline is never mixed with the calibrated 75.47 mm.
4. Transfer valid pixels to the colour pinhole grid, keeping the nearest point per pixel. Holes stay invalid.
5. Anchor. Per frame, take the median of `stereo / monocular` on pixels valid in both, excluding masks when supplied. Report the
   median over frames and keep per-frame values. Unmasked support is not separated, and excluding masks does not prove a static
   background.

Known weakness: sampled rectified pairs show a median vertical residual of about 2 px (two frames of public episode 41), and a row-wise
matcher loses accuracy there. Possible fixes are refining the pair rotation from static background, or a learned matcher, which is a new
dependency and needs a licence check. Neither has been chosen.

## Output

`data/outputs/stereo_depth/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Meaning |
| --- | --- | --- |
| `rectification.npz` | `R1, R2, P1, P2, Q`, `map_left_x/y`, `map_right_x/y` | Matrices and remap tables |
| `frames/frame_XXXXXX.npz` | `disparity_px`, `depth_m`, `valid` | Rectified left frame |
| | `rgb_depth_m`, `rgb_valid`, `rgb_intrinsics_pixels` | Same depth on the colour grid; K must equal the depth slice's |
| | `left_right_residual_px`, `rgb_match_confidence` | Uncalibrated matching quality on newer archives, not a probability; older ones lack it |
| `anchor.json` | per frame, per episode | Scale, support, spread |
| `run.json` | | Request, hashes, per-frame status, timing |

A frame with too little support is `insufficient_support` and gets no scale. `anchor` fails the episode when too few frames have one.

## Independent geometry (never run)

`correspondences` uses ORB with reciprocal ratio matching (0.75), a 2000-feature ceiling, vertical admission 3 px and reprojection
admission 2 px. It reads the depth producer's grid only, never its depth values. Textureless areas stay missing, so the report cannot show
full surface accuracy. Query IDs are `a:{row}:{column}`, and residue zero modulo five of a SHA-256 hash is reserved as holdback. Limits: 32
frames, maximum side 384, 1200 s.

`report` needs a frozen protocol ([template](geometry-protocol.example.json), unusable unchanged) that binds candidate, baseline, queries and
frames before residuals are read, with one frozen metric scale per candidate. `geometry.json` gives fixed-scale log-depth, metre and relative
residuals, reprojection, vertical residual, coverage and paired comparisons, by stratum: unknown, static, hands, objects. Nothing ranks candidates
automatically. Static support needs a producer that does not exist yet, so temporal scores stay null with semantic masks. Thresholded vertical
statistics are conditional, so use the producer's pre-admission distribution when checking calibration tails. `anchor.holdback_modulus=5` and the
`log_median` and Huber aggregations sit behind `anchor_research_authorized` (false).

[anchor-comparison.json](anchor-comparison.json) is a research design and not a CLI request.

## Open questions

- Does the public calibration apply to the evaluation recordings? This design assumes the same rig.
- May scored inference use the stereo streams and calibration? The kit says to reconstruct from the released videos, which include all three streams.
- Is the 2 px vertical residual calibration error, timing or rolling shutter?
