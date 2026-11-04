# Stereo depth and metric scale

Pipeline role: step 3 (static poses). Calibrated stereo gives metric depth, and the static placement of each object in
the 19 static episodes uses it (through `pose_insurance`). It also fixes the scale of MoGe-2 depth.

The scorer removes one rigid transform per episode but no scale. A scene 5% too large puts every object 5% too far.
The calibrated stereo baseline is the only measured ruler we have that does not depend on knowing the object.

## Run

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth rectify --request src/features/stereo_depth/rectify.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth generate --request src/features/stereo_depth/generate.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth anchor --request src/features/stereo_depth/anchor.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth correspondences --request src/features/stereo_depth/correspondences.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.stereo_depth report --request src/features/stereo_depth/report.json
```

CPU only, OpenCV and NumPy. `runtime.json` holds the switches and episode lists. `correspondences`, `report` and the
extra anchor variants are off and have never been run. Episodes 41, 43 and 44 are always refused. No pose label is read.

## Method

1. Rectify. Camera C is left, B is right. One `stereoRectify` call with all 14 distortion terms gives `R1, R2, P1, P2, Q`.
   (The stored rectification rotations fail a closure test by 0.039, so we do not use them.)
2. Match with semi-global block matching, a left-right check and speckle removal.
3. `depth = P1[0,0] * baseline / disparity`, with the calibrated 75.47 mm baseline (never the nominal 75 mm).
4. Move valid pixels onto the colour camera's pinhole grid, nearest point per pixel. Holes stay invalid.
5. Anchor MoGe-2: per frame, the median of `stereo / monocular` over pixels valid in both, masks excluded when given.
   Report the median over frames and keep per-frame values.

Known weakness: rectified pairs show about 2 px median vertical error (two frames of public episode 41). A row-wise
matcher loses accuracy there. No fix was chosen. Independent accuracy is not established.

## Output

`data/outputs/stereo_depth/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Meaning |
| --- | --- | --- |
| `rectification.npz` | `R1, R2, P1, P2, Q`, `map_left_x/y`, `map_right_x/y` | Matrices and remap tables |
| `frames/frame_XXXXXX.npz` | `disparity_px`, `depth_m`, `valid` | Rectified left frame |
| | `rgb_depth_m`, `rgb_valid`, `rgb_intrinsics_pixels` | Same depth on the colour grid |
| | `left_right_residual_px`, `rgb_match_confidence` | Matching quality (uncalibrated; newer runs only) |
| `anchor.json` | per frame, per episode | Scale, support, spread |
| `run.json` | | Request, hashes, per-frame status, timing |

A frame with too little support is `insufficient_support` and gets no scale. `anchor` fails the episode when too few
frames have one.

## Unused checks (never run)

`correspondences` matches ORB features between the two views to test geometry independently. `report` scores a candidate
against a protocol frozen before any residual is read ([template](geometry-protocol.example.json), not usable as is).
[anchor-comparison.json](anchor-comparison.json) is a design note, not a request.

## Open questions

- Does the public calibration hold for the evaluation recordings? We assume the same rig.
- Is the 2 px vertical error calibration, timing or rolling shutter?
