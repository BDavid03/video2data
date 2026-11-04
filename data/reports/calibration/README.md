# Foundation 04: calibration and frame conventions

Audit date: 2026-09-24. Dataset revision: `5f68335f3acc802033d1e80728c1633197521de8`.
Upstream source pin: `43a3f5b4d479581f2e8bdf5e6ed7ca361d39ec5a`.

**Use decoded ego images without an additional 180-degree rotation. Keep C as stereo left and B as stereo right. Recompute rectification from K, the complete 14-element D, and the pair's R/t. Factory geometry remains provisional: the sparse checks show material stereo and RGB alignment residuals.**

The machine-readable policy is [ego.json](../../configs/calibration/ego.json), and reproducible numerical results are [numeric-audit.json](numeric-audit.json). The original calibration was not changed. Its SHA-256 is `23e2d9c46369292cdda492a777e7a6083815369a25d1af4bb82560669fde5d52`.

## Evidence and findings

The pinned [Track 3 README](../../datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/README.md) documents frame-exact public video/parquet correspondence, metres, w-first quaternions, and `world_T_object` in scanned-mesh coordinates. Invisible all-zero poses are invalid. Each episode has independently centered x/y coordinates; z is not centered. Camera-to-mocap-world extrinsics are absent. Never use a shared global episode origin or pretend factory rig extrinsics provide camera motion/world registration.

The [factory calibration](../../datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public/meta/camera_calibration.json) records `rotation_deg=180` on the ego device, three cameras, and stereo pair. Its note explicitly swaps pair ordering to `ego_cam_c` left and `ego_cam_b` right. Rotation metadata describes the rig; it is not an instruction to rotate already oriented decoded pixels again.

Sparse CPU geometry diagnostics used public episode **41**, the lowest released `Pan_Solo` ID, at frames **0 and 149** of 299. Saved full-resolution first-frame images were inspected: [A RGB](episode_000041_ego_cam_a_frame000000.jpg), [B mono](episode_000041_ego_cam_b_frame000000.jpg), [C mono](episode_000041_ego_cam_c_frame000000.jpg). The scene and hands are already upright. Decoder autorotation was explicitly disabled. Geometry evidence remains limited to these two frame indices.

A later [RGB contact sheet](episode_000041_rgb_review_contact_sheet.jpg) adds uniformly spaced review frames **0, 99, 199, 298**, with MP4 PTS 0.00/4.95/9.95/14.90 s. These are review samples, not detected contact/release events. Visual inspection shows one white handled pot, a visible support table and substantial static background; hands manipulate the pot in the middle two samples and are clear of it in the first/last samples. The associated [pilot profile](../../configs/calibration/pilot-episode-41.json) proposes episode41 for an initial reconstruction, within the frozen development split. This does not validate unsampled frames or accurate hand contact timing.

SIFT matching used 4,000 candidate features, mutual Lowe-ratio 0.7 matching, and no fitted fundamental-matrix inlier rejection. Full 14-coefficient distortion was inverted with 50 iterations and a 1e-10 stopping epsilon. C-to-B results use newly computed OpenCV rectification with zero-disparity alignment and alpha=0. The two orientation hypotheses share the same descriptor matches; the alternate hypothesis maps each pixel through `(width-1-u,height-1-v)` before applying calibration.

| Pair / frame | Mutual matches | Decoded-as-calibrated residual | Extra 180 residual |
|---|---:|---:|---:|
| C to B / 0 | 523 | median vertical 2.081 px; p90 4.158 px | median vertical 13.256 px; p90 23.305 px |
| C to B / 149 | 555 | median vertical 1.916 px; p90 4.024 px | median vertical 12.402 px; p90 20.256 px |
| A to B / 0 | 407 | median Sampson 4.572 equivalent px | median Sampson 15.461 equivalent px |
| A to B / 149 | 370 | median Sampson 2.990 equivalent px | median Sampson 13.669 equivalent px |

RGB Sampson residuals are normalized-camera residuals multiplied by B's focal length, so they are approximately stereo-pixel-equivalent and are not directly interchangeable with vertical errors. Positive C-left/B-right disparities account for 97.51% and 96.40% of the matched pairs. These checks support the chosen orientation, order, and transform direction, but **do not validate dense metric depth or physical exposure synchronization**. The remaining error can include factory calibration error, mismatches, rolling shutter, motion, or timing; this test does not distinguish them. No calibration was fitted to these samples, and no ground-truth trajectories were used.

## Transforms, distortion, and rectification

Adopt column vectors, with `A_T_B` mapping B coordinates into A. The optical frame interpretation is x right, y down, z forward. This is the project's OpenCV convention, not an assertion that the README specifies all camera axes.

The field named `extrinsics_to_stereo_left` still uses historical socket B as its anchor: C's field matches the explicitly named C-to-B pair transform within factory rounding, while B has no such field. A's field is therefore interpreted as `B_T_A`; derive `C_T_A = inverse(B_T_C) * B_T_A`. The equality residual and composed matrix are recorded in the numerical audit. Do not reinterpret A's field as C-to-A simply because the rotated pair now names C as left. RGB image checks support the B anchor, with the residual limitations above.

All six camera entries have Perspective distortion with **14 coefficients**, ordered `[k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y]`. The final two tilt terms are nonzero; passing only 5 or 8 coefficients loses part of the model. This order is documented by [Luxonis's calibration API](https://docs.luxonis.com/software-v3/depthai/api/cpp) and [OpenCV's camera model](https://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html). The source's `Perspective` label must not be treated as a fisheye model.

All tested extrinsic/rectification rotations are proper and orthonormal within 1e-5 tolerance; factory rounding produces errors of order 1e-7. Numerical validity does not imply geometric correctness. In particular, the supplied ego rectification matrices fail the consistency relationship `R_right * R_left_to_right * inverse(R_left) = I`: maximum element error is **0.0385761**. Their transformed translation is approximately `[-0.0754405,-0.00170010,0.00131052]` m, with nonzero y/z. They must not be used as a validated horizontal stereo rectification. Recompute R1/R2/P1/P2/Q together from the pair extrinsics and inspect correspondence residuals.

The nominal ego baseline is **0.0750000048 m**, while the norm of the calibrated translation is **0.0754710207 m**, a **0.471016 mm** difference (about 0.628%). Use the resulting rectified P/Q matrices for depth. Mixing nominal baseline with a Q matrix derived from the calibrated translation introduces a scale inconsistency.

For image transforms, the stdlib helpers explicitly implement `u'=W-1-u`, `v'=H-1-v`, and half-pixel resize `u'=sx*(u+0.5)-0.5`. `K'=H*K` preserves the original optical frame and intentionally has negative focal entries after180. APIs requiring canonical positive focal lengths need an explicit optical basis change `S=diag(-1,-1,1)`: `K'=H*K*S` and `A'_T_B'=S4*A_T_B*S4`. Distortion must also be handled consistently; the helpers deliberately do not silently modify the 14 coefficients. Prefer distortion/rectification in the preserved calibration domain, and record output K/P, resolution, crop, and zero-distortion state for subsequent consumers.

## Pairing and clocks

The completed [data audit](../data-audit/audit.json) and [video records](../data-audit/videos.json) provide full decoding evidence: 120 clips, 31,281 decoded frames, 40 episodes across public/evaluation; every episode has equal stream counts and identical integer PTS. All 20 public parquet row counts match video counts. Every clip has 20/1 fps, 1/10240 timebase, zero PTS origin and 0.05-second increments. No duplicate/backwards PTS, adjacent identical decoded-frame checksum candidates, or display-matrix/rotation metadata lines were found.

The data audit compares integer PTS exactly (zero tolerance). The configuration's 1e-6-second tolerance is a prospective serialization check, not a claimed exposure tolerance. Pair streams and public rows using `(split,episode_index,frame_index)`. The public metadata nominal rate is **20.25411910142744 Hz**, so parquet `timestamp` and 20 Hz MP4 PTS diverge by up to **0.2333751678 seconds**. Preserve video PTS, parquet timestamp and parquet capture_time separately; nearest-timestamp matching would corrupt the documented frame map.

Equal encoded PTS and frame counts establish container correspondence. They do not prove the three sensor exposure times coincide or exclude dropped capture frames before encoding. Per-camera hardware exposure timestamps are not available in the audited artifacts. The calibration explicitly says separate ego/exo devices are **not hardware synchronized**, with reported ego-versus-master skew mean **7.39 ms**, max **7.41 ms**. Those values concern cross-device timing; they cannot be used as an estimated within-ego RGB/mono offset.

## Validation and remaining work

`python -m unittest discover -s tests -p 'test_calibration*.py' -v` passes **9 tests**. Coverage includes 180-degree corner mapping/involution, half-pixel resize, transformed projection equivalence, optical basis changes, rigid inversion, improper/nonfinite matrix rejection, full distortion preservation, pinned intrinsics/order/baseline, and corrupted calibration rejection.

Reproduce the numeric and sparse-image audit from repository root:

```powershell
. ./scripts/use-local-artifacts.ps1
python -m track3.calibration.audit --snapshot datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8 --output reports/calibration --image-check
```

The core parser/helpers/tests use only the Python standard library. The optional image check used already installed OpenCV 4.13.0 and NumPy, one CPU thread, no downloads/installations and no GPU. The measured audit runtime including the contact sheet is about **3.7 seconds**; image/report outputs occupy less than **2 MB**. Peak RAM was not measured. All outputs, including sparse images, remain in this repository; the CLI uses the shared output guard to reject outside-root and immutable-input locations.

Before dense metric depth, expand checks to static background and moving-hand segments in several public episodes, report epipolar residual distributions at the actual processing resolution, and investigate factory calibration refinement without validation-label leakage. Test RGB-to-stereo registration across depth with explicit occlusion handling. Physical synchronization needs sensor timestamp evidence or a separate motion-based timing estimate with uncertainty; do not infer it from PTS. New sources/evaluation preprocessing need an orientation spot check before reusing the public-pilot settings.
