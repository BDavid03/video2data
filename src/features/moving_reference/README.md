# M1 video masks

Fixed Grounding DINO + SAM2.1 Hiera-Large masking queue: public 39, then evaluation
38. Only `ego_cam_c` videos and their clock inventories are opened. This is the
stream stereo_depth rectifies as its left view. Masks use the **raw distorted C
pixel grid**, with no autorotation or rectification; a maximum-side 1024 resize is
recorded. Consumers must transform the masks when they rectify the images.
No object labels, mocap, scans, camera tracker or object tracker are read or built.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference
```

`runtime.json` freezes the ordered queue and rules before any detections: frame-0
pan prompts `dust pan`, `yellow dustpan`, `scoop` in that order, stopping at the
first success. G3 box/text thresholds are 0.40/0.30 on both clips and hands. Hands
are detected at frames 0,10,20,..., with NMS IoU 0.70, at most two boxes, and anonymous
slots assigned by minimum center displacement. There is no per-episode tuning.
Hands entering between detection frames can be recovered by reverse propagation;
these are model predictions, not observations or anatomical identities.

SAM2 uses its pinned 1024-square model input, bfloat16 inference, CPU-offloaded
video and state, no hole filling or CUDA connected-component extension. CUDA's
allocator is capped at 15000 MiB, below the 16 GiB budget. All GPU work is sequential
under `src.gpu.GpuLease`; workers verify parent ownership. The handler stops GPU
work at 07:49:20 UTC on 4 November, allowing teardown before 07:50. Worker alarms
and frame checks implement that deadline. No process outside the job is signalled.

The pan is seeded only at frame 0 and uses forward masks. Hands are re-seeded at
the fixed cadence. The official video predictor runs forward and backward over
the full clip, then each hand uses the direction of its nearer seed (ties use
forward). Forward/reverse IoU is saved as a diagnostic, including the pan. Masks
are thresholded at logit 0; no connected-component or area filtering is applied.

Each `data/outputs/moving_reference/epNN/masks/` contains:

- `request.json`: frozen settings, stream clocks, source/input/weight hashes and
  prepared JPEG hashes; `input_frames/` keeps the model's exact input.
- `detections.json`: all prompt attempts and hand keyframes, scores and assignments.
- `frames/frame_XXXXXX.npz`: boolean `masks` in order pan/hand_0/hand_1,
  `frame_index`, `timestamp_seconds` (video PTS), `pts_integer`, `track_ids`,
  `selected_direction` (0 forward, 1 reverse), `forward_reverse_iou` (-1 unavailable),
  and `reverse_processed`. Overlap between tracks is retained.
- `summary.json`, `contact-sheet.png` (12 evenly spaced frames), stage reports and
  logs, `run.json` (artifact hashes and GPU telemetry), and append-only `ledger.jsonl`.

Completed stages are skipped on resume only after input, weight, settings and
source identities match. Incomplete segmentation restarts deterministically and
replaces its own derived frame files. A completed run is skipped. Coverage means
**nonempty mask**, not usable or correct segmentation: missing hands can be real
absence, and a nonempty wrong pan cannot establish the M1 90% usability gate.

## Real-data results, 4 November 2026

| Episode | Frames, both directions | Pan / hand_0 / hand_1 nonempty | Longest missing, all tracks | Pan area min / median / max (px) | Seconds | Peak device / torch reserved (MiB) | Visual finding |
|---|---:|---:|---:|---:|---:|---:|---|
| public 39 | 319 | 319 / 319 / 319 | 0 | 5446 / 9431 / 16419 | 203.66 | 9956 / 5286 | Pan body holds in all 12 inspected samples; gripping hand obscures the handle. |
| evaluation 38 | 263 | 263 / 263 / 263 | 0 | 62577 / 75658 / 83651 | 159.20 | 9424 / 4572 | **FAIL: the mask tracks the tabletop, not the pan**, already at frame 0 and through the final sampled frame. |

Both clips used `dust pan` without a fallback, with DINO scores 0.50457 and
0.55165 respectively. On 38, a threshold-passing detection was the wrong target:
the highest-scoring box surrounds the table, and SAM2 tracks the table with the
pan largely excluded as a hole. **Do not consume 38 track 0 as an object mask.**
No prompt, detection threshold, seed, direction rule or mask threshold was changed
after viewing either result; there was no segmentation retry or fallback tuning.
Nonempty coverage and high direction agreement do not establish the M1 usability gate.

I inspected both 12-frame contact sheets, plus the raw-versus-overlay detail sheet
on 39. Its visible pan body holds through lifting and rotation, with no apparent
sampled switch to the background. Around the gripping hand, especially sampled
frames 87, 116 and 260, object/hand overlap makes the handle boundary uncertain.
These are visual judgments on the sampled frames, not pixel ground truth or a
measured 90% usable-frame claim. Hand slots are not anatomically verified.

Both source clips are 1280x800, resized by **0.8** to 1024x640. Peak sampled device
use was 9956 MiB (9.72 GiB), below 16 GiB. The SAM2 stage used 168.75 seconds on 39
and 129.42 seconds on 38 (see stage reports for exact values). GPU execution ended
before 07:21 UTC; the cooperative lock was released well before 07:50.

The first queue completed 39, then refused 38 before inference because the initial
11000 MiB admission allowance plus 1024 MiB headroom exceeded the available free
memory. That refused attempt, including source snapshots, is preserved under
`ep38/masks/admission-refused/`. The allowance was corrected to **10000 MiB**, with
the same 1024 MiB headroom, and only 38 was resumed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference --episode 38
```

The only source changes between the two successful episodes are the operational
admission correction and this single-episode resume entry point. Their original
source identities remain in their respective requests. The default queue refuses
an existing request with a changed source identity; it does not silently replace
the completed 39 result. This reservation correction did not change model inputs
or inference settings. `visual-inspection.json` records the semantic failure on
38; `output-audit.json` independently checks each episode's frame identities,
timestamps, dimensions, counts and original artifact hashes.

Checks run: Python compilation, CPU imports, focused old-versus-new mask-statistic
and IoU equivalence, and these two real-data runs. No test suite ran. The full list
of written files is `ep39/masks/files-written.txt`; each `run.json` inventories its
artifacts. No camera or object tracker was built.

## Candidate A: sealed CPU stereo reference

The CPU tracker reads only the selected C/B videos, calibration, masks and clocks,
our canonical `thin_tool_profile` dimensions/profile, and the frozen M1 protocol.
It never opens episode parquet labels, mocap or earlier camera trajectories.
Episode 38 is admitted only from `masks-smallest/summary.json`; the table masks in
`ep38/masks/` are refused. Mask producer modules and artifacts stay unchanged.

Run 39 first (use a fresh attempt name only for a documented code defect):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference track --episode 39 --attempt candidate-a-cpu-support
```

Freeze the code, settings, protocol, body constant and 39 reference after review:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference freeze
```

Run 38 once with the frozen sources and settings:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference track --episode 38 --attempt candidate-a
```

A completed identical attempt is skipped; a changed identity is refused. The
append-only episode tracker ledger preserves completed attempts. The once marker
on 38 is written before tracking: a failed 38 operation is preserved and cannot
be retried through this handler. There are no sweeps or threshold searches.

`camera.py` contains CPU geometry and tracking functions; `_object.py` contains
object pose, rendering and smoothing mathematics; `gates.py` tabulates the frozen
limits; `track.py` orchestrates I/O; `support.py` runs the installed evaluator's
CPU physics in its existing isolated interpreter. All six source modules and the
shared constructor/path/provenance helpers are frozen by SHA-256 before 38.

The world origin is the nearest point of the stereo table plane to camera 0.
World z is its upward normal; x is the initial viewing direction projected onto
the table. A level table is assumed. Raw distorted SAM masks are transformed
with the same calibrated rectification maps, respecting their half-pixel resize.
Stereo runs at 960x600 with left/right consistency; reprojection residuals are
scaled to a 640-pixel width for the protocol gates.

A deterministic quarter of ORB landmarks in each spatial cell are reserved
before tracking. They never enter PnP or the fitted map; their neighbourhoods and
descriptors are excluded when adding map features. Held-out tracks are followed
independently by forward/backward LK and measured against their stereo positions
at birth. PnP-RANSAC/LM rejects inconsistent fitted matches; failed camera frames
hold the preceding estimate and are explicitly invalid. Forearms have no masks:
a conservative thick ray from each hand away from the object to the image border
is excluded, alongside dilated hands and object masks.

Frame-0 heading comes from the stereo object cloud's in-plane principal axis,
with four silhouette fits of our own mesh about the table normal. Initial height
comes from stereo; it is **not** snapped to the table. This assumes the scoop
plane is parallel to the table at rest, but does not assume its heading matches
world x. Opposite headings within 0.02 IoU are unresolved and invalidate the
reference. Both the initialization hypotheses and the base gap/penetration are
saved, including failures. Inter-frame forward/backward LK plus stereo and
RANSAC Kabsch provide object motion; retained frame 0 and the two latest
10-frame keyframes provide direct object-coordinate refinement. Loss of
keyframe support invalidates a frame; pairwise fallback poses remain visible
as diagnostics and are never silently admitted as observed poses.

Smoothing was stated before the first run: **one centered Savitzky–Golay pass,
window 5, polynomial 2**, with interpolation at segment edges, on translation and
continuous equivalent rotation vectors. Only contiguous valid segments are
smoothed; invalid fallbacks remain unsmoothed. Speed/rate gates cover **every
written pose**, including those invalid fallbacks. Mask usability is an
operational texture/stereo-support proxy (at least 30 non-hand eroded object
corners); sampled semantic inspection is reported separately.

The sole public scan use computes `scan_T_asset` once from the pan's oriented
bounding axes/center. The narrow end sets +x toward the handle, surface-area
occupancy sets the upward thin axis, and +y completes a proper rotation. No scan
size, geometry or motion is used to fit or select the reference. The same JSON
constant is copied byte-identically to 38. The output pose is
`world_T_asset @ inverse(scan_T_asset)`, so the kit can apply its scan in its own
body convention. This use must be disclosed in the report.

The zero-action check uses the evaluator's CPU composition for one second at
20 Hz, with frame-0 reference held fixed, zero actions and no resets. Our own
mesh/URDF is supported by a z=0 cylinder fixture; the shipped robot hand template
is placed 5 m away and above, with no contact labels. This is an isolated support
check, not a reconstruction of either episode's hands and not an RL rollout.

Each `epNN/<attempt>/` writes `request.json` with all direct input/source hashes,
`world.json`, `initialization.json`, per-frame camera/object diagnostics,
`tracking.npz`, our `canonical-asset.obj`, `support/` with evaluator input/logs and
measurements, `body-convention.json`, `reference.npz`, `reprojection.npz`,
`gates.json`, `overlay-strip.png`, and an artifact-hashed `run.json`.
The reference, gates, body convention and overlay are also copied to `epNN/`.
`reference.npz` has metric `position`, kit-order `quaternion`/`quaternion_xyzw`,
`frame_index`, video-PTS `time`, Boolean `valid`, `world_T_object`, `pose` (xyz +
xyzw), and scalar `gates_passed`. Frame identity, rather than a label clock, is
used for the sealed scoring hand-off. A failing reference is still written.

The overlay has 12 evenly spaced frame columns, raw video above and projected
reference below. Green outlines are the observed pan mask, orange outlines are
our mesh, and blue shading marks excluded hands. INVALID labels are retained.
`reprojection.npz` uses NaN on invalid frames, and reprojection coverage is gated
separately, so good-looking valid snippets cannot establish whole-clip success.

### Candidate-A result, 4 November 2026

**FAIL on 39; FAIL on 38.** Each passes 11 of the 23 tabulated checks. Full gate
values and limits are in `data/outputs/moving_reference/ep39/results.md` and each
episode's root `gates.json`. The same frozen code ran 38 exactly once, after the
07:46:31 UTC freeze. No label or mocap scoring ran. Neither reference is admitted
as a policy target.

| Measurement | 39 | 38 | Frozen requirement |
|---|---:|---:|---:|
| Frames written / valid | 319 / 51 | 263 / 49 | ≥90% valid reprojection coverage |
| Table RMS (mm) | 3.46 PASS | 3.95 PASS | ≤5 |
| Disjoint table normal disagreement (deg) | 6.54 FAIL | 4.03 FAIL | ≤2 |
| Held-out camera median / p95 (px at 640 width) | 0.942 / 7.181 FAIL | 0.804 / 6.901 FAIL | ≤0.333 / 1 |
| Held-out world residual median (mm) | 36.6 FAIL | 29.3 FAIL | ≤10 |
| Camera coverage | 56.1% FAIL | 62.7% FAIL | ≥98% |
| ≥30 object correspondences | 81.8% FAIL | 86.7% FAIL | ≥90% |
| FB disagreement (mm / deg) | 4.25 / 0.501 PASS | 4.06 / 0.471 PASS | ≤10 / 5 |
| Speed at asset center (m/s) / angular rate (deg/s) | 0.832 / 223.8 PASS | 0.724 / 247.1 PASS | ≤1.5 / 540 |
| Reference reprojection IoU median / p10 | 0.493 FAIL / 0.481 PASS | 0.551 FAIL / 0.540 PASS | ≥0.60 / 0.30 |
| Reprojection coverage | 16.0% FAIL | 18.6% FAIL | ≥90% |
| Frame-0 penetration (mm) | 54.5 FAIL | 84.9 FAIL | ≤5 |
| CPU zero-action translation / rotation | 58.70 m / 146.5° FAIL | 0.536 m / 174.7° FAIL | ≤0.01 m / 2° |

The support simulation becomes unstable from the deeply penetrating initial
placement; its large displacement is a failed initialization diagnostic, **not**
a measured object motion in the video. It uses the actual evaluator composition,
VOC scale 0, CPU device, zero actions and no resets. The saved speed gate uses
our asset's bounding-center origin; the independent output audit also measures
the submitted scan-convention origin: maxima 0.846 and 0.727 m/s, both passing.
The origin distinction changes neither episode's speed pass/fail nor any pose.

The other passing checks on both are minimum held-out tracks (265/269), four
image regions, operational mask usability (97.2%/99.6%), longest unusable mask
interval (one frame each), and zero positive base gap. Zero gap does not rescue
the failing penetration. Initial opposite-heading hypotheses clear the fixed
0.02 IoU margin on both (0.0407/0.0835); this is algorithmic disambiguation, not
an independently labelled proof of orientation.

I viewed all twelve-frame strips and enlarged frame pairs on both episodes,
plus the diagnostic graphs. On 39 the mesh is oversized at frame 0; after lifting,
it drifts and tilts away from the green pan boundary, especially sampled frames
87 and 202. On 38 the corrected green mask follows the visible pan in the sampled
frames, but the orange mesh extends too far and its angle/edge alignment is wrong
under gripping, clearly at sampled frames 71 and 167. The handle boundary remains
occluded by hands. These are visual judgments on twelve samples each; there is
no semantic ground-truth segmentation measurement. Keyframe refinement first
fails at frames 56/50 (2.80/2.50 s), and never recovers afterward.

Audits: all recorded input/source/artifact hashes match; frame indices and PTS
are intact; all written poses are finite; quaternions have maximum norm error
2.22e-16; reconstructing `world_T_asset` from the reference and body constant
agrees to 2.22e-16. The body convention JSON is byte-identical on both episodes.
The first completed 39 track and the final 39 track agree exactly. Direct
rectification equivalence gives zero maximum difference for K, Q and maps;
Kabsch versus SciPy differs by at most 4.44e-16 on 50 real stereo correspondences.
Python compilation and all six CPU imports passed. No test suite ran.

The first 39 attempt failed before rectification because OpenCV required its
translation as contiguous 3x1. Its failed record is preserved. The second attempt
completed tracking; the third corrected speed/rate gate aggregation to include
invalid written fallbacks; the final attempt added the available CPU evaluator
support check. These changes did not move thresholds or alter the tracking
poses. All attempts and exact commands are preserved in the episode's ledger
and `commands-run.txt`. No source changed after the freeze.

Artifacts for review:

- `ep39/reference.npz`, SHA-256 `9310781d072d25ea87361757dddb4ea8cdb12af297fcd7e2ca2d0560837d5e1b`.
- `ep38/reference.npz`, SHA-256 `7592ccb5456043006e38e91ceb98a033a9d3f0e58573129fc74846f7e6c3be55`.
- `epNN/overlay-strip.png`, `overlay-detail.png`, `tracking-diagnostics.png`,
  `output-audit.json`, and `visual-inspection.json`.
- Final producer directories: `ep39/candidate-a-cpu-support/` (162.13 s) and
  `ep38/candidate-a/` (135.12 s).
- `ep39/code-freeze.json`, `equivalence.py`, `equivalence.json`,
  `audit-and-report.py`, `results.md`, `commands-run.txt`, `files-written.txt`.

The lead owns review, commit, the separate chain outcome/report documents and
any sealed scoring decision. Candidate B, RL training and uploads were outside
this CPU job. The sealed 39 labels remain unread by this job.

## Candidate C: historical placement-only result (08:16 UTC)

**C fails the measured 39 support rotation check; camera/object tracking and 38
were not run.** This follows the ordered stop rule in the method-revision brief.
A's source modules and `candidate-a*` outputs were unchanged. At that point no C
full-clip reference was written. The completed continuation below supersedes this stop.

Build/resume C's frame-0 geometry and its own-hull S2 check:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.candidate_c asset
```

The command reads only 39 frame 0 for fitting. It measures the silhouette's
handle-hole center, heading, planar dimensions, width profile, head/neck and
handle-ring dimensions. A refitted `thin_tool_profile` and a watertight mask
extrusion are compared by frame-0 IoU; the closer fit is selected. The same asset
is intended for 38. Its thickness is the 25th percentile of eroded stereo point
heights above the table, **a floor-height proxy rather than an observed underside**.
The canonical asset origin is its planar bounding center at half thickness;
the resting pose constrains its base to table z=0, without inheriting world x as
its heading. The pan scan is loaded once solely to re-derive the same bounding
axis/origin body convention; the old and new transforms are recorded and agree.

The earlier brief gated this camera-only command on placement; the continuation
brief removes that restriction:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.candidate_c camera
```

**At that stage this operation had not run.** KLT, sparse BA, table refinement,
model tracking, freezing and 38 were subsequently completed in the continuation below.

The placement defect in A was an unconstrained height adjustment: its silhouette
optimizer could lower the oversized mesh by 6 cm to make the projection smaller.
C derives metric footprint from calibrated rays intersecting the table-parallel
plane, and fixes contact height. Its 39-only measurements yield 0.18176 ×
0.16621 m, with a 6.628 mm extrusion. The ring hole is retained. The selected mesh
has 2744 vertices and 5488 faces and is watertight.

| Measured gate | Frozen limit | A39 | C39 | A38 | C38 |
|---|---:|---:|---:|---:|---|
| Frame-0 base gap (mm) | ≤5 | 0 PASS | 0 PASS | 0 PASS | NOT RUN |
| Frame-0 penetration (mm) | ≤5 | 54.5 FAIL | numerical zero PASS | 84.9 FAIL | NOT RUN |
| Zero-action translation (mm) | ≤10 | 58695 FAIL | 4.920 PASS | 536.2 FAIL | NOT RUN |
| Zero-action rotation (deg) | ≤2 | 146.45 FAIL | 4.754 FAIL | 174.65 FAIL | NOT RUN |

`ep39/candidate-c/results.md` tabulates every original gate alongside A39/A38;
unreached C gates are explicitly NOT RUN. C's frame-0 IoU improves from 0.4156
for the original mesh resting flat to 0.6682 for the measured profile and 0.9105
for the selected mask extrusion. This is a fit on that same frame, not a passing
whole-clip reprojection gate. The initial table is still A's frame-0 definition
(3.46 mm RMS, 6.54° patch disagreement); the scheduled multi-frame table revision
was not reached and is not claimed as passing.

The support check uses **this task's eight imported Newton/CoACD hulls**, a
footprint-covering cylinder exactly 1 mm below their lowest vertex, mass 0.3 kg
and diagonal inertia 0.001 kg m². No M2 pose, fixture placement or collision hull
is reused. Shipped template hands are translated clear, contact references are
zero, the evaluator runs on CPU with VOC scale 0 and zero actions, and no resets
occur. The prospectively retained horizon is one second/20 steps. Its maximum
rotation through step 19 is 0.721°; step 20 reaches 4.754°. The gate **fails** over
the full horizon. The protocol states 1 cm/2° limits without a duration, so this
gate is not fully specified; no shorter horizon is substituted to claim a pass.

I viewed `asset/A-versus-C-frame0.png`, the full-frame C overlay and
`support-diagnostic.png`. C's outer footprint fits much better, but the projected
hole is slightly shifted/narrowed and the uniform thin extrusion flattens the
visible raised walls. No motion/occlusion performance was measured or viewed.

Provenance caveat: reading the named S2 `candidate.json` for setup also exposed
pre-existing M2-derived motion/grasp metadata. No episode parquet label file was
opened and none of those fields were used. Subsequent S2 reads were limited to
its URDF and implementation. This incidental exposure is recorded in
`output-audit.json` for the lead's research-integrity review.

The first C support worker failed because the isolated FlashCHORD interpreter
has no OpenCV. Imports were made lazy; nothing was installed. That failed log is
preserved, and the corrected real CPU run produced the failure above. Compilation
and CPU imports of all three new modules passed. No test suite or camera run ran.
The final audit independently recomputes support motion, checks watertightness,
base/table alignment, imported-hull clearance and input hashes, and verifies A's
frozen sources remain unchanged. Artifacts and exact commands are inventoried
inside `ep39/candidate-c/`; nothing in A's run directories was rewritten.

## Candidate C: completed continuation, frozen 08:39:52 UTC

**39 FAIL: 16/23 checks pass. 38 FAIL: 17/23 checks pass.** Both references are
written with `gates_passed=false`; neither is admitted for policy training.
Per-frame `valid` means an observed camera/model estimate, not global admission.
The labelled overlays make this distinction explicit. No source module changed
after the C freeze; 38 ran once. No labels or mocap were read or scored during
this continuation, and no M2 task or motion fields were consumed.

Run/resume the fixed two-item CPU support queue (S2, then exactly one S3):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.support_drop run
```

Run the full ordered 39 pipeline; support failure is recorded and does not stop it:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.full_c run --episode 39
```

Freeze C's source hashes, settings, reference, visual asset and body convention:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.full_c freeze
```

Run 38 once with those exact sources/settings and the 39-only asset:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.full_c run --episode 38
```

The handler refuses changed frozen code on 38 and writes `once.json` before work.
Cached stages avoid repeating finished work. An initial 35-evaluation BA and its
full 39 output are preserved under `initial-35-evaluations/`; the final numerical
budget is 200 evaluations on both episodes. Both hit that cap without convergence.
The early OpenCV failures, rectification/mask identities and command logs are
retained. The 39 post-processing reruns corrected complete-triangle reprojection
and speed at the submitted body origin; they reuse the fitted tracks, do not move
a threshold, and precede the freeze.

`camera_ba.py` tracks a dense 16-pixel grid, using KLT forward/backward checks,
calibrated LR-consistent stereo, and subpixel stereo LK where available. The
fallback remains calibrated stereo, never monocular depth. Empty KLT tracks are
handled explicitly. A deterministic quarter of grid tracks is held out at birth;
its observations never enter PnP initialisation or BA. Training-track RANSAC
outliers define moving-region exclusions, including the person at the left.
BA optimises all frame poses and up to 1400 training landmarks with a sparse
Jacobian, Huber loss and left/right pinhole residuals in metric scale. First camera
pose fixes the gauge. Held-out metrics use the unchanged `gates.py` definitions.

The table uses at most the first 12 frames, stopping at first dilated hand/pan
intersection. The original four image-patch predicates and their size are kept.
The pooled plane defines z and a rigid gauge update applied to all camera poses.
Frame-0 placement is independently recovered on each episode with the fixed 39
asset resting flat; it does not copy a world-axis heading or an old trajectory.

`model_track.py` fits SE(3) from previous world pose and KLT motion prediction,
using two-way distance-transform silhouette chamfer outside hands and robust
stereo-to-sampled-own-surface distance. Weak fits use additional 180° starts about
the long and normal axes. Forward and backward passes are compared at each frame.
Only one five-frame, order-two Savitzky–Golay smoothing pass is applied. Final IoU
uses every triangle of the unchanged visual mesh; the cap-outline approximation
is used only inside the bounded optimiser. The independent audit reproduces the
exact IoU on twelve real frames per episode with zero difference.

| Measurement | A39 | C39 | A38 | C38 |
|---|---:|---:|---:|---:|
| Camera coverage | 56.1% FAIL | 100% PASS | 62.7% FAIL | 100% PASS |
| Held-out median / p95 px at 640 width | .942 / 7.181 FAIL | .359 / 1.641 FAIL | .804 / 6.901 FAIL | .312 PASS / 1.421 FAIL |
| Held-out world residual mm | 36.6 FAIL | 25.4 FAIL | 29.3 FAIL | 19.5 FAIL |
| Table patch disagreement ° | 6.54 FAIL | 2.286 FAIL | 4.03 FAIL | 2.634 FAIL |
| Original ≥30 KLT correspondence coverage | 81.8% FAIL | 81.8% FAIL | 86.7% FAIL | 86.7% FAIL |
| C ≥30 stereo-pixel coverage, separate diagnostic | — | 100% | — | 100% |
| FB translation mm / rotation ° | 4.25 / .501 PASS | .703 / .697 PASS | 4.06 / .471 PASS | .415 / .407 PASS |
| Mesh IoU median / p10 | .493 FAIL / .481 PASS | .859 / .827 PASS | .551 FAIL / .540 PASS | .864 / .810 PASS |
| Reprojection coverage | 16.0% FAIL | 100% PASS | 18.6% FAIL | 100% PASS |
| Initial gap / penetration mm | 0 / 54.5 FAIL | .465 / 0 PASS | 0 / 84.9 FAIL | 0 / .673 PASS |

Every frozen threshold and pass/fail value appears in `epNN/candidate-c/gates.json`
and the full A/C table in `results.md`. The original 30-correspondence gate is
explicitly **A's KLT/RANSAC inlier count**. C's dense supported mask pixels are
reported separately and are not substituted into that gate. This gate remains
well-defined numerically but is specific to the feature-tracking method; both
counts are retained for the lead. A's archived speed is at its asset center;
C measures speed at the submitted body origin. Both pass their unchanged limit.

The two 60-step hands-cleared drops are both failures:

| Collision variant | Thickness | Hulls | Max translation | Max rotation | First >2° |
|---|---:|---:|---:|---:|---:|
| S2 | 6.628 mm | 8 | 1.876 m FAIL | 177.18° FAIL | step 20, 1.00 s |
| S3, exactly one change | ≥10 mm | 12 | .2023 m FAIL | 168.45° FAIL | step 6, .30 s |

Both use unedited CoACD, 0.3 kg, diagonal 0.001 inertia, one-mm hull clearance,
zero actions and no resets, with 81 stationary reference frames to cover the full
60-step horizon. S3 changes only z thickness of the collision proxy; the tracking
and exported visual mesh hash is unchanged. The isolated drop importer uses the
collision proxy file; no thicker shape is rendered into the video reference.
The gate table records the shared S3 asset drop on both episodes, not an extra
38 support trial. S2 is also reported in full. Nothing was iterated beyond these
two queued drops.

The first S2 jump is **tilt-dominated**, axis [-.379, -.925, -.022] in world xyz.
Own hull shape 110 rises from 691 N to 1155 N peak contact force at that interval;
shape 114 peaks at 1313 N. S3's first failed rotation is also tilt-dominated;
contact is lost at that control step. `support-drop-summary.json` retains per-hull
forces around the first failure and the largest later rotation jump;
`support-drop-comparison.png` shows rotation/tilt, displacement and hull-force
heatmaps. These large numerical excursions are failed support diagnostics,
not motion measured in the video.

I viewed both full twelve-frame strips, enlarged samples, tracking plots and
world top/side trajectories. The reconstructed outline follows lifting and
rotation much better than A. On 39 the grip/ring around sampled frames 87 and
202 remains occluded, and the low walls are flattened. On 38 the frame-0 fit is
weaker than the later samples, the rim/handle at 71/167 remains uncertain under
the hand, and the late returned pan has a small boundary mismatch. No sampled
identity switch or gross off-object drift is apparent. High silhouette IoU and
small forward/backward error do **not** prove world pose accuracy; camera gates
still fail. Each trajectory uses its own table/initial-view world gauge, with
video time colours and the submitted body origin, and contains no mocap curve.

Final reference SHA-256:

- 39: `46a6fc97eb42c09e5611cd4853455f55e730564b6269ad22b1e594a97a34b56e`.
- 38: `1c84c24447e3c901ab03315a733c901a2a7860fb872676214f92d468245a58cb`.

The independent campaign audit finds no input/source hash mismatch, verifies the
same frozen code/body/asset on 38, exact PTS and frame identities, finite unit
quaternions, body-convention composition to 2.22e-16, and exactly one smoothing
pass. Independent gate aggregates agree within 1.4e-12. All five C modules compile
and CPU-import. No test suite, GPU, install, upload, mutating git or label scoring
ran. The earlier placement-only report is preserved in `placement-only-report/`;
its script is historical and must not be used to overwrite the completed run.

## Amendment 1, 09:15 UTC hand-off

The protocol amendment is explicit and was not edited. The frozen `gates.json`
files remain byte-for-byte intact. `gates-amended.json` keeps their values beside
the new instruments and thresholds. **Both episodes still FAIL.**

| Rule/instrument | 39 | 38 |
|---|---:|---:|
| Original frozen rule | FAIL, 16/23 pass | FAIL, 17/23 pass |
| Amended rule | FAIL, 22/28 pass | FAIL, 24/28 pass |
| Held-out median / p95, px | .359 / 1.641 PASS | .312 / 1.421 PASS |
| Patch normal, <=3 degrees | 2.286 PASS | 2.634 PASS |
| Table max frame-median offset, <=10 mm | 92.62 FAIL | 93.46 FAIL |
| Table p95 frame medians, <=15 mm | 79.31 FAIL | 82.25 FAIL |
| Late table normal, <=2 degrees | 3.46 FAIL | 49.80 FAIL |
| Table/landmark usable coverage, >=80% | 72.41% FAIL | 91.63% PASS |
| Long-baseline max median / p95, <=10/20 mm | 6.29 / 4.63 PASS | 5.22 / 4.23 PASS |
| Stereo-pixel coverage, >=90% | 100% PASS | 100% PASS |
| Old KLT count retained beside new | 81.8% | 86.7% |

The noise-floor control does **not** explain the original 3D residual: near-static
same-track controls are 15.21/12.53 mm, versus old 25.37/19.46 mm. The original
instrument therefore stands and fails. The perpendicular number is recorded but
is not used to claim a pass. Near-static means <=1 mm camera translation and
<=0.2 degrees rotation between consecutive frames, fixed before measurement.
The conservative explanation rule and all control frame identities are recorded.

No camera change was retained. On 39 only 21.3% of the high-error tail lies in the
outer image tenth, versus 17.1% of all tracks; its median location is [.481,.441],
depth 1.03 m and age 33 frames. That is not an isolated edge/calibration defect
justifying a correction. Held-out selection and camera matrices are unchanged.

The independent table region is a fixed frame-0 image ellipse centered at
[.52W,.72H], radii [.17W,.25H], excluding the 21-pixel-dilated frame-0 pan/hands.
Training-only points define its frame-0 plane; held-out IDs are lifted at first
sight, never re-anchored, and projected back to that image-region rule. No pan
pixels enter the table checks. Too few points is a failure. The fixed disc/cloth
region does not supply a stable plane here; the failed offsets/normals are not
presented as a precise measurement of camera error alone.

The body constant **was wrong**: asset coordinates were treated as bounding
axes. The orthographic overlay exposed a 12.3407-degree in-plane offset. The
near-equal planar bounding dimensions are disambiguated by the asset's handle
hole, and the known construction +z avoids the symmetric extrusion's ambiguous
area-median sign. The scan floor sign itself is correct: the interior floor lies
below the rim. The corrected bounding-frame transform and both old/new values
are in `asset/body-convention.json`; `asset/body-overlay.png` shows labelled top,
side and end views and was inspected. Both references were re-expressed under
this fixed transform; geometry, trajectories and camera were not refitted.
Old references are preserved as `reference-before-amendment.npz`.

Run 39 amended diagnostics: `PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.amendment run --episode 39`.
Freeze: `PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.amendment freeze`.
Run 38 once: `PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.amendment run --episode 38`.
The amended freeze precedes the single amended 38 diagnostic pass. No label
scoring ran. `results-amended.md` lists every amended gate and final hashes.
Support is retained as a diagnostic and excluded from amended admission.


### Table instrument audit after sealing

The table instrument had contamination and fitting defects. Its fixed ellipse included the lap below the table; its SVD/clipping fit produced frame-0 normals 63.21 degrees (39) and 48.80 degrees (38) away from world up. RANSAC on frame-0 training tracks identifies a table footprint, carried by a plane-induced image mapping. Held-out errors never select members. The 15 mm RANSAC tolerance is inherited from the original estimator, fixed for both clips before corrected measurements; all gate limits remain unchanged. The late plane uses the same RANSAC rule. References, camera poses, old gates and masks are unchanged.

| Instrument | Limit | 39 before → after | 38 before → after |
|---|---:|---:|---:|
| Maximum frame median offset | 10 mm | 92.62 → 4.03 | 93.46 → 3.54 |
| p95 frame median offset | 15 mm | 79.31 → 3.22 | 82.25 → 3.19 |
| Late normal change | 2 degrees | 3.460 → 0.409 | 49.801 → 0.621 |
| Table / landmark coverage | at least 80% | 72.41 → **23.51 FAIL** | 91.63 → **33.08 FAIL** |
| Maximum in-plane frame median | 10 mm | 6.29 → 2.28 | 5.22 → 2.47 |

**Both still fail: 25/28 gates each.** Only 75/319 and 87/263 frames retain at least 30 table tracks. Those sparse subsets do not establish full-clip camera accuracy. The original world residual remains failed. No conditional rule based on labels was applied. With all original contaminated observations retained, robust fitting alone gives late normal changes 0.287/0.587 degrees, while maximum offset remains 44.06/36.78 mm. This isolates both fitting and contamination effects. Old offset calculations reproduce exactly (0 m maximum difference).

Batch audit, 39 then 38, refusing to overwrite existing audit results:

```bash
MPLCONFIGDIR=data/outputs/moving_reference/ep39/candidate-c/figure-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.table_audit
```

Independent real-data equivalence/control:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-audit-verify.py
```

Report generation:

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-audit-report.py
```

Outputs in each candidate-c directory: `gates-amended-2.json`, `table-drift-before-after.png`, `table-points-audit.png`, `table-region-corrected.png`, `table-audit-points.npz`, `table-audit-equivalence.json`, `run-table-audit.json`. Full interpretation, commands and hashes: episode 39's `results-table-audit.md`. Early diagnostic outputs remain under `table-audit-initial/`. The inspected overlays show lap and table-edge contamination; grip/arm boundaries remain imperfect, particularly 39 frame 60. Coverage is reported as failure rather than enlarging the region or adding tracks after measurement.


### Independent dense table measurement, one attempt

The same fixed code ran once on 39, then 38: an 8-pixel KLT grid seeded every 5 frames, avoiding active tracks within 4 pixels. The exact corrected footprint, original 21-pixel exclusions and forearm rays, forward/backward flow and stereo rules are retained. These new tracks never enter the frozen camera fit. First-sight stereo anchors persist for each ID through every seeding event; dead IDs are not revived. No cadence, mask, footprint or threshold adjustment followed either result.

| Measure | Limit | 39 previous → fresh | 38 previous → fresh |
|---|---:|---:|---:|
| Maximum frame-median offset | 10 mm | 4.032 → 4.815 | 3.536 → 3.182 |
| p95 frame-median offset | 15 mm | 3.220 → 3.217 | 3.195 → 2.830 |
| Late normal change | 2 degrees | 0.409 → 0.204 | 0.621 → 0.494 |
| Table and landmark coverage | at least 80% | 23.51 → **98.75% PASS** | 33.08 → **100% PASS** |
| Maximum in-plane median | 10 mm | 2.279 → 8.010 | 2.471 → 8.188 |
| p95 in-plane medians | 20 mm | 1.786 → 5.195 | 1.876 → 5.939 |

**All table and landmark gates pass on both clips. Complete records remain FAIL, 27/28 each, solely on the unchanged world-residual gate.** The references, camera poses and all older gate records are byte-for-byte unchanged. No conditional label-based rule was applied here.

39 has 315/319 covered frames; 69, 89, 108 and 109 remain uncovered, with 14/27/20/16 observations. They stay in the denominator. 38 covers all 263 frames, minimum 176 observations. No frame has fewer than 30 visible grid candidates under the unchanged masks; 39's remaining misses reflect temporal/stereo attrition between fixed seed frames. The lowest unmasked footprint fraction is 19.77% on 39 and 63.87% on 38. Grip/wrist segmentation is imperfect, so mask-based capacity is not exact physical visibility.

The independent real-data audit found zero first-sight anchor error, no revived/reset IDs, and at most 4.34e-19 m difference when recomputing the metrics. Observation ages have median/p95 13/59 frames (39), 17/91 (38); 62 and 195 landmarks last at least 100 frame intervals. Fresh births count in the unchanged statistic, so this does not claim every track spans the clip.

Fixed queue batch (skips matching completed records; never retries a started incomplete attempt):

```bash
MPLCONFIGDIR=data/outputs/moving_reference/ep39/candidate-c/figure-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.table_tracks --queue data/outputs/moving_reference/ep39/candidate-c/table-tracks-queue.json
```

Anchor and metric audit:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-tracks-verify.py
```

Report generation:

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-tracks-report.py
```

Each episode's `candidate-c/` contains `gates-amended-3.json`, `table-tracks-observations.npz`, `table-tracks-inputs.json`, `table-tracks-overlay.png`, `table-tracks-worst.png`, `table-tracks-diagnostic.png`, `table-tracks-verification.json`, and `run-table-tracks.json`. Episode 39 also holds the queue, ledger, summary and `results-table-tracks.md` with commands, reference hashes and the inspected overlay findings. No camera fit or reference generation is part of this operation.
