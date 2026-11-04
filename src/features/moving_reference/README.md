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
