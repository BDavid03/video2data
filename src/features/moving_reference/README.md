# Moving reference

Builds the object path a moving policy follows, for tuning episode 39 and evaluation episode 38. The tracker reads no scan, but the body convention reads the public scan's axes once, disclosed. It produces masks, a stereo camera track, a model-based object track, and a
gate record for each candidate. Protocols and results:
[moving-m1-protocol.md](../../../docs/chain/moving-m1-protocol.md),
[moving-m1-sealed.md](../../../docs/chain/moving-m1-sealed.md),
[moving-m1-outcome.md](../../../docs/chain/moving-m1-outcome.md). Output is under
`data/outputs/moving_reference/epNN/`.

Short version: candidates A and B failed. Candidate C failed its first frozen checks and Amendment 1, which had already moved three limits (held-out median 1/3 to 0.5 px, p95 1 to 2 px, patch normal 2 to 3 degrees) and changed instruments. It passes 28 of 28 only in the final record, after the table check was repaired and re-measured and a conditional rule replaced the world-residual measurement. That rule was written after seeing both clips' gates and before any label, so this is weaker than a prospective validation. High silhouette overlap and small forward/backward error do not show world pose accuracy.

## Masks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference --episode 38
```

Fixed Grounding DINO and SAM2.1 Hiera-Large queue: 39, then 38. It reads only `ego_cam_c` videos
and clock inventories, on the raw distorted C pixel grid, resized to a maximum side of 1024.
Consumers must transform the masks when they rectify. `runtime.json` freezes the rules: pan prompts `dust pan`,
`yellow dustpan`, `scoop` at frame 0, stopping at first success; box and text thresholds 0.40 and 0.30;
hands detected every 10 frames with anonymous slots. There is no per-episode tuning. Hands
recovered by reverse propagation are model predictions and not anatomical identities. All GPU work is
sequential under `src.gpu.GpuLease`.

Each `masks/` holds `request.json`, `detections.json`, `frames/frame_XXXXXX.npz` (`masks` in order
pan, hand_0, hand_1, plus clock and direction fields), `summary.json`, `contact-sheet.png`, `run.json`
and `ledger.jsonl`. Resume checks inputs, weights, settings and sources.

| Episode | Frames | Pan area min / median / max (px) | Finding |
| --- | ---: | ---: | --- |
| 39 | 319 | 5446 / 9431 / 16419 | Pan body holds in the 12 inspected samples; the gripping hand hides the handle |
| 38 | 263 | 62577 / 75658 / 83651 | FAIL: the mask tracks the tabletop |

Do not use track 0 in `ep38/masks/` as an object mask. Candidate A onward admits 38 only from
`masks-smallest/summary.json`. "Coverage" means a nonempty mask, and these are visual judgments on
sampled frames, with no pixel ground truth. The first queue run refused 38 on a memory allowance
(11000 MiB plus 1024 MiB headroom), corrected to 10000 MiB. That attempt is in
`ep38/masks/admission-refused/`.

## Candidate A: stereo tracker (failed)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference track --episode 39 --attempt candidate-a-cpu-support
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference freeze
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference track --episode 38 --attempt candidate-a
```

CPU only. Reads C and B videos, calibration, masks, clocks and our `thin_tool_profile` shape, with no
labels, mocap or earlier camera tracks. World z is the table normal, assuming a level table. A
quarter of ORB landmarks are held out for checking. Smoothing is one Savitzky-Golay pass, window 5,
order 2. The only public scan use computes `scan_T_asset` once from the pan's oriented bounding axes,
so the kit can apply its scan in its own body convention; this must be disclosed in the report.

`camera.py` is geometry and tracking, `_object.py` pose and smoothing maths, `gates.py` the limits,
`track.py` I/O, and `support.py` the evaluator's CPU physics. Each attempt writes `request.json`,
`world.json`, `tracking.npz`, `reference.npz` (metric position, kit-order quaternions, frame index,
PTS, `valid`, `gates_passed`), `gates.json`, `overlay-strip.png` and `run.json`. A failing reference
is still written. Episode 38 runs once; its marker is written first, so a failure cannot be retried.

Result: FAIL on both, 11 of 23 checks each. Neither reference is admitted. Valid frames 51/319 and
49/263. Reprojection coverage 16.0% and 18.6%. Frame-0 penetration 54.5 and 84.9 mm. The support
simulation's large excursions come from the bad start and are not motion in the video. Keyframe
refinement fails at frames 56 and 50 and never recovers. Details: `ep39/results.md`, `gates.json`.

## Candidate C: refit asset, bundle adjustment, model tracking

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.candidate_c asset
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.candidate_c camera
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.support_drop run
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.full_c run --episode 39
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.full_c freeze
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.full_c run --episode 38
```

The asset reads only episode 39's frame 0 for fitting, and the same asset is used on 38. It is 0.18176
by 0.16621 m with a 6.628 mm mask extrusion. Thickness is a floor-height proxy, without an observed
underside. The pan scan is loaded once to re-derive the body convention. Its first support check failed
on rotation (4.754 degrees, limit 2) and stopped the run under the stop rule. A follow-up brief removed
that stop; the earlier report is
preserved in `placement-only-report/`, and that script must not be rerun over the completed output.

`camera_ba.py` tracks a dense 16-pixel grid with KLT and calibrated stereo and holds a quarter of
tracks out, then runs bundle adjustment over all frame poses and up to 1400 landmarks. Both episodes
hit the 200-evaluation cap without converging. `model_track.py` fits SE(3) per frame with silhouette
chamfer and stereo distance. Forward and backward passes are compared. Smoothing is one five-frame
Savitzky-Golay pass. Episode 38 runs once, refuses changed code, and writes `once.json` first.

Provenance note: reading the S2 `candidate.json` during setup also exposed M2-derived motion and grasp
metadata. No episode label file was opened and those fields were not used. This is recorded in
`output-audit.json`.

| Measurement | A39 | C39 | A38 | C38 |
| --- | ---: | ---: | ---: | ---: |
| Camera coverage | 56.1% FAIL | 100% | 62.7% FAIL | 100% |
| Held-out median / p95 px at 640 width | 0.942 / 7.181 FAIL | 0.359 / 1.641 FAIL | 0.804 / 6.901 FAIL | 0.312 / 1.421 FAIL |
| Held-out world residual mm | 36.6 FAIL | 25.4 FAIL | 29.3 FAIL | 19.5 FAIL |
| Table patch disagreement degrees | 6.54 FAIL | 2.286 FAIL | 4.03 FAIL | 2.634 FAIL |
| Mesh IoU median / p10 | 0.493 FAIL / 0.481 | 0.859 / 0.827 | 0.551 FAIL / 0.540 | 0.864 / 0.810 |
| Reprojection coverage | 16.0% FAIL | 100% | 18.6% FAIL | 100% |

C failed its frozen checks, 16/23 on 39 and 17/23 on 38. Both 60-step hands-cleared drops failed: S2 1.876 m
and 177.18 degrees, S3 0.2023 m and 168.45 degrees. These are failed support diagnostics and not video
motion. C's outlines follow lifting and rotation better than A's. The grip and ring remain occluded,
low walls are flattened, and 38's frame-0 fit is weaker.

## Later records on C

Camera poses are unchanged by these steps, and each older gate file is kept. Amendment 1 re-expressed both references under the corrected body constant without refitting, and nothing after sealing changes them.

Amendment 1 (`gates-amended.json`): FAIL, 22/28 on 39 and 24/28 on 38. The near-static noise-floor
control (15.21 and 12.53 mm) does not explain the world residual (25.37 and 19.46 mm), so that failure
stands. The body constant was wrong by 12.3407 degrees in plane and by an origin offset, and was corrected; old references are kept as `reference-before-amendment.npz`. The constant still comes from the public scan, and 39's overlay keeps an unresolved 2 cm ring offset.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.amendment run --episode 39
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.amendment freeze
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.amendment run --episode 38
```

Table audit (`gates-amended-2.json`): the fixed ellipse included the lap and the plane fit had defects.
After correction, table offsets pass but coverage fails (23.51% and 33.08%). 25/28 each.

```bash
MPLCONFIGDIR=data/outputs/moving_reference/ep39/candidate-c/figure-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.table_audit
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-audit-verify.py
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-audit-report.py
```

Dense table measurement, one attempt (`gates-amended-3.json`): an 8-pixel KLT grid seeded every 5 frames,
outside the camera fit. Coverage 98.75% and 100%. All table and landmark gates pass; 27/28 each, failing only
the unchanged world-residual gate. On 39, frames 69, 89, 108 and 109 stay uncovered. Masks at the grip and
wrist are imperfect.

```bash
MPLCONFIGDIR=data/outputs/moving_reference/ep39/candidate-c/figure-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.table_tracks --queue data/outputs/moving_reference/ep39/candidate-c/table-tracks-queue.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-tracks-verify.py
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/moving_reference/ep39/candidate-c/table-tracks-report.py
```

Final record (`gates-final.json`):

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.final_gates
```

This copies `gates-amended-3.json` and swaps the world-residual measurement for the perpendicular
residual at the same 10 mm limit, but only when `reference_passed` and `allowance_passed` are both true
in 39's sealed record. Otherwise it refuses. The 3D value and the failed noise-floor control stay in the
record. The one-line rule is clause 2 of `docs/chain/moving-m1-sealed.md`. Both final records pass 28 of 28,
with that substitution. Rerunning verifies and does not replace, and the packer rebuilds the decision from
its inputs.
