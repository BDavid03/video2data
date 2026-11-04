# Spoon admission, episodes 35–37

The fixed rules are in `docs/chain/moving-spoon-protocol.md`. The ordered queue is
`spoon_queue.json`; the handler is `spoon.py`. Existing dust-pan modules and historical
38/39 records are left unchanged. No spoon labels or scans are read.

Run from the checkout with the project interpreter:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.spoon_queue
```

The queue skips recorded failed episodes, never retries a measured miss, appends
`data/outputs/moving_policy/spoon-ledger.jsonl`, and publishes events immediately to
`data/outputs/moving_policy/spoon-status.md`. A passing mask screen yields to the camera
phase before any later episode starts. It never launches training by itself.

The object-name prompt is `spoon`. The existing DINO/SAM2 workers, thresholds, smallest
qualifying box rule, hand cadence, propagation and GPU lease are reused. `pan` in mask
worker record keys is a legacy slot name for object track zero. It does not change the
prompt or the object. CPU work runs under `nice`.

Before a camera solve, the screen measures existing final M1 mask gates: at least 90%
of frames have at least 30 stereo-supported texture points in the eroded, hand-excluded
object mask, with no unusable interval longer than two frames. The amended object
correspondence gate counts stereo pixels at the same 30 / 90% limits. These are exactly
`gates.evaluate` and `gates.stereo_pixel_gate` definitions. Nonempty masks alone do not
meet them. A miss stops that episode immediately, keeping v5; every later gate is
explicitly recorded as unmeasured, with its unchanged limit. No camera, tracker, Stage 0,
policy or pack is run after that stop.

The D1 spoon PLY files for 35, 36 and 37 are byte-identical:
`28c4f67eecd0296c898dc63b3ec312d03381ee47ba7e44aa310a49eb99e9c48d`.
The preserved copy and provenance are at `data/outputs/moving_reference/ep35/asset/`.
It retains its own body frame, 634 vertices, 1,260 faces and extents
0.469352 × 0.072401 × 0.033261 m. S2's 0.3 kg and diagonal 0.001 kg m² are modelling
constants, not measured spoon dynamics. No simulation occurs unless admission passes.

All three episodes stopped at the unchanged mask gates:

| Episode | Usable frames (required 90%) | Longest unusable interval (limit 2) |
| --- | --- | ---: |
| 35 | 46/421, 10.9264% | 94 |
| 36 | 36/407, 8.8452% | 83 |
| 37 | 1/151, 0.6623% | 102 |

Stereo pixel support was 100% on each clip. The masks followed the spoon on episode 35
in the 12-frame contact sheet, a visual observation without pixel ground truth. Gate
measurements and later episode results are in each `epNN/mask-gates.json`,
`epNN/gates-final.json`, and `moving_policy/epNN-v1/result.md`.

One code defect was corrected before any gate measurement: the new input selector
joined a legacy cache path directly. It now uses `src.paths.project_path`, matching
the existing handlers. The first failed path-resolution marker and traceback are
preserved; no gate was retried. Details: `ep35/path-resolution-defect.json` and
`moving_policy/spoon-queue.log`. The mask handler had CPU entry additions while the
initial fixed workers ran; that source-identity detail is disclosed in the same file.

Verification uses syntax compilation, CPU imports and an equivalence script on the
real measured counts at `data/.runtime/tmp/spoon-equivalence/check.py`. Its report is
`data/outputs/moving_policy/spoon-equivalence.json`. It also rehashes 41 preserved
source/scientific files. No test suite, install, download, label access or upload runs.
