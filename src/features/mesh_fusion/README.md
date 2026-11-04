# F1 mesh fusion: Stage A

CPU-only, video-only feasibility for the white pot (public 42) and dish rack (public 12), under
`docs/chain/mesh-fusion-protocol.md` (277c7df). This operation never loads a scan, computes CD-O,
builds a triangle mesh or starts Stage B.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_fusion stage-a \
  --request src/features/mesh_fusion/queue.json
```

`queue.json` contains one request per object. Completed trials resume by request/source identity;
`data/outputs/mesh_fusion/ledger.jsonl` records each completed object and `summary.md` reports the two verdicts.
Every object and queue execution owns a `run.json`, hashed inputs/sources and hashed output inventory.
New stereo observations go through the unchanged CPU `stereo_depth.generate` handler and are cached by
single-frame request. No CUDA is required. All local caches go under `data/.runtime/tmp/mesh-fusion/cache/`.

## Method and fixed choices

Camera-A RGB is inspected every fifth frame and refined frame by frame around contact. The queue records
first possible contact and first clear contact separately and conservatively excludes the uncertain interval.
For public 12, the cut is the first scene hand-object contact (the blue cup beside the rack), not a later
contact with the rack. This keeps the entire background scene static. Candidate frames are 0/5/10/15 in both clips.

Backproject the cached opening clusters using calibrated stereo, fit a local metric table plane (MoGe supplies
only the normal orientation hint), and keep seed points >6 mm above it. The opening seed's 0.5–99.5 percentile
axis-aligned bounds grow by 30 mm. Later object points must lie inside those bounds, >6 mm above the plane,
and within 30 mm of the seed. This is a bounded 3D region, not a learned tracker.

Background ORB (6,000 features) excludes every opening foreground cluster's projected 3D box with 30 mm margin
and the lower 20% hand-entry strip. Two PnP passes update that projected exclusion. Reuse the camera-motion
feature/PnP functions; refine against opening background stereo using the existing trimmed ICP (10 rigid
iterations, 80% retained, 10 mm downsampling, <=12,000 points). No map growth or chained registration.

Reference landmark index modulo five == zero is held out before PnP. Its current matches are also excluded
from PnP, and 8 px patches around the validation correspondences are excluded from dense ICP. The independent
error is the median *untrimmed* Euclidean disagreement of these correspondences after refinement; at least ten
are needed. PnP acceptance uses >=30 RANSAC inliers, post-ICP median reprojection <=1 px on the original
<=3 px consensus, and independent disagreement <=3 mm. Paired Kabsch is recorded as a diagnostic only.

Angular span uses rays between accepted camera centres and the fixed opening object centre, not head rotation.
Additional surface is `(fused 3 mm voxels absent from opening)/(all fused 3 mm voxels)`, with a fixed opening-grid
origin. It is sensitive to depth noise, registration and sampling; it does not establish reconstructed unseen shape.
Only accepted frames enter fusion. Frame 0 is explicitly an identity reference, not an independent viewpoint.

Kill if fewer than three accepted frames, OR angle <10 degrees AND additional support <10%. The slice has no
Stage B implementation. A pass requires the lead to freeze one separate recipe before scans/scoring.

## Results

Both objects are **KILL** under the frozen protocol. Only frame 0 (identity) is accepted for each, out of
candidates 0/5/10/15. Accepted-view angular span and additional supported surface are both zero. The safe
intervals are [0,18) for public 12 and [0,19) for public 42, with first possible/clear contact recorded separately.

| Object | Later-frame PnP inliers (5/10/15) | Post-ICP reprojection px | Independent disagreement mm |
| --- | --- | --- | --- |
| Rack | 455 / 265 / 273 | 2.185 / 4.526 / 4.090 | 31.314 / 59.998 / 59.764 |
| Pot | 224 / 150 / 121 | 1.646 / 3.383 / 5.131 | 34.409 / 62.798 / 86.769 |

All later frames have enough PnP inliers but fail both the <=1 px and <=3 mm requirements. The opening cloud
contains 7,717 supported surface voxels for the rack and 5,157 for the pot. No independent additional view is
fused. This establishes failure of this registered-stereo adapter under F1; it does not establish zero true camera
motion or impossibility of reconstruction with another method. Both objects retain v4; Stage B is not proposed.

The initial execution had a coverage-accounting bug: the grown frame-0 cloud was compared against its smaller
mask seed and falsely counted opening-region growth as novelty. It also excluded only matched reserved landmarks
from reference ICP, rather than the entire reserved reference set. Both were corrected without changing thresholds
or registration parameters, and the complete queue reran under new source identities. Historical run records and
ledger rows remain intact; `data/outputs/mesh_fusion/corrections.md` identifies superseded and current runs.

See `data/outputs/mesh_fusion/summary.md` and the per-object summaries for figures, acceptance tables and run records.
`verification.json` records direct real-data recomputation and library equivalence: coverage 0% by an independent
voxel-set calculation, camera feature outputs exactly equal, ICP adapter/direct-call error 0, angular formula
agreement within 1.54e-13 degrees, and all 20 final per-object artifact hashes matching. All nine modules compile
and import on CPU. A second invocation reused both completed trial identities without adding ledger rows.

## F1b: PnP-only amendment 1

Amendment 1 (`3db3325`) changes only the pose refinement: `registration: "pnp"` skips ICP.
The request default is `"pnp_icp"` for F1. The original F1 records, ledger and root summary are
unchanged; F1b owns `data/outputs/mesh_fusion/f1b/`, including its own ledger and per-object records.
The existing torn-ledger recovery and `data/.runtime/tmp/mesh-fusion/cache/` path are retained.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_fusion stage-a \
  --request src/features/mesh_fusion/queue_f1b.json
```

The same frames 0/5/10/15 were rerun with the same masks, independent correspondences, region,
coverage definition and acceptance thresholds. All later PnP counts and raw PnP reprojection
statistics exactly match F1. Recomputed gate reprojection uses the unchanged PnP-consistent
training correspondence set, so it differs slightly from the PnP solver's own inlier median.

| Object / episode | Frame | Inliers | Gate reprojection px | Independent error mm | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| Rack / 12 | 0 | 1281 | 0.000000 | 0.000000 | yes (identity) |
| Rack / 12 | 5 | 455 | 0.681745 | 29.625675 | no |
| Rack / 12 | 10 | 265 | 0.944408 | 47.516520 | no |
| Rack / 12 | 15 | 273 | 0.906566 | 63.472658 | no |
| Pot / 42 | 0 | 888 | 0.000000 | 0.000000 | yes (identity) |
| Pot / 42 | 5 | 224 | 0.845983 | 37.513348 | no |
| Pot / 42 | 10 | 150 | 0.898393 | 60.583652 | no |
| Pot / 42 | 15 | 121 | 0.940500 | 79.271350 | no |

Every later frame passes the inlier/reprojection gates but fails independent agreement <=3 mm.
Both objects accept only [0], with accepted-view angular span 0 degrees and additional supported
surface 0%; both are **KILL** under the unchanged rule. These zero diversity measurements reflect
the absence of accepted additional views, not a claim of zero true camera motion. Under amendment 1,
the fusion route ends for this deadline; no Stage B recipe is proposed or run, and v4 remains.

`data/outputs/mesh_fusion/f1b/summary.md` links both real-data run records and figures. The completed
object operations took 3.152 s (rack) and 2.878 s (pot); all stereo inputs reused existing producers.
`verification.json` confirms 46 original F1 files byte-identical, matching PnP statistics, no ICP
refinement, and all 20 F1b artifacts matching their hashes. Reinvoking the queue resumes both identities.
