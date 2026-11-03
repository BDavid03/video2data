# F1 mesh fusion: Stage A

CPU-only, video-only feasibility for the white pot (public 42) and dish rack (public 12), under
`docs/chain/mesh-fusion-protocol.md` (277c7df). This operation never loads a scan, computes CD-O,
builds a triangle mesh or starts Stage B.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .runtime/linux-cuda/bin/python -B -m track3.features.mesh_fusion stage-a \
  --request track3/features/mesh_fusion/queue.json
```

`queue.json` contains one request per object. Completed trials resume by request/source identity;
`outputs/mesh_fusion/ledger.jsonl` records each completed object and `summary.md` reports the two verdicts.
Every object and queue execution owns a `run.json`, hashed inputs/sources and hashed output inventory.
New stereo observations go through the unchanged CPU `stereo_depth.generate` handler and are cached by
single-frame request. No CUDA is required. All local caches go under `.runtime/tmp/ctx20/fusion-cache/`.

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
ledger rows remain intact; `outputs/mesh_fusion/corrections.md` identifies superseded and current runs.

See `outputs/mesh_fusion/summary.md` and the per-object summaries for figures, acceptance tables and run records.
`verification.json` records direct real-data recomputation and library equivalence: coverage 0% by an independent
voxel-set calculation, camera feature outputs exactly equal, ICP adapter/direct-call error 0, angular formula
agreement within 1.54e-13 degrees, and all 20 final per-object artifact hashes matching. All nine modules compile
and import on CPU. A second invocation reused both completed trial identities without adding ledger rows.
