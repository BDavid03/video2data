# Mesh fusion

A failed route, kept as evidence. It tested whether later video frames could add surface to the white pot
(public 42) and dish rack (public 12) by registering stereo views before hand contact. Video only and
CPU only: it loads no scan, computes no CD-O and builds no mesh. Protocol and outcome:
[mesh-fusion-protocol.md](../../../docs/chain/mesh-fusion-protocol.md),
[mesh-fusion-outcome.md](../../../docs/chain/mesh-fusion-outcome.md).

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_fusion stage-a \
  --request src/features/mesh_fusion/queue.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_fusion stage-a \
  --request src/features/mesh_fusion/queue_f1b.json
```

Completed trials resume by identity. Output: `data/outputs/mesh_fusion/` (F1) and
`data/outputs/mesh_fusion/f1b/` (F1b, PnP only). Caches: `data/.runtime/tmp/mesh-fusion/cache/`.

Method: candidate frames 0, 5, 10 and 15. Opening clusters are backprojected with calibrated stereo into a
bounded 3D region above a fitted table plane. Background ORB and PnP give each frame's pose, optionally
refined by trimmed ICP. A held-out landmark set (index modulo 5) measures independent error. A frame is
accepted with at least 30 inliers, reprojection at most 1 px and independent error at most 3 mm. The
route is killed with fewer than three accepted frames, or with angle under 10 degrees and added surface
under 10%.

Result: KILL for both objects in F1 and F1b. Only the identity frame 0 was accepted.

| Object | Frames 5 / 10 / 15 | F1 reprojection px | F1 independent mm | F1b reprojection px | F1b independent mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Rack, 12 | Inliers 455 / 265 / 273 | 2.185 / 4.526 / 4.090 | 31.3 / 60.0 / 59.8 | 0.682 / 0.944 / 0.907 | 29.6 / 47.5 / 63.5 |
| Pot, 42 | Inliers 224 / 150 / 121 | 1.646 / 3.383 / 5.131 | 34.4 / 62.8 / 86.8 | 0.846 / 0.898 / 0.941 | 37.5 / 60.6 / 79.3 |

This shows that this registered-stereo adapter failed. It does not show zero camera motion, or that
reconstruction is impossible by another method. Both objects stayed on v4, and no Stage B exists.

The first F1 run had a coverage bug that counted opening-region growth as new surface, and it excluded only
matched reserved landmarks from ICP. Both were fixed without changing thresholds and the queue reran;
`data/outputs/mesh_fusion/corrections.md` lists superseded runs.
