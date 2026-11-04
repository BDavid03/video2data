# Mesh fusion

Pipeline role: step 2 (shapes), a failed test kept as evidence. Not in the submitted entry.

We tested whether later video frames could add surface to the white pot (public episode 42) and dish rack (public 12)
by registering stereo views before the hand touches them. Video only, CPU only: no scan, no CD-O, no mesh built.
Protocol (written before the run) and outcome: [mesh-fusion-protocol.md](../../../docs/chain/mesh-fusion-protocol.md),
[mesh-fusion-outcome.md](../../../docs/chain/mesh-fusion-outcome.md).

## Run

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_fusion stage-a \
  --request src/features/mesh_fusion/queue.json
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_fusion stage-a \
  --request src/features/mesh_fusion/queue_f1b.json
```

Finished trials resume by identity. Output: `data/outputs/mesh_fusion/` (F1) and `data/outputs/mesh_fusion/f1b/`
(F1b, PnP only). Cache: `data/.runtime/tmp/mesh-fusion/cache/`.

## Method

Candidate frames 0, 5, 10, 15. The opening table clusters are lifted to 3D with calibrated stereo. Background ORB
features and PnP give each frame's pose, optionally refined by trimmed ICP. A held-out landmark set (index mod 5)
measures error independently. A frame passes with at least 30 inliers, reprojection at most 1 px and independent error
at most 3 mm. The route stops with fewer than three passing frames, or with under 10 degrees of view change and under
10% new surface.

## Result: stopped for both objects

Only frame 0 (the reference frame itself) passed.

| Object | Inliers, frames 5 / 10 / 15 | F1 reprojection px | F1 independent mm | F1b reprojection px | F1b independent mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Rack, 12 | 455 / 265 / 273 | 2.185 / 4.526 / 4.090 | 31.3 / 60.0 / 59.8 | 0.682 / 0.944 / 0.907 | 29.6 / 47.5 / 63.5 |
| Pot, 42 | 224 / 150 / 121 | 1.646 / 3.383 / 5.131 | 34.4 / 62.8 / 86.8 | 0.846 / 0.898 / 0.941 | 37.5 / 60.6 / 79.3 |

This shows this stereo registration failed. It does not show the camera was still, or that another method could not
work. Both objects kept their v4 shapes. A coverage bug in the first F1 run was fixed without changing thresholds and the
queue reran; `data/outputs/mesh_fusion/corrections.md` lists the replaced runs.
