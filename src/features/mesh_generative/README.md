# Pretrained mesh shapes

Pipeline role: step 2 (shapes), a side route. It generates Hunyuan3D-2mini shapes for pots and pitchers from a masked
first frame. These shapes went into the separate D1 v5 mesh file (CD-O 0.53562 cm on the leaderboard). They are not in
the submitted combined file: that file has no recorded mapping from these shapes to the simulation frame, so it uses
procedural shapes only.

Generation runs in its own Python 3.10 environment under the shared GPU lock, with no texture or background-removal
models. Install record: `data/.runtime/hunyuan-mesh/INSTALL.md`. Protocols written before each run, and the outcome:
[mesh-generative-protocol.md](../../../docs/chain/mesh-generative-protocol.md),
[mesh-generative-g3-protocol.md](../../../docs/chain/mesh-generative-g3-protocol.md),
[mesh-generative-outcome.md](../../../docs/chain/mesh-generative-outcome.md).

All scores are CD-O (cm, lower is better) on reused public scans, with objects chosen after earlier results on the same
scans. They are development numbers, not an independent test.

| Campaign | Result | Output |
| --- | --- | --- |
| G1 pilot: rack, pitcher, pot | Fail: rack got worse past the 0.15 cm cap. v4 kept | `data/outputs/mesh_generative/g1-pilot/` |
| G2 pots and pitcher, video-checked | Pass. Evaluation pot 5 and pitcher 20 replace v4 in D1 v5 | `data/outputs/mesh_generative/g2/`, `g2-packed-check/` |
| G3 automatic pot masks | Packed check failed. v5 stands; there is no v6 | `data/outputs/mesh_generative/g3/` |
| Insights: seeds and admission | Diagnostic only; no validated signal | `data/outputs/mesh_generative/insights/` |

These commands record past runs. At this commit G1 and G2 refuse because their source hashes changed; run them at the
recorded commit. Finished G3 and insights calls just return their paths. The v5 repack still works here.

## G1 pilot

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative smoke
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative pilot --request src/features/mesh_generative/request.json
```

Crops come from the masks in `queue.json`, white outside the mask, padded by 10%. Seed 12345, 30 steps, octree 380.
Scored on v4's 12 pairs. Object-mean gain +0.259 cm, but rack 0.865 to 1.096 cm (+0.232, over the cap), so it failed.
About 18 s per shape; peak GPU memory 6865 MiB.

## G2: pots and pitcher (in D1 v5)

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative g2 > data/outputs/mesh_generative/g2-handler.log 2>&1
bash scripts/reproduce/d1_meshes_v5.sh "https://github.com/BDavid03/video2data/commit/$(git rev-parse HEAD)" --run
```

`g2_queue.json` freezes pairs, masks and fit settings. The admission gate (a pass/fail check fixed before scoring) is
video-only: fit both the generated and the v4 mesh to the masked depth points with Sim(3) ICP, and compare median
residuals. Gate results are sealed in `admissions.json` before any scan score. The repack refuses unless both the video
gate and the packed scan check passed. Evaluation pot 8 failed the video gate (4.37 times v4's residual).

| | Object mean (cm) | Pair mean (cm) |
| --- | ---: | ---: |
| v4 packed | 0.65338396 | 0.66292148 |
| G2 development | 0.56244822 | 0.61972984 |
| G2 packed (D1 v5) | 0.56443301 | 0.62080727 |

No object got worse. The other 29 PLYs are byte-identical to v4. V5 reproduces at commit `b161723`.

The video gate has no proven link to CD-O: in G3 it also passed a pot 6 shape that the scan check vetoed.

## G3: automatic pot masks (failed)

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative g3 > data/outputs/mesh_generative/g3-handler.log 2>&1
```

Grounding DINO plus SAM2.1 made first-frame pot masks (environments under `data/.runtime/grounding-dino/` and
`data/.runtime/sam2/`). Development improved (0.56085 / 0.61604 cm), but the packed check failed (0.56702 / 0.62678 cm
against v5's 0.56443 / 0.62081 cm). Evaluation pot 6 was the only change.

## Insights

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_generative insights
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/mesh_generative/insights/critic-repair/repair_driver.py
```

After-the-fact analysis only. Admission ratio against CD-O change: Spearman rho 0.383 (n=9), 0.143 within pots (n=7).
So the video gate does not predict CD-O. Seed-to-seed CD-O spread is large (SD up to 0.570 cm for public pot 42).
`finalize.py.txt` is a kept record; do not run it.
