# Pretrained mesh shapes

Generates Hunyuan3D-2mini shapes from masked first-frame crops, and tests whether they should replace
procedural pot and pitcher meshes. Generation uses a separate Python 3.10 environment and the GPU lease,
with no texture or background-removal models. Install record: `data/.runtime/hunyuan-mesh/INSTALL.md`.
Protocols and outcome: [mesh-generative-protocol.md](../../../docs/chain/mesh-generative-protocol.md),
[mesh-generative-g3-protocol.md](../../../docs/chain/mesh-generative-g3-protocol.md),
[mesh-generative-outcome.md](../../../docs/chain/mesh-generative-outcome.md).

All scores below are development evidence on reused public scans, with objects picked after earlier
results on those scans. None is an untouched assessment.

The commands describe recorded campaigns. At this checkout G1 and G2 refuse their changed source
inventories, possibly after preparing crops or cached observations, so reproduce them at the recorded
commit. A new campaign needs a new protocol and output identity. Completed G3 and insights calls return their
existing paths at once. The v5 repack still works here.

| Campaign | Result | Output |
| --- | --- | --- |
| G1 pilot: rack, pitcher, pot | FAIL, rack regressed above the cap. v4 stays | `data/outputs/mesh_generative/g1-pilot/` |
| G2 pots and pitcher, video-admitted | PASS. Evaluation pot 5 and pitcher 20 replace v4 in D1 v5 | `data/outputs/mesh_generative/g2/`, `g2-packed-check/` |
| G3 automatic pot masks | Packed veto FAIL. v5 stands, no v6 exists | `data/outputs/mesh_generative/g3/` |
| Insights: seeds and admission | Diagnostic only, no validated signal | `data/outputs/mesh_generative/insights/` |

The final combined v5 submission does not use the G2 shapes for pot 5 or pitcher 20, because they have no
recorded mapping to the rollout frame. See [the submission ledger](../../../docs/chain/SUBMISSION.md).

## G1 pilot

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative smoke
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative pilot --request src/features/mesh_generative/request.json
```

Crops: masks from `queue.json`, white outside the mask, padded by 10% of the longest extent. Sampling: seed
12345, 30 steps, octree 380, `mc_algo="mc"`, no re-rolls. Scoring uses the kit's 4096-vertex budget and
`mesh_eval.pairs.score_pair` on v4's 12 pairs (rack 5, pitcher 1, pot 6). The handler always stops at the gate.

Equal-object gain +0.25936561 cm; weighted contribution 0.42159881 to 0.40244651 cm. Pitcher 1.487 to
0.607 cm, pot 0.858 to 0.729 cm, rack 0.865 to 1.096 cm (+0.232 cm, above the 0.15 cap). Generation took
about 18 s per shape, peak torch allocation 4620.69 MiB, sampled device peak 6865 MiB. Environment: torch
2.7.1+cu128, Transformers 4.48.3, Diffusers 0.32.2. Keep wrapper logs outside the run directory, so the final
CLI print cannot invalidate an artifact hash.

## G2: pot and pitcher follow-up

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative g2 > data/outputs/mesh_generative/g2-handler.log 2>&1
bash scripts/reproduce/d1_meshes_v5.sh "https://github.com/BDavid03/video2data/commit/$(git rev-parse HEAD)" --run
```

`g2_queue.json` freezes pairs, masks, fallbacks and fit settings. Evaluation 5, 8 and 20 generate from their
own crops. Public pot 0 and 26 and evaluation pot 1, 3, 6 and 28 lack usable assignments and keep v4. A
proper Sim(3) ICP fits both generated and v4 meshes to the masked depth points, in the frame's unanchored MoGe
units, and the ratio of median residuals is the admission check. Admissions are sealed in `admissions.json`
before any scan score. The repack script refuses unless both gates passed.

| | Equal-object (cm) | Pair-weighted (cm) |
| --- | ---: | ---: |
| v4 packed | 0.65338396 | 0.66292148 |
| G2 development | 0.56244822 | 0.61972984 |
| G2 packed | 0.56443301 | 0.62080727 |

Maximum object regression is zero in both stages. Evaluation pot 8 was rejected by video residual (4.37493
times v4) before scoring. All 29 other PLYs are byte-identical to v4. Placeholder pack:
`g2-packed-check/track3_meshes_v5_placeholder.parquet`, SHA-256
`8545628a8cfc3a23639942f37fbafb0aaebdc5070b3c5038cb0829c9d23eaa9b`. V5 itself reproduces at commit `b161723`.

## G3: automatic first-frame pot masks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.mesh_generative g3 > data/outputs/mesh_generative/g3-handler.log 2>&1
```

Grounding DINO (revision `a2bb814`, prompt `white cooking pot.`, thresholds 0.4 and 0.3) and SAM2.1 Hiera-Large
(source `2b90b9f5ceec907a1c18123530e92e794ad901a4`, box prompt) run in environments under
`data/.runtime/grounding-dino/` and `data/.runtime/sam2/`. Controls must reach IoU 0.70. Candidates need
10,000 to 45,000 pixels, at most 10% border contact, at least 2,000 depth points and a residual at most 1.5 times v4.
Incomplete markers refuse retries. At HEAD a completed call returns its path and changes nothing; find
the recording commit with `git log --format=%h -1 -- src/features/mesh_generative/g3_queue.json`.

Control IoUs 0.914, 0.748, 0.752. Public 26 and evaluation 6 admitted at ratios 1.33446 and 0.96201; evaluation
3 and 28 rejected at 1.68157 and 1.56437. Development passed (0.56084957 and 0.61604065 cm). The packed veto
failed: 0.56702215 and 0.62678221 cm against v5's 0.56443301 and 0.62080727 cm, with evaluation pot 6 the only
change. Placeholder: `g3-packed-check/track3_meshes_v6_placeholder.parquet`, SHA-256
`54939f3d5dfd326b18e593e722fb01732f1a41ebdcc2b5ffe1e44abf83e3a481`.

## Insights

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.mesh_generative insights
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B data/outputs/mesh_generative/insights/critic-repair/repair_driver.py
```

Post-hoc only, behind `insights_authorized`: nine existing views plus four extra seeds for three views. The
second command rerenders and audits on CPU. `finalize.py.txt` is a retained driver; do not execute it. Scores
never select a seed or change a gate or pack.

Admission ratio against CD-O change has Spearman rho 0.38333 (n=9), or 0.14286 within pots (n=7), from
dependent observations, so there is no validated predictive signal. Admission still passes the harmful G3 pot 6
shape. Generation-seed CD-O sample SDs are 0.09122, 0.56984 and 0.33492 cm for public pitcher 21, public pot 42
and evaluation pot 5. Pooled pot seed variance is 7.49 times the variance across seven original views, which is
descriptive and not a fraction of error explained. Handle visibility is confounded with mask source. Pitcher
replication covers public 21 only, and pot 42's five-seed mean is worse than v4.
