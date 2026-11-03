# Frozen G1 mesh pilot

Generates one Hunyuan3D-2mini shape for each frozen frame-0 camera-A crop: rack public 12,
pitcher public 21, pot public/tuning 42. The handler admits exactly those three jobs and requires
`pilot_authorized`. Generation uses a separate Python 3.10 venv and the cooperative `GpuLease`.
The main CUDA environment is read-only. No texture extensions or background-removal models run.

Run from the checkout after the approved installation in `.runtime/hunyuan-mesh/INSTALL.md`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_generative smoke
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_generative pilot --request track3/features/mesh_generative/request.json
```

`queue.json` pins the three recorded mask producers. The unchanged left half of each producer's
lossless overlay is the remapped source RGB frame. Labels use assignment cluster ids plus one.
Pixels outside those masks become white. The square crop adds 10% of its longest mask extent
on each side, padding off-image regions white. The model's own image preprocessing remains
unchanged and is recorded. Shape sampling is seed 12345, 30 steps, octree 380, `mc_algo="mc"`.
No scan data is read until generation finishes. There are no re-rolls or crop adjustments.

The resumable loop skips completed generations and appends one completed object per ledger row.
An incomplete generation marker refuses automatic repetition. CPU scoring uses the existing kit
budget converter (4096 vertices and faces) and `mesh_eval.pairs.score_pair` with the original
v4 ledger's 12 scoring pairs/seeds (rack 5, pitcher 1, pot 6), plus an incumbent equivalence check.
Those rows reuse three shapes, rather than generate the full panel. The gate uses equal-object
mean gain, contribution to the original 26-pair mean, and the 0.15 cm regression cap.

The smoke mesh, timing and VRAM report live under `.runtime/tmp/g1/smoke/`.
Pilot inputs, sources, weights/archive hashes, timings, GPU telemetry and output hashes live in
`outputs/mesh_generative/g1-pilot/run.json`; each object has `crop.png`, `raw.ply`, generation
JSON/log, `scores.json`, `scores.npz` (budgeted generated/v4/scan geometry), and score log.
The loop writes `ledger.jsonl`, `summary.json`, `summary.md` and `comparison.png`.

Latest result (2026-11-03 UTC): **FAIL**, stopped at the pilot gate. Equal-object gain is
**+0.25936561 cm**; weighted contribution improves **0.42159881 → 0.40244651 cm**.
Pitcher improves **1.48717155 → 0.60673095 cm**, pot **0.85836290 → 0.72912484 cm**;
rack regresses **0.86484402 → 1.09642586 cm** (**+0.23158184 cm**, above the 0.15 cap).
V4 stays. The three incumbent equivalence errors are exactly **0.0 cm**. Generations took
**17.99 / 18.18 / 18.67 s** (rack / pitcher / pot), excluding 15.56–17.18 s model loading;
peak torch allocation / reservation was **4620.69 / 5142 MiB**, device-wide sampled peak
**6865 MiB** for each. Smoke passed in 18.75 s generation, 17.07 s loading.

The final isolated environment uses torch 2.7.1+cu128, torchvision 0.22.1+cu128,
Transformers 4.48.3 and Diffusers 0.32.2. See the install record for the failed pre-sampling
compatibility attempts and complete package lock. No upstream source or checkpoint was modified.

See [the summary](../../../outputs/mesh_generative/g1-pilot/summary.md) and
[the figure](../../../outputs/mesh_generative/g1-pilot/comparison.png).
The external CLI log is `outputs/mesh_generative/g1-pilot-handler.log`; keep wrapper logs outside
the run directory so the final CLI print cannot invalidate an artifact hash.

The handler always stops at that gate. Scans were reused and object selection used v4 scores,
so this pilot is not an untouched validation. No full panel or packing is authorized here.

## G2: fixed pot/pitcher follow-up

G2 is a separately authorized campaign under the committed protocol's final section. G1's
records remain untouched. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_generative g2 > outputs/mesh_generative/g2-handler.log 2>&1
```

`g2_queue.json` freezes all relevant public development and evaluation pot/pitcher pairs,
recorded mask producers, fallback cases, fit settings and the development aggregation. Only
public 21/42 reuse their own G1 meshes. Evaluation 5/8/20 generate from their own frame-0 crops.
Public pot 0/26 and evaluation pot 1/3/6/28 lack usable assignments and keep v4. No new mask is made.

One PCA-initialized proper Sim(3) ICP aligns at most 4096 observed points to 8192 deterministic
mesh-surface samples, with 40 iterations, no reflection and a 1e-7 cost stopping threshold.
Final point-to-triangle medians use **every valid masked depth point**, with bounded chunks and
trimesh's exact closest-point library. Both meshes use the same procedure. Residuals share the
frame's unanchored MoGe units; their ratio is the admission criterion. Fits and every fallback
are appended to `outputs/mesh_generative/g2/ledger.jsonl` before any scan scoring. A sealed
`admissions.json` precedes development and, conditionally, the packed veto.

Development uses public pot 0/26/42 and pitcher 21, kit seed episodes 1/3/5, matched v4 geometry
and the existing object folds. It averages views within each object/seed, then the ten objects;
fixed roster counts give the 26-pair weighted mean. The other eight objects retain their
matched v4 results. There is no adjustable parameter. Evaluation meshes remain separate.

Only a development PASS enters `g2-packed-check/`: copies v4's complete production PLY tree,
replaces only video-admitted evaluation pot/pitcher pairs, checks unchanged PLY hashes and packed
coordinates, uses the manifest's official packer, and runs the existing `mesh_eval.pairs.run_queue`
and `write_summary` code with the original pair seeds. This keeps scoring outputs and scratch
inside this job's permitted directories. Both kit policy-track and mesh-track preflights run.
The preflight checks environment/sample availability; packing and row/geometry comparisons check
the placeholder file. A failure stops G2 with no repair or runner-up.

The lead can repack a **passing** frozen production tree after committing the new sources:

```bash
bash scripts/reproduce/d1_meshes_v5.sh "https://github.com/BDavid03/video2data/commit/$(git rev-parse HEAD)" --run
```

The script refuses unless both gates passed. This worker never executes a real-URL pack.
G2 selects objects after their G1 results on the same public scans; all results are reused-scan
development evidence and not an untouched assessment. See `outputs/mesh_generative/g2/summary.md`
for the completed verdict, all residuals, gate thresholds, seed spread, resources and figures.

Latest G2 result (2026-11-03 UTC): **development PASS; packed veto PASS**. Development
equal-object / pair-weighted means are **0.56244822 / 0.61972984 cm**. Exact packed means
are **0.56443301 / 0.62080727 cm**, against v4 **0.65338396 / 0.66292148 cm**.
Maximum object regression is zero in both stages. Only evaluation pot 5 and pitcher 20
replace v4. Evaluation pot 8 was rejected by video residual (4.37493 x v4), before scoring.
All 29 non-target production PLY files are byte-identical to v4; 278,528 unchanged packed
coordinate rows match exactly. Both kit preflights report Ready to submit.

The placeholder file is
`outputs/mesh_generative/g2-packed-check/track3_meshes_v5_placeholder.parquet`, SHA256
`8545628a8cfc3a23639942f37fbafb0aaebdc5070b3c5038cb0829c9d23eaa9b`.
The real-URL repack command above must run **after the lead commits the new G2 sources**.
A dry plan was checked; no real-URL file, commit, push or upload was created by this job.

## G3: automatic frame-zero pot masks

The committed [G3 protocol](../../../docs/chain/mesh-generative-g3-protocol.md) freezes six
new pot pairs, three recorded-mask controls and every threshold. Run once or resume with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_generative g3 > outputs/mesh_generative/g3-handler.log 2>&1
```

`g3_queue.json` pins the inputs. `.runtime/grounding-dino/` and `.runtime/sam2/` contain
separate Python 3.10 environments, immutable weights, install records and complete package
locks. DINO uses revision `a2bb814`, Transformers 4.48.3, the exact prompt
`white cooking pot.`, box threshold 0.4 and text threshold 0.3. SAM2.1 Hiera-Large uses
source `2b90b9f5ceec907a1c18123530e92e794ad901a4`, the official checkpoint, a box-only
prompt and the highest predicted-IoU mask. Both use torch 2.7.1+cu128 / torchvision
0.22.1+cu128. Optional SAM CUDA extensions, compilation and mask postprocessing are disabled;
only the largest eight-connected component is retained.

The lid rule runs first. The first control below IoU 0.70 stops the campaign before candidate
inference, generation or scan scoring. Candidate masks must have 10,000–45,000 pixels,
at most 10% of their inner boundary within Euclidean distance 2 pixels of the image border
or remap-invalid pixels, and at least 2,000 observed depth points. Boundaries use
eight-neighbour erosion. Each surviving pair reuses G2's exact crop, generation worker,
mesh budget and Sim(3) residual fitting; generated median residual must be at most 1.5 × v4.
No completed prediction is rerun. Incomplete inference/generation markers refuse retries.

All video-only choices enter `ledger.jsonl`, then immutable `admissions.json`, before any
new scan score. G2's public 42/pitcher 21 and evaluation 5/8/20 decisions remain intact.
Development reuses G2's view/seed aggregation and folds, with G3's strict limits and at least
one newly admitted public pair. Only a PASS builds `g3-packed-check/` from v5's exact PLY
bytes plus newly admitted evaluation pairs, compares against real-URL v5, and runs the
existing `mesh_eval.pairs` loop. Numeric packed failure stops before kit preflight.

`outputs/mesh_generative/g3/` contains the input/model/source hashes, frame inputs, masks,
mask checks, crops, observed clouds, generated meshes, admissions, gate results,
`summary.md`, `summary.json` and a labelled `comparison.png` for all six candidates.
`run.json` records wall time and artifact inventories; each GPU subprocess records torch
peaks and cooperative lease telemetry. Logs sit beside the run directories, outside their
artifact inventories. The v6 reproduction script refuses unless control, development and
packed veto all passed. This worker never executes a real-URL pack.

G3 retains G1's rack-failure disclosure and G2's post-pilot object selection: every score is
development evidence on reused public scans. Existing G1/G2 run records and the v5 script
remain untouched. The v5 dry repack still works at HEAD; a G2 campaign rerun refuses the
changed slice source inventory, as intended. V5 itself reproduces at URL commit `b161723`.

Latest G3 result (2026-11-03 UTC): **control PASS; development PASS; packed veto FAIL;
v5 stands**. Control IoUs were **0.91397899 / 0.74798665 / 0.75204068** (public 42,
evaluation 5/8). Public 0 and evaluation 1 kept v4 under the lid rule. All four inferred
candidate masks passed area, border and depth checks. Public 26 and evaluation 6 were
admitted at residual ratios **1.33446 / 0.96201**; evaluation 3/28 were rejected at
**1.68157 / 1.56437**. No completed prediction or generation was repeated.

Development equal-object / pair-weighted means were **0.56084957 / 0.61604065 cm**
(limits **0.56244822 / 0.61972984**), with one newly admitted public pair. The packed
candidate changed only evaluation pot 6. Exact packed means regressed to
**0.56702215 / 0.62678221 cm** against v5 **0.56443301 / 0.62080727 cm**. Pot mean
regressed **0.81727092 → 0.84316233 cm** (+0.02589141); all other object means matched.
This fails both strict mean gates, despite meeting the 0.15 cm object-regression cap.
All **35** unchanged production PLY files and **286,720** unchanged packed coordinate
rows match v5. Kit preflights and real-URL repacking were not run after the numeric veto.

The placeholder is `outputs/mesh_generative/g3-packed-check/track3_meshes_v6_placeholder.parquet`,
SHA256 `54939f3d5dfd326b18e593e722fb01732f1a41ebdcc2b5ffe1e44abf83e3a481`.
Final run wall time was **583.00 s**; torch allocation/reservation peaked at
**4620.69 / 5142 MiB**, with device-wide sampled peak **7058 MiB**. See
[summary.md](../../../outputs/mesh_generative/g3/summary.md) and
[comparison.png](../../../outputs/mesh_generative/g3/comparison.png) for every candidate,
mask measurement, fit ratio, gate and resource row. The sealed `admissions.json` predates
development scoring. Two preparation-only failures (inherited video provenance and a
nonzero-frame measurement reference) were corrected before any model inference; their
logs are preserved under `.runtime/tmp/g3/`.
