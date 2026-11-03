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
