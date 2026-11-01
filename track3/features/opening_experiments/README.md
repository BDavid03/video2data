# Opening geometry experiments

The authorized 18 October 2026 run completed rectification, stereo depth, scale
anchoring and camera estimation. Geometric clustering rejected an above-table
cluster touching the image border (`opening_not_quiet`), blocking both mesh
variants. See the [stage results](../../../outputs/opening_experiments/20261018T120039.470932Z-e0b15dc0/result.json)
on Z:. The initial OpenCV matrix-shape failure is retained in its earlier failed
run; explicit contiguous calibration matrices resolved it in this rerun.

This independent slice owns a
bounded episode-39 comparison using existing public feature APIs. No teacher,
GPU model, learned mask, pose label, package, download or licensed asset is needed.

Supply `depth_run` in `run.json`: the completed **contiguous** MoGe2 teacher for
public episode 39, containing frames 0–19. Sparse whole-clip teachers cannot serve
this opening chain. `frame_count` is bounded to 1–20; `mesh_experiments` defaults
to true. Requests never include episode 41, validation or evaluation episodes.

After explicit authorization, run in `Z:/hugging-face/video2data` using the existing
interpreter selected by `scripts/use-scratch-depth.ps1`:

```text
python -m track3.features.opening_experiments --request track3/features/opening_experiments/run.json
```

The chain recomputes 768-side stereo rectification, generates CPU SGBM depth,
observes monocular metric scales and estimates an opening camera trajectory/table
world without masks. Existing defaults retain disparity range 128, block size 5,
one-pixel reverse consistency, 64 anchor-support pixels and 50% frame support.
Camera initialization requires actual table-plane and static-feature support;
failure is an experiment result, never a reason to substitute poses or geometry.

Optional geometric opening components use the actual calibrated depth and table
world. Hull and observed TSDF reconstruction are attempted independently only
when a completed persistent opening contains exactly one component and the
recorded episode roster is exactly `mini_dust_pan`. The component's actual array
position supplies its assignment; the producer's instance label is retained.
Merged, absent or ambiguous components block reconstruction. Hull completion and
open observed TSDF remain separate research products; neither creates production
full-video object masks or establishes reconstruction accuracy.

The owning `outputs/opening_experiments/<run-id>/run.json` records each stage's
parent hashes, status, timings and child run identity. Successful children retain
their own artifact inventories. A unique new matching failed producer is retained
when the older handler does not attach a failed-run path. Dependent stages are
blocked; disabled optional stages remain unattempted. `result.json` reports actual
completed/failed/blocked counts. A completed orchestration may include failed
experiments: inspect stage outcomes before consuming any child product.

The wrapper requires its `runtime.json` `run_authorized` flag. Participating
slices separately require stereo `rectify_authorized`, `generate_authorized`,
`anchor_authorized`; camera `estimate_authorized`; and, for mesh experiments,
masks `opening_clusters_authorized` plus mesh `reconstruct_authorized`. Each needs
public episode 39 in its allowlist. This wrapper leaves those gates unchanged.
No tests, smoke checks, validation commands, notebooks, compilation, simulation
or submission upload is part of this experiment. Existing environment and sources
remain unchanged; temporary files and bytecode caches stay in the canonical root.
