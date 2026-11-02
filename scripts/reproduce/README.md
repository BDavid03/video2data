# Reproduce Track 3 deliverables

Run from this checkout with the existing project environments. No installation is
performed. The three deliverable scripts default to **plan only**: they read
recorded metadata and print commands without creating run directories or importing
GPU libraries. `--run` enables execution. Commands are printed with shell quoting;
execution stops on the first failing command. Nothing is deleted by these scripts.

```bash
bash scripts/reproduce/d1_meshes.sh
bash scripts/reproduce/d2_poses.sh
bash scripts/reproduce/d3_report.sh
bash scripts/reproduce/verify.sh submissions/20261101T1605Z-d1-v2/manifest.json
bash scripts/reproduce/verify.sh submissions/20261101T1703Z-d2-v1/manifest.json
```

## D1: recorded measures → vessel-rule meshes → CD-O → pack → preflight

```bash
bash scripts/reproduce/d1_meshes.sh 'https://github.com/<owner>/<repo>/commit/<sha>' --run
# Optional: --manifest submissions/<recorded-d1-v2>/manifest.json
```

The default source is the newest timestamped `submissions/*-d1-v2/manifest.json`.
Its `build_run` identifies the recorded build. The script checks that its request
equals the committed `scripts/track3/requests/d1/build-v2-r2.json`, then rebuilds
from that request's **23 recorded measure runs**. It does not regenerate depth or
measurements. The build handler writes a fresh immutable
`outputs/mesh_extents/build/<new-run>/run.json`; its printed path supplies the
candidate and pair directories to subsequent commands.

The kit interpreter runs `scripts/track3/local_cdo.py` on the new object meshes
(default sampling seeds from episodes 1/3/5), then the manifest's kit packing and
preflight commands. Absolute command paths are rebased to the current checkout;
mesh directory, output file and commit URL are substituted. CD-O uses public scans
only as scoring references. The recorded fitted score is **0.9251 cm**; the
cross-fitted adoption headline remains **0.92572 cm**. The CPU reproduction below
reran the fitted score and recovered **0.9251 cm**.

A fresh `.runtime/tmp/reproduce-d1-<unique>/` holds `build.log`, `score.log`,
`cdo.json`, `pack.log`, `preflight.log`, the Parquet and a new `manifest.json`
with commands, hashes and the new build-record path. The source manifest and
historical build are retained. D1 is CPU only, but its rebuild is still opt-in.

## D2: version-aware campaign or composition → pack → preflight

The newest timestamped `submissions/*-d2-v*/manifest.json` is the default.
`--manifest` selects a particular recorded package. The script reads `version`
when present, otherwise the packed filename's `_v1`/`_v2` suffix (the historical
manifests have no explicit version key). Unsupported versions are refused.
Execution requires an explicit commit URL; plans default to the recorded URL.
Episodes 41/43/44 are refused.

### v2: CPU composition

```bash
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/c7f90116f11b41970d0e910375713037faab126f' \
  --manifest submissions/20261101T2315Z-d2-v2/manifest.json --run
```

The script verifies every export's recorded `source` SHA-256 before copying any
files. It checks the recorded composition: **18 unchanged v1 exports** from
`outputs/pose_insurance/results-v1/` and **K1a episodes 1/28** from
`outputs/pose_insurance/rebuilt-v2/`. Episode 30 remains on v1. It copies those
20 sources into a new `.runtime/tmp/reproduce-d2-<unique>/results/`; the historical
results directories remain untouched. CUDA is hidden from its child commands.
It never imports or launches the GPU campaign for this branch.

The manifest's official pack and preflight commands then use that fresh results
directory, explicit URL and fresh CSV path. `--skip-crosscheck` and `--skip-kaggle`
are preserved. The work directory holds `sources.json`, `results/`, `pack.log`,
`preflight.log`, `command_results.json`, the CSV and a new `manifest.json` recording
source/copy hashes, commands, exit codes and version. Preflight must exit 0 and
report `Ready to submit`. This reproduces the recorded composition from evaluated
exports; it does not regenerate their GPU trajectories.

### v1: existing frozen loop campaign (GPU)

```bash
bash scripts/reproduce/d2_poses.sh .runtime/tmp/reproduce-d2-campaign/results \
  'https://github.com/<owner>/<repo>/commit/<sha>' \
  --manifest submissions/20261101T1703Z-d2-v1/manifest.json --run
```

Use a new, empty results directory. The original `RESULTS_DIR COMMIT_URL` interface,
queue budget and `tasks` then `evaluation` commands are retained. The script copies
the slice queue into its fresh work directory, taking the learned budget and
export episode list from the manifest: **512 steps, one world, replay minimum 32**
on **20 evaluation episodes**. Budget selection and tuning are not repeated.

The existing loop lacks a results-directory CLI option. The Python invocation
binds `loop.RESULTS` before `loop.main()`; its handlers, resume ledger, summaries,
completion marker and GPU lease are unchanged. Run this branch only in an
authorised GPU job when the GPU schedule is free. The current slice code is used;
changing a path does not reconstruct historical code. The CPU comparison below
checks all four printed v1 commands against the original script; it does not
execute the v1 campaign. Packing/preflight and their logs retain the same behavior.

## D3: figures → PDF

```bash
bash scripts/reproduce/d3_report.sh --run
# Optional source argument after --run: docs/report/track3/main.tex
```

This CPU command runs `docs/report/track3/figures/make_figures.py` to regenerate
the current qualitative panel from recorded measurements, meshes and scores,
then `bash scripts/report/build_pdf.sh docs/report/track3/main.tex`.
It performs no metric rerun. The existing figure script owns its recorded input
paths; [the report README](../../docs/report/track3/README.md) documents their
provenance, display transforms and the **2-page** PDF evidence.

Execution refreshes the figure PDF and `main.pdf`/TeX logs beside their sources.
A fresh `.runtime/tmp/reproduce-d3-<unique>/build.log` records both commands and
their output; Matplotlib uses `.runtime/tmp/reproduce-report-matplotlib/`.
The existing builder uses the pinned Tectonic and its in-checkout cache, accepts
only the documented microtype tracking error and rejects other TeX errors.

## Run records and hash coverage

| Deliverable | Recorded source |
| --- | --- |
| D1 | `submissions/20261101T1605Z-d1-v2/{manifest,run}.json`; the manifest's `build_run` and its request's measure producers |
| D2 | v1: `submissions/20261101T1703Z-d2-v1/{manifest,run}.json`, loop ledger/completion; v2: `submissions/20261101T2315Z-d2-v2/{manifest,run}.json`, per-export `source`/SHA-256 and `composition`, `outputs/pose_insurance/v2/report.md` |
| D3 | `docs/report/track3/README.md` provenance table; `docs/report/track3/main.log`; new command/output log when executed |

`verify.sh <manifest.json>` is read-only and CPU only. It streams SHA-256 for the
packed file and **every nested entry with a path/file and sha256**, including
exports, inputs and source hashes. Missing files print `MISMATCH`. Exit status is
0 when all match, 1 on any mismatch. Paths must resolve inside this checkout.

Coverage is exactly what the supplied manifest records; it does not recursively
follow unhashed run-record paths. In particular, the old D1 manifest lists only
the packed-file hash, so its **one** matching entry does not independently verify
all 23 measure producers. The build handler admits its recorded inputs on rebuild.
Hash agreement also does not establish numerical or bitwise reproducibility of a
new GPU run. Align the checked-out sources with the recorded source hashes before
claiming an exact historical reproduction.

## CPU evidence, 2026-11-02

Both real-data `--run` paths completed with **exit 0**. D1 build, local CD-O,
packing and preflight each exited 0; D2 v2 source verification/copy, packing and
preflight completed, with both kit commands exiting 0. Both preflights reported
`Ready to submit`. No GPU stage, install, upload or test suite was run.

Commands executed with the exact placeholder URLs recorded in the manifests:

```bash
bash scripts/reproduce/d1_meshes.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/874ff6793721faa0035a71788b8f5a22f9b4b2d3' --run
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/c7f90116f11b41970d0e910375713037faab126f' \
  --manifest submissions/20261101T2315Z-d2-v2/manifest.json --run
bash -n scripts/reproduce/d1_meshes.sh
bash -n scripts/reproduce/d2_poses.sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .runtime/linux-cuda/bin/python -B \
  .runtime/tmp/ctx05-d5-evidence/check_scripts.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .runtime/kit-env/bin/python -B \
  .runtime/tmp/ctx05-d5-evidence/compare_deliverables.py
bash scripts/reproduce/verify.sh submissions/20261101T1605Z-d1-v2/manifest.json
bash scripts/reproduce/verify.sh submissions/20261101T2315Z-d2-v2/manifest.json
```

| Comparison | Recorded packed SHA-256 | Fresh SHA-256 | Result |
| --- | --- | --- | --- |
| D1 Parquet | `43fa6ac02353520aac4d019f7d6d45e0aab923f1c8f43d6cf03e30c242ba95c2` | `43fa6ac02353520aac4d019f7d6d45e0aab923f1c8f43d6cf03e30c242ba95c2` | Byte-identical; 294,912 rows |
| D2 v2 CSV | `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741` | `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741` | Byte-identical; 9,120 rows |

D1's pandas comparison is exact, including row order; x/y/z float bits, Arrow
schema metadata and Parquet metadata also match. All **47 PLY files** (11 object
meshes plus 36 scored-pair copies) match the historical build bytes, so the geometry
is identical. There are no row-order, float, metadata or timestamp differences in
the packed file. The fresh build is
`outputs/mesh_extents/build/20261102T000114.850697Z-1a4c354e/run.json`.
The fitted local CD-O is **0.9251 cm**, matching the recorded **0.9251 cm** at the
scorer's four-decimal precision (delta **0.0000 cm**), over **10 public-scan objects**
and the kit seeds for episodes **1/3/5**. This is the fitted reconstruction score,
not a new cross-fitted assessment or host score.

D2 verifies **20/20 recorded source hashes** and **20/20 copied export hashes**;
pandas also confirms identical CSV values and row order. The v1 old-versus-new
plan comparison recovers the same four commands (tasks, evaluation, pack,
preflight). `bash -n` passed for both D1/D2 scripts; the edited D2 script's embedded
Python passed `py_compile`. D1 needed no script change.

Evidence directories (paths relative to this checkout):

| Evidence | Directory / files |
| --- | --- |
| D1 reproduction | `.runtime/tmp/reproduce-d1-wbr3u3xl/`: `console.log`, `build.log`, `score.log`, `cdo.json`, `pack.log`, `preflight.log`, `track3_meshes_v2.parquet`, `manifest.json`, `comparison.json` |
| D2 reproduction | `.runtime/tmp/reproduce-d2-wi7g6bmn/`: `console.log`, `sources.json`, `results/`, `pack.log`, `preflight.log`, `command_results.json`, `track3_poses_v2.csv`, `manifest.json`, `comparison.json` |
| Comparisons and syntax | `.runtime/tmp/ctx05-d5-evidence/`: `comparisons.log`, `syntax.log`, `script-checks.json`, `script-checks.log`, `v1-plan-{old,new}.log`, comparison/check scripts |
| Source-manifest verification | `.runtime/tmp/ctx05-d5-evidence/verify-{d1,d2}-v2.log` |

The read-only D1 source-manifest verifier exited **0** (1/1 hash matches). The D2
source-manifest verifier exited **1** (**106/107 match**): its only mismatch is the
pinned `docs/chain/pose-placement-protocol.md`, expected
`baa3b86154a5eb9007586c5737702246a7b16af6a0b0bba6d26d6bf70c4022e5`, actual
`bf724b9cb50dda0571f8de2c0b785eed54df38534f1a2cfc44935dcc6742a9c8`.
The current committed protocol already differs from that package's recorded hash;
this job did not edit it. The packed CSV, all exports, and the other recorded
hash entries match. Package reproduction succeeds despite this separate historical
provenance mismatch; a fully matching historical source tree is not established.

Remaining gap: **D2 GPU regeneration was not executed**, including the physical
K1a runs for episodes 1/28 and the v1 campaign. The CPU result proves reproduction
from recorded evaluated exports, not retraining or numerical GPU reproducibility.
**D3 reproduction is handled separately.** Placeholder URLs remain nonfinal;
Kaggle authentication/checks were skipped and no upload was attempted. The D1
preflight also retains the recorded `no --results given` warning.
