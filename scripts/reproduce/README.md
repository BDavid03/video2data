# Reproduce Track 3 deliverables

Before uploading, run the complete CPU release check:

```bash
bash scripts/reproduce/check_release.sh
```

It hashes the two v5 upload files, verifies both manifests and pose provenance,
runs both kit preflights with Kaggle checks skipped, and rebuilds the report.
Every check prints OK or FAIL; READY means all matched `release.json`.
Logs go into a fresh directory under `data/outputs/report/`. A matching PDF is published
as `data/outputs/report/track3-abstract-62967a99.pdf`; a different existing file is never overwritten.
The command never uploads, pushes or contacts Kaggle.

Run from this checkout with the existing project environments. No installation is
performed. The three deliverable scripts default to **plan only**: they read
recorded metadata and print commands without creating run directories or importing
GPU libraries. `--run` enables execution. Commands are printed with shell quoting;
execution stops on the first failing command. Nothing is deleted by these scripts.

```bash
bash scripts/reproduce/d1_meshes_v5.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d
bash scripts/reproduce/d2_poses_v5.sh
bash scripts/reproduce/verify.sh submission/20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json
bash scripts/reproduce/verify.sh submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json
bash scripts/reproduce/d1_meshes.sh
bash scripts/reproduce/d2_poses.sh
bash scripts/reproduce/d3_report.sh
bash scripts/reproduce/verify.sh submission/20261101T1605Z-d1-v2/manifest.json
bash scripts/reproduce/verify.sh submission/20261101T1703Z-d2-v1/manifest.json
```

## D1: recorded measures → vessel-rule meshes → CD-O → pack → preflight

```bash
bash scripts/reproduce/d1_meshes.sh 'https://github.com/<owner>/<repo>/commit/<sha>' --run
# Optional: --manifest submission/<recorded-d1-v2>/manifest.json
```

The default source is the newest timestamped `submission/*-d1-v2/manifest.json`.
Its `build_run` identifies the recorded build. The script checks that its request
equals the committed `src/features/mesh_extents/requests/build-v2-r2.json`, then rebuilds
from that request's **23 recorded measure runs**. It does not regenerate depth or
measurements. The build handler writes a fresh immutable
`data/outputs/mesh_extents/build/<new-run>/run.json`; its printed path supplies the
candidate and pair directories to subsequent commands.

The kit interpreter runs `src/features/mesh_eval/local_cdo.py` on the new object meshes
(default sampling seeds from episodes 1/3/5), then the manifest's kit packing and
preflight commands. Absolute command paths are rebased to the current checkout;
mesh directory, output file and commit URL are substituted. CD-O uses public scans
only as scoring references. The recorded fitted score is **0.9251 cm**; the
cross-fitted adoption headline remains **0.92572 cm**. The CPU reproduction below
reran the fitted score and recovered **0.9251 cm**.

A fresh `data/.runtime/tmp/reproduce-d1-<unique>/` holds `build.log`, `score.log`,
`cdo.json`, `pack.log`, `preflight.log`, the Parquet and a new `manifest.json`
with commands, hashes and the new build-record path. The source manifest and
historical build are retained. D1 is CPU only, but its rebuild is still opt-in.

## D2: version-aware campaign or composition → pack → preflight

The newest timestamped `submission/*-d2-v*/manifest.json` is the default.
`--manifest` selects a particular recorded package. The script reads `version`
when present, otherwise the packed filename's `_v1`/`_v2` suffix (the historical
manifests have no explicit version key). Unsupported versions are refused.
Execution requires an explicit commit URL; plans default to the recorded URL.
Episodes 41/43/44 are refused.

### v2: CPU composition

```bash
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/c7f90116f11b41970d0e910375713037faab126f' \
  --manifest submission/20261101T2315Z-d2-v2/manifest.json --run
```

The script verifies every export's recorded `source` SHA-256 before copying any
files. It checks the recorded composition: **18 unchanged v1 exports** from
`data/outputs/pose_insurance/results-v1/` and **K1a episodes 1/28** from
`data/outputs/pose_insurance/rebuilt-v2/`. Episode 30 remains on v1. It copies those
20 sources into a new `data/.runtime/tmp/reproduce-d2-<unique>/results/`; the historical
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
bash scripts/reproduce/d2_poses.sh data/.runtime/tmp/reproduce-d2-campaign/results \
  'https://github.com/<owner>/<repo>/commit/<sha>' \
  --manifest submission/20261101T1703Z-d2-v1/manifest.json --run
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

## D3: committed report tree → fresh figures → fresh PDF

```bash
bash scripts/reproduce/d3_report.sh        # plan only, no directories created
bash scripts/reproduce/d3_report.sh --run  # CPU; always creates a fresh directory
```

The script pins HEAD, exports `docs/report/track3` with read-only `git archive`
and `tar` into `data/.runtime/tmp/reproduce-d3-<unique>/src/`, then regenerates the
mesh panel, `--pose-offsets` panel and `--bars` legacy chart there. The checked-out
figure generator reads the recorded inputs from this checkout; its only change
from the archived generator is the output-directory option. All figure drawing,
values and metadata configuration are unchanged. Its used source hash is saved in
`figure-generator.sha256`. The unused legacy `d1-v1-cdo.pdf` is preserved by
archive copy, not regenerated; both figures referenced by `main.tex` are rebuilt.

The archived `main.tex`, classes, bibliography and assets are then built with the
pinned Tectonic 0.17.0 using the exact `build_pdf.sh` engine arguments:
`-X compile -Z continue-on-errors --keep-logs`. Matplotlib, temporary files and
Tectonic's writable cache are private to the fresh work directory; the existing
`data/.cache/tectonic` is copied as a read-only input. The wrapper's same microtype
error filter is retained. `scripts/report/build_pdf.sh` needed no edit.

`SOURCE_DATE_EPOCH` comes from the last commit touching `docs/report/track3`;
commits outside that tree preserve the PDF bytes only while the figure inputs read
from `data/outputs/` and upstream, and `data/.cache/tectonic`, also stay the same.
Tectonic may need the network for fonts missing from that cache.
A commit touching any report-tree file, including a README, bumps the epoch and
changes the hash; `check_release.sh` detects the change.
Existing report PDFs, logs and other report files are never rewritten. No metrics,
GPU operation, install or upload is performed.

### Executed twice, 2026-11-02

The final script was run twice serially by a resumable two-entry ledger script;
both invocations of `bash scripts/reproduce/d3_report.sh --run` exited **0**.
Both used commit `9fb178a3fc74435ea193524cfddc15f753f4ea6a` and
`SOURCE_DATE_EPOCH=1793578088` (2026-11-02 00:08:08 UTC).

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B \
  data/.runtime/tmp/reproduce-d3-audit-eTmAqxDt/run_reproductions.py
PYTHONDONTWRITEBYTECODE=1 data/.runtime/linux-cuda/bin/python -B \
  data/.runtime/tmp/reproduce-d3-audit-eTmAqxDt/compare_pdfs.py
bash -n scripts/reproduce/d3_report.sh
bash -n scripts/report/build_pdf.sh
```

The first command resumes completed entries; each new invocation of the D3 shell
script itself always creates a fresh work directory. Shell syntax passed; the
edited figure generator also passed `py_compile` and a CPU import. The final
report has **3 pages: 2 main-text pages plus 1 references-only page**, with both
figures. The only TeX error remains the allowed XeTeX/microtype tracking error;
there are no overfull boxes or undefined references.

SHA-256 comparison of every figure PDF and the report (the `main.pdf` reference
is the prior in-place build, which is untracked, rather than a committed PDF):

| PDF | Committed / prior SHA-256 | Run 1 SHA-256 | Run 2 SHA-256 | Result |
| --- | --- | --- | --- | --- |
| figures/d1-v1-cdo.pdf | `65716f91a28826bcbd7c0621e0a6f3e2cf197a919d1c7f1c1abf70f996bfafc9` | `65716f91a28826bcbd7c0621e0a6f3e2cf197a919d1c7f1c1abf70f996bfafc9` | `65716f91a28826bcbd7c0621e0a6f3e2cf197a919d1c7f1c1abf70f996bfafc9` | archive copy; unused legacy figure |
| figures/d1-v2-cdo.pdf | `c3ec50578a5fdfae0833895672119b3bbd4895e4c6074acf2d0a4b43fb94b70f` | `ebeb0edd890afd1050ba6894a06d1861809cd3f0a10afa0ef748b4cd28c4ecb9` | `ebeb0edd890afd1050ba6894a06d1861809cd3f0a10afa0ef748b4cd28c4ecb9` | CreationDate and startxref offset only |
| figures/d1-v2-qualitative.pdf | `53f6c1842a34c458e7b923985d6b66f2bc8944b0e640580a06ededd757b65e8a` | `3a3008a81143ce215de0969f0285fb00f9e2c8c22cf84e88224b96b8d19d6666` | `3a3008a81143ce215de0969f0285fb00f9e2c8c22cf84e88224b96b8d19d6666` | CreationDate and startxref offset only |
| figures/d2-slot1-offset.pdf | `71e9e9a64c09ca43f0953dd02d564cd79527744078f9b60759ccf3d63ddf71df` | `71e9e9a64c09ca43f0953dd02d564cd79527744078f9b60759ccf3d63ddf71df` | `71e9e9a64c09ca43f0953dd02d564cd79527744078f9b60759ccf3d63ddf71df` | byte-identical |
| main.pdf | `62117515559c989bc1a58edf0667175ad1c5cf5714031b748ef0f29d7fb27e37` | `476c27e70900550b606c2c494534dad6f03cb12d248a41f15baf55fc61c91170` | `476c27e70900550b606c2c494534dad6f03cb12d248a41f15baf55fc61c91170` | CreationDate in object stream 11; compressed length grows by one byte; trailer ID and xref/startxref offset change; all other objects byte-identical |

Both runs match **every PDF byte-for-byte**. The historical differences are fully
accounted for:

- The two D1 v2 figure PDFs change only `CreationDate` and the `startxref` byte
  offset: the old date strings `D:20261102031035+10'00'` (bars) and
  `D:20261102084334+10'00'` (mesh panel) become `D:20261102000808Z`. The new
  string is six bytes shorter; `startxref` decreases by six. Normalising exactly
  those two fields makes the original/fresh bytes identical. Fonts and drawing
  streams do not change. The pose-offset PDF already omits CreationDate/ModDate
  and matches the committed bytes exactly.
- The prior report's CreationDate `D:20261102000438-00'00'` becomes
  `D:20261102000808-00'00'` inside object stream 11. Its compressed length grows
  from **5,653 to 5,654 bytes**. Of **55 direct objects**, only stream 11 and the
  cross-reference stream 206 change. Decompressing stream 11 proves its sole
  content difference is that date; all other direct objects, including fonts,
  page content and embedded figures, are byte-identical. Stream 206 changes the
  PDF ID and its own byte offset by +1; `startxref` changes **341,794 → 341,795**.
  Thus the report's historical hash difference is metadata/offsets only, with
  identical page content. No historical PDF was rewritten to force a hash match.

Final artifacts and evidence:

| Item | Path |
| --- | --- |
| Run 1 PDF | `data/.runtime/tmp/reproduce-d3-38r6BKup/src/docs/report/track3/main.pdf` |
| Run 2 PDF | `data/.runtime/tmp/reproduce-d3-1cmKxGAh/src/docs/report/track3/main.pdf` |
| Per-run records | Each run directory: `build.log`, `commit.txt`, `source-date-epoch.txt`, `figure-generator.sha256`, `sha256.txt`; figures and TeX logs under `src/docs/report/track3/` |
| Audit | `data/.runtime/tmp/reproduce-d3-audit-eTmAqxDt/`: `ledger.jsonl`, `summary.json`, `run-{1,2}.log`, `comparison.json`, `hash-table.md`, `syntax.log`, `pdfinfo.log`, `tex-evidence.log` |

The audit hashes the original report tree before/after the runs and confirms no
file changed during execution. The only report-tree edit is the allowed
`make_figures.py` output-directory option. Two preliminary builds before the
final path-handling adjustment also succeeded; their evidence is retained under
`data/.runtime/tmp/reproduce-d3-audit-xeEJsNaE/`. The complete generated/source file
inventory is `data/.runtime/tmp/reproduce-d3-audit-eTmAqxDt/files-changed.txt`.

## Run records and hash coverage

| Deliverable | Recorded source |
| --- | --- |
| D1 | `submission/20261101T1605Z-d1-v2/{manifest,run}.json`; the manifest's `build_run` and its request's measure producers |
| D2 | v1: `submission/20261101T1703Z-d2-v1/{manifest,run}.json`, loop ledger/completion; v2: `submission/20261101T2315Z-d2-v2/{manifest,run}.json`, per-export `source`/SHA-256 and `composition`, `data/outputs/pose_insurance/v2/report.md` |
| D3 | `docs/report/track3/README.md` provenance table; `docs/report/track3/main.log`; new command/output log when executed |

`verify.sh <manifest.json>` is read-only and CPU only. It accepts either a top-level
`file` + `sha256` or `file: {path, sha256}`, and streams SHA-256 for the packed file
and every explicit nested path/file + sha256 entry. If an original path is missing,
`scripts/reproduce/moved-paths.json` redirects it to the moved file with the same
pinned digest and prints the relocation. Missing files or differing digests remain
`MISMATCH`. Exit status is 0 when all match, 1 on any mismatch. Paths must resolve
inside this checkout. It does not consult Git or interpret source-hash dictionaries.

Coverage is exactly what the supplied manifest records; it does not recursively
follow unhashed run-record paths. In particular, the old D1 manifest lists only
the packed-file hash, so its **one** matching entry does not independently verify
all 23 measure producers. The build handler admits its recorded inputs on rebuild.
Hash agreement also does not establish numerical or bitwise reproducibility of a
new GPU run. Align the checked-out sources with the recorded source hashes before
claiming an exact historical reproduction.

## Current verify results

Checked on 2026-11-04 with `bash scripts/reproduce/verify.sh <manifest>` for every
`*manifest*.json` under `submission/`, including `manifest-final.json`.
Manifest paths below are relative to `submission/`. These checks hash the current
checkout and preserved artifacts; they do not run historical reproductions.

| Manifest | Result (OK / entries) | Mismatches at HEAD |
| --- | --- | --- |
| `20261101T1605Z-d1-v2/manifest.json` | PASS: 1/1 | None |
| `20261101T1703Z-d2-v1/manifest.json` | FAIL: 29/35 | `src/features/pose_insurance/__main__.py`, `src/features/pose_insurance/_scene.py`, `src/features/pose_insurance/handler.py`, `src/features/pose_insurance/loop.py`, `src/features/pose_insurance/pack.py`, `src/features/pose_insurance/scene.py` |
| `20261101T2315Z-d2-v2/manifest.json` | FAIL: 103/107 | `src/features/pose_insurance/__main__.py`, `src/features/pose_insurance/_placement.py`, `docs/chain/pose-placement-protocol.md`, `scripts/reproduce/verify.sh` |
| `20261102T0032Z-d2-v3/manifest-final.json` | PASS: 42/42 | None |
| `20261102T0032Z-d2-v3/manifest.json` | FAIL: 493/500 | `src/features/pose_insurance/__main__.py`, `src/features/pose_insurance/_placement.py`, `src/features/pose_insurance/runtime.json`, `src/features/stereo_depth/runtime.json`, `scripts/reproduce/d2_poses.sh`, `scripts/reproduce/README.md`, `scripts/reproduce/verify.sh` |
| `20261102T0313Z-d1-v3/manifest.json` | PASS: 1/1 | None |
| `20261102T0516Z-d1-v4/manifest-final.json` | FAIL: 2/3 | `scripts/reproduce/d1_meshes.sh` |
| `20261102T0516Z-d1-v4/manifest.json` | FAIL: 5/6 | `scripts/reproduce/d1_meshes.sh` |
| `20261103T0645Z-d2-v4/track3_poses_v4_final.manifest.json` | PASS: 60/60 | None |
| `20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json` | PASS: 69/69 | None |
| `20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json` | PASS: 4/4 | None |

The expected legacy mismatches are source files, runtime flags, protocol text
and reproduction scripts that later commits changed. Use each older pack's
recorded commit for historical reproduction; these HEAD mismatches are not
failures of the v5 upload files. Both v5 manifests pass, and `check_release.sh`
also hashes the actual files under `submission/` against the release digests.
The build-v4 request is accepted at its moved path with its original digest.
Full command logs: `data/outputs/report/critic-checks-20261104T012723Z/`.

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
  --manifest submission/20261101T2315Z-d2-v2/manifest.json --run
bash -n scripts/reproduce/d1_meshes.sh
bash -n scripts/reproduce/d2_poses.sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' data/.runtime/linux-cuda/bin/python -B \
  data/.runtime/tmp/ctx05-d5-evidence/check_scripts.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' data/.runtime/kit-env/bin/python -B \
  data/.runtime/tmp/ctx05-d5-evidence/compare_deliverables.py
bash scripts/reproduce/verify.sh submission/20261101T1605Z-d1-v2/manifest.json
bash scripts/reproduce/verify.sh submission/20261101T2315Z-d2-v2/manifest.json
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
`data/outputs/mesh_extents/build/20261102T000114.850697Z-1a4c354e/run.json`.
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
| D1 reproduction | `data/.runtime/tmp/reproduce-d1-wbr3u3xl/`: `console.log`, `build.log`, `score.log`, `cdo.json`, `pack.log`, `preflight.log`, `track3_meshes_v2.parquet`, `manifest.json`, `comparison.json` |
| D2 reproduction | `data/.runtime/tmp/reproduce-d2-wi7g6bmn/`: `console.log`, `sources.json`, `results/`, `pack.log`, `preflight.log`, `command_results.json`, `track3_poses_v2.csv`, `manifest.json`, `comparison.json` |
| Comparisons and syntax | `data/.runtime/tmp/ctx05-d5-evidence/`: `comparisons.log`, `syntax.log`, `script-checks.json`, `script-checks.log`, `v1-plan-{old,new}.log`, comparison/check scripts |
| Source-manifest verification | `data/.runtime/tmp/ctx05-d5-evidence/verify-{d1,d2}-v2.log` |

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
**D3 fresh-directory reproduction is recorded above.** Placeholder URLs remain nonfinal;
Kaggle authentication/checks were skipped and no upload was attempted. The D1
preflight also retains the recorded `no --results given` warning.

## D2 v3: stereo-scale hybrid composition

D2 v3 uses the v2 CPU composition path with the manifest's accepted `S1` exports
and unchanged `v2` exports. It verifies every source before copying, then runs the
official packer and preflight. It also requires the reproduced CSV hash to equal
the recorded v3 hash. This reproduces the package from evaluated exports; it does
not repeat stereo estimation, training or the physical evaluator.

```bash
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/bb8a3f02fac668e86ff7b6a3ad7c3ccaec40050d' \
  --manifest submission/<stamp>-d2-v3/manifest.json --run
```

New v3 reproduction artifacts are written beneath
`data/.runtime/tmp/ctx05-scored-reproduce-d2-<new>/`. The v3 campaign evidence and final
hash comparison are recorded in `data/outputs/pose_insurance/v3/report.md`.
