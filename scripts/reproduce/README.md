# Reproduce Track 3 deliverables

Run from the checkout with the existing environments. Nothing here uploads, pushes or
contacts Kaggle, and nothing is deleted. The D1, D2 and D3 scripts print a plan by
default and create nothing. Add `--run` to execute; they stop at the first failing command.

## Release check

```bash
bash scripts/reproduce/check_release.sh
bash scripts/reproduce/check_release.sh --combined submission/20261104T1316Z-d2-combined-v5/track3_combined_v5.manifest.json
```

The first form hashes the two v5 files, verifies both manifests and the pose provenance, runs both preflights from the 30 September kit with Kaggle checks skipped, and rebuilds the report. Its READY does not show board acceptance: D2 v5 passed it and was rejected on all five pose boards. `--combined` also checks that manifest's provenance and file hash and runs the current
kit's preflight on it. Each check prints OK
or FAIL, and READY means everything matched `release.json`. Logs go to a new directory
under `data/outputs/report/`. A matching PDF is published as
the path `scripts/reproduce/release.json` names (`data/outputs/report/track3-technical-report-e63bffef.pdf` since the moving v5 release; `62967a99` was the static abstract); a different existing file is never
overwritten.

## All commands

```bash
bash scripts/reproduce/d1_meshes_v5.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d
bash scripts/reproduce/d2_poses_v5.sh
bash scripts/reproduce/verify.sh submission/20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json
bash scripts/reproduce/verify.sh submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json
bash scripts/reproduce/d1_meshes.sh
bash scripts/reproduce/d1_meshes.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d --version v4
bash scripts/reproduce/d2_poses.sh
bash scripts/reproduce/d3_report.sh
bash scripts/reproduce/verify.sh submission/20261101T1605Z-d1-v2/manifest.json
bash scripts/reproduce/verify.sh submission/20261101T1703Z-d2-v1/manifest.json
```

`d2_poses_moving.sh` runs the moving-pose pack, `src.features.pose_insurance.pack_moving`,
and passes its arguments through (`--request`, `--code-commit`, `--out`, `--dry-run`,
`--verify-manifest`).

## D1: recorded measures to meshes, CD-O, pack, preflight

```bash
bash scripts/reproduce/d1_meshes.sh 'https://github.com/<owner>/<repo>/commit/<sha>' --run
# Optional: --manifest submission/<recorded-d1-v2>/manifest.json
```

The default source is the newest `submission/*-d1-v2/manifest.json`. The script checks
that its build request equals `src/features/mesh_extents/requests/build-v2-r2.json`, then
rebuilds from that request's 23 recorded measure runs. It does not regenerate depth or
measurements. It runs on CPU.

The kit interpreter then runs `src/features/mesh_eval/local_cdo.py` (sampling seeds from
episodes 1, 3 and 5), followed by the manifest's pack and preflight commands. CD-O uses
public scans only as scoring references. Output goes to a new
`data/.runtime/tmp/reproduce-d1-<unique>/` with logs, `cdo.json`, the Parquet and a new
`manifest.json`.

The recorded fitted score is 0.9251 cm and the reproduction recovered it. The cross-fitted
adoption figure is 0.92572 cm. The fitted score is neither a cross-fitted assessment nor a
host score.

## D2: campaign or composition to pack and preflight

The default is the newest `submission/*-d2-v*/manifest.json`; `--manifest` picks another.
Unsupported versions are refused, `--run` needs an explicit commit URL, and episodes 41,
43 and 44 are refused.

### v2: CPU composition

```bash
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/c7f90116f11b41970d0e910375713037faab126f' \
  --manifest submission/20261101T2315Z-d2-v2/manifest.json --run
```

This checks each recorded source hash, then copies 18 v1 exports from
`data/outputs/pose_insurance/results-v1/` and K1a episodes 1 and 28 from
`data/outputs/pose_insurance/rebuilt-v2/` into a new
`data/.runtime/tmp/reproduce-d2-<unique>/results/`. Episode 30 stays on v1. Preflight must
exit 0 and report `Ready to submit`. This reproduces the composition from evaluated
exports. It does not regenerate their GPU trajectories.

### v3: stereo-scale composition

```bash
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/bb8a3f02fac668e86ff7b6a3ad7c3ccaec40050d' \
  --manifest submission/<stamp>-d2-v3/manifest.json --run
```

Same path as v2, using the accepted `S1` and unchanged `v2` exports, and it also requires
the rebuilt CSV hash to equal the recorded one. It does not repeat stereo estimation,
training or the physical evaluator. Output goes under
`data/.runtime/tmp/ctx05-scored-reproduce-d2-<new>/`; the campaign record is
`data/outputs/pose_insurance/v3/report.md`.

### v1: frozen loop campaign (GPU)

```bash
bash scripts/reproduce/d2_poses.sh data/.runtime/tmp/reproduce-d2-campaign/results \
  'https://github.com/<owner>/<repo>/commit/<sha>' \
  --manifest submission/20261101T1703Z-d2-v1/manifest.json --run
```

Use a new, empty results directory and a free GPU. The budget comes from the manifest:
512 steps, one world, replay minimum 32, on 20 evaluation episodes. It uses the current
slice code, so changing a path does not reconstruct historical code. The v1 campaign has
never been re-executed by this script; only its four printed commands were compared with
the original script.

## D3: report to figures and PDF

```bash
bash scripts/reproduce/d3_report.sh        # plan only
bash scripts/reproduce/d3_report.sh --run  # CPU, always a new directory
```

The script pins HEAD, exports `docs/report/track3` into
`data/.runtime/tmp/reproduce-d3-<unique>/src/`, regenerates the figures, and builds the PDF
with Tectonic 0.17.0. `make_figures.py` reads recorded inputs from this checkout. Tectonic
may need the network for fonts missing from `data/.cache/tectonic`.

`SOURCE_DATE_EPOCH` comes from the last commit touching `docs/report/track3`. Commits
elsewhere keep the PDF bytes only while the figure inputs under `data/outputs/` and
upstream, and `data/.cache/tectonic`, stay the same. A commit touching any file in the
report tree, including a README, changes the epoch and the hash. `check_release.sh` detects it.

Two runs on 2 November 2026 at commit `9fb178a` were byte-identical to each other. They
differ from earlier builds only in dates, offsets and the PDF ID. That report had 3 pages;
the only TeX error was the allowed XeTeX/microtype one. Audit records:
`data/.runtime/tmp/reproduce-d3-audit-eTmAqxDt/`.

## What verify.sh covers

`verify.sh <manifest.json>` is read-only and CPU only. It hashes the packed file and every
explicit nested path or file entry with a SHA-256. If an original path is missing,
`moved-paths.json` redirects it to the moved file with the same digest. It exits 0 when
all match and 1 on any mismatch.

Coverage is only what the manifest records. It does not follow unhashed run-record paths.
The old D1 manifest lists only the packed file, so its one matching entry does not verify
the 23 measure producers. Matching hashes also do not show that a new GPU run would
reproduce numerically. Align your sources with the recorded hashes before claiming an
exact historical reproduction.

`verify.sh --historical <manifest>` (`V2D_HISTORICAL=1`, which `check_release.sh` sets) reads frozen
sources from the recorded commit from before the folder layout change. Without it, the D2 v5 manifest now
gives 68/69, because the layout change rewrote `scripts/reproduce/d2_poses_v5.sh`. With it, the script is
reported as drift and its frozen copy matches. The combined v5 manifest gives 130/130 without the flag.

Results below are from 2026-11-04, before the layout change, with paths relative to `submission/`. Logs:
`data/outputs/report/critic-checks-20261104T012723Z/`.

| Manifest | OK / entries | Mismatches |
| --- | --- | --- |
| `20261101T1605Z-d1-v2/manifest.json` | 1/1 | None |
| `20261101T1703Z-d2-v1/manifest.json` | 29/35 | `src/features/pose_insurance/__main__.py`, `_scene.py`, `handler.py`, `loop.py`, `pack.py`, `scene.py` |
| `20261101T2315Z-d2-v2/manifest.json` | 103/107 | `src/features/pose_insurance/__main__.py`, `_placement.py`, `docs/chain/pose-placement-protocol.md`, `scripts/reproduce/verify.sh` |
| `20261102T0032Z-d2-v3/manifest-final.json` | 42/42 | None |
| `20261102T0032Z-d2-v3/manifest.json` | 493/500 | `src/features/pose_insurance/__main__.py`, `_placement.py`, `runtime.json`, `src/features/stereo_depth/runtime.json`, `scripts/reproduce/d2_poses.sh`, `scripts/reproduce/README.md`, `scripts/reproduce/verify.sh` |
| `20261102T0313Z-d1-v3/manifest.json` | 1/1 | None |
| `20261102T0516Z-d1-v4/manifest-final.json` | 2/3 | `scripts/reproduce/d1_meshes.sh` |
| `20261102T0516Z-d1-v4/manifest.json` | 5/6 | `scripts/reproduce/d1_meshes.sh` |
| `20261103T0645Z-d2-v4/track3_poses_v4_final.manifest.json` | 60/60 | None |
| `20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json` | 69/69 | None |
| `20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json` | 4/4 | None |

The mismatches are source files, flags, protocol text and scripts that later commits
changed. They do not affect the packed v5 files. Use an older pack's recorded commit to
reproduce it.

## CPU reproduction record, 2 November 2026

Both real-data `--run` paths exited 0 and both preflights reported `Ready to submit`.

| File | SHA-256 | Result |
| --- | --- | --- |
| D1 v2 Parquet | `43fa6ac02353520aac4d019f7d6d45e0aab923f1c8f43d6cf03e30c242ba95c2` | Byte-identical, 294,912 rows |
| D2 v2 CSV | `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741` | Byte-identical, 9,120 rows |

All 47 D1 PLY files matched the historical build. The fresh build is
`data/outputs/mesh_extents/build/20261102T000114.850697Z-1a4c354e/run.json`. D2 verified
20/20 source and copy hashes. Evidence: `data/.runtime/tmp/reproduce-d1-wbr3u3xl/`,
`data/.runtime/tmp/reproduce-d2-wi7g6bmn/` and `data/.runtime/tmp/ctx05-d5-evidence/`.

Known gaps:

- D2 GPU regeneration has not been executed, including the K1a runs for episodes 1 and 28
  and the v1 campaign. The CPU result shows reproduction from recorded exports. It does
  not show retraining or numerical GPU reproducibility.
- At the time, the D2 v2 source-manifest check gave 106/107: the pinned
  `docs/chain/pose-placement-protocol.md` differed from the package's recorded hash, so a
  fully matching historical source tree is not established.
- Those runs used placeholder URLs, which are not final, and skipped Kaggle checks. The D1
  preflight also recorded a `no --results given` warning.
