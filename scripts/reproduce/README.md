# Reproduce Track 3 files

These scripts check and rebuild the files we submitted (README steps 2, 3, 5 and 6: shapes,
static poses, the submitted file, and the report).

Words used here:

- A **pack** is the single Parquet or CSV file we upload. Packing turns meshes and poses into it.
- A **manifest** is the JSON file written beside a pack. It records the SHA-256 hash of the pack and
  of every input and source file used to make it.
- **Preflight** is the organisers' format check (`tools/preflight.py` in the submission kit). It must
  print `Ready to submit`.
- **CD-O** is the shape score (object Chamfer distance, cm, lower is better). **AUC** is the main pose
  score (area under the success curve, higher is better).

Run every command from the checkout root with the existing environments. Nothing here uploads,
pushes, contacts Kaggle or deletes anything. The D1, D2 and D3 scripts only print a plan by default.
Add `--run` to execute. They stop at the first failing command.

## What is in `submission/`

| Folder | File | Status |
| --- | --- | --- |
| `20261104T1422Z-d2-moving-v5/` | `track3_combined_v5.parquet` (`b476eb1c`) | Counted file. Episode 38 moving, 19 episodes static |
| `20261104T1316Z-d2-combined-v5/` | `track3_combined_v5.parquet` (`ddfc98b3`) | All-static fallback, also uploaded |
| `20261104T1403Z-d2-moving-v5/` | (`1570037a`) | Earlier moving pack, never uploaded |
| `20261104T1418Z-d2-moving-v5/` | (`623a1a8a`) | Earlier moving pack, never uploaded |

Older packs (D1 v2 to v5, D2 v1 to v5) are not in this checkout's `submission/` folder. The commands
below that read them need those folders put back first.

## Check the submitted file

```bash
bash scripts/reproduce/verify.sh submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
bash scripts/reproduce/check_release.sh --combined submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
# the all-static fallback
bash scripts/reproduce/check_release.sh --combined submission/20261104T1316Z-d2-combined-v5/track3_combined_v5.manifest.json
```

`check_release.sh` checks the files that `release.json` names, then rebuilds the report. With no
arguments it hashes the separate D1 v5 mesh pack and D2 v5 pose pack, verifies both manifests and the
pose provenance, and runs both preflights with Kaggle checks skipped. `--combined` also checks that
manifest's provenance and file hash and runs the current kit's preflight on it. Each line prints OK or
FAIL. READY means everything matched `release.json`. READY does not mean the boards accept the file:
D2 v5 passed it and was rejected on all five pose boards.

Logs go to a new folder under `data/outputs/report/`. If the rebuilt PDF matches, it is written to the
path `release.json` names (`data/outputs/report/track3-technical-report-08021ded.pdf`). An existing
different file is never overwritten.

`d2_poses_moving.sh` runs the moving-pose packer (`src.features.pose_insurance.pack_moving`) and passes
its arguments through: `--request`, `--code-commit`, `--out`, `--dry-run`, `--verify-manifest`.

## Rules that apply to every rebuild

- Use the full commit URL. It is written into the pack, so a short one changes the bytes.
- The scripts refuse changed sources. Older packs need their recorded commit checked out.
- `--historical` (sets `V2D_HISTORICAL=1`; `check_release.sh` sets it itself) reads frozen sources from
  the commit recorded before the folder layout change, and reports the current file as drift.
- Episodes 41, 43 and 44 are refused.

## D1: meshes (step 2)

```bash
bash scripts/reproduce/d1_meshes_v5.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d
bash scripts/reproduce/d1_meshes.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d --version v4
bash scripts/reproduce/d1_meshes.sh 'https://github.com/<owner>/<repo>/commit/<sha>' --run   # v2
# optional: --manifest submission/<recorded-d1-v2>/manifest.json
```

`d1_meshes_v5.sh` repacks the generated-shape tree that passed its checks. Default output:
`data/outputs/mesh_generative/g2-real-pack/track3_meshes_v5.parquet`.

`d1_meshes.sh` reads the newest `submission/*-d1-<version>/manifest.json`. It checks that the build
request equals `src/features/mesh_extents/requests/build-v2-r2.json`, then rebuilds from that request's
23 recorded measure runs on CPU. It does not redo depth or measurements. The kit interpreter then runs
`src/features/mesh_eval/local_cdo.py` (sampling seeds from episodes 1, 3 and 5), then the manifest's
pack and preflight commands. Public scans are used only as scoring references. Output goes to a new
`data/.runtime/tmp/reproduce-d1-<unique>/` with logs, `cdo.json`, the Parquet and a new `manifest.json`.

The recorded fitted score is 0.9251 cm and the rebuild recovered it. The cross-fitted figure is
0.92572 cm. Neither is a host score.

## D2: static poses (step 3)

```bash
bash scripts/reproduce/d2_poses_v5.sh     # verify the frozen v5 sources; --run packs
bash scripts/reproduce/d2_poses.sh \
  'https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/c7f90116f11b41970d0e910375713037faab126f' \
  --manifest submission/20261101T2315Z-d2-v2/manifest.json --run
```

`d2_poses_v5.sh` runs `src.features.pose_insurance.scored_stereo_pack --version v5` on CPU.

`d2_poses.sh` defaults to the newest `submission/*-d2-v*/manifest.json`; `--manifest` picks another.
It refuses unknown versions, and `--run` needs an explicit commit URL.

- **v2 (CPU).** Checks each recorded source hash, then copies 18 v1 exports from
  `data/outputs/pose_insurance/results-v1/` and episodes 1 and 28 from
  `data/outputs/pose_insurance/rebuilt-v2/` into a new `data/.runtime/tmp/reproduce-d2-<unique>/results/`.
  Episode 30 stays on v1. Preflight must report `Ready to submit`. This rebuilds the file from saved
  rollouts. It does not rerun the GPU.
- **v3 (CPU).** Same path with the accepted stereo-scale exports. The rebuilt CSV hash must equal the
  recorded one. Output: `data/.runtime/tmp/ctx05-scored-reproduce-d2-<new>/`. Record:
  `data/outputs/pose_insurance/v3/report.md`.
- **v1 (GPU).** Pass a new empty results folder, a commit URL and
  `--manifest submission/20261101T1703Z-d2-v1/manifest.json`. Budget from the manifest: 512 steps, one
  world, replay minimum 32, 20 evaluation episodes. It uses current code, and has never been rerun end
  to end.

## D3: report (step 6)

```bash
bash scripts/reproduce/d3_report.sh        # plan only
bash scripts/reproduce/d3_report.sh --run  # CPU, always a new folder
```

The script exports the committed `docs/report/track3` into
`data/.runtime/tmp/reproduce-d3-<unique>/src/`, runs `figures/make_figures.py` on recorded inputs, and
builds the PDF with Tectonic 0.17.0. Tectonic may need the network for fonts missing from
`data/.cache/tectonic`. `SOURCE_DATE_EPOCH` comes from `release.json` (`report.epoch`). Two runs on
2 November 2026 were byte-identical. Audit: `data/.runtime/tmp/reproduce-d3-audit-eTmAqxDt/`.

## What `verify.sh` covers

`verify.sh <manifest.json>` is read-only and CPU only. It hashes the pack and every path or file entry
the manifest records. If a recorded path has moved, `moved-paths.json` points it to the moved file with
the same hash. Exit 0 means all match; exit 1 means at least one mismatch.

It checks only what the manifest records. It does not follow unhashed paths in run records. The old D1
manifest lists only the pack, so it does not verify the 23 measure runs. Matching hashes do not show
that a new GPU run gives the same numbers.

Recorded results on 4 November 2026, before the layout change (paths under `submission/`; logs in
`data/outputs/report/critic-checks-20261104T012723Z/`):

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

The mismatches are source files, flags, protocol text and scripts changed by later commits. They do
not affect the v5 packs. After the layout change, D2 v5 gives 68/69 without `--historical` (the layout
change rewrote `d2_poses_v5.sh`) and 69/69 with it. Combined v5 gives 130/130 without the flag.

## CPU rebuild record, 2 November 2026

Both real-data `--run` paths exited 0 and both preflights printed `Ready to submit`.

| File | SHA-256 | Result |
| --- | --- | --- |
| D1 v2 Parquet | `43fa6ac02353520aac4d019f7d6d45e0aab923f1c8f43d6cf03e30c242ba95c2` | Byte-identical, 294,912 rows |
| D2 v2 CSV | `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741` | Byte-identical, 9,120 rows |

All 47 D1 PLY files matched the original build. D2 matched 20/20 source and copy hashes. Evidence:
`data/.runtime/tmp/reproduce-d1-wbr3u3xl/`, `data/.runtime/tmp/reproduce-d2-wi7g6bmn/` and
`data/.runtime/tmp/ctx05-d5-evidence/`.

Known gaps:

- We have not rerun D2 training on the GPU (the episode 1 and 28 runs and the v1 campaign). The CPU
  rebuild shows the file can be rebuilt from saved rollouts. It does not show retraining gives the
  same numbers.
- At that time the D2 v2 source check gave 106/107: `docs/chain/pose-placement-protocol.md` differed
  from its recorded hash.
- Those runs used placeholder URLs and skipped Kaggle checks. D1 preflight also warned
  `no --results given`.
