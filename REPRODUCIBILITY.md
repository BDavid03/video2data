# Reproducibility

How to rebuild, check and trace every file in this Track 3 entry. The [README](README.md) explains what the
entry does; this page is the checklist. Every command runs from the checkout root on Linux with an NVIDIA GPU.

## 1. Inputs that are pinned

| Input | Where it is pinned | Value |
| --- | --- | --- |
| Dataset `nvidia/video_to_data_challenge` (track_3 only) | `data/configs/sources.lock.json` | commit `5f68335f3acc802033d1e80728c1633197521de8` |
| Dataset file hashes (169 files) | `data/manifests/dataset/inventory.json` | re-verified 4 Nov 2026, no change (`data/dataset-refresh/20261104T203500Z/diff.md`) |
| Submission kit (organisers') | `data/.runtime/kit-20261102/v2d_submission_kit/KIT_MANIFEST.json` | archive SHA-256 `7e1fdd00…77f0`, all 61 files match |
| FlashCHORD runtime | `data/configs/sources.lock.json` | commit `aee95f25ae0943f335d8bb42fa95b6a9acfa266f` |
| Train / tuning / holdout split | `data/configs/splits/public-v1.json` | train 0/11/12/18/21/23/26/31, tuning 39/42, holdout 43/44; episode 41 excluded |

Scored episodes have no labels. Public labels were read only on the train and tuning episodes above.

## 2. Environments

```bash
bash scripts/linux/install-linux-cuda.sh   # project venv  data/.runtime/linux-cuda  (Python 3.13, torch 2.7.1+cu128)
bash scripts/linux/install-kit-env.sh      # kit scorer    data/.runtime/kit-env
bash scripts/linux/install-flashchord.sh   # RL runtime    data/.runtime/flashchord/venv
```

Each script is idempotent and logs under `data/reports/installation/`. FoundationPose (frame-zero placement) has its
own install, described in [docs/INSTALL.md](docs/INSTALL.md). Mask models (SAM2.1, Grounding DINO) live in
`data/.runtime/sam2/` and `data/.runtime/grounding-dino/` with their own `INSTALL.md`.

## 3. Files we uploaded

The ledger is `docs/submissions/counted.json` (one row per uploaded or packed file, with SHA-256, size, code commit
and Kaggle result). Manifests are archived under `docs/submissions/manifests/`. Checkpoints behind each counted file
are in `models/policies/<pack>/` with a `SHA256SUMS`.

| File | SHA-256 | Code commit | Status |
| --- | --- | --- | --- |
| `track3_combined_v5.parquet` (moving v5: episode 38 moving, 19 static) | `b476eb1c2bc1ee6fab2df72f42f377d87e5d265c06140f3e47fc2fc67f8ff81d` | `c8ff8c13` | uploaded 4 Nov 14:28 UTC, the counted file (AUC 0.32339) |
| `track3_combined_v5.parquet` (all static) | `ddfc98b3e6d9bd10b31092903fdc93e1e22624d1f2ddc545b1d8f64c4ebf9342` | `02117aca` | uploaded, fallback |
| `track3_meshes_v5_final.parquet` | `bcb93d57d86b4dac06ed21af9af84ce39edffe52b3d9c35e28928391e47e4e58` | `b1617234` | uploaded, CD-O 0.53562 |
| `track3_combined_fp.parquet` (episode 25 frame-zero placement) | `dc7729413e2f3df74c9250e8b7edfb8194a707b74408e4a03820d0639d541ffb` | `07fdf671` | packed, not uploaded |

Check them without rebuilding anything:

```bash
bash scripts/reproduce/check_counted.sh            # OK / MISSING / CHANGED for every ledger file and checkpoint
bash scripts/reproduce/verify.sh submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
```

## 4. Rebuild

Every builder refuses changed sources: check out the commit recorded in the pack's manifest first. Use the full
commit URL; it is written into the packed file.

```bash
# plans only; add --run to execute
bash scripts/reproduce/d1_meshes_v5.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d
bash scripts/reproduce/d2_poses_v5.sh
bash scripts/reproduce/d2_poses_moving.sh --verify-manifest submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
bash scripts/reproduce/d3_report.sh                 # archives docs/report/track3, rebuilds figures and the PDF
bash scripts/reproduce/check_release.sh --combined submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
```

`check_release.sh` prints one line per check and ends with READY or NOT READY. Logs land in
`data/outputs/report/release-*/`. Details and the per-script limits are in
[scripts/reproduce/README.md](scripts/reproduce/README.md).

Frame-zero placement (public scans registered to the first stereo frame, then a static policy retrained from that
start) is the `src/features/pose_refresh/` slice: joint-box masks and registration on the train pairs
(`train_frame0.py`), limits from those pairs (`calibrate.py`, rule in `data/outputs/refresh/<stamp>/protocol.md`
sealed by `protocol.sha256` before any result), retraining (`frame0_fp_retrain.py`) and packing (`pack.py`).

## 5. The technical report

Source `docs/report/track3/main.tex`; the built PDF is tracked at
[docs/report/final/track3-technical-report.pdf](docs/report/final/track3-technical-report.pdf), and its SHA-256 is
pinned in `scripts/reproduce/release.json` (`report.sha256`). `d3_report.sh --run` rebuilds it with a fixed
`SOURCE_DATE_EPOCH`; `check_release.sh` compares the rebuilt bytes with the pin.

## 6. Run records

Every operation writes its request snapshot, hashed inputs and sources, status, timing and artifact hashes under
`data/outputs/<slice>/…/run.json`. Consumers re-hash what they read. Rules written before a run and the verdicts
after it are `docs/chain/*-protocol.md` and `*-outcome.md`. Dated working notes are in `docs/chain/handoffs/`.

## 7. What is not in git

`data/outputs/`, `data/.runtime/`, `data/.cache/`, `models/`, `submission/`, `upstream/` and the dataset snapshot
are ignored (`.gitignore`). The hashes in the manifests and the ledger let anyone with those files confirm they are
the ones we used.
