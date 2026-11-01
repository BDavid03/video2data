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
cross-fitted adoption headline remains **0.92572 cm**. No score was rerun here.

A fresh `.runtime/tmp/reproduce-d1-<unique>/` holds `build.log`, `score.log`,
`cdo.json`, `pack.log`, `preflight.log`, the Parquet and a new `manifest.json`
with commands, hashes and the new build-record path. The source manifest and
historical build are retained. D1 is CPU only, but its rebuild is still opt-in.

## D2: frozen evaluation queue → official exports → pack → preflight

```bash
bash scripts/reproduce/d2_poses.sh outputs/pose_insurance/results-v1 \
  'https://github.com/<owner>/<repo>/commit/<sha>' --run
# Optional: --manifest submissions/<recorded-d2>/manifest.json
```

Both positional arguments default in plan mode to the newest timestamped
`submissions/*-d2-v*/manifest.json`'s `results_dir` and `code_commit_url`.
Execution requires an explicit commit URL. A later packed v2 manifest with the same
schema is selected without editing this script.

The script copies the slice queue into a fresh
`.runtime/tmp/reproduce-d2-<unique>/queue.json`, setting its learned budget from
the manifest and its evaluation episodes from the listed export filenames. For
v1 these are **512 steps, one world, replay minimum 32**, on **20 evaluation
episodes**. It runs only the loop's `tasks` and `evaluation` phases, in order.
Budget selection and public tuning are not repeated. Episodes 41/43/44 are refused.

The existing loop lacks a results-directory CLI option. The printed Python
invocation binds its existing `loop.RESULTS` to the requested directory before
calling `loop.main()`; no slice source is changed and no extra export copy is
needed. Its canonical task registry, resume ledger, summaries and completion
marker remain under `outputs/pose_insurance/v1/`; immutable producers remain
under `build-v1/` and `run-v1/`. The script uses the current checked-out slice
code and does not turn an older recipe into a newer one merely by changing a path.

**The evaluation phase can use the GPU.** Run `--run` only in an authorised GPU
job when the chain's GPU schedule is free. It executes the resumable loop once,
serially, through the slice handlers and their `track3/gpu.py` lease. It never
touches the lock itself. No GPU stage was run for this job.

Packing and preflight reuse the manifest's kit commands, substituting the results
directory, commit URL and fresh CSV path. The original `--skip-crosscheck` and
`--skip-kaggle` flags are preserved. The fresh temporary directory holds phase
logs, pack/preflight logs, CSV and a new manifest recording its queue, exports and
commands. Preflight must exit successfully and report `Ready to submit`.
The historical manifest records **9,120 rows**; this job only re-hashed that file.

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
| D2 | `submissions/20261101T1703Z-d2-v1/{manifest,run}.json`; `outputs/pose_insurance/v1/{ledger.jsonl,evaluation_finished.json}`; selected producer `run.json` files |
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

All four scripts passed `bash -n`. All three default plans exited 0. Their logs
and the verifier transcripts are in
`.runtime/tmp/d5-reproduction-j5wHti15/`:
`syntax.log`, `plan-d1.log`, `plan-d2.log`, `plan-d3.log`,
`verify-d1.log`, `verify-d2.log`.

Commands run:

```bash
for script in scripts/reproduce/*.sh; do bash -n "$script" || exit; done
bash scripts/reproduce/d1_meshes.sh
bash scripts/reproduce/d2_poses.sh
bash scripts/reproduce/d3_report.sh
bash scripts/reproduce/verify.sh submissions/20261101T1605Z-d1-v2/manifest.json
bash scripts/reproduce/verify.sh submissions/20261101T1703Z-d2-v1/manifest.json
```

D1 verification exited **0**. D2 verification exited **1**: its packed CSV and all
20 exports match, but two source files have changed since packing
(`__main__.py`, `scene.py`). They were not altered by this job.
Full verifier output:

```text
+ .runtime/linux-cuda/bin/python -B - submissions/20261101T1605Z-d1-v2/manifest.json
OK submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet
SUMMARY 1 entries: 1 OK, 0 MISMATCH
+ .runtime/linux-cuda/bin/python -B - submissions/20261101T1703Z-d2-v1/manifest.json
OK submissions/20261101T1703Z-d2-v1/track3_poses_v1.csv
OK outputs/pose_insurance/results-v1/episode_000001.parquet
OK outputs/pose_insurance/results-v1/episode_000003.parquet
OK outputs/pose_insurance/results-v1/episode_000005.parquet
OK outputs/pose_insurance/results-v1/episode_000006.parquet
OK outputs/pose_insurance/results-v1/episode_000008.parquet
OK outputs/pose_insurance/results-v1/episode_000010.parquet
OK outputs/pose_insurance/results-v1/episode_000014.parquet
OK outputs/pose_insurance/results-v1/episode_000015.parquet
OK outputs/pose_insurance/results-v1/episode_000016.parquet
OK outputs/pose_insurance/results-v1/episode_000019.parquet
OK outputs/pose_insurance/results-v1/episode_000020.parquet
OK outputs/pose_insurance/results-v1/episode_000025.parquet
OK outputs/pose_insurance/results-v1/episode_000028.parquet
OK outputs/pose_insurance/results-v1/episode_000030.parquet
OK outputs/pose_insurance/results-v1/episode_000032.parquet
OK outputs/pose_insurance/results-v1/episode_000033.parquet
OK outputs/pose_insurance/results-v1/episode_000035.parquet
OK outputs/pose_insurance/results-v1/episode_000036.parquet
OK outputs/pose_insurance/results-v1/episode_000037.parquet
OK outputs/pose_insurance/results-v1/episode_000038.parquet
OK track3/features/pose_insurance/__init__.py
MISMATCH track3/features/pose_insurance/__main__.py
  expected ad048be624d930e890204d5b99b11c24e43879333620a9e9e80af27b8f2b628a
  actual   7271f31b3971b4b111863b2788e4eefc8ac85db3814ac23e335cd4111133f5a4
OK track3/features/pose_insurance/_scene.py
OK track3/features/pose_insurance/diagnostic.py
OK track3/features/pose_insurance/handler.py
OK track3/features/pose_insurance/loop.py
OK track3/features/pose_insurance/pack.py
OK track3/features/pose_insurance/request.py
MISMATCH track3/features/pose_insurance/scene.py
  expected 932d294b33cd6ba3976e3c30afdeec20e57f4e20ad30835a0b612202e09eee07
  actual   f78f64519eb9f3d03ebba52a314b7d00ff2fd4990923b1ccecd2d8a29091f7c4
OK track3/paths.py
OK track3/gpu.py
OK track3/provenance.py
OK track3/runs.py
OK configs/sources.lock.json
SUMMARY 35 entries: 33 OK, 2 MISMATCH
```

No D1 rebuild, GPU operation, report build, pack or preflight was executed here.
The `--run` branches are written but not exercised. No test suite was run.
`docs/chain/SUBMISSION.md` was not edited.

Suggested replacement D5 row for the lead context:

```markdown
| D5 | One reproduction command per deliverable | D1/D2/D3 plan-first entry points and manifest hash verifier; shell syntax and all default plans pass. D1 packed hash matches; D2 CSV and 20 exports match, with two historical source-hash mismatches documented | `scripts/reproduce/README.md` (commands, run-record pointers and verifier output) | Execute rebuilds only in an authorised run; D2 GPU phase remains guarded by --run |
```

