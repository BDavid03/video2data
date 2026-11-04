# Track 3 submission container

This folder builds the Docker container that reruns our submitted file's policies (README steps 5
and 6). The container holds the code, the 20 trained policies, the meshes and the organisers'
evaluator, so a judge can replay each policy and repack the file.

## The file it stages

- Pack (the single file we upload): `submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.parquet`
- SHA-256: `b476eb1c2bc1ee6fab2df72f42f377d87e5d265c06140f3e47fc2fc67f8ff81d`
- Code commit: `c8ff8c135c53a54f9959e20def5920becab87eec`
- Manifest (the JSON that hashes the pack and its inputs) SHA-256:
  `df66435ed6d4fee15ca364ff4738ec4e42f336a754968945f49794dc9e90c331`
- Admission source (the record that let episode 38's policy in):
  `data/outputs/pose_insurance/moving-pack/evidence-promoted-v3/adoptions.json`

It replaces the earlier moving packs `1570037a` (1403Z) and `623a1a8a` (1418Z). Neither was uploaded.
The later repository HEAD is not this file's code commit.

## Stage and check on CPU

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  data/.runtime/linux-cuda/bin/python -B scripts/container/stage.py \
  --combined-manifest submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json \
  --counted-set single
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/linux-cuda/bin/python -B scripts/container/check_cpu.py
```

Output: `data/outputs/container/submission/`. Without `--combined-manifest`, `stage.py` picks the newest
local `*-d2-moving-v5/` manifest. A static combined manifest also works.

What gets staged:

- Meshes from the manifest. Static policies from its body-frame trace. Adopted policies from its
  moving admission.
- Episode 38: its trained checkpoint `b6145fcb…`, its moving task, its rollout and the simulated
  `mini_dust_pan.obj`. The other 19 policies are the static combined v5 ones.
- 20 checkpoints, 36 meshes, 72 simulated assets, 92 aliases for old file paths.

Every copy is hashed at both ends. Replaced files go to `data/outputs/container/history/`. The code
snapshots are exported with read-only `git archive` / `ls-tree`. Records: `copy-hashes.json`,
`generated-hashes.json`, `stage-plan.json`, `provenance/reproduction-trace.json`,
`provenance/environments.json`. The frozen input manifest is copied to `submission/provenance/`.

## Build (owner only)

1. Copy `data/outputs/container/kaggle-ids.example.json` to `data/outputs/container/kaggle-ids.json`.
   Fill in all six Kaggle submission IDs for this file, including the CD-O one. IDs of older files do
   not count. `stage.py --finalize` refuses to write `submission.json` until all six exist.
2. Run (needs Docker and the NVIDIA Container Toolkit):

```bash
DATASET_ROOT=/absolute/path/to/released/dataset bash scripts/container/owner-build.sh
```

This finalizes the IDs, builds the image, checks the copied files with the organisers' verifier,
rebuilds the D1 mesh pack, runs the 20 policies one at a time, compares the new rows, saves the image
as a zstd tarball and prints its hash and image ID. It stops on the first failure. It never pushes,
uploads or submits anything. No Kaggle API is called during staging.

## Inside the container

```bash
run.sh verify | verify-copies | verify-mesh | reproduce | pack [options]
run.sh reproduce --label poses-combined-v5
```

`reproduce` runs each checkpoint through the unedited kit evaluator, one policy at a time, with one
world and a reset at frame zero. Static policies trained for 512 environment steps; episode 38's
policy trained for 66,763,776 steps. Nothing is retrained. `image_setup.py` maps old task and asset
paths inside the image; the file bytes are unchanged. Caches, logs and the GPU lock live under
`/output`. One packing log also goes to `code/data/outputs/container/`.

`pack` rebuilds the file from the new rollouts. For a moving manifest it uses
`src.features.pose_insurance.pack_combined_moving` with the frozen `provenance/adoptions.json`; it
checks episode, sources and checkpoint against the admission, then calls the unedited kit packer on the
new rollouts and the 36 staged meshes (`--skip-crosscheck` kept). A static manifest uses
`pack_combined` without adoptions. It does not rerun the full release packer, which would discard the
new rollouts. It does not re-run the original admission checks on the new outputs.

Check stored rollouts on CPU, no GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/kit-env/bin/python -B scripts/container/reproduce.py pack \
  --manifest data/outputs/container/submission/submission.draft.json \
  --label poses-combined-v5 \
  --adoptions data/outputs/container/submission/provenance/adoptions.json \
  --rollouts data/outputs/container/submission/outputs/poses-combined-v5/rollouts \
  --output data/outputs/container/cpu-moving-repack
```

The stage also keeps D1 v5, its code-mesh snapshot and the old kit as records. `split.draft.json` is
only a diagnostic: the kit allows one code commit per submission, so it cannot check the old two-commit
setup. `run.sh verify-mesh` adds the old kit's mesh packer because the verifier has no mesh repack.

## Runtime

Evaluation ran on Ubuntu 26.04.1, RTX 5080 16 GiB, Python 3.11.15, FlashCHORD 0.4.0 at
`aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, JAX 0.10.2 (CUDA 13), Warp 1.14.0, Newton 1.2.0.
Four-module evaluation hash: `b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`.
The new kit names runtime `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. Equal evaluation hashes do not
prove the revisions are the same, and the organisers have not ruled on this difference.

The Docker context copies the evaluator, robot assets, both environments and both CPython builds. It
downloads no Python packages or models. The Dockerfile pulls `ubuntu:26.04` and apt packages, which are
not pinned by hash. Needs a CUDA 13 NVIDIA driver (>=580), the NVIDIA Container Toolkit, ideally 16 GiB
GPU memory, and about 30 GiB of disk.

## What has been checked

CPU only, no dataset read, no GPU:

- `dependency-checks-final-moving.log`: all five CPU checks pass (interpreters, imports, task and asset
  files).
- `verifier-final-moving.json` / `.log`: organisers' verifier, unedited: 11 pass, 0 fail, 0 warn,
  0 skip, including clean code at `c8ff8c13`, all 20 checkpoint hashes and repack equality.
- `cpu-final-moving-repack.log`: the adapter repacks all 9,120 rows byte-identical to the counted file.
- `verify-copies-final-moving.log`: copied-file and generated-file hashes.

Not tested: the Docker build, apt and system libraries, the relocated venvs and path aliases inside the
image, GPU driver compatibility, offline policy loading, CUDA compilation, and fresh GPU scores. Kaggle
IDs, receipts, the tarball upload, repository sharing and the submission form are the owner's to do. A
local image ID is not a registry digest; `owner-build.sh` prints both.
