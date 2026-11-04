# Frozen Track 3 submission container

The single file staged for all six boards is the moving v5 release:
`submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.parquet`.
Its SHA-256 is `b476eb1c2bc1ee6fab2df72f42f377d87e5d265c06140f3e47fc2fc67f8ff81d`;
its code commit is `c8ff8c135c53a54f9959e20def5920becab87eec`.
This final release supersedes the 1403Z file (`1570037a…`) and the 1418Z file
(`623a1a8a…`); neither earlier file was uploaded. The selected manifest SHA-256 is
`df66435ed6d4fee15ca364ff4738ec4e42f336a754968945f49794dc9e90c331`.
Its adoption source is `data/outputs/pose_insurance/moving-pack/evidence-promoted-v3/adoptions.json`.
The later repository HEAD is not the counted file's provenance commit.
The frozen input manifest is copied to `submission/provenance/` within the staging tree.

## Stage and check on CPU

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  data/.runtime/linux-cuda/bin/python -B scripts/container/stage.py \
  --combined-manifest submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json \
  --counted-set single
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/linux-cuda/bin/python -B scripts/container/check_cpu.py
```

Output: `data/outputs/container/submission/`. Without `--combined-manifest`, stage.py
selects the newest local `*-d2-moving-v5/` manifest. An explicit static combined manifest
remains supported. Mesh paths come from the selected manifest, static policy paths from
its body-frame trace, and adopted policy paths from its moving admission. Episode 38 uses
its learned checkpoint `b6145fcb…`, its moving task, achieved rollout and actual simulated
`mini_dust_pan.obj`; the other nineteen policies remain static combined v5's.

Staging checks both ends of each copy. Changed files and superseded code/mesh trees are
preserved under `data/outputs/container/history/`. The code snapshots have independent Git
object stores and clean detached HEADs: read-only `git archive`/`ls-tree` export tracked
files and a standard v2 index; no clone, checkout or other mutating Git command runs.
The source repository, submission packs and native environments are unchanged.
`copy-hashes.json`, `generated-hashes.json`, `stage-plan.json` and
`provenance/reproduction-trace.json` record the inputs and staged closure.

## Supply IDs and build as the owner

Copy `data/outputs/container/kaggle-ids.example.json` to
`data/outputs/container/kaggle-ids.json`. Fill all six nulls with positive integer IDs for
this moving file, including its CD-O upload. IDs for older files do not substitute.
`stage.py --finalize` refuses a final `submission.json` until all six IDs exist and copies
any receipts beside the original pack files. No Kaggle API is called during staging.

```bash
# Owner account: Docker and NVIDIA Container Toolkit must already be available.
DATASET_ROOT=/absolute/path/to/released/dataset bash scripts/container/owner-build.sh
```

This command finalizes IDs, builds, checks copied bytes and stored artifacts with the
unedited verifier, reproduces the historical D1 mesh pack, evaluates twenty frozen policies
serially, compares fresh rows/provenance, saves the image as a zstd tarball and prints its
hash and identity. It stops on failure. The owner inspects the results, uploads the tarball
and supplies the public link, checksum, image identity, GitHub URL and report themselves.
The script never pushes, uploads or submits a form. If host zstd is unavailable, it uses
the already staged Python zstandard package in a CPU container.

## Reproduction

`run.sh reproduce --label poses-combined-v5` evaluates the manifest's checkpoints with the
unedited kit evaluator and `src.gpu.GpuLease`, one policy at a time. It preserves explicit
frame-zero reset and one world. Static policies trained for 512 environment steps; the
adopted episode 38 checkpoint trained for 66,763,776 steps. No policy is retrained here.
Historical task and asset paths are read from artifacts and aliased inside the image by
image_setup.py; task/checkpoint/OBJ/URDF bytes stay unchanged. No chain context is named
in implementation paths. Caches, copied logs and the container-local lease live under `/output`. The release
command helper also retains a uniquely named packing log under the code checkout
(`code/data/outputs/container/`), because its provenance helper requires a code-relative path.

The final CPU packing adapter imports `src.features.pose_insurance.pack_combined_moving`
for a moving manifest and passes the frozen `provenance/adoptions.json` to `pack_pose`.
It checks its episode, source paths and checkpoint against the release admission, then
uses that packer's existing `base.command` to call the unedited kit packer on **fresh
achieved exports** and the 36 staged moving meshes. A static manifest imports
`pack_combined` and uses its `command` without adoptions. `--skip-crosscheck` is preserved.
The release packer's full `pack()` function is intentionally not rerun: it re-admits and
copies original frozen exports, has no fresh-rollout input, and would discard the new
rollouts. Re-admission/reconstruction and their dataset dependencies are outside this
container reproduction. The adapter preserves the original admissions rather than claiming
fresh policy outputs have passed the original scientific gates.

The same CPU packing path can check stored exports without a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/kit-env/bin/python -B scripts/container/reproduce.py pack \
  --manifest data/outputs/container/submission/submission.draft.json \
  --label poses-combined-v5 \
  --adoptions data/outputs/container/submission/provenance/adoptions.json \
  --rollouts data/outputs/container/submission/outputs/poses-combined-v5/rollouts \
  --output data/outputs/container/cpu-moving-repack
```

The stage also retains D1 v5 as historical evidence, its clean code-mesh snapshot and old
kit. `split.draft.json` is diagnostic: the kit supports only one global code commit, so it
cannot validate that historical two-commit arrangement as an aggregate. Its generic
verifier has no object_mesh re-pack branch; `run.sh verify-mesh` supplements it with the
old kit's mesh packer. The counted single moving file avoids the commit conflict.

## Runtime and remaining work

Native evaluation used Ubuntu 26.04.1, RTX 5080 16 GiB, Python 3.11.15, FlashCHORD 0.4.0 at
`aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, JAX 0.10.2 CUDA 13, Warp 1.14.0 and Newton 1.2.0.
The four-module evaluation hash is
`b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`.
The new kit instead names runtime `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.
Equal evaluation hashes do not prove complete revision equivalence; organiser acceptance
of this runtime deviation remains unresolved. The actually evaluated runtime is preserved.

The Docker context copies the native evaluator, robot assets, both installed environments
and both CPython distributions. It downloads no Python packages or models. Dockerfile pulls
ubuntu:26.04 and its listed apt dependencies; those versions are not content-pinned.
Use a CUDA 13 capable NVIDIA driver (runtime documentation says >=580), NVIDIA Container
Toolkit, preferably 16 GiB GPU memory, and an estimated 30 GiB build/output space.
Package inventories are in `provenance/environments.json`.

The moving stage contains 20 checkpoints, 36 combined meshes, 72 simulated assets and
92 historical path aliases. `dependency-checks-final-moving.log` records all five CPU checks
passing, including both copied interpreter/package imports and task/asset closure.
`verifier-final-moving.json` / `.log` records the **unedited organiser verifier: 11 pass,
0 fail, 0 warn, 0 skip**, including clean code at c8ff8c13, all twenty checkpoint hashes
and stored-rollout re-pack equality. `cpu-final-moving-repack.log` records the container adapter
packing all 9,120 rows byte-identically to the counted moving file, with the SHA-256 above.
`verify-copies-final-moving.log` records copied-file and generated-plumbing hashes.
No dataset was read and no GPU launched during these CPU checks.

Docker build, apt/system linkage, image
venv relocation and aliases, GPU/driver compatibility, offline policy loading, CUDA
compilation, fresh GPU outputs/scores and variation remain **untested**. No Docker command
runs in this job. Kaggle IDs, identity checks, receipts, tarball delivery, repository sharing
and submission-form access remain the owner's work. A local image config ID is not a
registry manifest digest; owner-build.sh prints both available identities accurately.
