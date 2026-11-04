# Frozen Track 3 submission container

CPU staging is complete; Docker build and GPU reproduction require the owner's account.
No image has been built, no Kaggle identity check has run, and no final submission.json
exists until the owner supplies real submission IDs.

## Release changed during preparation

The brief named the combined pack `1c6e3e9b…`, commit `efa8f39…`, in
`submission/20261104T1303Z-d2-combined-v5/`. The lead removed that draft while this job
was running. The available replacement is `submission/20261104T1316Z-d2-combined-v5/`,
SHA-256 `ddfc98b3e6d9bd10b31092903fdc93e1e22624d1f2ddc545b1d8f64c4ebf9342`,
commit `02117aca7ac721c29e9a77d9fb8942347584d63b`.
The recorded direction in `docs/chain/SUBMISSION.md` and commit `73059fb` is now one
combined file for all six boards. This stage explicitly follows that release. It preserves
D1 v5 (`bcb93d57…`, `b1617234…`, owner-reported public CD-O 0.53562) as historical evidence.
No counted artifact was changed by container preparation. Read stage-plan.json for the
exact input manifest and its hash; finalization uses this frozen copy, not a later release.

## Prepare, supply IDs, build

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 \
  data/.runtime/linux-cuda/bin/python -B scripts/container/stage.py \
  --combined-manifest submission/20261104T1316Z-d2-combined-v5/track3_combined_v5.manifest.json \
  --counted-set single
```

Output: `data/outputs/container/submission/`. Staging is resumable and hash-checks both
ends of each copy. Independent local `git clone --no-hardlinks --no-checkout` plus detached
checkouts change only the new clones. Neither the original worktree nor its refs/index is
modified. The clones include `.git` and all locally available history: 291 reachable commits
at preparation time, not shallow, about 74 MiB per .git. Historical paths/docs remain in that
history; no excluded dataset episode is loaded.

Copy `data/outputs/container/kaggle-ids.example.json` to
`data/outputs/container/kaggle-ids.json`, and fill all six nulls with positive integer IDs
for this combined file, including its new `cd_o_cm` ID. The earlier D1 ID cannot substitute
for that new upload. The five pose IDs and new combined CD-O ID do not yet exist in this
job. No Kaggle API is called. Existing receipts beside the two pack files are copied when
present; `stage.py --finalize` adds later receipts from those directories. A receipt stored
elsewhere must be copied unchanged to submitted/ and independently hashed by the owner.
Finalization refuses to create final submission.json without all six IDs. The kit verifier
itself does not require IDs; passing a draft check is not a final container declaration.

```bash
# Docker and NVIDIA Container Toolkit must already be available.
DATASET_ROOT=/absolute/path/to/released/dataset bash scripts/container/owner-build.sh
```

The script stops at the first failing command. It finalizes IDs, builds, verifies stored
artifacts, checks the historical D1 mesh re-pack, reproduces all twenty policies serially,
checks reproduced rows/provenance, saves with docker save | zstd, hashes the tarball and
prints form fields. It never pushes, uploads, accepts competition rules or submits a form.
The owner should inspect reproduced scores before submitting the form. The host's runtime
is a prerequisite for stage.py; the prepared Docker context itself is self-contained.
If the host lacks the zstd executable, owner-build.sh streams through the already copied
Python zstandard package in a CPU container instead; it requires no host installation.

## Original two-file arrangement: exact schema limitation

The guide permits different counted files per board, but the verifier supports only one
global code.path/code.commit_url. It accepts both track_3_mesh and track_3 from config,
and a kit override per entry; its prose field table simply omits track_3_mesh. There is no
separate enforced JSON Schema in this kit. An extra entry-level code_commit_url or code_path
is ignored. Thus D1 at b161723 fails check_provenance against a global 02117ac (or efa8f39)
checkout. The stage includes split.draft.json so this failure can be reproduced honestly.

There is also no object_mesh re-pack branch. Without outputs a mesh entry receives SKIP;
with outputs the verifier invokes the generic pose packer, rather than
pack_reconstruction.py track3-meshes. Our verify-mesh command runs the exact old kit's
mesh pack and compares every row, geometry value and provenance. It reproduces D1 byte for
byte on the host. It supplements the unedited verifier; it does not alter its checks.

Two ways out for the original arrangement:

1. Organisers explicitly accept two clean code checkouts with two independent manifests,
   or provide a verifier/schema supporting code selection per submission. Separate
   manifests are included for diagnosis. Only after that acceptance, a split build can use
   ORGANISER_ACCEPTS_TWO_COMMITS=1 with owner-build.sh. Its aggregate check still fails.
2. Upload/select the same combined file on all six boards. This is now the recorded plan.
   The removed 1c6e3e9b draft used simulated boxes for episode 5's pot and episode 20's
   pitcher; the current ddfc98b3 pack instead retains their procedural v4 shapes. Both
   sacrifice D1's generated shapes. Episode 19's rigid body-frame rotations can also change
   finite sampled registration. No upload or board selection happened in this job.

## Layout and reproduction dependencies

- code/ is clean at 02117ac; code-mesh/ is clean at b161723; both have independent .git.
- kit/ and verifier_kit/ are the exact new kit; kit-old/ is D1's exact original kit.
- submitted/ has both exact Parquets and the current combined file's exact derived CD-O
  Parquet; their frozen manifests are in provenance/.
- weights/poses-combined-v5/ holds twenty original .safetensors checkpoints.
- outputs/poses-combined-v5/rollouts/ holds twenty unchanged Parquets and twenty original
  official_metrics JSON files, renamed only to the guide's episode naming convention.
- meshes/poses-combined-v5/ has the 36 combined meshes; meshes/meshes-d1-v5/ has D1's 36.
- tasks/ holds twenty unchanged task Parquets and 72 simulated OBJ/URDF assets.
- runtime/ holds the evaluated FlashCHORD source/assets, installed evaluator and kit
  environments, and both CPython distributions. Native package bytes are copied, not installed.
- provenance/reproduction-trace.json binds each task, checkpoint, asset and original run.
  path-aliases.json maps 92 historical absolute file paths to staged files.

The checkpoints contain absolute task paths; tasks and URDFs contain absolute asset paths.
The unchanged runtime's evaluation.source_root remaps only relative paths or absolute
/workspace paths, so it cannot remap ours. image_setup.py creates filesystem aliases only
inside the image, preserving task, checkpoint, mesh and URDF bytes. Historical prefixes
are derived from the artifacts; implementation adds no fixed chain-context path. Venv
pyvenv.cfg, interpreter links and the editable source .pth are relocated; evaluator modules
and installed package payloads stay unchanged. No model weights are downloaded.

Policies originally ran through src/features/pose_insurance/handler.py and its scored
handlers, with the unchanged kit evaluator, explicit frame-zero reset, one world and a
512-step FlashSAC budget. The container calls that same evaluator serially at those frozen
settings, using src.gpu.GpuLease from the clean code checkout. The lease/cache/logs live
under /output. It does not retrain, reconstruct from video or regenerate tasks/meshes.
The combined pack used --skip-crosscheck; reproduction preserves that flag, which skips
the packer's reference-score cross-check, not any artifact hash or organiser comparison.

Mount the released challenge dataset read-only at /data and output read-write at /output.
These frozen evaluation tasks already contain their references and simulated assets; this
reproduction path needs no dataset file access. It never opens validation 2/13/22/24/27,
episode 41, or holdout 43/44. The historical mesh path only packs stored reconstructions.

## Runtime provenance and hardware

Native evaluation used Ubuntu 26.04.1, RTX 5080 16 GiB, Python 3.11.15, FlashCHORD 0.4.0 at
`aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, JAX 0.10.2 CUDA 13, Warp 1.14.0, Newton 1.2.0.
The four-module evaluation hash is
`b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`, matching the pose file.
The new kit README instead names runtime `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.
Equal evaluation hashes do not establish equivalence of the complete revisions. This
runtime-pin deviation is disclosed and organisers may reject it. Do not replace the
runtime that actually ran or edit provenance to hide it.

There was no native evaluation Docker image. Dockerfile uses ubuntu:26.04 and copies the
native installed environments. It pulls that base and apt's git, ca-certificates, libstdc++6,
libgomp1, GL/EGL/X11/font libraries explicitly listed in Dockerfile, plus dependencies.
No pip/uv, source, kit or model downloads. Base/apt versions are not content-pinned; keep the
built image and its checksum. Full package/version inventories are in provenance/environments.json.
The copied 7.0 GB runtime closure is smaller than the original 9.4 GiB installation because
unrelated video_to_data components, runtime tests and bytecode caches are omitted.

Use CUDA 13 capable NVIDIA drivers (runtime's Dockerfile says driver >=580), NVIDIA
Container Toolkit and preferably the original 16 GiB GPU class. Budget 16 GiB host RAM
and 30 GiB free build/output space; these are estimates, not measured container peaks.
Original twenty-policy train-plus-evaluate times total about 18 minutes; container compile
and evaluation time are unmeasured. Allow 15–60 minutes for first reproduction with uncertainty.

## Checks and unresolved items

The sibling verifier-current.json/log records the unedited verifier's checks using its
--manifest path option; it is not hard-coded to /submission. It passed all 11 checks for
the current single-file draft, including clean commit, all provenance hashes, twenty
checkpoint hashes, and stored-rollout re-pack equality. D1's separate old-kit re-pack
matches all 294,912 rows and is byte-identical. Other CPU evidence is in the sibling
CPU results report; copy-hashes.json and verify-copies cover every copied regular file.

Known untested: Docker build, apt availability/dependency resolution, system linkage,
venv relocation and aliases in the image, GPU/driver compatibility, offline policy loading,
CUDA compilation, fresh reproduced rows/provenance/scores and run-to-run variation, Kaggle
copies/IDs, repository sharing with the organiser account, tarball delivery and form access.
Runtime pin approval is unresolved. A local image generally has a config image ID and no
registry RepoDigest; owner-build.sh prints both, without calling a config ID a registry
digest. If organisers require a registry manifest digest, clarify that separately.

The owner uploads the saved tarball to a public file-sharing link, then supplies the link,
sha256, image identity, required GitHub URL(s), and two-page report through the form.
