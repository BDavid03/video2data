# Pose insurance v1

Train a FlashSAC policy with parked Sharpa hands in a static, video-derived box
scene, then export achieved poses with the unchanged official evaluator. This is
an insurance entry; it does not reconstruct manipulation or establish semantic
object frames.

The owner's 2026-10-18 execution authorisation and the D2 briefs authorise the 20
scored episodes plus public tuning 39/42. `runtime.json` gates each operation.
Episodes 41/43/44 are refused. No packages are installed or submissions uploaded.

## Run

Use the read-only project environments and run one GPU operation at a time:

```bash
export PYTHONPATH=. PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
.runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase budgets
.runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase choose_budget
.runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase tasks
.runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase evaluation
.runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase tuning
.runtime/kit-env/bin/python -m track3.features.pose_insurance.pack
```

`queue.json` is the editable experiment queue and canonical learned budget. Budget
selection chooses the largest measured learner-update count among passing pilots
that took at most 300 seconds. Each pilot/training operation uses the slice handlers,
hashed `run.json` records, and `track3.gpu.GpuLease`. All repeated runs belong to the
loop. A failed evaluation episode stops the campaign.

A standalone operation uses `python -m track3.features.pose_insurance run --request
track3/features/pose_insurance/requests/learned_10.json`. The contract request is a
one-step, zero-update envelope check and is never published into the v1 results.

## Learned budget and measured pilots

The selected campaign budget is **512 environment steps, one world, replay minimum 32**.
All three episode-10 pilots passed, but 16,384 steps produced a nonfinite training
state on episode 3 after 6.70 minutes, and 4,096 steps produced a nonfinite state
on episode 5. These recipes are retained as rejected evidence. The queue uses the
largest remaining tested budget; `v1/budget_selection.json` records the decision.
This yields 479 learner updates and 240 actor Adam steps after warm-up.
Replay capacity is 64, n-step is three, batch size eight, with one update per
collection after warm-up. Training uses float32; actor/critic widths are 32/64 with
one residual block each. Evaluation always uses one world, explicit reset, 20 Hz,
static references and zero VOC assistance. Hands remain parked far from the objects.

| Episode 10 steps | Learner updates | Actor Adam steps | Drift cm / degrees | Minutes | Peak device MiB |
|---:|---:|---:|---|---:|---:|
| 512 | 479 | 240 | 0.047694 / 0 | 1.70 | 1940 |
| 4,096 | 4,063 | 2,032 | 0.047694 / 0 | 1.20 | 1951 |
| 16,384 | 16,351 | 8,176 | 0.047694 / 0 | 2.34 | 1951 |

These are measured training-plus-evaluation times including setup and compilation;
cache state differs. Peak memory is sampled device-wide allocation, not isolated
process memory. Saved actor and critic Adam counters independently confirmed these
update counts, with no skipped updates. Passing exports require finite unit poses,
valid environments, official slot order, complete roster keys, no duplicates, and
maximum drift below 1 cm / 2 degrees from the first exported pose.

## Geometry and reviewed identities

D1 v1 box proportions come from build `20261101T103838.400965Z-117fbcff`. Episode
MoGe-2 geometric-mean size restores their metric dimensions where measured; otherwise
pooled D1 size is retained. MoGe depth scale is unanchored and uncalibrated, with no
established absolute error bound. The table normal defines world +z and its plane
becomes z=0. Box centres use projected cluster centres and half-height seating;
heading is in-plane PCA. `_scene.py` contains only pure numerical functions.

`placements.json` records directly inspected frame-0 overlays, explicit cluster
IDs (or a documented default), notes, and `reviewed` / `uncertain` confidence for
every slot missing a reviewed measurement assignment. Cluster size never assigns
an object name. Episode 14 is corrected to cup=1, rack=0. Episode 30's cluster 0 is
the rack holder; cluster 1 mixes the rack rim and sweeper, so the sweeper retains
an uncertain default rather than being assigned the rack holder. Episode 28's
merged rack/pot cluster cannot establish separate placements; both use defaults.
Episode 1's merged pot/lid likewise retains defaults. Hand-contaminated cup centres
in 20/32 and the partial rack centre in 30 remain uncertain.

Original assignment notes and identity confidence propagate into scene records,
`v1/tasks.json`, ledger and summaries. Episode 25's near-identical board names remain
**arbitrary and uncertain**; inspection does not identify which named board is which.

Clearance uses the separating-axis test on oriented table footprints. Disjoint
boxes are unchanged. An overlap gets the minimum separating translation plus a
10 micrometre numerical margin, preferentially moving a placement override.
The old sphere rule's 33–93 cm shifts disappear. Episode 28's revised default boxes
need a 3.776 cm correction. The complete before/after table and per-slot IDs are
in `outputs/pose_insurance/v1/displacements.md` and `displacements.json`.

**Main known limit: semantic object frames are unresolved.** PCA axes and visible
centroids are not reference object frames. Host alignment of slot 0 can rotate the
relative displacement and orientation of slot 1 incorrectly. Public 39/42 each
have one object; static-oracle agreement there cannot measure this two-object risk
or establish absolute scale accuracy. Semantic-frame work is explicitly deferred.

## Outputs and resume

V0 run directories, `results/`, and the v0 submission are retained intact. V1 uses:

- `outputs/pose_insurance/build-v1/` and `run-v1/`: immutable producers and assets.
- `outputs/pose_insurance/results-v1/`: official trajectory Parquets only.
- `outputs/pose_insurance/v1/`: queue ledger, task registry, budget selection,
  summaries, checkpoints and completion markers.
- `outputs/pose_insurance/diagnostic-v1/`: recorded public-label diagnostics.
- `submissions/<UTC stamp>-d2-v1/`: CSV, manifest, pack/preflight logs and run record.

Resume re-admits producer records and checkpoint/trajectory hashes, restores any
missing or stale flat result on both execution and resume, and verifies the entire
selected export set before writing `evaluation_finished.json`. Packing repeats that
admission, checks the exact 9,120-row roster and requires kit preflight's “Ready to
submit” verdict. The requested commit URL is a placeholder using the current HEAD;
`manifest.json` also records the actual working-source hashes for owner review.

## Recorded public diagnostics

The tuning queue trains the chosen policy for public 39/42 and calls the CPU recorded
operation, also available directly:

```bash
.runtime/linux-cuda/bin/python -m track3.features.pose_insurance.diagnostic \
  --episode 39 --run outputs/pose_insurance/run-v1/<run>/episode_000039/run.json
```

Each `pose_insurance.diagnostic` record hashes the admitted trajectory and policy
producer, public pose Parquet, public meshes, baseline and official metric code,
`diagnostic_sampling.json` (1,000 surface points/object, seed zero) and runtime gates.
Public labels never enter task construction or the scored policy lineage. Sampled
public mesh points approximate the host's vertices. The oracle static 17-episode
mean is AUC 0.460898 / SP-SR 0.411765; it starts from true initial poses.

The shipped tissue task contract and old/new PCA extents equivalence remain
confirmed: one row with 390 nested frames at 30 Hz, 22 joints, 67 robot frames and
16 contact slots per hand, plus the accepted zero-articulation placeholder. The
new pure axes helper preserves all original extent fields exactly on 8,313 real
points. The unchanged runtime is pinned at `aee95f25`; evaluator package SHA-256 is
`b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`.

## Completed v1 pack

`submissions/20261101T1703Z-d2-v1/track3_poses_v1.csv`: 9,120 rows; preflight **Ready to submit**.

SHA-256: `fa3b1c320128a196648e0a966eb81b8e5cb8e65cb8ad6b1de0a3fbbb4523c26b`.

All 20 policies passed. Complete per-episode timing, drift, update counts, placement
confidence, resume evidence and recorded tuning metrics are in
`outputs/pose_insurance/v1/report.md`. All saved v0 file hashes remain unchanged.
