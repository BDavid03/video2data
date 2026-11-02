# Pose insurance

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

## Kinematic static-placement diagnostic

This CPU operation implements `docs/chain/pose-placement-protocol.md`, including
Amendment 1. It constructs static scenes for public train 0/11/12/18/21/23/26/31
in explicit evaluation-analogue slot order, holds their poses over the mocap
reference timeline, and scores them with the kit's own functions. The shared
`_placement.place_boxes` function also supplies the unchanged v1 scene builder.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.placement_loop
```

`placement_queue.json` sets the fixed candidates and trial budget. The loop calls
the gated `placement_diagnostic` handler, hashes each candidate/episode identity,
re-admits finished producers and inputs on resume, appends `placement/ledger.jsonl`,
and writes `summary.md`, `summary.json`, `per_episode_auc.png` and `finished.json`
under `outputs/pose_insurance/`. Each trial owns a hashed `run.json` and
`metrics.json`, with per-episode metrics, frame-zero transform decomposition,
one-factor and joint oracle substitutions, and the slot-0 heading-flip ceiling.
A single request can also use `python -m track3.features.pose_insurance
placement_diagnostic --request <json>` (`episode_index`, `candidate`, and optional
`k1` for the cumulative candidate).

`placements_train.json` records overlay inspection before scoring: merged cluster
0 in 0/26/31, and the border-cut cup cluster 1 in 12. All seven missing independent
placements retain the exact v1 defaults in the comparator. K1a places the upper
object above its support (lid over pot, pot over rack, sweeper over rack) and
skips horizontal clearance for vertically disjoint boxes; K1b uses existing
footprint separation. Both use merged-cluster centroid and PCA yaw.
K2 estimates signed body-up from other train episodes' frame-zero rotations,
uses normalized means and singleton z-up fallback, then applies minimal up-axis
rotation followed by PCA yaw. Geometry rotation and the full inertia tensor are
re-expressed consistently in the diagnostic record. No simulator assets are
exported by this operation.

**Amendment 1 train result (2026-11-02): recommend none under its fixed gates.** All 40 trials completed. Each
candidate has mean AUC **0.268844**, SP-SR **0/8**, MP-SR **0/8**. K1a wins the
predeclared AUC tie by queue order, but fails the required +0.02 AUC improvement.

| Candidate | RPE cm | MPPE cm | Amendment 1 outcome |
|---|---:|---:|---|
| v1 | 47.508 | 26.661 | Comparator |
| K1a | 36.796 | 21.632 | Fails AUC gain |
| K1b | 39.198 | 22.441 | Fails AUC gain |
| K2 alone | 48.698 | 27.623 | Fails AUC gain and MPPE tolerance |
| K1a+K2 | 36.782 | 21.747 | Fails AUC gain versus K1a |
| Oracle static | 15.114 | 7.403 | AUC 0.531919; SP-SR 5/8; MP-SR 0/8 |

The joint oracle replacement recovers the static reference benchmark; the
per-episode best slot-0 heading flip adds no AUC. Those are diagnostics using
reference outcomes, never candidates. The signed pot convention reverses between
its two episodes; singleton fallback also leaves six objects uncalibrated. K2
therefore cannot resolve every semantic frame from these eight episodes.

The refactor was compared bit-for-bit against the old computation and all 20
selected immutable v1 scenes (36 slots): positions, wxyz quaternions and metric
dimensions are identical; before-clearance positions and clearance shifts also
match. Evidence: `outputs/pose_insurance/placement/equivalence.json`; script:
`.runtime/tmp/d2-placement/equivalence.py`. Historical runs were not rewritten.

Caveats: this is **kinematic, not physical**. K2 mocap conventions are supervised
calibration from public train labels, with leave-one-episode-out scoring; any
future submitted K2 convention must instead fit all eight train episodes, with
wooden_spoon z-up. Public scan surfaces are used **only for scoring**, with our
1,000 samples per object (seed 0), which approximate the host's vertices. Scan
sizes and origin offsets never enter placement. No shared video/mocap heading
frame is released, so K3 is omitted. Any future passing candidate still needs
real evaluator exports on affected scored episodes, with first poses within
1 cm / 2 degrees of intended placement and the existing drift check, before a
repack. This job ran no FlashCHORD evaluator or packing.

A separate real-data metric audit (`placement/audit.json`) reproduced every
candidate's metrics exactly. Slot 1 has zero AUC in all 40 trials: its smallest
per-frame mean ADD is **9.950 cm**, above the kit's largest **9 cm** threshold.
This explains why the substantial K1 distance improvements do not change AUC.
The joint oracle differs from its reference by at most 1.25e-14 across the five
metrics. A second queue invocation resumed **40/40** trials without new scoring.

## Amendment 2 confirmation

`placement_confirmation_queue.json` fixes public 2/27 and candidates v1/K1a only.
Preparation clones the depth requests of train 0/26 (including stride 12/5,
full-video stop, 768 px and 1,200 tokens) and the measure requests' frame lists
(`[0, 12]` and `[0, 5, 10, 15]`) and thresholds. It resumes completed stages and
runs teacher depth sequentially through `depth.generate` and its GPU lease:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.confirmation_prepare
```

After inspecting the two resulting overlays, record `placements_confirm.json`
before scoring. Its `episodes` map must contain keys `2` and `27`, each with
`measurable`, `note`, `measure_run`, `overlay`, `merged_cluster_ids`, `stacked_slot`
and explicit `slots` entries (`cluster_ids` or null, note and confidence).
An episode without a genuine merged opening cluster is marked unmeasurable;
it contributes no candidate score and is reported as such. Train placements
and the eight-episode K2 fitting roster remain separate.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.placement_loop --queue track3/features/pose_insurance/placement_confirmation_queue.json
```

The confirmation writes `confirmation_ledger.jsonl`, `confirmation.md`,
`confirmation.json` and `confirmation_finished.json` under the placement output.
It requires lower RPE and MPPE and no lower AUC/SP-SR/MP-SR on every measurable
episode. No measurable episodes means keep v1. A passing confirmation is required
before any scored scene changes, physical evaluation or D2 v2 packing.

**Confirmation result (2026-11-02): adopt under Amendment 2, based on one episode.**
After the owner enabled the separate validation flags, depth and measurement
completed for 2/27. Episode 2 has merged cluster 0; episode 27 instead has separate
rack cluster 1 and pot clusters 0/2, so it is excluded under the pre-registered
cannot-measure clause without scoring. Overlay entries and hashes were recorded
before scoring. On episode 2, K1a lowers RPE **54.389 -> 21.789 cm**, MPPE
**28.287 -> 12.433 cm**, and frame-zero offset error **45.109 -> 12.887 cm**;
AUC stays **0.334275**, SP-SR and MP-SR stay zero. This satisfies all Amendment 2
conditions. Independent recomputation reproduced the five metrics exactly.
See `outputs/pose_insurance/placement/confirmation.md` and `confirmation_audit.json`.
The original eight-episode Amendment 1 failure remains recorded above; Amendment 2
was registered after that result and confirmed on the fresh episode.

The original placement ledger's 40 blank lines were removed after copying it to
`.runtime/tmp/backup/20261101T225423.627578Z/ledger.jsonl`; record bytes and order
are unchanged. Both queue writers now append exactly one JSON record per line.
Evidence: `outputs/pose_insurance/placement/ledger_normalization.json`.


## D2 v2: latest result

**Packed 9,120 rows; kit preflight Ready to submit; verification 107/107 hashes
OK, zero mismatches.** The committed “Decision after the confirmation” activates
K1a only for scored **1 and 28**. Episode **30 and the other 17 unchanged
episodes retain v1**: eighteen v1 exports plus two K1a exports in total.

`placements_v2.json` contains only 1/28, with `enabled: true`; the scene builder
returns the original v1 construction for every other episode. The reused 1/28
policies were trained at **512 steps, one world, replay minimum 32**, with 479
learner updates and 240 actor Adam steps. Their maximum first-pose errors are
0.062205 / 0.062201 cm; maximum drifts are 0.057650 / 0.057644 cm, all 0 degrees.
No new GPU run was required for finalisation.

The exactness proof rebuilds the active twenty scenes through the CPU scene
handler: **18 complete scene JSON byte matches against v1, plus two against the
evaluated K1a scenes**. Task fields and visual/collision assets also match after
normalising generated asset paths. Evidence:
`outputs/pose_insurance/v2/active_scene_equivalence.json`.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .runtime/flashchord/venv/bin/python -B -m track3.features.pose_insurance.loop --phase reuse --queue track3/features/pose_insurance/queue_v2.json --version v2
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/kit-env/bin/python -B -m track3.features.pose_insurance.pack --version v2
```

The first command admits the already evaluated 1/28 exports against the current
scene proof and budget. The second copies eighteen unchanged files from
`results-v1/` and those two exports into `results-v2/`, then runs the official
packer (`--skip-crosscheck`), preflight (`--skip-kaggle`) and the recorded
`bash scripts/reproduce/verify.sh <manifest>` command. The manifest keeps its two
kit commands in v1 format for `d2_poses.sh`; verification is recorded separately.

The confirmation rests on **one fresh episode, public 2**. Public 27 had separate
clusters and was excluded before scoring. Train 26, the analogue for scored 28,
was part of the selection evidence, so rack/pot behaviour has no independent
confirmation here. The original fixed gates failed, Amendment 2 was registered
after those outcomes, and the lead subsequently applied the physical check per
scene. Observed AUC and SP-SR remain unchanged at these placement errors; no
host-score improvement is claimed.

Episode 30's K1a trial failed first-pose error **1.471503 cm** and drift
**17.002592 cm** because the elevated sweeper lay outside the rack footprint and
fell. That failed producer is excluded; its original accepted v1 export is used.
The raw physical results and exact earlier trial sources remain under `v2/`.

CSV: `submissions/20261101T2315Z-d2-v2/track3_poses_v2.csv`

SHA-256: `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741`

The timestamped directory contains `manifest.json`, completed `run.json`,
`pack.log`, `preflight.log` and `verify.log`. The placeholder URL uses the
packing-time HEAD; working-source hashes are also recorded for review. Complete
composition, copied-file hashes, caveats and checks are in
`outputs/pose_insurance/v2/report.md`. Existing v1 results and package remain
unchanged.

## Scale test

Run the fixed stereo-anchored scale diagnostic (IDEAS 27) with:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -m track3.features.pose_insurance.scale_loop
```

`scale_queue.json` fixes eight train episodes, tuning controls 39/42 and the
incumbent/S1/S2 candidates. The loop calls unchanged stereo handlers on the exact
scene depth frames, resumes completed identities, appends `scale/ledger.jsonl`,
and writes `outputs/pose_insurance/scale/summary.md`, `summary.json`, the per-episode
RPE/MPPE figure and hashed trial `run.json` records. `_scale.py` contains only pure
estimation, scene transforms and gates. The existing placement code stays pinned.
Invalid estimates retain D2 v2. This CPU diagnostic never constructs scored scenes,
runs the physical evaluator or packs a submission. Results and the registered
adoption decision are in the summary; physical adoption remains a separate stage.

Latest result: all ten stereo estimates are valid; all 30 kinematic trials completed
and a second invocation resumed all 30 without rerunning them. S1 changes mean
RPE/MPPE from 39.363846/22.818618 cm to 35.335583/20.843565 cm; S2 gives
35.024862/20.683371 cm. Both pass all three numerical gates. Incumbents reproduce
all five historical metrics exactly. The strict decision is blocked because S2's
single-object episode-39 MPPE changes by 7.1054e-15 cm under rigid alignment;
`numerical_control.json` traces this to a 2.7756e-17 m position difference, consistent
with floating-point cancellation. No equality tolerance was introduced after scores.
The numerical tie-break favours S1; physical follow-up remains the lead's decision.
The lead then adopted S1 by a post-result, per-candidate reading of the invariance
clause (`docs/chain/pose-scale-protocol.md`, last section); D2 v3 applies it (see
`SCALE_V3.md`). After S1 the train distance ratio is still 1.18-2.23 (3.58 on episode
31), so part of the gain may come from shorter offsets hedging wrong directions rather
than from a corrected scale.
