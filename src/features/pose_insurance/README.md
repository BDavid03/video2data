# Pose insurance

Train FlashSAC with parked hands in static, video-derived box scenes and export
achieved poses with the unchanged official evaluator. This insurance entry does
not reproduce manipulation or establish semantic object frames. Runtime gates,
source hashes and immutable producer records apply to every operation. Episodes
41/43/44 are excluded. The owner pushes code and uploads submissions.

## Run

The fixed queues drive repeated trials through the slice handlers, with one GPU
operation at a time. The recorded budget is 512 transitions / one world / replay
minimum 32 (479 learner and 240 actor Adam updates). Larger pilots produced
nonfinite states on episodes 3/5 and were rejected. CPU diagnostics below use
`data/.runtime/linux-cuda/bin/python`; policy loops invoke the existing FlashCHORD
runtime under `src.gpu.GpuLease`.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.scored_stereo_loop
```

## Geometry and reviewed identities

`_scene.py` contains pure scene geometry. Table +z comes from the measured plane;
box heading comes from in-plane PCA. Oriented-footprint clearance replaces the
old sphere separation. MoGe-2 depth is uncalibrated: it establishes no absolute
scale bound. `placements.json` records reviewed/uncertain slot assignments;
cluster size never determines an object name. Merged or hand-contaminated masks
keep documented defaults. Episode 25's board identities remain arbitrary.

Semantic object frames are unresolved. Slot-0 alignment can rotate slot-one
offsets incorrectly. Single-object tuning controls 39/42 cannot measure that
risk. Public reference poses and scan surfaces enter diagnostic scoring only;
they do not construct scored scenes. Scan sampling approximates host scoring.

## Outputs and resume

`data/outputs/pose_insurance/` owns immutable `build-v*/`, `run-v*/` producers,
`results-v*/` official trajectory exports, and per-campaign ledgers/reports.
Resume checks source identities, producer records, checkpoints and artifact
hashes; failures fall back only where the fixed queue permits, without retry.
Historical records remain unchanged. Source changes invalidate execution resume;
historical report rendering is a separate CPU operation.

## D2 v4: direct stereo placement

`scored_stereo_queue.json` changes only slot-one XY in exact v3 tasks. The scored
loop above writes `stereo-placement-scored/{plans,ledger,composition,summary}.*`,
policy records and `results-v4/`. `--phase prepare` records plans without policies.
All 12 eligible T1 policy runs pass; episode 30 lacks an independent slot-zero
cluster and keeps v3. The composition has twelve T1 plus eight v3 exports.

Reproduce the frozen composition without a new policy run:

```bash
bash scripts/reproduce/d2_poses_v4.sh --compose-only
```

Packing requires an exact committed-source URL and the reproduction script's
`--run --code-commit <URL> --out <CSV>` options. It writes a CSV and manifest;
nothing uploads. The packed manifest under `submission/20261103T0645Z-d2-v4/`
is authoritative for the final file and pinned source list.

## Scored stereo-anchored S1 follow-up

The earlier v3 queue scales slot-one table-plane offsets, using sixteen two-object
scored episodes and four untouched single-object episodes. Its 512/1/32 budget,
physical checks and final per-episode fallback decisions remain frozen.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.scored_scale_loop
```

Writes `data/outputs/pose_insurance/v3/{ledger,plans,decisions,finished}.json*`,
`build-v3/`, `run-v3/`, `results-v3/` and the packed v3 artifact. `v3/report.md`
records the physical checks, composition, hashes and preflight. `--phase prepare`
records plans only. `--phase pack` composes the frozen decisions. Shorter offsets
may hedge incorrect directions; no host-score gain is claimed.

## Placement, scale and length diagnostics

These are kinematic train/tuning measurements, not physical policy evaluations.
Each queue writes a resumable ledger, immutable trials and readable summaries
under `data/outputs/pose_insurance/<campaign>/`.

| Diagnostic | Module after `python -B -m src.features.pose_insurance.` | Output directory | Latest result |
|---|---|---|---|
| Static placement | `placement_loop` | `placement/` | All 40 trials complete; no candidate passes +0.02 AUC. Static oracle AUC 0.531919 versus candidate 0.268844. |
| Scale | `scale_loop` | `scale/` | 30 trials; S1 RPE/MPPE 35.335583/20.843565 cm versus incumbent 39.363846/22.818618. Strict joint decision blocked by S2's 7.1e-15 cm control difference; S1 adopted separately as disclosed in the protocol. |
| Stereo placement | `stereo_placement_loop` | `stereo-placement/` | T1 mean train AUC 0.276704640, RPE/MPPE 29.013682/17.678603 cm. Only slot-one XY changes. |
| Length | `length_diagnostic --request src/features/pose_insurance/length_request.json` (via slice dispatcher) | `length/` | Five-episode stereo horizontal MAE 0.581 cm versus 11.754 cm incumbent; all five improve. |
| Length confirmation | `length_diagnostic --request src/features/pose_insurance/length_confirm_request.json` (via dispatcher) | `length-confirm/` | Not confirmed: mandatory episode 13 is merged; available-three MAE 1.138 cm cannot satisfy the four-episode rule. |
| Identical support | `support_diagnostic --request src/features/pose_insurance/support_request.json` | `support-diagnostic/` | 11 episodes; common-support depth gap 11.8859 cm, support contribution below 0.472 cm; depth-dominates reading 5/5, ratio correction 4/5. |

For dispatcher operations the command is `python -B -m src.features.pose_insurance
<operation> --request <full-slice-request-path>`. Queue files and protocols in
`docs/chain/` define rosters and gates. `_placement.py`, `_scale.py`, `_length.py`
and `_stereo_placement.py` hold pure computations. Length confirmation preparation uses
`length_prepare.py` (`setup`, `prepare --reuse-only`, `freeze`); producer records retain the
original assignment provenance and disclose merged/missing offsets.

## Pose shrink control

The fixed shrink queue compares I/S1/G, seven shifts and C0/C25/C50/C75, computes
six oracle grids and 40,320 scale assignments, and conditionally evaluates H.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.shrink_loop --queue src/features/pose_insurance/shrink_queue.json
```

Writes `data/outputs/pose_insurance/shrink/summary.{json,md}`, immutable trials,
ledger segments and previous-report backups. Torn ledger tails are retained;
new appends use continuation segments. `_shrink.py` computes, `shrink_diagnostic.py`
admits/records, and `shrink_report.py` renders.

Latest: all 156 trials completed and resumed. I/S1 reproduce all five historical
metrics exactly on ten episodes. G's MPPE gain over S1 is only 0.042320834 cm.
C0 reaches 14.133047791 cm versus S1 20.843564645 cm but is implausible.
H gains 3.823015849 cm MPPE yet loses 0.000180288462 AUC and worsens episode 31
by 0.693391880 cm, exceeding tolerance: **H fails; retain D2 v3**. The recorded
`verification.json` and `critic-fixes-verification.json` retain exact comparisons.
The episode-5 failure inspection rules out initial box overlap; its first
nonfinite training value remains unknown.

## Heading diagnostics and HR2

The heading queue records oracles, leave-one-episode-out up-axis priors and
separate post-hoc descriptive constructions; none is a selected scored policy.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.heading_loop
```

Writes `data/outputs/pose_insurance/heading/summary.{json,md}` and immutable trials.
The engineering rerun reproduces all 320 original metric values exactly.
H2S-desc (saved registrations on train 18/23, ignoring HR2 declines) has mean
AUC 0.322286160 versus T1 0.276704640. It is descriptive, post-hoc and has no
adoption gate. It does not change HR2's failure.

`registration_verdict.py` recomputes the frozen HR2 verdict in its own record;
`registration_verdict_queue.json` owns pairs, rosters and thresholds. Latest:
**FAIL**, pairs 0/4 accepted, up-axis 5/11. Selected b/C errors on episode 18
are 4.20/5.63 degrees; on 23, 9.46/13.69 degrees. Frozen HR2 sources and method
hashes remain pinned; registration predictions precede label assessment.

## H3 table-world heading

H3 places registered rotations through T1's signed camera-to-table basis,
keeping positions. The scored episode-19 adapter admits exact D2 v4 producers;
train assessment uses episode 18 plus seven historical T1 comparator rows.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.registration_h3_loop && PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.heading_h3_loop
```

Writes `data/outputs/pose_insurance/registration-h3/latest.json` and
`heading-h3/summary.{md,json}`, ledgers and immutable reports. The maintained
`heading_h3_check` module independently recomputes geometry and writes
`heading-h3/independent-check.json`. Predictions and scored comparisons precede
train labels; queue contract hashes fix rosters, thresholds and rivals.

Latest: required G1/G2/G3 pass. Episode-18 AUC is 0.472910765; eight-episode mean
0.298238811 versus T1 0.276704640. The equal-weight flip robustness gain is
0.010767086, only 0.000767086 above G2's margin. Episode 19's sweeper remains
ambiguous (3.442% rival margin, 179.978 degrees), so G2 is required. Resume
reuses two registrations/four heading rows. A D2 v5 build still requires the
protocol's owner approval and physical checks; H3 alone is evidence.

## S1 silhouette diagnostic

Measure saved HR2/H3 basins against depth-derived partial masks, with no pose
refitting. `_silhouette.py` owns geometry/scores, `_silhouette_labels.py` owns
pure correctness/identity rules, `_silhouette_render.py` owns presentation, and
`silhouette_loop.py` admits data, records predictions and assesses separately.
The 13-object roster/thresholds remain pinned in `silhouette_queue.json`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.silhouette_loop
```

Writes `data/outputs/pose_insurance/silhouette/`: immutable `trials/` with prediction
JSON/render NPZ, `predictions/` with candidate table and `candidate-outlines.png`,
`assessments/` with label errors/verdict, and root summary/pointer files.
`--operation predict` stops before labels; `--operation assess --prediction-report
<run.json>` requires a completed full panel. Changed source identities require
fresh trial records; the old ledger and prediction/assessment records stay intact.

Latest: **FAIL; episode-19 application withheld**. All 13 predictions finished
14:08:18 UTC before assessment began 14:08:49 UTC. Episode 18's correct sweeper
basin wins, but its minimum IoU margin 0.02481043 is below 0.05; best/rival IoUs
are 0.792341/0.764124, C errors 5.6309/176.8835 degrees. No wooden flip resolves.
Only 2/7 controls have complete matches; neither vetoes. Episode 19 abstains
(best/rival IoUs 0.782869/0.787093). Independent raw-input IoU difference is zero.
Both panels surrounding the unmatched-control reporting correction remain saved;
thresholds, sweeper scores and verdicts agree. `silhouette/report.md` holds the
full evidence. H3 and submissions are unchanged.

## D2 v5: frozen episode-19 TW headings

The owner approved the H3 TW selection on 3 November 2026. The pinned queue is
`scored_heading_v5_queue.json`: episode 19 only, 512 environment steps, one world,
replay minimum 32, one policy attempt and no retries or basin switching. G1/G2/G3
must pass in the hashed H3 summary; the dated approval, H3 report, v4 manifest and
full-precision H3 poses are admitted before numerical imports. The dedicated
`scored_stereo_authorized` flag controls this stage and the stereo runner/packer.

CPU preparation (writes assets and task; never runs the policy):

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 data/.runtime/linux-cuda/bin/python -B \
  -m src.features.pose_insurance.scored_heading_v5 \
  --queue src/features/pose_insurance/scored_heading_v5_queue.json --phase prepare
```

The same command with `--phase all` is the single GPU command, to be launched by
the lead in GPU mode after review, source freeze and completion of any other GPU
job. It reuses the CPU build. `plan.json` freezes source/input hashes; any later
change refuses execution. `attempt.json` reserves the sole attempt atomically;
`ledger.jsonl` records preparation and resolution. The policy handler requires the
reserved plan and run path before any policy write; exclusive creation of
`attempt-claim.json` prevents a second handler invocation from running it again.
Packing admits only that reserved run and its recorded claim. An interrupted/failed attempt
retains v4 and cannot be retried. The 300-second queue budget is a planning bound
based on the v4 33-second warm / 178-second cold runs; a live training process is
never terminated at that boundary. Completion is `heading-v5/finished.json`.

Output is under `data/outputs/pose_insurance/heading-v5/`. Counter-rotation uses
`G = Rnew.T Rold`: OBJ vertices become `G v`; the URDF collision origin becomes
`G Rcollision` / `G tcollision`; COM inertia becomes `G Ibody G.T`. URDF RPY uses
extrinsic XYZ (`Rz(yaw) Ry(pitch) Rx(roll)`). The full six-entry tensor has identity
inertial RPY, preventing a second rotation. Mesh/URDF read-back equivalence covers
world vertices, collision corners, COM and inertia; masses, collision dimensions,
positions, parked hands, mesh radii and all 235 frames are preserved. The CPU
`cpu-task-projection.parquet` is explicitly a task/export-contract projection,
never a policy trajectory or simulation evidence. `cpu-inspection.json` also
inspects the real v4 trajectory against the TW scene and records its expected
heading failure. Actual first-pose and drift gates apply after the GPU run.

On success the runner copies the other nineteen v4 Parquets verbatim into
`data/outputs/pose_insurance/results-v5/`, checking their v4 SHA-256 values, and records
full/local kit preflight separately. After the lead commits the frozen sources:

```bash
bash scripts/reproduce/d2_poses_v5.sh --run --code-commit "$URL" \
  --out "$PACK/track3_poses_v5_final.csv"
```

`URL` must contain the full committed SHA. Without `--run`, the wrapper verifies
admission only; `--compose-only` reproduces and runs local preflight on CPU. Packing
compares all 8,650 non-19 CSV rows field-for-field as strings, preserving signed
zero text and excluding only `code_commit_url`; total rows remain 9,120. The v5
manifest includes the hashed parent, approval/H3/plan/build/policy/checkpoint,
invariance, twenty export hashes, committed sources, script, CSV comparison and
preflight result. Historical v4 outputs, queue and reproduction script stay intact.

Scientific limitation: world-geometry/inertia equivalence and physical/export
checks establish consistency, not scored accuracy. The sweeper rival is only
3.44% worse in registration and G2 cleared by 0.00077 AUC; the basin may be wrong.
The CPU build and verification findings are recorded in
`data/outputs/pose_insurance/heading-v5/cpu-verification.json`; GPU results remain pending.

CPU preparation passed: the serialized world mesh/collision residuals were at most
5.55e-17 / 8.33e-17 m, and world inertia residual was at most 6.51e-19 kg m².
Both 0.1 kg masses, 26 task fields and 235 frames are preserved. The projected task
covers 470/470 rows; the real v4 export is correctly rejected against TW at
178.19 degrees first-pose rotation error. All 12 changed/new Python modules compile
and import on CPU; the convention extraction is bit-identical on 16 recorded calls
(32 slots). Preparation resumes with one build and zero GPU attempts. All 5,658
inventoried historical output/pack files retain their hashes.

## Frame-zero panel Stage C

The fixed CPU queue implements `docs/chain/frame0-panel-protocol.md`, including
Amendment 1. It checks the A/B seal and the exact authorized protocol-history
exception, freezes all arm poses and recorded-rival combinations before label
access, then scores only train 0/11/12/18/21/23/26/31 with the unchanged kit metric.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.panel_gates
```

`data/outputs/pose_insurance/frame0-panel/stage-c/` contains hashed `candidates.json`,
a resumable trial ledger, seed 0/1/2 metrics, every frame-zero limit and gate
clause in `summary.json` and `summary.md`, and the provenance/timing `run.json`.
Seed 0 decides the gates; the original placement keep arms are descriptive only.
These are disclosed train-label kinematic results, with no policy or simulation.

Latest Stage C result: 138 unique pose/seed trials completed in 7.40 seconds;
seed-zero T1 reproduced exactly. P passes (RPE -1.556 cm, MPPE -0.822 cm,
AUC unchanged); P-free is identical on 26/31. TW4 fails despite AUC +0.048302:
only 26 meets the new-pair limits, 23 exceeds the original-T1 offset allowance,
and flip stress fails. AN4 and all three original-placement keep arms fail.
Seeds 1/2 give the same verdicts. Episode 31's 5.124 cm vector error exceeds
5 cm while its 26.015-degree b and 9.096-degree C errors meet their limits.
No scored-stage adoption or physical-evaluation claim follows from this job.

## Episode 30: scored P route

`panel_scored` applies Amendment 1 items 8–11 to episode 30 alone after P passed
and TW failed. It reuses the frozen DINO/SAM workers and exact inherited T1
stereo/remap/table frame, preserves T1 rotations, and saves each stage separately.
Run these commands in order, inspecting admission and feasibility before proceeding;
an existing stage directory prevents a repeated policy attempt.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.panel_scored stage-a
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.panel_scored construct
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/flashchord/venv/bin/python -B -m src.features.pose_insurance.panel_scored policy
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.pose_insurance.panel_scored decision
```

Outputs live in `data/outputs/pose_insurance/frame0-panel/scored-30/`: masks and contact
sheet, frozen construction, fresh box assets/task, one 512-step/one-world policy,
official evaluator trajectory, physical checks and `decision.json`. Run the CPU
geometry review before policy execution; its saved `geometry-check.json` is consumed
as evidence. The review script is `data/.runtime/tmp/panel-scored-30-review/check_geometry.py`.
The train queue's pre-amendment protocol pin is disclosed; this adapter records
the current amended protocol directly and asserts the original model settings.
No scored label or score is read. Physical acceptance establishes simulation
consistency only; proxy shrink does not establish realistic geometry or accuracy.
The existing v5 exports remain unchanged; this command never packs or uploads.

Latest result: **FAIL, keep v5 (adoption rule 5)**. Stage A admitted 865/16,351
stereo pixels; `d = (0.019340175, 0.011406293, -0.075975808) m`. The maximal
common XY factor was 0.0281486625, with 1 cm clearance and both masses 0.1 kg.
First-pose errors were 3.600222 cm / 0.000002415 degrees; drift was
16.526426 cm / 89.999970 degrees. All 20 incumbent export hashes match v5.
One policy/evaluator attempt used 196.085 GPU-lease seconds, plus 33.884 for
mask inference; elapsed stage-to-stage wall time was 320.419 seconds.

## Gated moving-pose composition (CPU)

`python -m src.features.pose_insurance.pack_moving` is separate from the pinned
static dispatcher. Static packing and its 1 cm / 2 degree drift refusal remain
unchanged. Moving packing replaces only admitted episode exports in frozen v5,
invokes the unedited kit packer/preflight, and compares every unadopted CSV field
as text, preserving signed zeros and excluding only the global commit URL.
Outputs stay under `data/outputs/pose_insurance/moving-pack/`; `submission/` is
always forbidden. Dry outputs stay under `data/.runtime/tmp/mpack-dry/` and are
never eligible for release.

### From finished ep38-v1 to a release entry

These commands are for the lead after the producer finishes. They do not train,
render simulation, push or upload. Both runs must have completed training and
final evaluator outputs; 39 must also have its sealed mocap GO. Both final M1
records and Stage 0 must pass. **If those gates fail, retain v5.** The lead
first reviews/commits the implementation and the actual report disclosure, and
uses that commit's URL below. A changed abstract also requires the usual D3
report build/hash update before the final release check can pass.

1. Generate the admission and request from the two real producer folders, with
   no hand-written metrics or manual JSON editing. The evidence directory must
   be new:

   ```bash
   bash scripts/reproduce/d2_poses_moving.sh \
     --make-admission data/outputs/moving_policy/ep38-v1 \
     --m2v-output data/outputs/moving_policy/ep39-v1 \
     --abstract-disclosure docs/report/track3/main.tex \
     --out data/outputs/pose_insurance/moving-pack/evidence-v1
   ```

   This reads the existing training/checkpoint/evaluator records, hashes their
   inputs, renders achieved poses on the frozen M1 video evidence on CPU, and
   writes per-frame IoU arrays and producer records for 38 and 39. It writes
   `admission.json` (schema 2) and, **only after full admission succeeds**,
   `adoptions.json`. A failure is a refusal, not a request to edit a measurement.

2. Pack using the lead's full committed URL, into a new output directory:

   ```bash
   MOVING_CODE_SHA="$(git rev-parse HEAD)"
   bash scripts/reproduce/d2_poses_moving.sh \
     --request data/outputs/pose_insurance/moving-pack/evidence-v1/adoptions.json \
     --code-commit "https://github.com/BDavid03/video2data/commit/$MOVING_CODE_SHA" \
     --out data/outputs/pose_insurance/moving-pack/packed-v1/track3_poses_moving.csv
   ```

   The URL must contain the exact current computation sources. The output
   directory contains the twenty selected exports in `results/`, the CSV and
   sibling `.manifest.json`, kit pack/preflight logs, and
   `release-pose-entry.json`. The manifest records every episode's `v5 static`
   or `moving policy` origin, checkpoint/admission hashes, producer inputs and
   code URL. It has no direct `AGENTS.md` hash entry. The historical v5 parent
   manifest is hashed without expanding its old source inventory.

3. Check the packed evidence before selecting it:

   ```bash
   bash scripts/reproduce/verify.sh \
     data/outputs/pose_insurance/moving-pack/packed-v1/track3_poses_moving.manifest.json
   bash scripts/reproduce/d2_poses_moving.sh --verify-manifest \
     data/outputs/pose_insurance/moving-pack/packed-v1/track3_poses_moving.manifest.json
   ```

   Moving verification re-admits both policies and M1/M2v evidence, re-renders
   the video IoUs, checks the common simulator asset, then requires an exact
   official CSV repack. Dry/fake packs and empty moving releases fail.

4. **Lead-only release selection**, after review. This uses the generated entry
   rather than manually copying hashes. It changes only the pose selection:

   ```bash
   data/.runtime/linux-cuda/bin/python -B - <<'PY'
   import json
   from pathlib import Path
   release_path = Path('scripts/reproduce/release.json')
   release = json.loads(release_path.read_text())
   release['pose'] = json.loads(Path(
       'data/outputs/pose_insurance/moving-pack/packed-v1/release-pose-entry.json'
   ).read_text())
   release_path.write_text(json.dumps(release, indent=2) + '\n')
   PY
   bash scripts/reproduce/check_release.sh
   ```

   The entry has `path`, `sha256`, `manifest` and `moving_manifest` (the latter
   names the same manifest). Only this explicit optional selection activates
   moving verification in `check_release.sh`. The lead must see `READY`, review
   the abstract, and decide promotion/upload. This job leaves `release.json`
   selecting v5 and writes no file under `submission/`.

### Admission checks and producer contract

The request lists `{episode, trajectory, checkpoint, admission, reference}` per
adopted episode. `reference` is **`reference-task.npz`**, the moved file used by
the task. The schema-2 admission builder derives these paths from the finished
folder; it never invents `optimizer_updates`, `actor_adam_steps`, `split` or
`episode_index` fields that the producer does not write.

Actual learning requires a completed `train.run.json`, its Stage 0 hash, the
recorded trainer command (from `argv` or immutable `commands.json`), the frozen
66,763,776-step recipe, a `train.log` completion count at least that high, and
the actual checkpoint's safetensors metadata. The latter must be a FlashSAC
policy with positive `user.global_update_step`, matching
`user.environment_steps`, recipe total and task path. Metadata is read on CPU
with the already installed producer interpreter; no package is installed.

The final evaluator is bound through `score-self.run.json` and
`commands.json:evaluator_final` (the original M2 `score-*.run.json:argv` format
is also understood). Its checkpoint, trajectory and metrics paths must match.
The export must have the kit's unchanged script/code hashes, checkpoint hash,
complete official row roster, official slot names/order, finite unit quaternions,
valid worlds, and the full uniform frame clock. Explicit reset mode, end fraction
1 and no object/wrist failure establish one attempt reaching the end. Training
and evaluator records are bound to the same immutable Stage 0/task.

M1 comes exclusively from each source reference's
`candidate-c/gates-final.json`. Earlier gate files are retained as hashed
historical evidence, not direct admission files. The admission hashes **both** the amended gates file and
the source reference; no `reference_sha256` field is required inside the gates
file. The amended producer's `frozen_rule.label_free` supplies provenance when
there is no top-level `label_free`. All amended measurements are checked against
these limits; `passed=true` alone is insufficient:

| Measure | Limit |
| --- | --- |
| Held-out camera reprojection median / p95 at 640 width | ≤0.5 / 2 px |
| Held-out camera world residual | ≤1 cm; perpendicular instrument only under the verified sealed reference-and-allowance pass; old 3D residual retained descriptively |
| Held-out tracks / image regions / camera coverage | ≥30 / 4 / 98% |
| Table fit residual / disjoint-patch normal | ≤5 mm / 3° |
| Usable masks / longest missing interval | ≥90% / ≤2 frames |
| ≥30 stereo-supported eroded-mask pixels | ≥90% of frames; old KLT count remains descriptive |
| Forward/backward position / orientation disagreement | ≤1 cm / 5° median |
| Object speed / angular rate | ≤1.5 m/s / 540°/s |
| Valid reference frames | ≥90% |
| Reference IoU median / p10 / usable coverage | ≥0.60 / 0.30 / 90% |
| First-frame base gap / penetration | ≤5 mm / 5 mm |
| Maximum per-frame median table offset / p95 of medians | ≤1 cm / 1.5 cm |
| Late table-normal change | ≤2° |
| Table drift usable fraction | ≥80%, as recorded by the amended producer |
| Maximum per-frame median long-baseline in-plane displacement / p95 of medians | ≤1 cm / 2 cm |
| Long-baseline frames with ≥30 never-reanchored withheld table landmarks | ≥80% |

M1 support is **not** a pack gate: Amendment 1 moved support to each policy's
Stage 0. Stage 0 must contain/pass all seven named checks, including that support
proof. No support limit is relaxed.

`reference-source.npz`, `reference-task.npz`, `reference.json` and the actual task
parquet are checked together. Source and moved poses must differ only by the
recorded pure translation, with rotations unchanged, on the entire sequence.
The task's first pose **and all later object poses** must equal that moved M1
reference. Thus a reset placing achieved at the task's first pose cannot conceal
a wrong task reference. The original and moved clocks are compared to the task's
uniform clock at `1e-6 s`; observed residuals were only `1.78e-15 s` (the producer's
equivalent division expression records zero).

One shared asset manifest must bind the same OBJ, URDF and body-convention hashes
on 39 and 38, in the actual task mesh/URDF paths, Stage 0 and checkpoint task.
M1's reference/body constant is bound to this asset in the admission. The tracking
extrusion remains M1's reprojection mesh, as the protocol states; the simulator
uses the shared profile, not that extrusion. This distinction is recorded rather
than pretending they are the same mesh.

`achieved_reprojection` is a **hash entry for `producer.json`**, not three naked
numbers. Its producer binds the exact trajectory, source reference, recorded
translation, common simulator asset, frozen camera, body constant, tracking mesh
and each cached frame's masks. It writes per-frame IoUs/usable flags in a hashed
npz. Admission verifies those inputs, recomputes the aggregate values **and
re-renders every frame** to require equality. A producer for another trajectory,
reference or asset, a changed mask/camera, empty source inventory, forged summary
or unhashed dictionary is refused. The producer rejects fake/plumbing trajectories.
The achieved and M2v IoU limits remain median ≥0.60, p10 ≥0.30 on ≥90% of frames;
this reprojection is explicitly blind to camera drift, so M1's table/landmark gates
remain independently mandatory.

M2v's real `score-66763776/mocap_metrics.json` must belong to a completed
`score-mocap.run.json`, its learned checkpoint and moved reference. Its GO/AUC
≥0.45, self-reference bounds, no-reset end, amended M1 evidence and achieved
reprojection must all pass before 38 can be adopted. For adopted achieved poses,
the packer re-derives `T`, `Q`, `E`, `Theta`, held-still baselines and first-pose
errors. Each slot must meet `T,Q ≥0.5`, `E ≤min(0.5 E_S,3 cm)`,
`Theta ≤min(0.5 Theta_S,15°)`, and first errors ≤1 cm / 2°. Undefined positional
T/Q from a stationary reference are refused. No scored labels are read.

### CPU evidence, 4 November 2026

Evidence is under `data/.runtime/tmp/mpack-dry/`. The follow-up uses new
`amended-proof/` and `amended-evidence/` subdirectories, retaining the original
proof artifacts. `prove-amended.py` runs the roster/schema/protocol/error/output
refusal loop; `prove-evidence.py` runs real learning/task checks and the new
asset/reference/reprojection refusal loop. They append ledgers and resume cases.

- The genuine M2 checkpoint/log record 66,763,776 steps and 130,382 optimizer
  updates; the new learning check accepts them without invented producer fields.
- Both real video Stage 0 task/reference/asset checks were exercised. Pure
  translations have zero source/moved error; task serialization differs by at
  most `3.1e-8 m` and `5.7e-6°`, inside the unchanged tolerances.
- A **clearly fake**, scratch-only replay of the moved M1 reference reprojects
  identically to M1's original per-frame IoUs (maximum difference 0). This checks
  geometry/translation plumbing, not a learned achieved rollout. It cannot be
  packed or produce admission evidence.
- Empty adoption must preserve all 9,120 v5 CSV rows and the whole-file SHA-256
  `0b2606aed015fe0a64e31037ff790b5483680c3da7e7a5ae2d503edeb8261035`. Its dry
  URL is deliberately v5's old URL; a real lead pack changes the URL column
  globally and records the new manifest/source/input identities.
- The separate fake 0..5° rotation changes 262/263 episode-38 rows and leaves
  all 8,857 other-episode rows text-identical. The fake remains marked and refused;
  its successful kit shape/preflight is not scientific admission.
- **At the earlier follow-up, amended M1 results on 38 and 39 failed.** The packer preserves those
  refusals. No successful real moving admission or moving-release READY is
  claimed. The current release remains v5; see final scratch command logs and
  evidence summaries for the final refusal counts and `check_release.sh` output.

Only compilation, CPU imports, equivalence/refusal scripts, real CPU packing and
kit/release checks run here. The producer slice, upstream, models, pinned data,
submission tree and release selection are not edited by this job. The lead still
reviews the report's actual wording before selecting a new release.

Final follow-up checks: **117 refusals** (86 roster/protocol/packing cases and
31 producer/reference/asset/reprojection cases); the final empty CSV is byte
identical to v5; `release-amended-final.log` ends **`READY`**. The actual
`--make-admission` command exits 1 on failed amended M1 and creates no adoption
request. While the other producer job rebuilt its live folders, geometry refusal
fixtures used the immutable first snapshots; those fixtures are explicitly
blocked by `inspect_policy` and are not real learning/admission evidence. The
real M2 learning check remains on its original completed process/checkpoint.
The final float32 fake-reference replay differs from M1's double-precision
reference raster IoUs by at most `8.84e-5` (median/p10 remain 0.86443/0.81040);
the earlier geometry replay had zero difference. Neither replay is admissible.

### Final M1 handoff, 4 November 2026

The current admission file is **`candidate-c/gates-final.json`**. Earlier
`gates-amended*.json` files remain hashed historical inputs. Generate it once with:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.final_gates
```

The final record preserves `gates-amended-3.json`, every other limit and the old
3D world residual. Under the sealing record's clause 2 it gates the perpendicular
world residual at the unchanged 10 mm, only after both sealed reference and
allowance pass. It hashes all earlier gate records and the actual sealed decision.
Admission re-derives the entire final record from those inputs; the old noise-floor
requirement does not override the explicitly authorized sealed-pass condition.
A failed seal, altered limit, forged measurement or missing provenance is refused.
Every gate still has to pass. Both actual records pass 28/28; the references keep
their sealed hashes, and no Stage 0 record changes.

The existing `--make-admission` sequence now passes M1 on both episodes, then
refuses with an explicit list of missing finished 38-policy/M2v records. No
adoption request or pack is written until those producer outputs exist and pass.
The release remains v5. Current evidence and final per-gate table are under
`data/.runtime/tmp/mpack-dry/final-gates-proof/`; command logs and the empty v5
reproduction are under `data/.runtime/tmp/mpack-dry/`.
