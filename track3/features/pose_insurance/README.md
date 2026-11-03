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
`.runtime/linux-cuda/bin/python`; policy loops invoke the existing FlashCHORD
runtime under `track3.gpu.GpuLease`.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.scored_stereo_loop
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

`outputs/pose_insurance/` owns immutable `build-v*/`, `run-v*/` producers,
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
nothing uploads. The packed manifest under `submissions/20261103T0645Z-d2-v4/`
is authoritative for the final file and pinned source list.

## Scored stereo-anchored S1 follow-up

The earlier v3 queue scales slot-one table-plane offsets, using sixteen two-object
scored episodes and four untouched single-object episodes. Its 512/1/32 budget,
physical checks and final per-episode fallback decisions remain frozen.

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.scored_scale_loop
```

Writes `outputs/pose_insurance/v3/{ledger,plans,decisions,finished}.json*`,
`build-v3/`, `run-v3/`, `results-v3/` and the packed v3 artifact. `v3/report.md`
records the physical checks, composition, hashes and preflight. `--phase prepare`
records plans only. `--phase pack` composes the frozen decisions. Shorter offsets
may hedge incorrect directions; no host-score gain is claimed.

## Placement, scale and length diagnostics

These are kinematic train/tuning measurements, not physical policy evaluations.
Each queue writes a resumable ledger, immutable trials and readable summaries
under `outputs/pose_insurance/<campaign>/`.

| Diagnostic | Module after `python -B -m track3.features.pose_insurance.` | Output directory | Latest result |
|---|---|---|---|
| Static placement | `placement_loop` | `placement/` | All 40 trials complete; no candidate passes +0.02 AUC. Static oracle AUC 0.531919 versus candidate 0.268844. |
| Scale | `scale_loop` | `scale/` | 30 trials; S1 RPE/MPPE 35.335583/20.843565 cm versus incumbent 39.363846/22.818618. Strict joint decision blocked by S2's 7.1e-15 cm control difference; S1 adopted separately as disclosed in the protocol. |
| Stereo placement | `stereo_placement_loop` | `stereo-placement/` | T1 mean train AUC 0.276704640, RPE/MPPE 29.013682/17.678603 cm. Only slot-one XY changes. |
| Length | `length_diagnostic --request track3/features/pose_insurance/length_request.json` (via slice dispatcher) | `length/` | Five-episode stereo horizontal MAE 0.581 cm versus 11.754 cm incumbent; all five improve. |
| Length confirmation | `length_diagnostic --request track3/features/pose_insurance/length_confirm_request.json` (via dispatcher) | `length-confirm/` | Not confirmed: mandatory episode 13 is merged; available-three MAE 1.138 cm cannot satisfy the four-episode rule. |
| Identical support | `support_diagnostic --request track3/features/pose_insurance/support_request.json` | `support-diagnostic/` | 11 episodes; common-support depth gap 11.8859 cm, support contribution below 0.472 cm; depth-dominates reading 5/5, ratio correction 4/5. |

For dispatcher operations the command is `python -B -m track3.features.pose_insurance
<operation> --request <full-slice-request-path>`. Queue files and protocols in
`docs/chain/` define rosters and gates. `_placement.py`, `_scale.py`, `_length.py`
and `_stereo_placement.py` hold pure computations. Length confirmation preparation uses
`length_prepare.py` (`setup`, `prepare --reuse-only`, `freeze`); producer records retain the
original assignment provenance and disclose merged/missing offsets.

## Pose shrink control

The fixed shrink queue compares I/S1/G, seven shifts and C0/C25/C50/C75, computes
six oracle grids and 40,320 scale assignments, and conditionally evaluates H.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.shrink_loop --queue track3/features/pose_insurance/shrink_queue.json
```

Writes `outputs/pose_insurance/shrink/summary.{json,md}`, immutable trials,
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
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.heading_loop
```

Writes `outputs/pose_insurance/heading/summary.{json,md}` and immutable trials.
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
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.registration_h3_loop && PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.heading_h3_loop
```

Writes `outputs/pose_insurance/registration-h3/latest.json` and
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
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.silhouette_loop
```

Writes `outputs/pose_insurance/silhouette/`: immutable `trials/` with prediction
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
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python -B \
  -m track3.features.pose_insurance.scored_heading_v5 \
  --queue track3/features/pose_insurance/scored_heading_v5_queue.json --phase prepare
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

Output is under `outputs/pose_insurance/heading-v5/`. Counter-rotation uses
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
`outputs/pose_insurance/results-v5/`, checking their v4 SHA-256 values, and records
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
`outputs/pose_insurance/heading-v5/cpu-verification.json`; GPU results remain pending.

CPU preparation passed: the serialized world mesh/collision residuals were at most
5.55e-17 / 8.33e-17 m, and world inertia residual was at most 6.51e-19 kg m².
Both 0.1 kg masses, 26 task fields and 235 frames are preserved. The projected task
covers 470/470 rows; the real v4 export is correctly rejected against TW at
178.19 degrees first-pose rotation error. All 12 changed/new Python modules compile
and import on CPU; the convention extraction is bit-identical on 16 recorded calls
(32 slots). Preparation resumes with one build and zero GPU attempts. All 5,658
inventoried historical output/pack files retain their hashes.
