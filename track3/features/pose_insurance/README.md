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
