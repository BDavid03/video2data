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

`pack_moving.py` is a separate module entry point: the existing slice dispatcher is
pinned by v5 and stays unchanged, as does `handler.inspect_export`'s static drift
refusal. The moving operation verifies the frozen v5 CSV and manifest identities,
copies its twenty exports, and replaces only explicitly admitted scored episodes.
It invokes the unedited kit packer and preflight (Kaggle skipped), then compares
all unadopted CSV fields as **text**, including signed zeros, ignoring only the
new global `code_commit_url`. It never writes under `submission/`.

After the lead commits the implementation and has genuine passing evidence:

```bash
bash scripts/reproduce/d2_poses_moving.sh \
  --request data/outputs/pose_insurance/moving-pack/adoptions.json \
  --code-commit https://github.com/BDavid03/video2data/commit/<FULL_LEAD_COMMIT_SHA> \
  --out data/outputs/pose_insurance/moving-pack/adopted/track3_poses_moving.csv
```

The output's **parent directory must be new**. This writes `results/` with all
20 selected parquets, `pack.log`, `preflight.log`, the CSV and its sibling
`.manifest.json`. Normal packing requires the URL commit to contain the exact
current implementation, metric helpers, path/provenance helpers, reproduction
and release scripts, and M1/M2 protocols. The lead fills the actual commit URL
at pack time; this command does not push, upload or promote a release.

The request has one list, which may be empty for a dry reproduction:

```json
{
  "adoptions": [
    {
      "episode": 38,
      "trajectory": "data/outputs/<producer>/trajectory.parquet",
      "checkpoint": "data/outputs/<producer>/training/policy_<steps>.safetensors",
      "admission": "data/outputs/<producer>/admission.json",
      "reference": "data/outputs/<reference-producer>/reference.npz"
    }
  ]
}
```

All paths are inside the current checkout. Duplicate episodes and any episode
outside the kit's scored evaluation roster are refused **before adoption inputs
are opened**. In particular, tuning 39/42, validation 2/13/22/24/27 and excluded
41/43/44 cannot be adopted. Episode-39 M2v **summary evidence** is read; no tuning
trajectory or labelled dataset is opened by this operation.

`reference.npz` uses M1's existing contract: `pose` is `T x 7` for a single object,
or `T x B x 7` in official slot order; positions are metres and quaternions are
xyzw. `frame_index` and `time` give the full source clock. The independently
hashed reference must agree with the evaluator's embedded reference, including
quaternion sign equivalence. No frame alignment or per-frame transformation is
fitted during admission.

The admission document uses the following envelope. Each `{path, sha256}` is a
real content hash entry, not a placeholder at runtime:

```json
{
  "schema_version": 1,
  "episode": 38,
  "inputs": {
    "trajectory": {"path": "...", "sha256": "..."},
    "checkpoint": {"path": "...", "sha256": "..."},
    "reference": {"path": "...", "sha256": "..."}
  },
  "policy_run": {"path": ".../run.json", "sha256": "..."},
  "official_metrics": {"path": ".../official_metrics.json", "sha256": "..."},
  "m2v": {"path": ".../m2v-adoption.json", "sha256": "..."},
  "m1": {"path": ".../gates.json", "sha256": "..."},
  "achieved_reprojection": {
    "median_iou": 0.60, "p10_iou": 0.30, "usable_fraction": 0.90
  },
  "abstract_disclosure": {"path": "docs/report/track3/main.tex", "sha256": "..."}
}
```

The numbers in that example illustrate the limits, **not measurements**. The
operation fails closed on absent, nonfinite or failing evidence:

- The evaluator schema, units, episode, checkpoint hash, evaluator script/code
  hashes, complete official roster, names and slot order, valid environments,
  unit quaternions and clock must agree. Evaluation modules must equal pinned
  upstream after the installation's CRLF-to-LF normalization, with no code edits.
- `policy_run` follows the existing pose producer record: `status=completed`,
  `split=evaluation`, matching `episode_index`, positive `optimizer_updates` and
  `actor_adam_steps`, checkpoint path/hash, `artifacts` hashes for
  `trajectory.parquet` and `official_metrics.json`, and one recorded kit evaluator
  command binding the exact checkpoint, trajectory and metrics paths. This is
  provenance evidence of learning; the packer does not retrain the checkpoint.
- M2v's sealed summary requires `episode=39`, `passed=true`, `verdict=GO`,
  `AUC >= 0.45`, `tracking` containing `T`, `Q`, `E_m`, `E_S_m`, `Theta_deg`,
  `Theta_S_deg` satisfying Amendment 1's bounds, and a `reprojection` object with
  the same three keys/limits as `achieved_reprojection`.
- M1's final gates document requires `episode`, `label_free=true`, `passed=true`,
  and `reference_sha256` matching the adopted npz. Its `gates` retain the existing
  M1 producer's `{value, passed, ...}` rows. Every measurement in `M1_LIMITS`
  is checked against the protocol's fixed limits, including held-out camera
  tracks, table, masks, object motion, reference reprojection and **measured**
  zero-action support. An unmeasured support gate does not pass.
- `official_metrics.json` must record `reset_mode=explicit`, start 0, one world,
  the full step count, no nonfinite worlds, end fraction 1 and zero object/wrist
  failure fractions. Together with the export clock and evaluator command this
  establishes one attempt reaching the end without an in-rollout reset.
- `T` and `Q` reuse the frozen displacement-gain and 15-frame increment functions.
  The packer re-derives them, mean position error `E`, mean geodesic orientation
  error `Theta`, held-still baselines and initial errors from the exported achieved
  poses and the independent reference. Each object must have `T,Q >= 0.50`,
  `E <= min(0.5 E_S, 0.03 m)`, `Theta <= min(0.5 Theta_S, 15 deg)` and initial
  errors at most 1 cm / 2 degrees. An exactly stationary positional reference has
  undefined `T/Q` and is refused. This conservative per-object rule also applies
  to multi-object adoptions.

The packer checks the supplied achieved-reprojection measurements; it does not
rerender the video or independently regenerate M1/M2v's evidence. Those measured,
hashed producer records and the abstract's actual wording still need lead review.
It re-derives tracking and initial errors rather than trusting admission claims.
The manifest records the code URL, all input/source hashes and each episode's
`v5 static` or `moving policy` origin, with checkpoint/admission hashes for moving
policies. It has no direct `AGENTS.md` hash entry. The immutable v5 parent is
hashed as an input without expanding its historical source inventory.

To select a real moving release later, the lead updates the `pose` entry in
`release.json` with the new CSV path/hash and manifest, and adds
`"moving_manifest": "<the same manifest path>"` inside that entry. Only then does
`check_release.sh` call `--verify-manifest` instead of v5's static verifier. It
re-admits the evidence, verifies per-episode provenance, repacks all admitted
exports through the kit and requires exact CSV equality. A missing selector,
a dry pack or an empty moving release fails. The existing release configuration
is unchanged and still selects v5.

### CPU proof, 4 November 2026

The scratch evidence lives in `data/.runtime/tmp/mpack-dry/`:

```bash
bash scripts/reproduce/d2_poses_moving.sh \
  --request data/.runtime/tmp/mpack-dry/empty.json \
  --code-commit https://github.com/BDavid03/video2data/commit/db7b13fc196785e9bf10714cd6e8231d808d06b8 \
  --out data/.runtime/tmp/mpack-dry/empty-reviewed/poses.csv --dry-run
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  data/.runtime/linux-cuda/bin/python -B data/.runtime/tmp/mpack-dry/prove.py
```

Dry mode permits only `mpack-dry/`, labels its manifest **NOT ADMISSIBLE**, and
skips the new implementation's commit-content check. Its syntactically valid URL
above is v5's old URL, so that the equivalence run can compare whole-file bytes;
it does **not** claim this new implementation existed at that commit. A later
production URL changes the URL column globally and the new manifest records the
new implementation and per-episode inputs.

| Evidence | Result |
| --- | --- |
| Empty adoption | All 9,120 rows and whole CSV bytes identical to v5, SHA-256 `0b2606aed015fe0a64e31037ff790b5483680c3da7e7a5ae2d503edeb8261035`; no URL difference in this dry proof |
| Fake rotation plumbing | Existing episode-38 rows rotated smoothly by 0..5 degrees; 262/263 rows change, all 8,857 rows in other episodes remain text-identical |
| Refusals | 93 distinct producer, schema, protocol, tracking, first-pose, episode and output-boundary probes refused; `refusals.jsonl` and `proof-summary.json` |
| Kit shape | Empty and fake compositions pass local preflight, Kaggle skipped; this does not establish scientific admissibility |
| Scientific adoption | Zero moving episodes admitted; the operation rejects the fake marker and all dry packs at release |

The fake's CSV is named `FAKE_ROTATION_ONLY/poses_FAKE_NOT_ADMISSIBLE.csv`; its
parquet is explicitly marked `plumbing_only`, and `NOTICE.json` says it is not an
admitted pack. Only the low-level scratch composition/official kit shape was
exercised with this fake; admission was never bypassed for a real pack. The
refusal script uses real v5 data plus helper-level probes and synthetic evidence
envelopes for failing gates, not a successful moving-policy producer. It appends
a ledger and resumes completed cases. Compilation, CPU import and the existing
`check_release.sh` are the other checks; no suite, GPU job, install or upload ran.

The post-change release check ended `READY` (`mpack-dry/release-after.log`), with
v5's pose hash unchanged. The optional selector was exercised against a dry
manifest without editing `release.json`: a selected dry pack failed moving
provenance, and a moving manifest missing its selector failed declaration
(`release-selector-proof.json`). No successful real moving-release check is
claimed. `metric-equivalence.json` records exact agreement with the existing
T/Q functions for held-still, half-motion and full-motion controls (tolerance
`1e-12`).
