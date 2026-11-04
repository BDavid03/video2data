# Joint moving policy with a resting companion

This route builds one task containing both official object slots. The hand
policy follows A's video reference; B has a constant reference at its existing
v5 start. Both are independent free rigid bodies. The unedited evaluator can
therefore export both bodies from one checkpoint and one world.

The protocol is `docs/chain/moving-two-object-protocol.md`; its frozen Git
blob hash is recorded in `data/outputs/moving_policy/two-object-status.md`.
The queue is `two_object_queue.json`. It fixes all 16 scored two-object
episodes, their ascending order, the time budget and the seed/transition count.
No scored labels, scans or excluded episodes enter this route.

## Run

From the checkout root, using the existing environments:

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B \
  -m src.features.moving_policy.two_object_screen
```

Run the loop detached for a long screen. It reuses the existing DINO and SAM2
workers under `src/gpu.py`'s lease, chooses roles from background-corrected
video mask motion, and applies the unchanged spoon/38 mask gates. It stops
an episode at its first failed gate, retaining the next queue entry. For a
passing mask, the existing M1 stereo/KLT, metric bundle adjustment, object
tracker, reprojection and table-audit instruments run on our procedural mesh.
The first passing reference stops the queue. An infrastructure error stops
the route without retrying. Completed terminal episodes are skipped on resume;
changed queue, protocol or source identities are refused.

Outputs are under `data/outputs/moving_policy/two-object/`:

- `screen.log`, `screen.pid`: detached process and log.
- `identity.json`, `ledger.jsonl`, `summary.json`, `summary.md`: resumable queue evidence.
- `epN/masks-0/`, `masks-1/`: unedited worker requests, masks, logs and coverage.
- `epN/roles.json`, `mask-gates.json`, `result.json`: role and terminal gate evidence.
- `epN/candidate-c/`: M1 reference evidence, only after the mask gate passes.

`two_object_task.py` builds the joint longitudinal/transverse candidates from
an admitted reference. It delegates unchanged single-object construction to
`handler.construct`; `_two_object.py` composes the companion in official slot
order and remaps hand contact IDs to A. It never combines separate exports.
The procedural meshes and body-frame maps come from the existing combined-v5
trace, and both mesh and URDF bytes are hashed.

`two_object_train_worker.py` retains the original training entry point and
clears B's virtual wrench after A's existing controller is applied, before
physics integration. It changes neither the RL recipe nor the official
evaluator. The evaluator's existing zero-VOC composition makes both objects
passive during learned-policy evaluation. `resting_metrics` measures the frozen
panel limits without clamping or resetting B.

## Build evidence

`data/outputs/moving_policy/two-object/build-equivalence.json` records exact
Parquet byte equality for the single-object path: 263 frames, 62 columns,
SHA-256 `20748480c60c84e7f942b8d2346df8bb576bac2d000390f01a2de505cea4bf41`.
`build-joint-loader.json` records the real installed CPU loader/environment for
both slot orders: two free roots, B virtual wrench exactly zero, A gravity
assistance 2.9430003 N. These checks establish task/controller plumbing; they
are not physical admission or evidence of a useful learned two-object policy.

Every new module passed `py_compile` and CPU import. No test suite ran.
Screen outcomes and any subsequent training/admission result are recorded in
the status file and queue summary. Until every reference, Stage-0 and final
policy gate passes, the incumbent pack stays selected.

## Recorded screen outcome (4 November, 14:56 UTC)

Episode 1 passed the three operational mask gates (1.000 usable, zero-frame
gap, 1.000 stereo support) and the five measured camera subset gates (median
0.2662 px640, p95 1.2282 px640, minimum 140 held-out tracks, four regions,
coverage 1.000). The complete reference did not pass: the process stopped
because the original v5 manifest path was no longer available when binding
assets. A separate audit also found zero frame-0 visible stereo pixels after
excluding B, so independent A initialization was unavailable. No episode was
retried; the remaining fifteen were not attempted. No Stage 0, training,
learned-policy export or candidate pack ran. See `summary.md` and
`summary.json` in the output directory for the full stop and its limits.

The delivered empty-cloud guard was added after the running process imported
its original reference module; it is not credited to this execution. Both
source hashes and the corrected timing record are retained in
`reference-boundary-correction.json`. Resuming with this changed source is
refused by the original source identity, as intended; this failed route must
not silently restart.

## Disclosed infrastructure resume

After the lead restored the two parent manifests and verified their hashes
against the uploaded moving manifest, a one-time resume was authorized at
episode 3. `two_object_resume.py` calls the same scientific handlers with the
recorded empty-cloud guard active. It records episode 1 from saved evidence
without rerunning it. Its execution wrapper continues the entire screen after
the 15:35 UTC training cutoff; no later training launch is allowed. The frozen
protocol, queue, thresholds, prompts and role rule are unchanged. Run it once:

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B \
  -m src.features.moving_policy.two_object_resume
```

`resume-once.json` binds the restored parent hashes, original source identity,
guard correction and wrapper code hash. `resume-screen.log` is the new log.
The per-episode A/B and first-failure table remains in `summary.md`. Original
stop summaries are preserved under `data/.runtime/tmp/two-object-resume-before/`.

## Current amended campaign

The owner-corrected close is 22:00 UTC, with last training start 19:45 UTC.
`two_object_campaign.py` screens 8, 1, 10, 14, 15, 16, 19, 20, 25, 28, 30,
32, 33, 3, 5, 6. Both texture-usability measures are waived at early and final
M1 gates; stereo support and every other reference/physical/policy limit remain.
The dated protocol revision freezes joint DINO instance selection: minimum box
IoU, then smallest-box ties, requiring box and propagated frame-0 mask IoU <0.5.
Existing distinct masks are retained. The specified mask reruns for 8/1 preserve
their earlier duplicate-instance evidence. New corrected outputs live under
`instance-separated/`; the campaign and its ledger live under `campaign/`.

The companion modules are `two_object_instances.py` (unchanged DINO/SAM workers,
cooperative lease and distinct-instance rule), `two_object_stage0.py` and
`two_object_cpu.py` (original M2 checks plus both-body support),
`two_object_run.py` (one frozen training attempt, official export, admission and
cumulative packing), and `two_object_admission.py` (A tracking, B rest, evidence
and report binding). `two_object_existing.py` performs CPU-only preparation of
already-masked episodes while the training job owns the GPU. Pure composition,
rest measurements and instance pairing remain in private modules.

Every passing episode may train once. Check `training-queue.json`, `summary.md`,
`campaign-instances-fixed.log` and `two-object-status.md` for current evidence.
The new `pose_insurance.pack_combined_joint` module preserves the uploaded parent
on every unadopted row and keeps the pinned old packer untouched. Exact required
report strings are in `report-disclosure.md`; this job never edits the report.

## Clip-wide instances and symmetry instruments (4 November, 16:15 UTC)

The current campaign reads `data/outputs/moving_policy/two-object/control.json`
between episodes. Set `pause: true` to wait without a lease, or edit its ordered
`queue` and explanatory `note`. Completed outcomes are reused. The allowed
population stays the sixteen scored two-object episodes. The initial requested
order was 20, 25, 28, 30, 32, 33, 3, 5, 6, 19, 1, 8, 10; the lead may reorder it
through the control file. Every passing reference and Stage-0 result goes
immediately to the unchanged episode-38 training recipe before the cutoff.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 setsid nohup \
  data/.runtime/linux-cuda/bin/python -B \
  -m src.features.moving_policy.two_object_campaign \
  > data/outputs/moving_policy/two-object/campaign-30g.log 2>&1 < /dev/null &
```

Do not launch this beside an existing campaign. Its current PID/log are recorded
in `two-object-status.md`. Results/ledger are under `campaign-30g/`; terminal
markers are `completed.json` or `error.json`. `training-queue.json` records
starts, ends, admission and cumulative candidates. Scientific stops remain
stops; changing the queue does not grant another attempt.

`two_object_clip_instances.py` and `two_object_clip_worker.py` retain separate
frame-zero instances and, when the qualifying boxes overlap, search frames
5, 10, ... with the same thresholds and assignment rule. Both object masks are
seeded at the first separated frame and propagated toward both clip ends. The
frame-zero mask-overlap gate remains. Existing hands use the original worker.
The explicitly revised episodes 19/1 use new `clip-instance-separated/` outputs;
all old results remain. Episode20's interrupted operation retains the completed
slot0 segmentation and completes only slot1.

`_two_object_rotation.py` contains pure swing-twist measurements. The wrapper in
`two_object_rotation.py` captures forward/backward closure matrices while
returning the unchanged original pose fits and observed flags. Cups discard
axial twist; the lid reduces it modulo 180 degrees. Joint policy and support
orientation checks use the same instrument and unchanged thresholds.
`two_object_rescore.py` reads saved episode8/10 tracks without re-tracking or
changing their bytes. Their old scalar-only forward/backward angles cannot be
exactly decomposed; they are retained conservatively and explicitly counted.
No validity flag is restored without an explicit saved rotation-only rejection
reason. These changes do not fix translation, stereo, table or coverage failures.

The disclosed protocol revisions precede execution. `equivalence-30g.json`
records twelve CPU imports, unchanged shared source hashes and exact old/new
episode38 policy-metric equivalence. The required report specification now has
nine shared sentences. No new training/admission success is implied by the
implementation or its CPU checks.

## Conditional camera-region decision (4 November, 16:31 UTC)

`_two_object_camera.py` composes the original M1 result, then accepts exactly
three regions only when all five unchanged direct-camera checks pass: held-out
reprojection median/p95, track count, coverage and final perpendicular world
residual. Four regions retain the original composition exactly. The early
camera subset can proceed with three regions provisionally; this does not grant
PASS before the final world-residual instrument exists. Joint admission
independently recomposes the condition. Shared M1, spoon and episode38 files
remain untouched.

The authorized reference-only reruns are19,14,16, using the same masks, instance
results, A/B roles and mask-gate numbers. Outputs live in
`campaign-camera-regions/`, with a fresh ledger and execution record. Existing
scientific stops remain in previous campaign directories. Subsequent episodes
use this same conditional rule uniformly. The current control queue starts
19,14,16, then25,28,30,32,33,3,5,6. Training, cutoff, evaluator and packer remain
unchanged. `camera-regions-equivalence.json` records exact old four-region
composition, nine protected source identities, and denial when any of the five
prerequisites fails. The report specification/admission check now includes ten
shared sentences.

## Whole-clip facing, own-mesh flip and three table patches

The disclosed revision first retains declared scorer symmetry, then checks the
own mesh's 180-degree invariance. `_two_object_facing.py` measures symmetric
vertex Hausdorff distance with SciPy cKDTree and requires a convex-body or
mapped-triangle certificate before accepting its <=2mm surface bound. The
original winner remains unchanged when this resolves the tie. Otherwise
`two_object_facing.py` captures all four unchanged initial fits, tracks the
winner and best opposite seed, and compares the existing finite reprojection
IoU medians. A strict difference >0.02 selects one; otherwise the tie remains.
Both comparison arrays, initialization poses and hashes are saved. All original
quality gates apply to the selected run. Admission recomputes the decision.

`two_object_table.py` reuses the original plane, gauge, frame selection, patch
predicates and >=100 inlier-point rule. Exactly three available patches now
supply the same maximum pairwise normal angle at the unchanged final <=3deg
limit. Four patches return the original result exactly; fewer than three remain
unavailable. The sidecar records normals and inlier counts. Shared M1 and spoon
sources are unchanged.

Current outputs: `campaign-facing-table/`, log `campaign-facing-table.log`.
The boundary reload allowed the in-progress episode30 to finish; its STOP was
retained. New queue19,25,15,32,33,3,5,6,14,20 excludes completed30. Authorized
reruns reuse masks, instance results, roles, mask measurements and the exact
camera solve files. Completed frame caches are reused where available. Every
reference+Stage0 pass trains immediately, one attempt each, unchanged recipe
and cutoff. Control-file steering still applies between episodes and to the
CPU-only preparer during training.

The CPU-only diagnosis can also be run once:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  data/.runtime/linux-cuda/bin/python -B \
  -m src.features.moving_policy.two_object_validity_diagnosis
```

`--campaign-pid PID` refreshes the analysis until that process exits. Cached
per-reference results avoid recomputing unchanged data. `validity-diagnosis.md`
contains one table and a dominant-cause line per completed reference, linked to
per-frame CSVs. It separates actual rejection predicates from diagnostic gate
exceedances. First/last hand-mask overlap supplies explicit phase proxies;
physical contact, carrying and placement are not labelled. Analysis reads only
saved video/reference outputs and our own procedural assets and never uses GPU.

`facing-table-equivalence.json` records exact real-input initial pose/dictionary
comparison, unchanged admitted ep38 four-patch gauge/report, protected source
hashes, and actual ep25 mesh-flip certificate. Required report specification and
joint admission now contain thirteen shared disclosure sentences. See the live
summary for empirical rerun outcomes; no policy success follows from these
implementation checks alone.

## Admission review corrections

`two_object_contract.py` and `_two_object_contract.py` provide the shared
training/admission boundary. All eight named Stage0 checks must exist and have
`passed is True`; the actual task, support USD and independent task-reference
NPZ must be in the checked inventory. A's full source pose sequence and clock
are compared with the task, and the official evaluator's reference columns are
compared with that hash-bound task reference. A completed evaluator command
binds Stage0, final checkpoint, both output paths and their hashes.

Checkpoint metadata is read on CPU from the actual safetensors resolved config.
Seed0,1024worlds,ratio2 and the entire curriculum dictionary must match the saved
episode38 checkpoint configuration. The parent training job waits/reaps its
single CPU preparer before evaluation or further preparation, including failure
paths; a preparer failure propagates and stops the campaign.

`packed_starts` checks the uploaded moving-v5 baseline hash and the matching
achieved export, verifies official slot order and converts xyzw->wxyz. The
builder uses this start for B and retains A's existing translation anchor.
Training and admission independently re-check this binding. Old untrained tasks
must be preserved and rebuilt; the review audit found zero production joint
tasks and zero training attempts, so no rebuild was necessary.

`review-fixes-status.json` holds training, admission and cumulative packing
until verification passes. `review-fixes-verification.json` records nineteen
negative boundary cases, eight CPU imports, ten protected source identities,
and the real episode19 start correction0.406439mm/0.111775deg. These checks do
not constitute a new trained policy, official rollout or successful admission.
After the natural boundary at episode32, `campaign-reviewed.log` and
`campaign-reviewed-current.json` identify the continuing process and terminal
paths. Previous results remain in `campaign-facing-table/`.
