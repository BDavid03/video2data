# Pose insurance

Train a small FlashSAC parked-hand policy for each video-derived static box scene,
then run the unchanged Submission Kit evaluator to export achieved object poses.
The final `results/` directory contains official trajectory Parquets only. No packing
or upload is performed here.

Execution is authorised by the owner's 2026-10-18 “all flags are go” direction and
this D2 brief for the 20 scored episodes and public tuning episodes 39/42. Runtime
flags and the episode allow-list are in `runtime.json`. Episodes 41/43/44 are refused.
The FlashCHORD source and environment are read-only. No packages were installed.

## Run

From the worktree, with no other GPU operation running:

```bash
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" PYTHONPATH=. \
  .runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase pilots

# After reviewing successful pilots and confirming <=30 minutes per episode:
setsid nohup env PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" PYTHONPATH=. \
  .runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase evaluation \
  > outputs/pose_insurance/loop.log 2>&1 &
```

`queue.json` sets the ordered pilots, scored episode list and editable budgets
(environment steps, worlds, replay minimum). Each trial calls the slice handlers;
they enforce gates, create hashed `run.json` records and acquire `track3.gpu.GpuLease`.
The loop hashes requests and code, checks completed exports before skipping them,
appends `ledger.jsonl`, and writes `summary.md` and `checkpoints.json`. A phase ends
with `<phase>_finished.json`; any failure stops the queue. Failed records are retained.
Use a changed budget for a new identity. A diagnostic uses public labels only after
export and writes a separate `public_label_diagnostic` artifact.

The initial contract budget is one world / one step / replay minimum one, n-step
three, so replay has no committed transitions and no update should occur. The
learned budget is one world / three steps / replay minimum one, n-step three.
The handler reads `global_update_step` from the saved learner state and requires
it and the actor Adam step counter to be positive for every learned export. Training uses float32 and finite state
checks; these tiny budgets establish optimizer use, not meaningful manipulation skill.
Three is the minimum budget with this one-world, three-step replay: the first two
collections remain pending and the third commits one sample and performs an update.
The initial eight-step pilots recorded six learner updates and three actor Adam steps.
The actor/critic widths are 32/64 with one residual block each, batch eight, replay
capacity 64. Evaluation always uses one world, deterministic inference, explicit
reset, 20 Hz and zero VOC assistance.

The CPU task batch is resumable as well:

```bash
PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" PYTHONPATH=. \
  .runtime/flashchord/venv/bin/python -m track3.features.pose_insurance.loop --phase tasks
```

It has built all 20 scored tasks; `outputs/pose_insurance/tasks.json` records their
producer paths and every fallback decision. Each policy run retains its own complete
task and assets beside its producer run.

## Geometry and fallback

Task positions use MoGe-2 monocular metric-depth predictions from the recorded
`mesh_extents` producers. Their scale is unanchored: there is no measured absolute
scale error bound. Opening-frame spread is not a calibration uncertainty estimate.
D1 v1 box proportions come from build `20261101T103838.400965Z-117fbcff`; the
per-episode measured geometric-mean dimensions restore size where available.
Otherwise the pooled D1 size is used. These boxes simplify visible object shape.

The table normal defines world +z; its plane becomes z=0. Camera +x projected into
the plane defines world +x. Each object centre is projected horizontally into this
frame and seated at half its metric box height. Its yaw is the frame-0 cluster's
in-plane PCA heading, with a deterministic sign and right-handed axes. The private
pure helper adds axes while retaining the original extents numbers. The old slice
is unchanged. `outputs/pose_insurance/geometry_equivalence.json` records exact
agreement on all three original fields for 8,313 real episode-38 points.

Unassigned slots use the largest unclaimed frame-0 cluster, without claiming a name
match, with pooled D1 size. If no frame-0 cluster is available, use documented
0.5 m slot spacing and identity orientation. Only fallback placements are shifted
when bounding radii would overlap. `initial_scene.json` records every decision.
Fully unassigned scored episodes: 3, 14, 28, 30. Additional missing initial slots:
1 (pot and lid: its reviewed assignment starts at frame 110), 6 (pot), 20 (cup),
32 (cup). No public 0/26/31 tasks are generated: they are outside this slice's scope.

The host aligns only slot 0's first pose. Our PCA box frames are arbitrary relative
to the reference's object frames; slot 1 orientation/placement error remains scored.
Tuning episodes 39 and 42 are single-object, so they do not measure that two-object
risk. Both are nevertheless exported and scored with the kit's pose functions.
The oracle static reference over 17 public episodes is AUC 0.460898, SP-SR 0.411765;
it starts from true public poses and is not an expected reconstructed-scene score.

## Evidence

`outputs/pose_insurance/contract.json` inventories the shipped tissue-box Parquet:
one row, 390 source frames at 30 Hz, 22 joints and 67 robot frames per hand, 16
contact slots, and one object. It has no schema_version, uses an all-zero articulation
placeholder, and includes extra MANO and task-frame metadata. This matches the
legacy reader's required columns; this slice retains only required robot fields,
translates wrist and frame positions consistently, and disables every contact ID.
No MANO assets or pose labels are used to build scored tasks.

The queue's latest results, including failures, are in `summary.md`; per-run logs
and telemetry are beside the checkpoint and exported trajectory. Peak MiB is sampled
device-wide usage including unrelated/display allocation, not isolated process VRAM,
and short peaks can be missed. Drift is maximum motion from the first exported pose;
the evaluator records frame zero after one control step. Successful exports require
finite unit poses, valid environments, official slot names, no duplicate keys, every
roster key, and maximum drift <=1 cm and <=1 degree. Absolute initial seating is a
separate uncertainty.

Installed runtime identity: commit `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`,
evaluator package SHA-256 `b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1`.
See `outputs/pose_insurance/runtime_inventory.json` for file hashes and package versions.

## Latest measured result

The final three-step learned pilots all passed: evaluation 38/10 and public 39/42,
one learner update and one actor Adam step each, full roster coverage, <0.048 cm
drift, and 0.73–1.12 minutes per episode with warm caches. Public AUC was 0.174961
and 0.194066, respectively, identical to each episode’s oracle static AUC; both
SP-SR values were zero. The cold contract pilot took 5.14 minutes.

Detailed tables and immutable run paths: `outputs/pose_insurance/pilot_report.md`.
The 20-episode queue was launched detached; its PID is in `loop.pid`, its output
in `loop.log`, and successful completion is marked by `evaluation_finished.json`.
The initial eight-step pilots remain in the ledger as measured history.
