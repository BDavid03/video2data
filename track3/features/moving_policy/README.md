# M2: tuning-39 contact-driven task

Stage 0 implements `docs/chain/moving-m2-protocol.md`. **STOP:** the fixed grasp fails
the wrist-speed gate and the zero-action CPU support gate. No training or GPU work ran.
The gates were not changed. One permitted grasp repair was used, before training.

## Build and inspect

From the checkout root, using the existing environments:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.moving_policy stage0
```

This one command builds the asset, support USDA and task; runs the installed
FlashCHORD loader and CPU FK; measures the six checks; attempts the evaluator's
physics on CPU; renders a four-frame strip; and writes dry training/scoring argv.
It refuses to overwrite an existing `stage0.json`. Results and caches belong only
to `outputs/moving_policy/ep39/`. The implementation reads episode 39 explicitly;
it does not discover or iterate other episode files.

`request.py`, `runtime.json` and `queue.json` fix the scope. `geometry.py` contains
pure geometry. `handler.py` owns files, provenance, subprocesses and admission;
`cpu.py` uses the installed Newton/Warp/FlashCHORD runtime. No dependencies are added.

## Asset, frame convention and grasp

The asset is D1's existing `thin_tool_profile` constructor and frozen dust-pan
profile, sized using **only public-39 video-depth extents**: 0.256454 × 0.213388 ×
0.054363 m. No dust-pan scan is read in Stage 0. Its broad tapered scoop, low lip
and elliptical grip are our procedural reconstruction, not the captured scan.
The reconstruction is rotated into the mocap body frame so it starts level; object
poses retain their exact mocap values. This is an explicit upper-bound coordinate
choice, not a recovered camera-to-mocap transform. A static support cube is placed
at the reconstructed initial bottom height. Mass is 0.1 kg with bounding-box inertia.

Visual and collision geometry use the same OBJ in a free-root rigid URDF.
FlashCHORD's `scene/objects.py` convex-decomposes mesh collision with CoACD's
defaults. This task has 28 convex collision pieces. A concave mesh without
decomposition would collide as one filled convex hull; we do not use that shortcut.

The deterministic template frame is the first maximum contact-coverage sample of
the shipped tissue-box sequence after it rises 5 cm: **frame 132**. Captured finger
joints stay fixed. Each hand's middle fingertip is placed near a longitudinal
exterior face: **left at the scoop front lip, right at the distal grip**. Four fixed
wrist rolls are compared by geometry alone, keeping the hand solids outside the
asset's enclosing hull. Two hands supply the opposing exterior-face contacts;
one-hand sufficiency has not been established. The handle is too narrow for its
side faces to count as opposite exterior faces of the full asset under the frozen
clearance criterion, so the rule does not claim such a handle pinch passes that gate.

For 0.5 seconds, both pre-shaped hands approach from 6 cm outward with a smooth
fixed interpolation. Afterwards each wrist and all 67 robot frames follow
`T_WH(t) = T_WO(t) T_OH`. Contacts and normals transform with the object, with one
middle-finger contact group per hand and no contacts during acquisition. There is
no inactive hand. Legacy MANO fields are transformed template placeholders, not
estimates of the human's hands. Installed Sharpa FK derives the actual robot targets.
The object reference is copied from mocap; the builder never generates object
motion for a rollout. Only a learned policy through the unedited evaluator could
produce submission trajectories.

## Stage 0 result

| Check | Final measurement | Verdict |
| --- | --- | --- |
| 1. Schema | 319 frames, exact 62-field physical template schema; 2 hands × 22 joints/67 frames/16 contacts; installed loader and binding succeed | PASS |
| 2. Closure | 308/308 post-acquisition frames (100%); distinct middle-tip groups on opposite exterior faces | PASS |
| 3. Penetration | 0 m after repair; initial rule 0.00241566 m | PASS |
| 4. Kinematics | Limit excess 5.02e-9 rad; wrists 1.40309/1.59152 m/s versus example maximum 0.214100 m/s; ratio 7.43354, limit 2 | **FAIL** |
| 5. Support | CPU, 11 zero-residual steps: translation 10.51667 m, rotation 0.731965°; limits 0.01 m/2° | **FAIL** |
| 6. Reference truth | Maximum translation difference 2.19011e-7 m after the actual control-rate loader; kit AUC 1.0 | PASS |

The one repair moves both grasp targets 5 mm farther outward (3 → 8 mm), with
the same deterministic roll-selection rule. It removes the initial 2.42 mm
penetration but does not repair kinematics. Both attempts are retained separately;
there is no further grasp variation. The penetration check solves linear programs
over full convex collision solids, rather than sampling fingertip centers. All
post-acquisition hand/object geometry is rigid, so one collision pose covers that
interval; measured FK group variation is below 0.9 micrometres, including source-rate rounding.

The initial physics attempt exposed a Hydra quoting bug in a path containing `=`.
That implementation bug was corrected without changing the grasp or checks. The
corrected CPU run completed physics and failed support. Its raw poses are retained
for independent checking. Initial hand geometry extends **6.05/4.19 cm below the
support plane**, a likely explanation for the large impulse; geometric object
closure does not establish a physically feasible grasp with the table present.
An identical CPU run reproduced both maxima exactly; an independent float64
recalculation from its raw poses gives 10.516669935 m and 0.731965249°.

The collision audit uses the exact installed Newton URDF import before CoACD,
including float32 vertex conversion, rather than a separate float64 mesh
decomposition. The initial diagnostic snapshots are retained. Final checks also
use the actual simulator control rate: the exact template schema stores `fps` as
float32, while the simulator accepts the effective rate as a Python float.
Those implementation corrections change neither the grasp rule nor any gate.

`stage0.json` hashes the scientific inputs, source files and output artifacts.
`grasp-attempt-0.json` and `grasp-attempt-1.json` preserve both rule outcomes;
`geometry-audit.json` records support interference, and `strip.png` shows frames
0, 11, 231 and 318. This is a reference visualization, not a policy rollout.

## Motion and transfer limits

39 has **319 frames, 15.700501 s**, effective rate **20.254130 Hz**. Float timestamp
rounding differs from a uniform grid by at most 0.783 microseconds. We use that
effective rate in both task and simulator, unit motion speed, and the complete
frame grid: no 20 Hz/half-speed truncation. Object displacement reaches **39.10 cm**,
path length 191.61 cm and maximum speed **0.71293 m/s**. The example reaches
17.38 cm, 40.94 cm path and 0.09080 m/s. The pan origin rises **27.34 cm** (example
17.21 cm): it is lifted, not merely slid. With our leveled asset, 178 frames have
the mesh bottom over 1 cm above support; maximum clearance is 23.04 cm. The same
approximate asset intersects support by up to 2.70 cm elsewhere in the reference.

The builder currently admits **39 only**. A later scored-38 extension needs a
new admitted request and a video-derived reference in the same metric body/frame
convention, with camera motion removed and gravity aligned. It must preserve
frame timing, names/order, complete trajectories and body-frame mapping; replacing
the mocap file alone cannot prove those contracts. Stage 0 failures here block
training and scored adoption under the current protocol. Whether another grasp
could work, GPU physics equivalence, force closure and learned tracking remain
unproved. No conclusion about all video-reference policies follows from this failure.

## Dry training and scoring commands

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.moving_policy train
PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.moving_policy score
```

These print exact upstream argv, also saved in `commands.json`, `train.argv.txt`,
`score-32814080.argv.txt` and `score-66763776.argv.txt`. Training reuses the existing
`dynamic_feasibility.handler.training_command` without editing it: fresh seed 0,
1,024 worlds, ratio 2.0, replay 1,048,576/minimum 6,400, interval 504,832, unchanged
D1 thresholds, 66,763,776 transitions, LR horizon 130,398. The same upstream
cooperative worker enforces a 75-minute deadline when authorized.

Adding `--run` requires the lead's corresponding runtime flag, six passing gates,
unchanged hashes and `GpuLease`; all flags remain **false**. Existing completed
jobs are skipped, unfinished jobs are not silently relaunched. The score operation
evaluates the boundary and final checkpoints once with one explicit-reset world,
then reuses M0's motion functions and the kit scorer against tuning-39 mocap.
Public scan surface points would be used **only at that later measurement step**,
never in the simulated asset or Stage 0 builder.

If CPU physics had been unavailable, the smallest separately gated GPU support
command would be the following. CPU physics was available and failed, so this is
not needed to fill a missing result:

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python -B -m track3.features.moving_policy support-gpu --run
```
