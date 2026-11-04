# W1.0 protocol: what is frame-0 position worth alone? (`COMPLETION-PLAN.md` W1; fixed before any new number below exists)

## Question

The plan's figure of 0.387 for correcting train 0/26/31 replaces those rows with O3, which corrects rotation and offset together. It
says nothing about position alone. W1.0 measures the position term by itself, on train labels, so that W1 and the rotation routes can
be ranked by marginal gain per hour. These arms are oracles: they use mocap and are never candidates for a pack.

## What was already known when this was written (disclosure)

Recorded and read by the author, per episode: T1, O1, O2, O3 (`outputs/pose_insurance/heading/summary.md`). Means: T1 AUC 0.276705,
RPE 29.01 cm, MPPE 17.68 cm; O2 0.352713, 20.27, 11.35; O3 0.531919, 15.11, 7.40. O2's gain is 45 % train 18. T1's gain over S1 was
+0.0077 AUC, the only measured value for a correct stereo offset without its direction. On T1's clean episodes the direction after
the kit's alignment was off by -51 / -98 / +95 / -130 / -2 degrees (`pose-stereo-placement-protocol.md`). No arm below that is marked
new has been computed.

## Why direction gets its own arm

The kit aligns one rigid transform from slot 0's first pose. An offset measured correctly in the camera or table frame is therefore
rotated by slot 0's unknown heading error, and it lands in the wrong direction. W1.1's masks can fix the length and the camera-frame
vector, but they do not fix this. PO assumes it solved; PL-W1 shows how much depends on it.

## Arms (eight train episodes 0, 11, 12, 18, 21, 23, 26, 31; recorded T1 start poses; slot order and fill as the heading protocol)

Define `(C_ref, b_ref) = relative_transform(reference_start)`, `R0 = Rotation.from_quat(T1_start[0, 3:])`, `R0_ref` the reference's slot-0
rotation, and `A = R0_ref R0^T`, the rotation the kit's alignment applies. `d_ref = p1_ref - p0_ref` is the reference offset in the
reference world.

| Arm | New? | Definition | What it shows |
| --- | --- | --- | --- |
| T1, O2, O3 | no | recomputed by `_heading.oracle` | reproduction and comparison |
| PO | new | rotations as T1; `p1 = p0 + R0 b_ref` on all eight | slot 1 exact at frame 0 after alignment, position alone |
| PO-W1 | new | PO on 0/26/31; T1 elsewhere | W1's roster with direction and height assumed solved |
| PL-W1 | new | on 0/26/31, `p1 = p0 + A^T Rz(k * 15 deg) d_ref` for each `k` in 0..23, every metric averaged over the 24; T1 elsewhere | **conditional yaw-sensitivity diagnostic, not an estimate of W1's benefit.** After alignment it keeps the reference's height and horizontal radius. It assumes tilt solved; the uniform yaw weights are stipulated, not inferred |
| RC | new | positions and anchor pose as T1; slot 1 rotation `R0 C_ref` | pure relative rotation, no position or direction change |

Metrics: the kit's AUC, RPE, MPPE, SP-SR and MP-SR from `static_pose_baseline.score`, one world, 1,000 scan samples, seed 0, as every
earlier static diagnostic. PL-W1 also reports each episode's minimum and maximum over the 24 directions.

**These are not upper bounds on whole-clip static AUC.** Translation can compensate rotation error, and an intermediate static position
can beat the initial one (episode 18's H2S scored 0.616 against O3's 0.606). They also repair full XYZ, whereas the existing placement
code edits XY only, so they are not forecasts of that code.

## Validity checks (any failure blocks the ranking)

1. T1, O2 and O3 reproduce `outputs/pose_insurance/heading/summary.json` per episode within 1e-9 for all five metrics.
2. PO: on the unwarped poses, slot 0 and all rotations equal T1's bit for bit. After `_warp_to_reference_start`, slot 1's frame-0
   position equals the reference within 1e-9 m on all eight.
3. PL-W1: `k = 0` equals PO-W1 within 1e-9 per metric. After alignment, for every `k`, slot 1's vertical displacement from slot 0 and its
   horizontal radius equal the reference's within 1e-9 m.
4. RC: positions and the anchor pose equal T1's bit for bit, and the frame-0 relative rotation equals `C_ref` within 1e-9 rad.
5. Single-object tuning 39/42 have no slot 1, so they are not run. Episodes 41, 43 and 44 and validation 2/13/22/24/27 are not read.

## Reporting

A per-episode table and an eight-episode mean for every arm, with pure marginals in both orders, because position and rotation interact:
position first (`PO - T1`, then `O3 - PO`) and relative rotation first (`RC - T1`, then `O3 - RC`). O2 is reported separately, as relative
rotation plus the existing anchor-direction intervention.

## Scenario table (fixed now; a calculation, not a scheduling rule)

Each route gets `g x r / (20 h)`. `g` is its mean AUC gain per treated train episode, `r` its scored reach out of 20, `h` its hours.
Episode 19 already ships H3, so train 18, its analogue, is excluded from rotation gains. Reach assumes transfer from labelled train
episodes to unlabelled scored ones; episode 5 (spoon pair) and episode 8 have no exact train-pair analogue. RPE and MPPE are shown beside
every AUC figure.

| Route | `g` from | Treated train episodes | `r` | `h` |
| --- | --- | --- | --- | --- |
| W1, length with unsolved yaw | PL-W1 minus T1 | 0, 26, 31 | 4 (episodes 1, 5, 28, 30) | 3 |
| W1, direction solved | PO-W1 minus T1 | 0, 26, 31 | 4 | 3 |
| Rotation (W2a or H3) | O2 minus T1 | 0, 11, 12, 21, 23, 26, 31 | 8 (episodes 1, 8, 10, 14, 20, 25, 28, 30) | 4.5 |
| Rotation and offset by pose estimation (W2b) | O3 minus T1 | same seven | 8 | 4, unverified |

The rotation rows are perfect-model figures and the first W1 row is not, so this favours rotation by construction. It says how much there
is to win, not what will be won.

## What the result does

W1.0 kills nothing and sets no order by itself. The order is fixed separately, before its numbers: a bounded W2b pilot on train 12 and 31
goes before W2a training (it needs no trained model and can recover both rotation and offset), and W1.1's masks are built in any case,
because W1.3, W2b and the merged episodes all need them. A weak PL-W1 does not close W1, since it does not test real video-derived offsets;
W1.1's own gate decides that. A pack still needs a gate of its own. The table above reorders this only if a later route's figure exceeds an
earlier route's by more than a factor of 2 on both AUC and RPE, which the lead records with the reason.

**Scope.** CPU only, new files only, under `outputs/pose_insurance/position-oracle/`. Scans enter only the scoring, as in every earlier
static diagnostic. No pack manifest file, `AGENTS.md` or recorded `run.json` is changed. The outcome goes in
`docs/chain/position-oracle-outcome.md`.
