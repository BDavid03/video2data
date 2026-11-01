# Pose placement protocol (fixed 2026-11-02, context 04, before any placement candidate was scored)

D2 v1 is a static entry: each object stays at the initial pose the video gives it (drift
< 0.05 cm on all 20 scored episodes). Its tuning diagnostics (39, 42) have one object each, so
they equal the oracle floor by construction and say nothing about two-object episodes, where
most pose points are lost. This protocol fixes, before scoring, how two-object placement is
measured and when a change is adopted for a repack.

## Why placement is scored in slot 0's body frame

The kit (`metric_code/track_3/AUC.py:113`, `_warp_to_reference_start`) applies one rigid
transform per world that maps the achieved first pose of slot 0, rotation included, onto the
reference first pose of slot 0. Slot 1 is therefore scored by its pose relative to slot 0's
body frame: an error in slot 0's frame convention (up axis, yaw, a 180-degree PCA flip) swings
slot 1's whole offset vector, and slot 1's own frame convention sets its orientation error.

## Episodes

Public development episodes with two objects, all from the frozen train set: 0, 11, 12, 18, 21,
23, 26, 31. Each has an evaluation analogue with the same object pair in the same slot order
(public parquet order = `track_3_evaluation_objects.json` order):

| Train | Objects (slot 0, slot 1) | Scored analogue | Video state in frame 0 |
| --- | --- | --- | --- |
| 0 | white_pot, white_pot_lid | 1 | merged cluster (as ep 1) |
| 11 | blue_cup, beige_cup | 10 | two clean clusters |
| 12 | blue_cup, plastic_dish_rack | 14 | cup cut by the image border |
| 18 | mini_sweeper, mini_dust_pan | 19 | two clean clusters |
| 21 | blue_cup, water_pitcher | 20 | two clean clusters |
| 23 | wooden_piece_1, wooden_piece_2 | 25 | two clusters, identities arbitrary |
| 26 | plastic_dish_rack, white_pot | 28 | merged cluster (as ep 28) |
| 31 | mini_sweeper, plastic_dish_rack | 30 | merged cluster (as ep 30) |

Tuning 39/42 (one object) add nothing here. Episodes 41, 43, 44 are never used. The older
"validation" public episodes (2, 13, 22, 24, 27, 40, 45) are outside the frozen protocol and are
not used.

## What is scored

- **Candidate scene:** the D2 v1 scene construction (`pose_insurance/scene.py`, `_scene.py`),
  unchanged except for the candidate under test: MoGe-2 frame-0 depth, the `mesh_extents`
  measure clusters, the recorded video assignments, D1 v1 boxes at MoGe-2 metric size, defaults
  where no cluster is assigned. No mocap pose, scan geometry or scan dimension enters a scene.
- **Static rollout:** the scene's initial poses held for every frame (measured drift < 0.05 cm).
  The adopted candidate is then run through the real evaluator on the affected scored episodes
  before any repack (drift check of the slice).
- **Reference:** the public mocap poses (`observation.objects`, `[x, y, z, qw, qx, qy, qz]`,
  invisible frames filled from the nearest visible one, as `scripts/track3/static_pose_baseline.py`).
- **Metrics:** the kit's own functions (AUC, SP-SR, MP-SR, RPE, MPPE) after
  `_warp_to_reference_start`, one world. AUC uses 1,000 surface samples of each public scan
  (seed 0), an approximation of the host's vertices; the scans are used only for scoring.
- **Floor:** the oracle static scene (mocap frame-0 poses held), from the same code.

## Decomposition (diagnostic only, never a candidate)

Per episode, with the relative transform of slot 1 in slot 0's body frame, achieved
`(R01_a, t01_a)` and reference `(R01_r, t01_r)`, report: `|t01_a - t01_r|` (cm), the distance
ratio `|t01_a| / |t01_r|` (scale), the angle between `t01_a` and `t01_r` (placement direction
plus slot-0 frame), the geodesic angle between `R01_a` and `R01_r` (slot-1 frame), and the kit
metrics with one factor at a time replaced by its reference value: offset direction, offset
length, slot-1 relative rotation. Where the public data allow a common frame between the video
world (table plane z = 0) and the mocap frame, also report each object's up-axis convention and
heading error directly.

## Candidates (at most three, fixed now)

- **K1 merged-cluster placement.** A slot whose object shares one frame-0 cluster with the
  other slot (train 0, 26, 31; scored 1, 28, 30) is placed at that cluster's centroid in the
  table plane instead of the 0.5 m default: the lid seated on the pot's top face (same table-plane
  centre), any other pair resolved by the slice's existing minimal footprint separation; identity
  yaw replaced by the merged cluster's PCA yaw. The cluster dimensions stay the D1 boxes.
- **K2 up-axis convention.** For each object seen in public train mocap, the body axis that is
  closest to the support normal at frame 0 (majority over that object's train episodes,
  leave-one-episode-out when scoring an episode that contains it) is mapped onto the table
  normal in our initial pose; objects never seen in train mocap (wooden_spoon) keep z up.
- **K3 heading convention.** For each object, the in-plane angle between its mocap body heading
  and the video PCA long axis, estimated mod 180 degrees, leave-one-episode-out as K2, applied on
  top of K2. Only if the route call shows a common frame between video and mocap exists in the
  public data; otherwise K3 is not run and the reason is recorded.

Candidates are scored alone and cumulatively in the order K1, K1+K2, K1+K2+K3.

## Adoption (for a repack)

A candidate is adopted only if, over the eight train episodes:

1. mean AUC rises by at least 0.02 (about the AUC change of 1 cm of slot-1 offset, from the
   context-01 5 cm sensitivity: 0.12 AUC per 5 cm);
2. mean SP-SR and MP-SR do not fall, and mean MPPE does not rise;
3. no single episode loses more than 0.10 AUC.

The adopted scene must then pass the slice's evaluator drift check on every affected scored
episode. The repack, its run records and these tables go in `docs/chain/SUBMISSION.md` before the
file is listed for the owner. Losers are logged in `docs/chain/IDEAS.md`.
