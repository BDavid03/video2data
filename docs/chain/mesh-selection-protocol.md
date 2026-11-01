# Mesh selection protocol (fixed 2026-11-01, context 03, before any new mesh was scored)

D1 v1's box-versus-ellipsoid choice was made on the same 10 public scans that report its score.
From now on every mesh change follows this protocol, written down before scoring. The scans stay
measurement-only: no scan geometry, dimension or template enters a submitted mesh.

## Baseline

D1 v1 (one video-measured box per object, build run
`outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff`), scored with
`scripts/track3/local_cdo.py` over kit seeds of episodes 1, 3, 5 (per-object table in
`outputs/mesh_eval/d1-ctx02/v1-box.json`).

## Rules, not objects

A candidate is a **rule** that maps video measurements (and the object's name category, written
below before scoring) to a shape. Picking a shape per object by its scan score is not allowed.

Categories, fixed now from the object names and how they look in the video:

| Category | Objects |
| --- | --- |
| round vessel (revolved profile plausible) | beige_cup, blue_cup, white_pot, white_pot_lid, water_pitcher |
| flat or blocky | wooden_piece_1, wooden_piece_2, plastic_dish_rack |
| thin tool | mini_sweeper, mini_dust_pan, wooden_spoon (no scan) |

## Two-fold leave-objects-out

Folds, fixed now (each holds part of every category that has a scan):

- Fold A: beige_cup, white_pot_lid, wooden_piece_1, plastic_dish_rack, mini_dust_pan
- Fold B: blue_cup, white_pot, water_pitcher, wooden_piece_2, mini_sweeper

Any free parameter of a rule (a threshold, a handle fraction, a taper) is chosen on fold A and
scored on fold B, then chosen on fold B and scored on fold A. The **cross-fitted score** is the
mean of the held-out per-object scores over all 10 scanned objects. A rule with no free parameter
is scored once on all 10; that is its cross-fitted score.

## Adoption

A rule replaces the box in a repack only if all three hold:

1. its cross-fitted per-object mean beats the box's (1.008 cm) by more than 0.03 cm (several times
   the largest per-object seed spread in v1, 0.036 cm on the rack, divided by sqrt(10));
2. no held-out object gets worse by more than 0.15 cm;
3. the pair-weighted mean over the 26 scanned pairs does not get worse.

Report the cross-fitted numbers as the headline, beside the in-sample ones, and log every rule
tried (including losers) in `docs/chain/IDEAS.md`. At most three rules per context, to bound
selection on 10 objects.

## The spoon (no scan)

The spoon cannot be scored. A spoon shape changes only through a rule that also covers scanned
objects (for example "thin tool: ellipsoid head plus box handle, sizes from the measured extents")
and passes the adoption test on the scanned thin tools and the rest; or through a video-only proxy
(silhouette IoU of the projected shape against the opening-frame depth cluster) that has first
been shown to rank shapes the same way CD-O does on the 10 scanned objects (Spearman >= 0.6 over
box, ellipsoid and the candidate). Otherwise the spoon keeps its box.

## Repack

Only when a rule is adopted. The repack, its run records and the cross-fitted table go in
`docs/chain/SUBMISSION.md` (Packed files) before the file is listed for the owner.
