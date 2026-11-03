# Pose heading protocol: is a frame-0 heading cue worth building, and is it identifiable from video? (IDEAS 46; fixed 2026-11-03 08:03 UTC, context 16, before any number below was computed)

## Why

D2 v4 (T1, `docs/chain/pose-stereo-placement-protocol.md`) fixed most of slot 1's offset *length* (stereo error
0.58 cm on the clean five train episodes) but not its *direction*: after the kit's alignment slot 1 is off by
-51 / -98 / +95 / -130 / -2 degrees on 11 / 12 / 18 / 21 / 23. The scorer
(`upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/AUC.py:113`) aligns each world on slot 0's
**full** first pose, so for a static plan the only frame-0 quantities that still count are slot 1's offset in slot
0's body frame, `b = R0^T (p1 - p0)`, and slot 1's relative rotation `C = R0^T R1`. Both depend on the predicted
frame-0 rotations, which today carry no heading information. This protocol decides, in two cheap stages, whether a
heading cue is worth building before 12:00 UTC on 4 November.

## Data and integrity

- Train episodes 0/11/12/18/21/23/26/31 only (tuning 39/42 are single-object: invariance controls at most). Episodes
  41, 43, 44 are never read. Public validation 2/13/22/24/27 are spent and are not used here. No leaderboard use.
- Kinematic scoring of static plans exactly as in the stereo-placement protocol ("Scoring" section): the kit's five
  pose metrics on the full reference trajectories, train means over the eight episodes, T1 reproduced exactly (AUC
  0.2767, RPE 29.01 cm, MPPE 17.68 cm to the recorded precision) before any variant is believed.
- Every artifact is written under `outputs/pose_insurance/heading/` (or `.runtime/tmp/` for scratch), with
  `run.json` records; nothing pinned by the D2 v4 manifest is edited.

## Stage A: oracle ceiling (uses train mocap labels, descriptive; it decides only whether to continue)

Variants, each applied to every train episode with two or more objects, T1 positions and heights unchanged:

- **O1 (direction oracle).** Rotate slot 0's predicted frame-0 rotation about the world vertical (and hold it static)
  by the angle that makes the horizontal direction of the predicted `b` equal the mocap frame-0 `b`'s. Slot 1's
  predicted rotation is rotated by the same yaw (its relative rotation `C` stays T1's).
- **O2 (direction and relative rotation oracle).** O1, then set slot 1's rotation to `R0' C_ref`, where `C_ref` is the
  mocap frame-0 `R0^T R1`.
- **O3 (full frame-0 oracle, bound only).** O2 with the mocap frame-0 `b` vector itself (length and height too).

**Decision rule (fixed now).** Stage B runs only if the train-mean AUC of O2 exceeds T1's by at least **+0.010**
(T1's whole gain over S1 was +0.0077). If O2 is below that, the heading lever is closed, IDEAS 46 is recorded as
"ceiling too low", and nothing is built. RPE, MPPE, SP-SR and MP-SR of every variant are reported beside AUC,
per episode, with no further verdict. O3 is reported as a bound only.

## Stage B: identifiability kill test (only if Stage A passes)

A video-only frame-0 heading estimate for each object (route chosen by the lead after a Codex route call; the
default is registering the object's public scan to its frame-0 calibrated-stereo points with a yaw search). The
method, its parameters and any symmetry rule are frozen in an amendment to this file, committed **before** any
error on episodes 12/18/21/23 is computed. Development may use episode 11 only.

Evaluation, once, on clean train episodes 12/18/21/23 (frame-invariant, so no camera-to-mocap extrinsic is needed):
the horizontal angle between the predicted and mocap `b`, and the geodesic angle between predicted and mocap `C`.

**Pass (fixed now):** horizontal `b` direction error < 30 degrees on all four episodes, and `C` error < 30 degrees
on at least three of four. An episode the method declines (object symmetric, too few points) counts as a failure for
the `b` test. Fail means stop: IDEAS 46 is recorded with the per-episode errors and nothing is built.

## Stage C: candidate (only if Stage B passes)

A candidate H1 = T1 plus the frozen heading method, under its own amendment with the stereo-placement protocol's train
gates (mean RPE and MPPE not worse, AUC/SP-SR/MP-SR not lower, worst-episode AUC loss <= 0.02), fixed before it is
scored. If it clears them by 06:00 UTC on 4 November, it goes through the scored stage and the physical check as D2
v5; otherwise D2 v4 stands.

## Disclosure

Stage A uses train labels by design; its numbers are a ceiling, never a result. The report may mention the ceiling
and the kill-test outcome as findings.

## Amendment 1 (2026-11-03 08:15 UTC, context 16, after Stage A and before any up-axis-prior number)

**Stage A outcome (scratch run `.runtime/tmp/heading-ceiling/`, T1 reproduced to 3.6e-15):** train-mean AUC T1
0.276705, O1 0.286239, O2 0.352713, O3 0.531919. O2 - T1 = +0.0760 >= +0.010, so Stage B runs. Direction alone is
+0.0095; the relative-rotation oracle carries +0.0665.

**What the lead knew before writing this (disclosure).** Stage A's diagnostic `up_axes.json` (train labels) shows the
scan frames are not z-up: world +z expressed in each object's frame-0 body frame is about body -z for blue/beige cups,
+x for white_pot (episode 0), white_pot_lid, wooden pieces, plastic_dish_rack and mini_dust_pan, +z for mini_sweeper,
and tilted for water_pitcher; the pot is reported inverted in episode 26. T1's rotations are yaw-only boxes, so its
relative rotation C is wrong by 88-179 degrees mostly through this up-axis convention. The candidate below was chosen
because of that, so its train evaluation is not blind; leave-one-episode-out fitting is the only protection.

**Stage B2 (new): up-axis prior U1.** For each object type, the body-up axis `u_type` is the normalised mean, over the
type's train-episode occurrences, of world +z expressed in the object's frame-0 mocap body frame. Rules fixed now:

- Consistency: if any two occurrences' up vectors differ by more than 30 degrees, or the mean's norm is below 0.8, the
  type has no prior and keeps T1's rotation.
- Construction: each object with a prior gets frame-0 rotation `R = Rz(psi_T1) Q(u_type)`, where `Q` is the
  shortest-arc rotation taking `u_type` to world +z and `psi_T1` is T1's recorded box yaw (no heading information is
  added). Positions are T1's, unchanged. Objects without a prior keep T1's rotation.
- Train evaluation (the gate): leave-one-episode-out, so an episode's own labels never enter its prior; a type with
  no other train occurrence keeps T1's rotation in that fold. Scored on the eight train episodes with the same scorer
  as Stage A.
- Gates for U1 against T1 (train means, LOO): AUC gain >= +0.010; RPE and MPPE not worse; SP-SR and MP-SR not lower;
  worst single-episode AUC loss <= 0.02. In-sample (full-train) U1 is reported beside it, descriptively.
- If U1 passes, the candidate for the scored stage fits each prior on all eight train episodes (types seen in one
  train episode use that one occurrence, disclosed); types never seen in train keep T1's rotation. The scored stage
  then needs the box/collision geometry counter-rotated so the physical placement is unchanged, and the physical
  check (first pose <= 1 cm and 2 degrees, as for D2 v4) before any pack.
- If Stage B (registration) also passes, the lead chooses between U1 and H1 on the train LOO numbers, disclosed.

Stage B (registration kill test) is unchanged and continues independently.
