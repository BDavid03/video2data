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

## Amendment 1 (2026-11-03 08:09 UTC commit time, context 16, after Stage A and before any up-axis-prior number)

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

## Amendment 2 (2026-11-03 08:14 UTC, context 16): U1 outcome, and Stage B method HR1 frozen before any held-out error

**U1 (scratch `.runtime/tmp/heading-ceiling/summary_u1.md`, T1 reproduced exactly): FAILS.** Leave-one-episode-out
train means vs T1: AUC +0.000000 (needs +0.010), RPE -0.236 cm, MPPE +0.292 cm (worse), SP-SR/MP-SR 0. The prior
fixes the up axis (errors < 1 degree; mini_sweeper 4-8) but slot 1's relative rotation stays 92-180 degrees wrong
because the heading about up is still arbitrary; the inverted pot (0 vs 26) makes each leave-one-out pot prior
179 degrees wrong. In-sample U1 is 0.268844 (lower). Per Amendment 1, U1 is closed; no rule is re-tuned. A recorded
re-run of these numbers (`heading_loop.py`) is an independent implementation, not a new test.

**Stage B method HR1 frozen** as `docs/chain/pose-heading-method-hr1.json` (sha256
ef2fdfc3ff8718657ff2dcb2f2c151f177a54bdf9224a0fea25bbbeadf99ba15), developed on episode 11 only, label-blind (mocap
read only after the freeze): gravity-constrained scan-to-stereo registration, both up signs x 5-degree yaw starts,
trimmed ICP, decline if another basin > 30 degrees away is within 3 % of the best. On episode 11 it declines both cups
(flat multi-basin curves, rival basins 0.8-1.6 % worse). It is now run once, unchanged, on 12/18/21/23; a declined
slot fails the b test as the protocol says.

## Outcome (2026-11-03 08:20 UTC, context 16): Stage B FAILS; IDEAS 46 closed under this protocol

HR1 ran once, unchanged (hash checked), on 12/18/21/23; registrations and declines were saved
(`heldout_registrations.json`, sha256 e402f605...) before mocap was read. b direction < 30 degrees on 0 of 4 episodes
and C < 30 degrees on 0 of 4: blue_cup declined on 12 and 21 (rival basins within 0.56-2.16 %), the rack, pitcher and
both cups declined, mini_sweeper and both wooden pieces "cannot represent" (no `<name>.glb`; only `_visual`/`_collision`
files), mini_dust_pan accepted but 79 degrees off in up axis. Before declines, episode 12 would have been 103 / 129
degrees off (b / C) and episode 21 +11 / 168, so declining was protective. Even choosing the best of the 72 x 72 yaw
starts with the truth known could not meet both tests on 12 or 21 (best b + C 60.8 and 136.9 degrees), because the
frozen {+z, -z} up set cannot represent the rack (+x up), dust pan or tilted pitcher. Reading (narrow, after the
critics): this shows that HR1 as frozen fails, and that for the cups the stereo shape gives the up axis but not the
heading (flat multi-basin yaw curves; 1 cm markers under ~3 mm stereo noise). It does NOT show that geometry cannot
recover heading for asymmetric objects (rack, dust pan, pitcher, sweeper, wooden pieces): HR1 could not represent
their up axes or load three of their scans, so for them the question is untested. Either route (geometry with a full
up-axis search, or RGB marker detection for the cups) is a new protocol, not a repair of this one.

**Disclosures added after the critics.** (1) HR1's "label-blind" means its developer read no mocap before the freeze;
the context as a whole had already read all train mocap in Stage A, and the {+z, -z} up set was frozen after Stage A's
up-axis table (train labels, including 12/18/21/23) showed the rack, dust pan, pot and wooden pieces to be +x-up. That
knowledge biased HR1 towards failing on those objects, not towards passing; the HR1 verdict stands (C fails on 12 and
21 even before declines, 128.8 and 168.4 degrees), but it is not evidence about those objects. (2) The loader's
`<name>.glb` assumption failing on three scans was found after the freeze. (3) O1 matches the angle of b's body-frame
x/y components; that is a horizontal direction only when slot 0's body frame is z-up or z-down. On episode 23 (slot 0
+x-up) it matches the wrong angle; on 0 and 26 it is a no-op. So the "direction +0.0095 / relative rotation +0.0665"
split depends on this convention; the Stage A decision (+0.0760 vs +0.010) does not. (4) Amendment 1's heading time
was corrected to its commit time (08:09 UTC). Episodes 12/18/21/23 now carry spent HR1 errors: a new heading protocol
must name its own evaluation set before it is built. Evidence:
`outputs/pose_insurance/heading/summary.md` (recorded Stage A and U1, Codex) and
`outputs/pose_insurance/heading/scratch-evidence/` (scratch Stage A/U1 and Stage B, Claude sub-agents).
