# Pose heading protocol H3: placing HR2 registrations in the scene, flip cost, scored episode 19 (IDEAS 50; written before any H3 number; commit time is the record)

## Why

HR2 (`docs/chain/pose-heading-hr2-protocol.md`) failed its own decline rule, but its selected rotations were accurate on
the asymmetric train pairs (`docs/chain/pose-heading-hr2-outcome.md`). The owner has been asked whether to adopt them for
scored episode 19 only (sweeper / dust pan; `docs/chain/NEEDS-YOU.md`). Two things must be fixed or measured before any
such adoption: how a registered rotation is placed in the T1 scene, and what it costs when the method picks the
sweeper's near-tied rival basin. This protocol fixes both, and registers episode 19, before any of its numbers exist.
It decides nothing on its own: D2 v5 is built only if `AGENTS.md` carries the owner line
"2026-11-03: approve D2 v5 heading for episode 19" and every gate below passes by 06:00 UTC on 4 November 2026.

## Disclosure (what is already known)

- HR2 selected rotations on train 18 were scored post hoc (H2S-desc) under two constructions: shortest-arc onto T1's
  offset direction ("A", recorded): episode 18 AUC 0.6161; oracle-style yaw-only on T1 ("B", Claude critic 16b):
  0.5582. T1 episode 18 AUC is 0.3006 (eight-episode mean 0.2767). Train 18 labels, HR2's frame-0 errors (sweeper /
  dust pan b 4.2 deg, C 5.6 deg) and the sweeper rival (179.0 deg away, 2.06 % worse) are known.
- The construction chosen below ("TW") has never been scored on any episode. Its episode-18 number is unknown at the
  commit of this file.
- Episode 19 has no public labels; it has been used only by the D2 v4 scored stage (T1 placement, physical check).
  No HR2 registration of episode 19 exists.

## Construction TW (fixed now, by a physical argument)

The scorer aligns only slot 0's first scored pose with one rigid transform, so for static objects what is scored is
the relative pose of slot 1 in slot 0's body frame: C = R0^T R1 and b = R0^T (p1 - p0). Each input measures one thing:

- the calibrated-stereo centroids of T1 (D2 v4, passed its train gates) measure where each object is in the table
  frame;
- the HR2 registration measures how each object's scan is oriented relative to the camera (its translation is a
  scan-origin estimate from a trimmed fit and was never gated).

TW keeps each measurement in its own role: R_i = B R_i^cam for both slots, where R_i^cam is the registered
(search 0, best basin) scan-to-camera rotation and B is the T1 build's own camera-to-table basis
(`table_basis` of the build's recorded plane normal, with any recorded extrinsic between the registration camera and
the build camera applied first); positions are T1's, unchanged. Then C equals the registered relative rotation exactly
and b is T1's measured world offset seen from the registered slot-0 orientation.

The two constructions already seen each discard a measurement: A rotates both objects so the registered offset
direction lands on T1's, which overrides the measured orientation in the table frame (including the measured tilt);
B keeps T1's yaw-only box frame for slot 0, which imposes body z = table up on scans whose up axis is not z (the dust
pan's is +x) and discards slot 0's measured tilt. TW is therefore the construction; A and B are reported beside it as
disclosed, already-seen alternatives and are never selected.

## Measurements on train episode 18 (labels allowed; only episode 18 changes; other seven episodes keep T1)

- **H3-sel:** both slots' selected rotations under TW.
- **H3-flip:** each slot that has a recorded rival basin within HR2's 5 % margin and more than 30 deg away (the sweeper;
  the dust pan if it has one) replaced by that rival, one combination per such rival; others selected.
- For each: the five kinematic metrics on episode 18 and on the eight-episode mean, scored exactly as H2S-desc was
  (same T1 builds, same scorer path), plus frame-0 b and C errors against mocap. A and B are recomputed in the same
  run as descriptive rows.

## Gates (applied once, unchanged)

- **G1 (selected):** H3-sel versus T1 on the eight-episode mean: AUC gain >= +0.010; RPE and MPPE not worse; SP-SR and
  MP-SR not lower; worst single-episode AUC loss <= 0.02.
- **G2 (flip risk):** with the selected basin and each H3-flip combination weighted equally, the expected
  eight-episode mean AUC gain is >= +0.010 and expected RPE and MPPE are not worse than T1; and no H3-flip combination
  lowers episode 18's AUC by more than 0.02 below T1.
- **G3 (episode 19, label-free):** the frozen HR2 method (`docs/chain/pose-heading-method-hr2.json`, sha256
  c1fd8677...; the frozen `_registration.py` functions, unchanged) registers both episode-19 objects at frame 0 with the
  inputs the D2 v4 scored stage used. Each slot must have a mask, a loadable scan and subsampling stability <= 10 deg.
  G2 is required only if a slot of episode 19 has a rival basin within 5 % and more than 30 deg away; if neither slot
  has one, G1 and G3 suffice.

If any required gate fails, D2 v4 stands and the result is recorded. If all pass and the owner line is present, D2 v5
is D2 v4 with only episode 19 changed to TW rotations (box/collision geometry and inertia counter-rotated so the
physical placement is unchanged; physical check first pose <= 1 cm and <= 2 deg), packed with preflight, ledger and
upload command by 12:00 UTC on 4 November.

## Episode 19 registration (no labels exist; recorded separately)

Its own queue and roster file (the frozen HR2 evaluation roster is not edited), run through the recorded registration
operation, with the episode-19 frame-0 rotations, basins, margins and stability saved before any construction is
applied. Reported: selected and rival basins per slot, relative margin, separation, stability, the TW relative pose
(C, b) and its change from D2 v4's episode-19 relative pose.

Episodes 41, 43, 44 are never read; validation 2/13/22/24/27 are not used; no leaderboard use.
