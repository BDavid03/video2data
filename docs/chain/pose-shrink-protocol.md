# Pose shrink protocol: calibrated scale or plain shrinkage? (IDEAS 28/30; fixed 2026-11-02, context 06, before any candidate below was scored)

D2 v3 scales slot 1's table-plane offset from slot 0 by the per-episode stereo/MoGe-2 scale `s`
(S1, `docs/chain/pose-scale-protocol.md`). On the 8 train episodes S1 lowered mean RPE
39.36 -> 35.34 cm and MPPE 22.82 -> 20.84 cm. After S1 the video/mocap distance ratio is still
1.18-2.23 (3.58 on episode 31), and slot 1's direction error is 2-130 degrees. With a wrong
direction, any shorter offset lowers the error, so S1's gain could be plain shrinkage (a hedge)
rather than a corrected metric scale. This protocol fixes, before scoring, the comparators, the
numeric tolerances, the scope of every control and what may follow from the result.

This is a **diagnostic**. Nothing here repacks D2. The only route to a repack is the separate,
pre-registered follow-up H at the end, with its own gates, physical check and fallback.

## Scoring (unchanged from the scale protocol)

The kinematic static diagnostic of `docs/chain/pose-placement-protocol.md` (Amendment 1 slot
order, nearest-visible fill, the kit's own AUC, SP-SR, MP-SR, RPE and MPPE after
`_warp_to_reference_start`, one world, 1,000 scan surface samples per object, seed 0), on the 8
train two-object episodes 0, 11, 12, 18, 21, 23, 26, 31, with equal-episode means. Tuning 39 and
42 run as single-object controls. Episodes 41, 43 and 44 are never read. No mocap pose, scan
geometry or scan dimension enters any candidate scene; mocap and scans enter scoring and the
oracle diagnostic only.

## Candidates (fixed now)

Every candidate is the D2 v2-rule incumbent scene (K1a on 0 and 26, v1 elsewhere) with slot 1's
table-plane offset `d = p1 - p0` replaced by `c * d`. Heights, rotations, sizes and slot 0 are
unchanged. On the K1a-stacked episodes 0 and 26, `d = 0` and every candidate equals the incumbent.

| Name | Factor `c` | Reads |
| --- | --- | --- |
| I (incumbent) | 1 | nothing |
| S1 | `s_e`, the episode's own stereo scale | stereo of that episode |
| G (global shrink) | `s_bar = 0.856952`, the mean of the 8 train `s` values in `outputs/pose_insurance/scale/summary.md` (recomputed from the `stereo_*.json` files, must agree to 1e-6) | stereo of all 8 |
| P1-P7 (shuffled scale) | `s` of another episode: cyclic shift k = 1..7 of the 8 train `s` values in episode order 0, 11, 12, 18, 21, 23, 26, 31 | stereo of another episode |
| C0, C25, C50, C75 (shrink curve) | 0, 0.25, 0.50, 0.75 (C0 is IDEAS 28, zero offset) | nothing |

C0 places slot 1 at slot 0's table-plane position. It is physically implausible (interpenetrating
boxes) and is scored kinematically only, as a bound.

**Oracle diagnostic (reads mocap; never a candidate):** per episode, the factor `c*` in
0.00, 0.01, ..., 1.50 that minimises that episode's MPPE, and the length ratio
`mocap distance / incumbent distance` at frame 0.

## Integrity controls (numeric tolerance and scope fixed now)

- **Reproduction.** The new code's I and S1 rows must equal the recorded rows of
  `outputs/pose_insurance/scale/summary.md` per episode within 1e-9 (AUC, SP-SR, MP-SR) and
  1e-9 cm (RPE, MPPE). A failure blocks **every** verdict (the scorer is not the recorded one).
- **Single-object invariance.** For each candidate, every kinematic metric on 39 and 42 must
  equal the incumbent's within 1e-9 (cm for RPE/MPPE). A failure disqualifies **only that
  candidate** from the verdicts and from H; it is reported, not repaired.
- **Shift sanity.** G's `s_bar` and each shift's assignment are written into the ledger before
  scoring.

## Verdicts (tolerances fixed now; means over the 8 train episodes)

Let `M(x)` and `R(x)` be candidate x's mean MPPE and mean RPE in cm.

1. **Per-episode calibration.**
   - *Calibration adds information* when `M(G) - M(S1) >= 0.25` and `R(G) - R(S1) >= 0.50`
     and `M(S1) <= M(Pk)` for at least 6 of the 7 shifts.
   - *Not separable from a global shrink* when `|M(G) - M(S1)| < 0.25` (whatever R does).
   - *Global shrink is better* when `M(S1) - M(G) >= 0.25`.
   - Anything else is reported as *mixed*, with the numbers.
2. **Hedge.** Let `c_best` be the factor among {0, 0.25, 0.50, 0.75, s_bar, 1} with the lowest
   mean MPPE (ties to the larger factor).
   - *Hedge-dominated* when `c_best <= 0.50` and `M(S1) - M(C_best) >= 1.0`: shrinking well below
     any calibrated scale keeps lowering the error, so S1's gain is mostly a hedge.
   - *Consistent with scale* when `c_best >= 0.75` or `M(S1) - M(C_best) < 1.0`.
3. **Oracle reading** (descriptive, no verdict): median and range of `c*` over the 6 changed
   episodes beside `s_e`, and the Spearman correlation of `s_e` with the length ratio.

Every verdict is reported as it comes out, with the 6 changed-episode means beside the 8-episode
means. No verdict changes D2 v3. The abstract states the verdict in one sentence.

## Pre-registered follow-up H (the only route to a repack)

H exists because the leaderboard ranks RPE and MPPE as two of the six Track 3 metrics, and
when slot 1's direction is unobservable a shorter offset is the lower-expected-error choice. H
is run only when verdict 2 is *hedge-dominated*.

- **Construction.** `c_H` is the factor among {0.25, 0.50, 0.75} (0 is excluded as physically
  implausible) with the lowest train mean MPPE, ties to the larger. Per episode with `d != 0`,
  the applied factor is `max(c_H, c_sep)`, where `c_sep` is the smallest factor at which the two
  D1 boxes' table-plane footprints (oriented rectangles from the scene's dimensions and yaw) are
  at least 1 cm apart, found by bisection to 1e-4 with a separating-axis test. Episodes with
  `d = 0` are unchanged. `c_H` and every `c_sep` are recorded before scoring.
- **Train gates, against S1** (the shipped rule): mean RPE falls by at least 2 cm and mean MPPE
  by at least 1 cm; mean AUC, SP-SR and MP-SR do not fall (1e-12); no episode loses more than
  0.02 AUC or gains more than 0.5 cm MPPE; the single-object invariance control above passes
  for H.
- **Scored stage.** If the gates pass, H is applied to the same 13 scored episodes S1 changed
  (3, 6, 8, 10, 14, 15, 16, 19, 20, 25, 30, 32, 33), with `s` and scenes otherwise as in v3.
  Each changed scene runs through the real evaluator (the slice's existing budget, one world)
  and must pass the slice's existing physical check (first exported pose within 1 cm /
  2 degrees of the intended pose, drift below 1 cm / 2 degrees). Per episode, a failure or
  a run with no valid export keeps the D2 v3 export. Episodes 1, 5, 28 and 35-38 keep v3.
  If at least one episode changes, the result is packed as D2 v4 with a manifest pinning this
  file; otherwise D2 v3 stays.
- **Disclosure.** The report says H is a deliberate shrink chosen on train episodes because
  slot 1's direction is unobservable, gives `c_H`, and lists changed and fallback episodes.

If verdict 2 is not *hedge-dominated*, H is not built and D2 v3 stays.
