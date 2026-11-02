# Pose stereo-placement protocol: slot 1 placed from direct calibrated stereo (IDEAS 44/45; fixed 2026-11-02 UTC, context 12, before any candidate below was built or scored)

## Why, and on whose authority

D2 v3 places slot 1 at the S1 offset: the MoGe-2 table-plane offset from slot 0 times the episode's
stereo/MoGe-2 scale `s` (`docs/chain/pose-scale-protocol.md`). The length diagnostic
(`docs/chain/pose-length-diagnostic-protocol.md`, `outputs/pose_insurance/length/summary.md`) found
that the same offset measured directly from calibrated ego_cam_b/c stereo points (V_stereo) has a
horizontal length error of 0.58 cm against the mocap origins on the clean five train episodes
(S1: 11.75 cm, lower on 5/5). Its fresh-episode confirmation (Amendment 1) did not pass by the
letter, because counted episode 13 is one merged cluster; on the three measurable fresh episodes
22/24/27 V_stereo was 1.14 cm off vs 7.21 cm for the MoGe-2 x `s` stand-in, lower on 3/3. The
length protocol therefore said no candidate follows. **The owner overrode that on 2026-11-03
("needs-you now approved", `AGENTS.md`, answering `NEEDS-YOU.md` item 7).** This protocol is the
separately pre-registered candidate that item asked for.

**What was already known when this was written (disclosure).** The committed length summary
contains, for the clean five, the slot-1 frame-0 horizontal error after the kit's slot-0 rigid
alignment. The lead read it before writing this file: V_stereo 15.1 / 28.1 / 32.8 / 59.3 / 0.8 cm
on 11 / 12 / 18 / 21 / 23 against S1 16.4 / 54.6 / 42.4 / 69.1 / 8.2 cm, with direction errors of
-51 / -98 / +95 / -130 / -2 degrees. So the train gates below are not blind: frame-0 errors predict
the trajectory metrics well for static objects. The gates check that the full-trajectory kit
metrics agree and that nothing regresses; they are not an independent test of the idea. The only
fresh evidence is the descriptive 3/3 on 22/24/27, and the report says so.

**Expected effect.** The gain is expected on RPE and MPPE. Slot 1's direction after alignment is
wrong by 50-130 degrees on four of the five clean episodes, so AUC, SP-SR and MP-SR are guarded,
not expected to rise (episode 23, where the direction happens to be right, is the exception that
may move AUC).

## Candidate T1 (fixed now)

The incumbent on train is S1, rebuilt exactly as `scale_geometry(geometry, 'S1', s, valid)` from
the recorded scenes (it must reproduce the recorded S1 positions within 1e-6 m, as the length
diagnostic already checks).

T1 changes one thing: slot 1's table-plane position becomes `p1 = p0 + d_T`, where `p0` is slot 0's
S1 table-plane position (unchanged) and `d_T` is the V_stereo offset of the length diagnostic,
computed by the same code and definitions:

- the observed slot masks of the recorded frame-0 clusters (train: `placements_train.json`
  observed cluster ids; scored episodes: the cluster ids the D2 v3 scene of that episode used);
- frame-0 calibrated stereo points (`stereo_depth.generate`, on the MoGe-2 pixel grid of the
  depth run the scene used), valid pixels only, at least 300 valid pixels per object;
- each object's point mean projected into the table basis of that run's measured plane normal;
  `d_T` = slot-1 centre minus slot-0 centre.

Heights, rotations, object sizes, masses and slot 0 are unchanged. **Clearance:** if the two D1
boxes' oriented table-plane footprints (from the scene's dimensions and yaw) are not at least
1 cm apart at `d_T`, the offset becomes `c * d_T` with the smallest `c` in [1, L_S1 / |d_T|] that
gives 1 cm clearance (separating-axis test on the rectangles' unit edge normals, bisection to a
factor width of 1e-4, feasible upper endpoint), where L_S1 is the S1 offset length. If no such `c`
exists, the episode keeps S1. `c` is recorded before scoring.

**Fallback to the incumbent (S1 on train, D2 v3 on scored episodes), fixed now, reading no
reference label or score:** a merged cluster, a slot without its own observed cluster (default
placement), fewer than 300 valid stereo pixels on either object, no frame-0 stereo run, or no
clearance as above. On train this means 0 and 26 (merged) and 31 (default slot) keep S1 by rule;
12 uses its recorded observed slot-0 mask (the S1 scene uses a default slot 0 there; this is an
assignment difference as well as a depth-source change, and is reported).

No mocap pose, scan geometry or scan dimension enters any candidate scene. Mocap and scans enter
scoring and the descriptive reading only.

## Scoring (unchanged from the scale and shrink protocols)

The kinematic static diagnostic of `docs/chain/pose-placement-protocol.md` (Amendment 1 slot order,
nearest-visible fill, the kit's own AUC, SP-SR, MP-SR, RPE and MPPE after
`_warp_to_reference_start`, one world, 1,000 scan surface samples per object, seed 0), on the eight
train two-object episodes 0, 11, 12, 18, 21, 23, 26, 31, equal-episode means. Tuning 39 and 42 run
as single-object controls. Episodes 41, 43 and 44 are never read.

## Integrity controls

- **Reproduction.** The new code's S1 rows must equal the recorded S1 rows
  (`outputs/pose_insurance/scale/summary.json` or the per-trial `metrics.json`) per episode within
  1e-9 (AUC, SP-SR, MP-SR) and 1e-9 cm (RPE, MPPE). A failure blocks the verdict.
- **Offset reproduction.** Each `d_T` must equal the recorded V_stereo vector of
  `outputs/pose_insurance/length/summary.json` within 1e-9 m. A failure blocks the verdict.
- **Single-object invariance.** On 39 and 42 T1 has nothing to change; every metric must equal
  S1's within 1e-9. A failure blocks the verdict.
- A missing or non-finite metric blocks the verdict; the denominator is never reduced.

## Train gates (T1 against S1, means over the eight train episodes)

1. mean RPE falls by at least 2 cm and mean MPPE by at least 1 cm;
2. mean AUC, SP-SR and MP-SR do not fall (1e-12);
3. no episode loses more than 0.02 AUC or gains more than 0.5 cm MPPE.

Pass all three and the controls: T1 goes to the scored stage. Otherwise D2 v3 stays and the result
is recorded in `IDEAS.md` as a finding. No post-result reading of these gates is adopted.

## Descriptive direction reading (no verdict; item 2 of hand-off 11)

For every train episode and for S1 and T1: the signed angle in degrees between slot 1's aligned
predicted offset and the aligned mocap offset after the kit's slot-0 alignment (as
`_length.ray_errors` transports it), the frame-0 slot-1 aligned position error, and whether it is
inside each of the kit's AUC thresholds. Also, per episode, the error T1 would have at an exact
direction (`| |d_T| - |M_o| |`) and the angle tolerance that would put it inside 9 cm
(`2 asin(9 / (2 |M_o|))`). This states how far a correct length alone can move AUC and what a
heading cue would have to achieve. It chooses nothing.

## Scored stage and the physical check (only if the train gates pass)

- T1 is computed for the 13 scored episodes S1 changed (3, 6, 8, 10, 14, 15, 16, 19, 20, 25, 30, 32,
  33), with every fallback above. Episodes 1, 28 (stacked), 5 (v3 fallback) and single-object
  35-38 keep their D2 v3 exports. The evaluation split ships no calibration; the public stereo
  calibration is assumed to hold, as for S1.
- Each changed scene runs through the real evaluator (the slice's existing budget, one world) and
  must pass the slice's existing physical check: first exported pose within 1 cm / 2 degrees of the
  intended initial pose and drift below 1 cm / 2 degrees. Per episode, a failure or a run without a
  valid export keeps the D2 v3 export. This fallback reads no score or reference label.
- If at least one scored episode changes, the hybrid is packed as D2 v4 with
  `scripts/reproduce/d2_poses.sh` at a new commit, with a manifest pinning this file and the real
  GitHub commit URL form; the upload command goes into `NEEDS-YOU.md`. Otherwise D2 v3 stays.

## Disclosure in the report

The report says: the idea came from a train diagnostic; its fresh confirmation failed by the letter
and the owner approved the candidate anyway; the train gates were set after the frame-0 errors were
known; the expected effect is RPE/MPPE; direction stays unobservable; and which scored episodes
changed or fell back, and why.
