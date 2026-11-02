# Pose scale protocol (IDEAS 27; fixed 2026-11-02 00:05 UTC, context 05, before any stereo scale on a two-object episode was computed or any scale candidate was scored)

D2 v2 builds every scene at MoGe-2 metric size. Two measurements suggest that size is inflated:
on the clean two-object train episodes the video inter-object distance is 1.37-1.80 times the
mocap distance (`docs/chain/IDEAS.md` row 27), and on tuning episode 39 the existing
`stereo_depth.anchor` run gives stereo/MoGe-2 depth ratios of about 0.76-0.78 per frame
(`outputs/stereo_depth/public/episode_000039/20261101T095021.311860Z-79e8971a/anchor.json`).
The kit removes one rigid transform per world and no scale, so placement scale stays scored.
This protocol fixes, before scoring, how a stereo-anchored scale is tested and when it is
adopted.

Known ceiling (diagnostic, already recorded): replacing slot 1's offset length by the mocap
length lifts the eight-episode mean AUC only from 0.268844 to 0.276705
(`outputs/pose_insurance/placement/summary.md`, "Diagnostic ceilings"), because slot 1's
direction error (2-130 degrees) and relative rotation (88-179 degrees) remain. The expected
effect of this protocol is therefore on RPE and MPPE; AUC, SP-SR and MP-SR are guarded, not
expected to rise.

## Scale estimate (one estimator, no tuning)

- Per episode, run the existing `stereo_depth` operations `rectify`, `generate` and `anchor`
  unchanged (camera C left, B right, calibrated 75.47 mm baseline, SGBM, median of
  stereo/monocular over pixels valid in both) against the MoGe-2 depth run that the D2 scene of
  that episode already uses, on that run's frames.
- The episode scale `s` is the median of the per-frame `scale` values over frames with
  `scale_status == "observed"`, as the slice already reports it.
- **Validity filter, decided now and reading no reference label or score:** `s` is valid when at
  least 3 frames are observed, median `support_share` >= 0.10, median `log_ratio_mad` <= 0.30 and
  0.40 <= `s` <= 1.20. An invalid `s` means the episode keeps its D2 v2 scene.

## Candidates (two, fixed now)

The incumbent is the D2 v2 construction rule applied to each train episode's scored analogue:
K1a on train 0 and 26 (as scored 1 and 28), v1 placement on the others (train 31 keeps v1, as
scored 30).

- **S1, offset scale.** Slot 1's initial table-plane position becomes
  `p0 + s * (p1 - p0)`, where `p0`, `p1` are the incumbent table-plane positions; heights,
  rotations, object sizes and slot 0 are unchanged. On K1a-stacked episodes `p1 = p0` and S1
  changes nothing.
- **S2, scene scale.** The whole incumbent scene is scaled by `s` about slot 0's table-plane
  position: table-plane offsets, heights above the table and the D1 box dimensions (mesh,
  collision box, mass properties consistent with the slice's density rule). Rotations unchanged.

No mocap pose, scan geometry or scan dimension enters a scene. Scans enter scoring only (1,000
surface samples per object, seed 0, as the placement diagnostic).

## Scoring

The kinematic static diagnostic of `docs/chain/pose-placement-protocol.md` (Amendment 1 slot
order, nearest-visible fill, the kit's own AUC, SP-SR, MP-SR, RPE and MPPE after
`_warp_to_reference_start`, one world), on the eight train two-object episodes 0, 11, 12, 18,
21, 23, 26, 31, equal-episode means. Tuning 39 and 42 (one object) are run as invariance
controls: their kinematic metrics must be identical to the incumbent's for both candidates
(a static single object is unaffected by scale); any change is a defect, not a result.
Episodes 41, 43 and 44 are never read.

## Adoption

A candidate passes when, against the incumbent over the eight episodes:

1. mean RPE falls by at least 2 cm and mean MPPE by at least 1 cm;
2. mean AUC, SP-SR and MP-SR do not fall (1e-12 tolerance);
3. no single episode loses more than 0.02 AUC or gains more than 0.5 cm MPPE.

Episodes whose `s` is invalid enter the means with the incumbent scene. If both candidates pass,
S1 is adopted unless S2's mean MPPE is lower by at least 1 cm (S1 changes no physics). If neither
passes, D2 v2 stays and the result is recorded in `IDEAS.md` as a finding.

## Scored episodes and the physical check (fixed now)

For an adopted candidate, `s` is computed for every scored two-object episode with the same
estimator and validity filter. Each changed scene runs through the real evaluator (the slice's
existing budget, one world) and must pass the slice's existing check: first exported pose
within 1 cm / 2 degrees of the intended initial pose and drift below 1 cm / 2 degrees. **Per
episode, a scene that has an invalid `s` or fails the physical check keeps its D2 v2 export.**
This fallback reads no score or reference label. The repack is then a per-episode hybrid; the
report lists which episodes changed, which fell back and why. A repack happens only if at least
one scored episode changes; its manifest pins this file, so this file is not edited afterwards.

## Disclosure

The report says: the scale idea came from mocap distance ratios on train episodes; selection used
train episodes, because tuning 39/42 have one object and cannot measure placement (the chain's
recorded disagreement with `PROGRAM.md` section 4); the AUC ceiling above; and any per-episode
fallback.
