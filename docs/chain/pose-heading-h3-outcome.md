# H3 outcome (context 18, 13:15 UTC 3 Nov 2026): all required gates pass; D2 v5 waits on the owner line

Kept apart from `docs/chain/pose-heading-h3-protocol.md` (committed `dc02fc1`, 12:57:59 UTC, before any H3 number).
Episode-19 registration ran 13:04:44-13:14:08 UTC and was saved before train 18 was scored (from 13:15:05). Evidence:
`outputs/pose_insurance/heading-h3/summary.md`, `verification.json`; `outputs/pose_insurance/registration-h3/`.

**Train 18 (only episode changed; other seven keep T1):**

| Row | Ep 18 AUC | Ep 18 RPE cm | Ep 18 MPPE cm | 8-ep AUC | 8-ep RPE | 8-ep MPPE | b err deg | C err deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 | 0.3006 | 29.58 | 19.48 | 0.2767 | 29.01 | 17.68 | 89.7 | 117.8 |
| H3-sel (TW) | 0.4729 | 10.22 | 5.56 | 0.2982 | 26.59 | 15.94 | 9.3 | 5.6 |
| H3-flip (sweeper rival) | 0.3006 | 41.71 | 25.16 | 0.2767 | 30.53 | 18.39 | 140.3 | 176.9 |
| A (shortest-arc, seen) | 0.6161 | 9.58 | 4.13 | 0.3161 | 26.51 | 15.76 | 4.2 | 5.6 |
| B (yaw-only, seen) | 0.5582 | 9.69 | 4.81 | 0.3089 | 26.53 | 15.84 | 5.5 | 5.6 |

TW scores below A and B on episode 18: T1's stereo offset direction seen from the registered slot-0 orientation is
9.3 degrees off the mocap direction, while A imposes the registered direction (4.2). TW was fixed by argument before
any number, so it stands; A and B are descriptive only.

**Gates:** G1 PASS (mean AUC +0.0215, RPE -2.42 cm, MPPE -1.74 cm, SP-SR +0.125, no episode worse). G2 required and
PASS, narrowly: selected and flip at equal weight give mean AUC +0.01077 (threshold +0.010, margin 0.00077), RPE
-0.45 cm, MPPE -0.52 cm; the flip lowers episode-18 AUC by 0 (it returns to T1's) but raises RPE/MPPE. G3 PASS: both
episode-19 masks (870 / 3,212 stereo pixels) and scans admitted, stability shifts <= 0.39 degrees.

**Episode 19 (label-free):** sweeper selected / rival losses 1.3858 / 1.4335 mm, rival 3.44 % worse and 179.98 degrees
away (HR2 would decline it; this is why G2 was required); dust pan rival 76 % worse (accepted). TW changes the episode-19
relative rotation by 118.2 degrees and the body-frame offset by 32.5 cm from D2 v4 (world positions unchanged).

**Reading (critic-18 corrections applied):** G2's equal weighting is the protocol's stipulated stress scenario, not a
calibrated probability; the break-even selected-basin weight for the AUC gate is 46.4 % (0.010 / 0.021534), conditional
on train outcomes transferring. The flip's zero episode-18 AUC loss was guaranteed in advance: T1's slot 1 on episode 18
is already beyond the AUC thresholds (AUC floor), so the flip arm could not lose AUC there; on episode 19, where D2 v4's
state relative to that floor is unknown, a wrong basin could lower AUC. G1 carried little new information (A and B had
already cleared it; C error is construction-invariant). All of this rests on one train pair (episode 18) and one
sampling seed. No extrinsic was needed in TW: route-h3-18 and review-h3-18 both found that HR2 and the T1 build work
in ego camera A (stereo transferred by `A_T_C`), with no code path for another camera. Checks: lead re-ran the independent raw-NPZ/raw-mocap recomputation (b 9.3229, C 5.6309 deg, agrees);
A reproduces H2S-desc's five metrics exactly; G2 arithmetic re-derived by the lead.

**Decision state:** D2 v5 needs the owner line "2026-11-03: approve D2 v5 heading for episode 19" in `AGENTS.md` by
about 06:00 UTC 4 Nov; none is present at 13:20 UTC 3 Nov. Without it, D2 v4 stands.
