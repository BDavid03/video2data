# M1 outcome (4 Nov 2026, 09:50 UTC): a scan-free video path reaches AUC 0.833 on tuning 39, by a route with amended gates

Protocol `moving-m1-protocol.md`; sealing record `moving-m1-sealed.md`. Evidence: `data/outputs/moving_reference/`. The tracker reads no scan. The body convention reads the public scan's axes once, disclosed. **This is the reference alone: a path, not a rollout, and not a challenge result.**

## The sealed comparison (run once, 09:17 UTC, after hashes were committed)

| Measure against mocap on 39 | Value | Limit |
| --- | ---: | ---: |
| Kit AUC | **0.8331** | at least 0.60 |
| Mean position error, after the scorer's single first-pose alignment | 2.16 cm (p95 4.50, max 5.23) | at most 2.5 cm |
| Mean rotation error | 8.57 deg (p95 14.0, max 17.2) | at most 12 deg |
| SP-SR, MPPE | 1.0, 2.26 cm | |
| Allowance: M2's measured policy error composed on it | AUC 0.8005 | at least 0.45 |

Beside it: held still on 39, 0.18; shipped T1, 0.175; a perfect position with held rotation, 0.413. **Lead's check:** a separate script with other surface points and its own alignment reproduces the kit's 0.833 under trapezoidal integration. Motion correlates with mocap at 0.83, 0.996 and 0.997 per axis; travel 2.06 m against 1.92 m; net rotation 114 against 109 degrees. Note a mean position error near its limit: 2.16 against 2.5.

## How it got there: read this before trusting 38

| Step | Result | What happened |
| --- | --- | --- |
| Candidate A, as frozen | **FAIL**, 11 of 23 | ORB camera lost coverage; features lost once lifted; placement defect |
| Candidate B, FoundationPose | **FAIL** | lost on 39 (102 deg forward-backward), jumps on both, noncommercial licence. Closed |
| Candidate C, method revision, mocap sealed | **FAIL** the frozen rule, 16 and 17 of 23 | bundle adjustment plus silhouette and stereo fit, refit mesh (frame-0 IoU 0.50 to 0.91) |
| Amendment 1: thresholds moved, instruments changed, stricter drift checks added | **FAIL**, 22 and 24 of 28 | noise floor did not explain the world residual; table check looked wrong |
| Body constant check, from critic 27 | **a real bug fixed** | 12.34 deg of yaw and an origin offset, before the sealed pass |
| Sealing record, conditional rule, breaking "only one amendment" | written before any label | see `moving-m1-sealed.md` |
| Sealed pass | **PASS** | above |
| Table check's defect: its region held the lap and table edges, and its fit was faulty | fixed, limits unchanged | then coverage fell to 24 and 33 % |
| Fresh withheld table tracks, one attempt | **PASS** on both, limits unchanged | below |

**So M1 does not pass its first rule, or its second. It passes a third, written after seeing the gate values on both clips and before reading a label, with one direct truth on 39.** Reasonable readers will discount it. What guards 38 is below; what does not is the reprojection, since camera error cancels there (critic 27), and IoU is close to C's own objective.

## 38's evidence (no labels exist, none were read)

| Rule (final) | Limit | 39 | 38 |
| --- | ---: | ---: | ---: |
| Held-out median, moved from 1/3 px | 0.5 px | 0.359 | 0.312 |
| Held-out p95, moved from 1 px | 2 px | 1.641 | 1.421 |
| Patch normal, moved from 2 deg | 3 deg | 2.286 | 2.634 |
| World residual, perpendicular instrument (old 3D: 25.4, 19.5 mm, kept on record) | 10 mm | 2.35 | 1.94 |
| Table offset, max of frame medians | 10 mm | 4.82 | 3.18 |
| Late table normal change | 2 deg | 0.204 | 0.494 |
| Landmark in-plane displacement, max of frame medians | 10 mm | 8.01 | 8.19 |
| Landmark p95 of medians | 20 mm | 5.20 | 5.94 |
| Table and landmark coverage | 80 % | 98.75 % | 100 % |
| Mesh IoU median, p10 | 0.60, 0.30 | 0.859, 0.827 | 0.864, 0.810 |
| Forward-backward | 1 cm, 5 deg | 0.70 mm, 0.70 deg | 0.42 mm, 0.41 deg |

**Limits on what this shows.** Passing these does not measure 38's path against truth; there is none. One seed-free label pass on another clip of the same object says less about 38's own motion. 39's reference has a body-convention constant from the scan, and an unresolved 2 cm ring offset in the overlay. M1's `reference.npz` hash is what M2v and the 38 run train on.

## Consequence

By the sealing record, clause 2: **M2v runs for adoption**, and M1's rule passes on 38. M2v launched 09:28 UTC. Outcome in `moving-m2v-outcome.md`.
