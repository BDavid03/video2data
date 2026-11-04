# Refresh 20261104T2048Z: outcome (written 20:59 UTC, 4 Nov 2026)

Protocol `protocol.md` (SHA-256 in `protocol.sha256`) was sealed at 20:48 UTC before any registration ran.

## Dataset and kit
Re-downloaded `track_3` resolves to the pinned commit 5f68335f; all 169 files unchanged; the installed kit matches its
manifest and the organisers' online archive (`data/dataset-refresh/20261104T203500Z/diff.md`).

## Train/tuning frame-zero registrations (labels used only to validate)
Joint box assignment (highest total Grounding DINO score among box combinations with pairwise IoU < 0.5), SAM2.1 masks,
SGBM stereo depth, FoundationPose registration of both public scans, B-in-A relative pose compared with the label at frame 0.

| Episode | Objects | Masks | B-in-A error vs label | Valid (<= 2 cm, 10 deg) | min IoU | max depth m | B margin |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 11 | blue_cup, beige_cup | distinct (box IoU 0) | 0.56 cm, 2.0 deg | yes | 0.924 | 0.0109 | 0.0035 |
| 12 | blue_cup, plastic_dish_rack | distinct | 5.19 cm, 14.7 deg | no | 0.896 | 0.0096 | 0.0014 |
| 21 | blue_cup, water_pitcher | distinct | 1.47 cm, 8.4 deg | yes | 0.291 | 0.487 | 0.0050 |
| 23 | wooden_piece_1, wooden_piece_2 | distinct | 55.8 cm, 179.6 deg (slots swapped) | no | 0.895 | 0.0035 | 0.0004 |
| 26 | plastic_dish_rack, white_pot | distinct | 0.86 cm, 3.6 deg | yes | 0.630 | 0.0109 | 0.0042 |
| 0 | white_pot, white_pot_lid | no box pair with IoU < 0.5 | not registered | - | - | - | - |
| 18 | mini_sweeper, mini_dust_pan | no box pair with IoU < 0.5 | not registered | - | - | - | - |
| 31 | mini_sweeper, plastic_dish_rack | no box pair with IoU < 0.5 | not registered | - | - | - | - |

Episode 11 is the case that failed before (independent per-prompt boxes gave both cups the same mask); the joint rule
separates them and the registration is valid. Episode 23 registers both pieces but with the slots swapped: the detector
cannot tell piece 1 from piece 2. Nested objects (lid on pot, brush in dustpan or rack) defeat the box-overlap rule.

## Limits by the sealed rule (`limits.json`)
mask IoU >= 0.9236, depth residual <= 0.4868 m, B rival margin >= 0.003542. With five pairs the IoU limit lands on the
single strictest valid pair because the best-separating threshold ties and the rule breaks ties toward the stricter
value; the valid pairs 21 and 26 (IoU 0.29 and 0.63) show that silhouette IoU does not predict a valid relative pose.

## Applied once to the saved scored registrations (`decisions.json`)
No scored episode passes (8, 10, 14, 19, 20, 25, 30 all fail the IoU limit; 10 and 19 also the margin). Under the
protocol nothing is adopted and no limit is moved after this result. No policy was trained in this refresh and no file
was packed; moving v5 (b476eb1c) stays the counted file. The other packed candidate of the evening
(`fp-candidate-20261104T203551.465560Z`, episode 25 adopted under limits relaxed after results, disclosed in the report)
is unchanged by this work.

## Not run (time)
Full-clip masks, stereo depth and camera for all frames, mesh rebuild, train-episode RL rollouts, and kit-score tables of
held frame-zero poses.
