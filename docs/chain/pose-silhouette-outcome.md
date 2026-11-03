# S1 outcome (context 19, 14:08 UTC 3 Nov 2026): FAIL by rule; episode-19 application withheld

Kept apart from `docs/chain/pose-silhouette-protocol.md` (committed `3de737b`, 13:56:13 UTC, before any silhouette
number). Predictions for all 13 objects finished 14:08:18 UTC; label assessment 14:08:49-14:08:50 UTC. Evidence:
`outputs/pose_insurance/silhouette/report.md`, figure `.../predictions/20261103T140806.126752Z-061e5fe8/candidate-outlines.png`.

| Case | Result | Detail |
| --- | --- | --- |
| 18 sweeper (known: depth-best correct) | abstain, so FAIL | the correct basin wins every search and mask variant, but the minimum IoU lead is 0.0248 < 0.05 (search 0: 0.7923 vs 0.7641; C errors 5.63 vs 176.88 deg) |
| 23/24 wooden piece 1 | not testable | no upright candidate in the near-tie set |
| 23/24 wooden piece 2 | abstain (both) | no candidate leads by the margin |
| Controls (7 accepted objects) | 2/7 complete, no veto | 39 pan and 27 pot keep the depth-best; five lack one-to-one repeated matches |
| 19 sweeper | withheld | prerequisites failed; its own prediction also abstains |

Reading: the outline leans the right way on the one known sweeper flip, but by half the margin fixed in advance, so it
is not evidence enough to break the 3.44 % tie on episode 19. The risk the owner weighs for D2 v5 is unchanged (break-even
46.4 %). Checks: an independent script (no `_silhouette.py` import) recomputed all 12 sweeper IoUs exactly; 221
protected files, including 46 manifest pins, unchanged. The masks are depth-derived partial regions, the likeliest
reason the outline margin is small.
