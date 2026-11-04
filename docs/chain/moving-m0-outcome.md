# M0 outcome (4 Nov 2026, about 02:09 UTC): NULL

Protocol `docs/chain/moving-m0-protocol.md`, committed `1ec1e90` before any transport number existed. Evidence: `outputs/dynamic_feasibility/m0/` (`summary.md`, `final.json`, `audit.json`, `jump_hold_controls.json`).

D1's two partial policies do not carry the box. About 22M transitions on a hand reference earned no credit on this screen, with training assistance still at VOC 0.05 and the evaluator's assistance removed.

| Arm | T | Q | Perpendicular cm | Mean position cm | ADD-AUC | Max displacement cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S (held still) | 0 | 0 | 0 | 6.889 | 0.4591 | 0 |
| Z (zero updates) | 0.0070 | 0.0220 | 0.594 | 7.012 | 0.4280 | 1.50 |
| L1 seed 0 | 0.0052 | 0.0128 | 0.665 | 7.026 | 0.4276 | 1.62 |
| L1 seed 1 | 0.0005 | 0.0080 | 0.305 | 6.954 | 0.4471 | 0.72 |
| D1p seed 0 (21,580,800) | 0.0352 | 0.0751 | 0.478 | 6.803 | 0.4438 | 2.03 |
| D1p seed 1 (22,604,800) | 0.0094 | 0.0091 | 0.399 | 6.828 | 0.4527 | 0.78 |

The reference travels up to 17.38 cm and ends 3.18 cm from its start. SIGNAL needed `T >= 0.20`, `Q >= 0.10` and error at most S's on one seed; the best is `T` 0.035 and `Q` 0.075. D0's old rule also says NO-GO.

- **Validity:** all five checks passed, including exact reproduction of Z, L1 and S; no jump-and-hold control met GO or SIGNAL (the largest `T` among them, 1.37, came with `Q` -0.20, so `Q` did its job). Both exports admitted; no retry; 42 GPU-seconds, sampled peak 1,883 MiB.
- **Independent check:** the lead's separate script recomputed `T`, `Q`, position error and displacement from the raw parquets, and the checkpoint hashes (`f9750297...`, `1cf3b6a1...`); all agree. The protocol required this only for a positive result; it was done anyway.

**Reading.** M0 does not show that a longer curriculum cannot learn this, or close W3: the protocol says NULL means this screen was not met. D1p seed 0 moves the box a little further than Z and has the best `Q`, but one seed at 0.075 is not a signal, and the differences among all arms are within what disturbing a resting box produces. The finding is narrower and still useful: **stopping the curriculum at VOC 0.05 and about 22M transitions yields no transport once assistance is removed**, so a compressed schedule that does not finish cannot be expected to help.

**Consequence, by the protocol's table.** M2 still tests the object-driven reference. D1p counts as M2's hand arm only if M2's object arm runs D1's exact schedule, seeds and transitions; otherwise M2 runs both arms, or says it cannot compare them. A completed unassisted schedule costs about 2.6 h per episode at best, so M2 must choose between a matched-to-D1 test at about 30 minutes per seed and a real completed curriculum. That goes in M2's protocol.
