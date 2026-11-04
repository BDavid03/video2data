# Frame-0 panel outcome (4 Nov 2026, about 03:05 UTC): TW and AN FAIL, P PASS

Protocol `frame0-panel-protocol.md`, Amendment 1 committed `cca187a` before any Stage C number. Evidence: `outputs/pose_insurance/frame0-panel/stage-c/` (`summary.md`, `summary.json`, `candidates.json`, `run.json`). Kinematic, train labels, eight-episode means; this is a disclosed train panel and not independent validation.

| Arm | AUC | SP-SR | RPE cm | MPPE cm | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| T1 | 0.2767 | 0 | 29.01 | 17.68 | baseline |
| P (26, 31 at `p1 = p0 + d`) | 0.2767 | 0 | 27.46 | 16.86 | **PASS** (diagnostic) |
| TW4 | 0.3250 | 0.125 | 25.81 | 15.70 | **FAIL** |
| AN4 | 0.2688 | 0 | 25.81 | 16.14 | **FAIL** |
| TW5 (descriptive; adds 18) | 0.3465 | 0.25 | 23.39 | 13.96 | not gated |
| TW4-keep, the original rule (descriptive) | 0.2830 | 0 | 32.16 | 19.09 | FAILS every clause of the original gate |

**Frame-0 errors.**

| Pair | Offset residual cm (T1) | `b` angle deg | `C` TW deg | TW limits |
| --- | ---: | ---: | ---: | --- |
| 0 | 12.98 (12.98) | 94.4 | 132.3 | FAIL, input admission |
| 26 | 2.99 (10.11) | 4.4 | 4.8 | PASS |
| 31 | 5.12 (30.05) | 26.0 | 9.1 | FAIL by 0.124 cm |
| 23 (inspected earlier) | 3.94 (1.17) | 8.1 | 13.7 | PASS |

**Why TW failed, clause by clause.**
1. Frame-0 limits on 1 of 3 (required 2): 0 failed input admission, and 31 is 0.124 cm over 5 cm.
2. AUC +0.0483 and SP-SR 0.125: passes.
3. RPE -3.21 and MPPE -1.98 cm: passes.
4. Every changed episode improves: passes.
5. Pair 23's offset error rises 2.77 cm against a 0.5 cm allowance: fails. Rotating slot 0's frame moves a T1 offset that was already good.
6. Flip stress fails: AUC +0.0085 against +0.01, MPPE +0.08 cm, and a flipped combination costs 18.4 cm MPPE and 0.063 AUC on one episode.

AN failed on clauses 1, 2, 4 and 6: a rotated anchor with no relative rotation loses 0.063 AUC on 23. Surface seeds 1 and 2 give the same verdicts. **No failure is repaired on these pairs; two of TW's misses are narrow, and that changes nothing.**

**Independent check.** The lead's separate script rebuilt P from Stage A's saved `d`, took T1 from W1.0's record, took mocap from raw parquets, and computed offset, `b` and `C` with its own formula. It agrees on every printed digit, including flip stress, and the registered relative rotations match Stage B to 3e-15.

**Reading.**
- The measured offset is real information: on both contact pairs it cuts offset error (10.1 to 3.0 cm, 30.1 to 5.1 cm) and RPE (18.7 to 15.5, 30.4 to 21.1). AUC does not move, as W1.0 predicted for position alone.
- Registered rotation plus measured offset is where AUC appears, as W1.0 also predicted: on fresh pairs, 26 goes 0.500 to 0.617 and 31 goes 0.098 to 0.317 with SP-SR 1. Of TW's three failed clauses, flip stress is the one with material size: 31's sweeper rival is within 0.117 % at 179 degrees, and a flip costs up to 18.4 cm MPPE. Clause 1 missed by 0.124 cm, and clause 5 is a 2.77 cm rise on a pair whose T1 offset was already 1.17 cm. **Our reading, not a gate result: flip ambiguity limits this route more than registration accuracy.** A later route would need a disambiguator that does not rest on the same geometric loss, and it would have to face fresh pairs, not these.
- Amendment 1 mattered and did not change the result: under the original rule TW4-keep fails on every clause it was built to test, and its offset error rises 21 cm.

**Consequence, by the protocol.** Gate TW and Gate AN close their scored rows. Gate P opened one route: episode 30 alone, with amended P positions and T1 rotations, under adoption rules 3 and 5 and the proxy rule.

**Episode 30: FAIL, v5 kept (`outputs/pose_insurance/frame0-panel/scored-30/decision.json`).** Stage A admitted it: 865 / 16,351 stereo pixels, box IoU 0.071, `d` (1.93, 1.14, -7.60) cm. Clearance of 1 cm needed a common XY factor of 0.0281, so the proxies were all but gone. One evaluator run on the existing budget gave first pose 3.60 cm against 1 cm, translation drift 16.53 cm against 1 cm, and rotation drift 90.0 degrees against 2. Nothing was retried, no scored label or score was read, and all 20 incumbent exports stayed byte-identical. A body placed 7.6 cm off its partner with almost no footprint falls over, which is what the check exists to catch.

**So the panel yields no new pack; D2 v5 stays.** Of the scored rows, 25, 28, 1, 3, 5, 6, 15 and 16 close with Gates TW and AN, and 30 closes here.
