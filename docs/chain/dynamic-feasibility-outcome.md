# D0 outcome (context 19, 14:30 UTC 3 Nov 2026): NO-GO by rule

Protocol `docs/chain/dynamic-feasibility-protocol.md` (committed `c287d5e`, 14:12:17 UTC, before any run). Evidence:
`track3/features/dynamic_feasibility/README.md`, `outputs/dynamic_feasibility/` (ledger, summaries, `audit.json`).

| Arm | Seed | Mean pos cm | P90 pos cm | Mean ori deg | ADD-AUC | Completion | Wall s | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| S (object held still, computed) | - | 6.889 | 16.549 | 2.217 | 0.4591 | - | - | - |
| Z (zero updates) | 0 | 7.012 | 16.266 | 3.323 | 0.4280 | 390/390 finite | 119 | 3183 |
| L1 (64 worlds, 128k transitions) | 0 | 7.026 | 16.396 | 3.415 | 0.4276 | 390/390 finite | 77 | 2915 |
| L1 | 1 | 6.954 | 16.352 | 2.150 | 0.4471 | 390/390 finite | 76 | 2915 |

- GO needed both learned seeds <= 3 cm and <= 0.5 x S; neither is close. L2 was skipped by its gate (best L1 seed improved
  on Z by 0.83 % < 20 %). Total GPU 4.5 of the 75 allowed minutes. No deviation, no setting changed after launch.
- Z: the hand reference alone moved the box at most 1.5 cm against the reference's 17.4 cm. The hands following their
  reference do not carry the object; transport has to be learned.
- L1 seed 1 comes closest to S because it disturbs the box least (0.72 cm of motion against Z's 1.50 cm), not because
  it transports it better (critic-19). An "improvement over Z" gate therefore also rewards staying still; a later
  test should gate on transport (displacement along the reference), not only on error.
- What D0 does NOT show: L1 is about 1/2000 of the released recipe (237 learner updates), so D0 rules out the cheapest
  budget only. Whether a larger budget that fits 16 GB learns transport is a separate question (D1, its own protocol).
- Checks: an independent script reproduced position errors and ADD-AUC exactly (orientation within 2.1e-9 deg); 85
  input hashes matched; resume appended nothing.
