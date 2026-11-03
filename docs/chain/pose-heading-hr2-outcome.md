# HR2 outcome (context 16, 12:32 UTC 3 Nov 2026): fails both panels; D2 v4 stands

Kept apart from `docs/chain/pose-heading-hr2-protocol.md`, whose bytes the recorded HR2 trials hash (sha256 687debcf...); the protocol is restored to its frozen commit `9135746`.

Frozen evaluation run once (`outputs/pose_insurance/registration-hr2/phase2-report.md`; predictions saved 12:30:29 UTC
before labels were read at 12:30:37; 11 trials, 67 min CPU; code and method hashes unchanged). **Pair panel FAIL:** 0/4
pairs accepted (18: sweeper declined, rival basin 179 degrees away within 2.06 %; 23 and 24: all wooden pieces declined,
rivals within 1.7-3.5 %; 27: rack declined, subsampling shift 11.9 degrees). **Up panel FAIL:** 5/11 accepted (needs 6),
all accepted up errors < 5.4 degrees.

**Construction caveat (Claude critic 16b):** the H2S-desc train scores depend on how the registered relative pose is placed in the T1 world: shortest-arc onto T1 (recorded) gives episode 18 AUC 0.6161, the oracle-style yaw-only construction 0.5582 (eight-episode mean with only 18 changed 0.3161 vs 0.3089); a candidate must fix its construction before scoring.

**Descriptive only (selected rotations, ignoring declines; lead re-derived from the saved NPZ and raw parquet frame 0,
all objects visible at frame 0):** 18 b 4.2 / C 5.6 degrees (offset length 22.55 vs 22.56 cm); 23 b 9.5 / C 13.7;
24 b 172.6 / C 11.7 (consistent with the documented arbitrary wooden-piece identity); 27 b 8.5 / C 3.0. The declined
wooden-piece selections are inverted in absolute up (about 179 degrees) while their relative pose is close. So on
these spent examples the selected rotations of the asymmetric pairs are accurate and the decline rule rejected them;
whether relaxing that safeguard transfers to new episodes is unproven (the sweeper's rival was a 179-degree flip only
2 % worse). The rule is not relaxed after the fact: adopting the selected rotations would override a failed rule and is put to the owner
(`docs/chain/NEEDS-YOU.md`), as the stereo placement was.
