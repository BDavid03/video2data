# M1 sealing record, committed before the sealed pass reads any 39 pose (4 Nov 2026, 09:18 UTC)

**Sealed references, candidate C, with the corrected body constant:**
- 39: `44cd03d9da3a9b017610eab00439662eb8d7bc83ff00bd0a386cff618d30b3e1`
- 38: `c5da9fbe4802a62ccfeca8627f2b9aa69a2acf14eaeecbc565b3a728fb275f12`

Neither reference changes after this line. Script: `data/.runtime/tmp/sealed-m1-27.py`, which refuses a second run and any other hash. Copied to the sealed output after running.

**Status at sealing. C has failed twice, and both stand.** The first frozen rule: 16 of 23 on 39, 17 of 23 on 38. Amendment 1: 22 of 28 on 39, 24 of 28 on 38. Failing under Amendment 1 on both clips: world residual, with the old instrument retained (25.4 and 19.5 mm against 10), because the noise-floor control did not explain it (15.2 and 12.5 mm), although the perpendicular part is 2.35 and 1.94 mm; table offset (93 mm on both against 10) and late normal change (3.46 and 49.8 degrees against 2). On 39 only: table and landmark coverage 72 % against 80 %. Passing, and new: long-baseline in-plane displacement, 6.3 and 5.2 mm against 10.

**This next part breaks Amendment 1's "it is the only one". It is written here, before any label, so that it can be judged as such.** The rule as written closes M1. The sealed pass is free on the CPU and is the only direct truth available, so it runs now whatever happens, once, with limits unchanged (AUC 0.60, 2.5 cm, 12 degrees; allowance 0.45).

1. **Sealed pass fails:** M1 closes. M2v does not run for adoption. D2 v5 stands. The finding goes in the report. No revision of the reference.
2. **Sealed pass passes:** the reading is that 39's own mocap has shown a 2.5 cm stereo residual was not reference error. Only then, on both clips, the world-residual gate uses the perpendicular instrument at the same 1 cm. **Nothing else moves.** Table offset and normal are under investigation as a code defect, for a physical reason: 49.8 degrees of normal change cannot coexist with under 6 mm of in-plane displacement from the same camera. If they are a defect in the code and it is fixed, the same limits apply. **If they are not a defect, or still fail, 38 is not adopted,** and M2v runs only as a report finding. 39's coverage miss is covered on 39 by its labels. 38 has its own coverage, which passes.
3. Whichever way this falls, the abstract and report state: M1 failed its first frozen rule, failed its amendment, and proceeded, if it did, under a conditional rule fixed before the label pass. Adoption on 38 then rests on video rules that were moved after seeing both clips, and one label pass on another clip. That is weaker than a prospective validation. A reader should discount it accordingly.

The owner is told in `NEEDS-YOU.md`, and can stop this with a dated line.
