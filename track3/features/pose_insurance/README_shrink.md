# Pose shrink control

CPU-only implementation of `docs/chain/pose-shrink-protocol.md`. The fixed queue scores
I, S1, G, P1–P7 and C0/C25/C50/C75 on eight train episodes and single-object controls
39/42, then computes six 151-point MPPE oracle grids and all 40,320 scale assignments
for the direction-free length diagnostic. Mocap/scans enter diagnostics and scoring,
never candidate geometry. Episodes 41/43/44 are never accessed.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.shrink_loop --queue track3/features/pose_insurance/shrink_queue.json
```

The queue is resumable: a trial identity hashes settings, protocol, assignments, stereo
records and anchor producers, scoring inputs, scene parents and computation sources.
Only a completed matching run with admitted inputs and an intact metrics artifact is
reused. The ledger is appended and fsynced. Assignment and conditional H construction
rows are recorded before their trials. Budget exhaustion leaves completed trials ready
for the same command. All 156 possible trials fit the default budget (140 candidates,
six oracle grids, ten H trials if triggered). A failed comparator is reported with the
fixed denominator retained and its dependent verdicts blocked.

`_shrink.py` contains pure computations; `shrink_diagnostic.py` owns gating, input
admission and recorded trials; `shrink_loop.py` owns orchestration and resume;
`shrink_report.py` writes `outputs/pose_insurance/shrink/summary.{json,md}`. The existing
`scale_diagnostic_authorized` flag and `diagnostic_python` interpreter apply. Shared
`heavy_execution` uses its default project cache and temporary directories, exactly as
the scale diagnostic does. Summary backups go to `.runtime/tmp/backup/<UTC stamp>/`.
Existing source files and runtime flags stay pinned.

Reproduction compares I and S1 against the full-precision scale summary for all ten
episodes with tolerance 1e-9. Each candidate's single-object control has its own eligibility
gate. Verdicts follow the registered practical margins and MPPE tie tolerance. The length
rank is the competition rank (one plus strictly better assignments); exact ties are reported.
Global scale is recomputed at full precision and checked against the registered rounded
0.856952 within 1e-6. Oracle grid MPPE calls the kit's own alignment and MPPE functions;
scan-based AUC is unnecessary when only MPPE is minimized.

H's train construction and scoring run only if verdict 2 triggers. Its factors act once
on the v2 offset, with the registered SAT clearance bisection and S1 fallback. No scored
H evaluator, packing or training is run here.

## Recorded result (2026-11-02)

All 156 trials completed; unchanged-code resume admitted all 156 without scoring again.
I and S1 reproduced all five recorded metrics exactly on all ten episodes. Every candidate,
including H, passed both single-object controls. All 76 pre-existing pose/stereo files
kept their original hashes. Full results: `outputs/pose_insurance/shrink/summary.{md,json}`.

- Verdict 1: difference below the registered practical margin. G minus S1 is
  −0.042320834 cm MPPE and −0.059553770 cm RPE; S1 beats only three of seven shifts.
- Verdict 2: further shrinkage helps. C0 has mean MPPE 14.133047791 cm versus
  S1 20.843564645 cm (gain 6.710516854 cm). C0 remains an implausible diagnostic bound.
- Oracle c*: median 0.025, range 0–0.50; Spearman(s, full-distance ratio) 0.542857143.
- Length information not shown: six-episode errors I/S1/G are
  20.532477704 / 14.713075833 / 14.554395566 cm; S1 ranks 26,559/40,320 (65.87%).
- H uses c_H=0.25 with six per-episode clearance floors. Its RPE/MPPE gains against
  S1 are 8.360113551 / 3.823015849 cm, but mean AUC falls 0.000180288462
  (episode 23 loses 0.001442307692 AUC) and episode 31's MPPE rises 0.693391880 cm,
  exceeding 0.5 cm. **H fails; retain D2 v3.**

The optional CPU episode-5 inspection is in `outputs/pose_insurance/shrink/episode5_diagnosis.md`:
only the task position field changes; finite/unit poses and positive dimensions/inertias
pass, and footprints remain separated by a projected 7.40 cm. V3 has six non-finite
actor BatchNorm-statistic tensors and 21 non-finite state tensors, including critic/target
BatchNorm statistics and temperature/Adam state. Initial box overlap is ruled out; the
first invalid value and its cause remain unknown.

The final independent arithmetic/oracle/clearance recheck is `verification.json`;
`commands.md` records exact commands, and `changed_files.txt` lists every generated
file, including scratch and cache files. No test suite was run.
