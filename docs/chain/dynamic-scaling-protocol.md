# Dynamic scaling D1: does a budget that fills our GPU learn object transport? (context 19; written before any D1 run; commit time is the record)

D0 (`docs/chain/dynamic-feasibility-outcome.md`) is NO-GO and stands. It tested only the cheapest budget (64 worlds,
128k transitions, about 1/2000 of the released 250M-transition recipe) and used 4.5 GPU-minutes. D1 asks the question
D0 left open, with its own fixed rule: does the largest budget that fits this machine in about 25 minutes per seed make
the shipped tissue-box object follow its reference with the virtual object controller off? Still a capability test on a
shipped example: never a submission, no challenge episode read.

## Method (fixed now)

- Task, evaluator, metrics, comparator S and code: exactly D0's slice (`track3/features/dynamic_feasibility/`), reused,
  with a new queue entry D1; D0 records are not touched.
- **Sizing probe (before training, reads no tracking error):** at most 5 GPU-minutes measuring environment steps per
  second and peak memory at 256, 512 and 1024 worlds (others in between if one fails). Choose the world count with the
  highest throughput whose peak stays <= 13.5 GiB; set total transitions = throughput x 1,350 s (rounded down to a
  multiple of the world count). The VOC curriculum thresholds scale with total transitions exactly as in L1 (assistance
  reaches zero before the end). Other training settings as L1. Write the chosen values into the queue and the ledger
  before the first D1 training run.
- **Arms:** D1 seed 0, then D1 seed 1; evaluation as D0 (final checkpoint, full 390 frames).
- Stop rule: 65 GPU-minutes in all, probe and compilation included. A crash or non-finite state is that seed's failure;
  no retry with new settings.

## Go / no-go (fixed now, same thresholds as D0)

**GO** if both seeds complete finite full rollouts with mean object position error <= 3 cm and <= 0.5 x S (S = 6.889 cm).
Otherwise **NO-GO**. Also reported: each seed's improvement over Z (7.012 cm), ADD-AUC, p90 error, wall time, memory.

What a verdict means: GO says the policy stage can learn transport on this machine at about 25 min per episode, so
video-derived hand and object references become the blocker (they need the installs proposed in `NEEDS-YOU.md`). It
does not by itself make a pack. NO-GO says that with this GPU and before 12:00 UTC 4 Nov, moving-object policies are
out of reach, and the structural effort goes to meshes (`PROGRAM.md` section 0, to-do 3).

Episodes 41, 43, 44 are never read; no challenge episode is read; no leaderboard use.
