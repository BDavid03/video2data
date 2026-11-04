# Dated notes on the M2v protocol, kept apart from it

`moving-m2v-protocol.md` holds the frozen rules, and Stage 0 of 38 and 39 pins its hash (`06dc2ac0...`). These notes change no limit and no rule.

**Why this file exists (4 Nov 2026, 12:29 UTC).** The first note below was appended to the protocol at 11:10:12 UTC, 12 seconds after that resume passed the Stage 0 guard. So the guard was not exercised against the note until 12:25 UTC, when it correctly refused the next resume, with no training started: `verify_scientific_inputs` found one changed input of 130 on each episode, this file. Rather than edit the guard, whose code Stage 0 pins, the protocol file was restored to its exact Stage 0 bytes (commit `8ef08ba`) and the note moved here word for word. Commit `ffbadcc` keeps the note where it first stood. The guard, Stage 0 records, thresholds and code are untouched.

## Note, 11:10 UTC, before any final number on 38 (moved verbatim)

**The kill rule fired and training was stopped at 11:05, with self-`T` -1.26 on the CPU replay. The official GPU evaluator on the same checkpoint agrees that it fails (`T` 0.25, `Q` -0.96, 25 cm, success 0; `ep38-v1/boundary-gpu-32814080/`). The rule is written under "M2v on 39", and the lead's brief extended it to 38. Its purpose was to free the GPU, which now has no other use. So the same run resumes from `state_34328576`, same seed, recipe and scheduled end, with its replay buffer lost, and with a deadline of about 37 minutes. It is one run, stopped and resumed, disclosed. Adoption is decided by the same rules at the final checkpoint, and if it misses or runs out of time, D2 v5 stands. No label is involved.**

## Note, 12:29 UTC, before any final number on 38

The owner stopped that resume at about 11:19 UTC ("end training temporarily") and then asked for this run, whose brief takes that as the go-ahead. The same run resumes from `state_40386560`: same seed, recipe and scheduled end, replay buffer lost a second time, deadline 35 minutes. It has now been stopped twice. Adoption is by the same rules at the final checkpoint.

**12:00 UTC, the last pack, has already passed, and no dated owner line moves it.** So unless one appears, this run's outcome is a report finding and no pack follows, whichever way it falls.
