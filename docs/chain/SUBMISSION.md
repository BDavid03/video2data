# Submission checklist

Kept current by every chain context (see `PROGRAM.md` section 1). Freeze: 4 November 2026,
17:00 EST (22:00 UTC). Package-ready target: 2 November 2026.

| Id | Deliverable | State | Evidence | Blocked on |
| --- | --- | --- | --- | --- |
| D1 | Track 3 mesh Parquet (CD-O) | not started: mask and mesh slices implemented, never run | - | first run of `object_masks` / `object_mesh` on the scored episodes |
| D2 | Track 3 pose CSV (five pose metrics) | not started: policy slices implemented, never run | - | Isaac Lab / FlashCHORD install (approved 2026-11-01, not yet done) |
| D3 | Extended abstract PDF | draft LaTeX from the depth line only: `outputs/auto_research/report/20261029T040958.830501Z-d8ea4610/main.tex` | depth results only | LaTeX engine install (approved 2026-11-01, not yet done) |
| D4 | Code at a GitHub commit URL | local git repository initialised 2026-11-01 | `git log` | owner: GitHub remote (NEEDS-YOU 1) |
| D5 | One reproduction command per deliverable | depth line only (`scripts/research_loop.py --finalize --report`) | `docs/design/handoff-2026-10-29.md` | - |
| D6 | Packed files, preflight clean, ready for the owner to upload | not started | - | D1 or D2 first; upload is the owner's (NEEDS-YOU 2) |

## Scored episodes and objects

Source: `upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json` and
`track_3_sample_submission.csv` (9,120 pose rows). The first context lists every scored episode
and its objects here.

## Challenge rules on external data, models and assets

Not read yet. The first context that relies on anything external reads the rules and records
here what they allow (`PROGRAM.md` section 3).

## Packed files

| Time (UTC) | File | SHA-256 | Commit | Final? | Scores (from the owner) |
| --- | --- | --- | --- | --- | --- |
