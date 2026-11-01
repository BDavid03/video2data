# Submission checklist

Kept current by every chain context (see `PROGRAM.md` section 1). Freeze: 4 November 2026,
17:00 EST (22:00 UTC). Package-ready target: 2 November 2026.

| Id | Deliverable | State | Evidence | Blocked on |
| --- | --- | --- | --- | --- |
| D1 | Track 3 mesh Parquet (CD-O) | started: first stage (MoGe2 depth) ran on scored episode 38; local CD-O scorer built; primitive reference baselines measured (cube 2.52 cm, sphere 2.82 cm: per-object mean over the 10 scanned objects, three kit seeds; not the leaderboard's 36-pair mean, and the spoon, 10 of 36 pairs, has no scan) | `outputs/depth/evaluation/episode_000038/20261101T030402.023623Z/run.json`; `outputs/mesh_eval/*.json`; `scripts/track3/local_cdo.py` | nothing external: video-measured shapes for 11 objects, then packing (`.runtime/kit-env` installed) |
| D2 | Track 3 pose CSV (five pose metrics) | runtime installed and smoke-tested; no task parquet, checkpoint or rollout yet | `scripts/linux/install-flashchord.sh`, `.runtime/flashchord/install-logs/device.log`, `.scratch/fc-replay.log` (worktree) | object 6-DoF tracks from video, a task parquet per episode, hand references (no hand model installed), per-episode training |
| D3 | Extended abstract PDF | draft LaTeX from the depth line only: `outputs/auto_research/report/20261029T040958.830501Z-d8ea4610/main.tex` | depth results only | LaTeX engine install (approved 2026-11-01, not yet done) |
| D4 | Code at a GitHub commit URL | local git; this context's work is on branch `worktree-chain-ctx01` | `git log` | owner: GitHub remote (NEEDS-YOU 1) |
| D5 | One reproduction command per deliverable | depth line (`scripts/research_loop.py --finalize --report`); installs (`scripts/linux/install-flashchord.sh`, `install-kit-env.sh`); local CD-O (`scripts/track3/local_cdo.py`) | this file | - |
| D6 | Packed files, preflight clean, ready for the owner to upload | not started | - | D1 meshes first; upload is the owner's (NEEDS-YOU 2) |

## Triage (context 01, 2026-11-01 03:30 UTC, 90 h before the freeze)

**D1 can land by 2 November.** Every input exists: the 20 evaluation ego videos (3 streams each,
decodable, frame-exact), MoGe2 weights, the stereo and geometry-only mask slices, and the kit
environment for packing. A valid file needs only one closed mesh per (episode, object); the
measured floor for any closed primitive is about 2.5 cm mean CD-O. The decisive measurement:
a primitive scaled to the object's true principal extents scores 1.17-1.31 cm (analysis with
scan-derived extents, never a candidate), so measuring three extents per object from video is
worth more than half of the gap to a perfect mesh. Route for D1, cheapest first:

1. Per object, measure its three principal extents from video (stereo or MoGe points above the
   table in the opening frames, before the hand touches it), fused across every scored episode
   where the object appears, and submit a box/ellipsoid with those extents. Validate the
   measurement on the public episodes against the scans with `local_cdo.py`.
2. Replace objects one at a time with richer shapes (opening hull, revolved profiles for cups,
   pot, lid and pitcher; procedural spoon and rack), keeping a replacement only when it beats
   the extents primitive on public-episode CD-O. Priority by score weight: wooden_spoon (10 of
   36 pairs), white_pot (6), blue_cup (5), plastic_dish_rack (5): 72 % of CD-O.
3. Pack with `pack_reconstruction.py track3-meshes` in `.runtime/kit-env`, with the local commit
   in a placeholder GitHub URL, marked not final until the owner pushes (NEEDS-YOU 1).

**D2 cannot land by 2 November; a minimal valid D2 by 4 November is possible but not proven.**
Measured: FlashCHORD at aee95f25 needs no Isaac Sim or Isaac Lab (it is Newton 1.2.0 +
MuJoCo-Warp 3.8.1 + JAX); it installed natively in `.runtime/flashchord/venv`; Warp and JAX see
the RTX 5080; a headless replay built the Sharpa-hand scene and compiled every kernel on sm_120
under WSL (exit 0). What blocks it:

- The evaluator only rolls out a real FlashSAC/PPO checkpoint trained by FlashCHORD, whose
  embedded config names a per-episode *task parquet*: hand wrist/finger references, object body
  poses, contacts and object mesh/URDF paths. None exists for any evaluation episode.
- The scored poses are the simulated objects, moved by simulated hands. Hand references need a
  hand estimator; WiLoR and MANO weights are not present (new dependency, owner approval).
- Object meshes in simulation must be our reconstructions too (FAQ), so D2 waits on D1's meshes.
  `track3/features/object_mesh/sources.json` still maps ten objects to `public_scan`: that
  mapping must change before any pose work.
- FlashCHORD's README asks 48 GiB GPU for full training; this GPU has 16 GiB (use few worlds).

Measured floor (`scripts/track3/static_pose_baseline.py`, `outputs/pose_eval/static-baseline.md`):
if every object stays at its first pose, the kit's pose metrics on 17 public episodes give AUC
0.46, SP-SR 0.41 (7 of 17 pass), MP-SR 0, RPE 12.8 cm, MPPE 9.3 cm (AUC approximate: our own
vertex samples). Objects move 4-16 cm on average. Moving slot 1's start by 5 cm costs 0.12 AUC
and 0.18 SP-SR, so first-frame relative placement matters. A static-objects rollout is therefore
a real insurance entry for D2 (it needs initial object poses from video, our meshes and a
checkpoint whose hands leave the objects alone); beating it needs metric object tracks and a
policy that moves the objects. Shortest route: (1) 6-DoF object tracks from stereo with our own meshes
(idea 11), (2) a task parquet per episode, with hand references from a hand model (NEEDS-YOU 5)
or scripted approach paths, (3) a short FlashSAC run per episode with few worlds, (4) the kit's
`evaluate_policy.py`, then pack. Still worth one valid static-object D2 file as insurance if
time allows: every metric competition needs some file before the freeze to rank at all.

## Scored episodes and objects

Source: `upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_evaluation_objects.json`
(object order is the scorer's slot order) and `track_3_sample_submission.csv` (9,120 pose rows,
one world per episode). Public episodes with the same object are where it can be measured
against a scan (episodes 41, 43, 44 excluded everywhere).

| Episode | Objects (slot 0, slot 1) | Pose rows | Frames | Public episodes with a scan of these objects |
| --- | --- | --- | --- | --- |
| 1 | white_pot, white_pot_lid | 670 | 335 | 0, 2 (pot also 26, 27, 42, 45) |
| 3 | white_pot, wooden_spoon | 626 | 313 | pot only; no spoon scan |
| 5 | white_pot, wooden_spoon | 576 | 288 | pot only |
| 6 | white_pot, wooden_spoon | 486 | 243 | pot only |
| 8 | beige_cup, white_pot | 438 | 219 | 11 (cup); pot as above |
| 10 | blue_cup, beige_cup | 322 | 161 | 11 |
| 14 | blue_cup, plastic_dish_rack | 316 | 158 | 12, 13 |
| 15 | plastic_dish_rack, wooden_spoon | 332 | 166 | rack 12, 13, 26, 27, 31 |
| 16 | plastic_dish_rack, wooden_spoon | 346 | 173 | rack as above |
| 19 | mini_sweeper, mini_dust_pan | 470 | 235 | 18 |
| 20 | blue_cup, water_pitcher | 404 | 202 | 21, 22 |
| 25 | wooden_piece_1, wooden_piece_2 | 712 | 356 | 23, 24 |
| 28 | plastic_dish_rack, white_pot | 440 | 220 | 26, 27 |
| 30 | mini_sweeper, plastic_dish_rack | 526 | 263 | 31 |
| 32 | blue_cup, wooden_spoon | 518 | 259 | cup 11, 12, 13, 21, 22 |
| 33 | blue_cup, wooden_spoon | 696 | 348 | cup as above |
| 35 | wooden_spoon | 421 | 421 | none (RPE = 0) |
| 36 | wooden_spoon | 407 | 407 | none (RPE = 0) |
| 37 | wooden_spoon | 151 | 151 | none (RPE = 0) |
| 38 | mini_dust_pan | 263 | 263 | 18, 39, 40 (RPE = 0) |

36 (episode, object) pairs over 11 objects: wooden_spoon 10, white_pot 6, blue_cup 5,
plastic_dish_rack 5, beige_cup 2, mini_dust_pan 2, mini_sweeper 2, white_pot_lid 1,
water_pitcher 1, wooden_piece_1 1, wooden_piece_2 1. Four episodes (35-38) have one object, so
RPE is zero for them and MP-SR needs only that object.

## Challenge rules on external data, models and assets

Read 2026-11-01 from the challenge FAQ (`https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf`,
copy in `.runtime/tmp/rules-20261101/faqs.pdf`), the challenge page and the kit `config/tracks.json`:

- "Pretrained models and external data are allowed for all tracks."
- Track 3: "A valid Track 3 submission requires object trajectories from learned policies and
  reconstructed object meshes for all 20 egocentric videos. While some objects in the eval set
  have meshes in the public data, participants should reconstruct the objects to generate object
  trajectories and submit their own reconstructed meshes."
  Our reading (fail-closed): the public scans are used only to *measure* reconstructions on public
  episodes. No scan geometry, dimension or template enters a submitted mesh or the simulated
  object assets. One reconstruction per physical object pooled across scored episodes is not
  addressed by the rules; it uses only released video and is described openly in the report.
- Sharpa Wave is the default embodiment; floating hands and arm-and-hand setups are welcome.
  Simulation only. Track 3 is ranked by the sum of per-metric ranks (six metrics).
- Submission limit: five per week per track, unlimited in the final three days. Winning code is
  released under Apache-2.0.
- The kit's host checks evaluator hashes, so trajectories must come from the official
  `evaluate_policy.py`; no hand-built trajectories.

## Packed files

| Time (UTC) | File | SHA-256 | Commit | Final? | Scores (from the owner) |
| --- | --- | --- | --- | --- | --- |
