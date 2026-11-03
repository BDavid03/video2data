# Submission checklist

Kept current by every chain context (see `PROGRAM.md` section 1). Freeze: 4 November 2026,
17:00 EST (22:00 UTC). Package-ready target: 2 November 2026.

| Id | Deliverable | State | Evidence | Blocked on |
| --- | --- | --- | --- | --- |
| D1 | Track 3 mesh Parquet (CD-O) | **v4 packed, preflight "Ready to submit"** (context 08; supersedes v3). **Round 4 (context 11, Amendment 5, frozen `f71e8f2` before scoring): no change, v4 kept.** R10 pooled vessel profiles were killed by their video-only test on tuning 42 (subset knots 45.7 % of radius apart; the camera sees the open pot's interior); R11 exact exterior rack boundary scored worse in development (rack 0.860 -> 0.904 cm, equal-object 0.6571 vs 0.6527), so no v5 (`outputs/mesh_extents/rules-round4/summary.md`, IDEAS 41-42). v4 details: rule R9 = R7 compartment rack + R8 thin-tool profiles (dust pan, sweeper, unscanned spoon), adopted under the pre-score Amendment 2 (`bc601cf`, 05:03:47 UTC; first score 05:06:45 UTC): cross-fitted 0.65270 cm per object vs incumbent R6/0.10 0.77315 (gate < 0.74315), R8 passed alone (0.70576) as the spoon safeguard requires; **exact packed CD-O 0.6629 cm pair-weighted / 0.6534 per object vs v3 0.8115 / 0.7711** (9/26 pairs better, 17 equal, 0 worse; stricter veto per object < 0.74108 passed); re-checked by the lead with `local_cdo` on rack ep 15, sweeper ep 19, dust pan ep 38 and from the pair ledger. Section "D1 v4" below. **History, v3** (context 07; supersedes v2): rule R6 peg-rack (R2 + flat rim handle for the pot, hollow thin-walled open vessels at wall/radius 0.10, video-counted two-row peg rack) adopted under the pre-score Amendment 1 of `docs/chain/mesh-selection-protocol.md` (`c7e2b56`): cross-fitted 0.77315 cm per object vs incumbent 0.92506, worst regression 0.0; **exact packed CD-O 0.8115 cm pair-weighted / 0.7711 per object vs v2 1.0850 / 0.9257** (19/26 pairs better, 0 worse); re-checked by the lead through `local_cdo` on built PLYs (pot, rack, blue cup to 5e-9 cm) and from the pair ledger. Section "D1 v3" below. **History, v2** (context 03): adopted R2 vessel-handle rule, aspect threshold 1.3, handle height fraction 0.6; v1's exact video measure runs, 36 pairs. Cross-fitted per-object CD-O **0.92572 cm** vs v1 box 1.00764; cross-fitted pair-weighted **1.08742 cm** vs 1.25662; all three adoption gates pass. Fresh score of the final all-object fitted build: **0.9251 cm** (development estimate, public scans). The spoon keeps its v1 box and remains unmeasured. **Exact local CD-O of the packed files (context 06):** each of the 26 scannable (episode, slot) pairs scored through the kit's own `score` with its own seed: v2 pair-weighted 1.0850 cm, per-object 0.9257; v1 1.2547 / 1.0091 (`outputs/mesh_eval/pairs/summary.md`, slice `track3/features/mesh_eval/`); within 0.003 cm of the earlier reused-object estimates | `submissions/20261102T0313Z-d1-v3/`; build `outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295/run.json`; `outputs/mesh_eval/rules-round2/`, `outputs/mesh_eval/pairs/mesh-round2-v2-v3/`; v2: `submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet`; build `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/run.json`; `outputs/mesh_eval/d1-ctx03/`; packed-file details below | owner: GitHub push, then repack with the real URL (NEEDS-YOU 1, 2) |
| D2 | Track 3 pose CSV (five pose metrics) | **v4 packed (context 12; supersedes v3), preflight "Ready to submit":** `submissions/20261103T0645Z-d2-v4/track3_poses_v4_final.csv` (SHA-256 7221dd3a..., 9,120 rows, URL commit `d1d1da5`; directory stamp is 17 min late, packed 06:28 UTC, kept because the manifest records the path). T1 (`docs/chain/pose-stereo-placement-protocol.md`, `f18664c` + Amendment 1 `c7c079b`, owner-approved): slot 1 at the direct calibrated-stereo centroid offset. Train gates vs S1 passed: RPE 35.34 -> 29.01 cm, MPPE 20.84 -> 17.68, AUC 0.2690 -> 0.2767 (episode 23 only), no episode worse, S1 reproduced exactly (lead re-derived). Scored: 12 episodes changed (3, 6, 8, 10, 14, 15, 16, 19, 20, 25, 32, 33), all pass the physical check (first pose <= 0.049 cm, drift <= 0.048 cm); 30 keeps v3 (no slot-0 cluster); 1, 5, 28, 35-38 keep v3; the eight unchanged episode exports are byte-identical to v3; in the 12 changed episodes slot 0 is numerically equal to v3 (policies re-ran: max 7.45e-9 m, 451 rows differ only in signed-zero formatting), only slot 1 moves. The 12 GPU runs recorded mid-refactor hashes of six length/scored modules (a parallel refactor job); recomputing every construction with the committed code at `d1d1da5` matches the frozen plans, task files and all 20 export hashes exactly (`outputs/pose_insurance/stereo-placement-scored/source-drift-check.md`) (critic-12 correction; commit message of `d1d1da5` says byte-identical, which holds for the unchanged episodes only). Caveats: train gates not blind (frame-0 errors known before the protocol); fresh confirmation failed by the letter (13 merged) and the owner approved anyway; slot-1 direction still unobservable. `outputs/pose_insurance/stereo-placement{,-scored}/summary.md`; reproduce `bash scripts/reproduce/d2_poses_v4.sh --run --code-commit <url> --out <csv>`. **History, v3:** **v3 packed, preflight "Ready to submit"** (context 05; supersedes v2): v2 plus the stereo-anchored slot-1 offset scale S1 (`docs/chain/pose-scale-protocol.md`, IDEAS 27): calibrated C/B stereo gives stereo/MoGe-2 scale s 0.82-0.92 on the 8 train two-object episodes, and scaling slot 1's table-plane offset from slot 0 by s lowers train RPE 39.36 -> 35.34 cm and MPPE 22.82 -> 20.84 (every changed episode better; AUC 0.2688 -> 0.2690; SP-SR/MP-SR 0). Adopted by a post-result lead reading of the invariance clause (recorded in the protocol). Scored: S1 on 13 episodes (3, 6, 8, 10, 14, 15, 16, 19, 20, 25, 30, 32, 33; s 0.77-0.95; all pass the physical check, first pose <= 0.049 cm, drift <= 0.048 cm); v2 kept on 1, 28 (stacked, S1 no-op), 5 (training produced non-finite optimizer state; fallback, no retry) and single-object 35-38. Caveats: train distance ratio still 1.18-2.23 after S1 (3.58 on default-offset episode 31), so part of the gain may be a hedge effect of a shorter offset; public calibration assumed for the evaluation rig; AUC/SP-SR not expected to move. **Shrink control (context 06, `docs/chain/pose-shrink-protocol.md`):** a global shrink (mean train s) matches S1 within 0.042 cm MPPE and S1 beats only 3/7 shuffled scales, so S1 did not meet the registered calibration-benefit criterion and its gain is consistent with shrinkage (equivalence not established); zero offset would give 14.13 cm (physically implausible); the pre-registered clearance-limited H failed two gates, so v3 stays | `submissions/20261102T0032Z-d2-v3/` (`manifest.json`), `outputs/pose_insurance/scale/summary.md`, `outputs/pose_insurance/v3/report.md`; v2 history below | slot-1 direction (2-130 deg) and relative rotation stay unobservable without a semantic object frame (IDEAS 29), so AUC/SP-SR still need moving objects or frames |
| D3 | Extended abstract PDF | **anonymous abstract "Primitive Meshes and Static Poses", two main pages plus references.** **Context 15: pose section now describes packed D2 v4** (stereo placement of the second object; train RPE 35.34 -> 29.01 cm, MPPE 20.84 -> 17.68, AUC 0.2690 -> 0.2767 from one episode, no episode worse; train-informed selection, incomplete fresh confirmation and adoption anyway, 51-130 deg direction errors, mask reassignment, calibration transfer and simulation-consistency-only checks disclosed; metric definitions in Table 2). Codex wrote it (`d3-pose-v4-15`), a cold reading (`cold-d3-15`, Presentation 3/5) was applied (`d3-fix-15`) plus lead edits; `d3_report.sh --run` on `6590235`: main.pdf c0dfdf26..., main text ends on page 2, known microtype error only (`.runtime/tmp/reproduce-d3-JBNeVro8/`). Context 10: fourth cold reading (Codex `cold-d3-10`, Presentation 4/5, Visualisations 3/5) applied in `81537f2`: opening claim bounded to development on inspected public scans, plain selection wording and Table 1 labels/caption, spoon paragraph separates the three-view mean IoU (+0.23) from Figure 2's single-frame IoU, pose sentence limited to Table 2; Figure 2 renders v3/v4 in one neutral material with a translucent spoon mask and "IoU is not a validated CD-O proxy". `d3_report.sh --run` on `81537f2`: main.pdf eb10ac18..., main text ends on page 2, known microtype error only (`.runtime/tmp/reproduce-d3-OW3oLL47/`). Then both critics (P4/V4) and their fixes in `3550af6` (v3/v4 defined before Figure 1, readable Table 1 caption, the pre-registered proxy criterion with 5/8, pose sentence in the conclusion, Figure 2 compartment/grip labels, zero-padded date): `d3_report.sh --run` on `3550af6`: main.pdf 749af1c0..., main text ends on page 2 (`.runtime/tmp/reproduce-d3-Uo3oHLyQ/`). **History** (context 09): third cold reading (Codex, Presentation 3/5, Visualisations 3/5) applied: constructors described as MoGe-2-derived extents plus manually read video attributes, with scan-guided design stated up front; 26/36 coverage beside every mesh headline (object mean -15.3 %, pair mean -18.3 %, 9 better / 17 tied / 0 worse); Figure 1 is now an all-object v3->v4 dumbbell (per-object packed CD-O, asserted against the pair ledger; spoon row 'unmeasured'); round 3 told plainly; pose section a negative result; reproducible normalisation sentence; spoon proxy result (NOT VALIDATED; v4 outline IoU +0.23, descriptive) in the spoon paragraph. `d3_report.sh --run` on HEAD cfc8be5: main.pdf fde60527..., 3 pages, main text ends on page 2 (`.runtime/tmp/reproduce-d3-9ZEhdVeg/`). Then critic-09 fixes (critics: Presentation 4/4, Visualisations 3/4 before them): Figure 2 a qualitative strip (rack and dust pan v3/v4/scan, spoon video crops with box and bowl outlines), CD-O invariance sentence, one hedge plus one disclosure paragraph, motivation sentence, Table 2 without the all-zero SP-SR column; `d3_report.sh` now refuses a figure generator that differs from the built commit. `d3_report.sh --run` on HEAD 5b63d2d: main.pdf 1229a6b3..., 3 pages, known microtype error only (`.runtime/tmp/reproduce-d3-6WL89nGq/`). Not rescored. **History** (context 08): second cold reading (Presentation 3/5, Visualisations 2/5) applied (scan exposure up front, our protocol vs challenge rules, reconstructible exploratory pose selection, MP-SR and units, uniform-shrinkage series, larger v3/v4/scan Figure 1), then updated to D1 v4 (0.6534 vs v3 0.7711 cm per object), then critic fixes (round 3 labelled fixed-rule in-sample, sampling SDs, spoon-risk paragraph with the 0.39 cm break-even, spoon row in Figure 1, separate conclusion); `d3_report.sh --run` on HEAD 5a10c91 builds with only the known microtype warning, main text ends on page 2 (`.runtime/tmp/reproduce-d3-IeA3SqSu/`, main.pdf 23a29416...). **History** (context 07): cold reviewer reading applied (plain names, defined rules, matched packed headline), describes shipped D1 v3 (0.7711 vs v2 0.9257 cm per object), round-2 gates and veto, scan-render and one-object-per-rule disclosures, seed spread, thickness trade-off, conclusion; `d3_report.sh --run` on HEAD 9522e17 builds cleanly (`.runtime/tmp/reproduce-d3-wPepz3Lp/`, main.pdf cdd34c6a...) | `docs/report/track3/` (`bash scripts/reproduce/d3_report.sh --run`; provenance table in its README) | - |
| D4 | Code at a GitHub commit URL | local git; this context's work is on branch `worktree-chain-ctx01` | `git log` | owner: GitHub remote (NEEDS-YOU 1) |
| D5 | One reproduction command per deliverable | **Context 10:** `d3_report.sh --run` on `81537f2` builds cleanly; the spoon proxy command reproduces 149/149 identical IoUs after the handler slimming (`c04c0a6`) and the dispatcher change (`7e1fabe`), 0 new trials on resume; new `python -m track3.features.mesh_extents proxy-insights --request track3/features/mesh_extents/proxy-insights.json` replays 135/135 trials exactly, 0 new on resume. **D3 executed (context 09):** `d3_report.sh --run` on cfc8be5 builds cleanly (known microtype error only). **Spoon proxy (context 09):** `python -m track3.features.mesh_extents proxy --request track3/features/mesh_extents/proxy-spoon.json` reproduces 149/149 IoUs after the engineering fixes; a second run makes 0 new trials. **D1 v4 executed (context 08):** `bash scripts/reproduce/d1_meshes.sh <url> --version v4 --run` rebuilt v4 with 47/47 PLY files identical to the veto build and a byte-identical Parquet (6663a1c7...); `--version v3 --run` re-run on the same script still rebuilds v3 byte-identically; `verify.sh` on the v4 manifest 6/6 OK (`outputs/mesh_extents/reproduction/20261102T052748Z/summary.md`). **D1 v3 executed (context 07):** `bash scripts/reproduce/d1_meshes.sh <url> --version v3 --run` rebuilt v3 into a fresh build (`20261102T031152.089350Z-ea88a295`), 36 meshes byte-identical to the veto build, preflight Ready to submit; that Parquet is the packed v3; `verify.sh` on the v3 manifest 1/1 OK. **Earlier, executed (context 05):** `d1_meshes.sh --run` rebuilt D1 v2 from the 23 recorded measure runs into a fresh directory, byte-identical Parquet (43fa6ac0...) and CD-O 0.9251 cm; `d2_poses.sh` is version-aware (v1 GPU campaign; v2/v3 CPU composition from verified exports) and its `--run` reproduced D2 v2 (6ea6530b...) and v3 (1dcbca75...) byte-identically; `d3_report.sh --run` builds from an archive of HEAD into fresh directories; two runs give identical `main.pdf`. `verify.sh` on the v3 manifest: 500/500. Not executed: GPU regeneration of the D2 policies | `scripts/reproduce/README.md` | - |
| D6 | Packed files, preflight clean, ready for the owner to upload | **Context 12: D2 v4 packed (`submissions/20261103T0645Z-d2-v4/`, commit `d1d1da5`), replaces D2 v3 for upload; D1 v4 unchanged.** D1 v4 and D2 v3 packed with placeholder commit URLs (not final); kit preflight `--skip-kaggle`: "Ready to submit" for both | `submissions/20261102T0516Z-d1-v4/`, `submissions/20261102T0032Z-d2-v3/` | GitHub remote for the real URL; upload is the owner's (NEEDS-YOU 1, 2) |

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

## D1 v0 (context 02, 2026-11-01)

Method (`track3/features/mesh_extents/README.md`): MoGe2 depth on opening frames 0-19 of each
episode (no stereo: a per-frame scale does not change proportions); floor plane, then the table
as the sharpest height peak above it (the camera_motion world plane was the floor on ep 39, which
broke the opening-clusters route); clusters above a locally refitted table plane; a recorded
visual assignment of frame-0 clusters to object names (`assignments.json`; 23 of 30 episodes
used, the rest left out with a reason: merged or border-cut objects); per-frame in-plane PCA
extents and 98th-percentile height, frames kept only while the mask size stays within
0.8-1.25x frame 0's; per object, proportions pooled by per-axis median over episodes (2-9
episodes per object). The same box goes to every scored pair of that object.

Shape rule: box for every object, chosen once on the per-object mean over the 10 scans (box
1.009, ellipsoid 1.044). **Disclosure:** this is selection on the public scans, and the
thresholds were set while looking at public episode 39's scan score (ep 39 is tuning data). No
scan geometry or dimension enters a mesh, but these local CD-O numbers are development
estimates, not an untouched assessment; say so in the report. Per-object shape picking was not
done. From context 03 on, any mesh choice is made on a fixed tuning protocol written down before
scoring (see the hand-off).

Reproduce (from the commit in the packed file):

```bash
bash scripts/track3/depth_opening_frames.sh evaluation 1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38
# plus public 39 (frames 0-19) and evaluation 1 at stride 10: requests in each depth run.json
PYTHONPATH=. .runtime/linux-cuda/bin/python scripts/track3/d1_measure.py --out .runtime/tmp/d1-runs.json
PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_extents build --request <{"measure_runs": [...], "shape": "box"}>
.runtime/kit-env/bin/python upstream/v2d_submission_kit/v2d_submission_kit/tools/pack_reconstruction.py track3-meshes \
  --meshes outputs/mesh_extents/build/<run>/meshes \
  --sample upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_mesh_sample_submission.parquet \
  --commit https://github.com/<owner>/<repo>/commit/<sha> --out submissions/<stamp>/track3_meshes.parquet
```

`assignments.json` names each episode's depth run by path, so a fresh rerun must regenerate the
depth runs and update those paths (the run ids are timestamps).

## Packed files

### D2 v4 (2026-11-03 06:28 UTC, context 12; supersedes D2 v3 for upload)

Code commit `d1d1da5`; URL `https://github.com/BDavid03/video2data/commit/d1d1da5880eca24413e43965bc78599cb4e9c648`
(the owner must push it). Kit preflight "Ready to submit" with Kaggle checks skipped; full preflight fails only
on the missing Kaggle token. Not uploaded.

| File | SHA-256 | Command | Identity |
| --- | --- | --- | --- |
| `submissions/20261103T0645Z-d2-v4/track3_poses_v4_final.csv` | 7221dd3ad8cabed7037d35a61293f1aac26b1dbcfd9498fa617a02791d9f57cf | `bash scripts/reproduce/d2_poses_v4.sh --run --code-commit <URL> --out <csv>` | 9,120 rows; manifest `track3_poses_v4_final.manifest.json` pins the protocol, the 12 GPU policy runs and the composition |

### Final real-URL files (2026-11-03, owner approval "all owner requirements approved")

Code commit `15060c1` (Apache-2.0 `LICENSE`, identity-checked reproduction scripts); URL `https://github.com/BDavid03/video2data/commit/15060c13b07bda03b056f5cc507f0e4f68a083f0`.
Rebuilt by the recorded scripts; only the URL column differs from the packed v4/v3 files; kit preflight
"Ready to submit" for both. Not uploaded yet: the commit must be pushed and the upload needs the owner's
Kaggle token (NEEDS-YOU 1, 2).

| File | SHA-256 | Command | Identity |
| --- | --- | --- | --- |
| `submissions/20261102T0516Z-d1-v4/track3_meshes_v4_final.parquet` | a4c7f8a5675885234ccdf77e7d6eb6f4ff26263fda420ce2ed4914e22480935f | `bash scripts/reproduce/d1_meshes.sh <URL> --version v4 --run` (`manifest-final.json`) | 36/36 PLY byte-identical; 294,912 rows equal except `code_commit_url` |
| `submissions/20261102T0032Z-d2-v3/track3_poses_v3_final.csv` | 903ebaef4696192c7268017407bcd20870b4de4e8cb7a852ab6d216dede41192 | `bash scripts/reproduce/d2_poses.sh <URL> --version v3 --run` (`manifest-final.json`) | 9,120 rows equal except `code_commit_url` |

| Time (UTC) | File | SHA-256 | Commit | Final? | Scores (from the owner) |
| --- | --- | --- | --- | --- | --- |
| 2026-11-01 10:35 | `submissions/20261101T1035Z-d1-v0/track3_meshes_v0.parquet` (D1, 294,912 rows, 1.6 MB) | `f814da46bdaaa747012cc078d6ca29274070633ab4745bcbae5319fabc24b710` | `0f8619373f2953eada625e25aa53d98f0f06d11a` (branch `worktree-chain-ctx01`) in placeholder URL `https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/...` | **no**: repack with the real URL after the push | - |
| 2026-11-01 15:00 | `submissions/20261101T1500Z-d1-v1/track3_meshes_v1.parquet` (D1 v1, supersedes v0) | `a76143b933ba09453a02b8100f2fd15af39ab51ca1064b7dea274c0eff411a9b` | `93038d282478c8f7923425cdb2648ce8ef09d2c6` in the placeholder URL | **no**: repack with the real URL after the push | - |
| 2026-11-01 16:05 | `submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet` (D1 v2, supersedes v1; 294,912 rows, 1,568,410 bytes) | `43fa6ac02353520aac4d019f7d6d45e0aab923f1c8f43d6cf03e30c242ba95c2` | `874ff6793721faa0035a71788b8f5a22f9b4b2d3` in `https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/...` | **no**: requested placeholder URL; repack with the real URL after the push | - |
| 2026-11-01 16:17 | `submissions/20261101T1617Z-d2-v0/track3_poses_v0.csv` (D2 v0 insurance; 9,120 rows, 4,815,348 bytes; kit preflight `--results outputs/pose_insurance/results --skip-kaggle`: Ready to submit) | `f19c365a176daaaa21d18b8c164a202fc5b0224e258069c51ef033e3c47f7714` | `67036c321432792dbb7440db4e08122a5c8fde7f` in the placeholder URL (the slice itself is in `e133bde`; the final repack must use a pushed commit that contains it) | **no**: placeholder URL | - |
| 2026-11-01 17:03 | `submissions/20261101T1703Z-d2-v1/track3_poses_v1.csv` (D2 v1, supersedes v0; 9,120 rows; preflight Ready to submit) | `fa3b1c320128a196648e0a966eb81b8e5cb8e65cb8ad6b1de0a3fbbb4523c26b` | `aa7d5566a6af8596fbb85a91dac7885e94290f50` in the placeholder URL (slice code in `7bc1a36`; repack from a pushed commit that contains it) | **no**: placeholder URL | - |
| 2026-11-01 23:15 | `submissions/20261101T2315Z-d2-v2/track3_poses_v2.csv` (D2 v2, supersedes v1; 9,120 rows; K1a on episodes 1, 28; preflight Ready to submit; `verify.sh` 107/107 OK) | `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741` | HEAD at pack time in the placeholder URL (`manifest.json: code_commit_url`); slice code committed after packing in the context-04 commits, so the final repack must use a pushed commit that contains them | **no**: placeholder URL | - |
| 2026-11-02 00:32 | `submissions/20261102T0032Z-d2-v3/track3_poses_v3.csv` (D2 v3, supersedes v2; 9,120 rows, 4,817,499 bytes; S1 offset scale on 13 episodes; preflight Ready to submit; `verify.sh` 500/500 OK) | `1dcbca75223873e233f187c781457cd5cbeab8b0fcce0bf955933e83fe6ca416` | `bb8a3f02fac668e86ff7b6a3ad7c3ccaec40050d` in a placeholder URL | no | - |
| 2026-11-02 03:13 | `submissions/20261102T0313Z-d1-v3/track3_meshes_v3.parquet` (D1 v3, supersedes v2; 294,912 rows; R6 peg-rack; reproduced by `d1_meshes.sh --version v3 --run`; preflight Ready to submit; `verify.sh` 1/1 OK) | `a61f02b12078a05c940037034e37ec4201e3e919a55e4bea9e0058dfa2f45f40` | `e5b47a233c5a96b033eef4af5f695b31e2de1cde` in the placeholder URL | **no**: placeholder URL | - |
| 2026-11-02 05:16 | `submissions/20261102T0516Z-d1-v4/track3_meshes_v4.parquet` (D1 v4, supersedes v3; 294,912 rows, 1,662,623 bytes; R9 compartment rack + thin-tool profiles; reproduced by `d1_meshes.sh --version v4 --run`; preflight Ready to submit) | `6663a1c7e0491c0cfa53b27fe36d7904455351dad548181e4e59a2263215af50` | `bc601cf474a8ccff3e77a7c10dd0025158b0c388` in the placeholder URL | **no**: placeholder URL | - |
| 2026-11-03 06:28 | `submissions/20261103T0645Z-d2-v4/track3_poses_v4_final.csv` | 7221dd3a... | `d1d1da5` (real URL, not yet pushed) | final once pushed | - |

### D1 v4: adopted R9 rack-and-tools rule (context 08)

Pre-score Amendment 2 of `docs/chain/mesh-selection-protocol.md` (commit `bc601cf`, 05:03:47 UTC;
first round-3 score 05:06:45 UTC) fixed the incumbent (packed v3; development R6/0.10 0.77315 cm
per object, 0.81675 pair-weighted), three parameter-free rules with video attributes in
`track3/features/mesh_extents/attributes-round3.json`, the spoon safeguard (R8 must pass alone
before R9 is eligible) and a stricter packed veto (per object < 0.74108, pair <= 0.81150, no object
+0.15 cm). The wall-thickness round was dropped: the best per-object choice of already-tested walls
would gain only 0.004 cm, and choosing thin pot walls now would be scan-informed (route call).

| Rule | Cross-fitted per object (cm) | Pair-weighted (cm) | Worst regression (cm) | Gates |
| --- | ---: | ---: | ---: | --- |
| R7 compartment rack | 0.72009 | 0.71472 | 0.0 | pass |
| R8 thin-tool profiles | 0.70576 | 0.76492 | 0.0 | pass (spoon safeguard met) |
| **R9 = R7 + R8** | **0.65270** | **0.66288** | 0.0 | pass; selected |

Exact packed veto (`outputs/mesh_eval/pairs/mesh-round3-veto/`): v4 0.66292148 pair-weighted /
0.65338396 per object vs v3 0.81149513 / 0.77107703; rack 1.368 -> 0.865, dust pan 1.054 -> 0.518,
sweeper 0.719 -> 0.582 cm per object; nothing else changed. Caveats: a third adaptive round on the
same ten scans raises selection optimism (size unknown); the spoon (10 of 36 leaderboard pairs)
changed from a box to a bowl-and-handle profile without any scan, so its leaderboard effect is
unknown; buried internal faces are 10.5 % of the rack's area (IDEAS 38).

### D1 v3: adopted R6 peg-rack rule (context 07)

Pre-score Amendment 1 of `docs/chain/mesh-selection-protocol.md` (commit `c7e2b56`, 02:55:35 UTC,
before the first round-2 score) fixed the incumbent (R2 at threshold 1.3: 0.92506 cm per object,
1.08768 pair-weighted, three seeds), the gates, an exact veto against packed v2, three nested rules
and the video attributes (`track3/features/mesh_extents/attributes.json`, each with frame evidence).
The lead had seen the public scan renders (abstract Figure 1) before the attributes were fixed;
the attributes are read from video frames, but the choice is not blind to the scans.

| Rule | Wall/radius chosen on A / B | Cross-fitted per object (cm) | Cross-fitted pair-weighted (cm) | Worst regression (cm) | Gates gain / object / pair |
| --- | --- | ---: | ---: | ---: | --- |
| R6 peg-rack (adopted) | 0.10 / 0.10 | 0.77315 | 0.81675 | 0.0 | pass / pass / pass |
| R5 hollow-vessels | 0.10 / 0.10 | 0.84555 | 0.95600 | 0.0 | pass / pass / pass |
| R4 rim-handle | - | 0.92061 | 1.07740 | 0.0 | fail / pass / pass |

Exact veto (each pair's own kit seed, 26/36 pairs, `outputs/mesh_eval/pairs/mesh-round2-v2-v3/`):
v3 **0.8115 / 0.7711 cm** (pair-weighted / per object) vs v2 1.0850 / 0.9257: passes; 19 pairs better,
7 unchanged, 0 worse. Wall thickness 0.10 is the largest grid value; thicker walls were not tried and
are not retuned now (no retuning after the veto). Selection among three nested rules on ten reused
scans is optimistic. The spoon keeps its box.

- Production build: `outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295/run.json` (request
  `scripts/track3/requests/d1/build-v3.json`), byte-identical to the veto build
  `20261102T030138.153550Z-118eeb04`; the v2 request still rebuilds its 47 meshes byte-identically.
- Pack: `submissions/20261102T0313Z-d1-v3/manifest.json` (reproduction log
  `.runtime/tmp/reproduce-d1-kq3jjpqk/`).
- Reproduce: `bash scripts/reproduce/d1_meshes.sh <commit-url> --version v3 --run`.

### D1 v2: adopted R2 vessel-handle rule

Fixed protocol: `docs/chain/mesh-selection-protocol.md`. Three rules were evaluated; R2 was
adopted after passing all three gates. These are development estimates against the ten public
scans, used only for scoring. No scan geometry, dimension or template enters a candidate.

| Rule | Threshold chosen on A → scored B | Threshold chosen on B → scored A | Cross-fitted per-object mean (cm) | In-sample mean (cm) | Cross-fitted pair-weighted mean (cm) | Worst held-out regression (cm) | Gain / object / pair gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R2 vessel-handle | 1.3 | 1.05 | 0.92572 | 0.92506 | 1.08742 | +0.10599 | pass / pass / pass |

The final build fixes the threshold at **1.3**, selected on all ten objects, and handle height
fraction at **0.6**. Its fresh local score is **0.9251 cm** over seed episodes 1/3/5; this fitted
score does not replace the cross-fitted adoption headline. `wooden_spoon` stays a box.

Evidence and run records:

- Build: `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/run.json` (v1's exact 23 measure runs).
- Legacy equivalence: `outputs/mesh_eval/d1-ctx03/v2-equivalence.json`; old/new box and ellipsoid
  builds have byte-identical vertex arrays, face arrays and PLY files for all 47 meshes each;
  the box rebuild also matches all 47 historical v1 meshes.
- Candidate equivalence: `outputs/mesh_eval/d1-ctx03/v2-mesh-equivalence.json`; all ten scanned
  meshes match the loop's R2/1.3 candidates byte-for-byte; all 36 pair copies match their object
  meshes; the spoon matches v1.
- Fresh score: `outputs/mesh_eval/d1-ctx03/v2-cdo.json`; selection evidence:
  `outputs/mesh_eval/rules/summary.md` and `ledger.jsonl`.
- Pack run: `submissions/20261101T1605Z-d1-v2/run.json`; SHA-256, commands and row checks:
  adjacent `manifest.json`; kit preflight: adjacent `preflight.log`, exit 0, **Ready to submit**.
  Warnings: no pose results supplied and Kaggle checks skipped. Packed row count and row order
  exactly match the kit mesh sample (294,912 rows); numeric values are finite and row ids unique.

Reproduce using the existing measure runs:

```bash
PYTHONPATH=. .runtime/linux-cuda/bin/python -m track3.features.mesh_extents build --request scripts/track3/requests/d1/build-v2-r2.json
.runtime/kit-env/bin/python scripts/track3/local_cdo.py --candidates outputs/mesh_extents/build/<new-run>/objects --out outputs/mesh_eval/<new-score>.json
.runtime/kit-env/bin/python upstream/v2d_submission_kit/v2d_submission_kit/tools/pack_reconstruction.py track3-meshes \
  --meshes outputs/mesh_extents/build/<new-run>/meshes \
  --sample upstream/v2d_submission_kit/v2d_submission_kit/data/track_3_mesh_sample_submission.parquet \
  --commit https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/874ff6793721faa0035a71788b8f5a22f9b4b2d3 \
  --out submissions/<new-stamp>-d1-v2/track3_meshes_v2.parquet
.runtime/kit-env/bin/python upstream/v2d_submission_kit/v2d_submission_kit/tools/preflight.py --track track_3 --skip-kaggle
```

### D2 v2: K1a merged-cluster placement on episodes 1 and 28 (context 04)

Protocol `docs/chain/pose-placement-protocol.md` (fixed and committed before scoring, amended
twice before the data each amendment uses existed, final decision recorded before packing).

| Stage | Data | Result |
| --- | --- | --- |
| Fixed gates (Amendment 1) | public train 0, 11, 12, 18, 21, 23, 26, 31; kinematic static placement, kit metrics | no candidate passes: AUC 0.2688 for every candidate (slot-1 frame-0 error 10-74 cm > 9 cm), oracle static 0.5319; K1a RPE 47.51 -> 36.80 cm, MPPE 26.66 -> 21.63; K2 no gain |
| Confirmation (Amendment 2) | fresh public 2 (27 unmeasurable: separate clusters) | K1a RPE 54.39 -> 21.79 cm, MPPE 28.29 -> 12.43, AUC/SP-SR/MP-SR unchanged: adopt, on one episode |
| Physical check | real evaluator, scored 1, 28, 30 | 1 and 28 pass (first pose 0.062 cm, drift 0.058 cm); 30 fails (1.47 cm, drift 17.0 cm) and keeps v1 |

Evidence: `outputs/pose_insurance/placement/{summary.md,confirmation.md,ledger.jsonl}`,
`outputs/pose_insurance/v2/{report.md,physical_checks.json,equivalence.json}`, pack manifest and
preflight log in `submissions/20261101T2315Z-d2-v2/`. Reproduce: `bash scripts/reproduce/d2_poses.sh`
(plan), `bash scripts/reproduce/verify.sh submissions/20261101T2315Z-d2-v2/manifest.json`.
