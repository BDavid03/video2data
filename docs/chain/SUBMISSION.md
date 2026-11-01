# Submission checklist

Kept current by every chain context (see `PROGRAM.md` section 1). Freeze: 4 November 2026,
17:00 EST (22:00 UTC). Package-ready target: 2 November 2026.

| Id | Deliverable | State | Evidence | Blocked on |
| --- | --- | --- | --- | --- |
| D1 | Track 3 mesh Parquet (CD-O) | **v2 packed, preflight "Ready to submit"** (context 03): adopted R2 vessel-handle rule, aspect threshold 1.3, handle height fraction 0.6; v1's exact video measure runs, 36 pairs. Cross-fitted per-object CD-O **0.92572 cm** vs v1 box 1.00764; cross-fitted pair-weighted **1.08742 cm** vs 1.25662; all three adoption gates pass. Fresh score of the final all-object fitted build: **0.9251 cm** (development estimate, public scans). The spoon keeps its v1 box and remains unmeasured | `submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet`; build `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/run.json`; `outputs/mesh_eval/d1-ctx03/`; packed-file details below | owner: GitHub push, then repack with the real URL (NEEDS-YOU 1, 2) |
| D2 | Track 3 pose CSV (five pose metrics) | **v2 packed, preflight "Ready to submit"** (context 04; supersedes v1): v1's static FlashCHORD entry with K1a merged-cluster placement on scored episodes 1 and 28 (lid stacked on the pot; pot stacked on the rack, at the merged frame-0 cluster instead of the 0.5 m default); the other 18 exports are v1's, byte-identical. Episode 30's K1a scene was physically rejected (the sweeper falls 17 cm) and keeps v1. Chosen under `docs/chain/pose-placement-protocol.md`: the fixed gates failed (slot-1 error 10-74 cm on all 8 train two-object episodes, beyond the 9 cm top AUC threshold, so AUC/SP-SR cannot move; v1 AUC 0.269 vs oracle static 0.532); K1a lowers RPE 47.5 -> 36.8 cm and MPPE 26.7 -> 21.6 on train, confirmed on fresh public 2 (RPE 54.4 -> 21.8, MPPE 28.3 -> 12.4; 27 unmeasurable) under a pre-registered post-result amendment. Expected effect: lower RPE/MPPE on episodes 1 and 28 only; AUC and SP-SR unchanged | `submissions/20261101T2315Z-d2-v2/` (`manifest.json`), `outputs/pose_insurance/placement/summary.md`, `confirmation.md`, `outputs/pose_insurance/v2/report.md` | slot-0 heading is unobservable from video (no camera-to-mocap transform), so slot-1 credit needs moving objects or a new idea (IDEAS 24, 27, 28) |
| D3 | Extended abstract PDF | **anonymous abstract, 2 pages, rendered and inspected** (context 04: poppler installed without root; qualitative mesh panel for pot, pitcher, rack replaces the bar chart; pose section updated for D2 v2 and the placement diagnostic) | `docs/report/track3/` (`bash scripts/reproduce/d3_report.sh --run` or `bash scripts/report/build_pdf.sh docs/report/track3/main.tex`; provenance table in its README) | - |
| D4 | Code at a GitHub commit URL | local git; this context's work is on branch `worktree-chain-ctx01` | `git log` | owner: GitHub remote (NEEDS-YOU 1) |
| D5 | One reproduction command per deliverable | `scripts/reproduce/`: plan-first `d1_meshes.sh`, `d2_poses.sh` (reads the newest D2 manifest), `d3_report.sh`, and `verify.sh <manifest.json>` (re-hashes every recorded file). D1 v2 manifest 1/1 OK; D2 v2 manifest 107/107 OK; D2 v1's manifest shows two source-hash mismatches only because context 04 refactored `scene.py` and `__main__.py` (geometry equivalence proven, `outputs/pose_insurance/placement/equivalence.json`) | `scripts/reproduce/README.md` | - |
| D6 | Packed files, preflight clean, ready for the owner to upload | D1 v2 and D2 v2 packed with placeholder commit URLs (not final); kit preflight `--skip-kaggle`: "Ready to submit" for both | `submissions/20261101T1605Z-d1-v2/`, `submissions/20261101T2315Z-d2-v2/` | GitHub remote for the real URL; upload is the owner's (NEEDS-YOU 1, 2) |

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

| Time (UTC) | File | SHA-256 | Commit | Final? | Scores (from the owner) |
| --- | --- | --- | --- | --- | --- |
| 2026-11-01 10:35 | `submissions/20261101T1035Z-d1-v0/track3_meshes_v0.parquet` (D1, 294,912 rows, 1.6 MB) | `f814da46bdaaa747012cc078d6ca29274070633ab4745bcbae5319fabc24b710` | `0f8619373f2953eada625e25aa53d98f0f06d11a` (branch `worktree-chain-ctx01`) in placeholder URL `https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/...` | **no**: repack with the real URL after the push | - |
| 2026-11-01 15:00 | `submissions/20261101T1500Z-d1-v1/track3_meshes_v1.parquet` (D1 v1, supersedes v0) | `a76143b933ba09453a02b8100f2fd15af39ab51ca1064b7dea274c0eff411a9b` | `93038d282478c8f7923425cdb2648ce8ef09d2c6` in the placeholder URL | **no**: repack with the real URL after the push | - |
| 2026-11-01 16:05 | `submissions/20261101T1605Z-d1-v2/track3_meshes_v2.parquet` (D1 v2, supersedes v1; 294,912 rows, 1,568,410 bytes) | `43fa6ac02353520aac4d019f7d6d45e0aab923f1c8f43d6cf03e30c242ba95c2` | `874ff6793721faa0035a71788b8f5a22f9b4b2d3` in `https://github.com/OWNER-PLACEHOLDER-not-final/video2data/commit/...` | **no**: requested placeholder URL; repack with the real URL after the push | - |
| 2026-11-01 16:17 | `submissions/20261101T1617Z-d2-v0/track3_poses_v0.csv` (D2 v0 insurance; 9,120 rows, 4,815,348 bytes; kit preflight `--results outputs/pose_insurance/results --skip-kaggle`: Ready to submit) | `f19c365a176daaaa21d18b8c164a202fc5b0224e258069c51ef033e3c47f7714` | `67036c321432792dbb7440db4e08122a5c8fde7f` in the placeholder URL (the slice itself is in `e133bde`; the final repack must use a pushed commit that contains it) | **no**: placeholder URL | - |
| 2026-11-01 17:03 | `submissions/20261101T1703Z-d2-v1/track3_poses_v1.csv` (D2 v1, supersedes v0; 9,120 rows; preflight Ready to submit) | `fa3b1c320128a196648e0a966eb81b8e5cb8e65cb8ad6b1de0a3fbbb4523c26b` | `aa7d5566a6af8596fbb85a91dac7885e94290f50` in the placeholder URL (slice code in `7bc1a36`; repack from a pushed commit that contains it) | **no**: placeholder URL | - |
| 2026-11-01 23:15 | `submissions/20261101T2315Z-d2-v2/track3_poses_v2.csv` (D2 v2, supersedes v1; 9,120 rows; K1a on episodes 1, 28; preflight Ready to submit; `verify.sh` 107/107 OK) | `6ea6530b5408d2d24a4bf9a4f038b99aa6da659d6af3faf2020dea23a2923741` | HEAD at pack time in the placeholder URL (`manifest.json: code_commit_url`); slice code committed after packing in the context-04 commits, so the final repack must use a pushed commit that contains them | **no**: placeholder URL | - |

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
