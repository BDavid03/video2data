# Pose heading protocol HR2: registration for non-cup objects (IDEAS 49; context 16, written before any HR2 number; commit time is the record)

## Why

`docs/chain/pose-heading-protocol.md` measured a train oracle ceiling of +0.076 AUC over T1 for the true frame-0
relative pose, and found that its registration HR1 failed: cup heading is not recoverable from stereo shape, and the
non-cup objects were never represented (up set {+z, -z}, `<name>.glb` loader, 9 cm crop). A pair can only gain if
BOTH slots get a heading (a cup's unknown yaw rotates b and C together). Clean train pairs with two non-cup objects:
18 (mini_sweeper / mini_dust_pan) and 23 (wooden pieces), whose oracle gains are +0.275 and +0.140 AUC (+0.052 on the
eight-episode mean). Scored pairs that HR2 could change in practice: 19 (sweeper / dust pan) and possibly 25 (wooden
pieces); spoon pairs (no public scan), merged pairs (1, 28) and 30 (no slot-0 cluster) cannot. Route:
`outputs/chain/codex/*route-hr2-16.answer.md`.

## Disclosure (what is already known)

All train labels were read in Stage A (up axes in scan frames: rack, dust pan, pot, lid, wooden_piece_1 +x;
wooden_piece_2 -x; sweeper +z; cups -z; pitcher tilted; pot inverted in 26). Episodes 12/18/21/23 carry spent HR1
errors. Validation 22/24/27 were spent on the length confirmation; this is their first use for heading. So the
evaluation below is prospective rejection evidence for a frozen method, not an untouched holdout.

## Method HR2 (fixed in outline now; parameters frozen in a method file before evaluation)

Reuse HR1's observed-to-model trimmed ICP with Kabsch refinement, with these changes only:

- Scan: `<name>.glb`, else `<name>_visual.glb`; never collision geometry. Surface samples as HR1.
- Starts: six signed scan axes as up candidates x 5-degree yaw, unrestricted rotational refinement; rotations compared
  geodesically across starts (not by a scan-axis yaw).
- No cup-sized crop: the observed cloud is the object's stereo-valid mask points above the table.
- Decline if: mask or scan missing; another final rotation more than 30 degrees away within 5 % of the best loss; or the
  best rotation moves more than 10 degrees under fixed repeated point subsampling.
- Development on tuning episode 39 (mini_dust_pan, single object) only; tuning labels may be used there (up error vs
  the table normal). Parameters and decline thresholds are frozen in `docs/chain/pose-heading-method-hr2.json`, committed
  with its sha256 in an amendment BEFORE any evaluation episode is registered.

## Evaluation (run once, unchanged, after the freeze; predictions saved before any label is read)

- **Pair panel 18 / 23 / 24 / 27** (both slots non-cup with independent masks). For each pair with both slots accepted:
  angle between predicted and mocap frame-0 b as full 3D vectors, and geodesic C error.
  **Pass:** 18 and 27 both accepted with b < 30 degrees and C < 30 degrees, and every other accepted pair also under
  both. A declined pair does not pass.
- **Up panel**: every non-cup slot with an independent mask in 12/18/21/23/24/27/42 (about 11 slots): up error =
  angle between `R_est^T n_table` and `R_ref^T e_z`. **Pass:** at least 6 accepted over at least 4 episodes and
  3 object types, every accepted up error < 15 degrees. (Tilt only, not yaw.)
- HR2 passes only if both panels pass. If either fails, stop: D2 v4 stands, and the result is recorded.

## Candidate H2 and train gates (only if HR2 passes)

H2 = T1 with HR2 rotations on pairs where both slots are accepted, box/collision geometry and inertia counter-rotated
so the physical placement is unchanged (as `_placement.py` does); every other pair keeps T1. Train gates vs T1 on the
eight train episodes: mean AUC gain >= +0.010; RPE and MPPE not worse; SP-SR and MP-SR not lower; worst single-episode
AUC loss <= 0.02. If H2 clears them by 06:00 UTC on 4 November it goes to the scored stage and physical check (first
pose <= 1 cm and 2 degrees) as D2 v5; otherwise D2 v4 stands.

Episodes 41, 43, 44 are never read; no leaderboard use.

## Amendment 1 (context 16, before any evaluation episode was loaded; commit time is the record): HR2 frozen

Phase 1 registered tuning episode 39 only (`outputs/pose_insurance/registration-hr2/phase1-report.md`): dust pan
accepted, up error 6.2 degrees, distant rival basin 26 % worse, subsampling shifts <= 1.9 degrees, 434 s CPU per
object; all eight non-cup scans load. Method frozen as `docs/chain/pose-heading-method-hr2.json` (sha256
c1fd86776f0a5e39eaf0f79273cbb8c6098ef585b59783b00a30ef1d65772e25); code `_registration.py`,
`registration_loop.py` at this commit. The evaluation roster is `track3/features/pose_insurance/registration_eval_queue.json`
(11 slots: 12 rack; 18 sweeper, dust pan; 21 pitcher; 23 and 24 wooden pieces; 27 rack, pot; 42 pot). Phase 2 runs it
once, predictions saved before any label is read, then applies the pass rules above unchanged. Note: the pair panel's
wooden-piece identities in 24 follow the recorded confirmation assignments.
