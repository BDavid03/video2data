# Refresh 20261104T2048Z: frame-zero placement limits calibrated on train/tuning before any scored decision

Written 2026-11-04 20:48 UTC, before any registration of this refresh ran. The dataset and kit were re-downloaded and are
byte-identical to the pinned snapshot (`data/dataset-refresh/20261104T203500Z/diff.md`).

## Inputs
- Train/tuning two-object episodes with public scans for both objects: 0, 11, 12, 18, 21, 23, 26, 31 (labels read only
  for validation on these episodes). Episodes 41, 43, 44 are never read. Scored episodes: the saved registrations of
  8, 10, 14, 19, 20, 25, 30 (`data/outputs/pose_insurance/frame0-fp/corrected/ep*/corrected-placement.json`), unchanged.
- Frame-zero masks: Grounding DINO boxes per object prompt on the raw ego_cam_c frame 0 (box 0.4, text 0.3), assigned
  JOINTLY: among all combinations of one box per object with every pairwise box IoU < 0.5, choose the highest total score,
  ties by smaller total area. SAM2.1 image predictor from the chosen boxes. Stereo depth: calibrated rectification and SGBM
  (`moving_reference.camera`), the same instrument as the scored registrations.
- Registration: the unchanged FoundationPose adapter via `src/features/pose_refresh/frame0_fp_worker.py` (seed 0).

## Validation of a train registration (labels used here only)
A pair is VALID when the FoundationPose B-in-A relative pose (scan frames) agrees with the label relative pose at frame 0
within 2 cm offset and 10 degrees after the declared symmetry (cups axial, lid 180 degrees), the same tolerances as the
rollout agreement rule. Otherwise INVALID.

## Limit rule (fixed now)
For each instrument (silhouette IoU, higher is better; median masked depth residual, lower is better; symmetry-collapsed
distinct-rival relative margin of object B, higher is better), the limit is the threshold on the train pairs that best
separates VALID from INVALID pairs (maximum accuracy over the observed values), ties broken toward the stricter value; a
pair counts as passing an instrument when both objects pass it. If every pair is VALID the limit is the worst observed
value among them. A scored episode is adopted when all three limits pass, then trained with the frozen v5 static recipe
(seed 0, 512 steps, 1 world, replay minimum 32), rolled out through the kit's evaluator, and admitted only under the
unchanged physical guards (first pose within 1 cm / 5 degrees, displacement within v5 + 1 cm, rollout B-in-A within 2 cm /
10 degrees). Unadopted episodes keep their moving-v5 rows byte-for-byte. No limit changes after a scored result.

## Not in this refresh (time)
Full-clip mask rebuild, stereo depth and camera for all frames, procedural mesh rebuild, train-episode RL rollouts, and the
kit-score table of held frame-zero poses are not run; they remain open items.
