# Pose silhouette diagnostic S1: break near-tied registration basins by mask outline (IDEAS 51; written before any silhouette number; commit time is the record)

Route: `outputs/chain/codex/*route-sil-19.answer.md` (Codex, read-only; no projection or overlap computed).

## Purpose and disclosure

HR2's trimmed depth ICP cannot separate the mini sweeper from its 180-degree flip (rival 2.06 % worse on train 18,
3.44 % on scored 19), and H3's flip gate G2 passed by only 0.00077 because of that coin flip. S1 tests whether
agreement between the scan rendered at each saved basin and the frame-0 camera-A object mask separates near-tied
basins, without refitting anything.

Labels and errors on the known-answer cases are spent: on episode 18 the selected sweeper basin is correct relative to
its accepted dust pan and its rival is the flip; the selected wooden pieces in 23/24 are inverted in absolute up while
their relative rotation is close. Episode 19 is label-free. Earlier mesh silhouette-proxy results (IoU not validated
as a CD-O proxy) are known. This is prospective diagnostic rejection evidence on spent cases, not an untouched
validation set and not a calibrated flip probability.

## Inputs (frozen)

Registration trials, by exact directory (never a `latest` pointer): HR2 `outputs/pose_insurance/registration-hr2/trials/`
`20261103T112911.312560Z-c6cf948f` (18 sweeper), `20261103T113226.572792Z-041aa9e0` (18 pan),
`20261103T115050.799956Z-3bcea8bd` / `20261103T115403.973957Z-3ca3b440` (23 wooden pieces 1 / 2),
`20261103T115627.877075Z-3bb2c1de` / `20261103T115942.452435Z-d33e43ff` (24 wooden pieces 1 / 2),
`20261103T111232.770514Z-4b289c63` (39 pan), `20261103T112312.414779Z-40655a67` (12 rack),
`20261103T114035.178074Z-572cd4e6` (21 pitcher), `20261103T120807.011747Z-fbb78022` (27 pot),
`20261103T121947.451604Z-6e9beb58` (42 pot); H3 `outputs/pose_insurance/registration-h3/trials/`
`20261103T130445.626124Z-878a4f5c` (19 sweeper), `20261103T130707.891203Z-5122ba19` (19 pan).
Each trial's `run.json`, `prediction.json` and `registrations.npz` are re-admitted and hashed, as are the scans, mask
label files, assignments, depth preprocessing grids, this protocol, the queue and the source code.

Frame 0, camera A. Stored transforms are scan to camera A: `p_A = R p_scan + t`. Intrinsics are
`input_intrinsics_pixels` from each trial's recorded depth producer `preprocessing.npz`. Masks are the recorded
`outputs/mesh_extents/.../labels_000000.npz` cluster unions (`label == cluster_id + 1`, the cluster ids the trials
used) on the 576x768 undistorted grid. They are depth-derived, height-filtered (>= 8 mm) regions, not complete RGB
segmentations; they are never intersected with stereo validity. Scans: the path each trial recorded, loaded as the HR2
loader does (scene transforms applied; no recentring, rescaling, collision substitution or component splitting).

## Execution

Pure array functions in `track3/features/pose_insurance/_silhouette.py`; a CPU handler, resumable loop and queue in the
same slice, writing immutable run records, candidate tables, predictions, a ledger and `summary.md` under
`outputs/pose_insurance/silhouette/`. All 13 object predictions are saved before any label is read; labels are read
only in a separate assessment step. Frozen HR2/H3 code, methods, queues and artifacts are not modified.

## Candidates

For each of the four searches independently: keep final rotations with loss <= 1.05 x that search's minimum; sort by
loss, then index; greedily keep representatives, removing rotations within 30 degrees of a kept one. Search 0 defines
candidate identities, its depth-best included. Representatives of searches 1-3 must map one-to-one onto search 0's
within 10 degrees; a missing, extra or ambiguous match makes the object abstain. Each representative uses its own
saved translation; nothing is refitted, recentred or rescaled. Fewer than two candidates: not testable.

## Rendering and decision

Transform scan vertices by `R, t`, require finite values and positive depth for every vertex, project with the pinhole
`K`, round to the nearest pixel, and fill each triangle separately (OpenCV `fillConvexPoly`, LINE_8) into one binary
union; no back-face culling, alpha or hull. Empty masks or renders, invalid intrinsics, and a render or mask touching
the image border reject the object (abstain); nothing is repaired or dropped. Score on the full grid, no crop.

Primary score: IoU with the mask. The same score is taken with the mask eroded and dilated once (3x3 square kernel);
no other cleanup. Margin m = 0.05 absolute IoU (an engineering safeguard, about 150-190 pixels at these mask areas, not
a calibrated threshold). A candidate is selected only if it beats every other candidate by >= m in all three mask
variants and all four searches, with the same matched identity; otherwise abstain.

Descriptive only, fixed now and never used to decide: symmetric outer-contour chamfer, distances capped at 8 px, in
pixels, per candidate and mask variant.

## Known-answer assessment (after predictions are saved)

- **18 sweeper:** pan fixed to its recorded search-0 selection; a candidate is correct if the geodesic error of
  `R_sweeper^T R_pan` against frame-0 mocap C is < 30 degrees; exactly one correct candidate identity is required,
  otherwise not testable.
- **23/24 wooden pieces:** correct means `angle(R^T n_table, R_ref^T e_z) < 15` degrees (uprightness, not yaw), with
  frozen name/scan/mask assignments and no label permutation. The classification must agree with
  `angle(R^T n_table, signed canonical up) < 15` degrees (+x piece 1, -x piece 2); disagreement is identity-uncertain,
  so not testable. Several upright yaws may all be correct.

All five cases are reported with reasons; the roster is never shrunk. **Pass:** 18 sweeper selects its depth-best
basin with margin, and no testable case selects an incorrect candidate. Abstentions are reported separately; no
wooden-piece decision means no evidence of resolving wooden flips.

## Controls

Accepted objects: pans 18/19/39, rack 12, pitcher 21, pots 27/42. For each, compare search 0's depth-best with the
basin representative farthest from it (any loss; ties by lower loss, then index), matched one-to-one within 10 degrees
in each repeated search, with the same margin and mask variants. Any robust preference for the far basin in any control
vetoes S1. A missing comparison makes the control panel incomplete, which is not a pass. Controls test consistency
with clear depth fits, not truth.

## Episode 19 and what S1 changes

Only if the known-answer test passes and the controls do not veto: apply the frozen decision to the episode-19 sweeper
and report agree (selects H3's depth-best), disagree, or abstain; otherwise report "application withheld". No episode-19
label exists or is read. H3's verdict is unchanged; S1 changes only the risk the owner weighs for D2 v5, recorded in
`docs/chain/NEEDS-YOU.md`. It changes no submission, rotation, approval or pack by itself.

Episodes 41, 43 and 44 are never read. No leaderboard feedback, no outcome-driven threshold change; any other metric is
descriptive and needs its own protocol before scoring.
