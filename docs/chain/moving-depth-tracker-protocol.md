# Feature-free depth tracker, revision 1

Frozen before tracking or label access, 2026-11-04 UTC. CPU development only.
The seal in `data/outputs/moving_reference/depth-track/protocol-seal.json` records
this file's SHA256 and UTC time. Do not amend this revision after its first run.
This is a reference-tracker experiment, not an RL rollout or challenge result.

## Population and admission

Only public train episodes 0, 11, 12, 18, 21, 23, 26, 31 and tuning 39, 42
may be inventoried, tracked or evaluated. Never access episodes 41, 43, 44,
scored labels, scan geometry or leaderboard results. No GPU, installs or uploads.
Use all frames of every available full-clip SAM2 object/hand + metric stereo cache.
Partial frame-zero FoundationPose inputs are inventoried but cannot establish
motion accuracy, hand exclusion or full-clip coverage and are not tracking trials.
Sparse stereo caches without SAM2 masks are missing inputs, not failed tracks.
No labels, label-derived poses or previous object trajectories initialize tracking.
Existing label-free camera BA poses and their held-out diagnostics may be reused;
background features in that camera solve are disclosed, but object fitting uses none.

Known cache inventory before freezing: only 39 has full masks and stereo (319
frames). Other allowed clips have sparse depth but no full SAM2 cache. Object
classes are pot/lid (0), cups (11), cup/rack (12), sweeper/pan (18), pitcher/cup
(21), wooden pieces (23), pan/rack (26), sweeper/rack (31), pan (39), frying pan
(42). No released clip in this population supplies a spoon: record that class
as unmeasured rather than substituting scored spoon episodes.

## Mesh, coordinates and fixed fit

Use byte-verified procedural meshes from the campaign's
`data/outputs/pose_insurance/combined-v5/body-frames.json`, first traced occurrence
of each class. Never load a dataset scan, scan-derived convention or scan points.
Record mesh identity and its asset coordinate system. For 39 use the two-object
campaign's `mini_dust_pan` mesh (episode 19's trace), not C's separately refitted
pan or C's old object track. No mesh scaling or shape fitting in this revision.

Frame zero: masked stereo median centre, table-up orientation, four PCA-based
quarter-turn yaw starts. Symmetric mesh axis is its largest coherent cap normal.
For non-symmetric objects use asset z as table-up. Centre height starts from the
masked cloud, without forcing contact. Later frames start only at the previous
written world pose, transformed into the current camera. No object features,
optical flow, keyframes, mocap seeds or motion clipping; no smoothing/interpolation.

Fixed seed 31. Area sample 8000 mesh surface points plus mesh vertices for
silhouette association. At most 160 uniformly indexed non-hand depth points and
180 contour points per direction. Optimizer image width 320; final silhouette
metrics use the original cached 960-pixel camera grid. Three outer correspondence
updates, SciPy least_squares (soft_l1, f_scale 1, max_nfev 20 each, diff_step 1e-4).
Silhouette term: bidirectional boundary distance via nearest contour associations;
model contour points under the dilated hand mask are excluded. Depth term:
point-to-triangle distances using the triangles attached to eight nearest sampled
surface points (an explicitly approximate nearest-triangle search). Residual scales:
3 optimizer pixels, 0.01 m depth, 0.05 m translation continuity, 0.35 rad rotation
continuity; normalize each data block by sqrt(number of residual scalars).
Continuity weights 0.1 each. No grid/sweep or label-guided parameter changes.

Blue/beige cups: two local swing parameters perpendicular to the declared mesh
axis; twist is free and parallel-transported from the previous pose. Solve position
and axis direction only. Lid: full SO(3), choose the representative among identity
and a pi local-axis rotation nearest the previous orientation. Other classes:
full SO(3) with the same rotation-continuity prior. Directed cup axis error is
acos of the axis dot product; do not make an upside-down cup equivalent.

## Frozen validity and label-free pass bars

A frame is valid iff camera observed AND at least 50 object pixels remain outside
a seven-pixel dilated hand mask AND at least 6 finite stereo object points remain
(after three-pixel object erosion) AND exact rendered mesh IoU outside those hands
>= 0.30 AND point-to-mesh median <= 0.015 m AND p90 <= 0.030 m AND pose is finite
with positive camera depth. Insufficient evidence is invalid; retain a finite
previous pose as fallback if the fit cannot be attempted, including that fallback
in speed/rate maxima. No feature-count predicate or initial-facing ambiguity gate.
A nonsymmetric wrong-facing fit must fail the later orientation evaluation.

Compute the campaign instruments with its unchanged `gates.evaluate` and its
symmetry-aware angular-rate amendment. Record the old correspondence/FB/support
checks as failed/unmeasured, never fabricate feature evidence. Quality bars stay:
valid fraction >= 0.90; IoU median >= 0.60, p10 >= 0.30; reprojection coverage
>= 0.90; all-frame speed <= 1.5 m/s; angular rate <= 540 deg/s. IoU is included
only on valid frames exactly as in the campaign. Also save raw all-frame IoU.
This experiment cannot grant campaign admission: independent camera/table/support
checks remain required and unchanged. `reference.npz:gates_passed` stays false.

## Sealing and labelled assessment

Track every ready clip first, seal its reference, tracking arrays, fit diagnostics,
mesh identity, settings, source hashes and all input hashes, then write the batch
`outputs-seal.json` BEFORE the evaluation handler can open a label parquet.
Evaluation refuses changed sealed files, settings, code or protocol. Only the
evaluation function may read public mocap, explicitly selecting the allowed
episode parquet and the tracked object by name. Match exact frame indices, check
video/label times, use visible finite nonzero unit-quaternion mocap only. Report
label availability separately. Every finite labelled frame is in the accuracy
denominator, including invalid tracker frames. Unavailable labels count against
label coverage, not as successful frames.

Use a single initial rigid alignment, `truth[0] @ inverse(prediction[0])`, as the
kit aligns initial poses. This removes constant world/body gauge and reports motion
accuracy, not absolute registration. Also report raw unaligned errors as diagnostics;
do not fit a transform over the clip or a scale. The first frame must have a valid
label; otherwise evaluation is blocked. Compare asset-frame cup/lid symmetries after
initial alignment: transport the declared axis through the initial prediction-to-label
body rotation, so a differing body convention cannot silently change symmetry axes.
No scan-derived frame mapping is inferred.

**Per-clip accuracy pass fixed now:** >= 0.90 label coverage, >= 0.90 of labelled
frames with position < 0.03 m, >= 0.90 with symmetry-aware angle < 30 degrees, >=
0.90 jointly below both, median position <= 0.02 m and median angle <= 15 degrees,
AND all the label-free quality bars above. Missing classes keep the overall
cross-class gate false. Every clip/class needed by the scored two-object campaign
must pass before a dated campaign revision may track scored clips once. This CPU
job does not touch scored clips or modify the running campaign. Any later recipe
change needs a new protocol and an independently fixed assessment; do not retune
this revision against its reported mocap outcomes.

## Outputs and missing inputs

Write only below `data/outputs/moving_reference/depth-track/`: inventory, editable
queue, protocol seal, per-clip run records/checkpoints/reference/tracking/reprojection/
fit diagnostics, append-only ledger, output seal, evaluation CSV/JSON and summary.md.
Resume only exact input/settings/source identities; skip completed sealed tracks.
The queue is a batch, not an agent-operated parameter sweep. Missing masks and
dense stereo/camera caches receive explicit per-object GPU follow-up commands;
record unavailable spoon coverage. GPU follow-up is a separate job after the
current campaign releases its lease, never started here.

## Evaluator correction and authorized tuning follow-up — 2026-11-04

The owner explicitly authorized an evaluator correction and at most three tuning
revisions (2..4), on episode 39 only, in the same session. Revision 1 remains FAIL.
Its original protocol and source files are preserved byte-for-byte in
`data/outputs/moving_reference/depth-track/frozen-v1/`, with a SHA256 manifest.
The original protocol text above is not edited; new entries are appended.

M1's actual evaluator is `data/outputs/moving_reference/ep39/sealed/sealed-m1-27.py`.
It matches arrays by exact frame index and equal length, then uses
`turn = mocap_q[0] * inverse(predicted_q[0])`,
`aligned_p = turn.apply(predicted_p - predicted_p[0]) + mocap_p[0]`,
`aligned_q = turn * predicted_q`. There is no time interpolation. The timestamp
mismatch is retained as metadata; removing this tracker's unsupported 1ms rejection
is an evaluator correction, not a change to tracking, fit validity or accuracy bars.
`evaluation-m1/evaluator-correction.json` freezes the script and both input hashes
before evaluation. Candidate C reproduces its saved M1 error arrays exactly in
position and within 1.1e-14 degrees in rotation. Revision 1's scores are unchanged.

Development is explicitly tuning, selected on 39, not an independent assessment.
A single queue fixes revisions 2..4. The handler appends each dated method entry
before its run, copies and hashes the whole protocol into that revision's output,
runs the full clip, seals it, and only then evaluates labels. It stops if a revision
passes the unchanged protocol bars and beats candidate C: higher joint <3cm/<30deg
fraction and neither median error larger. The 90% joint bar stays, even though
candidate C itself reaches only 78.68% under these new accuracy measures. No
holdout, scored tracking, GPU execution, new geometry, scale fitting or mask
annotation is authorized in this CPU follow-up.

Diagnosis before these revisions: frame 0 is invalid (depth p90 30.958mm); first
position error >3cm is frame 1. No known hand pixel is in the depth cloud; only
16.5% of 5,341 sampled >30mm residuals are near a hand, 64.5% outside projection,
and 4.0% behind the nearest surface by >30mm. These do not establish background
or hand contamination as the cause. The traced procedural mesh's canonical up
is 77.08 degrees from asset z; using asset z as table-up is an initialization bug.
Its dimensions also differ from C's refitted mesh; mesh bytes and scale stay fixed.

Trimming depth distances is permitted only in the optimization term. All revisions
use the same baseline full non-hand cloud and thresholds to measure validity;
trimming cannot hide residual failures. None consumes C's object trajectory or
scan-derived body convention during tracking. C is a saved evaluator baseline only.

## Tuning revision 2 — 2026-11-04T17:32:39.575550+00:00

Frame zero already fails; asset z differs 77.08 degrees from the traced canonical up. Initialize using the recorded own-mesh canonical rotation and mask handle-hole heading, with four yaw starts. Apply the existing label-free table gauge. Trim the largest 20% depth distances only in optimization, with soft-L1 scale 0.1 on normalized residuals; keep the full non-hand depth cloud for validity. Known hand pixels were already removed in revision 1. Mesh bytes, dimensions and thresholds unchanged.

Settings fixed before this run: `{"continuity_weight": 0.1, "depth_trim_fraction": 0.8, "distance_transform": false, "reason": "Frame zero already fails; asset z differs 77.08 degrees from the traced canonical up. Initialize using the recorded own-mesh canonical rotation and mask handle-hole heading, with four yaw starts. Apply the existing label-free table gauge. Trim the largest 20% depth distances only in optimization, with soft-L1 scale 0.1 on normalized residuals; keep the full non-hand depth cloud for validity. Known hand pixels were already removed in revision 1. Mesh bytes, dimensions and thresholds unchanged.", "revision": 2, "robust_scale": 0.1, "rotation_grid": false}`.

Same mesh, validity residuals, 90% accuracy bars and all campaign instrument thresholds. M1 evaluator correction applies unchanged. This is development on tuning 39 only. Stop early only if this revision passes AND has a higher joint success fraction than candidate C with both median errors no larger. No scored tracking or admission.

## Tuning revision 3 — 2026-11-04T17:33:33.057108+00:00

Add a bilinearly sampled silhouette boundary distance-transform term plus reciprocal contour distance to revision 2, replacing frozen nearest-boundary targets. Increase the continuity weight from 0.1 to 0.5 to discourage local depth/contour jumps. Retain canonical initialization and trimmed optimization; validity still measures all baseline non-hand depth points.

Settings fixed before this run: `{"continuity_weight": 0.5, "depth_trim_fraction": 0.8, "distance_transform": true, "reason": "Add a bilinearly sampled silhouette boundary distance-transform term plus reciprocal contour distance to revision 2, replacing frozen nearest-boundary targets. Increase the continuity weight from 0.1 to 0.5 to discourage local depth/contour jumps. Retain canonical initialization and trimmed optimization; validity still measures all baseline non-hand depth points.", "revision": 3, "robust_scale": 0.1, "rotation_grid": false}`.

Same mesh, validity residuals, 90% accuracy bars and all campaign instrument thresholds. M1 evaluator correction applies unchanged. This is development on tuning 39 only. Stop early only if this revision passes AND has a higher joint success fraction than candidate C with both median errors no larger. No scored tracking or admission.

## Tuning revision 4 — 2026-11-04T17:35:18.910038+00:00

Add a six-start plus/minus 15 degree local-axis rotation grid to revision 3 only when fitted p90 exceeds both 30mm and 1.5 times the preceding frame's p90, or IoU falls below 0.45. Select by the same label-free IoU plus median-depth score. No pose clipping, scaling, label initialization or parameter selection inside a run.

Settings fixed before this run: `{"continuity_weight": 0.5, "depth_trim_fraction": 0.8, "distance_transform": true, "reason": "Add a six-start plus/minus 15 degree local-axis rotation grid to revision 3 only when fitted p90 exceeds both 30mm and 1.5 times the preceding frame's p90, or IoU falls below 0.45. Select by the same label-free IoU plus median-depth score. No pose clipping, scaling, label initialization or parameter selection inside a run.", "revision": 4, "robust_scale": 0.1, "rotation_grid": true}`.

Same mesh, validity residuals, 90% accuracy bars and all campaign instrument thresholds. M1 evaluator correction applies unchanged. This is development on tuning 39 only. Stop early only if this revision passes AND has a higher joint success fraction than candidate C with both median errors no larger. No scored tracking or admission.

## Defect fix: rev 3 + init fix — 2026-11-04T17:48:33.527369+00:00

Defect fix, not a tracker revision: retain revision 3's mesh, optimizer and full-cloud validity. Use full_c.initialize on C's original canonical procedural mesh and full frame-zero stereo cloud, including C's fixed base height. Map that exact canonical pose through the campaign mesh's recorded coordinate_rotation; no new scan or labels. Do not optimize or select another start at frame zero. Verify the canonical pose equals C's saved forward frame zero before tracking. Subsequent frames use revision 3 unchanged. C's reference stays the baseline; all pass bars unchanged.

## Revision 5 hybrid regression on tuning 39 — 2026-11-04T17:50:49.355075+00:00

Resume the existing sealed C reference on 39, unchanged; it is 100% valid. Select C's exact pose/validity where valid, otherwise fit from C's last valid pose, with cups solving only position and axis. No fit may override a valid C frame; bridge results never alter C's tracker state/keyframes. No last-valid C pose means fallback-alone from C's frame-zero initializer, disclosed. Keep the frozen full-cloud residual validity and pass bars. Confirm equality of every array and reference bytes on 39. Train uses the unchanged campaign two_object_facing.reference_tracks; saved 39 instead uses its own historical C model tracker. These are two disclosed C implementations, not a claim that 39 tests missing-keyframe recovery.

## Revision 5 C baseline public 11 beige_cup — 2026-11-04T17:52:13.464455+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 11 beige_cup — 2026-11-04T17:52:48.635920+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 11 blue_cup — 2026-11-04T17:54:41.438070+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 11 blue_cup — 2026-11-04T17:55:24.225028+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 12 blue_cup — 2026-11-04T17:57:35.475434+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 12 blue_cup — 2026-11-04T17:58:22.462202+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 21 blue_cup — 2026-11-04T17:59:24.308640+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 21 blue_cup — 2026-11-04T18:00:11.538560+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 0 white_pot — 2026-11-04T18:00:17.737320+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 0 white_pot — 2026-11-04T18:01:52.909368+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 0 white_pot_lid — 2026-11-04T18:03:52.492528+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Global reprojection instrument correction — 2026-11-04T18:05:01.377621+00:00

The hybrid originally reported IoU outside its dilated fitting hand mask, while C reports raw-hand IoU. For global campaign instruments use the existing _object.render and _object.iou with frame hands unchanged for every method. Keep original sealed gates.json; write campaign-instruments.json and campaign-reprojection.npz as separate measurements. No pose, per-frame fit validity, symmetry rule, label correspondence or threshold changes. The same all-valid reference must have the same IoU under the same mesh/camera/mask. The correction script is frozen in reprojection-correction.json before measurement; summary reports corrected global instruments.

## Revision 5 hybrid public 0 white_pot_lid — 2026-11-04T18:05:19.683471+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 21 water_pitcher — 2026-11-04T18:06:49.323650+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 21 water_pitcher — 2026-11-04T18:07:39.532352+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 26 white_pot — 2026-11-04T18:07:45.624639+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 26 white_pot — 2026-11-04T18:08:22.708422+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 5 C baseline public 42 white_pot — 2026-11-04T18:09:40.873540+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 hybrid public 42 white_pot — 2026-11-04T18:10:48.145463+00:00

C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.

## Revision 6: position-only decisive test — 2026-11-04T18:57:03.383031+00:00

One fixed CPU recipe on every completed public train/tuning cache, episodes
0/11/12/18/21/23/26/31/39/42 only. No scored or excluded episode, scan geometry,
GPU, new mask, parameter sweep or pass-bar change. Episode 11 remains identity
ambiguous. Reuse each saved C's frame-zero own-mesh pose and recorded camera
trajectory; compute missing C baselines through the unchanged campaign reference
handler. Never use C's subsequent object positions in the position estimator.

For each frame erode the object mask by 3x3 pixels, exclude the existing hand mask
dilated by 7x7 pixels, and take finite depth_valid stereo points at z>0.05m.
Depth cut: retain points within two robust sigmas of median camera z, where sigma
is max(1.4826 * median absolute depth deviation, 0.001m). Position observation
is the coordinate-wise median of the retained points. N=30 retained points is
fixed before running. Fewer points carries the previous final output position;
if frame zero is unsupported, use C's frame-zero position.

Visible-surface correction uses only our existing candidate mesh, sampled at
40,000 area-weighted surface points with seed 6. Project those points at C's
frame-zero mesh pose and retain the nearest z sample per in-image rounded pixel.
The arithmetic mean of that frontmost visible cloud estimates its visible-surface
centroid. Project (mesh origin minus that centroid) onto its unit camera view ray,
fixing one signed distance per object before tracking. Each frame adds this fixed
distance along that observation's view ray before transforming into the recorded
camera's world. No scale, offset, ray sign or mesh choice is fitted to labels.
No visible mesh sample is an explicit failure, never a zero-offset substitution.

Orientation is the exact saved C frame-zero orientation held in world coordinates
through the clip. Apply a causal coordinate-wise median of the last five raw
world positions (startup uses the available prefix); unsupported frames append
the held raw position but preserve the preceding final filtered output exactly.
No other smoothing, reset or pose fit. The point-count observation flag is recorded
separately. Per-frame validity still requires the frozen full non-hand depth and
silhouette residual thresholds and camera observation; N=30 is an additional
requirement, not a replacement for fit validity. Preserve existing pass bars and
report campaign admission as false.

Freeze source/input hashes before tracking, seal every output and then the whole
batch before any new label access. Only then call unchanged kit_score on static,
C and position variants with identical own meshes, conventions, native object
order and common-world camera transforms; never filter invalid frames. Public
reference geometry remains evaluation-only and its host proxy limitation explicit.
Static holds C's saved frame-zero pose. Missing C is reported explicitly if its
unchanged handler cannot run. No method choice is made on these train outcomes.

Report each object's median centre error after mocap motion onset (>5mm for three
consecutive frames, same frozen audit definition). Primary medians use the kit's
registered body centres and shared scene alignment, replaying its unchanged array
functions from the exact scored Parquet rows and checking all four scalar metrics.
Also retain independent M1 initial-pose-aligned centre diagnostics for comparison.
A motionless object has no post-onset median (N/A), rather than a fabricated zero.
This is one disclosed position-only revision, not proof of the original pass bar.

## Revision 5 C baseline public 18 mini_sweeper — 2026-11-04T19:00:22.499407+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 18 mini_dust_pan — 2026-11-04T19:02:46.167472+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 31 mini_sweeper — 2026-11-04T19:04:11.315766+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 12 plastic_dish_rack — 2026-11-04T19:06:17.067131+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 26 plastic_dish_rack — 2026-11-04T19:07:41.243481+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 31 plastic_dish_rack — 2026-11-04T19:08:51.288571+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 23 wooden_piece_1 — 2026-11-04T19:10:53.610873+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Revision 5 C baseline public 23 wooden_piece_2 — 2026-11-04T19:12:06.427145+00:00

Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. Independent camera/table/support admission remains outside this experiment.

## Arm G: motion-gated revision 6 — 2026-11-04T19:30:45.662472+00:00

One fixed video-only gate per object on all18 objects in the same ten public
train/tuning episodes. No forced one-carried-object assumption. Set R=0.05m,
K=5 frames. The object is declared carried iff its sealed revision-6 world/body
position leaves radius R around its own frame-zero position for K consecutive
frames AND a hand is in contact on those same frames. Contact = intersection of
the raw rectified hand-slot mask with an 11x11-dilated object mask, divided by
min(raw object area, hand area), at least0.05; empty masks have contact0. G uses
the maximum contact across slots only for the binary gate, never an average of
hand positions. Decisions and first qualifying frame are frozen before labels.
Carried objects copy revision6's exact whole-clip poses; others hold C's saved
frame-zero body pose. No C-valid/pan substitution and no class-dependent rule.
The offline whole-clip decision is disclosed; orientation remains held initially.

For evaluation only, define the mocap carried/moving proxy using exactly R=5cm
and K=5 consecutive original mocap frames, based on body-origin displacement
from frame zero. Mocap has no semantic grasp/contact annotation: report this as
an operational translation proxy, not a labelled secure-grasp truth. Report TP,
FP, TN, FN and per-object agreement, both all18 and excluding ambiguous episode11.
Post-onset medians keep the previous >5mm/three-frame onset and kit-registered
centres, with all mocap-carried objects included even when the video gate misses.
Keep original quality/accuracy bars; no scored tracking or campaign admission.

## Arm H: separate-hand attachment — 2026-11-04T19:30:45.662765+00:00

Use exactly G's sealed video-only carried decision. For non-carried objects
hold C's frame-zero body pose. Recover SAM2 track_ids1/2 individually from original
mask NPZs, using track.masks_at's exact half-pixel resize and nearest-neighbour
rectification; verify recovered object mask and hand union equal cached arrays.
Slots are persistent SAM2 IDs, not asserted anatomical left/right identities.
Never average their points, centres or displacements.

Palm proxy = each hand mask's distance transform >=0.5 times its maximum interior
distance, excluding the raw object mask. On finite stereo-valid z>0.05m points use
the same median-z +/- two robust sigmas (1.4826 MAD, floor1mm) as revision6; require
30 retained points. A missing palm observation stays missing. Smooth each hand's
world centres with a causal median of supported observations within five frames;
the current frame must itself be supported. No hand/object scale or offset is
fit to labels and no new model, scan, video decode or stereo estimation is used.

On a G-carried object, confirm grasp when one slot has contact>=0.05 and supported
palm depth on all five consecutive frames, object revision6 displacement over
that window >=1cm, and the difference between its five-frame displacement and
that slot's displacement <=3cm. Choose the slot with highest mean contact in the
window (ties lower ID). Backdate grasp to the window's first frame g (offline
five-frame confirmation). Anchor object at revision6 position[g] and palm at its
smoothed position[g]. Through attachment write object_position[g] + palm[t]-palm[g],
with C's initial orientation held. Before grasp use the initial C pose.

Carry the previous output on a missing selected-palm observation. Release after
five consecutive frames of selected-hand contact loss, missing selected-palm
depth, OR five-frame hand/object displacement disagreement >3cm. At the confirming
release frame preserve the preceding output; hold it afterwards. Re-arm for a
new grasp only after every hand has been out of contact for five frames. Each
regrasp anchors to revision6's position at that grasp frame. Record selected slot,
ambiguous two-slot contact, missing depth, grasp/release frames and release reason.
This conservative test does not handle continuous handovers without a contact gap.

Freeze both entries, source/input identities and queue before the first new arm
run; seal all36 G/H trajectories and all18 carried decisions before any new label
access. Run the unchanged kit on G/H and compare with the sealed static/C/rev6
baseline rows using identical meshes, native object order, exact frame IDs and
camera-derived shared worlds. Replay all four metrics before publishing registered
centre medians. Reference scans/anchors remain evaluation-only public proxies.
New-arm per-frame validity uses the original full non-hand cloud fit residuals and
camera observation, plus availability of its current motion source. No lowered
threshold, gate waiver, label initialization, scored episode, GPU or background
process. Train/tuning selection is disclosed development, not an unbiased test.
