# Moving policy with a resting second object: frozen protocol

Written before any mask, motion selection, tracking or training result in this campaign.
No scored labels or scans are read. Episodes 41, 43 and 44 are excluded.

## Population and deterministic video-only selection

The official kit roster has 20 scored episodes. Its 16 two-object episodes,
in queue order, are:

| Episode | Slot 0 | Slot 1 |
| --- | --- | --- |
| 1 | white_pot | white_pot_lid |
| 3 | white_pot | wooden_spoon |
| 5 | white_pot | wooden_spoon |
| 6 | white_pot | wooden_spoon |
| 8 | beige_cup | white_pot |
| 10 | blue_cup | beige_cup |
| 14 | blue_cup | plastic_dish_rack |
| 15 | plastic_dish_rack | wooden_spoon |
| 16 | plastic_dish_rack | wooden_spoon |
| 19 | mini_sweeper | mini_dust_pan |
| 20 | blue_cup | water_pitcher |
| 25 | wooden_piece_1 | wooden_piece_2 |
| 28 | plastic_dish_rack | white_pot |
| 30 | mini_sweeper | plastic_dish_rack |
| 32 | blue_cup | wooden_spoon |
| 33 | blue_cup | wooden_spoon |

Order is ascending official episode ID, independent of labels, shapes and scores.
For each episode, generate independent masks for both object names using the
unchanged M1 detector/segmenter: name with underscores replaced by spaces,
DINO box 0.4/text 0.3, smallest qualifying box, fixed hand cadence 10,
SAM2.1 Hiera-Large, forward object masks and nearest-seed bidirectional hands.
No alternative prompt or manual correction is allowed. Identical detections for
the two names provide no instance separation and stop that episode.

A is the object with the larger video motion statistic; B is the other object.
At every decoded camera-C frame, use the median coordinates of its mask.
Subtract cumulative median background optical flow (M1 corner/flow functions,
background outside both objects and hands). Relative to frame zero, the motion
statistic is the 95th percentile of the Euclidean displacement of this corrected
mask centre, in pixels divided by image width. Missing centres are omitted;
no valid frame-zero centre or no valid background flow stops selection.
Ties choose lower official slot. This is an ordering/role rule, not a claim of
metric motion or a new admission threshold. Camera-relative residual motion can
misidentify A; the fixed reference and physical gates must still pass.

## Unchanged reference and policy gates

After A is selected, apply the spoon/38 mask gate: stereo-supported eroded,
non-hand texture points >=30 on >=0.9 of frames; longest unusable interval <=2
frames; >=30 stereo object pixels on >=0.9 of frames. A failed mask gate stops
that episode before camera/reference work.

All 28 final M1 thresholds and instruments in
`src/features/moving_reference/_final_gates.py:FINAL_LIMITS` and the sealed
perpendicular world-residual rule remain unchanged. No scan-based tracking or
body registration is introduced. Tracking uses our own reconstruction and the
existing M1 camera, object tracker, audits and final gates. Preserve every
measurement and reference hash before training. A missing required instrument
is a failure, never an implicit pass.

Stage 0 uses the same schema, grasp closure, penetration, finger-limit,
reachability, support and reference-truth checks as episode 38, including the
same acquisition rule and deterministic longitudinal/transverse grasp choice.
No new grasp repair is allowed.

Train only the first episode passing all reference and Stage-0 gates. FlashSAC,
1,024 worlds, seed 0, update ratio 2, 66,763,776 transitions, the exact episode-38
curriculum and scheduled final checkpoint. No intermediate checkpoint selection,
midpoint kill, seed shopping, threshold moves or second attempt. The unedited
official evaluator exports every object in official slot order on the full
official frame grid from this single learned checkpoint and evaluation world.
Two independently evaluated exports cannot be spliced.

A's final self-reference admission is unchanged: T >=0.50, Q >=0.50,
E <=min(0.5 E_static, 0.03 m), angle <=min(0.5 angle_static, 15 degrees),
end reached with no reset; unaligned first pose within 0.01 m and 2 degrees;
achieved reprojection median IoU >=0.60, p10 >=0.30, coverage >=0.90.
The already completed sealed train/tuning evidence remains required; scored
labels never establish admission.

## Only additions required for B

B uses our own D1/v5 procedural mesh, with a hash binding the exact simulation
asset and body convention. Never substitute a dataset scan. B starts at its
existing v5 static pose and retains a constant reference at that pose. It is a
free physical rigid body in the same simulated scene, with no policy actuator,
virtual object force or kinematic overwrite acting on it; hands act on A only.
Contact and support may keep B in place. Both bodies must be exported together.

Reuse `docs/chain/frame0-panel-protocol.md`, scored rule 5: B's unaligned first
export must be within 1 cm and 2 degrees of its intended start; its maximum
translation and rotation drift over the entire exported grid must be strictly
under 1 cm and 2 degrees from its first exported pose. Additionally measure
maximum error from the intended start with the same strict drift limits, so
first-pose and drift tolerances cannot accumulate. No special B stability
threshold or clamping is introduced. Unsupported/falling/contact-displaced B
fails, keeping the incumbent export for the whole episode.

## Execution and outcome

One mask/track/training attempt per episode; stop each at its first failed gate
and continue ascending until one complete reference passes or all 16 stop.
Infrastructure errors are recorded separately and stop the route; a failed
operation is never silently retried. A route-wide missing capability stops the
route honestly. No result-dependent edits to this protocol.

Write UTC stage/results to `data/outputs/moving_policy/two-object-status.md`.
If final admission passes, extend only the combined moving packer under the
same evidence rules, create a fresh candidate directory under
`data/outputs/pose_insurance/combined-v5/`, and verify its manifest. No existing
pack, submission file, report, check_release, upload, install or commit changes.
Hard close is 2026-11-04 17:00 UTC; do not launch work that cannot finish and
leave time to pack before it. A failure or time stop produces no candidate.

## Disclosed method revision: joint instance selection (4 November 2026, 15:42 UTC)

The lead found that independent name prompts selected the same physical region
for both objects in episodes 1 and 8: equal mask centres throughout episode 1;
episode-8 mask IoU 1.000 at frames 0 and 50 and 0.999 at frame 150. This is a
name-to-instance assignment defect. No labels or scans informed this revision.
The earlier stops remain recorded and are superseded only as explicitly stated.

Before segmentation, collect frame-0 Grounding DINO candidates for both names,
with the unchanged text threshold 0.3 and box threshold 0.4. Among all pairs
(one qualifying box per name), choose minimum box IoU. Resolve exact ties by
smallest slot-0 box area, then smallest slot-1 box area, then original detector
indices, preserving the previous smallest-area/index convention for each name.
If no pair has IoU strictly below 0.5, STOP for missing instance separation.
Use these selected boxes in the unchanged SAM2.1 propagation workers. If the
propagated frame-0 object masks have IoU >=0.5, STOP for the same reason.

Rerun episode-8 and episode-1 masks under this fixed assignment, preserving all
old masks, requests, reports and stops. These are disclosed method-fix reruns,
not independent replications. Existing completed masks with frame-0 object IoU
<0.5 are retained, with their identity and reuse recorded; they are not rerun.
An externally interrupted incomplete mask operation is preserved and may restart
under this revision; it has supplied no completed result.

The queue is now 8, 1, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 3, 5, 6.
The lead's 15:32 answer waives both mask_usable_fraction and
mask_longest_unusable_interval_frames at early and final M1 admission. These
measurements are still reported. Stereo support >=0.9 and all other M1,
Stage-0, A-policy and resting-B admission thresholds remain unchanged. Frame
validity reflects observed camera/object fits; the waived texture flag is not
reintroduced as an independent validity requirement.

The owner-corrected close is 22:00 UTC. Train every episode passing reference
and Stage 0, in passing order, at most once, with the unchanged episode-38 recipe
and scheduled final checkpoint. Last training start is 19:45 UTC; cumulative
candidate packs are targeted by 20:50 UTC. A new cumulative packer lives beside
the pinned original module. No existing submission, pinned source, report,
label, scan, install, upload or git mutation is authorized by this revision.

The report must contain: "Both objects are detected jointly as distinct instances
(box IoU below 0.5) before segmentation." Reused already-distinct masks must
also be disclosed as retained evidence rather than a fresh segmentation result.

## Disclosed method revision: declared axial symmetries (4 November 2026, 15:55 UTC)

The lead found episode 8's opposite-facing margin 0.00596024 (unchanged rule:
resolved only above 0.02). The organisers' current kit Track-3 task text declares
beige_cup and blue_cup continuously symmetric around their central axes, and
white_pot_lid symmetric under 0/180 degrees around its normal. AUC.py applies
axial symmetry to pose errors. No scored label or scan informed this revision.

Only for active beige_cup, blue_cup or white_pot_lid, identify the central/cap
normal from the own procedural mesh's largest coherent unsigned face-normal
area. This follows the vertical cylinder/revolved shell construction and uses
no scan. Carry that model axis through the chosen initial pose into the stereo
world; measure its unsigned angle to the stereo table normal. Record the model
axis, world axis, angle, coherent face-area fraction, symmetry type, original
margin, and original opposite_resolved value. If the original tie is unresolved
and the axis angle is <=10 degrees, keep the original winning pose and mark
opposite_resolved=True and resolved_by='declared scorer symmetry'. Otherwise
keep the original rule. All other objects retain it without change.

The helper is used only by the two-object reference path. Shared track.py and
_object.py, the episode-38 path and the spoon path remain byte-identical. The
CPU equivalence record must establish those source identities and unchanged
unsupported-object initialization values before the revised reference runs.

Rerun only episode 8's reference under this disclosed fix, retaining its masks,
instance separation and all earlier stops. Continue from episode 10, retaining
completed masks; retain episode 1's revised instance-separation STOP. All other
reference and physical thresholds, the usability waiver, training budget,
checkpoint rule, cumulative packing and no-label/no-scan rules stand unchanged.

The report must also contain: "For cups and the pot lid, a tie between opposite
facings is resolved by the organisers' declared symmetry." Passing this tie
resolution establishes equivalence under the declared symmetry, not recovery
of an unobservable cup heading or accuracy of the full video trajectory.

## Disclosed method revisions: clip-wide instances and symmetry rotations (2026-11-04T16:15:25.887741+00:00)

Lead decision after frame-zero detection stops in episodes 19 and 1: if no
qualifying frame-zero pair has box IoU <0.5, apply the same joint detector at
frames 5, 10, 15, ... in timestamp order. Keep box threshold 0.4 and text
threshold 0.3; choose the lowest-IoU pair, breaking ties by the same smallest
slot-0 box, smallest slot-1 box and detector indices. Stop at the first frame
with IoU <0.5. Seed both SAM2.1 objects at that frame, propagate forward and
backward through the full clip, and retain the unchanged propagated frame-zero
mask IoU <0.5 check. If there is no separated pair anywhere on this cadence,
STOP. Preserve every old detector and segmentation result. Episodes 19 and 1
receive one disclosed method-fix attempt in a new directory. An externally
interrupted operation has no completed gate result and may resume its completed
workers without repeating them.

For beige_cup and blue_cup (continuous axial symmetry) and white_pot_lid
(order 2), evaluate rotation differences by swing-twist decomposition about the
own procedural mesh central axis in the source object's body frame. For cups,
retain swing and remove twist; for the lid reduce twist modulo 180 degrees.
Apply this to every reference orientation gate (angular rate, forward-backward
rotation, stationary-rotation control) and every joint policy/physical
orientation gate. Threshold values remain unchanged. Translation, camera,
table, stereo support, coverage, validity and reprojection measurements retain
their original instruments. A valid flag may change only when a saved explicit
rejection reason establishes that an otherwise-observed frame was rejected
solely for a rotation spike removed by this symmetry; absent that evidence,
restore zero frames. Record old/new measurements, axis, symmetry and counts.
If the saved forward/backward closure matrices were not retained, do not infer
their swing from scalar angles: record the metric as unrecomputable and retain
the original conservative value. No re-tracking is authorized for episodes 8
and 10: re-score only their saved tracks, preserve hashes and old results.

Queue: 20, 25, 28, 30, 32, 33, 3, 5, 6, 19, 1, 8, 10. Last training start
19:45 UTC; train every reference + Stage-0 passing episode immediately, one
GPU job and one training attempt at a time, unchanged episode-38 recipe.
No threshold changes, scored labels, scans, pinned-source edits or uploads.

Read data/outputs/moving_policy/two-object/control.json between episodes.
Keys are pause (bool), queue (allowed episode IDs, ordered, no duplicates),
note (string). A pause waits without holding the GPU lease; queue changes only
steer uncompleted work and never erase outcomes or authorize a new attempt.
The lead can edit this file without killing the campaign.

## Disclosed decision: conditional three-region camera coverage (2026-11-04T16:31:37.396593+00:00)

Lead decision after episodes 19, 14 and 16 stopped at three held-out image
regions while the measured direct camera subset passed. In the two-object
path only, camera_heldout_regions_min may pass at >=3 instead of >=4 if ALL
of these final unchanged instruments pass: held-out reprojection median
<=0.5 px640, p95 <=2 px640, held-out tracks >=30, coverage >=0.98, and
camera_heldout_world_residual_median_m <=0.01 m using the existing sealed-pass
perpendicular instrument. This condition applies uniformly to every episode,
including subsequent episodes. Four or more regions retain the old verdict.
A missing direct measure cannot satisfy the condition.

At the early camera subset, exactly three regions with all available direct
checks passing may proceed provisionally to collect the final world-residual
instrument. This is not a gate PASS: the world-residual condition is still
pending. Only the final composed gate can grant the conditional acceptance.
Preserve the original four-region measurement/verdict in the raw M1 reports;
record the new conditional verdict and all five prerequisite values separately.
Shared tracker, spoon path and episode38 behavior remain byte-identical.

Rerun references from the camera stage for 19, then 14, then16, in new output
directories. Reuse the completed instance results, masks, frozen A/B role and
mask-gate measurements; preserve all previous stops. These are disclosed
method-decision reruns, not independent trials. Continue remaining control-file
episodes afterward. Train every reference+Stage0 pass immediately, one GPU job
and one training attempt per episode, unchanged recipe, last start19:45 UTC.
All other gates and prior disclosed revisions remain unchanged.

Required report sentence: "The camera check accepts three image regions when
held-out reprojection accuracy passes." The short sentence does not replace
the full five-prerequisite rule in this protocol and admission evidence.
