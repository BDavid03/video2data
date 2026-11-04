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
