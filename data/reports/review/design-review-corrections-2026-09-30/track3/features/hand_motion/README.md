# Hand motion

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

This slice estimates the pose of both hands in every frame and places them in the
metric world. The robot's reference motion is retargeted from these hands, so they
decide how the robot approaches, grasps and moves each object. `RG/` means
`upstream/video_to_data-flashchord-aee95f2/robotic_grounding/`; the source reading
behind the upstream facts is
[upstream-retargeting.md](../../../reports/contract-evidence/2026-09-30/upstream-retargeting.md).

## Role in scoring

No leaderboard metric measures hands. They matter through the policy: a hand
reference that reaches the wrong place, or passes through the object, trains a
policy that knocks the object away or never lifts it, and the object trajectory is
what is scored.

## What the upstream retargeter requires

All **Official**, from the pinned upstream source.

| Requirement | Source |
|---|---|
| Per hand and per frame: 21 joint positions in the world, metres, in MANO order (wrist, then four joints for each of thumb, index, middle, ring and pinky) | `RG/source/robotic_grounding/robotic_grounding/retarget/params.py:7-29` |
| Per hand and per frame: 21 joint orientations, scalar-first quaternions, in MANO's joint-frame convention. The wrist and the five fingertips are active orientation targets. | `retarget/params.py:191-219`, `retarget/hand_kinematics.py:236-251` |
| Both hands on every frame. There is no validity field and no single-hand mode. | `RG/scripts/retarget/ego_recon_to_sharpa.py:307-312,521-535` |
| A Z-up world with gravity along negative z | `ego_recon_to_sharpa.py:227-243`, `retarget/support_recon.py:39,480-491` |

The MANO parameters themselves are carried through and never read by the
retargeter. The joint orientations are the hard part: upstream gets them by
running MANO forward kinematics in a separate loader, because the MANO model files
are licence-gated and the library that runs them is GPL-3.0
(`RG/workflow/data_pipeline.md:43-59`).

## Operations

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `estimate` | Per-frame hand detections and MANO parameters in the camera frame | Perception environment | Yes |
| `kinematics` | 21 joint positions and orientations per hand from the MANO parameters | Isolated Windows MANO environment at `.runtime/mano/` | No |
| `lift` | The same joints placed in the metric world | Perception environment | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `ego_cam_a` clip on the pinhole grid | Dataset, [depth](../depth/README.md) | Colour frames, intrinsics |
| `outputs/object_masks/.../frames` | [object_masks](../object_masks/README.md) | The two hand layers, to seed and check detection |
| Depth in metres | [depth](../depth/README.md), [stereo_depth](../stereo_depth/README.md) | Depth on hand pixels |
| `outputs/camera_motion/.../trajectory.npz` | [camera_motion](../camera_motion/README.md) | `world_T_camA` |
| `models/hand/models/MANO_RIGHT.pkl`, `models/hand/models/MANO_LEFT.pkl` | The user, from the MANO site | Read by `kinematics` only |

## Method

1. **Detect and assign sides.** Find up to two hands per frame and label each
   left or right. At most one hand per side is kept. A side's track is continued
   by position between frames, so a detector that flips a label for one frame
   does not swap the hands.
2. **Regress.** For each hand crop, a hand-pose network gives MANO wrist
   orientation in the camera frame, 15 finger joint rotations and a shape vector.
   Networks of this family store left-hand parameters in mirrored right-hand
   form (`RG/scripts/retarget/ego_recon_to_dexmate_sharpa.py:313-326`). The output
   records which form it holds for both wrist and finger rotations. `kinematics`
   un-mirrors both together once, using the same convention. The final
   `hands.npz` contains both in anatomical, un-mirrored form.
3. **Fix the shape.** Use one shape vector per side for the whole episode, the
   median over confident frames. A hand does not change size during a clip.
4. **Run forward kinematics.** From the parameters, compute the 21 joints and
   their orientations in MANO's convention, relative to the wrist.
5. **Place in depth.** A network estimating from one crop is reliable in image
   position and unreliable in distance. Keep the image position. Set the distance
   so the visible hand surface agrees with metric depth on the hand's mask
   pixels, using a robust fit over those pixels.
6. **Lift.** Transform to the world with `world_T_camA` only where the camera
   pose and the hand estimate are valid.
7. **Mark validity.** A hand is valid in a frame when it was detected, its score
   is above the floor, and enough of it is inside the image. Everything else is
   invalid and left as NaN. `world_valid` also requires a valid camera pose; world
   joints and rotations are NaN where it is false. [scene_assembly](../scene_assembly/README.md) decides
   what an absent hand does, because upstream needs both hands on every frame.

**Without MANO (Design fallback).** If the MANO files are not available, joint
positions can still come from a landmark model, and the wrist and fingertip
orientations can be built geometrically: the wrist frame from the wrist and the
index and pinky knuckles, each fingertip frame from the direction of its last
bone. The construction has to reproduce MANO's axis convention, which would be
written from the published model description and checked only by looking at
retargeted hands. It is a fallback because a convention error there is silent.

## Outputs

`outputs/hand_motion/<split>/episode_XXXXXX/<run-id>/hands.npz`

| Field | Type and shape | Meaning |
|---|---|---|
| `sides` | string 2 | `["left", "right"]`, the index into every array's second axis |
| `valid` | bool T×2 | Camera-relative hand estimate satisfies step 7 |
| `world_valid` | bool T×2 | Hand `valid` AND camera_motion `valid`; mask for world joints and rotations |
| `joints_world` | float64 T×2×21×3 | Metres, MANO joint order |
| `joint_rotations_world` | float64 T×2×21×3×3 | MANO joint-frame convention |
| `wrist_orient_camA` | float32 T×2×3 | Axis-angle, anatomical and un-mirrored after `kinematics`; same form as `finger_pose` |
| `finger_pose` | float32 T×2×45 | Axis-angle, anatomical and un-mirrored after `kinematics`; same form as wrist orientation |
| `betas` | float32 2×10 | One shape per side |
| `score` | float32 T×2 | Detector score |
| `in_frame_fraction` | float32 T×2 | Share of the 21 joints that project inside the image |
| `depth_support` | int32 T×2 | Hand pixels with valid depth used in step 5 |
| `frame_index`, `pts` | int64 T | Frame identity |

`run.json` records the hand model, the orientation source (`mano_fk` or
`geometric`), the regressor's input parameter form and the output parameter
form (`anatomical_unmirrored`). It records whether the joint un-mirroring of
wrist and finger rotations was applied; already anatomical input is not converted
again. Intermediate regressor output preserves both rotations as regressed in
the same declared form. Intermediate kinematics output records the conversion
and camera-relative joints; these files pass between the two Windows processes.

## Configuration and authorization

`runtime.json` holds `estimate_authorized`, `kinematics_authorized` and
`lift_authorized`, all false.

## Dependencies

Proposed, not installed. Each needs the user's confirmation, and the first two
need the user's own action.

| Item | Licence | Note |
|---|---|---|
| MANO model files | MANO licence; registration and download by the user | Never committed, never redistributed |
| A MANO forward-kinematics library | `manotorch` is GPL-3.0 | Kept in its own environment and called as a separate process, as upstream does |
| WiLoR weights and code | Model under CC BY-NC-ND | 8.66 GB with the HaMeR archive, publisher figures in `reports/model-readiness/README.md` |
| The hand detector used by WiLoR | Ultralytics terms apply | Needs a licence decision |

The challenge's release terms exempt pretrained models under incompatible
licences from the release requirement (**Official**, `KIT/config/tracks.json:335`).
Whether non-commercial models may be used at all is the organisers' call.

## Resources

One GPU stage at a time through the shared lock. Unmeasured.

## Failure handling

An episode in which a side is never detected is not a failure: the side is
recorded as absent throughout. An episode in which no hand is ever detected fails
with `no_hands`.

## Separation of public labels

The public release has no hand labels. Nothing to separate.

## Acceptance evidence

Once runs are authorized: overlays of the 21 projected joints sit on the hands at
approach, grasp and release; fingertip positions at grasp lie within a centimetre
or two of the object's surface; the hand's distance from the camera changes
smoothly.

## Open questions

| Question | Who can answer |
|---|---|
| Will the user register for and download MANO? | The user |
| May models under non-commercial licences be used? | Organisers |
| What did the organisers' own baseline do with frames in which a hand is out of view? | Organisers |
