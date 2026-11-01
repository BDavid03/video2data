# Object and hand masks

Status: design only. No code exists for this slice. Nothing was run while writing
this document.

This slice finds each tracked object and each hand in the colour video and follows
it through the clip, producing one mask per instance per frame. Every later
perception slice depends on it: masks tell camera motion what to ignore, tell mesh
reconstruction and pose tracking what to look at, and tell hand tracking where the
hands are.

## Role in scoring

Indirect. A mask that drifts onto the wrong object produces a confident, wrong
pose track, and the scorer has no way to know. Object identity matters most:
slot 0 of each episode anchors the scorer's alignment
(**Official**, `KIT/v2dlb/object_pose.py:122-163`), so confusing slot 0 with slot 1
corrupts every metric for that episode.

## Operations

Disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `generate` | Per-frame instance masks for one episode | Perception environment | Yes |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `ego_cam_a` clip | Dataset | Decoded frames and integer MP4 PTS |
| `outputs/depth/.../preprocessing.npz` | [depth](../depth/README.md) | The colour pinhole grid, so masks and depth share pixels |
| `KIT/data/track_3_evaluation_objects.json` | Organisers | Object names per evaluation episode, in slot order |
| `reports/data-audit/trajectories.json` | Data audit | The `objects` name list per public episode, and nothing else |
| `prompts.json` | This slice | Text prompt for each object name |

## Method

1. **Instances.** One instance per object slot, in the official slot order, then
   `left_hand` and `right_hand`. Slot order is not alphabetical in every episode
   (**Official**, `KIT/data/track_3_evaluation_objects.json:4-5`), so the order is
   read from the file and never sorted.
2. **Detect.** On a small set of keyframes, run an open-vocabulary detector with
   the object's text prompt, for example "white pot" for `white_pot`. Keep the
   best box per instance. Prompts are written once and tuned on development
   episodes only.
3. **Propagate.** Seed a video segmentation model with the keyframe boxes and
   propagate forward and backward through the clip.
4. **Recover.** When an instance's mask vanishes or its score falls below the
   configured floor, mark it absent. Re-detect at the next keyframe and resume.
   Absent is a recorded state. No mask is carried across a gap.
5. **Separate hands.** Hands are detected with the same detector and prompt
   "hand". Left and right are assigned later by [hand_motion](../hand_motion/README.md),
   which has a hand-specific model. Until then the two hand instances are stored
   as `hand_0` and `hand_1`.

Both models run on the pinhole image produced by the depth slice's remap, not on
the raw distorted frame, so a mask pixel and a depth pixel are the same ray.

**Model choice (Design).** Grounding DINO for detection and SAM 2.1 for
propagation, the pair the upstream baseline uses. For Grounding DINO, prefer the
implementation in the `transformers` library over the original repository: the
original compiles a custom CUDA operator whose build recipe does not list the
RTX 5080's architecture (source reading in `reports/model-readiness/README.md`), and
the library version is plain PyTorch. This is unverified until a run.

## Outputs

`outputs/object_masks/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Type and shape | Meaning |
|---|---|---|---|
| `run.json` | `instances` | list of names | Index into every per-frame array |
| | `grid` | | Hash of the depth slice's `preprocessing.npz` |
| `frames/frame_XXXXXX.npz` | `masks` | bool K×H×W | One layer per instance |
| | `present` | bool K | False when the instance was not found in this frame |
| | `score` | float32 K | The segmentation model's own score, not a calibrated probability |
| | `source` | uint8 K | 0 absent, 1 keyframe detection, 2 propagated |
| `keyframes.json` | | | Keyframe indices, boxes, detector scores and the prompt used |

Layers may overlap where a hand covers an object. Consumers decide how to resolve
overlap; this slice does not.

## Configuration and authorization

`runtime.json` holds `generate_authorized: false`. The request names the episode,
frame range, keyframe interval and score floor. `prompts.json` maps each of the
11 object names to a prompt.

## Dependencies

Proposed, not installed. Each needs the user's confirmation.

| Item | Size | Licence |
|---|---|---|
| Grounding DINO Swin-T weights | 694 MB | Apache-2.0 |
| SAM 2.1 Hiera-Large checkpoint | 898 MB | Apache-2.0 |
| BERT tokenizer and text encoder files | Small | Apache-2.0 |
| `transformers`, `sam2` and their requirements | Unresolved | Apache-2.0 |

Sizes are the publisher's figures from `reports/model-readiness/README.md`.

## Resources

One GPU stage at a time, through the shared lock in `track3/gpu.py`. Memory and
runtime are unmeasured. The video model holds features for many frames, so long
clips may need to be processed in overlapping windows; the longest scored episode
has 421 frames.

## Failure handling

An instance never found in any keyframe fails the episode with
`instance_not_found` and the prompt that was tried. A partial run keeps its frames
and records the last completed index.

## Separation of public labels

No pose label is read. Object names for public episodes come from the name list
the data audit copied out of each Parquet file's schema metadata. This slice never
opens a public Parquet file.

## Acceptance evidence

Once runs are authorized: overlay videos for development episodes show each
instance on the right object at the start, at first contact, at release and at the
end, and identity never swaps between two similar objects such as the two cups in
evaluation episode 10.

## Open questions

| Question | Who can answer |
|---|---|
| Is a manual click or box on an evaluation video allowed when detection fails? The design keeps manual prompts disabled until answered. | Organisers |
| Do text prompts separate `wooden_piece_1` from `wooden_piece_2`, and `beige_cup` from `blue_cup`? | Development runs |
