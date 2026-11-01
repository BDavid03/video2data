# Object and hand masks

Status: opening geometry and full-video generation implemented, never run.
The optional detector/segmenter environment and models are not provisioned.
All execution flags remain false. No dependency requirements were changed.

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
| `opening-clusters` | Instance masks for the opening frames only, from geometry, with no model | Perception environment | No |

`opening-clusters` serves the resting entry
([entry tiers](../../../docs/design/entry-tiers.md)) and needs no new dependency.
`generate` serves the tracked entry.

### Implemented full-video generation

`GenerateMasksRequest`, `generate.py`, `_identity.py`, `_models.py`,
`_generate_common.py`, `generate.json`, `backends.json` and `prompts.json` own the
full-video path. The reference entry point is
`python -m track3.features.object_masks generate`; do not run it under the
current no-testing/no-execution instruction.

The request selects an explicit completed `depth.generate` run. Grid identity,
dataset revision, episode, original RGB file hash and the pinned MP4 frame/PTS
inventory must agree. Public object identities come only from the historical
object-name inventory; evaluation slot order comes from the kit roster. No
Parquet pose label is opened. The output order is exactly the object roster,
followed by `hand_0` and `hand_1`.

The handler decodes the full clip once, applies the depth producer's pinhole
remap and valid-pixel mask, and retains numbered JPEG frames at quality 95 in
the run's `input_frames/`. Those files are hashed and recorded as artifacts.
Detection and video segmentation use those same images. JPEG compression is an
explicit intermediate representation; its quality and storage costs are
unmeasured. All images, temporary window copies and resulting arrays stay in the
project's output directory.

Detection runs once per distinct prompt at every configured keyframe, including
the final frame. The default interval is ten frames. Boxes undergo score
filtering and nonmaximum suppression. Assignment uses prompt identity, box
overlap and recent center displacement; near-tied alternatives remain absent.
Anonymous hand tracks receive arbitrary initial slot order and are never called
anatomical left/right at this stage. Hand estimation assigns anatomy later.

Named instances sharing a prompt, such as the two wooden pieces, cannot acquire
identity through image ordering. For those episodes, supply both `opening_run`
(`object_pose.initialise`) and `opening_masks_run`
(`object_masks.opening_clusters`). They must use the selected depth run and
the opening pose must record that exact cluster producer. Distinct frame-zero
clusters and the opening's object assignments provide the identity anchors.
Merged clusters do not establish two named identities. A stale or ambiguous
identity stays absent; arbitrary relabeling after a long loss is refused.

After the detector phase releases its model, SAM2 processes adjacent keyframe
windows, with independent forward and backward states anchored by the endpoint
box observations. Each state uses CPU frame/state offload. Windows contain at
most `keyframe_interval + 1` frames, bounded by the request to 61. Each direction
stops publishing an instance after its first small or low-score mask. It cannot
resume without another detector anchor. If both directions yield masks but
disagree beyond the configured IoU limit, that instance/frame is absent.
Otherwise the nearer anchor supplies the output, with score breaking distance
ties. No mask is copied across an invalid interval. Every clip frame gets one
output file, including frames where every instance is absent.

The score is the mean sigmoid of positive foreground pixel logits on valid
image pixels. It is a mask heuristic, not a calibrated object-presence score or
SAM predicted IoU. `run.json.score_kind` declares it explicitly. The request
sets its floor and the minimum mask area. Detector scores remain separately in
`keyframes.json`. The NPZ also includes `anchor_frame_index` (int64 K, -1 when
absent) and scalar `frame_index`/`pts`. Absent layers are all false with NaN
scores and source 0. Source 1 denotes a detector-anchor frame; source 2 denotes
propagation from its recorded anchor.

### Optional model runtime

`backends.json` targets an isolated `.runtime/object-masks` environment and
keeps its installation-record pins null. The adapter targets the
[Transformers 4.56.2 Grounding DINO API](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/grounding-dino)
with a local detector/tokenizer directory, safetensors weights, remote code
disabled and custom detector kernels disabled. No online model identifier is
passed to its loaders. SAM2 is constructed with a local checkpoint/configuration
using the video-predictor interface described by the existing upstream source.
Optional SAM2 postprocessing is explicitly disabled.

Future separately approved provisioning must supply an immutable detector
revision and SAM2 source commit, still null in configuration, and a completed
hash-pinned installation record. That record requires:

- `backend: grounding_dino_sam2`, `status: completed`, `user_approved: true`,
  the exact environment path and matching detector/model/source identities;
- a `packages` version mapping including `numpy`, `torch`, `transformers`,
  `SAM-2` and `opencv-python`, with Transformers matching the configured version;
- a unique `files` inventory of installed `path`/`sha256` entries, including
  imported detector/processor modules, SAM2 builder/predictor modules and the
  installed SAM2 configuration;
- an `assets` list of `path`/`sha256` entries covering every file in the local
  detector/tokenizer directory and the selected SAM2 checkpoint.

The checkpoint revision remains the previously recorded SAM2.1 Hiera-Large
revision. This is an integration contract, not a resolved dependency lock or an
installation approval. Nothing was installed or downloaded for this feature.
Runtime gates require the recorded environment, versions, files and assets before
model construction. Offline flags are set; cache/temp paths stay inside the
project. The shared GPU lease spans both model phases and checks headroom around
their work. Admission allowances are configurable and unmeasured.

Identity, occlusion recovery, foreground scoring, window boundaries, GPU use and
end-to-end consumer compatibility still need future authorized runtime evidence.
Fixed prompts are initial implementation choices, not tuned or validated settings.

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
   `hand_0` and `hand_1`. Slot order is not alphabetical in every episode
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

### `opening-clusters` (Design)

Before a hand arrives, the objects stand apart on the table. Geometry alone can
separate them.

1. Take the metric point cloud of each opening frame from depth scaled by
   stereo_depth's anchor, and the table plane from camera_motion's `world.json`.
2. Keep points between a small height above the plane and a height limit, inside
   the table's inlier extent. The table itself and the background drop out.
3. Group the remaining points into connected clusters on the pixel grid, joining
   neighbours whose 3D distance is small. Drop clusters below a minimum size.
4. Write one mask layer per cluster, named `cluster_0`, `cluster_1` and so on,
   in order of size. This operation does not name objects.
   [object_pose](../object_pose/README.md) assigns names when it registers the
   scans.
5. The opening ends at the first frame where the cluster count changes or a new
   cluster enters from the image border. Frames after that are not processed.

The output uses the same per-frame files as `generate`, with `source` value 3
for a geometric cluster, and `run.json` records `method: opening_clusters` and
the opening's frame range. It fails with `opening_not_quiet` when the first
frame already has a cluster touching the image border above the table. When
fewer clusters than listed objects are found, the run completes and records
`fewer_clusters_than_objects`. Objects that touch at the start, such as a lid on
a pot or a spoon in a pot, form one cluster. object_pose tries to separate them
by registering the scans one after another. If that fails, the episode is left
to `generate`.

Thresholds are request fields, set on development episodes.

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
| | `score` | float32 K | Mean foreground sigmoid logit score for `generate`; not a calibrated probability |
| | `source` | uint8 K | 0 absent, 1 keyframe detection, 2 propagated, 3 geometric cluster |
| `keyframes.json` | | | Keyframe indices, boxes, detector scores and the prompt used |

Layers may overlap where a hand covers an object. Consumers decide how to resolve
overlap; this slice does not.

## Configuration and authorization

`runtime.json` holds `generate_authorized: false` and
`opening_clusters_authorized: false`. The request names the episode,
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

A named object with no accepted mask anywhere in the clip fails the episode with
`instance_not_found`; its prompt and detections remain recorded. Hands may remain
absent. A partial run keeps completed frame artifacts and records failure or
interruption; downstream consumers require a completed run. Missing frames,
changed assets or model output dimensions/identities fail explicitly instead of
becoming successful empty observations. Window copies are cleaned up; calibrated
input frames remain as hashed artifacts. No automatic resume is implemented.

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

## Bounded correspondence and visibility operation

`CorrespondencesRequest` and `correspondences()` in `correspondences.py` extend
the existing mask feature. The CLI operation `correspondences` reads the adjacent
request JSON. Its new `correspondences_authorized` flag is **false**. No new
runtime, model or dependency installation is included. Source only; no checks or
execution have been performed.

The handler requires exact completed, video-only `depth.generate` and
`object_masks.generate` parents, the mask parent's recorded consumption of that
depth run, the pinned revision/roster/source clock, matching grid and every
consumed artifact hash. Outputs require the canonical Z: checkout before optional
imports. Admission allows only public episodes 0/11/12/18/21/23/26/31/39/42 and
an explicit episode runtime allowlist; 41/43/44 and evaluation stay excluded.

At the first requested frame, exclusive incumbent masks seed bounded image
corners. The initial identity is inherited from that mask producer and is not
independently established. Subsequent frames use OpenCV pyramidal Lucas–Kanade
flow, forward/backward consistency, finite/bounds/displacement rejection,
exclusive slot-mask support and bounded RANSAC similarity consensus. At least
six surviving queries are required by default. This is a conservative image
geometry baseline, not a rigid 3D object model: perspective change, deformation,
featureless surfaces and imperfect masks can reject correct tracks. No automatic
replenishment or reinitialization follows complete loss. Unsupported identities
are `unknown`; image flow cannot distinguish occlusion from leaving the view.
Anonymous hand slots remain non-anatomical.

Work is bounded to 384 frames, 256 queries per slot, sixteen slots and 1800 wall
seconds checked between frames. A single OpenCV call is not preempted. Default
work is 32 frames and 128 queries per slot. No GPU allocation or new weights are
needed, but existing incumbent mask artifacts remain a prerequisite.

The completed run kind is `object_masks.correspondences`, schema version 1.
It records `grid`, `instances`, `query_namespace=run_id`, exact hashed parent
`inputs`, and frame rows with `frame_index`, `video_pts`, `status`, `artifact`
and `sha256`. Each `frames/frame_NNNNNN.json` records grid/RGB hash, source
frame/PTS and:

- `tracks`: `query_id` scoped to the run, `roster_id`, `origin_frame_index`, `origin_video_pts`,
  `origin_xy`, `xy` (nullable), `visible`, `geometric_accepted`,
  `rejection_reason`, `discontinuity_id`, `support_kind` and optional
  `forward_backward_error_px`. Rejected coordinates are not observations.
  A rejected query appears in that frame's evidence and is never propagated again.
- `identities`: `roster_id`, `visibility` (`observed` or `unknown`),
  `identity_supported`, `gap_frames`, `discontinuity_id`, `rejection_reason`.
  Gaps count decoded source frames; source PTS permits seconds-based reporting.
- `corrections`: the exact admitted image-only correction rows applied here.

Optional corrections require a file and separately pinned SHA256. The strict
document keys are `schema_version:1`, `provenance:"video_only"`,
`masks_run_sha256`, `grid`, `frames`. Each frame row has only `frame_index`,
`video_pts`, `rgb_sha256`, `operator`, `reason`, `revision`, `roster_id`,
`points_xy` (Nx2, finite, unique, within the exclusive slot mask and query budget).
The RGB hash identifies the incumbent's stored calibrated JPEG used for tracking;
it is not a hash of an original lossless source frame. Corrections start a new
discontinuity and new query IDs. Box/mask edits and cross-slot reassignment are
not supported in this baseline. No label/reference-scan field or diagnostic
producer is admitted. Operator provenance asserts image-only adjudication; the
software cannot infer how an operator obtained their knowledge.

SAM3/3.1 and CoTracker remain separately investigated challengers. Stored-result
ingestion is deferred because no pinned completed challenger artifact exists and
its visibility/confidence/identity semantics cannot safely be inferred. The new
artifact contract enables a future explicit adapter without adding model imports
or treating query IDs as roster identities. No experimental reliability gain is
claimed; mask provisioning, canonical write access and event adjudication remain
blocked.
