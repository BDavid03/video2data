# Artifact contracts

These contracts describe implemented boundaries as of 5 October 2026. They are
not generated schemas or runtime evidence. Exact optional fields and request
limits belong to the owning feature's README and source. No operation was run to
produce this document.

## Run ownership and identity

The research backlog's 2026-10-18 storage correction makes shared
`track3.paths.output_path` refuse generated writes unless the checkout resolves
to `Z:/hugging-face/video2data` or `/mnt/z/hugging-face/video2data`. An
implementation checkout may be edited/read but cannot substitute for canonical
artifact storage. This admission change was read as source only, with no checks.

A feature produces an operation record, normally `run.json`, beside its outputs.
Episode artifacts generally live under
`outputs/<feature>/<split>/episode_XXXXXX/<run-id>/`; policy and submission
features document their own layouts. Consumers name explicit producer records.
A path, a matching filename or a newer timestamp is insufficient lineage.

[provenance.py](../../track3/provenance.py) supplies these shared fields, and
[runs.py](../../track3/runs.py) owns the lifecycle around them (`recorded` writes `run.json`
before and after the body, closes running frame rows on failure, and hashes every file of
the run directory into `artifacts`):

| Field | Contract |
|---|---|
| `kind` | Explicit feature and operation identity, such as `hand_motion.lift` |
| `status` | Initial `running`; consumers require `completed`; failed/interrupted records cannot become inputs |
| `request` | The owning operation's resolved request |
| `inputs` | Explicit project-relative paths and SHA-256 values, including derived producer records |
| `sources` | Feature Python source, shared technical primitives, source lock and additional recorded integration sources |
| `provenance` | One of the three defined provenance tags |

Features additionally record artifact inventories, frame outcomes, timings,
backend identity and resource evidence appropriate to their operation.
`run_fields` does not itself inventory every generated artifact. `read_run`
checks a completed producer's kind/provenance and reads its identity; the consumer
must separately compare consumed artifact bytes with that producer's recorded
hashes. Do not interpret a successful producer-record read as full artifact
admission.

A completed perception run may contain invalid frames. Completion means the
operation finished its declared work; the validity arrays say which measurements
exist. An interrupted operation may preserve partial diagnostics, but its run
status prevents downstream use. JSON metadata uses finite values or explicit
nulls; array artifacts can use NaN under their validity masks.

## Provenance and data separation

| Tag | Meaning and permitted use |
|---|---|
| `video_only` | Production-compatible lineage, subject to each feature's additional asset and split rules |
| `public_asset_diagnostic` | Diagnostic lineage involving a public reference asset; not eligible for submission |
| `public_label_diagnostic` | Public pose-label diagnostic lineage; not eligible for production reconstruction or submission |

Inheritance takes the strongest tag across parents. Missing legacy tags and
unknown tags are refused rather than guessed. Copying a diagnostic artifact into
another directory cannot make it production-compatible.

`video_only` does not mean that public tracking scans are forbidden. It means the
lineage satisfies the production path's policy. Submission separately requires
CD-O mesh metadata to say `source = reconstructed`, with the correct source
episode. A video-only public scan remains ineligible as a reconstructed submission
mesh. Human interpretation of a development report must not introduce label arrays
as hidden producer inputs.

## Episode, grid, clock and frame conventions

Cross-feature consumers bind dataset revision, split, episode, object ordering,
producer identity and the calibrated image grid. Same dimensions alone do not
establish equal intrinsics, remaps, padding or orientation. Depth owns the
pinhole preprocessing used by masks and geometric consumers.

| Quantity | Meaning |
|---|---|
| `frame_index` | Dataset video-frame identity; array rows must match the declared grid |
| `pts` / `video_pts` | Integer MP4 presentation timestamp in the producer's documented field |
| Public nominal timestamps | Public-reference timing, kept separate from decoded PTS |
| Capture time | Recording time; not a substitute for simulator playback time |
| Reference/rollout clock | Exact 20 Hz correspondence used by the implemented policy/scoring path |
| `world_T_camA` | Transform from colour-camera optical coordinates to metric world |
| `camA_T_object` / camera-relative object transform | Object-body coordinates to camera coordinates; use the exact field spelling of the producer |
| `world_T_object` | Composition of camera-to-world and object-to-camera transforms |

The optical frame is x right, y down, z forward. World +z is table-up; table-level
is assumed. Mesh body axes are distinct from world axes. Metres apply throughout
metric geometry. Matrix, axis-angle and quaternion representations must be
converted only at explicit boundaries; upstream hand export uses scalar-first
quaternions. Never infer quaternion order from four numeric values.

## Perception interfaces

The symbols `T`, `H`, `W` and `B` below mean frame count, grid height, grid width
and ordered object count. These are concise interface summaries; linked feature
pages define all fields and failure outcomes.

| Producer | Principal artifact | Required interpretation |
|---|---|---|
| [Depth](../../track3/features/depth/README.md) | Per-frame depth, validity and preprocessing metadata | Depth and intrinsics belong to the recorded pinhole grid |
| [Stereo depth](../../track3/features/stereo_depth/README.md) | `rectification.npz`, per-frame disparity/depth | Rectified depth and colour-grid `rgb_depth_m` are different coordinate representations |
| Stereo anchor | `anchor.json` | Scale, support and spread per frame; only observed scales justify monocular metric geometry |
| [Masks](../../track3/features/object_masks/README.md) | Per-frame masks and instance records | Named full-video object slots followed by anonymous hand slots; opening clusters cannot replace this contract |
| [Camera](../../track3/features/camera_motion/README.md) | `trajectory.npz`, `world.json` | `world_T_camA`: float64 `T×4×4`; `valid`, `observed`: bool `T`; invalid transforms NaN |
| [Object pose](../../track3/features/object_pose/README.md) | `opening.json`, `poses.npz` | Camera validity and world validity are separate; body-frame and visual/collision mesh identities must match |

Direct stereo takes precedence over scaled monocular depth in metric consumers.
Missing per-frame scale can leave direct stereo available; it does not authorize
an interpolated scale. Object tracker confidence is a geometric heuristic, not a
calibrated probability. Surface support is distinct from rendered visibility;
unavailable visibility remains explicitly invalid.

## Hand interfaces

[Hand motion](../../track3/features/hand_motion/README.md) owns four distinct
boundaries: regressed parameters, MANO forward kinematics, depth-fitted/world
hands, and the global neutral parked template.

| Lifted-hand field | Type/shape | Contract |
|---|---|---|
| `sides` | Two strings | `[left, right]`, anatomically associated |
| `valid` | bool `T×2` | Camera-relative estimate admitted |
| `world_valid` | bool `T×2` | Subset of `valid`, with an admitted camera pose |
| `joints_camA`, `joints_world` | float64 `T×2×21×3` | Metres, MANO21 ordering; invalid entries NaN |
| `joint_rotations_world` | float64 `T×2×21×3×3` | Absolute anatomical MANO joint frames |
| `wrist_orient_camA`, `finger_pose` | float32 `T×2×3`, `T×2×45` | Anatomical, unmirrored axis angles |
| `betas` | float32 `2×10` | One fixed shape per anatomical side |
| `frame_index`, `pts` | int64 `T` | Exact frame identity |

The scene boundary requires `mano_fk` orientation provenance and the declared
metric MANO21/anatomical convention. A geometric orientation approximation is
not interchangeable. Mirrored model input is unmirrored once, where declared;
already anatomical estimates are not converted again.

The parked-template run has model-asset scope, `observed: false`, and no episode
or split. Its `model_rest.npz` and `parked_hand.json` bind raw MANO geometry,
per-side wrist-relative joints, anatomical frames and 778 surface vertices.
Reusing this template is not cross-episode reuse of measured hands.

Preview consumes exact completed lift/depth producers and selected frame IDs.
Its images and report retain the source clock, validity and rejection outcomes.
`review_only: true` and `quality_approved: false` prevent a visual preview from
being treated as an acceptance record.

## Mesh interfaces

[Object mesh](../../track3/features/object_mesh/README.md) records source type,
body-frame rule, visual and collision file hashes, method and reconstruction
source episode/frames. Consumers must not substitute a visual asset for a
separately declared collision asset by filename similarity.

Opening-hull reconstruction assumes table-footprint completion and closes a
convex surface. Opening TSDF reports observed surface extraction and its support
requirements. Its output may be unsuitable for collision. Point-fusion metadata
records weighting and capacity; overflow fails instead of silently coarsening.
A proposed method has no producer contract until its handler exists.

## Scene interface

[Scene assembly](../../track3/features/scene_assembly/README.md) writes
`scene.json`, `trajectories.npz`, asset copies and a report. It requires exact
pose, camera, mesh and, in full mode, hand lineage, plus the explicit parked
producer in both modes.

| Content | Contract |
|---|---|
| Object trajectories | Complete metric body poses on the declared frame grid; stable object order |
| `object_state` | Resting/moving classification, recorded as 0/1 |
| `object_filled` | Explicit completion mask |
| Hands | Complete `T×2×21` positions and corresponding anatomical rotations |
| `hand_source` | 0 observed; 1 interpolated; 2 parked or parking transition |
| `hand_completion` | Recorded interpolation algorithm and parent tree; resting mode uses a constant parked template |
| Contacts | Per frame, side and **16-link** slot; exact upstream link order |
| Contact body | One-based object ID; 0 means no contact |
| Inactive contact geometry | Zero position and zero normal, not stale coordinates |
| `contact_layout`, `contact_link_names` | Explicit sampling rule and ordered names for retargeting admission |

A contact proxy, an interpolated hand and a seated object are derived reference
geometry. Their flags must survive export. Scene completion does not retroactively
mark perception observations valid. A support relation must be acyclic; excessive
seating or unsupported object gaps fail at this boundary.

## Training and checkpoint interfaces

[Policy training](../../track3/features/policy_training/README.md) binds the
resolved JSON/YAML recipe to its owning compose run. The permitted training
variation is the output directory. Checkpoint admission reads bounded safetensors
metadata, checks format/version/algorithm/policy identity, exact configuration,
reference clock/count and training counters against the inventory. It does not
decode tensors or establish simulator compatibility.

A zero-update checkpoint cannot claim optimizer updates; a final trained
checkpoint requires them. Rollout and selection recheck file inventories and
source bindings. A passing metadata boundary is still followed by the upstream
runtime's own tensor/schema handling when execution is authorized.

[Scratch depth](../../track3/features/scratch_depth/README.md) has its own
schema-2 checkpoint contract. It binds network source bytes, model configuration,
training data, coursework identity, owning run and counters. Training checks
finite weights before save; prediction requires a strict state load and owned
checkpoint identity. These are implemented checks, not checks performed during
this documentation work. No trained scratch checkpoint currently exists.

## Campaign and submission interfaces

Campaign pins are append-only choices with reasons, producer roles and hashes.
Opening/full-camera and resting/tracked roles remain independent. Lineage follows
actual run paths, including unpinned ancestors; a cycle is an error. A superseded
but intact ancestor is distinct from one whose bytes changed.

Submission snapshots its selections in an immutable plan. Later check, pack and
upload records live under `operations/<operation>-<run-id>/`; they do not replace
the plan's `run.json`. Packing requires a named passing check and performs its
own input admission when authorized.

After the unchanged kit packer exits, `_packed.py` admits the actual output:

- Pose CSV: exact unique roster/columns, staged achieved values after six-decimal
  formatting, nonzero rounded quaternions and selected checkpoint/source identities.
- Mesh Parquet: exact roster/commit, contiguous indices, finite vertices, integer
  in-range faces and at least one positive-area triangle per mesh. Degenerate
  padding faces are allowed by this structural boundary.

`packed-check.json` binds file and plan hashes. Upload requires that same intact
identity before reserving an attempt. This proves neither pose accuracy nor mesh
shape quality, and it does not reproduce decimation or establish acceptance.

## Changing a contract

Document the producer, every consumer and the migration consequence together.
Keep old records immutable. A newly required identity field generally means
older outputs must be regenerated after authorization; do not backfill a claim
that was never measured. New optional fields must not silently change the meaning
of validity, provenance, clocks or object ordering. See the
[feature authoring guide](feature-template.md) for a source-only change workflow.
