# Implementation status

## Expanded research comparison completed — 2026-10-18

The expanded campaign completed in the canonical Z: checkout in 1470.21 seconds.
It recorded 35 completed experiment outcomes and two failed primary-environment
compiler attempts: nine new training operations, a reused transformer incumbent,
and tuning/holdout assessment for all 13 individual/ensemble candidates. These
counts do not mean 35 independently trained models. The earlier 12-trial run
below remains a separate historical baseline.

Sparse teacher sampling generated 192 frames across 12 public development
episodes. Maximum-side-128 preparation accepted every pair: 128 training samples,
32 tuning samples from episodes 39/42, then 32 holdout samples from episodes
43/44 after selection was frozen. The five original coursework-CNN variants
trained on a separately prepared dataset containing only the eight training
episodes, with ten epochs, batch size four and seed 31. Research neural refits
used 20 epochs and batch size four. Episode 41 stayed excluded.

The frozen winner is `top2_equal`, averaging scratch eager and BF16 predictions
equally in log depth. Its tuning log-depth MAE was **0.1239556042** and holdout MAE
**0.1521616317**. The old transformer incumbent scored 0.1958149432 tuning and
0.2190753796 holdout on expanded frames. Top-three equal scored 0.1494962110
on holdout; this outcome does not replace the tuning-frozen winner. These are
teacher pseudo-label comparisons, not measured physical depth accuracy or
complete reconstruction evidence. The canonical
[result](../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/result.json),
[frozen selection](../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/frozen-selection.json)
and [owning run](../outputs/learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/run.json)
retain exact candidate definitions, scores, lineage, timings and failures.

TorchInductor failed with `TritonMissing`. The primary CUDA 13.3 attempt failed
when CCCL rejected MSVC's traditional preprocessor. Both remain failed attempts;
no primary compiled-backend parity or speedup is claimed. A separate external
CUDA experiment subsequently completed in the existing read-only
PyTorch `2.12.1+cu132` environment. Compilation took 59.7557 seconds, total time
83.1539 seconds. One real batch produced loss difference
`1.1920928955078125e-7` and maximum gradient difference
`1.862645149230957e-9`, within `rtol=1e-4, atol=1e-6`. Warm eager/CUDA
forward-plus-backward timings were 0.0048373/0.0043066 seconds. The
[external record](../outputs/scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json)
retains exact environment and scope; these single-sample measurements establish
no general speedup, broad correctness or primary-environment compatibility.

The first episode-39 opening-geometry attempt failed at rectification with an
OpenCV `gemm` error. The integration now supplies contiguous 3×3 rotation and
intrinsic matrices plus a 3×1 translation. The
[rerun result](../outputs/opening_experiments/20261018T120039.470932Z-e0b15dc0/result.json)
records four completed stages (rectify, stereo, anchor and camera), one failed
stage (clusters: `opening_not_quiet: an above-table cluster touches the image border`)
and two blocked stages (hull and TSDF). The guard remained enforced; no object
mesh, complete reconstruction or full-episode tracking success is claimed.

Separately, the procedural synthetic-depth pilot completed 128 training-only
samples in 3.330 seconds. Its
[dataset](../outputs/synthetic_depth/20261018T115837.781490Z-06a78723/dataset.json),
[owning run](../outputs/synthetic_depth/20261018T115837.781490Z-06a78723/run.json)
and [preview](../outputs/synthetic_depth/20261018T115837.781490Z-06a78723/preview.png)
retain analytic optical-axis depth and procedural instance IDs. The separate
`synthetic_depth_training_pairs` contract carries `training_only: true`; these
samples were not mixed into the primary measured campaign. Generation supplies
neither a measured training improvement nor evidence of real-scene fidelity.

The completed [research report](../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/report.md),
[comparison figure](../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/comparison.png),
[candidate CSV](../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/candidates.csv)
and [report run](../outputs/learning_experiments/report/20261018T120125.303833Z-cb2e5071/run.json)
retain the combined metadata evidence with the original frozen winner. The
renderer loaded no model or checkpoint weights. These generated-artifact links
resolve in the canonical Z: checkout.

Execution flags remain authorized under the expanded instruction. No tests,
smoke checks, dependency experiments/installations, notebook cells, simulation or
submission uploads ran. All generated state remains on Z:, existing external
environments stay read-only, and full version 1 remains incomplete. These
documentation updates themselves ran no execution or checks.

## Authorized bounded research run completed — 2026-10-18

The user selected the default 12-trial tuning workflow, including teacher target
generation and preparation. Its feature-owned
[request and handler](../track3/features/learning_experiments/execute.py) call the
existing public depth, preparation, planning and tuning APIs and retain stage
progress and child artifact identities. The canonical Z: copy was staged with
backups before execution; environments, dataset inputs and upstream source trees
were not modified.

The run completed 200 MoGe2 teacher frames across the eight training episodes and
tuning episodes 39/42. Preparation retained all 40 selected pairs at maximum side
128: frames 0/5/10/15 per episode, with 32 training pairs and eight tuning pairs.
The smaller grid keeps every default transformer trial within its token bound.
The default seed-31 plan produced three trials per technique; all 12 completed
with zero failures. Workflow time was 841.99 seconds, including 659.03 seconds
for tuning.

| Technique | Best equal-episode tuning log-depth MAE |
|---|---|
| ANN | 0.4028088411 |
| CNN | 0.2804249727 |
| Transformer | 0.1581543745 |
| Gradient boosting | 0.3841079499 |

Selection chose `transformer-c4d255682d988b57440a753f`: width 32, two layers,
four heads, patch size 16, 20 epochs, batch size one, learning rate 0.001 and
zero weight decay. Its checkpoint is
`outputs/learning_experiments/train/20261018T105953.726069Z-440ed4d1/checkpoints/epoch_0020.pt`.
The canonical checkout retains
[the result](../outputs/learning_experiments/execute/20261018T105355.351623Z-37d0aa68/result.json),
[selection](../outputs/learning_experiments/tune/20261018T105658.305032Z-7407936a/selection.json)
and [complete trial records](../outputs/learning_experiments/tune/20261018T105658.305032Z-7407936a/run.json).
Artifact links resolve on Z:; generated data and checkpoints remain there.

These are scores against teacher pseudo-labels on four early frames per episode,
not measured depth accuracy or full-clip evidence. Episode 41, final holdout
43/44, official validation/evaluation, original scratch CNN training, ensembles,
custom compilation, notebooks, simulation and uploads were excluded. No tests,
smoke checks, dependency experiments or installations ran. Execution flags were
returned to false after the selected run. Full version 1 remains incomplete.

## Learning correctness and bounded admission refinements — 2026-10-18

Three sub-agents refined tuning, experimental checkpoints and scratch-depth
optimization; the primary agent reviewed their source edits and aligned the
prediction consumers and documentation. The changes remain inside their feature
slices and reuse existing dependencies.

Tuning now requires training/tuning episode coverage before any trial, pins its
plan/dataset/producer identities throughout the comparison, and records each
successful child output immediately so a later failure retains lineage. Score
admission also binds coverage thresholds, complete tuning frame counts and unused
final holdout. Neural checkpoints bind the dataset to their input inventory,
require unique frame rows and full training episode coverage, and retain ordered
consecutive epoch checkpoints. Boosting admission strengthens prepared-pairs,
calibration, depth and camera contracts.

Scoring and saved-output ensembles now require calibrated RGB ray masks and
prediction validity contained within them. File identity sets replace pairwise
alias scans. Scratch samples require floating depth/boolean masks, preserve
scalar-bound precision before and after float32 conversion, and count the labels
actually supervised. Scratch and research loss adapters refuse shape
broadcasting. Brightness-jitter tensors are released after use.

No tests, smoke checks, validation commands, project imports, planning runs,
preparation, training, inference, notebooks, compiler execution, dependency
experiments or installations ran. No generated artifacts were created. Runtime
flags remain false; correctness, compatibility and performance remain unmeasured.

## Single-file documentation compiler — 2026-10-05

Added [scripts/documentation/compile.py](../scripts/documentation/compile.py)
and its adjacent usage/scope documentation. The standard-library utility collects
root Markdown plus project documentation under docs, features, configurations,
scripts and reports, retaining historical review snapshots with source paths.
It writes a stable table of contents, original-source links, source byte hashes
and collected text to `docs/compiled/all-documentation.md` by default.

The utility excludes vendored/runtime/data trees, symlinks and prior compilations;
reports NUL padding without modifying sources; confines output to the project's
generated documentation directory; and refuses to replace unmarked files. A
temporary file beside the destination is atomically published after successful
source reads. Document-body links and heading anchors are preserved rather than
rewritten; each section links to its original navigation context.

No compilation, tests, smoke checks, validation commands, project imports,
dependency experiments or model/simulator operations ran. No combined document
has been generated. Dependencies and runtime flags remain unchanged.

## Extensive design documentation — 2026-10-05

Rebuilt the unreadable core pages in [docs/design](../docs/design/README.md)
from existing feature documentation and source. The set now covers architecture
and feature flow; run/artifact ownership, provenance, grids and clocks; hand,
mesh, scene and checkpoint boundaries; operating order and release procedures;
entry tiers and D17 comparison; current decisions, error propagation, risks,
delivery milestones and vertical-slice contribution guidance. The new operating
reference describes disabled operations without authorizing them.

The root README now reflects the full implemented feature set and links the
expanded documentation. The version 1 ledger links documentation coverage without
claiming runtime progress. Camera-motion documentation now describes the actual
origin fallback: the first camera's principal ray meets the table when no
explicit slot-0 point is supplied. Existing historical reports and test-command
references remain labelled as historical. The rebuilt decision register retains
readable cross-references without claiming to recover the lost decision history.

Documentation edits only. No tests, smoke checks, validation commands, project
imports, dependency experiments, models, notebooks, simulation or uploads ran.
No dependencies or runtime flags changed. Full version 1 remains incomplete.

## Version 1 buildout: hand completion without bone collapse — 2026-10-05

Source review of [scene assembly](../track3/features/scene_assembly/README.md)
found that independently averaging world-space joints during missing-hand gaps
and parking transitions could shorten or collapse bones as the hand rotated.
Completion now traverses the MANO21 tree, blends anatomical joint rotations and
local bone directions, and interpolates endpoint bone lengths. Equal endpoint
lengths remain constant by construction. Opposing directions have an explicit
deterministic rule; degenerate endpoint bones fail interpolation.

The scene records this completion contract and retains the existing `hand_source`
flags. Observed poses remain unchanged. Hand admission also requires the declared
metric MANO21/anatomical frame convention and boolean world validity contained
within camera validity. Completion is still synthetic reference geometry, with
no joint-limit, collision-free transit or accuracy guarantee.

No tests, smoke checks, validation commands, project imports, models, simulation
or dependency changes ran. These are source/documentation edits only; execution
flags remain false and version 1 remains incomplete.

## Version 1 buildout: hand projection review — 2026-10-05

The [hand-motion slice](../track3/features/hand_motion/README.md) now owns a disabled
`preview` operation for 1–64 explicitly selected frames. It consumes intact,
completed lifted hands and their exact depth producer, binds them to the RGB
video and inventoried PTS, and prepares calibrated RGB images beside MANO21 joint
overlays. The local HTML/JSON report retains camera/world validity, projected
joint counts and the original fit/rejection outcomes. Invalid estimates remain
undrawn. Output files and source artifacts retain hashes in the run record.

Preview uses existing NumPy/OpenCV dependencies without model imports, new
installations or an external viewer service. Estimation and preview now share the
feature-local RGB clock reader. Reports explicitly grant no quality approval;
selected projections do not establish full-clip identity, metric contact or
world-orientation correctness, and do not promote campaign inputs.

No preview was generated. No tests, smoke checks, validation commands, project
imports, video decoding, model execution, simulation or installation ran. The new
preview flag and all existing execution flags remain false. Backend provisioning,
licensed assets and version 1 runtime evidence remain outstanding.

## Version 1 buildout: bounded opening-point accumulation — 2026-10-05

[Object reconstruction](../track3/features/object_mesh/README.md) no longer keeps
every selected point cloud from every opening frame before reducing it. The
private point accumulator projects selected pixels in chunks and merges sums and
counts per occupied voxel, retaining point weighting across frames. Both hull and
TSDF methods use the same accumulator. The default cap is 500,000 occupied voxels
per object; overflow fails explicitly without dropping measurements or silently
coarsening the grid. Counts, capacity and weighting are retained in run and mesh
metadata. TSDF extraction also bounds generated vertices, including those later
discarded with degenerate triangles.

This bounds retained point and surface state; loaded frame sizes, hull library
working memory and total runtime remain unmeasured. Chunked floating-point sums
can differ slightly from the previous all-at-once reduction.

Source and documentation only. No tests, smoke checks, validation commands,
project imports, reconstruction, model execution, simulation or installation
ran. Runtime flags remain false and version 1 remains incomplete.

## Version 1 buildout: reconstruction comparison and scan coverage — 2026-10-05

[Public mesh diagnostics](../track3/features/public_diagnostics/README.md) now
accept an optional baseline reconstruction for each requested episode and require
the same object set on both sides. The unchanged pinned metric uses the same
reference scan and sampling seed for both methods; reports retain source hashes,
method names, distances and candidate-minus-baseline deltas. Reports remain
diagnostic-only and cannot select or modify production geometry.

The known missing `wooden_spoon` scan now produces an explicit unscorable row
instead of aborting a mixed-object report. Coverage and scored/unscorable counts
remain visible; unavailable distances never become zero. Empty scoring coverage
has null means. Unexpectedly missing published scans still fail. The report also
states that original meshes are compared without submission-budget decimation.
The mesh-only branch no longer imports the unused Parquet dependency.

Source and documentation work only. No tests, smoke checks, validation commands,
project imports, diagnostics, mesh loading, model execution, simulation or
installation ran. Runtime flags remain false and full version 1 evidence is
still incomplete.

## Version 1 buildout: observed opening-depth fusion — 2026-10-05

[Object mesh](../track3/features/object_mesh/README.md) now implements an optional
`opening_tsdf` reconstruction method alongside the closed opening hull. It uses
the same pinned masks, metric depth, observed scale, camera and opening object
assignments. A bounded grid accumulates masked projective signed distances;
local marching-tetrahedra extraction uses only fully observed cubes and records
contributing-frame counts, bounds, resolution and the absence of unseen-surface
completion. It uses existing NumPy/trimesh dependencies without adding a package.

The method can represent observed concavities that a hull fills, but has no
measured accuracy advantage and may leave open, noisy or fragmented surfaces.
Grid and surface limits fail explicitly. Total memory/time, view independence,
coverage and geometry quality remain unmeasured. Mesh metadata distinguishes this
CD-O surface from the spoon's completed provisional collision frame.

Campaign queue requests can explicitly choose the submission reconstruction
method. Spoon collision tasks now bind `opening_hull` in their request and
candidate matching. The default submission method remains the hull. Documentation
also now distinguishes implemented reconstruction from proposed registration and
generative reconstruction.

Implementation and source reading only. No tests, smoke checks, validation
commands, project imports, depth/mesh operations, model execution, simulation,
dependency installation or notebooks ran. All execution flags remain false;
version 1 completion remains unproven.

## Version 1 buildout: packed submission admission — 2026-10-05

[Submission](../track3/features/submission/README.md) now admits the produced
file after the unchanged kit packer exits. The private feature contract checks
exact columns and unique sample row IDs. Pose admission compares each value with
the selected staged trajectory after six-decimal formatting and checks per-row
checkpoint, evaluator, script and commit provenance, including nonzero rounded
quaternions. Mesh admission checks episode-object pairs, finite vertices,
contiguous arrays, integer in-range faces and a positive-area surface while
allowing degenerate padding.

The packing operation saves `packed-check.json` with the output and plan hashes.
Upload requires the intact report and matching completed packing record before
reserving an attempt. This closes the previous handoff that relied on packer exit
status alone. Structural mesh admission does not reproduce decimation or measure
shape accuracy, and neither report establishes leaderboard acceptance.

The pinned packer and schema sources were read only. No tests, smoke checks,
validation commands, project imports, Parquet reads, packing, model execution,
simulation, installation or upload ran. Runtime flags remain false. Version 1
still requires complete artifacts and separately authorized runtime evidence.

## Version 1 buildout: scratch-depth checkpoint provenance — 2026-10-05

[Scratch depth](../track3/features/scratch_depth/README.md) now records standard
output artifact hashes and includes coursework hashes in its source inventory.
Training requires dataset ownership by its preparation run and rechecks sample
bytes on each read. Schema 2 checkpoints bind the exact network source, owning
run, training request, dataset, coursework and preprocessing/target contract.
Prediction requires matching artifact ownership and reconciles epoch and update
counters with the completed fixed-epoch training request. Saving and prediction
reject nonfinite model tensors; strict state loading remains required.

The notebooks referenced by `coursework.json` are absent from this workspace.
Their earlier documented text reading does not establish current availability;
restoring them remains a prerequisite to any future authorized operation.

Source and documentation changes only. No tests, smoke checks, validation
commands, project imports, checkpoint reads, notebook execution, preparation,
training, inference or installation ran. All scratch-depth execution flags remain
false, episode 41 remains excluded, and version 1 completion remains unproven.

## Version 1 buildout: resolved policy/checkpoint admission — 2026-10-05

[Policy training](../track3/features/policy_training/README.md) now preserves
complete resolved job configurations as JSON as well as YAML. A future training
run must match its composition except for its own output directory. Checkpoint
admission compares the saved recipe with that snapshot and checks upstream's
added reference metadata, exact episode clock, rigid-object contract, actor
format/algorithm, whole-collection counters and filename. Final trained policies
must contain optimizer updates; zero-update policies must not.

The training-owned standard-library admission contract is also consumed by
rollout and checkpoint selection. Configuration/schema metadata hashes and
counters are retained in the checkpoint inventory and rechecked by consumers.
This is metadata integrity, not tensor loading or runtime compatibility evidence.
The pinned upstream training entry point and checkpoint writers were read as
source references only.

No tests, smoke checks, validation commands, project imports, Hydra composition,
checkpoint reads, training, model execution, simulation or installation ran.
All operation flags remain false. Version 1 completion remains unproven.

## Version 1 buildout: MANO contact-link integration — 2026-10-05

Source reading found that scene contacts selected arbitrary joints 0–15 while
retargeting labelled their 16 slots with 21 joint names. The pinned upstream
contract instead defines one palm link and three links for each of five fingers.
Scene assembly now samples those exact joint groups, reduces nearest triangle
queries to one contact per link, and records explicit link-order metadata.
Its existing 21-joint penetration diagnostic remains separate.

Retargeting export now compares the scene layout and array slot names with the
pinned installed upstream definition and writes the correct 16 `mano_link_names`.
The future inspection checks that order and inactive contact fields as well as
body IDs and active normals. Parked-hand templates no longer own contact slots.

This is a bounded sampled proximity method, not an observed-contact or complete
hand-surface collision claim. The upstream references were read only. No tests,
smoke checks, validation commands, project imports, model execution, retargeting,
simulation or installation ran. All operation flags remain false, and runtime
accuracy/cost and version 1 completion remain unproven.

## Version 1 buildout: sourced parked-hand templates — 2026-10-05

The [hand-motion slice](../track3/features/hand_motion/README.md) now owns the
independently disabled `parked-template` operation. Its implemented method uses
the existing optional MANO adapter to evaluate zero-shape flat-rest hands,
recentre at the wrist, and rotate geometry and anatomical frames together into
a defined neutral world convention. It preserves raw model geometry and source
asset hashes and marks the result as a model template, not an observation.

Both scene modes now require an explicit completed template producer. They retain
its artifact and provenance, and parking clearance includes its surface bounds.
The campaign queue can bind one shared template into every episode's resting and
full scene drafts. The old null template file is retained but no longer consumed.

No template was generated. MANO provisioning, licensed assets and future
orientation/retargeting evidence remain required. This was source/documentation
work only: no tests, smoke checks, validation commands, project imports, model
execution, simulation, dependency installation or notebooks ran. All execution
flags, including the new template flag, remain false.

## Version 1 buildout: submission/campaign integration — 2026-10-05

[Submission](../track3/features/submission/README.md) and rollout self-scoring
now use the campaign-owned selection metadata contract. The pin log is read and
hashed as one snapshot. Full-scene selection is independent of resting-scene
pins, and submission stale warnings compare matching producer roles across the
actual selected ancestry. Intact superseded inputs remain warnings; changed,
incomplete or diagnostic selected ancestors and lineage cycles fail the plan.
Mesh selections receive the same lineage treatment as rollouts.

Submission plans save the role selection snapshot and automatic-comparison
evidence, including the common scene, resting fallback and candidate scores.
Initial baselines remain resting; subsequent automatic comparisons choose the
highest eligible in-area challenger only when it exceeds the stated margin.
Inspection/rollout/training policy kinds and checkpoint ownership are checked
by the future handler. The campaign graph now includes resting-fallback scoring
against the same full scene as the trained candidate.

Implementation and documentation only. No tests, smoke checks, validation
commands, project imports, scoring, packing, model execution, simulation,
installation or upload ran. Runtime flags remain false and version 1 completion
remains unproven.

## Version 1 buildout: campaign planning — 2026-10-05

The [campaign slice](../track3/features/campaign/README.md) now plans through the
tracked reconstruction and policy path. Its graph includes opening-camera
bootstrapping, named masks, a full camera pass, matching opening geometry, hand
reconstruction, object tracking, both scene/reference/policy paths, checkpoint
selection and explicit rollout promotion. Submission mesh work includes every
object, including scanned simulation objects in spoon episodes.

Pin history distinguishes producer roles, so a full camera or tracked policy
selection can retain its opening/resting ancestors. Stale traversal follows
actual run paths and exact recorded hashes, including unpinned ancestors. Entry
tiers require nonblocking inspections of the matching policy kind. Unqualified
artifact pins do not erase the last explicit entry promotion.

Plans contain request drafts, exact selected-input bindings, unresolved task
dependencies and outstanding input requirements. Every row remains disabled;
unknown timings remain null. Submission's separate stale-warning reader still
needs integration with these producer roles. Runtime evidence, provisioning,
licensed assets and the sourced parked-hand template remain outstanding.

Source and documentation edits only: no tests, smoke checks, validation commands,
imports, planner execution, model execution, simulation or dependency-resolution
experiments ran. No dependencies were added and all runtime flags remain false.

## Version 1 buildout: full-video masks — 2026-10-05

The [object-mask slice](../track3/features/object_masks/README.md) now implements
the disabled `generate` operation: calibrated frame preparation, fixed prompts,
detector keyframes, explicit object/anonymous-hand assignment, and independent
forward/backward SAM2 propagation in bounded windows. It writes every clip frame
with the named layers, original clock and producer hashes required by tracking
and hand estimation. Shared-prompt object identities require distinct measured
opening assignments; ambiguous matches remain absent.

The optional detector/segmenter adapter requires a separately approved isolated
environment, pinned code/model revisions and complete file/asset inventories.
The detector and SAM2 source pins still need provisioning decisions. The
Transformers interface and local upstream SAM2 reference were read as source
documentation. No models or packages were downloaded or installed; no test,
validation, dependency-resolution, import, decoding or inference command ran.
All flags remain false and runtime behavior remains unmeasured.

## Version 1 buildout: hand estimation — 2026-10-05

The [hand-motion slice](../track3/features/hand_motion/README.md) now implements
disabled estimation and its optional WiLoR-mini adapter, completing the source
path from anatomical hand parameters through MANO kinematics to world lifting.
It uses calibrated virtual-camera input, preserves real detector confidence,
matches masks and anatomical sides, and fixes episode shape from accepted frames.
The adapter follows the already-recorded upstream source commit and requires a
separately approved, hash-pinned runtime and model assets before it can load.

The pinned pipeline/model source was read for API and camera/colour conventions.
No packages, models or environments were installed; no tests, validation commands,
dependency-resolution experiments, imports, video decoding or inference were run.
All operation flags remain false and installation-record pins remain null.
Full-video mask generation, approved model provisioning and runtime evidence are
still required for a complete hand reconstruction and version 1.

## Version 1 buildout: hand kinematics — 2026-10-05

The [hand-motion slice](../track3/features/hand_motion/README.md) now includes the
disabled `kinematics` request, handler and isolated MANO adapter. It records the
estimate/mask/depth lineage, converts mirrored left-hand wrist and finger
parameters together, evaluates the side-specific model and anatomical joint
frames, and exports the joints/rotations/surface samples needed by lifting.
Partial batches remain explicitly invalid until successfully computed.

The optional backend requires the planned isolated environment, a pinned completed
installation record, package/source pins and user-provided licensed model assets.
Those records/assets have not been created or supplied by this work. No package
requirements or installed dependencies were changed, and no test, validation,
import, model or provisioning command ran. All execution flags remain false.
Hand estimation and full-video mask generation remain implementation gaps.

## Version 1 buildout: object tracking — 2026-10-05

The [object-pose slice](../track3/features/object_pose/README.md) now includes
the geometric full-clip `track` handler and request. It uses the complete opening,
named masks, depth/stereo and camera trajectory, compensates camera motion,
excludes hand occlusion, and records explicit loss and reacquisition. `compose`
accepts the tracked mask/stereo lineage. Runtime flags remain false; no dependency
was added and no test, validation, import, decoding or execution command ran.

Full-video mask generation and hand detection/kinematics remain implementation
gaps. This is progress toward the complete version 1, not a completion claim.
The [version 1 delivery ledger](../docs/design/version1.md) preserves the full
pipeline scope and the evidence still required for completion.

## Hand lifting implementation — 2026-10-05

The [hand-motion slice](../track3/features/hand_motion/README.md) now implements
the CPU `lift` operation, its request/CLI, disabled runtime configuration, input
contracts and artifact persistence. It fits metric camera-relative hand surfaces
to depth along the wrist ray, then transforms valid joints and MANO orientations
through the selected camera trajectory. Invalid world poses remain NaN; no hand
or camera gaps are filled. Producer hashes, frame clocks, grid identity,
anatomical side assignments and video-only provenance are explicit requirements.

Detection, MANO forward kinematics and full-video hand masks are still missing
upstream producers. Their required handoff is documented; no substitute detections,
licensed model files or generated outputs were created. This closes one
implementation boundary, not the complete reconstruction milestone.

All hand execution flags remain false, and the episode allowlists are empty.
No tests, smoke checks, validation commands, imports, dependency-resolution
experiments, model execution or simulation were run. No dependencies were added.
Runtime behavior and accuracy remain unmeasured. Earlier dated sections below
describe their historical state rather than the current inventory of source code.

## Contract and design — 2026-09-30

The organisers opened submissions on 2026-09-29 and published a Submission Kit.
The contract is recorded in [contract.md](contract.md), from a read-only reading
of the kit and of the FlashCHORD source it names. Nothing from either was run or
installed. The design set is in [docs/design](../docs/design/README.md), with one
README per feature slice under `track3/features/`.

| Work | Status | Where |
|---|---|---|
| Official contract | Recorded. Eight of twelve earlier unknowns resolved, three partly, one open. | `reports/contract.md`, `configs/evaluator/contract.json` |
| Source readings | Four reports | `reports/contract-evidence/2026-09-30/` |
| New read-only inputs | Submission Kit; FlashCHORD and retargeting source at `aee95f25`, without LFS assets | `upstream/`, pinned in `configs/sources.lock.json` |
| System design | Written | `docs/design/` |
| Feature designs | Eleven new slices designed; no code | `track3/features/*/README.md` |
| Policy environment | Does not exist | Needs the user's confirmation |
| Git repository, Kaggle access | Do not exist | Need the user |
| Runs | None since the no-testing instruction | |

What changed in the project's understanding:

- The scored output is the object trajectory from a policy rollout in the
  FlashCHORD simulator, plus reconstructed object meshes. No hand or scene
  reconstruction is submitted.
- One policy is trained per scored episode. There are 20.
- The runtime documents 48 GiB of GPU memory for full training. This machine has
  16 GB. Whether training fits is the first thing to measure.
- `wooden_spoon` appears in ten scored episodes and has no public scan.
- The released floating-hand recipe plays the reference at half speed, which
  would misalign a submission silently. The design trains at full speed.

The dated plan is the [roadmap](../docs/design/roadmap.md). It needs decisions
from the user by 2 October.

The sections below are the record of earlier work and are unchanged.

## First implementation batch — 2026-09-24

The foundation tooling and audits are implemented. **The first batch is not yet
complete:** a full video reconstruction and a simulator step/render remain open.
At the initial foundation checkpoint, no new production dependencies, model weights,
simulator packages or container images had been installed. All project-generated
artifacts are stored in this repository.

**Current user direction: absolutely no testing.** All test, smoke, validation and
dependency-resolution runs are stopped. Results below were obtained before that
direction. The combined check runner is disabled; subsequent implementation and
documentation edits are not tested. See `AGENTS.md` and `configs/project.json`.

| Work | Status | Evidence |
|---|---|---|
| Project folders and source pins | Complete | `configs/sources.lock.json`; clean detached upstream checkout at `43a3f5b4d479581f2e8bdf5e6ed7ca361d39ec5a` |
| Dataset checksums and full decoding | Passed | 169/169 files, 1,469,035,134 bytes; 120/120 clips, 31,281 decoded frames; `reports/data-audit/audit.json` |
| Public trajectory and asset checks | Passed for audited properties | 20 episodes, 5,246 rows, 8,492 valid object poses; mesh and URDF references resolve |
| Calibration and pilot selection | Initial diagnostics complete | Episode 41, Pan_Solo; geometry checked at frames 0/149; RGB sheet reviewed at 0/99/199/298 |
| Frozen development/validation split | Complete | 13 development, 7 validation; `configs/splits/public-v1.json` |
| Standard-library adapters and metrics | Implemented | `track3/data/`, `track3/calibration/`, `track3/metrics/`; tests under `tests/` |
| Native CUDA smoke | Passed, synthetic only | Existing read-only torch 2.12.1+cu132 environment; inference, one update, exact checkpoint reload; `reports/runtime/gpu-smoke.json` |
| Model and dependency audit | Initial audit complete | `manifests/models/readiness.json`, `reports/model-readiness/README.md`; no full environment or model compatibility claim |
| Official Track 3 contract | Blocked on publication/access | 12 explicit unresolved fields; named evaluator absent from four inspected complete source trees; `reports/contract.md` |
| Full scene reconstruction | Not run | Model/code prerequisites, moving-camera backend, licensed hand asset and measured GPU compatibility outstanding |
| Simulator stepping and recording | Not run | No installed runtime found; official robot/task/simulator contract unresolved |

## Findings affecting implementation

All camera stream counts and container PTS agree within each episode, but that
does not prove physical exposure synchronization. MP4 videos use 20 fps. Public
nominal timestamps differ from MP4 PTS by up to 0.233375 seconds; capture times
form another clock. The adapter preserves those fields separately.

Episode 41's sampled decoded pixels are already upright. Adding another 180°
rotation worsens stereo residuals. The stored stereo rectification rotations are
inconsistent with the supplied transform; recompute from K, distortion, R and t.
Even the recomputed sample geometry has roughly 2 px median vertical residual,
and RGB geometry residual is larger. These are diagnostics on two frames, not a
validated dense depth pipeline or a universal calibration approval.

The root reviewed the four-frame RGB contact sheet: a single white pot is visible
on the table at the endpoints, with hand manipulation in the middle samples.
These are uniformly sampled review frames, not detected contact/release events.

Independent review identified and corrected failure cases in audit exit status,
unknown camera handling, pose-convention validation and quaternion tolerance.
Invalid or missing evidence now fails the relevant audit check. Public pose
norm tolerance matches the diagnostic metric protocol at 1e-3. Supplied URDF
physical parameters remain unverified; a sampled pot URDF calls its mass/inertia
placeholders, so asset reference validity is not simulator dynamics validation.

Before the no-testing direction, the integrated suite passed **36 tests with no skips**. The strict audit
report, GPU checkpoint hash, exact upstream commit and clean checkout were also
verified. The all-stages PowerShell entry point passed syntax validation; its
individual commands were tested separately, avoiding another unnecessary full
video decode during integration.

The CUDA smoke took 4.18 seconds with sampled total device memory peaking at
2,903 MiB, including the desktop. This confirms basic CUDA execution on the
RTX 5080, not model kernels, simulation graphics, robot control or task success.
Peak RAM and full reconstruction runtime have not been measured.

## Reproducibility and storage

`manifests/runs/foundation.json` is a coordination record linking historical
evidence. Dataset hashes and checkpoint hashes are in their existing reports;
the latest complete source/configuration/artifact hash snapshot has not been
generated. Test evidence is saved in
`reports/foundation/tests.log`. The one-command CPU verification entry point is
`scripts/run-foundation.ps1`; see the repository README for individual commands.

All new environments, temporary files, caches, model files, logs, recordings and
checkpoints belong beneath this directory. Dot-source
`scripts/use-local-artifacts.ps1` before tool runs. Reusing the external existing
CUDA interpreter is read-only. The shared Ubuntu/Docker setup was not changed.
Any future isolated Linux virtual disk/container storage must also be inside this
repository. Large artifacts stay here even when `.gitignore` excludes them from
future commits.

## Next executable step

Use the existing dependency-resolution evidence for a single MoGe2 depth stage to
request the workspace-required confirmation for its exact new dependencies and
one pinned 1.324 GB weight file. The pilot uses episode 41 and produces local depth
evidence; it does not close the full reconstruction milestone. Simulator setup
must preserve the eventual official contract. Licensed MANO availability and
compiled perception/camera components remain separate prerequisites.

The [install-only proposal](../configs/perception/moge/README.md) is now concrete:
an independent repository-local Python 3.13 environment, 69 locked wheel packages,
three pinned source trees and the one MoGe2 weight. The earlier Python 3.14
resolution failed on ModernGL wheel availability; the Python 3.13 resolution
succeeded before the no-testing direction. CUDA wheel sizes and runtime behavior
remain unverified. Installation approval would not authorize a test, smoke run,
model execution or simulator run.

On 2026-09-25 the user approved that bounded installation. Its execution record is
`reports/installation/moge-20260925/installation.json`, with a separate log for each
installation step. All 69 locked packages, three exact source checkouts and the
model download completed. The environment is `.runtime/moge-python313`, the
checkpoint is `models/moge/model.pt`, and `scripts/use-moge.ps1` configures the
current shell without running Python. No tests, validation commands, model
inference or simulation were run during installation. The no-testing instruction
remains in force; installation does not close the reconstruction milestone.

## Depth implementation — 2026-09-25

The user selected depth-stage implementation only and requested vertical slice
architecture. `track3/features/depth/` now owns the request, operation handler,
RGB/timestamp selection, calibrated preprocessing, MoGe2 integration, GPU resource
ownership, output persistence, configuration and artifact documentation.

The implementation targets public development episode 41 with one RGB frame per
model call and explicit depth validity. It preserves container timestamps, avoids
Parquet pose labels and records both input and model camera intrinsics. All run
artifacts are designed to stay inside the repository. See the feature README for
its NPZ contract and partial-result behavior.

No tests, linting, compile/import checks, validation commands, video decoding or
model inference were run for this implementation. No dependencies were added.
The existing model execution flag remains false. Runtime correctness, memory use
and depth accuracy remain unmeasured; no generated depth results are claimed.

Six Astra-high foundation assignments were used, with no more than six subagents
active concurrently. Later perception, experiments and final-review assignments
have not been dispatched as completed stages.

## Coursework depth implementation — 2026-09-25

The user additionally requested a depth model from scratch using `coursework/`.
`track3/features/scratch_depth/` implements an original convolutional
encoder–decoder with random initialization, dropout and masked log-depth regression.
The local beginner guide and Week 4 CNN lab informed the design; the feature's
`coursework.json` maps the source cells to the implementation. No notebook was run.

The vertical slice owns requests, preparation of aligned RGB/pseudo-depth pairs,
training, checkpoint persistence and calibrated video prediction. Future training
uses explicitly supplied completed MoGe2 artifacts as pseudo-targets, not measured
ground truth. The CNN does not import MoGe, reuse pretrained weights or add new
dependencies. It reuses the installed PyTorch environment. GPU ownership is now a
shared technical primitive in `track3/gpu.py`, used by both depth slices.

Preparation, training and prediction authorization flags are all false. No tests,
validation commands, compile/import checks, training, decoding or inference were
run. No paired training data, trained checkpoint or depth result was generated.
The implementation's runtime behavior, memory use and accuracy remain unmeasured.

## Learning and optimization buildout — 2026-10-11

The user requested continued implementation with sub-agents, PyTorch optimization,
custom CUDA kernels, ensembles, gradient boosting, ANNs, CNNs, transformers and
hyperparameter tuning. The user approved available GPT sub-agents for this
buildout because the requested Opus/Sonnet models were unavailable in this
session. Three bounded assignments own training optimization, the optional CUDA
loss, and the independent learning-experiments slice respectively.

Scratch-depth adds opt-in optimization choices and separately disabled compilation
flags. Source inventory now includes nested C++/CUDA files; compiler caches have
project-owned paths. Learning experiments own bounded plans and experimental
model components without changing the official policy runtime or production
checkpoint admission. The [learning design](../docs/design/learning-buildout.md)
records model separation, future comparison rules and remaining execution evidence.

No tests, smoke checks, validation commands, module imports, CUDA compilation,
benchmarks, dependency installation, data preparation, training, inference or
notebook execution were performed. No generated plans, checkpoints, models,
metrics or performance results are claimed. All operation flags remain false.

## Owned learning operations — 2026-10-11 continuation

Source buildout closes the factory-only gap in learning experiments. The slice
now owns neural training/prediction, a strict public prepared-data adapter,
source/data/epoch-bound checkpoints, persisted boosting operations, complete-group
pseudo-depth scoring and a sequential finite tuning runner. Selection records
the exact tuning score, candidate checkpoint and child run hashes; failed trials
remain visible. Final-holdout access has a separately disabled flag and is never
used for tuning selection. Prediction support is derived from calibrated RGB
rays independently of the teacher-label mask.

Three GPT sub-agents implemented prepared-data/scoring, checkpoint/boosting and
saved-output ensemble contracts while the primary agent implemented neural
handlers and orchestration.
Integration was reviewed by reading source. No tests, validation commands, import
checks, data preparation, training, prediction, scoring, tuning, compiler runs,
dependency installation or notebooks were executed. No trained or selected model
exists. This is implementation progress toward first-run readiness, not proof
of a working end-to-end version. Prepared teacher artifacts, coursework restoration,
approved optional runtimes/assets and execution evidence remain outstanding.

## First-run campaign preparation — 2026-10-11 continuation

Public queues now use the locked historical public identity inventory without
requiring an already generated producer record. Queue rows carry authoritative
object order and source/template identities; conflicting producer rosters block
that episode. Request drafts now merge source templates with actual dataclass
field metadata, CLI routes, task parameters and explicit dependency bindings.

The disabled `materialize` operation writes reviewable per-task request JSON and
a path/hash manifest from a completed queue. It preserves unresolved dependencies,
refuses changes to identity/bound inputs and records unplanned episodes. Physics
choices remain explicit and no new pin or execution permission is inferred.
Integration source review removed an unnecessary manual-camera-anchor requirement
because the camera producer already has a principal-ray table-origin path.

Two GPT sub-agents implemented queue bootstrap and request-schema metadata;
the primary agent implemented request-bundle persistence and integration.
No tests, validation commands, import checks, campaign operation, materialization,
producer run, model execution, simulation, package installation or notebook ran.
No request bundle was generated. All flags remain false. First-run runtime
correctness and full version readiness remain unproven.

## Learning execution boundaries — 2026-10-11 continuation

The direct tensor ensemble API now requires its own ensemble authorization in
addition to model and prediction authorization. Stored boosting predictions are
rechecked against depth bounds in float64 after float32 conversion, preventing
scalar-bound rounding from admitting values outside the dataset contract.

A GPT sub-agent implemented the boosting correction; the primary agent reviewed
the edited source and corrected the ensemble gate. No tests, validation commands,
imports, model execution, compilation or installations ran. All execution flags
remain false; these changes have no runtime correctness evidence.

The subsequent source pass applied the same depth-bound comparison correction to
prepared-data reads, neural prediction, ensemble member admission and scoring.
Prepared targets are checked both before and after float32 conversion. A GPT
sub-agent changed neural prediction while the primary agent changed the other
three consumers. Persisted array dtypes and configured bounds are unchanged.
No tests or operations ran; first-run readiness remains unproven.

## Existing inputs and continued learning buildout — 2026-10-11

Both referenced coursework notebooks are now present under `coursework/`.
The dataset source remains `datasets--nvidia--video_to_data_challenge`, with its
revision pinned in `configs/sources.lock.json`. Earlier missing-coursework notes
are historical and superseded. Notebook text was read, never executed. Read-only
storage inspection also located the existing Z: MoGe checkpoint, environment and
installation record; incomplete wider searches do not establish global absence
of other assets.

Three GPT sub-agents located existing inputs and corrected the PowerShell cache
setup guard. The primary agent corrected stale documentation, confined policy
compiler caches and added neural weight-decay tuning with training-compatible
learning-rate bounds. The finite search budget and episode partition are retained.
No notebooks, tests, validation commands, models, compilers or installers ran;
all operation flags remain disabled. No performance or end-to-end claim is made.
