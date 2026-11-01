# Current design decisions

Reconstructed source-based register, 5 October 2026. The previous file contained
no readable decision history. Existing identifiers below are retained only where
readable feature or contract documentation refers to them. The unnumbered entries
record current choices without inventing historical numbering or approval dates.

## Referenced decisions

### D03 — Follow the published contract as well as the packer

Use actual policy rollouts, metres, exact frame correspondence and reproducible
source identity even where the pinned packer does not enforce those requirements.
The [contract report](../../reports/contract.md) distinguishes documentary rules
from code enforcement. Empty evaluator allowlists or permissive input parsing do
not authorize a substitute result. Submission wraps unchanged pinned kit source
and admits the resulting file, while runtime evidence remains a separate need.

### D07 — Metric scale needs an observed ruler

Calibrated stereo is the baseline metric anchor. Known public scan dimensions
are documented as a cross-check and proposed fallback in
[stereo depth](../../track3/features/stereo_depth/README.md). This is not evidence
of an implemented general scan-scale fallback: current geometry consumers use
direct stereo or the frame's observed monocular anchor. They do not silently use
an episode median when a scale observation is absent. Any alternative must have
its own explicit implementation and provenance.

### D10 — Preserve gaps until the completion boundary

Camera/object/hand perception emits validity and invalid arrays. World validity
requires both the observation and a valid camera transform. Scene assembly owns
reference completion and its flags. Filling an input in place would erase the
difference between measured and synthetic geometry and hide camera failure.
Object gap limits, parking transitions and seating limits remain explicit.

### D17 — Compare challengers against the same full scene

Later automatic submission choices compare eligible replay/trained candidates
and the resting fallback against the current selected full scene. A challenger
must stay in the working area and beat the resting score by the recorded margin.
The initial baseline still selects resting rollouts. Explicit overrides are
recorded separately. See [entry tiers](entry-tiers.md) for the complete selection
boundary and its limitations.

## Architecture and operating decisions

| Choice | Reason | Consequence |
|---|---|---|
| Vertical slices | Changes remain understandable at a feature boundary | Request, handler, adapters, persistence, config and docs stay together |
| Explicit run paths and hashes | Reproducibility requires the exact consumed bytes | No implicit latest-run lookup in producers; changed ancestors fail admission |
| Role-aware campaign pins | Opening/full and resting/tracked passes serve different purposes | A later pin in another role does not erase a prior dependency |
| Inherited provenance | Public diagnostic evidence must not become a hidden production input | Unknown and diagnostic lineage cannot be silently retagged |
| Disabled execution | Current user instruction allows source work only | No documentation change enables a flag or grants run permission |
| Isolated optional backends | Native/model dependencies have separate compatibility and asset requirements | Null installation records and revisions remain blockers |
| Project-contained generated state | Preserve the agreed storage boundary | New caches, runtimes and outputs stay under the project root; external environments stay read-only |
| Shared GPU lease | One GPU requires coordinated headroom and scheduling | Arithmetic budgets remain unmeasured until an authorized run records costs |

## Geometry and model decisions

**One colour pinhole grid.** Depth preprocessing owns the grid; stereo, masks,
geometry and previews bind it. Equal width/height without equal intrinsics and
preprocessing is insufficient.

**Table-defined world.** Table normal supplies up, projected first-camera heading
supplies x, and metadata records the actual origin rule. This avoids requiring
labels for world extrinsics but assumes a level table and reliable plane fit.

**Separate simulation and submission meshes.** Unchanged public scans preserve
body-frame identity for tracking. Every CD-O pair gets same-episode reconstructed
geometry. Closed hulls can support collision but fill concavities; observed TSDF
surfaces preserve measured shape without guaranteeing closure. Cross-episode
reuse remains unresolved under Q06.

**Anatomical MANO frames.** Hand rotations must come from the declared MANO
kinematics boundary and preserve left/right conventions. Neutral parking is a
sourced model asset, not fabricated joint coordinates or another episode's hand.
Parent-frame completion preserves equal endpoint bone lengths, with explicit
limits: no joint-limit or collision-free transit guarantee.

**Explicit contact approximation.** Sixteen upstream link slots use sampled
geometric proximity. Keeping the approximation in the contract prevents a
consumer from mistaking it for observed contact forces or exact skin collision.

**Scratch depth from random initialization.** The feature implements an original
CNN informed by coursework concepts. Optional MoGe2 pseudo-depth supervision is
explicit; pretrained teacher labels do not make the student's initialization
pretrained. Coursework files and trained artifacts remain missing, and notebook
execution is not authorized.

## Policy and release decisions

Use the exact 20 Hz reference/output correspondence, not a guessed conversion
from capture rate. Save the resolved training recipe and bind checkpoints to it.
Metadata admission is deliberately separate from upstream tensor/runtime
compatibility and performance evidence.

Keep the submission plan immutable, write later operations separately, and admit
the exact packed bytes before upload. Reserve upload attempts before external
invocation and retain per-leg outcomes. An unknown outcome spends quota
conservatively until reconciled. Upload authorization applies to concrete files,
not to a standing intention to finish the project.

When changing one of these choices, update its owning feature and downstream
contracts, explain the compatibility consequence and retain existing run records.
A future measured result may justify a new design; no such result was collected
for this documentation update.
