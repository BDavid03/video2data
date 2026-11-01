# Operating reference

This is a future operating guide, dated 5 October 2026. **Do not execute these
operations under the current user instructions.** Even operations named `check`,
`inspect`, `board` or `preview` remain disabled. This document creates no run
records and grants no permission. It separates preparation decisions from the
operations they would eventually enable.

## Configuration map

| Location | Owns |
|---|---|
| [AGENTS.md](../../AGENTS.md) | Current authorization and architecture instructions |
| [Source lock](../../configs/sources.lock.json) | Dataset and upstream reference identities |
| [Public split](../../configs/splits/public-v1.json) | Development/validation separation |
| [Calibration policy](../../configs/calibration/ego.json) | Camera assignment, transforms and orientation policy |
| `track3/features/<feature>/runtime.json` | Operation flags and feature runtime boundaries |
| Feature request JSON beside each handler | Explicit producers, episode, limits and operation parameters |
| Feature backend/source configuration | Model assets, installation records and integration pins |
| [Policy environment](../../configs/runtime/policy-env.json) | Dedicated WSL runtime design and provisioning authorization |
| [Training budgets](../../track3/features/policy_training/budget.json) | Complete arithmetic recipes, not measured resource recommendations |
| [Campaign queue request](../../track3/features/campaign/queue.json) | Target tier, parked template and submission reconstruction method |

Null producer paths mean unresolved inputs. They are not instructions to discover
the latest run. Empty allowlists, false flags and missing backend pins must not be
changed merely to make a draft appear ready. A runtime flag is an additional
mechanical gate; it cannot override the user's no-execution instruction.

## Command shape, for reference only

Most implemented feature CLIs follow this shape:

```text
python -m track3.features.<feature> <operation> --request <request.json>
```

Depth is the exception: `track3.features.depth` takes `--request` directly, with
no `generate` positional argument. CLI operation names use hyphens where shown
in the [feature index](README.md); internal run kinds/configuration fields may
use underscores. Read the owning `__main__.py` and request dataclass for exact
syntax. These examples are intentionally not a runnable batch script.

Select the interpreter specified by the feature's environment contract. A
convenient active Python environment is not evidence that optional backends or
native dependencies are compatible. No environment discovery, installation,
import probing or dependency experiments are part of current documentation work.

## Before any future execution

The user must first change the relevant execution restriction. Then the concrete
operation needs its explicit request, permitted episode/split, complete producer
paths, backend assets and approved runtime. New production dependencies still
need separate confirmation. The existing MoGe2 approval is limited to its recorded
installation scope.

Keep all outputs and temporary state under the project storage root. Do not
modify external environments or upstream references. Model assets requiring a
user licence, such as MANO files, must be supplied through their documented
contracts. Missing installation records, revisions or checksums are blockers;
a source checkout is not an installation receipt.

## Dependency order

This sequence explains the dependency graph. It is not a promise that each stage
will succeed, and no stage has been scheduled by this document.

1. **Establish a public development pilot.** Use the frozen development split,
   preserve all clocks and input identities, and keep episode 41 excluded from
   the current depth scope. Do not use validation episodes to tune thresholds.
   The delivery ledger must distinguish historical foundation evidence from
   evidence produced by a new pilot.
2. **Generate metric geometry.** Select depth preprocessing, stereo geometry
   and observed scale artifacts with exact matching grids. Produce the opening
   camera and geometric clusters. This unmasked opening pass breaks the circular
   dependency between initial object assignments and named full-video masks.
3. **Resolve opening objects.** Resolve public tracking/collision scans and
   initialise geometric assignments. Spoon episodes need a scans-only assignment
   pass, same-camera spoon hull reconstruction, complete mesh resolution and a
   second initialisation. The spoon body-frame review remains a prerequisite.
   A scans-only result cannot become a complete scene.
4. **Create the shared parked template.** After MANO provisioning and separate
   authorization, generate one neutral model-asset template. Record its exact
   run path in scene requests; do not substitute hand observations from another
   episode. The template is needed even for the resting fallback.
5. **Build a resting fallback.** Assemble a resting scene with explicit masses
   and supports, retarget it, prepare simulator assets, compose the zero-update
   policy, run the authorized training path and actual rollout, inspect, then
   explicitly select/promote the resting result. Zero updates does not remove
   the simulator or checkpoint requirements.
6. **Build tracked perception.** Generate named full-video masks, then a masked
   full-camera trajectory. Rebuild opening geometry against that exact camera;
   do not reuse an opening run bound to the earlier camera. Estimate hands, run
   MANO kinematics and lift them with matching metric producers. Track objects
   and compose world poses, binding the selected hand producer where required.
7. **Build the full scene.** Admit the exact mesh, camera, object and hand
   producers plus parked template. Record gap completion, seating and contact
   approximations. A selected-frame hand preview can inform human review but
   cannot certify the whole scene.
8. **Train and compare a tracked policy.** Retarget the full scene, compose an
   explicit budget, train, roll out and inspect. Preserve the resting inspection
   as fallback. Self-score both results against the same full scene before
   applying the documented selection margin. Promotion is an explicit choice.
9. **Reconstruct submission meshes.** Produce same-episode geometry for every
   required object. Choose `opening_hull` or `opening_tsdf` explicitly for CD-O;
   spoon collision reconstruction still uses the hull. Do not submit a public
   tracking scan as reconstructed geometry.
10. **Complete the campaign and release.** Select current role-aware producers
    for every required episode, then follow the immutable submission sequence
    below. Campaign queue generation stops before packaging/upload.

The scratch-depth branch is separate: use the referenced files in `coursework/`
(present as of 2026-10-11), then, only after changed authorization, prepare explicitly selected MoGe2
pseudo-label pairs, train from random initialization and predict. Notebook
execution remains outside the requested workflow. Scratch outputs must satisfy
their own source/checkpoint contract before any proposed integration.

## Campaign selection procedure

`board` reports records; `pin` appends a deliberate selection and reason; `stale`
traverses lineage; `queue` writes a dependency-ordered plan. None runs another
feature. Producer requests continue to name exact paths even when a queue helped
assemble them.

A task is satisfied only by a current eligible pin with matching role, request
parameters and parent hashes. A completed candidate can produce a pin suggestion.
It cannot satisfy a downstream exact binding until selected. Separate roles keep
an opening camera, full camera, resting scene and tracked scene from overwriting
one another's meaning.

Queue `requirements` record deliberate choices and missing assets, not a complete
runtime-readiness audit. Unknown costs stay null. Plans retain `ready: false`
and `execution_authorized: false` even if all input selections are available.

## Submission sequence

The pinned local kit describes 20 evaluation episodes, 9,120 pose rows and 36
mesh pairs. The actual pinned roster/schema files are authoritative; do not use
these counts as a substitute for exact coverage. These are local source facts,
not a fresh statement about a live service.

1. `plan` selects inspected rollouts and reconstructed meshes, snapshots campaign
   choices and stages an immutable bundle. The first baseline selects resting
   rollouts. Later plans name a baseline and one stated change. Source provenance
   requires a clean project-root Git checkout whose local HEAD matches the actual
   remote branch HEAD and the recorded credential-free commit URL. The operation
   does not create, commit or push a repository.
2. `check` produces a separate operation record for that exact plan. It examines
   required coverage and producer/source contracts. No check is authorized now.
3. `pack-poses` and `pack-meshes` require an explicit passing check, re-admit inputs
   and call the unchanged pinned packers. Their child environments receive no
   Kaggle token. Each then admits the actual packed file and records its hash in
   `packed-check.json`.
4. `upload` takes the exact submission directory and pack run, requires explicit
   user authorization for the concrete files, and reserves a ledger attempt
   before external invocation. Packing alone never invokes upload.
5. Retain per-leg receipts and reconcile indeterminate results before a retry.
   A crash or unknown outcome conservatively spends the reserved attempt. Retry
   requests name only unconfirmed legs; accepted legs are not resent implicitly.

The local ledger enforces a conservative rolling seven-day cap of five uploads
per kind. The historical contract report explains why. Do not treat that local
cap, or a daily quota returned by preflight, as current remote quota authority.
Any future live-policy review belongs to the authorized release work.

## Failure and resumption

| Observed condition in a future run | Meaning and next source/operating action |
|---|---|
| Missing backend pin or installation record | Provisioning contract is incomplete; do not substitute an unrecorded environment |
| Changed input hash or grid/clock mismatch | Select/rebuild a coherent producer chain; do not relabel files |
| Completed tracking with gaps | Scene policy decides whether gaps can be filled; completion status alone is insufficient |
| Failed/interrupted producer with partial arrays | Retain diagnostics; do not promote or consume it as completed |
| Point/grid/face capacity exceeded | Make an explicit reviewed resolution/limit choice; no silent dropping or coarsening |
| Excessive seating, long object gap or ambiguous pose | Repair upstream evidence or use an eligible inspected fallback |
| Superseded but intact ancestor | Review advisory staleness and select deliberately; do not silently repin |
| Changed/incomplete/diagnostic submission ancestor | Hard lineage failure; a warning waiver cannot repair it |
| Unknown upload result | Reconcile the external outcome before explicitly retrying unconfirmed legs |

No general automatic resume guarantee exists across slices. Preserve immutable
records, choose new run IDs where appropriate and read the owning handler's
failure contract before planning a retry. Never edit a failed record's status
to manufacture a completed producer.
