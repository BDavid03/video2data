# Campaign

Queue requests accept `submission_mesh_method: opening_hull` (default) or
`opening_tsdf` for same-episode CD-O reconstruction. The method is included in
draft requests and candidate matching. Spoon collision reconstruction always
requires `opening_hull`; an observed TSDF surface cannot satisfy that task.
Both choices remain disabled and unrun.

Status: implemented, never run

Twenty evaluation episodes pass through a dozen slices in several environments on
one GPU, and most stages will run more than once. This slice keeps the record of
which run of each stage is the current one for each episode, shows what is
missing or out of date, and orders the work that remains. It runs nothing in
other slices. It reads run records and writes plans.

## Role in scoring

None directly. A submission is rejected whole when one episode is missing
(**Official**, `KIT/metric_code/track_3/AUC.py:399-405`), so the entry needs one
place that can say, for every scored episode, whether a valid rollout exists and
what it was built from.

## Operations

All disabled by default. All CPU. None opens a video, a model or a label.

| Operation | Produces | Runtime |
|---|---|---|
| `board` | A table of every episode against every stage: newest completed run, pinned run, status, provenance tag | Any environment with the standard library |
| `pin` | An appended entry in the pin log: this run is the chosen output of this stage for this episode | Same |
| `stale` | The list of pinned runs whose inputs are no longer the pinned upstream runs | Same |
| `queue` | An ordered list of the operations still needed, with the GPU time each is expected to take | Same |
| `materialize` | Individual reviewable request JSON files and a hash-bound manifest from a completed queue | Same |
| `evidence` | Explicit hash-bound metadata inventory, approach dispositions and missing-evidence report | Same |
| `freeze` | Immutable metadata choice receipt only when the full declared chain is admitted; blocked receipt otherwise | Same |

`evidence.py` owns `EvidenceRequest` and `campaign_evidence`. Its new
`evidence_authorized` and `freeze_authorized` flags are false. Source templates
are `evidence.json` (existing historical identities and item01–15 dispositions)
and `freeze.json` (deliberately no production candidate). These operations open
only explicit allowlisted metadata, never model/label/prediction arrays, and
start no producer. New receipts require canonical Z: storage. Exact named
historical hashes remain historical; source edits never revise old identities.

See [the source build record](../../../experiments/16-freeze-build.md) for
request fields, role admission, choice/ancestor constraints and explicit limits.
Freeze requests require one coherent episode chain, exact source-bound choices,
independent tuning support on39/42 and export metadata attachments. Missing
approaches/evidence/parents remain visible blockers; they cannot produce a
`frozen_metadata` receipt. Binary hashes are producer-declared and their payloads
are not read or verified. Metadata sealing is separate from numerical correctness,
production promotion, pinning or submission. Every receipt records consumed43/44,
excluded41 and `fresh_holdout_eligible: false`; no new holdout is designated.

Receipts use exclusive creation under
`outputs/campaign/<evidence|freeze>/<unique-id>/<manifest-sha256>.json` and are not
completed producer `run.json` records. Neither operation can select/promote pins.
No inventory/freeze operation has been run in this build.

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| Every `run.json` under `outputs/` and `runs/` | All slices | `kind`, `status`, `inputs`, `provenance`, `timing`, `gpu` |
| `KIT/data/track_3_evaluation_objects.json` | Organisers | The 20 scored episodes and their objects |
| `configs/splits/public-v1.json` | Frozen split | Public development and validation episodes |
| `reports/data-audit/trajectories.json` | Historical public identity inventory | Locked revision, episode identity and ordered object IDs only; no pose arrays |
| `budget.json` | [policy_training](../policy_training/README.md) | Named training templates; these are not measured costs |

Only the shared run-record fields are read
([artifact contracts](../../../docs/design/artifact-contracts.md)). A run record
that lacks them is shown as non-conforming and is never pinned. The implemented
depth and scratch-depth records now use the shared contract; older untagged runs
remain non-conforming.

The board includes full-video masks, all three hand operations, object tracking,
retargeting, policy composition/selection and rollout self-scoring. Its JSON keeps
both the latest stage pin and the latest pin for each producer role. Source
presence does not establish authorization, compatibility or runtime quality.

`queue` is implemented and disabled. `queue.json` defaults to
`target_tier: "tracked"`; `"resting"` limits the plan to the fallback path. The
campaign-local `_planning.py` defines phase-specific task IDs and explicit input
bindings. It imports no producer and starts no work.

Public queues can now bootstrap before the first producer run. Their ordered
object identities come from the pinned historical public inventory, constrained
by the frozen split and source lock. Existing producer `object_names` must agree
with that order; inconsistent or missing identity metadata blocks the episode.
The queue records roster path/hash identities and carries `object_names` on every
row. This does not inspect public pose arrays or use labels as production inputs.

`request_drafts.py` owns explicit metadata for the public request dataclasses and
CLI routes. It combines each operation's existing JSON template with episode
identity, task parameters and resolved bindings. Unknown keys fail rather than
being silently dropped. Rows contain `request_fields`, `request_template`,
`command_argv`, `missing_bindings`, `missing_choices` and `resolved_bindings`.
Template and request-class source path/hash entries are bound by the owning queue
run. During a future queue operation, the standard-library AST reader parses
source text to refuse a stale field catalog; no producer is imported.
Completeness concerns request fields, not runtime evidence or permission.
Camera's principal-ray table origin remains available: a manually supplied
slot-0 point requires a source description, but is not imposed on every draft.

## Reviewable request bundles

[materialize.py](materialize.py) owns `MaterializeRequest` and
`materialize_requests`; [materialize.json](materialize.json) defaults to no queue
and no overrides. The separately false `materialize_authorized` flag controls
writing these derived files. It starts no producer, chooses no new pin, installs
no package and modifies no execution flag.

The operation takes an exact completed `campaign.queue` run and admits its owning
`queue.json`, request-schema source identity, template bytes, selected record/pin
snapshot and public roster. Changed pins, records, templates or roster require a
new queue. Historical authorization-flag values remain in the producer receipt;
they need not be restored merely to write a review bundle. Older queue records
without the new schema/template metadata are refused.

Overrides are keyed by exact task ID: `<split>:<episode>:<stage>`, for example
`public:39:scene_assembly.assemble:resting`. Only fields in that operation's public
request contract are accepted. Task identity, bound inputs, candidate run,
artifact name and promotion tier cannot be replaced through overrides. Fixed
planner choices such as full-clip bounds, scene mode, mesh method and policy
budget are also preserved; changing them requires a new queue.
Explicit mass/support choices, reasons and tunable defaults can be supplied. Scene bundles
require a positive mass and declared support for every named object and refuse
unsupported names, self-support and cycles. No physical values are invented.

Generated bundles live at `outputs/campaign/materialize/<run-id>/`. Each request
has a distinct Windows-safe filename beneath `requests/`. `requests.json` records
its exact task identity, path/hash, argv, remaining dependencies/fields and
structural completeness. Unplanned episode blockers remain visible. The owning
`run.json` inventories partial outputs on failure. Writes require the canonical
`Z:/hugging-face/video2data` checkout or its `/mnt/z/` WSL mount.

All manifest rows retain `execution_authorized: false`. A complete request does
not establish producer admission, environment availability, resource fit or run
authorization. Queue/materialize operations have not been executed. Future CLI
reference only, disabled:

```text
python -m track3.features.campaign materialize --request track3/features/campaign/materialize.json
```

After separately authorized producer work and explicit pinning, regenerate the
queue and bundle to bind subsequent requests to the new exact producers.

## Producer graph

The tracked plan includes these dependencies:

1. Full-clip depth, stereo and scale anchors; an unmasked opening camera pass;
   opening clusters and geometric object assignments.
2. A resting scene, robot reference, zero-update policy, inspected rollout and
   checkpoint selection. Promoting that inspection remains an explicit pin.
3. Named full-video masks seeded from the opening; a masked full-clip camera
   pass; opening geometry rebuilt against that exact camera producer.
4. Hand estimation, MANO kinematics and metric lifting; object tracking and world
   composition; a full scene and its robot reference.
5. Trained policy, rollout using the explicit resting inspection as fallback,
   rollout inspection, checkpoint selection, self-scores for **both** the trained
   rollout and resting fallback against the same full scene, and an explicit
   tracked promotion.

Spoon episodes retain separate scan-only and complete initialisation passes for
each camera. Their reconstructed spoon must use the camera consumed by that
opening pose. A separate submission reconstruction covers **every object**,
including objects whose simulation meshes come from public scans. The queue
stops before submission packaging/upload; those operations retain their own
explicit requests and authorization.

Only a current, explicitly pinned, video-only run with the required role,
request parameters and exact parent hashes satisfies a task. An eligible
completed candidate produces a `campaign.pin` suggestion, never an automatic
selection. If an upstream task needs a new run or pin, downstream candidates
remain unresolved until a later plan can bind that exact input.

Rows contain `request_draft`, `request_bindings`, unresolved `depends_on` task
IDs, environment, GPU use, and `inputs_selected`. Drafts contain only the known
request fields; unresolved run paths are null. They are not executable requests.
`parked_hand_run` in the queue request names one completed neutral
`hand_motion.parked_template` producer shared across all episodes. It is carried
into both scene drafts, and an existing scene satisfies the task only when its
input hash matches that template. A null reference leaves scene selection
unresolved. `shared_inputs` records the template requirement and its disabled
producer request; the queue does not generate or pin a global template as an
episode observation.

The separate `requirements` field names inputs that still require a deliberate
choice or provisioning, such as licensed assets, the parked-hand template,
per-object masses/supports, checkpoint paths and pin reasons. These are standing
requirements, not a runtime/environment readiness audit. `ready` and
`execution_authorized` remain false in every plan.

Timings use medians of completed runs with the same operation, role and budget;
unknown costs remain null. Dependencies take precedence over ordering. Fallback
work precedes tracked upgrades, with CPU work before GPU work, training last,
and the configured episode priorities applied to training. Environment ties keep
nearby work together, including the isolated mask, WiLoR and MANO environments.
A public episode without a consistent object roster in existing producer records
is reported as blocked instead of guessing whether it needs spoon reconstruction.

## Method

1. **Pins.** A stage may have many completed runs for one episode. One run per producer role
   may be current; opening, full-camera, resting and tracked roles stay separate. The pin log is append-only: each entry has the split, episode,
   artifact name, producer role, run id, reason and time. The newest entry for a
   role is its pin. Older entries without a role are classified from their
   producer metadata. Nothing is pinned automatically. A run becomes the current one
   because a person, or an authorized `pin` request, said so.
2. **How slices use pins.** Producer requests still name input runs explicitly.
   The queue resolves selected inputs into request drafts and records their task
   bindings. It does not add implicit pin resolution to producer handlers.
3. **Stale.** A selected run becomes stale when a recorded ancestor is missing,
   incomplete, changed, or superseded by a pin in the same producer role.
   Traversal follows actual run paths, including unpinned ancestors. Repeating an
   operation in a later phase is not a dependency cycle. Ordinary artifact pins
   without a tier do not erase the last explicit rollout promotion.
   Submission and rollout self-scoring consume the public metadata contract in
   `selections.py`. Their full-scene selection survives later resting-scene pins.
   Submission saves a selection snapshot and reports stale warnings separately
   from refusing changed or incomplete selected lineage.
4. **Board.** For every evaluation episode, the board shows the best candidate
   that could be submitted today, by tier
   ([entry tiers](../../../docs/design/entry-tiers.md)):

   | State | Meaning |
   |---|---|
   | `none` | No inspected rollout. The submission cannot be packed. |
   | `resting` | A resting rollout passed inspection |
   | `tracked` | A trained rollout passed inspection and was selected over the resting one |

   A tier must match the inspection's actual policy kind (`resting` or `trained`),
   have `blocking: false`, and retain complete, unchanged lineage. Replay policies
   cannot be promoted as trained policies. The board's first line counts episodes
   in each state; a submission needs zero in `none`.
5. **Queue.** List the operations needed to move each episode to the next state,
   in this order: anything that unblocks an episode in `none`; then episodes
   whose only obstacle is a CPU stage; then GPU stages, shortest first within
   the perception group, with policy training last and ordered by the campaign's
   current priority list. GPU hours come from recorded timings of earlier runs
   of the same operation, and are blank until one exists.
6. **One GPU.** The queue is a plan. Each entry still needs its slice's
   authorization flag and takes the shared lock in `track3/gpu.py` when it runs.
   Perception jobs run on Windows and policy jobs in the Linux environment, so
   the queue groups consecutive jobs by environment to avoid switching.

## Outputs

`outputs/campaign/`

| File | Content |
|---|---|
| `pins.jsonl` | The pin log, append-only |
| `board/<run-id>/board.json`, `board.md` | The table in step 4 and the full episode-by-stage grid |
| `queue/<run-id>/queue.json`, `queue.md` | The ordered plan with expected GPU hours and the hours left before the freeze |
| `run.json` in each run directory | Request, inputs, status |

## Configuration and authorization

`runtime.json` holds `board_authorized`, `pin_authorized`, `stale_authorized` and
`queue_authorized`, all false. `priorities.json` holds the order in which
episodes receive training time.

## Dependencies

The Python standard library. No new package.

## Resources

CPU only. It reads a few hundred small JSON files. Unmeasured.

## Failure handling

A missing or malformed run record is listed on the board with the parse error.
`pin` refuses a run that is not `completed`, a run with a diagnostic provenance
tag on the evaluation split, and a run id that does not exist. Tier promotion
also refuses mismatched policy kinds, blocking inspections and stale lineage.

## Separation of public labels

No label is read. Public and evaluation episodes are shown in separate tables,
and a pin on one split cannot name a run from the other.

## Acceptance evidence

This buildout was edited and read as source only. No tests, smoke checks,
validation commands, imports, planner execution, model execution or simulation
were run. No dependency was added. All runtime flags remain false.

Once runs are authorized: after one episode has passed through every stage, the
board shows that episode as `resting` or `tracked` with the same run ids that
the submission plan then records, and re-running an upstream stage and pinning it
turns every downstream pin stale.

## Open questions

| Question | Who can answer |
|---|---|
| Does a pin log on `Z:` stay consistent when a Windows process and a Linux process append to it? The design allows one writer at a time, through a lock file of its own beside the log. | A run |
