# 01/16 — Metadata inventory and final-freeze source build

Implementation only; disabled and unrun. Ownership: `track3/features/campaign/`.
No array, weight, label, mesh or checkpoint was opened. No producer, check,
test, compiler, dependency installer or simulator was launched. No artifact was
generated. Existing historical outcomes were not used to select new recipes.

`evidence.py` adds explicit `EvidenceRequest` / `campaign_evidence` handlers and
CLI operations `evidence` and `freeze`. Both new authorization flags default to
false; existing flags are unchanged. Runtime writes require canonical Z: and
never fall back to this checkout or `/tmp`. Stdlib only; no added dependency.

`evidence.json` reuses the exact expanded owning-run/result/frozen-selection
identities already recorded in item01, plus explicit approach dispositions for
items01–15. It is an editable source request/catalog, not a generated report.
All record paths are individually supplied and SHA-256-bound. Reads are limited
to allowlisted metadata basenames,128 records,4MiB per record and64MiB combined,
without directory scans,
producer imports, binary payload reads or automatic ancestor discovery. Failed
and incomplete records can be inventoried with limitations; their source presence
never becomes completed empirical evidence. Additional completed/failed records
need their exact already-reconciled identities supplied before admission.

Evidence channels are teacher agreement, independent geometry, reference-label
diagnostics, achieved behaviour and runtime/resources. Rows preserve explicit
episodes, claims, failures/limitations and whether 43/44 was consumed. Recorded
episode fields must agree with declared row coverage. Diagnostic channels cannot
supply independent tuning support. The historical tuning winner is retained;
none of these files changes a pin or chooses weights/thresholds.

`freeze.json` intentionally has no candidate or inventory. An incomplete freeze
request can only produce a **blocked** receipt. A future complete request must
declare every item/approach disposition and blocker, all source-bound model,
weight, threshold, geometry and policy settings, and exact producer/artifact
lineage. Each choice's `settings_record` is an inventoried completed video-only
`run.json`, and `settings` must equal its entire recorded request. Choice artifact
hashes come from completed producer declarations; pretrained assets/source pins
can also bind exact raw inputs/sources of the settings producer. Known per-frame,
checkpoint and depth-preprocessing inventories are admitted without file scans.
Binary integrity is explicitly unverified by this metadata-only operation.

The production role map admits fixed producer kinds for depth, camera, identity,
object pose, observed surface, hands, contacts, scene, retargeting, policy and
achieved rollout. Per-episode roles must describe one split/episode. All selected
roles and choice sources must occur in the actual achieved-rollout ancestry;
disconnected records, missing sources/source-lock, missing parents, cycles and
consumed/excluded candidate ancestors block a freeze. Ancestors must already be
present in the explicit metadata inventory; traversal only follows in-memory
records and opens no extra paths. Both39/42 need completed, video-only independent
geometry support whose own recorded episodes match the evidence declaration.
The compatible producer is `stereo_depth.report`, with producer-owned
`geometry.json` schema `independent-geometry-report-v1`, independent-image channel,
fixed scales/protocol, reserved-query partition and positive reported support.
Each tuning report's actual ancestry must bind the selected model assets.
Caller relabelling of teacher metrics cannot satisfy this contract.

Scorecard, event-reel and fixed-scale comparison references are attachments to
exact inventoried metadata with a completed owning producer whose ancestry
includes the selected scene. Null or missing attachments block the freeze and
claim no exports. A manifest is not an encoded reel or a measured scorecard.
No rendered output or quality measurement exists from this build.

Future enabled operations write a unique receipt directory and a SHA-256-named
JSON file with exclusive creation. Prior receipts are not overwritten. Its
content records source identities, exact requests, inventory, choices,
dispositions, blockers, channels and limitations. `frozen_metadata` is available
only to a complete freeze request; it means sealed metadata, not numerical
correctness or production promotion. These receipts deliberately are not shared
completed producer `run.json` records and cannot directly satisfy campaign pins.
Promotion, submission packaging and upload remain separate operations.

Holdout43/44 remains consumed and unavailable for new selection/mixture tuning.
Every receipt fixes `fresh_holdout_eligible: false` and no fresh protocol. No new
untouched set is designated. A future unbiased assessment needs an independently
fixed protocol outside this handler. Episode41 is refused throughout.

Current blockers: canonical write access, upstream independent geometry/contact
evidence, approved challenger/license assets, executed policy/rollout evidence,
simulation authorization and absent exports. New code has no runtime correctness
or compatibility evidence because checks remain prohibited. This build closes a
source-handler gap; it does not close the empirical backlog or production freeze.

Last completed here:01/16 source inventory/freeze implementation. Current: blocked
production-chain admission. Next: parent integration of feature-owned build notes;
after canonical access and admitted evidence,03 then04 within existing scope.
